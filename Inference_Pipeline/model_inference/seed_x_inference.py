#!/usr/bin/env python3
"""
SEED-X model inference adapter

Supported modes:
  - understanding: see eval_img2text_seed_x_i.py
  - generation:    see eval_text2img_seed_x_i.py
  - editing:       see eval_img2edit_seed_x_edit.py

All configs are loaded via Hydra + OmegaConf; must be run from the SEED-X project root.
The pretrained/ directory must contain:
  seed_x / seed_x_i / seed_x_edit / seed_detokenizer /
  cvlm_llama2_tokenizer_... / QwenViT/qwen_vit_G.pt /
  stable-diffusion-xl-base-1.0 / Qwen-VL-Chat

The config JSON must specify:
  - seed_x_project_path: /home/.../SEED-X
"""
import os
import re
import sys
import torch
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from model_inference import generate_output_path, set_seed, load_pil_image

_cached_models: Dict[str, Any] = {}
_failed_models: Dict[str, Exception] = {}

BOI_TOKEN = "<img>"
BOP_TOKEN = "<patch>"
EOI_TOKEN = "</img>"
EOP_TOKEN = "</patch>"
IMG_TOKEN = "<img_{:05d}>"
NUM_IMG_TOKENS = 64
INSTRUCTION_PROMPT = "[INST] {instruction} [/INST]\n"
GENERATION_PROMPT = "[INST] Generate an image: {caption} [/INST]\n"

RESOLUTION_GRIDS = ["1x1", "1x2", "1x3", "2x1", "3x1", "1x4", "4x1", "2x2"]
BASE_RESOLUTION = 448


def _setup_seed_x_path(project_path: str) -> None:
    src = os.path.join(project_path, "src")
    for p in [src, project_path]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _build_grid_pinpoints():
    points = []
    for scale in RESOLUTION_GRIDS:
        s1, s2 = scale.split("x")
        points.append([int(s1) * BASE_RESOLUTION, int(s2) * BASE_RESOLUTION])
    return points


def _patch_xformers_fallback():
    """xformers in seedx env is not CUDA-compatible; replace with PyTorch SDPA fallback."""
    try:
        import xformers.ops as xops
        import xformers.ops.fmha as _fmha
        import torch.nn.functional as F

        _LTM = _fmha.attn_bias.LowerTriangularMask

        def _sdpa_fallback(query, key, value, attn_bias=None, p=0.0, scale=None, **kwargs):
            # xformers layout: [B, S, H, D] → SDPA needs [B, H, S, D]
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            is_causal = isinstance(attn_bias, _LTM)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal)
            return out.transpose(1, 2)

        xops.memory_efficient_attention = _sdpa_fallback
    except Exception:
        pass


def _do_load_model(project_path: str, gpu_id: int, mode: str) -> Dict[str, Any]:
    """Actually load model weights onto the GPU. Called by _load_model."""
    import hydra
    from omegaconf import OmegaConf

    configs_dir = os.path.join(project_path, "configs")
    pretrained_dir = os.path.join(project_path, "pretrained")
    device = f"cuda:{gpu_id}"
    dtype = torch.float16
    torch.cuda.set_device(gpu_id)

    print(f"[seed_x] Loading SEED-X ({mode}) on GPU {gpu_id}...")

    _orig_cwd = os.getcwd()
    try:
        os.chdir(project_path)

        tok_cfg = OmegaConf.load(os.path.join(configs_dir, "tokenizer", "clm_llama_tokenizer_224loc_anyres.yaml"))
        tokenizer = hydra.utils.instantiate(tok_cfg)

        img_tf_cfg = OmegaConf.load(os.path.join(configs_dir, "processer", "qwen_448_transform.yaml"))
        image_transform = hydra.utils.instantiate(img_tf_cfg)

        vis_enc_cfg = OmegaConf.load(os.path.join(configs_dir, "visual_encoder", "qwen_vitg_448.yaml"))
        visual_encoder = hydra.utils.instantiate(vis_enc_cfg)
        visual_encoder = visual_encoder.eval().to(device, dtype=dtype)

        if mode == "editing":
            llm_cfg = OmegaConf.load(os.path.join(configs_dir, "clm_models", "llm_seed_x_edit.yaml"))
        else:
            llm_cfg = OmegaConf.load(os.path.join(configs_dir, "clm_models", "llm_seed_x_i.yaml"))
        llm = hydra.utils.instantiate(llm_cfg, torch_dtype=dtype)

        if mode == "editing":
            agent_cfg = OmegaConf.load(os.path.join(configs_dir, "clm_models", "agent_seed_x_edit.yaml"))
        else:
            agent_cfg = OmegaConf.load(os.path.join(configs_dir, "clm_models", "agent_seed_x_i.yaml"))
        agent_model = hydra.utils.instantiate(agent_cfg, llm=llm)
        agent_model = agent_model.eval().to(device, dtype=dtype)

        adapter = None
        if mode in ("generation", "editing"):
            # diffusers in seedx env imports cached_download which was removed from newer huggingface_hub
            import huggingface_hub as _hfhub
            if not hasattr(_hfhub, 'cached_download'):
                from huggingface_hub import hf_hub_download as _hf_cached
                _hfhub.cached_download = _hf_cached
                import sys as _sys
                _sys.modules['huggingface_hub'].cached_download = _hf_cached
            from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler

            diffusion_path = os.path.join(pretrained_dir, "stable-diffusion-xl-base-1.0")
            noise_scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_path, subfolder="scheduler")
            vae = AutoencoderKL.from_pretrained(diffusion_path, subfolder="vae").to(device, dtype=dtype)
            unet = UNet2DConditionModel.from_pretrained(diffusion_path, subfolder="unet").to(device, dtype=dtype)

            if mode == "editing":
                adapter_cfg_name = "sdxl_qwen_vit_resampler_l4_q64_full_with_latent_image_pretrain_no_normalize.yaml"
            else:
                adapter_cfg_name = "sdxl_qwen_vit_resampler_l4_q64_pretrain_no_normalize.yaml"

            adapter_cfg_path = os.path.join(configs_dir, "sdxl_adapter", adapter_cfg_name)
            adapter_cfg = OmegaConf.load(adapter_cfg_path)
            adapter = hydra.utils.instantiate(adapter_cfg, unet=unet)
            adapter = adapter.eval().to(device, dtype=dtype)

            discrete_model = None
            if mode == "generation":
                discrete_cfg = OmegaConf.load(
                    os.path.join(configs_dir, "discrete_model", "discrete_identity.yaml")
                )
                discrete_model = hydra.utils.instantiate(discrete_cfg).to(device).eval()

            init_pipe_kwargs = dict(
                vae=vae,
                scheduler=noise_scheduler,
                visual_encoder=visual_encoder,
                image_transform=image_transform,
                dtype=dtype,
                device=device,
            )
            # editing adapter's init_pipe does not accept discrete_model (different class)
            if mode == "generation":
                init_pipe_kwargs["discrete_model"] = discrete_model

            adapter.init_pipe(**init_pipe_kwargs)

    finally:
        os.chdir(_orig_cwd)

    boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
    eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]
    bop_token_id = tokenizer.encode(BOP_TOKEN, add_special_tokens=False)[0]
    eop_token_id = tokenizer.encode(EOP_TOKEN, add_special_tokens=False)[0]

    print(f"[seed_x] Model ({mode}) loaded.")
    return {
        "tokenizer": tokenizer,
        "image_transform": image_transform,
        "visual_encoder": visual_encoder,
        "agent_model": agent_model,
        "adapter": adapter,
        "device": device,
        "dtype": dtype,
        "gpu_id": gpu_id,
        "mode": mode,
        "grid_pinpoints": _build_grid_pinpoints(),
        "boi_token_id": boi_token_id,
        "eoi_token_id": eoi_token_id,
        "bop_token_id": bop_token_id,
        "eop_token_id": eop_token_id,
    }


def _load_model(project_path: str, gpu_id: int, mode: str) -> Dict[str, Any]:
    """
    mode: "understanding" | "generation" | "editing"

    Attempts to load without evicting other cached modes (they may coexist on
    high-VRAM GPUs). Falls back to evicting all other modes on OOM.
    """
    cache_key = f"{project_path}_{gpu_id}_{mode}"
    if cache_key in _cached_models:
        return _cached_models[cache_key]
    if cache_key in _failed_models:
        raise _failed_models[cache_key]

    _patch_xformers_fallback()
    _setup_seed_x_path(project_path)

    try:
        cache = _do_load_model(project_path, gpu_id, mode)
    except torch.cuda.OutOfMemoryError:
        stale_keys = [k for k in list(_cached_models.keys())
                      if k.startswith(f"{project_path}_{gpu_id}_")]
        if not stale_keys:
            _failed_models[cache_key] = sys.exc_info()[1]
            raise
        print(f"[seed_x] OOM — evicting {len(stale_keys)} cached mode(s) to free VRAM, then retrying...")
        for k in stale_keys:
            del _cached_models[k]
        import gc; gc.collect()
        torch.cuda.empty_cache()
        try:
            cache = _do_load_model(project_path, gpu_id, mode)
        except Exception as e:
            _failed_models[cache_key] = e
            raise
    except Exception as e:
        _failed_models[cache_key] = e
        raise

    _cached_models[cache_key] = cache
    return cache


def _build_image_tokens(patch_length: int) -> str:
    """Build the image token string (including patch tokens)"""
    tokens = ""
    for _ in range(patch_length - 1):
        tokens += BOP_TOKEN + "".join(IMG_TOKEN.format(i) for i in range(NUM_IMG_TOKENS)) + EOP_TOKEN
    tokens += BOI_TOKEN + "".join(IMG_TOKEN.format(i) for i in range(NUM_IMG_TOKENS)) + EOI_TOKEN
    return tokens


def _encode_image(cache: Dict[str, Any], image_path: str):
    """Return (image_tensor, patch_position, embeds_cmp_mask, ids_cmp_mask_builder)"""
    from inference.any_res import process_anyres_image

    image = Image.open(image_path).convert("RGB")
    image_tensor, patch_pos_tensor = process_anyres_image(
        image, cache["image_transform"], cache["grid_pinpoints"], BASE_RESOLUTION
    )
    embeds_cmp_mask = torch.tensor(
        [True] * image_tensor.shape[0], device=cache["device"], dtype=torch.bool
    )
    patch_position = patch_pos_tensor  # will be cat'd at call site

    image_tensor = image_tensor.to(cache["device"], dtype=cache["dtype"])
    return image_tensor, patch_position, embeds_cmp_mask


def _build_ids_cmp_mask(input_ids: torch.Tensor, cache: Dict[str, Any]) -> torch.Tensor:
    boi_id = cache["boi_token_id"]
    eoi_id = cache["eoi_token_id"]
    bop_id = cache["bop_token_id"]
    eop_id = cache["eop_token_id"]
    ids_cmp_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    boi_indices = torch.where(torch.logical_or(input_ids == boi_id, input_ids == bop_id))[0].tolist()
    eoi_indices = torch.where(torch.logical_or(input_ids == eoi_id, input_ids == eop_id))[0].tolist()
    for boi_idx, eoi_idx in zip(boi_indices, eoi_indices):
        ids_cmp_mask[boi_idx + 1:eoi_idx] = True
    return ids_cmp_mask


def _understanding(cache: Dict[str, Any], image_path: Optional[str], prompt: str,
                   max_new_tokens: int = 512, seed: int = 666) -> str:
    set_seed(seed)

    tokenizer = cache["tokenizer"]
    visual_encoder = cache["visual_encoder"]
    agent_model = cache["agent_model"]
    device = cache["device"]
    dtype = cache["dtype"]

    if image_path and os.path.exists(image_path):
        image_tensor, patch_pos_tensor, embeds_cmp_mask = _encode_image(cache, image_path)
        # patch_pos_tensor is [N_patches, 2] (2D) — keep as-is (official script: torch.cat([patch_pos_tensor], dim=0))
        patch_position = torch.cat([patch_pos_tensor], dim=0)
        image_tokens = _build_image_tokens(image_tensor.shape[0])
    else:
        image_tensor = None
        patch_position = None
        embeds_cmp_mask = None
        image_tokens = ""

    text_prompt = INSTRUCTION_PROMPT.format(instruction=image_tokens + prompt)
    input_ids = tokenizer.encode(text_prompt, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + input_ids
    input_ids = torch.tensor(input_ids, device=device, dtype=torch.long)
    ids_cmp_mask = _build_ids_cmp_mask(input_ids, cache)
    input_ids = input_ids.unsqueeze(0)
    ids_cmp_mask = ids_cmp_mask.unsqueeze(0)

    with torch.no_grad():
        if image_tensor is not None:
            image_embeds = visual_encoder(image_tensor)
            output = agent_model.generate(
                tokenizer=tokenizer,
                input_ids=input_ids,
                image_embeds=image_embeds,
                embeds_cmp_mask=embeds_cmp_mask,
                patch_positions=patch_position,
                ids_cmp_mask=ids_cmp_mask,
                max_new_tokens=max_new_tokens,
                num_img_gen_tokens=NUM_IMG_TOKENS,
            )
        else:
            output = agent_model.generate(
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_img_gen_tokens=NUM_IMG_TOKENS,
            )

    text = re.sub("<[^>]*>", "", output["text"])
    return text.strip()


def _generation(cache: Dict[str, Any], prompt: str, output_path: str,
                num_steps: int = 50, seed: int = 666) -> str:
    set_seed(seed)

    tokenizer = cache["tokenizer"]
    agent_model = cache["agent_model"]
    adapter = cache["adapter"]
    device = cache["device"]

    gen_prompt = GENERATION_PROMPT.format(caption=prompt)
    prompt_ids = tokenizer.encode(gen_prompt, add_special_tokens=False)
    input_ids = torch.tensor(
        [tokenizer.bos_token_id] + prompt_ids, device=device, dtype=torch.long
    ).unsqueeze(0)

    with torch.no_grad():
        output = agent_model.generate(
            tokenizer=tokenizer,
            input_ids=input_ids,
            num_img_gen_tokens=NUM_IMG_TOKENS,
        )

    if not output.get("has_img_output", False):
        raise RuntimeError("[seed_x] Generation failed: model did not produce image output")

    images = adapter.generate(
        image_embeds=output["img_gen_feat"].to(device),
        num_inference_steps=num_steps,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    images[0].save(output_path)
    print(f"[seed_x] Generated image saved to: {output_path}")
    return output_path


def _editing(cache: Dict[str, Any], image_path: str, prompt: str,
             output_path: str, num_steps: int = 50, seed: int = 666) -> str:
    set_seed(seed)

    tokenizer = cache["tokenizer"]
    visual_encoder = cache["visual_encoder"]
    agent_model = cache["agent_model"]
    adapter = cache["adapter"]
    device = cache["device"]
    dtype = cache["dtype"]

    image_tensor, patch_pos_tensor, embeds_cmp_mask = _encode_image(cache, image_path)
    patch_position = torch.cat([patch_pos_tensor], dim=0)
    image_tokens = _build_image_tokens(image_tensor.shape[0])

    text_prompt = INSTRUCTION_PROMPT.format(instruction=image_tokens + prompt)
    input_ids = tokenizer.encode(text_prompt, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + input_ids
    input_ids = torch.tensor(input_ids, device=device, dtype=torch.long)
    ids_cmp_mask = _build_ids_cmp_mask(input_ids, cache)
    input_ids = input_ids.unsqueeze(0)
    ids_cmp_mask = ids_cmp_mask.unsqueeze(0)

    source_image = Image.open(image_path).convert("RGB").resize((1024, 1024))

    with torch.no_grad():
        image_embeds = visual_encoder(image_tensor)
        output = agent_model.generate(
            tokenizer=tokenizer,
            input_ids=input_ids,
            image_embeds=image_embeds,
            embeds_cmp_mask=embeds_cmp_mask,
            patch_positions=patch_position,
            ids_cmp_mask=ids_cmp_mask,
            max_new_tokens=512,
            num_img_gen_tokens=NUM_IMG_TOKENS,
        )

    if not output.get("has_img_output", False):
        raise RuntimeError("[seed_x] Editing failed: model did not produce image output")

    images = adapter.generate(
        image_embeds=output["img_gen_feat"],
        latent_image=source_image,
        num_inference_steps=num_steps,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    images[0].save(output_path)
    print(f"[seed_x] Edited image saved to: {output_path}")
    return output_path


def seed_x_inference_function(
    image_paths: Optional[Union[str, List[str]]],
    prompt: str,
    model_config: Dict[str, Any],
    index: Optional[int] = None,
    round_number: Optional[int] = None,
) -> str:
    project_path = model_config.get("seed_x_project_path", "")
    if project_path:
        _setup_seed_x_path(project_path)

    gpu_id = model_config.get("gpu_id", 0)
    inference_mode = model_config.get("inference_mode", "understanding")
    seed = model_config.get("seed", 666)
    num_steps = model_config.get("num_inference_steps", 50)

    if isinstance(image_paths, list):
        image_path = image_paths[0] if image_paths else None
    else:
        image_path = image_paths

    if inference_mode == "understanding":
        cache = _load_model(project_path, gpu_id, "understanding")
        max_new_tokens = model_config.get("max_new_tokens", 512)
        return _understanding(cache, image_path, prompt, max_new_tokens, seed)

    elif inference_mode == "generation":
        cache = _load_model(project_path, gpu_id, "generation")
        output_path = generate_output_path(model_config, index, "generation", round_number)
        return _generation(cache, prompt, output_path, num_steps, seed)

    elif inference_mode in ("editing", "unify"):
        if not image_path:
            print("[seed_x] Warning: editing mode requires image_path. Returning empty.")
            return ""
        cache = _load_model(project_path, gpu_id, "editing")
        mode_str = "editing" if inference_mode == "editing" else "unify"
        output_path = generate_output_path(model_config, index, mode_str, round_number)
        return _editing(cache, image_path, prompt, output_path, num_steps, seed)

    else:
        raise ValueError(f"[seed_x] Unknown inference_mode: {inference_mode}")
