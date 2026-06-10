#!/usr/bin/env python3
"""
Show-o2 model inference adapter

Supported modes:
  - understanding: multimodal QA (mmu_generate), see infer_understanding.py
  - generation:    text-to-image (ODE flow-matching sampler), see infer_t2i.py
  - editing:       not supported

Depends on the models / utils / transport / datasets modules under the Show-o project's show-o2/ subdirectory.
The config JSON must specify:
  - model_path:         showlab/show-o2-7B
  - llm_model_path:     Qwen/Qwen2.5-7B-Instruct
  - vae_model_path:     <show-o2 dir>/Wan2.1_VAE.pth (absolute path)
  - showo_project_path: /home/.../Show-o
"""
import os
import sys
import warnings
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union

from PIL import Image

# accelerate loads .bin checkpoints with weights_only=False, triggering a harmless PyTorch FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="accelerate")

from model_inference import generate_output_path, set_seed, load_pil_image

_cached_models: Dict[str, Any] = {}
_failed_models: Dict[str, Exception] = {}


def _setup_paths(project_path: str) -> None:
    showo2_path = os.path.join(project_path, "show-o2")
    # Insert project_path first, then showo2_path; the last inserted ends up at sys.path[0],
    # ensuring the models package in show-o2/ takes priority over the top-level Show-o/models/
    for p in [project_path, showo2_path]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _resolve_model_path(model_path: str) -> str:
    """Resolve a HuggingFace Hub ID to a local cache snapshot path.

    With HF_HUB_OFFLINE=1, diffusers' from_pretrained still calls the model_info() API for a Hub ID,
    causing an offline error. Passing a local path skips all API calls.
    """
    if os.path.isdir(model_path):
        return model_path
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        model_id_dir = "models--" + model_path.replace("/", "--")
        snapshots_dir = os.path.join(HF_HUB_CACHE, model_id_dir, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshots = sorted(os.listdir(snapshots_dir))
            if snapshots:
                resolved = os.path.join(snapshots_dir, snapshots[-1])
                print(f"[showo2] Resolved '{model_path}' to local path: {resolved}")
                return resolved
    except Exception:
        pass
    return model_path


def _load_model(model_path: str, llm_model_path: str, vae_model_path: str,
                gpu_id: int, project_path: str) -> Dict[str, Any]:
    cache_key = f"{model_path}_{gpu_id}"
    if cache_key in _cached_models:
        return _cached_models[cache_key]
    if cache_key in _failed_models:
        raise _failed_models[cache_key]

    # Inject a torch.nn.attention compatibility shim before the show-o2 module is first imported
    _patch_torch_nn_attention()
    _setup_paths(project_path)

    print(f"[showo2] Loading Show-o2 model on GPU {gpu_id}...")
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)
    # cuDNN in the showo2 conda env cannot initialize inside a subprocess (CUDNN_STATUS_NOT_INITIALIZED).
    # WanVAE uses 3D convolutions, and PyTorch has a native CUDA (non-cuDNN) fallback path, so globally disabling cuDNN works.
    torch.backends.cudnn.enabled = False

    try:
        from omegaconf import OmegaConf
        from models import Showo2Qwen2_5, WanVAE
        from models.misc import get_text_tokenizer
        from utils import path_to_llm_name, get_hyper_params

        # Load the official demo config (7B 432x432) and read all hyperparameters from it
        showo2_dir = os.path.join(project_path, "show-o2")
        cfg_path = os.path.join(showo2_dir, "configs", "showo2_7b_demo_432x432.yaml")
        config = OmegaConf.load(cfg_path)

        # Override the VAE path with an absolute path
        config.model.vae_model.pretrained_model_path = vae_model_path

        # Both tasks use float32 (the model is loaded as float32, and generation's attn_mask must match too)
        weight_type_und = torch.float32
        weight_type_gen = torch.float32

        llm_name = path_to_llm_name.get(llm_model_path, "qwen2.5")
        text_tokenizer, showo_token_ids = get_text_tokenizer(
            llm_model_path,
            add_showo_tokens=True,
            return_showo_token_ids=True,
            llm_name=llm_name,
        )
        config.model.showo.llm_vocab_size = len(text_tokenizer)

        # WanVAE: build with float32 (required by the understanding demo)
        vae_model = WanVAE(vae_pth=vae_model_path, dtype=weight_type_und, device=device)

        # show-o2-7B ships .bin shards, so use_safetensors=False is required
        # Resolve the Hub ID to a local path to avoid diffusers calling the model_info() API in offline mode
        local_model_path = _resolve_model_path(model_path)
        model = Showo2Qwen2_5.from_pretrained(local_model_path, use_safetensors=False)
        model = model.to(device).to(weight_type_und).eval()

        # add_time_embeds = True for 7B model
        if config.model.showo.add_time_embeds:
            config.dataset.preprocessing.num_t2i_image_tokens += 1
            config.dataset.preprocessing.num_mmu_image_tokens += 1

        hyper = get_hyper_params(config, text_tokenizer, showo_token_ids)
        (num_t2i_image_tokens, num_mmu_image_tokens, num_video_tokens,
         max_seq_len, max_text_len, image_latent_dim, patch_size,
         latent_width, latent_height,
         pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id,
         img_pad_id, vid_pad_id, guidance_scale) = hyper

    except Exception as e:
        _failed_models[cache_key] = e
        raise

    cache = {
        "model": model,
        "vae_model": vae_model,
        "text_tokenizer": text_tokenizer,
        "showo_token_ids": showo_token_ids,
        "config": config,
        "device": device,
        "weight_type_und": weight_type_und,
        "weight_type_gen": weight_type_gen,
        "gpu_id": gpu_id,
        # hyper params
        "num_t2i_image_tokens": num_t2i_image_tokens,
        "num_mmu_image_tokens": num_mmu_image_tokens,
        "max_seq_len": max_seq_len,
        "max_text_len": max_text_len,
        "image_latent_dim": image_latent_dim,
        "patch_size": patch_size,
        "latent_width": latent_width,
        "latent_height": latent_height,
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "boi_id": boi_id,
        "eoi_id": eoi_id,
        "img_pad_id": img_pad_id,
        "guidance_scale": guidance_scale,
    }
    _cached_models[cache_key] = cache
    print(f"[showo2] Model loaded.")
    return cache


def _understanding(cache: Dict[str, Any], image_path: Optional[str], prompt: str,
                   max_new_tokens: int = 512, seed: int = 666) -> str:
    """Multimodal understanding (see infer_understanding.py)"""
    set_seed(seed)

    from models import omni_attn_mask_naive
    from datasets.utils import image_transform

    model = cache["model"]
    text_tokenizer = cache["text_tokenizer"]
    showo_token_ids = cache["showo_token_ids"]
    vae_model = cache["vae_model"]
    config = cache["config"]
    device = cache["device"]
    weight_type = cache["weight_type_und"]
    num_mmu_image_tokens = cache["num_mmu_image_tokens"]
    bos_id = cache["bos_id"]
    boi_id = cache["boi_id"]
    eoi_id = cache["eoi_id"]

    top_k = 1

    sys_prompt_ids = text_tokenizer(
        "system\nYou are a helpful assistant.<|im_end|>", add_special_tokens=False
    )["input_ids"]
    role_a = text_tokenizer("\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    role_b = text_tokenizer("\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]

    if image_path and os.path.exists(image_path):
        resolution = config.dataset.preprocessing.resolution
        image_ori = load_pil_image(image_path)
        image = image_transform(image_ori, resolution=resolution).to(device)
        image = image.unsqueeze(0)
        image_latents = vae_model.sample(image.unsqueeze(2)).squeeze(2).to(weight_type)
        image_embeds_und = model.image_embedder_und(image_latents)
        image_embeds_gen = model.image_embedder_gen(image_latents)
        image_embeds_und = image_embeds_und + model.position_embedding(model.image_position_ids)
        image_embeds_und = model.und_trans(image_embeds_und)["last_hidden_state"]
        image_embeds = model.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))
        has_image = True
    else:
        image_embeds = None
        has_image = False

    input_ids = text_tokenizer(prompt, add_special_tokens=False).input_ids
    text_tokens_a = torch.tensor([bos_id] + sys_prompt_ids + role_a).to(device)[None, :]
    text_tokens_b = torch.tensor([boi_id, eoi_id] + input_ids + role_b).to(device)[None, :]
    text_embeds_a = model.showo.model.embed_tokens(text_tokens_a)
    text_embeds_b = model.showo.model.embed_tokens(text_tokens_b)

    add_time_embeds = config.model.showo.add_time_embeds
    if has_image:
        if add_time_embeds:
            time_embeds = model.time_embed(torch.Tensor([[1.0]]).to(device), text_embeds_a.dtype)
            if hasattr(model, "time_embed_proj"):
                time_embeds = model.time_embed_proj(time_embeds)
            input_embeds = torch.cat(
                [text_embeds_a, text_embeds_b[:, :1], time_embeds, image_embeds, text_embeds_b[:, 1:]],
                dim=1,
            ).to(weight_type)
            modality_positions = torch.tensor(
                [text_tokens_a.shape[1] + 2, num_mmu_image_tokens]
            )[None, None, :].to(device)
        else:
            input_embeds = torch.cat(
                [text_embeds_a, text_embeds_b[:, :1], image_embeds, text_embeds_b[:, 1:]],
                dim=1,
            ).to(weight_type)
            modality_positions = torch.tensor(
                [text_tokens_a.shape[1] + 1, num_mmu_image_tokens]
            )[None, None, :].to(device)

        attention_mask = omni_attn_mask_naive(
            B=input_embeds.size(0),
            LEN=input_embeds.size(1),
            modalities=modality_positions,
            device=device,
            inverted=True,
        ).to(input_embeds.dtype)
    else:
        input_embeds = torch.cat([text_embeds_a, text_embeds_b], dim=1).to(weight_type)
        attention_mask = None

    with torch.no_grad():
        output_tokens = model.mmu_generate(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            eos_token=text_tokenizer.eos_token_id,
        )

    output_tokens = torch.stack(output_tokens).squeeze()[None]
    text = text_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    return text[0].strip()


def _patch_torch_nn_attention() -> None:
    """PyTorch < 2.0 lacks the torch.nn.attention submodule; inject a compatibility shim so show-o2 transport/models can import correctly."""
    import sys
    import torch.nn as _nn
    if hasattr(_nn, "attention"):
        return
    from contextlib import contextmanager
    from enum import IntEnum

    class SDPBackend(IntEnum):
        ERROR = -1
        MATH = 0
        FLASH_ATTENTION = 1
        EFFICIENT_ATTENTION = 2
        CUDNN_ATTENTION = 3

    @contextmanager
    def sdpa_kernel(*args, **kwargs):
        yield

    attn_mod = type(sys)("torch.nn.attention")
    attn_mod.SDPBackend = SDPBackend
    attn_mod.sdpa_kernel = sdpa_kernel
    _nn.attention = attn_mod
    sys.modules["torch.nn.attention"] = attn_mod


def _generation(cache: Dict[str, Any], prompt: str, output_path: str,
                num_steps: int = 50, guidance_scale_override: Optional[float] = None,
                seed: int = 666) -> str:
    """Text-to-image generation (ODE flow-matching, see infer_t2i.py)"""
    set_seed(seed)

    from models.misc import prepare_gen_input
    from models import omni_attn_mask_naive
    from transport import create_transport, Sampler
    from utils import denorm

    model = cache["model"]
    vae_model = cache["vae_model"]
    text_tokenizer = cache["text_tokenizer"]
    showo_token_ids = cache["showo_token_ids"]
    config = cache["config"]
    device = cache["device"]
    weight_type = cache["weight_type_gen"]
    num_t2i_image_tokens = cache["num_t2i_image_tokens"]
    max_seq_len = cache["max_seq_len"]
    max_text_len = cache["max_text_len"]
    image_latent_dim = cache["image_latent_dim"]
    patch_size = cache["patch_size"]
    latent_width = cache["latent_width"]
    latent_height = cache["latent_height"]
    pad_id = cache["pad_id"]
    bos_id = cache["bos_id"]
    eos_id = cache["eos_id"]
    boi_id = cache["boi_id"]
    eoi_id = cache["eoi_id"]
    img_pad_id = cache["img_pad_id"]
    guidance_scale = guidance_scale_override if guidance_scale_override is not None else cache["guidance_scale"]

    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=num_t2i_image_tokens,
    )
    sampler = Sampler(transport)

    prompts = [prompt]
    batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null = \
        prepare_gen_input(
            prompts, text_tokenizer, num_t2i_image_tokens,
            bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id,
            max_text_len, device
        )

    z = torch.randn(
        (1, image_latent_dim, latent_height * patch_size, latent_width * patch_size),
        dtype=weight_type, device=device
    )

    if guidance_scale > 0:
        z = torch.cat([z, z], dim=0)
        text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
        modality_positions = torch.cat([batch_modality_positions, batch_modality_positions_null], dim=0)
    else:
        text_tokens = batch_text_tokens
        modality_positions = batch_modality_positions

    block_mask = omni_attn_mask_naive(
        text_tokens.size(0), max_seq_len, modality_positions, device
    ).to(weight_type)

    model_kwargs = dict(
        text_tokens=text_tokens,
        attention_mask=block_mask,
        modality_positions=modality_positions,
        output_hidden_states=True,
        max_seq_len=max_seq_len,
        guidance_scale=guidance_scale,
    )

    sample_fn = sampler.sample_ode(
        sampling_method=config.transport.sampling_method,
        num_steps=num_steps,
        atol=config.transport.atol,
        rtol=config.transport.rtol,
        reverse=config.transport.reverse,
        time_shifting_factor=config.transport.time_shifting_factor,
    )

    with torch.no_grad():
        samples = sample_fn(z, model.t2i_generate, **model_kwargs)[-1]

    if guidance_scale > 0:
        samples = torch.chunk(samples, 2)[0]

    samples = samples.unsqueeze(2)
    images = vae_model.batch_decode(samples)
    images = images.squeeze(2)
    images = denorm(images)   # → numpy uint8 array (B, H, W, 3)

    pil_img = Image.fromarray(images[0])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pil_img.save(output_path)
    print(f"[showo2] Generated image saved to: {output_path}")
    return output_path


def showo2_inference_function(
    image_paths: Optional[Union[str, List[str]]],
    prompt: str,
    model_config: Dict[str, Any],
    index: Optional[int] = None,
    round_number: Optional[int] = None,
) -> str:
    project_path = model_config.get("showo_project_path", "")
    if project_path:
        _setup_paths(project_path)

    model_path = model_config.get("model_path", "showlab/show-o2-7B")
    llm_model_path = model_config.get("llm_model_path", "Qwen/Qwen2.5-7B-Instruct")
    vae_model_path = model_config.get("vae_model_path", model_path)
    gpu_id = model_config.get("gpu_id", 0)
    inference_mode = model_config.get("inference_mode", "understanding")
    seed = model_config.get("seed", 666)

    if isinstance(image_paths, list):
        image_path = image_paths[0] if image_paths else None
    else:
        image_path = image_paths

    cache = _load_model(model_path, llm_model_path, vae_model_path, gpu_id, project_path)

    if inference_mode == "understanding":
        max_new_tokens = model_config.get("max_new_tokens", 512)
        return _understanding(cache, image_path, prompt, max_new_tokens, seed)

    elif inference_mode == "generation":
        output_path = generate_output_path(model_config, index, "generation", round_number)
        num_steps = model_config.get("num_inference_steps", 50)
        guidance_scale = model_config.get("guidance_scale", None)
        return _generation(cache, prompt, output_path, num_steps, guidance_scale, seed)

    elif inference_mode in ("editing", "unify"):
        print(f"[showo2] Warning: Show-o2 does not support '{inference_mode}' mode. Returning empty.")
        return ""

    else:
        raise ValueError(f"[showo2] Unknown inference_mode: {inference_mode}")
