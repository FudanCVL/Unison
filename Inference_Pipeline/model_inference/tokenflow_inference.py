#!/usr/bin/env python3
"""
TokenFlow model inference adapter

Supported modes:
  - understanding: i2t LLaVA-Qwen model, using VQTower to encode images (tokenflow_siglip_32k.pt)
  - generation:    t2i autoregressive model (autoregressive_infer_cfg), see run_llava_samples.py
  - editing:       not supported

The config JSON must specify:
  - understanding_model_path: ByteFlow-AI/TokenFlow-llava-qwen2.5-14B-finetuning
  - generation_model_path:    ByteFlow-AI/TokenFlow-t2i (corrected, without the -7B suffix)
  - tokenizer_path:           <TokenFlow dir>/pretrained_ckpts/tokenflow_clipb_32k_enhanced.pt
  - tokenflow_project_path:   /home/.../TokenFlow
"""
import os
import sys
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from model_inference import generate_output_path, set_seed, load_pil_image

_cached_models: Dict[str, Any] = {}
_failed_models: Dict[str, Exception] = {}

# TokenFlow VQ tokenizer .pt path (siglip for understanding; clipb for generation)
_SIGLIP_TOKENIZER = None   # lazy-loaded, set on first use


def _setup_paths(project_path: str, mode: str) -> None:
    if mode == "understanding":
        path = os.path.join(project_path, "i2t")
    else:
        path = os.path.join(project_path, "t2i")
    if path not in sys.path:
        sys.path.insert(0, path)


def _get_siglip_tokenizer_path(project_path: str) -> str:
    """Return the VQ tokenizer path used for understanding (tokenflow_siglip_32k.pt)"""
    return os.path.join(project_path, "pretrained_ckpts", "tokenflow_siglip_32k.pt")


def _load_understanding_model(model_path: str, gpu_id: int, project_path: str) -> Dict[str, Any]:
    cache_key = f"und_{model_path}_{gpu_id}"
    if cache_key in _cached_models:
        return _cached_models[cache_key]
    if cache_key in _failed_models:
        raise _failed_models[cache_key]

    _setup_paths(project_path, "understanding")

    print(f"[tokenflow] Loading understanding model {model_path} on GPU {gpu_id}...")
    try:
        from llava.model.builder import load_pretrained_model

        siglip_path = _get_siglip_tokenizer_path(project_path)

        # mm_vision_tower is an empty string in the HF config → build_vision_tower raises
        # "Unknown vision tower" due to the empty path. Monkey-patch the builder before loading to inject the correct path.
        import llava.model.multimodal_encoder.builder as _bld
        import llava.model.llava_arch as _arch
        _orig_bvt = _bld.build_vision_tower

        def _patched_bvt(vision_tower_cfg, **kwargs):
            vt = getattr(vision_tower_cfg, "mm_vision_tower", None) or ""
            if (not vt) and siglip_path and os.path.exists(siglip_path):
                vision_tower_cfg.mm_vision_tower = siglip_path
                vision_tower_cfg.mm_vision_vq_type = "TOKENFLOW"
            return _orig_bvt(vision_tower_cfg, **kwargs)

        _bld.build_vision_tower = _patched_bvt
        _arch.build_vision_tower = _patched_bvt

        try:
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=os.path.basename(model_path),
                device_map=f"cuda:{gpu_id}",
            )
        finally:
            _bld.build_vision_tower = _orig_bvt
            _arch.build_vision_tower = _orig_bvt

        # During device_map loading the checkpoint has no VQTower weights → vision_tower becomes a meta tensor (no-op warning).
        # After from_pretrained, the inner tokenflow model must be force-reloaded from the .pt file.
        vt = model.get_vision_tower()
        if vt is not None:
            from llava.model.multimodal_encoder.vision_tokenizer import tokenflow_model as _tfm
            # tokenflow_model is already eval()'d internally, but vq_model.train is replaced with a no-op (does not return self).
            # We cannot chain .eval(); fetch the model and operate step by step instead.
            _new_vt = _tfm(
                "TokenFlow", codebook_size=32768, teacher="siglip_384",
                pretrain_path=siglip_path
            )
            _new_vt.requires_grad_(False)
            _new_vt = _new_vt.to(device=f"cuda:{gpu_id}", dtype=torch.float16)
            vt.vision_tower = _new_vt
            vt.is_loaded = True
            if image_processor is None:
                from llava.model.multimodal_encoder.vision_tokenizer import SigLipImageProcessor
                image_processor = SigLipImageProcessor()

    except Exception as e:
        _failed_models[cache_key] = e
        raise

    cache = {
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "gpu_id": gpu_id,
    }
    _cached_models[cache_key] = cache
    print(f"[tokenflow] Understanding model loaded.")
    return cache


def _load_generation_model(model_path: str, tokenizer_path: str, gpu_id: int, project_path: str) -> Dict[str, Any]:
    cache_key = f"gen_{model_path}_{gpu_id}"
    if cache_key in _cached_models:
        return _cached_models[cache_key]
    if cache_key in _failed_models:
        raise _failed_models[cache_key]

    _setup_paths(project_path, "generation")

    print(f"[tokenflow] Loading generation model {model_path} on GPU {gpu_id}...")
    torch.cuda.set_device(gpu_id)
    try:
        import transformers
        from llava_t2i.model import LlavaLlamaForCausalLM

        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            attn_implementation="eager",
            mm_vision_tower=tokenizer_path or model_path,
        )
        model = model.to(torch.bfloat16).cuda(gpu_id).eval()
        model.config.mm_vision_vq_type = str(model.config.mm_vision_vq_type)
        model.config.use_cache = False

        text_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=model.config.tokenizer_model_max_length,
            padding_side="right",
            use_fast=False,
        )
        # Initialize image special tokens (required for t2i)
        model.reinit_image_token_start_end(text_tokenizer)

        # VQTower.vq_model inside a list is not necessarily handled correctly by model.to(bfloat16).cuda();
        # move it explicitly to the correct device + dtype
        vision_tower = model.get_vision_tower()
        if vision_tower is not None and hasattr(vision_tower, "vq_model"):
            vision_tower.vq_model = [
                vision_tower.vq_model[0].to(device=f"cuda:{gpu_id}", dtype=torch.bfloat16)
            ]

    except Exception as e:
        _failed_models[cache_key] = e
        raise

    cache = {
        "tokenizer": text_tokenizer,
        "model": model,
        "gpu_id": gpu_id,
    }
    _cached_models[cache_key] = cache
    print(f"[tokenflow] Generation model loaded.")
    return cache


def _understanding(cache: Dict[str, Any], image_path: Optional[str], prompt: str,
                   max_new_tokens: int = 512, seed: int = 666) -> str:
    """LLaVA-based image-text understanding (VQTower encodes the image)"""
    set_seed(seed)
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

    tokenizer = cache["tokenizer"]
    model = cache["model"]
    image_processor = cache["image_processor"]
    gpu_id = cache["gpu_id"]
    device = f"cuda:{gpu_id}"

    if image_path and os.path.exists(image_path) and image_processor is not None:
        image = load_pil_image(image_path)
        image_tensor = process_images([image], image_processor, model.config)
        if isinstance(image_tensor, list):
            image_tensor = [t.to(device, dtype=torch.float16) for t in image_tensor]
        else:
            image_tensor = image_tensor.to(device, dtype=torch.float16)
        qs = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
    else:
        image_tensor = None
        qs = prompt

    # The i2t model is Qwen2.5-based, so the corresponding conversation template is qwen_2_5
    conv = conv_templates["qwen_2_5"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_str = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    # llava_qwen.generate() calls super().generate() with inputs_embeds internally,
    # so output_ids contains only the newly generated tokens, without the input prefix; decode directly.
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text.strip()


def _generation(cache: Dict[str, Any], prompt: str, output_path: str,
                cfg_scale: float = 7.5, seed: int = 666) -> str:
    """Text-to-image generation (see run_llava_samples.py)"""
    set_seed(seed)

    from llava_t2i.dataset.process import crop_and_encode_text_and_img

    tokenizer = cache["tokenizer"]
    model = cache["model"]
    gpu_id = cache["gpu_id"]

    negative_prompt = (
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
        "fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, "
        "signature, watermark, username, blurry."
    )

    # Encode the positive and negative prompts; the results must be on the model's device
    device_str = f"cuda:{gpu_id}"
    input_id, _ = crop_and_encode_text_and_img(tokenizer, prompt, image=None, max_text_token_num=128)
    uncond_id, _ = crop_and_encode_text_and_img(tokenizer, negative_prompt, image=None, max_text_token_num=128)
    # Move to GPU (the official demo uses torch.set_default_tensor_type('cuda'); here we explicitly to(device))
    if isinstance(input_id, torch.Tensor):
        input_id = input_id.to(device_str)
    if isinstance(uncond_id, torch.Tensor):
        uncond_id = uncond_id.to(device_str)
    prefix_text_codes = [input_id, uncond_id]  # conditioned + unconditioned

    # autoregressive_infer_cfg internally calls operations like torch.zeros(),
    # which default to CPU; use set_default_device to ensure all new tensors are on the GPU
    _prev_device = torch.device("cpu")
    try:
        torch.set_default_device(device_str)
    except Exception:
        pass

    try:
        with torch.inference_mode():
            samples = model.autoregressive_infer_cfg(
                B=1,
                prefix_text_codes=prefix_text_codes,
                cfg=cfg_scale,
                topk_list=[600],
                topp_list=[0.6],
                g_seed=seed,
            )
    finally:
        try:
            torch.set_default_device("cpu")
        except Exception:
            pass

    # autoregressive_infer_cfg returns [B, H, W, 3] (HWC uint8, numpy or tensor)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if isinstance(samples, np.ndarray):
        Image.fromarray(samples[0].astype(np.uint8)).save(output_path)
    elif isinstance(samples, torch.Tensor):
        img = samples[0].cpu()
        if img.ndim == 3 and img.shape[-1] in (1, 3, 4):  # HWC
            Image.fromarray(img.numpy().astype(np.uint8)).save(output_path)
        else:  # CHW
            from torchvision.transforms.functional import to_pil_image
            to_pil_image(img.clamp(0, 1)).save(output_path)
    elif isinstance(samples, list):
        item = samples[0]
        if isinstance(item, Image.Image):
            item.save(output_path)
        elif isinstance(item, np.ndarray):
            Image.fromarray(item.astype(np.uint8)).save(output_path)
        else:
            from torchvision.transforms.functional import to_pil_image
            to_pil_image(item.cpu().clamp(0, 1)).save(output_path)
    elif isinstance(samples, Image.Image):
        samples.save(output_path)
    else:
        raise ValueError(f"[tokenflow] Unexpected samples type: {type(samples)}")

    print(f"[tokenflow] Generated image saved to: {output_path}")
    return output_path


def tokenflow_inference_function(
    image_paths: Optional[Union[str, List[str]]],
    prompt: str,
    model_config: Dict[str, Any],
    index: Optional[int] = None,
    round_number: Optional[int] = None,
) -> str:
    project_path = model_config.get("tokenflow_project_path", "")
    gpu_id = model_config.get("gpu_id", 0)
    inference_mode = model_config.get("inference_mode", "understanding")
    seed = model_config.get("seed", 666)

    if isinstance(image_paths, list):
        image_path = image_paths[0] if image_paths else None
    else:
        image_path = image_paths

    if inference_mode == "understanding":
        model_path = model_config.get("understanding_model_path", model_config.get("model_path", ""))
        cache = _load_understanding_model(model_path, gpu_id, project_path)
        max_new_tokens = model_config.get("max_new_tokens", 512)
        return _understanding(cache, image_path, prompt, max_new_tokens, seed)

    elif inference_mode == "generation":
        model_path = model_config.get("generation_model_path", model_config.get("model_path", ""))
        tokenizer_path = model_config.get("tokenizer_path", "")
        cache = _load_generation_model(model_path, tokenizer_path, gpu_id, project_path)
        output_path = generate_output_path(model_config, index, "generation", round_number)
        cfg_scale = model_config.get("cfg_scale", 7.5)
        return _generation(cache, prompt, output_path, cfg_scale, seed)

    elif inference_mode in ("editing", "unify"):
        print(f"[tokenflow] Warning: TokenFlow does not support '{inference_mode}' mode. Returning empty.")
        return ""

    else:
        raise ValueError(f"[tokenflow] Unknown inference_mode: {inference_mode}")
