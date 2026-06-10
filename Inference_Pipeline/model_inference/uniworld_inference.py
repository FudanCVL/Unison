#!/usr/bin/env python3
"""
UniWorld-V1 model inference adapter

Supported modes:
  - understanding: Qwen2.5-VL text generation (model.generate())
  - generation:    FLUX.1 text-to-image (FluxPipeline)
  - editing:       FLUX.1-based image editing (FluxPipeline + image condition)

Local weight paths (confirmed to exist):
  - model_path:   model_weights/UniWorld-V1/
  - flux_path:    model_weights/FLUX.1-dev/
  - siglip_path:  model_weights/siglip2-so400m-patch16-512/

The config JSON must specify uniworld_project_path pointing to the UniWorld-V1/ code directory.
"""
import os
import sys
import torch
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from model_inference import generate_output_path, set_seed, load_pil_image

_cached_models: Dict[str, Any] = {}
_failed_models: Dict[str, Exception] = {}


def _setup_paths(project_path: str) -> None:
    if project_path not in sys.path:
        sys.path.insert(0, project_path)


def _load_model(model_path: str, flux_path: str, siglip_path: str,
                gpu_id: int, project_path: str) -> Dict[str, Any]:
    cache_key = f"{model_path}_{gpu_id}"
    if cache_key in _cached_models:
        return _cached_models[cache_key]
    if cache_key in _failed_models:
        raise _failed_models[cache_key]

    _setup_paths(project_path)

    print(f"[uniworld] Loading UniWorld-V1 on GPU {gpu_id}...")
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)

    try:
        from transformers import AutoProcessor
        from transformers import SiglipImageProcessor, SiglipVisionModel
        from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import UnivaQwen2p5VLForConditionalGeneration
        from univa.utils.flux_pipeline import FluxPipeline

        # Load main model
        model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device).eval()

        processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=448 * 448,
            max_pixels=448 * 448,
        )

        # Load FLUX pipeline (uses model's denoise_tower)。
        # pipe.tokenizer/tokenizer_2 + text_encoder/text_encoder_2 are used for T5/CLIP prompt encoding.
        pipe = FluxPipeline.from_pretrained(
            flux_path,
            transformer=model.denoise_tower.denoiser,
            torch_dtype=torch.bfloat16,
        ).to(device)
        tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
        text_encoders = [pipe.text_encoder, pipe.text_encoder_2]

        # Load SigLIP (needed by the editing/image-condition branch)
        siglip_processor, siglip_model = None, None
        if siglip_path:
            siglip_processor = SiglipImageProcessor.from_pretrained(siglip_path)
            siglip_model = SiglipVisionModel.from_pretrained(
                siglip_path, torch_dtype=torch.bfloat16,
            ).to(device)

        # Load task head (understanding vs generation routing)
        task_head_path = os.path.join(model_path, "task_head_final.pt")
        if os.path.exists(task_head_path):
            import torch.nn as nn
            task_head_state = torch.load(task_head_path, map_location=device)
            in_dim = task_head_state["0.weight"].shape[1]
            mid_dim = task_head_state["0.weight"].shape[0]
            out_dim = task_head_state["3.weight"].shape[0]
            task_head = nn.Sequential(
                nn.Linear(in_dim, mid_dim),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(mid_dim, out_dim),
            ).to(device)
            task_head.load_state_dict(task_head_state)
            task_head.eval()
        else:
            task_head = None
    except Exception as e:
        _failed_models[cache_key] = e
        raise

    cache = {
        "model": model,
        "processor": processor,
        "pipe": pipe,
        "tokenizers": tokenizers,
        "text_encoders": text_encoders,
        "siglip_processor": siglip_processor,
        "siglip_model": siglip_model,
        "task_head": task_head,
        "device": device,
        "gpu_id": gpu_id,
    }
    _cached_models[cache_key] = cache
    print(f"[uniworld] Model loaded.")
    return cache


def _understanding(cache: Dict[str, Any], image_paths_raw: Optional[Union[str, List[str]]],
                   prompt: str, max_new_tokens: int = 512, seed: int = 666) -> str:
    """Qwen2.5-VL multimodal QA (supports multiple images, e.g. ME evaluation rounds passing the original + edited image together)"""
    set_seed(seed)
    from qwen_vl_utils import process_vision_info

    model = cache["model"]
    processor = cache["processor"]
    device = cache["device"]

    if isinstance(image_paths_raw, list):
        img_paths = [p for p in image_paths_raw if p and os.path.exists(p)]
    elif image_paths_raw and os.path.exists(image_paths_raw):
        img_paths = [image_paths_raw]
    else:
        img_paths = []

    messages_content = []
    for img_path in img_paths:
        messages_content.append({"type": "image", "image": img_path})
    messages_content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": messages_content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()


def _preprocess_siglip(siglip_model, siglip_processor, image_paths: List[str]):
    """SigLIP-encode the input images and return last_hidden_state (image condition)"""
    pixel_values = []
    for p in image_paths:
        pv = siglip_processor.preprocess(
            images=Image.open(p).convert("RGB"),
            do_resize=True, return_tensors="pt", do_convert_rgb=True,
        ).pixel_values
        pixel_values.append(pv)
    pixel_values = torch.concat(pixel_values).to(siglip_model.device, dtype=siglip_model.dtype)
    return siglip_model(pixel_values).last_hidden_state


def _generate_or_edit(cache: Dict[str, Any], image_path: Optional[str], prompt: str,
                      output_path: str, num_inference_steps: int = 28,
                      guidance_scale: float = 3.5, height: int = 1024, width: int = 1024,
                      seed: int = 666, is_editing: bool = False) -> str:
    """
    UniWorld-V1 generation/editing. Replicates the generation branch of univa/serve/cli.py:
      1) Build the conversation (includes the input image when editing) → processor → inputs
      2) When editing, SigLIP-encode the input image → siglip_hidden_states
      3) model(..., output_type="denoise_embeds") → LVLM image-condition embeds
      4) encode_prompt(T5/CLIP) → t5_prompt_embeds, pooled_prompt_embeds
      5) concat([t5, lvlm]) as prompt_embeds fed to FluxPipeline
    """
    set_seed(seed)
    from qwen_vl_utils import process_vision_info
    from univa.utils.denoiser_prompt_embedding_flux import encode_prompt
    from univa.utils.anyres_util import dynamic_resize

    model = cache["model"]
    processor = cache["processor"]
    pipe = cache["pipe"]
    tokenizers = cache["tokenizers"]
    text_encoders = cache["text_encoders"]
    siglip_model = cache["siglip_model"]
    siglip_processor = cache["siglip_processor"]
    device = cache["device"]
    generator = torch.Generator(device=device).manual_seed(seed)

    # ---- Build conversation content ----
    content = [{"type": "text", "text": prompt}]
    image_paths: List[str] = []
    new_h, new_w = height, width
    if is_editing and image_path and os.path.exists(image_path):
        content.append({"type": "image", "image": image_path,
                        "min_pixels": 448 * 448, "max_pixels": 448 * 448})
        image_paths.append(image_path)
        # When editing, the output size follows the input image (matching cli.py's update_size)
        with Image.open(image_path) as im:
            iw, ih = im.size
        new_h, new_w = dynamic_resize(int(ih), int(iw), "any_11ratio",
                                      anchor_pixels=height * width)

    conversation = [{"role": "user", "content": content}]
    chat_text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    # Drop the system segment (matching cli.py)
    parts = chat_text.split("<|im_end|>\n")
    if len(parts) > 1:
        chat_text = "<|im_end|>\n".join(parts[1:])
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[chat_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

    # ---- SigLIP image condition ----
    siglip_hidden_states = None
    if is_editing and siglip_model is not None and image_paths:
        siglip_hidden_states = _preprocess_siglip(siglip_model, siglip_processor, image_paths)

    # ---- LVLM denoise embeds ----
    with torch.no_grad():
        lvlm_embeds = model(
            inputs.input_ids,
            pixel_values=getattr(inputs, "pixel_values", None),
            attention_mask=inputs.attention_mask,
            image_grid_thw=getattr(inputs, "image_grid_thw", None),
            siglip_hidden_states=siglip_hidden_states,
            output_type="denoise_embeds",
        )

        # ---- T5/CLIP prompt embeds and concatenation ----
        t5_prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, 256, device, 1,
        )
        input_embeds = torch.concat([t5_prompt_embeds, lvlm_embeds], dim=1)

        result = pipe(
            prompt_embeds=input_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            height=new_h,
            width=new_w,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        )

    out_img = result.images[0]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_img.save(output_path)
    print(f"[uniworld] Image saved to: {output_path}")
    return output_path


def uniworld_inference_function(
    image_paths: Optional[Union[str, List[str]]],
    prompt: str,
    model_config: Dict[str, Any],
    index: Optional[int] = None,
    round_number: Optional[int] = None,
) -> str:
    """
    UniWorld-V1 inference entry function

    Returns:
        understanding mode → text string
        generation / editing / unify mode → saved image path
    """
    project_path = model_config.get("uniworld_project_path", "")
    if project_path:
        _setup_paths(project_path)

    model_path = model_config.get("model_path", "")
    flux_path = model_config.get("flux_path", "black-forest-labs/FLUX.1-dev")
    siglip_path = model_config.get("siglip_path", "")
    gpu_id = model_config.get("gpu_id", 0)
    inference_mode = model_config.get("inference_mode", "understanding")
    seed = model_config.get("seed", 666)
    num_inference_steps = model_config.get("num_inference_steps", 28)
    guidance_scale = model_config.get("guidance_scale", 3.5)
    height = model_config.get("height", 1024)
    width = model_config.get("width", 1024)

    if isinstance(image_paths, list):
        image_path = image_paths[0] if image_paths else None
    else:
        image_path = image_paths

    cache = _load_model(model_path, flux_path, siglip_path, gpu_id, project_path)

    if inference_mode == "understanding":
        max_new_tokens = model_config.get("max_new_tokens", 512)
        return _understanding(cache, image_paths, prompt, max_new_tokens, seed)

    elif inference_mode == "generation":
        output_path = generate_output_path(model_config, index, "generation", round_number)
        return _generate_or_edit(cache, None, prompt, output_path,
                                 num_inference_steps, guidance_scale, height, width,
                                 seed, is_editing=False)

    elif inference_mode in ("editing", "unify"):
        mode_str = "editing" if inference_mode == "editing" else "unify"
        output_path = generate_output_path(model_config, index, mode_str, round_number)
        return _generate_or_edit(cache, image_path, prompt, output_path,
                                 num_inference_steps, guidance_scale, height, width,
                                 seed, is_editing=True)

    else:
        raise ValueError(f"[uniworld] Unknown inference_mode: {inference_mode}")
