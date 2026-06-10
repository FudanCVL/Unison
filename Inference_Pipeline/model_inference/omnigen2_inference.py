#!/usr/bin/env python3
"""
OmniGen2 model inference adapter

Supported modes:
  - understanding: OmniGen2ChatPipeline returns results.text
  - generation:    OmniGen2Pipeline (no input image) returns results.images[0]
  - editing:       OmniGen2Pipeline (with input image) returns results.images[0]

The config JSON must specify:
  - model_path:             OmniGen2/OmniGen2 (HF repo or local path)
  - omnigen2_project_path:  /home/.../OmniGen2 (local code directory)
"""
import os
import sys
import torch
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from model_inference import generate_output_path, set_seed, load_pil_image

# Cache the two pipelines separately
_cached_chat_pipeline: Dict[str, Any] = {}
_cached_gen_pipeline: Dict[str, Any] = {}
_failed_pipelines: Dict[str, Exception] = {}


def _setup_paths(project_path: str) -> None:
    if project_path not in sys.path:
        sys.path.insert(0, project_path)


def _resolve_model_path(model_path: str) -> str:
    """Resolve a HuggingFace Hub ID to a local cache snapshot path.

    With HF_HUB_OFFLINE=1, from_pretrained still calls the model_info() API; passing a local path skips it.
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
                print(f"[omnigen2] Resolved '{model_path}' to local path: {resolved}")
                return resolved
    except Exception:
        pass
    return model_path


def _load_chat_pipeline(model_path: str, gpu_id: int, project_path: str) -> Any:
    """Load OmniGen2ChatPipeline (used for understanding)"""
    cache_key = f"chat_{model_path}_{gpu_id}"
    if cache_key in _cached_chat_pipeline:
        return _cached_chat_pipeline[cache_key]
    if cache_key in _failed_pipelines:
        raise _failed_pipelines[cache_key]

    _setup_paths(project_path)

    print(f"[omnigen2] Loading OmniGen2ChatPipeline on GPU {gpu_id}...")
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    try:
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2_chat import OmniGen2ChatPipeline
        from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

        local_path = _resolve_model_path(model_path)
        pipeline = OmniGen2ChatPipeline.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            local_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        pipeline = pipeline.to(device)
    except Exception as e:
        _failed_pipelines[cache_key] = e
        raise

    _cached_chat_pipeline[cache_key] = pipeline
    print(f"[omnigen2] Chat pipeline loaded.")
    return pipeline


def _load_gen_pipeline(model_path: str, gpu_id: int, project_path: str) -> Any:
    """Load OmniGen2Pipeline (used for generation / editing)"""
    cache_key = f"gen_{model_path}_{gpu_id}"
    if cache_key in _cached_gen_pipeline:
        return _cached_gen_pipeline[cache_key]
    if cache_key in _failed_pipelines:
        raise _failed_pipelines[cache_key]

    _setup_paths(project_path)

    print(f"[omnigen2] Loading OmniGen2Pipeline on GPU {gpu_id}...")
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    try:
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

        local_path = _resolve_model_path(model_path)
        pipeline = OmniGen2Pipeline.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            local_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        pipeline = pipeline.to(device)
    except Exception as e:
        _failed_pipelines[cache_key] = e
        raise

    _cached_gen_pipeline[cache_key] = pipeline
    print(f"[omnigen2] Generation pipeline loaded.")
    return pipeline


def _understanding(pipeline: Any, image_paths_raw: Optional[Union[str, List[str]]],
                   prompt: str, seed: int = 666, gpu_id: int = 0,
                   max_new_tokens: int = 512) -> str:
    """OmniGen2ChatPipeline text output"""
    set_seed(seed)
    device = torch.device(f"cuda:{gpu_id}")
    generator = torch.Generator(device=device).manual_seed(seed)

    if isinstance(image_paths_raw, list):
        img_paths = [p for p in image_paths_raw if p and os.path.exists(p)]
    elif image_paths_raw and os.path.exists(image_paths_raw):
        img_paths = [image_paths_raw]
    else:
        img_paths = []

    input_images = [load_pil_image(p) for p in img_paths] if img_paths else None

    # Note: OmniGen2ChatPipeline.__call__ does not accept max_new_tokens (text output length is controlled internally)
    results = pipeline(
        prompt=prompt,
        input_images=input_images,
        generator=generator,
        output_type="pil",
    )
    return (results.text or "").strip()


def _generate_or_edit(pipeline: Any, image_paths_raw: Optional[Union[str, List[str]]],
                      prompt: str, output_path: str, width: int = 1024, height: int = 1024,
                      num_inference_steps: int = 50, text_guidance_scale: float = 5.0,
                      image_guidance_scale: float = 2.0, seed: int = 666,
                      gpu_id: int = 0) -> str:
    """OmniGen2Pipeline image generation/editing"""
    set_seed(seed)
    device = torch.device(f"cuda:{gpu_id}")
    generator = torch.Generator(device=device).manual_seed(seed)

    negative_prompt = (
        "(((deformed))), blurry, over saturation, bad anatomy, disfigured, "
        "poorly drawn face, mutation, mutated, (extra_limb), (ugly), "
        "(poorly drawn hands), fused fingers, messy drawing, broken legs"
    )

    if isinstance(image_paths_raw, list):
        img_paths = [p for p in image_paths_raw if p and os.path.exists(p)]
    elif image_paths_raw and os.path.exists(image_paths_raw):
        img_paths = [image_paths_raw]
    else:
        img_paths = []

    input_images = [load_pil_image(p) for p in img_paths] if img_paths else None

    results = pipeline(
        prompt=prompt,
        input_images=input_images,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        max_sequence_length=1024,
        text_guidance_scale=text_guidance_scale,
        image_guidance_scale=image_guidance_scale,
        cfg_range=(0.0, 1.0),
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        generator=generator,
        output_type="pil",
    )

    out_img = results.images[0]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_img.save(output_path)
    print(f"[omnigen2] Image saved to: {output_path}")
    return output_path


def omnigen2_inference_function(
    image_paths: Optional[Union[str, List[str]]],
    prompt: str,
    model_config: Dict[str, Any],
    index: Optional[int] = None,
    round_number: Optional[int] = None,
) -> str:
    """
    OmniGen2 inference entry function

    Returns:
        understanding mode → text string
        generation / editing / unify mode → saved image path
    """
    project_path = model_config.get("omnigen2_project_path", "")
    if project_path:
        _setup_paths(project_path)

    model_path = model_config.get("model_path", "OmniGen2/OmniGen2")
    gpu_id = model_config.get("gpu_id", 0)
    inference_mode = model_config.get("inference_mode", "understanding")
    seed = model_config.get("seed", 666)

    if inference_mode == "understanding":
        pipeline = _load_chat_pipeline(model_path, gpu_id, project_path)
        max_new_tokens = model_config.get("max_new_tokens", 512)
        return _understanding(pipeline, image_paths, prompt, seed, gpu_id, max_new_tokens)

    elif inference_mode in ("generation", "editing", "unify"):
        pipeline = _load_gen_pipeline(model_path, gpu_id, project_path)
        mode_str = inference_mode if inference_mode in ("generation", "editing") else "unify"
        output_path = generate_output_path(model_config, index, mode_str, round_number)
        width = model_config.get("width", 1024)
        height = model_config.get("height", 1024)
        num_inference_steps = model_config.get("num_inference_steps", 50)
        text_guidance_scale = model_config.get("text_guidance_scale", 5.0)
        image_guidance_scale = model_config.get("image_guidance_scale", 2.0)
        return _generate_or_edit(
            pipeline, image_paths, prompt, output_path,
            width=width, height=height,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            image_guidance_scale=image_guidance_scale,
            seed=seed, gpu_id=gpu_id,
        )

    else:
        raise ValueError(f"[omnigen2] Unknown inference_mode: {inference_mode}")
