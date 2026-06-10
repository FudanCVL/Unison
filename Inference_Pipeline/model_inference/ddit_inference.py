"""D-DiT (Dual-Diffusion SD3) inference wrapper for the Unison pipeline."""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

_model_cache = {}   # cache_key -> pipeline
_failed = {}        # cache_key -> Exception


def _setup_path(project_path: str):
    p = str(Path(project_path).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_pipeline(model_path: str, project_path: str, gpu_id: int):
    """Load DualDiffSD3Pipeline onto a single GPU."""
    _setup_path(project_path)

    # Import after path setup so sd3_modules is findable
    from sd3_modules.dual_diff_pipeline import DualDiffSD3Pipeline

    device = f"cuda:{gpu_id}"
    pipeline = DualDiffSD3Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def _to_pil(images_tensor) -> Image.Image:
    """Convert VAE-decoded tensor (B,C,H,W) in [-1,1] to a PIL image."""
    img = images_tensor[0].float().cpu()
    img = (img / 2 + 0.5).clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(img)


def _format_vqa_prompt(prompt: str) -> str:
    """Wrap into the Q:/A: format from the official VQAv2 eval script."""
    return f"Q: {prompt} Answer the question using a single word or phrase. A: "


def _extract_answer(text: str, formatted_prompt: str) -> str:
    """Strip the Q:/A: prefix to return just the generated answer."""
    marker = "A: "
    idx = text.rfind(marker)
    if idx != -1:
        return text[idx + len(marker):].strip()
    # Fallback: remove the full formatted prompt if present
    if text.startswith(formatted_prompt):
        return text[len(formatted_prompt):].strip()
    return text.strip()


def ddit_inference_function(image_path, prompt, model_config, index=None, round_number=None):
    """
    D-DiT inference for understanding and generation modes.

    Understanding (i2t): image + question → text answer
    Generation (t2i):    text prompt → image path
    """
    project_path = model_config.get("project_path",
                                    "/path/to/Unified_Models/Dual-Diffusion")
    model_path = model_config.get("model_path")
    gpu_id = model_config.get("gpu_id", 0)
    seed = model_config.get("seed", 666)
    inference_mode = model_config.get("inference_mode", "understanding")

    cache_key = f"{model_path}_{gpu_id}"
    if cache_key in _failed:
        raise _failed[cache_key]

    if cache_key not in _model_cache:
        try:
            _model_cache[cache_key] = _load_pipeline(model_path, project_path, gpu_id)
        except Exception as e:
            _failed[cache_key] = e
            raise
    pipeline = _model_cache[cache_key]

    torch.manual_seed(seed)
    device = f"cuda:{gpu_id}"

    # ------------------------------------------------------------------ #
    # Understanding: image → text
    # ------------------------------------------------------------------ #
    if inference_mode == "understanding":
        if not image_path or not os.path.exists(image_path):
            return ""
        image = Image.open(image_path).convert("RGB")
        # D-DiT's masked diffusion text decoder was trained on "Q: ... A: " VQA format.
        # Wrapping the prompt ensures the model generates tokens after the "A: " marker.
        formatted_prompt = _format_vqa_prompt(prompt)
        pipeline.set_sampling_mode("i2t")
        with torch.inference_mode():
            text = pipeline(
                image=image,
                prompt=formatted_prompt,
                sequence_length=model_config.get("sequence_length", 128),
                num_inference_steps=model_config.get("num_inference_steps_i2t", 64),
                resolution=model_config.get("resolution", 512),
            )
        return _extract_answer(text, formatted_prompt)

    # ------------------------------------------------------------------ #
    # Generation: text → image
    # ------------------------------------------------------------------ #
    elif inference_mode == "generation":
        # Build output path
        output_dir = os.path.join(
            model_config.get("output_image_dir", "result/D-DiT/images"),
            model_config.get("task_id", ""),
        )
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"_round{round_number}_generation" if round_number is not None else "_generation"
        out_path = os.path.join(output_dir, f"{index:06d}{suffix}.png")

        res = model_config.get("resolution", 512)
        pipeline.set_sampling_mode("t2i")
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.inference_mode():
            images = pipeline(
                prompt=prompt,
                height=res,
                width=res,
                num_inference_steps=model_config.get("num_inference_steps_t2i", 28),
                guidance_scale=model_config.get("guidance_scale", 7.0),
                generator=generator,
            )
        _to_pil(images).save(out_path)
        return out_path

    else:
        raise ValueError(f"D-DiT does not support inference_mode='{inference_mode}'")
