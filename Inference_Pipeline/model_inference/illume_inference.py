#!/usr/bin/env python3
"""
ILLUME+ model inference adapter

Supported modes:
  - understanding: inference_mllm(is_img_gen_task=False) → output_text
  - generation:    inference_mllm(is_img_gen_task=True) + VQ decoder → image
  - editing:       same as generation, but the input includes the original image and uses the diffusion decoder

Depends on build_eval_model from the ILLUME_plus/ILLUME/generation_eval/ directory.
The config JSON must specify:
  - model_config_path:       configs/example/illume_plus_7b/...stage3.py
  - tokenizer_config_path:   configs/example/dualvitok/...max512.py
  - diffusion_decoder_path:  checkpoints/dualvitok-sdxl-decoder/
  - tokenizer_checkpoint:    checkpoints/dualvitok/pytorch_model.bin
  - illume_project_path:     /home/.../ILLUME_plus
"""
import fcntl
import os
import sys

# Suppress the "got forked after parallelism" warning from HuggingFace tokenizers.
# ILLUME's model loading triggers a fork after the fast tokenizer has already
# spun up background threads; setting this env var before any import silences it.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from model_inference import generate_output_path, set_seed, load_pil_image

_cached_engines: Dict[str, Any] = {}
_failed_engines: Dict[str, Exception] = {}

# Disk-IO serialization lock for concurrent loading across multiple GPU processes (avoids 8 processes reading ~18GB at once and contending for bandwidth)
_LOAD_LOCK_PATH = "/tmp/illume_model_load.lock"


def _setup_paths(project_path: str) -> None:
    illume_src = os.path.join(project_path, "ILLUME")
    vision_tok_src = os.path.join(project_path, "vision_tokenizer")
    for p in [illume_src, vision_tok_src, project_path]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_engine(model_config_path: str, tokenizer_config_path: str,
                 diffusion_decoder_path: str, tokenizer_checkpoint: str,
                 gpu_id: int, project_path: str) -> Any:
    """Load the ILLUME+ inference engine (with caching)"""
    cache_key = f"{model_config_path}_{gpu_id}"
    if cache_key in _cached_engines:
        return _cached_engines[cache_key]
    if cache_key in _failed_engines:
        raise _failed_engines[cache_key]

    _setup_paths(project_path)

    print(f"[illume] Loading ILLUME+ model on GPU {gpu_id}...")
    # ILLUME uses LOCAL_RANK internally to decide which GPU to load onto (see illume.py line 137).
    # It must be set before calling build_eval_model, otherwise everything defaults to GPU 0 and causes OOM.
    os.environ["LOCAL_RANK"] = str(gpu_id)
    torch.cuda.set_device(gpu_id)
    # All weights are already local; disable HuggingFace's network etag check to avoid each
    # from_pretrained stalling for tens of seconds when the network is slow (AutoTokenizer/AutoModel/SemanticEncoder/SDXL pipeline, 6+ call sites total).
    os.environ["HF_HUB_OFFLINE"] = "1"

    # Use a file lock to serialize disk IO when multiple GPU processes load at once, avoiding 8 processes reading ~18GB simultaneously and contending for bandwidth
    lock_file = open(_LOAD_LOCK_PATH, "w")
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    try:
        # Re-check the cache after acquiring the lock (another process may have finished loading)
        if cache_key in _cached_engines:
            return _cached_engines[cache_key]

        # The output_dir in the stage3 config is a relative path "./logdir/...",
        # so we must cd into the ILLUME_plus project root for it to resolve correctly
        _orig_cwd = os.getcwd()
        try:
            os.chdir(project_path)
            from generation_eval.models.builder import build_eval_model
            import generation_eval.models  # triggers @EVAL_MODELS.register_module() on class ILLUME

            eval_model_cfg = dict(
                type="ILLUME",
                config=model_config_path,
                tokenizer_config=tokenizer_config_path,
                diffusion_decoder_path=diffusion_decoder_path,
                tokenizer_checkpoint=tokenizer_checkpoint,
                torch_dtype="fp16",
            )
            engine = build_eval_model(eval_model_cfg)
        except Exception as e:
            _failed_engines[cache_key] = e
            raise
        finally:
            os.chdir(_orig_cwd)
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()

    _cached_engines[cache_key] = engine
    print(f"[illume] Model loaded.")
    return engine


def _understanding(engine: Any, image_paths_raw: Optional[Union[str, List[str]]],
                   prompt: str, seed: int = 666) -> str:
    """Text understanding"""
    set_seed(seed)

    if isinstance(image_paths_raw, list):
        img_path = image_paths_raw[0] if image_paths_raw else None
    else:
        img_path = image_paths_raw

    images_data = []
    if img_path and os.path.exists(img_path):
        images_data.append(load_pil_image(img_path))

    # prepare_inference_config does not accept do_sample; the default temperature=1.0 is fine
    inference_config = engine.prepare_inference_config(temperature=1.0)
    batch_data_item = dict(prompt=prompt)
    if images_data:
        batch_data_item["images_data"] = images_data
    else:
        # ILLUME reads image_sizes internally; set it empty explicitly when there is no image to avoid a KeyError
        batch_data_item["image_sizes"] = []
    outputs = engine.inference_mllm([batch_data_item], inference_config, is_img_gen_task=False)
    return (outputs[0].get("output_text", "") or "").strip()


def _generation(engine: Any, image_paths_raw: Optional[Union[str, List[str]]],
                prompt: str, output_path: str, seed: int = 666,
                use_diffusion: bool = False) -> str:
    """Image generation/editing"""
    set_seed(seed)

    if isinstance(image_paths_raw, list):
        img_path = image_paths_raw[0] if image_paths_raw else None
    else:
        img_path = image_paths_raw

    images_data = []
    if img_path and os.path.exists(img_path):
        images_data.append(load_pil_image(img_path))

    # llm_cfg_scale=1.0 disables LLM-CFG (CFG needs an unconditional_prompt, otherwise self.uncond=None crashes)
    # diffusion_cfg_scale only takes effect during editing (use_diffusion=True), going through the diffusion-internal CFG path
    inference_config = engine.prepare_inference_config(
        temperature=1.0,
        top_k=128 if not use_diffusion else 512,
        top_p=1.0,
        llm_cfg_scale=1.0,
        image_semantic_temperature=0.7 if use_diffusion else 1.0,
        image_semantic_top_k=512 if use_diffusion else 2048,
        image_semantic_top_p=0.8 if use_diffusion else 1.0,
        diffusion_cfg_scale=1.5 if use_diffusion else None,
        diffusion_num_inference_steps=50 if use_diffusion else None,
        resolution=(512, 512),
    )
    batch_data_item = dict(prompt=prompt)
    if images_data:
        batch_data_item["images_data"] = images_data
    batch_data = [batch_data_item]
    outputs = engine.inference_mllm(batch_data, inference_config, is_img_gen_task=True)
    out_images = engine.inference_tokenizer_decoder(
        outputs, inference_config, use_diffusion_decoder=use_diffusion
    )

    out_img = out_images[0] if out_images else None
    if out_img is None:
        raise RuntimeError("[illume] No image generated")

    if not isinstance(out_img, Image.Image):
        import numpy as np
        if isinstance(out_img, np.ndarray):
            out_img = Image.fromarray(out_img.astype(np.uint8))
        else:
            from torchvision.transforms.functional import to_pil_image
            out_img = to_pil_image(out_img.cpu())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_img.save(output_path)
    print(f"[illume] Image saved to: {output_path}")
    return output_path


def illume_inference_function(
    image_paths: Optional[Union[str, List[str]]],
    prompt: str,
    model_config: Dict[str, Any],
    index: Optional[int] = None,
    round_number: Optional[int] = None,
) -> str:
    """
    ILLUME+ inference entry function

    Returns:
        understanding mode → text string
        generation / editing / unify mode → saved image path
    """
    project_path = model_config.get("illume_project_path", "")
    if project_path:
        _setup_paths(project_path)

    model_config_path = model_config.get("model_config_path", "")
    tokenizer_config_path = model_config.get("tokenizer_config_path", "")
    diffusion_decoder_path = model_config.get("diffusion_decoder_path", "")
    tokenizer_checkpoint = model_config.get("tokenizer_checkpoint", "")
    gpu_id = model_config.get("gpu_id", 0)
    inference_mode = model_config.get("inference_mode", "understanding")
    seed = model_config.get("seed", 666)

    engine = _load_engine(
        model_config_path, tokenizer_config_path,
        diffusion_decoder_path, tokenizer_checkpoint,
        gpu_id, project_path,
    )

    if inference_mode == "understanding":
        return _understanding(engine, image_paths, prompt, seed)

    elif inference_mode == "generation":
        output_path = generate_output_path(model_config, index, "generation", round_number)
        return _generation(engine, image_paths, prompt, output_path, seed, use_diffusion=False)

    elif inference_mode in ("editing", "unify"):
        mode_str = "editing" if inference_mode == "editing" else "unify"
        output_path = generate_output_path(model_config, index, mode_str, round_number)
        return _generation(engine, image_paths, prompt, output_path, seed, use_diffusion=True)

    else:
        raise ValueError(f"[illume] Unknown inference_mode: {inference_mode}")
