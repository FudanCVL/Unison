#!/usr/bin/env python3
"""
Janus / Janus-Pro model inference adapter

Supported modes:
  - understanding: image-text VQA, using VLChatProcessor + language_model.generate()
  - generation:    text-to-image, using autoregressive CFG sampling
  - editing:       not supported (returns an empty string and logs a warning)
"""
import os
import sys
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from model_inference import generate_output_path, set_seed, load_pil_image

# Global model cache (reused within the process to avoid reloading)
_cached_models: Dict[str, Any] = {}
_failed_models: Dict[str, Exception] = {}


def _setup_janus_path(project_path: str) -> None:
    """Add the Janus project directory to sys.path"""
    if project_path not in sys.path:
        sys.path.insert(0, project_path)


def _load_model(model_path: str, gpu_id: int, project_path: str) -> Dict[str, Any]:
    """Load the Janus model (with caching)"""
    cache_key = f"{model_path}_{gpu_id}"
    if cache_key in _cached_models:
        return _cached_models[cache_key]
    if cache_key in _failed_models:
        raise _failed_models[cache_key]

    _setup_janus_path(project_path)

    print(f"[janus] Loading model {model_path} on GPU {gpu_id}...")
    torch.cuda.set_device(gpu_id)

    try:
        from transformers import AutoModelForCausalLM
        from janus.models import MultiModalityCausalLM, VLChatProcessor

        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        tokenizer = vl_chat_processor.tokenizer

        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda(gpu_id).eval()
    except Exception as e:
        _failed_models[cache_key] = e
        raise

    cache = {
        "processor": vl_chat_processor,
        "tokenizer": tokenizer,
        "model": vl_gpt,
        "gpu_id": gpu_id,
    }
    _cached_models[cache_key] = cache
    print(f"[janus] Model loaded successfully.")
    return cache


def _understanding(
    cache: Dict[str, Any],
    image_paths: Optional[Union[str, List[str]]],
    prompt: str,
    max_new_tokens: int = 512,
    seed: int = 666,
) -> str:
    """VQA inference"""
    _setup_janus_path("")  # already set up
    from janus.utils.io import load_pil_images as janus_load_pil

    set_seed(seed)

    vl_chat_processor = cache["processor"]
    tokenizer = cache["tokenizer"]
    vl_gpt = cache["model"]
    gpu_id = cache["gpu_id"]

    # For single or multiple images, use the first one
    if isinstance(image_paths, list):
        img_path = image_paths[0] if image_paths else None
    else:
        img_path = image_paths

    if img_path and os.path.exists(img_path):
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>\n{prompt}",
                "images": [img_path],
            },
            {"role": "Assistant", "content": ""},
        ]
    else:
        # Text-only QA when there is no image
        conversation = [
            {"role": "User", "content": prompt},
            {"role": "Assistant", "content": ""},
        ]

    pil_images = janus_load_pil(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
    ).to(f"cuda:{gpu_id}")

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    with torch.no_grad():
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer.strip()


@torch.inference_mode()
def _generation(
    cache: Dict[str, Any],
    prompt: str,
    output_path: str,
    cfg_weight: float = 5.0,
    temperature: float = 1.0,
    parallel_size: int = 1,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    seed: int = 666,
) -> str:
    """Text-to-image generation"""
    set_seed(seed)

    vl_chat_processor = cache["processor"]
    tokenizer = cache["tokenizer"]
    vl_gpt = cache["model"]
    gpu_id = cache["gpu_id"]

    conversation = [
        {"role": "User", "content": prompt},
        {"role": "Assistant", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    full_prompt = sft_format + vl_chat_processor.image_start_tag

    input_ids = tokenizer.encode(full_prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda(gpu_id)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros(
        (parallel_size, image_token_num_per_image), dtype=torch.int
    ).cuda(gpu_id)

    past_key_values = None
    for i in range(image_token_num_per_image):
        outputs = vl_gpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        hidden_states = outputs.last_hidden_state

        logits = vl_gpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
        ).view(-1)
        img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = vl_gpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(dec[0]).save(output_path)
    print(f"[janus] Generated image saved to: {output_path}")
    return output_path


def janus_inference_function(
    image_paths: Optional[Union[str, List[str]]],
    prompt: str,
    model_config: Dict[str, Any],
    index: Optional[int] = None,
    round_number: Optional[int] = None,
) -> str:
    """
    Janus / Janus-Pro inference entry function

    Returns:
        understanding mode → text string
        generation mode    → saved image path
        editing mode       → "" (not supported, logs a warning)
    """
    project_path = model_config.get("janus_project_path", "")
    if project_path:
        _setup_janus_path(project_path)

    model_path = model_config.get("model_path", "deepseek-ai/Janus-Pro-7B")
    gpu_id = model_config.get("gpu_id", 0)
    inference_mode = model_config.get("inference_mode", "understanding")
    seed = model_config.get("seed", 666)

    cache = _load_model(model_path, gpu_id, project_path)

    if inference_mode == "understanding":
        max_new_tokens = model_config.get("max_new_tokens", 512)
        return _understanding(cache, image_paths, prompt, max_new_tokens, seed)

    elif inference_mode == "generation":
        output_path = generate_output_path(model_config, index, "generation", round_number)
        cfg_weight = model_config.get("cfg_weight", 5.0)
        temperature = model_config.get("temperature", 1.0)
        parallel_size = model_config.get("parallel_size", 1)
        img_size = model_config.get("img_size", 384)
        patch_size = model_config.get("patch_size", 16)
        image_token_num_per_image = (img_size // patch_size) ** 2
        return _generation(
            cache, prompt, output_path,
            cfg_weight=cfg_weight,
            temperature=temperature,
            parallel_size=parallel_size,
            image_token_num_per_image=image_token_num_per_image,
            img_size=img_size,
            patch_size=patch_size,
            seed=seed,
        )

    elif inference_mode in ("editing", "unify"):
        print(f"[janus] Warning: Janus does not support '{inference_mode}' mode. Returning empty.")
        return ""

    else:
        raise ValueError(f"[janus] Unknown inference_mode: {inference_mode}")
