#!/usr/bin/env python3
"""
Show-o (v1) inference adapter — 1.3B (Phi-1.5 + MAGVITv2 discrete VQ tokens)

Supported modes:
  - understanding: CLIP ViT multimodal VQA (create_attention_mask_for_mmu_vit)
  - generation:    text-to-image (masked diffusion, t2i_generate)
  - editing:       not supported

Requires the top-level Show-o project directory (showo_project_path), which must contain:
  - models/          (Showo, MAGVITv2, CLIPVisionTower, get_mask_chedule)
  - training/        (UniversalPrompting, create_attention_mask_*)
  - llava/           (conversation templates)
  - configs/         (e.g. showo_demo_w_clip_vit_512x512.yaml)

Required config JSON fields:
  - model_path:         showlab/show-o-w-clip-vit-512x512 (or a local path)
  - llm_model_path:     microsoft/phi-1_5 (or a local path)
  - vq_model_path:      showlab/magvitv2 (or a local path)
  - clip_model_path:    openai/clip-vit-large-patch14-336 (or a local path)
  - showo_project_path: /path/to/Show-o
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

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
SYSTEM_PROMPT_LEN = 28


def _setup_paths(project_path: str) -> None:
    if project_path not in sys.path:
        sys.path.insert(0, project_path)


def _load_model(model_path: str, llm_model_path: str, vq_model_path: str,
                clip_model_path: str, gpu_id: int, project_path: str,
                resolution: int = 512) -> Dict[str, Any]:
    cache_key = f"{model_path}_{gpu_id}"
    if cache_key in _cached_models:
        return _cached_models[cache_key]
    if cache_key in _failed_models:
        raise _failed_models[cache_key]

    _setup_paths(project_path)
    print(f"[showo] Loading Show-o model on GPU {gpu_id}...")
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)

    try:
        # transformers 4.57+ enforces a safety check on the .bin format under torch<2.6 (CVE-2025-32434).
        # Show-o v1's Phi-1.5 backbone ships only .bin; bypassing the check for local trusted files is safe.
        import transformers.modeling_utils as _tm
        _tm.check_torch_load_is_safe = lambda: None

        from omegaconf import OmegaConf
        from models import Showo, MAGVITv2, CLIPVisionTower
        from training.prompting_utils import UniversalPrompting
        from transformers import AutoTokenizer

        cfg_name = "showo_demo_w_clip_vit_512x512.yaml" if resolution >= 512 else "showo_demo_w_clip_vit.yaml"
        cfg_path = os.path.join(project_path, "configs", cfg_name)
        config = OmegaConf.load(cfg_path)

        config.model.showo.pretrained_model_path = model_path
        config.model.showo.llm_model_path = llm_model_path
        config.model.vq_model.vq_model_name = vq_model_path

        tokenizer = AutoTokenizer.from_pretrained(llm_model_path, padding_side="left")
        uni_prompting = UniversalPrompting(
            tokenizer,
            max_text_len=config.dataset.preprocessing.max_seq_length,
            special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
                            "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
            ignore_id=-100,
            cond_dropout_prob=0.1,
        )

        vq_model = MAGVITv2.from_pretrained(vq_model_path).to(device)
        vq_model.requires_grad_(False)
        vq_model.eval()

        vision_tower = CLIPVisionTower(clip_model_path).to(device)

        # low_cpu_mem_usage=False forces construction on real CPU tensors: Showo.__init__ calls
        # resize_token_embeddings, and recent transformers default to mean_resizing=True, which
        # computes covariance over the embedding; on a meta device (low_cpu_mem_usage default True)
        # this raises "Tensor.item() cannot be called on meta tensors". The checkpoint weights then
        # overwrite this random init (the ckpt already contains the full 58498-row embedding).
        model = Showo.from_pretrained(model_path, low_cpu_mem_usage=False).to(device)
        model.eval()

        mask_token_id = model.config.mask_token_id

    except Exception as e:
        _failed_models[cache_key] = e
        raise

    _cached_models[cache_key] = {
        "model": model,
        "vq_model": vq_model,
        "vision_tower": vision_tower,
        "tokenizer": tokenizer,
        "uni_prompting": uni_prompting,
        "config": config,
        "device": device,
        "mask_token_id": mask_token_id,
    }
    print("[showo] Model loaded.")
    return _cached_models[cache_key]


def _understanding(cache: Dict[str, Any], image_path: Optional[str], prompt: str,
                   max_new_tokens: int = 512, seed: int = 666) -> str:
    """CLIP ViT + Phi-1.5 multimodal understanding (w_clip_vit=True path)."""
    set_seed(seed)

    from training.prompting_utils import create_attention_mask_for_mmu_vit
    from llava.llava import conversation as conversation_lib
    from training.utils import image_transform

    model = cache["model"]
    tokenizer = cache["tokenizer"]
    uni_prompting = cache["uni_prompting"]
    vision_tower = cache["vision_tower"]
    config = cache["config"]
    device = cache["device"]

    top_k = 1

    # Build the prompt with the phi1.5 template
    conv = conversation_lib.conv_templates["phi1.5"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt().strip()

    # System-prompt tokens (the phi-1_5 tokenizer always produces 28 tokens for SYSTEM_PROMPT)
    input_ids_system = tokenizer(
        SYSTEM_PROMPT, return_tensors="pt", padding="longest"
    ).input_ids.to(device)
    if input_ids_system.shape[-1] != SYSTEM_PROMPT_LEN:
        print(f"[showo] Warning: system prompt len={input_ids_system.shape[-1]}, expected {SYSTEM_PROMPT_LEN}")

    # Question tokens
    input_ids_q = tokenizer(
        [prompt_question], return_tensors="pt", padding="longest"
    ).input_ids.to(device)

    input_ids_llava = torch.cat([
        (torch.ones(1, 1) * uni_prompting.sptids_dict['<|mmu|>']).long().to(device),
        input_ids_system,
        (torch.ones(1, 1) * uni_prompting.sptids_dict['<|soi|>']).long().to(device),
        (torch.ones(1, 1) * uni_prompting.sptids_dict['<|eoi|>']).long().to(device),
        input_ids_q,
    ], dim=1)

    # CLIP image embeddings
    if image_path and os.path.exists(image_path):
        from transformers import CLIPImageProcessor
        clip_processor = CLIPImageProcessor.from_pretrained(
            vision_tower.vision_tower_name
        )
        image_ori = load_pil_image(image_path)
        pixel_values = clip_processor.preprocess(image_ori, return_tensors="pt")["pixel_values"][0]
        images_embeddings = vision_tower(pixel_values[None].to(device))
        images_embeddings = model.mm_projector(images_embeddings)
        has_image = True
    else:
        images_embeddings = None
        has_image = False

    text_embeddings = model.showo.model.embed_tokens(input_ids_llava)

    if has_image:
        # part1: <|mmu|>(1) + system(28) + <|soi|>(1) = 30 tokens
        part1 = text_embeddings[:, :2 + SYSTEM_PROMPT_LEN, :]
        # part2: <|eoi|>(1) + question
        part2 = text_embeddings[:, 2 + SYSTEM_PROMPT_LEN:, :]
        input_embeddings = torch.cat((part1, images_embeddings, part2), dim=1)
    else:
        input_embeddings = text_embeddings

    attention_mask = create_attention_mask_for_mmu_vit(
        input_embeddings, system_prompt_len=input_ids_system.shape[-1]
    )

    with torch.no_grad():
        cont_toks_list = model.mmu_generate(
            input_embeddings=input_embeddings,
            attention_mask=attention_mask[0].unsqueeze(0),
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            eot_token=tokenizer.eos_token_id,
        )

    cont_toks = torch.stack(cont_toks_list).squeeze()[None]
    text = tokenizer.batch_decode(cont_toks, skip_special_tokens=True)
    return text[0].strip()


def _generation(cache: Dict[str, Any], prompt: str, output_path: str,
                guidance_scale: float = 5.0, timesteps: int = 18,
                temperature: float = 1.0, seed: int = 666) -> str:
    """Text-to-image via MAGVITv2 masked diffusion."""
    set_seed(seed)

    from training.prompting_utils import create_attention_mask_predict_next
    from models.sampling import get_mask_chedule

    model = cache["model"]
    vq_model = cache["vq_model"]
    uni_prompting = cache["uni_prompting"]
    config = cache["config"]
    device = cache["device"]
    mask_token_id = cache["mask_token_id"]

    num_vq_tokens = config.model.showo.num_vq_tokens
    codebook_size = config.model.showo.codebook_size

    image_tokens = torch.ones((1, num_vq_tokens), dtype=torch.long, device=device) * mask_token_id
    input_ids, _ = uni_prompting(([prompt], image_tokens), 't2i_gen')

    if guidance_scale > 0:
        uncond_input_ids, _ = uni_prompting(([''], image_tokens), 't2i_gen')
        attention_mask = create_attention_mask_predict_next(
            torch.cat([input_ids, uncond_input_ids], dim=0),
            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
            rm_pad_in_image=True,
        )
    else:
        uncond_input_ids = None
        attention_mask = create_attention_mask_predict_next(
            input_ids,
            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
            rm_pad_in_image=True,
        )

    mask_schedule = get_mask_chedule("cosine")

    with torch.no_grad():
        gen_token_ids = model.t2i_generate(
            input_ids=input_ids,
            uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask,
            guidance_scale=guidance_scale,
            temperature=temperature,
            timesteps=timesteps,
            noise_schedule=mask_schedule,
            noise_type="mask",
            seq_len=num_vq_tokens,
            uni_prompting=uni_prompting,
            config=config,
        )

    gen_token_ids = torch.clamp(gen_token_ids, max=codebook_size - 1, min=0)
    images = vq_model.decode_code(gen_token_ids)

    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images = (images * 255.0).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    pil_img = Image.fromarray(images[0])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pil_img.save(output_path)
    print(f"[showo] Generated image saved to: {output_path}")
    return output_path


def showo_inference_function(
    image_paths: Optional[Union[str, List[str]]],
    prompt: str,
    model_config: Dict[str, Any],
    index: Optional[int] = None,
    round_number: Optional[int] = None,
) -> str:
    project_path = model_config.get("showo_project_path", "")
    if project_path:
        _setup_paths(project_path)

    model_path = model_config.get("model_path", "showlab/show-o-w-clip-vit-512x512")
    llm_model_path = model_config.get("llm_model_path", "microsoft/phi-1_5")
    vq_model_path = model_config.get("vq_model_path", "showlab/magvitv2")
    clip_model_path = model_config.get("clip_model_path", "openai/clip-vit-large-patch14-336")
    resolution = model_config.get("resolution", 512)
    gpu_id = model_config.get("gpu_id", 0)
    inference_mode = model_config.get("inference_mode", "understanding")
    seed = model_config.get("seed", 666)

    if isinstance(image_paths, list):
        image_path = image_paths[0] if image_paths else None
    else:
        image_path = image_paths

    cache = _load_model(model_path, llm_model_path, vq_model_path, clip_model_path,
                        gpu_id, project_path, resolution)

    if inference_mode == "understanding":
        max_new_tokens = model_config.get("max_new_tokens", 512)
        return _understanding(cache, image_path, prompt, max_new_tokens, seed)

    elif inference_mode == "generation":
        output_path = generate_output_path(model_config, index, "generation", round_number)
        guidance_scale = model_config.get("guidance_scale", 5.0)
        timesteps = model_config.get("generation_timesteps", 18)
        temperature = model_config.get("generation_temperature", 1.0)
        return _generation(cache, prompt, output_path, guidance_scale, timesteps, temperature, seed)

    elif inference_mode in ("editing", "unify"):
        print(f"[showo] Warning: Show-o does not support '{inference_mode}' mode. Returning empty.")
        return ""

    else:
        raise ValueError(f"[showo] Unknown inference_mode: {inference_mode}")
