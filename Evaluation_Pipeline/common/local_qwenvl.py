"""
Multi-GPU local Qwen3-VL generation engine.

Loads N copies of a Qwen3-VL checkpoint (one per GPU) in persistent worker
processes and exposes a blocking generate() call. Used by
common/local_judge.py (LocalQwenVLJudge, the judge backend for evaluate_unison.py).

Heavy deps (torch/transformers) are imported lazily inside the worker functions
so the pure helpers below can be imported and unit-tested without a GPU.
"""
import base64
import io
import os
from typing import Dict, List

from PIL import Image


def parse_gpu_ids(spec) -> List[int]:
    """Parse "0-7" / "0,1,2" / "0-1,4" / [0,1] / None into a sorted unique int list."""
    if spec is None:
        return [0]
    if isinstance(spec, (list, tuple)):
        return sorted(set(int(x) for x in spec))
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _decode_image(uri_or_path: str) -> Image.Image:
    """Decode a data-URI (data:image/...;base64,....) or a filesystem path to RGB PIL."""
    if isinstance(uri_or_path, str) and uri_or_path.startswith("data:"):
        b64 = uri_or_path.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
    else:
        img = Image.open(uri_or_path)
    return img.convert("RGB")


def _to_processor_messages(messages: List[Dict]) -> List[Dict]:
    """Convert engine intermediate messages (image value = data-uri or path) into
    the structure Qwen3-VL processor.apply_chat_template expects, with images
    decoded to PIL. Block order and roles preserved."""
    out = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue
        new_content = []
        for block in content:
            if block.get("type") == "image":
                new_content.append({"type": "image", "image": _decode_image(block["image"])})
            else:
                new_content.append({"type": "text", "text": block.get("text", "")})
        out.append({"role": role, "content": new_content})
    return out


import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

_MODEL = None
_PROCESSOR = None


def _init_worker(model_path, gpu_ids, counter):
    """ProcessPoolExecutor initializer: claim a distinct GPU, load model+processor once."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    global _MODEL, _PROCESSOR
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    gpu = gpu_ids[idx % len(gpu_ids)]
    torch.cuda.set_device(gpu)
    _MODEL = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype="auto", device_map=f"cuda:{gpu}"
    )
    _MODEL.eval()
    # Intermediate checkpoints (checkpoint-N/) only save weights; the full
    # processor config (preprocessor_config.json, tokenizer files) lives only
    # in the root output directory. Fall back to parent if needed.
    processor_path = model_path
    if not os.path.exists(os.path.join(model_path, "preprocessor_config.json")):
        parent = os.path.dirname(model_path)
        if os.path.exists(os.path.join(parent, "preprocessor_config.json")):
            processor_path = parent
    _PROCESSOR = AutoProcessor.from_pretrained(processor_path)


def _worker_generate(messages, max_new_tokens):
    import torch
    proc_messages = _to_processor_messages(messages)
    inputs = _PROCESSOR.apply_chat_template(
        proc_messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(_MODEL.device)
    with torch.no_grad():
        gen = _MODEL.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], gen)]
    text = _PROCESSOR.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return text[0].strip()


class LocalQwenVLEngine:
    """N persistent worker processes, one model per GPU; blocking generate()."""

    def __init__(self, model_path: str, gpu_ids):
        gpu_ids = list(gpu_ids)
        if not gpu_ids:
            raise ValueError("gpu_ids must be non-empty")
        self.model_path = model_path
        self.gpu_ids = gpu_ids
        ctx = mp.get_context("spawn")
        counter = ctx.Value("i", 0)
        self._pool = ProcessPoolExecutor(
            max_workers=len(gpu_ids),
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(model_path, gpu_ids, counter),
        )

    def generate(self, messages, max_new_tokens: int = 16) -> str:
        return self._pool.submit(_worker_generate, messages, max_new_tokens).result()

    def shutdown(self):
        self._pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
