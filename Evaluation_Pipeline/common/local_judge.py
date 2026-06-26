"""
LocalQwenVLJudge — drop-in replacement for ClosedSourceJudge that runs a locally
trained checkpoint instead of an API. It reuses the judge's prompts and
scoring/parsing verbatim by inheriting every higher-level
method and overriding only __init__ and _call.
"""
from typing import Dict, List, Optional

from common.judge import ClosedSourceJudge
from common.local_qwenvl import LocalQwenVLEngine, parse_gpu_ids


def _openai_to_engine_messages(messages: List[Dict]) -> List[Dict]:
    """OpenAI chat messages -> engine intermediate messages.
    image_url block -> image block carrying the url string. Order/roles preserved."""
    out = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        blocks = []
        for block in content:
            if block.get("type") == "image_url":
                blocks.append({"type": "image", "image": block["image_url"]["url"]})
            else:
                blocks.append({"type": "text", "text": block.get("text", "")})
        out.append({"role": role, "content": blocks})
    return out


class LocalQwenVLJudge(ClosedSourceJudge):
    """Same public interface as ClosedSourceJudge; runs a local model via a
    multi-GPU process pool. Only __init__ and _call are overridden."""

    def __init__(self, model_path: str, gpu_ids, max_workers: Optional[int] = None):
        # NOTE: intentionally does NOT call super().__init__ (no OpenAI client /
        # rate limiters). Inherited methods only use self.max_workers + self._call.
        gpu_list = parse_gpu_ids(gpu_ids)
        self.model = "local:" + model_path
        self.max_workers = max_workers or len(gpu_list)
        self._engine = LocalQwenVLEngine(model_path, gpu_list)

    def _call(self, messages: list, max_tokens: int = 16, text_only: bool = False) -> str:
        engine_messages = _openai_to_engine_messages(messages)
        try:
            return self._engine.generate(engine_messages, max_new_tokens=max_tokens)
        except Exception as e:  # mirror API _call's error contract for downstream parsers
            return f"ERROR: {e}"

    def close(self):
        self._engine.shutdown()
