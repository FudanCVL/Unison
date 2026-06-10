"""
Qwen3-VL-Plus judge via DashScope OpenAI-compatible API.

rate_edit uses ImgEdit benchmark prompts (PKU-YuanGroup/ImgEdit, NeurIPS 2025).
For multi-operation edits (compose), each sub-operation is scored independently
using its bbox annotation; the final score is the average.
"""

import base64
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from openai import OpenAI


class _RateLimiter:
    """Token bucket rate limiter — thread-safe, max N calls per second."""
    def __init__(self, rate: float):
        self._rate = rate
        self._tokens = float(rate)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self._rate, self._tokens + (now - self._last) * self._rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            time.sleep(wait)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYS_VQA = (
    "You are a precise VQA assistant. "
    "Answer questions with only 'yes' or 'no'. Do not explain."
)
_SYS_MISMATCH = (
    "You are evaluating predicted image-caption mismatches against ground truth. "
    "Respond with only a single integer from 0 to 5."
)
_SYS_RATER = (
    "You are an image quality evaluator. "
    "Respond with only a single integer from 0 to 5."
)


# ---------------------------------------------------------------------------
# ImgEdit rubric prompts  (action / style removed — not used in Unison tasks)
# {instruction} is replaced at call time; images follow after the prompt text.
# ---------------------------------------------------------------------------

_IMGEDIT_PROMPTS: Dict[str, str] = {
    "replace": (
        "You are a data rater specializing in grading image replacement edits. "
        "You will be given two images (before and after editing) and the corresponding editing instructions. "
        "Your task is to evaluate the replacement editing effect on a 5-point scale from three perspectives:\n\n"
        "Prompt Compliance\n"
        "1  Target not replaced, or an unrelated object edited.\n"
        "2  Only part of the target replaced, or wrong class/description used.\n"
        "3  Target largely replaced but other objects altered, remnants visible, or count/position clearly wrong.\n"
        "4  Correct object fully replaced; only minor attribute errors (colour, size, etc.).\n"
        "5  Perfect replacement: all and only the specified objects removed; new objects' class, number, position, scale, pose and detail exactly match the prompt.\n\n"
        "Visual Naturalness\n"
        "1  Image heavily broken or new object deformed / extremely blurred.\n"
        "2  Obvious seams, smears, or strong mismatch in resolution or colour; background not restored.\n"
        "3  Basic style similar, but lighting or palette clashes; fuzzy edges or noise are noticeable.\n"
        "4  Style almost uniform; tiny edge artefacts visible only on close inspection; casual viewers see no edit.\n"
        "5  Completely seamless; new objects blend fully with the scene, edit area undetectable.\n\n"
        "Physical & Detail Integrity\n"
        "1  Floating, interpenetration, severe perspective/light errors; key original elements ruined; background heavily warped.\n"
        "2  Missing shadows/occlusion; large background shifts or holes.\n"
        "3  Lighting, perspective and contact surfaces mostly correct; small but tolerable errors; background adjusted locally.\n"
        "4  New objects interact realistically with scene (shadows, reflections, texture) and preserve existing details; background change minimal.\n"
        "5  Physically flawless and enhances realism: accurate highlights, shadows, reflections, ambient effects; background untouched.\n"
        "The second and third score should no higher than first score!!!\n\n"
        "Example Response Format:\n"
        "Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\n"
        "Prompt Compliance: A number from 1 to 5.\n"
        "Visual Naturalness: A number from 1 to 5.\n"
        "Physical & Detail Integrity: A number from 1 to 5.\n"
        "editing instruction is : {instruction}.\n\n"
        "Below are the images before and after editing:\n"
    ),

    "add": (
        "You are a data rater specializing in grading image addition edits. "
        "You will be given two images (before and after editing) and the corresponding editing instructions. "
        "Your task is to evaluate the added object(s) on a 5-point scale from three perspectives:\n\n"
        "Prompt Compliance\n"
        "1  Nothing added or the added content is corrupt.\n"
        "2  Added object is a wrong class or unrelated to the prompt.\n"
        "3  Correct class, but key attributes (position, colour, size, count, etc.) are wrong.\n"
        "4  Main attributes correct; only minor details off or 1-2 small features missing.\n"
        "5  Every stated attribute correct and scene logic reasonable; only microscopic flaws.\n\n"
        "Visual Naturalness\n"
        "1  Image badly broken or full of artefacts.\n"
        "2  Obvious paste marks; style, resolution, or palette strongly mismatch.\n"
        "3  General style similar, but lighting or colours clearly clash; noticeable disharmony.\n"
        "4  Style almost uniform; small edge issues visible only when zoomed.\n"
        "5  Perfect blend; no visible difference between added object and original image.\n\n"
        "Physical & Detail Coherence\n"
        "1  Severe physical errors (floating, wrong perspective/light); key original elements blocked; background heavily distorted.\n"
        "2  Contact or occlusion handled poorly; minor background shifts, jaggies or noise; background visibly changed.\n"
        "3  Lighting, perspective, and contact mostly correct; remaining flaws small and acceptable; limited background change.\n"
        "4  Shadows, reflections, and material response believable; no loss of original detail; background changes are minute.\n"
        "5  Added object enhances overall realism: precise highlights, shadows, ambient effects; background essentially untouched.\n"
        "The second and third score should no higher than first score!!!\n\n"
        "Example Response Format:\n"
        "Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\n"
        "Prompt Compliance: A number from 1 to 5.\n"
        "Visual Naturalness: A number from 1 to 5.\n"
        "Physical & Detail Coherence: A number from 1 to 5.\n"
        "editing instruction is : {instruction}.\n\n"
        "Below are the images before and after editing:\n"
    ),

    "remove": (
        "You are a data rater specializing in grading object removal edits. "
        "You will be given two images (before and after editing) and the corresponding editing instructions. "
        "Your task is to evaluate the removal quality on a 5-point scale from three perspectives:\n\n"
        "Prompt Compliance\n"
        "1  Nothing removed, or an unrelated object edited.\n"
        "2  Target only partly removed, or a different instance/class deleted, or another object appears in the gap.\n"
        "3  Target mostly removed but extra objects also deleted, or fragments of the target remain.\n"
        "4  Only the specified objects removed, but a few tiny/background items deleted by mistake, or the count is wrong.\n"
        "5  Perfect: all and only the requested objects removed; every other element untouched.\n\n"
        "Visual Naturalness\n"
        "1  Image badly broken (large holes, strong artefacts).\n"
        "2  Clear erase marks; colour/resolution mismatch; background not restored.\n"
        "3  General look acceptable yet lighting/colour/style still clash; blur or noise visible.\n"
        "4  Style consistent; minor edge issues visible only when zoomed.\n"
        "5  Seamless: removal is virtually impossible to spot.\n\n"
        "Physical & Detail Integrity\n"
        "1  Severe physical errors (floating items, wrong perspective/light); key scene elements damaged; background heavily warped.\n"
        "2  Large un-filled gaps or obvious background shifts.\n"
        "3  Lighting, perspective and contacts mostly correct; flaws small and tolerable; background adjusted locally.\n"
        "4  Background reconstruction clean; existing details preserved; only minute changes outside the removal area.\n"
        "5  Physically flawless and even enhances realism: accurate light/shadow/texture infill, high-quality micro-details.\n"
        "The second and third score should no higher than first score!!!\n\n"
        "Example Response Format:\n"
        "Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\n"
        "Prompt Compliance: A number from 1 to 5.\n"
        "Visual Naturalness: A number from 1 to 5.\n"
        "Physical & Detail Integrity: A number from 1 to 5.\n"
        "editing instruction is : {instruction}.\n\n"
        "Below are the images before and after editing:\n"
    ),

    "adjust": (
        "You are a data rater specializing in grading attribute alteration edits. "
        "You will be given two images (before and after editing) and the corresponding editing instructions. "
        "Your task is to evaluate the attribute change on a 5-point scale from three perspectives:\n\n"
        "Prompt Compliance\n"
        "1  Target not adjusted, wrong object touched, or geometry changed.\n"
        "2  Right object but wrong attribute value/direction; only part edited; other objects also altered; slight stretch/crop.\n"
        "3  Mainly correct object and attribute, yet large hue/brightness/texture error; minor collateral edits; visible jaggies/distortion.\n"
        "4  All requested objects adjusted, only their attributes changed; shape kept; small inaccuracy in colour, material or amount.\n"
        "5  Exactly and only the requested objects adjusted; colour, material, gloss etc. match the prompt perfectly; shape 100% intact; zero unintended edits.\n\n"
        "Visual Seamlessness\n"
        "1  Massive colour spill, mosaics or heavy noise; image nearly unusable.\n"
        "2  Clear smears/bleeding on edges; abrupt resolution or tone shift; highlights/shadows clipped; background gaps.\n"
        "3  Overall palette OK but local tone or grain conflicts; soft edges; noticeable disharmony.\n"
        "4  Style unified, transitions smooth; only slight edge artefacts visible when zoomed.\n"
        "5  No detectable edit traces; colours/materials fuse with scene lighting; edit area practically invisible.\n\n"
        "Physical & Detail Fidelity\n"
        "1  Object floating, interpenetrating, or severe perspective/light mismatch; background badly warped.\n"
        "2  Missing shadows/highlights; wrong reflection direction; background visibly discoloured or distorted.\n"
        "3  Light, perspective and contact surface largely correct; minor acceptable flaws; background only locally affected.\n"
        "4  Adjusted material interacts believably with scene; shadows, highlights, reflections handled well; original details preserved.\n"
        "5  High physical realism: fine micro-highlights, diffuse bounce, subsurface effects present; overall scene realism improved.\n"
        "The second and third score should no higher than first score!!!\n\n"
        "Example Response Format:\n"
        "Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\n"
        "Prompt Compliance: A number from 1 to 5.\n"
        "Visual Seamlessness: A number from 1 to 5.\n"
        "Physical & Detail Fidelity: A number from 1 to 5.\n"
        "editing instruction is : {instruction}.\n\n"
        "Below are the images before and after editing:\n"
    ),

    "background": (
        "You are a data rater specializing in grading background editing. "
        "You will be given two images (before and after editing) and the editing instruction. "
        "Your task is to evaluate the background change on a 5-point scale from three perspectives:\n\n"
        "Instruction Compliance\n"
        "1  No change, or background unrelated to prompt, or foreground also replaced/distorted.\n"
        "2  Background partly replaced or wrong style/content; foreground noticeably altered.\n"
        "3  Main background replaced but elements missing/extra, or faint spill onto subject edges.\n"
        "4  Requested background fully present; foreground intact except minute artefacts or small prompt mismatch.\n"
        "5  Background exactly matches prompt (content, style, placement); all foreground pixels untouched.\n\n"
        "Visual Seamlessness\n"
        "1  Large tearing, posterisation, extreme blur/noise; edit area obvious at a glance.\n"
        "2  Clear cut-out halos, colour-resolution gap, or heavy smudge strokes.\n"
        "3  Blend acceptable but visible on closer look: slight edge blur, grain or palette shift.\n"
        "4  Nearly invisible seams; textures and sharpness aligned, only minor issues when zoomed in.\n"
        "5  Indistinguishable composite: edges, textures, resolution and colour grading perfectly continuous.\n\n"
        "Physical Consistency\n"
        "1  Severe mismatch: wrong horizon, conflicting light direction, floating subject, warped geometry.\n"
        "2  Noticeable but not extreme inconsistencies in light, shadows or scale; depth cues off.\n"
        "3  Overall believable; small errors in shadow length, perspective or ambient colour.\n"
        "4  Lighting, scale, depth, and camera angle well matched; only subtle discrepancies.\n"
        "5  Physically flawless: foreground and new background share coherent light, shadows, reflections, perspective.\n"
        "The second and third score should no higher than first score!!!\n\n"
        "Example Response Format:\n"
        "Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\n"
        "Instruction Compliance: A number from 1 to 5.\n"
        "Visual Seamlessness: A number from 1 to 5.\n"
        "Physical Consistency: A number from 1 to 5.\n"
        "editing instruction is : {instruction}.\n\n"
        "Below are the images before and after editing:\n"
    ),

    "compose": (
        "You are a data rater specializing in grading hybrid image edits (involving multiple operations on multiple objects). "
        "You will be given two images (before and after editing) and the editing instruction. "
        "Your task is to evaluate the overall editing quality on a 5-point scale from three perspectives:\n\n"
        "Instruction Compliance\n"
        "1  Neither object nor operations match the prompt; wrong items edited or shapes distorted.\n"
        "2  Only one object correctly edited, or both edited but with wrong/partial operations; collateral changes to other items.\n"
        "3  Both target objects touched, each with the requested operation broadly correct but missing details.\n"
        "4  Both objects receive the exact operations; tiny deviations in amount, position, or parameter. No unintended edits elsewhere.\n"
        "5  Perfect execution: each object fully reflects its specified operation, all other scene elements untouched.\n\n"
        "Visual Naturalness\n"
        "1  Large artefacts, obvious cut-outs, heavy blur/noise; edits conspicuous at a glance.\n"
        "2  Clear edge halos, colour or resolution mismatch, awkward scaling.\n"
        "3  Acceptable but visible on close look: slight edge softness, minor palette or focus shift.\n"
        "4  Edits blend smoothly; seams hard to spot, textures and sharpness largely consistent.\n"
        "5  Indistinguishable composite: colour grading, grain, resolution and style fully match the original image.\n\n"
        "Physical Consistency & Fine Detail\n"
        "1  Severe lighting/perspective mismatch, missing or wrong shadows; objects appear floating or warped.\n"
        "2  Noticeable but tolerable inconsistencies in illumination, scale, or depth cues.\n"
        "3  Generally plausible; small errors in shadow length, reflection angle, or texture alignment.\n"
        "4  Lighting, perspective, and material response closely match; only subtle flaws visible when zoomed.\n"
        "5  Physically flawless: shadows, highlights, reflections, depth and texture perfectly integrated.\n"
        "The second and third score should no higher than first score!!!\n\n"
        "Example Response Format:\n"
        "Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\n"
        "Instruction Compliance: A number from 1 to 5.\n"
        "Visual Naturalness: A number from 1 to 5.\n"
        "Physical Consistency & Fine Detail: A number from 1 to 5.\n"
        "editing instruction is : {instruction}.\n\n"
        "Below are the images before and after editing:\n"
    ),
}

# Map data operation keywords → prompt type
_OP_TYPE_MAP = {
    "add": "add",
    "remove": "remove",
    "replace": "replace",
    "alter": "adjust",
    "adjust": "adjust",
    "change": "adjust",
    "background": "background",
}


# ---------------------------------------------------------------------------
# Helper: operation parsing
# ---------------------------------------------------------------------------

def _parse_compose_ops(operation: Optional[str]) -> list:
    """
    Parse a numbered multi-op string into sub-operations.

    Input:  "1: Add, desc, [x,y,x,y]; 2: Remove, desc, [x,y,x,y]; 3: Alter, desc, [...]"
    Output: [(op_type, description, bbox_or_None), ...]

    Returns [] if the string is not in numbered format (→ treat as single op).
    """
    if not operation:
        return []
    if not re.search(r"\d\s*[:：]", operation):
        return []

    ops = []
    for part in re.split(r";\s*", operation.strip().rstrip(";")):
        part = part.strip()
        if not part:
            continue

        # Extract trailing bbox [x, y, x, y]
        bbox = None
        m = re.search(r",?\s*(\[[\d\s,.\-]+\])\s*$", part)
        if m:
            nums = re.findall(r"[-+]?\d*\.?\d+", m.group(1))
            if len(nums) >= 4:
                bbox = [float(n) for n in nums[:4]]
            part = part[: m.start()].strip()

        # Strip leading number "1: " or "1："
        part = re.sub(r"^\d+\s*[:：]\s*", "", part)

        # Split "Type, description"
        comma = part.find(",")
        if comma > 0:
            raw_type = part[:comma].strip().lower()
            description = part[comma + 1:].strip()
        else:
            raw_type = part.strip().lower()
            description = part.strip()

        op_type = _OP_TYPE_MAP.get(raw_type, "adjust")
        ops.append((op_type, description, bbox))

    return ops


def _detect_edit_type(operation: Optional[str]) -> str:
    """
    Infer single-op ImgEdit type from operation string.
    Returns: add | remove | replace | adjust | background | compose.
    """
    if not operation:
        return "compose"
    op = operation.lower()
    if re.search(r"\bremov", op):
        return "remove"
    if re.search(r"\badd\b", op):
        return "add"
    if re.search(r"\breplace\b", op):
        return "replace"
    if re.search(r"\bbackground\b|\bbg\b", op):
        return "background"
    if re.search(r"\bchange\b|\badjust\b|\balter\b|\bcolor\b|\bcolour\b|\btexture\b|\bscale\b|\brotate\b", op):
        return "adjust"
    return "compose"


# ---------------------------------------------------------------------------
# Helper: bbox annotation
# ---------------------------------------------------------------------------

def _annotate_bbox(image_path: str, bbox: list) -> Optional[str]:
    """
    Draw a red bounding box on a copy of image_path.
    Returns temp file path (caller must delete), or None if PIL is unavailable.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        for w in range(4):
            draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w], outline="red")
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(tmp.name, "JPEG", quality=95)
        return tmp.name
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

def _parse_three_scores(text: str) -> Optional[float]:
    """
    Parse 3-dimension ImgEdit response; apply cap rule; return raw average in [1,5].
    """
    scores = []
    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith("brief reasoning"):
            continue
        m = re.search(r":\s*([1-5])\b", line)
        if m:
            scores.append(int(m.group(1)))
    if not scores:
        return None
    s0 = scores[0]
    if len(scores) >= 3:
        return (s0 + min(scores[1], s0) + min(scores[2], s0)) / 3.0
    if len(scores) == 2:
        return (s0 + min(scores[1], s0)) / 2.0
    return float(s0)


def _extract_int_0_5(text: str) -> Optional[int]:
    m = re.search(r"\b([0-5])\b", text.strip())
    if m:
        return int(m.group(1))
    m = re.search(r"[0-5]", text.strip())
    return int(m.group()) if m else None


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def _encode_image(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/{mime};base64,{data}"


# ---------------------------------------------------------------------------
# Judge class
# ---------------------------------------------------------------------------

class QwenVLPlusJudge:
    def __init__(self, api_key: str, model: str = "qwen3-vl-plus", max_workers: int = 8,
                 max_rate: float = 10.0, text_rate: float = 20.0):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model
        self.max_workers = max_workers
        self._limiter = _RateLimiter(max_rate)        # for image-bearing requests
        self._text_limiter = _RateLimiter(text_rate)  # for text-only requests

    def _call(self, messages: list, max_tokens: int = 16, text_only: bool = False) -> str:
        limiter = self._text_limiter if text_only else self._limiter
        limiter.acquire()
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt == 2:
                    return f"ERROR: {e}"
                time.sleep(2 ** attempt)
        return ""

    # ------------------------------------------------------------------
    # Internal: score one editing operation
    # ------------------------------------------------------------------

    def _rate_single_op(
        self,
        original_image: str,
        edited_image: str,
        instruction: str,
        op_type: str,
        bbox: Optional[list] = None,
    ) -> float:
        """
        Score one editing operation using the ImgEdit rubric.
        If bbox is provided, both images are annotated with a red rectangle
        so the judge focuses on the correct region.
        Returns normalised score [0, 1]  (raw 1-5 → (score-1)/4).
        """
        prompt_template = _IMGEDIT_PROMPTS.get(op_type, _IMGEDIT_PROMPTS["compose"])
        prompt = prompt_template.format(instruction=instruction or "")

        orig_to_use = original_image
        edit_to_use = edited_image
        tmp_files = []

        if bbox:
            ann_orig = _annotate_bbox(original_image, bbox)
            ann_edit = _annotate_bbox(edited_image, bbox)
            if ann_orig:
                orig_to_use = ann_orig
                tmp_files.append(ann_orig)
            if ann_edit:
                edit_to_use = ann_edit
                tmp_files.append(ann_edit)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _encode_image(orig_to_use)}},
                    {"type": "image_url", "image_url": {"url": _encode_image(edit_to_use)}},
                ],
            }
        ]

        try:
            raw = self._call(messages, max_tokens=128)
            score = _parse_three_scores(raw)
            if score is None:
                fb = _extract_int_0_5(raw)
                score = float(fb) if fb is not None else None
        finally:
            for f in tmp_files:
                try:
                    os.remove(f)
                except OSError:
                    pass

        if score is None:
            return 0.0
        # raw in [1,5] → normalise to [0,1]
        return min(max((score - 1) / 4.0, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Public: rate_edit
    # ------------------------------------------------------------------

    def rate_edit(
        self,
        original_image: str,
        edited_image: str,
        instruction: str,
        operation: Optional[str] = None,
        bbox: Optional[str] = None,
    ) -> float:
        """
        Rate editing quality using the ImgEdit rubric.

        For multi-operation strings ("1: Add, ..., [bbox]; 2: Remove, ..."):
          - Each sub-operation is scored independently with its own bbox annotation.
          - Final score = mean of per-operation scores.

        For single-operation strings:
          - The external `bbox` parameter (string) is used for annotation if provided.

        Returns normalised score in [0, 1].
        """
        if not original_image or not os.path.exists(original_image):
            return 0.0
        if not edited_image or not os.path.exists(edited_image):
            return 0.0

        sub_ops = _parse_compose_ops(operation)

        if sub_ops:
            # Multi-operation: score each in order and average
            scores = []
            for op_type, description, sub_bbox in sub_ops:
                s = self._rate_single_op(
                    original_image, edited_image, description, op_type, sub_bbox
                )
                scores.append(s)
            return sum(scores) / len(scores)
        else:
            # Single operation
            op_type = _detect_edit_type(operation)
            parsed_bbox = None
            if bbox:
                nums = re.findall(r"[-+]?\d*\.?\d+", str(bbox))
                if len(nums) >= 4:
                    parsed_bbox = [float(n) for n in nums[:4]]
            return self._rate_single_op(
                original_image, edited_image, instruction, op_type, parsed_bbox
            )

    # ------------------------------------------------------------------
    # VQA / validation
    # ------------------------------------------------------------------

    def answer_yes_no(self, image_path: str, question: str) -> str:
        if not image_path or not os.path.exists(image_path):
            return "unknown"
        messages = [
            {"role": "system", "content": _SYS_VQA},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _encode_image(image_path)}},
                    {"type": "text", "text": question + "\nAnswer only 'yes' or 'no'."},
                ],
            },
        ]
        from .normalize import normalize_yes_no
        return normalize_yes_no(self._call(messages, max_tokens=8))

    def validate_generation(self, image_path: str, questions: Dict[str, str]) -> float:
        if not image_path or not os.path.exists(image_path):
            return 0.0
        if not questions:
            return 0.0
        results = []
        with ThreadPoolExecutor(max_workers=min(len(questions), self.max_workers)) as ex:
            futures = {
                ex.submit(self.answer_yes_no, image_path, q): qid
                for qid, q in questions.items()
            }
            for fut in as_completed(futures):
                results.append(1 if fut.result() == "yes" else 0)
        return sum(results) / len(results) if results else 0.0

    def validate_generation_with_answers(
        self, image_path: str, questions_with_answers: List[dict]
    ) -> float:
        if not image_path or not os.path.exists(image_path):
            return 0.0
        if not questions_with_answers:
            return 0.0
        results = []
        with ThreadPoolExecutor(max_workers=min(len(questions_with_answers), self.max_workers)) as ex:
            futures = {
                ex.submit(self.answer_yes_no, image_path, q["text"]): q["answer"]
                for q in questions_with_answers
            }
            for fut, expected in futures.items():
                results.append(1 if fut.result() == expected else 0)
        return sum(results) / len(results) if results else 0.0

    def rate_description_match(self, image_path: str, description: str) -> float:
        if not image_path or not os.path.exists(image_path):
            return 0.0
        prompt = (
            f"Rate how well this image matches the following description on a scale of 0 to 5.\n"
            f"Description: {description}\n\n"
            f"Respond with only a single integer from 0 to 5."
        )
        messages = [
            {"role": "system", "content": _SYS_RATER},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _encode_image(image_path)}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        raw = self._call(messages, max_tokens=8)
        score = _extract_int_0_5(raw)
        if score is None:
            return 0.0
        return min(max(score / 5.0, 0.0), 1.0)

    def score_mismatch_alignment(
        self, predicted: str, gt_operation: str, final_caption: str
    ) -> float:
        if not predicted or not gt_operation:
            return 0.0
        prompt = (
            "Rate how well the predicted image-caption mismatches align with the "
            "ground truth operations on a scale of 0 to 5.\n\n"
            f"Predicted mismatches: {predicted}\n"
            f"Ground truth operations: {gt_operation}\n"
            f"Target caption: {final_caption}\n\n"
            "5 = perfectly identifies all GT mismatches\n"
            "3 = partially correct\n"
            "0 = completely wrong\n\n"
            "Respond with only a single integer from 0 to 5."
        )
        messages = [
            {"role": "system", "content": _SYS_MISMATCH},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        raw = self._call(messages, max_tokens=8, text_only=True)
        score = _extract_int_0_5(raw)
        if score is None:
            return 0.0
        return min(max(score / 5.0, 0.0), 1.0)
