"""
For multi-operation edits (compose), each sub-operation is scored independently; the final score is the average.
"""

import base64
import csv
import json
import os
import re
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
    "Respond with only a single integer from 1 to 5."
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
        "editing instruction is : {instruction}.\n"
        "Evaluation scope: {bbox}\n"
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
        "editing instruction is : {instruction}.\n"
        "Evaluation scope: {bbox}\n"
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
        "editing instruction is : {instruction}.\n"
        "Evaluation scope: {bbox}\n"
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
        "editing instruction is : {instruction}.\n"
        "Evaluation scope: {bbox}\n"
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
        "editing instruction is : {instruction}.\n"
        "Evaluation scope: {bbox}\n"
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
        "editing instruction is : {instruction}.\n"
        "Evaluation scope: {bbox}\n"
        "Below are the images before and after editing:\n"
    ),
}

# Map data operation keywords → prompt type
_OP_TYPE_MAP = {
    "add": "add",
    "remove": "remove",
    "replace": "replace",
    "alter": "adjust",
    "change": "adjust",
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
    Infer single-op image editing type from operation string.
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


def _scope_text(bbox: Optional[list]) -> str:
    """Region-scoping instruction substituted into the ImgEdit ``{bbox}`` slot.

    Each of the three criteria is judged at a different spatial scope:

    1. Prompt/instruction compliance — INSIDE the target region (did the requested
       edit land on the intended target).
    2. Visual naturalness/seamlessness — IN AND AROUND the target region: seams,
       edges and how the edit blends with its immediate surroundings are local to the
       edit boundary, so this stays anchored near the box rather than the whole image.
    3. Physical & detail consistency — the WHOLE image: global lighting, shadows,
       reflections, background preservation and overall artefacts.
    """
    if bbox:
        coords = [int(v) for v in bbox]
        return (
            f"The target editing region (x1, y1, x2, y2 in [0, 1000]) is {coords}. "
            "Apply the three criteria at different spatial scopes: "
            "(1) judge the first criterion ONLY inside the target region (whether the "
            "requested edit was correctly applied to the intended target); "
            "(2) judge the second criterion IN AND AROUND the target region — seams, "
            "edges and how the edit blends with its immediate surroundings, anchored near "
            "the box rather than the whole image; "
            "(3) judge the third criterion over the WHOLE image — global lighting, "
            "shadows, reflections, background preservation and overall artefacts."
        )
    return "No target region is specified; judge all three criteria over the whole image."


def _strip_scope_line(tpl: str) -> str:
    """Remove the whole ``Evaluation scope: {bbox}`` line from an ImgEdit template.

    Used by the ``noscope`` bbox mode: the judge sees no region hint at all, not
    even the "No target region is specified" fallback text.
    """
    return "\n".join(ln for ln in tpl.split("\n") if "Evaluation scope:" not in ln)


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
    if not text or text.startswith("ERROR:"):
        return None
    m = re.search(r"\b([0-5])\b", text.strip())
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Per-call judge I/O logging (opt-in via env JUDGE_IO_LOG=<csv path>)
# One CSV row per single judge decision, flushed in real time. Used to build
# the Qwen3-VL-8B judge-distillation SFT set (each row = one inference + its
# integer score: VQA 0/1, ImgEdit per-dimension 1-5, mismatch 1-5).
# ---------------------------------------------------------------------------

_IO_LOCK = threading.Lock()
_IO_COLUMNS = ["call_type", "score", "dimension", "op_type", "instruction",
               "bbox", "question", "description", "operation", "predicted",
               "criteria", "img1", "img2", "raw"]
_DIM_CACHE: Dict[str, list] = {}


def _io_log(rec: dict) -> None:
    """Append one judge call as a CSV row (real-time flush). No-op unless JUDGE_IO_LOG is set."""
    path = os.environ.get("JUDGE_IO_LOG")
    if not path:
        return
    imgs = rec.get("images") or []
    bbox = rec.get("bbox")
    row = {c: "" for c in _IO_COLUMNS}
    row.update({
        "call_type": rec.get("call_type", ""),
        "score": rec.get("score", ""),
        "dimension": rec.get("dimension", ""),
        "op_type": rec.get("op_type", ""),
        "instruction": rec.get("instruction", ""),
        "bbox": json.dumps(bbox) if bbox is not None else "",
        "question": rec.get("question", ""),
        "description": rec.get("description", ""),
        "operation": rec.get("operation", ""),
        "predicted": rec.get("predicted", ""),
        "criteria": rec.get("criteria", ""),
        "img1": imgs[0] if len(imgs) > 0 else "",
        "img2": imgs[1] if len(imgs) > 1 else "",
        "raw": rec.get("raw", ""),
    })
    with _IO_LOCK:
        new = (not os.path.exists(path)) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_IO_COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)
            f.flush()


def _parse_ordered_scores(text: str) -> list:
    """All 1-5 integers from an ImgEdit response, in line order (dimension order)."""
    out = []
    for line in (text or "").split("\n"):
        line = line.strip()
        if line.lower().startswith("brief reasoning"):
            continue
        m = re.search(r":\s*([1-5])\b", line)
        if m:
            out.append(int(m.group(1)))
    return out


def _imgedit_dimensions(op_type: str) -> list:
    """[(dimension_name, criteria_block)] parsed from the ImgEdit rubric for op_type."""
    if op_type in _DIM_CACHE:
        return _DIM_CACHE[op_type]
    tpl = _IMGEDIT_PROMPTS.get(op_type, _IMGEDIT_PROMPTS["compose"])
    dims = []
    try:
        mid = tpl.split("three perspectives:\n\n", 1)[1].split("Example Response Format:", 1)[0]
        for seg in mid.split("\n\n"):
            lines = [l for l in seg.split("\n")
                     if not l.strip().lower().startswith("the second and third")]
            block = "\n".join(lines).strip()
            if not block:
                continue
            if re.search(r"\n\s*1\s", "\n" + block):   # has numbered 1..5 criteria
                dims.append((block.split("\n", 1)[0].strip(), block))
    except Exception:
        dims = []
    _DIM_CACHE[op_type] = dims[:3]
    return _DIM_CACHE[op_type]


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
# Mismatch P/R/F1 aggregation (pure — no API). Shared by score_mismatch_prf and
# any offline recompute over stored per-cell judge scores, so the two cannot drift.
# ---------------------------------------------------------------------------

def prf_from_matrix(matrix: List[List[float]], n_gt: int,
                    match_threshold: float = 0.5) -> dict:
    """Threshold-gated soft precision/recall/F1 from a P×G score matrix.

    matrix : n_pred rows × n_gt cols, each cell the normalised mismatch score in [0, 1]
             (raw 1-5 → (raw-1)/4: 3→0.5, 4→0.75, 5→1.0). Requires n_pred ≥ 1, n_gt ≥ 1.

    For each GT op take its best prediction (column max) and for each prediction its best
    GT op (row max); a best cell counts only when it reaches match_threshold (default 0.5
    == raw 3), otherwise it contributes 0 (a raw-2 "vaguely related" best match is a miss,
    not 0.25). recall = mean gated column-max, precision = mean gated row-max, F1 their
    harmonic mean. n_missed / n_extra are the below-threshold counts.
    """
    n_pred = len(matrix)
    col_max = [max(matrix[i][j] for i in range(n_pred)) for j in range(n_gt)]
    row_max = [max(matrix[i][j] for j in range(n_gt)) for i in range(n_pred)]
    recall = sum(v for v in col_max if v >= match_threshold) / n_gt
    precision = sum(v for v in row_max if v >= match_threshold) / n_pred
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    n_missed = sum(1 for v in col_max if v < match_threshold)
    n_extra = sum(1 for v in row_max if v < match_threshold)
    return {"precision": precision, "recall": recall, "f1": f1,
            "n_extra": n_extra, "n_missed": n_missed}


# ---------------------------------------------------------------------------
# Judge class
# ---------------------------------------------------------------------------

class QwenVLPlusJudge:
    # ImgEdit (UGG / ME-generation) bbox handling:
    #   "full"    – embed the GT region-scoping instruction (default, legacy behaviour)
    #   "noscope" – drop the bbox AND remove the "Evaluation scope:" line entirely
    # Mismatch (ME misalignment) ALWAYS keeps its bbox, regardless of this setting.
    # Inherited by LocalQwenVLJudge; set per-run from --imgedit-bbox-mode.
    imgedit_bbox_mode: str = "full"

    def __init__(self, api_key: str, model: str = "qwen3-vl-plus", max_workers: int = 8,
                 max_rate: float = 10.0, text_rate: float = 20.0,
                 enable_thinking: bool = False, thinking_max_tokens: int = 2048):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model
        self.max_workers = max_workers
        self._limiter = _RateLimiter(max_rate)        # for image-bearing requests
        self._text_limiter = _RateLimiter(text_rate)  # for text-only requests
        # Qwen3-VL "thinking" mode. DashScope requires stream=True with enable_thinking,
        # and the answer arrives after the (discarded) reasoning trace, so the token
        # budget is raised. Off by default — production scoring stays non-thinking.
        self.enable_thinking = enable_thinking
        self.thinking_max_tokens = thinking_max_tokens

    def _call(self, messages: list, max_tokens: int = 16, text_only: bool = False) -> str:
        limiter = self._text_limiter if text_only else self._limiter
        limiter.acquire()
        for attempt in range(3):
            try:
                if self.enable_thinking:
                    return self._call_thinking(messages, max(max_tokens, self.thinking_max_tokens))
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

    def _call_thinking(self, messages: list, max_tokens: int) -> str:
        """Streamed call with Qwen3-VL thinking enabled. Returns only the answer
        ``content`` (the ``reasoning_content`` trace is consumed and discarded), so the
        caller's parsing is unchanged from the non-thinking path."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
            extra_body={"enable_thinking": True},
        )
        parts: list = []
        for chunk in stream:
            if not chunk.choices:
                continue
            piece = getattr(chunk.choices[0].delta, "content", None)
            if piece:
                parts.append(piece)
        return "".join(parts).strip()

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
        If bbox is provided, coordinates are embedded in the prompt text so the
        judge focuses on the correct region.
        Returns normalised score [0, 1]  (raw 1-5 → (score-1)/4).
        """
        prompt_template = _IMGEDIT_PROMPTS.get(op_type, _IMGEDIT_PROMPTS["compose"])
        if self.imgedit_bbox_mode == "noscope":
            # judge sees no region at all: drop the bbox instruction AND the scope line
            prompt = _strip_scope_line(prompt_template).format(
                instruction=instruction or "", bbox="")
            bbox = None  # keep the I/O log honest about what the judge actually saw
        else:
            prompt = prompt_template.format(instruction=instruction or "", bbox=_scope_text(bbox))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _encode_image(original_image)}},
                    {"type": "image_url", "image_url": {"url": _encode_image(edited_image)}},
                ],
            }
        ]

        raw = self._call(messages, max_tokens=128)
        # log one row per scored dimension (the judge's raw 1-5 integer, uncapped)
        ints = _parse_ordered_scores(raw)
        for (dname, dcriteria), dscore in zip(_imgedit_dimensions(op_type), ints):
            _io_log({"call_type": "imgedit", "op_type": op_type, "dimension": dname,
                     "criteria": dcriteria, "instruction": instruction or "",
                     "bbox": [int(v) for v in bbox] if bbox else None,
                     "images": [os.path.abspath(original_image), os.path.abspath(edited_image)],
                     "score": dscore, "raw": raw})
        score = _parse_three_scores(raw)
        if score is None:
            fb = _extract_int_0_5(raw)
            score = float(fb) if fb is not None else None

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
        raw = self._call(messages, max_tokens=8)
        ans = normalize_yes_no(raw)
        if ans in ("yes", "no"):
            _io_log({"call_type": "vqa", "images": [os.path.abspath(image_path)],
                     "question": question, "score": 1 if ans == "yes" else 0, "raw": raw})
        return ans

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
            for fut in as_completed(futures):
                results.append(1 if fut.result() == futures[fut] else 0)
        return sum(results) / len(results) if results else 0.0

    def _score_single_mismatch_op(
        self, predicted: str, op_description: str,
        bbox: Optional[list] = None,
    ) -> float:
        if bbox:
            coords = [int(v) for v in bbox]
            bbox_line = f"Target region (x1, y1, x2, y2 in [0, 1000]): {coords}.\n"
        else:
            bbox_line = ""
        prompt = (
            "Rate how well the predicted image-caption mismatch identifies the following "
            "ground truth operation on a scale of 1 to 5.\n\n"
            f"Ground truth operation: {op_description}\n"
            f"{bbox_line}"
            f"Predicted mismatch: {predicted}\n\n"
            "5 = clearly and correctly identifies this specific operation with accurate location\n"
            "4 = mostly correct; minor location or description inaccuracy\n"
            "3 = partially identifies this operation but with significant gaps\n"
            "2 = vaguely related but does not correctly identify this operation\n"
            "1 = mentions relevant content but fails to identify this operation\n\n"
            "Respond with only a single integer from 1 to 5."
        )
        messages = [
            {"role": "system", "content": _SYS_MISMATCH},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        raw = self._call(messages, max_tokens=8, text_only=True)
        score = _extract_int_0_5(raw)
        if score is not None:
            _io_log({"call_type": "mismatch", "images": [],
                     "operation": op_description,
                     "bbox": [int(v) for v in bbox] if bbox else None,
                     "predicted": predicted, "score": score, "raw": raw})
        if score is None:
            return 0.0
        return min(max((score - 1) / 4.0, 0.0), 1.0)

    def score_mismatch_prf(
        self,
        predicted_items: List[str],
        gt_operation: str,
        max_pred: int = 30,
        match_threshold: float = 0.5,
    ) -> Optional[dict]:
        """
        Soft precision / recall / F1 of predicted mismatches against GT operations.

        Rather than scoring one merged blob against each GT op (recall only), this
        scores *each discrete predicted mismatch*
        against *each GT sub-operation*, building a P×G matrix M[i][j] in [0, 1]
        (the 1-5 mismatch rubric, normalised: raw 3→0.5, 4→0.75, 5→1.0). Then, with
        non-injective max-matching **gated at match_threshold**: a GT op (or a
        prediction) only counts as matched when its best cell reaches the threshold
        (default 0.5 == raw 3); below that the best cell contributes 0, not its raw
        partial value (so a raw-2 "vaguely related" best match is a miss, not 0.25):

            recall    = mean_j (max_i M[i][j] if max_i M[i][j] >= t else 0)   # GT ops covered ≥ raw 3
            precision = mean_i (max_j M[i][j] if max_j M[i][j] >= t else 0)   # preds hitting a GT op ≥ raw 3
            f1        = 2·P·R / (P + R)

        A matched mismatch keeps its graded value (0.5 / 0.75 / 1.0); only
        genuinely unmatched predictions (best cell below the threshold) lower
        precision, and uncovered GT ops lower recall. This gives the model a
        reason not to over-report mismatches, which the recall-only scorer lacked.

        predicted_items : list of discrete mismatch strings (already split & deduped).
        gt_operation    : GT operation string ("1: Add, ..., [bbox]; 2: Remove, ...");
                          parsed via _parse_compose_ops, single-op fallback.
        max_pred        : cap on predicted items scored (cost guard); overflow counted
                          in the returned n_dropped, not silently ignored.

        Returns {precision, recall, f1, n_pred, n_gt, n_extra, n_missed, n_dropped}
        (n_extra/n_missed are the counts below match_threshold; the same threshold gates P/R),
        or None if there are no GT operations to score against.
        """
        sub_ops = _parse_compose_ops(gt_operation)
        if not sub_ops:
            if not gt_operation:
                return None
            sub_ops = [("", gt_operation, None)]
        gt_descs = [(description, bbox) for _, description, bbox in sub_ops]
        n_gt = len(gt_descs)
        if n_gt == 0:
            return None

        items = [str(p).strip() for p in (predicted_items or []) if p and str(p).strip()]
        n_dropped = max(0, len(items) - max_pred)
        items = items[:max_pred]
        n_pred = len(items)

        if n_pred == 0:
            # Model identified nothing: recall 0, precision undefined → F1 0.
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "n_pred": 0, "n_gt": n_gt, "n_extra": 0, "n_missed": n_gt,
                    "n_dropped": n_dropped}

        # Build the P×G soft-score matrix concurrently (nested pool — same pattern
        # as validate_generation_with_answers; the shared rate limiter is the real throttle).
        matrix = [[0.0] * n_gt for _ in range(n_pred)]
        cells = [(i, j) for i in range(n_pred) for j in range(n_gt)]
        with ThreadPoolExecutor(max_workers=min(len(cells), self.max_workers)) as ex:
            fut_map = {}
            for i, j in cells:
                description, bbox = gt_descs[j]
                fut_map[ex.submit(self._score_single_mismatch_op, items[i], description, bbox)] = (i, j)
            for fut in as_completed(fut_map):
                i, j = fut_map[fut]
                try:
                    matrix[i][j] = fut.result()
                except Exception:
                    matrix[i][j] = 0.0

        agg = prf_from_matrix(matrix, n_gt, match_threshold)
        return {**agg, "n_pred": n_pred, "n_gt": n_gt, "n_dropped": n_dropped}
