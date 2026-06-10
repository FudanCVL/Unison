"""Answer normalization utilities."""

import re
import os
from typing import Optional


def normalize_yes_no(text: str) -> str:
    """Normalize model output to 'yes', 'no', or 'unknown'."""
    if not text or str(text).lower() in ("nan", "none", ""):
        return "unknown"
    cleaned = str(text).strip().lower()
    if cleaned in {"yes", "no"}:
        return cleaned
    # First complete word wins
    tokens = re.split(r"[^a-z]+", cleaned)
    for token in tokens:
        if token in {"yes", "no"}:
            return token
    # Substring fallback – both present is ambiguous
    has_yes = bool(re.search(r"\byes\b", cleaned))
    has_no = bool(re.search(r"\bno\b", cleaned))
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return "unknown"


def normalize_option(text: str, options: dict) -> str:
    """
    Extract the chosen option letter from model output.
    options: {"A": "...", "B": "...", ...}
    Returns a letter key or "unknown".
    """
    if not text or str(text).lower() in ("nan", "none", ""):
        return "unknown"
    raw = str(text).strip()
    valid = {k.upper() for k in options}

    # 1. Standalone letter at start: "B", "B.", "B)", "B:"
    m = re.match(r"^([A-Z])[^a-zA-Z]", raw + " ")
    if m and m.group(1) in valid:
        return m.group(1)

    # 2. "Answer: B" / "answer is B" / "(B)"
    m = re.search(r"(?:answer[:\s]+|answer\s+is\s+|\()([A-Z])[^a-zA-Z]", raw + " ", re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        if letter in valid:
            return letter

    # 3. Any standalone option letter in the text
    m = re.search(r"\b([A-Z])\b", raw)
    if m:
        letter = m.group(1).upper()
        if letter in valid:
            return letter

    # 4. Option text fuzzy match (lowercase comparison)
    raw_lower = raw.lower()
    for k, v in options.items():
        if v.lower() in raw_lower:
            return k.upper()

    return "unknown"


def parse_bbox(
    text: str,
    img_w: Optional[int] = None,
    img_h: Optional[int] = None,
) -> Optional[list]:
    """
    Parse a bounding box from model text.
    Returns [x_min, y_min, x_max, y_max] clipped to image boundaries, or None.
    If img_w/img_h are provided, coordinates are clipped to [0, img_w] x [0, img_h].
    Otherwise coordinates are clipped to >= 0 at minimum.
    """
    if not text or str(text).lower() in ("nan", "none", ""):
        return None
    text = str(text)
    # Find 4 consecutive numbers (int or float)
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if len(nums) >= 4:
        try:
            coords = [float(n) for n in nums[:4]]
            x_min, y_min, x_max, y_max = coords
            # Clip to image boundaries
            x_min = max(0.0, x_min)
            y_min = max(0.0, y_min)
            if img_w is not None:
                x_min = min(x_min, float(img_w))
                x_max = min(x_max, float(img_w))
            if img_h is not None:
                y_min = min(y_min, float(img_h))
                y_max = min(y_max, float(img_h))
            if x_max > x_min and y_max > y_min:
                return [x_min, y_min, x_max, y_max]
        except ValueError:
            pass
    return None


def parse_image_path(text: str) -> Optional[str]:
    """
    Check whether model response looks like an image file path and return it,
    or None if it looks like plain text.
    """
    if not text or str(text).lower() in ("nan", "none", ""):
        return None
    text = str(text).strip()
    # Must end with a known image extension
    if re.search(r"\.(png|jpg|jpeg|webp|bmp|gif)$", text, re.IGNORECASE):
        return text
    # Also accept paths that contain image extensions mid-string (e.g. from multi-line responses)
    m = re.search(r"([^\s\"']+\.(?:png|jpg|jpeg|webp|bmp|gif))", text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None
