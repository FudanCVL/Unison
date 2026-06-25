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
    pixel_space: bool = False,
) -> Optional[list]:
    """
    Parse a bounding box from model text.
    Returns [x_min, y_min, x_max, y_max] in 0-1000 relative coordinates, or None.

    pixel_space=True  — model output is absolute pixel coordinates; img_w/img_h
                        required to convert to 0-1000 space.
    pixel_space=False — relative coordinates: 0-1 range is auto-scaled to 0-1000;
                        values already in 0-1000 range are used as-is.
    """
    if not text or str(text).lower() in ("nan", "none", ""):
        return None
    text = str(text)
    # Prefer explicit bbox_2d JSON (e.g. OmniGen2 / UniWorld output)
    # to avoid extracting stray digits from field names like "bbox_2d".
    m = re.search(r'"bbox_2d"\s*:\s*\[([^\]]+)\]', text)
    if m:
        nums = re.findall(r"[-+]?\d*\.?\d+", m.group(1))
    else:
        nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if len(nums) >= 4:
        try:
            coords = [float(n) for n in nums[:4]]
            x_min, y_min, x_max, y_max = coords

            if pixel_space:
                # Convert absolute pixel coords to 0-1000 relative space.
                if img_w and img_h:
                    x_min = x_min / img_w * 1000
                    y_min = y_min / img_h * 1000
                    x_max = x_max / img_w * 1000
                    y_max = y_max / img_h * 1000
                else:
                    return None  # can't convert without image size
            else:
                # Auto-scale 0-1 normalized coords to 0-1000.
                if max(x_max, y_max) <= 1.0:
                    x_min, y_min, x_max, y_max = (
                        x_min * 1000, y_min * 1000, x_max * 1000, y_max * 1000
                    )

            x_min = max(0.0, x_min)
            y_min = max(0.0, y_min)
            x_max = min(x_max, 1000.0)
            y_max = min(y_max, 1000.0)
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
