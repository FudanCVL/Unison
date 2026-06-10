"""Bounding box and mask IoU utilities."""

import re
from typing import Optional


def bbox_iou(pred: list, gt: list) -> float:
    """Compute IoU of two bboxes [x_min, y_min, x_max, y_max]."""
    try:
        px1, py1, px2, py2 = [float(v) for v in pred]
        gx1, gy1, gx2, gy2 = [float(v) for v in gt]
    except (TypeError, ValueError):
        return 0.0

    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union_area = pred_area + gt_area - inter_area

    if union_area <= 0:
        return 0.0
    return min(1.0, max(0.0, inter_area / union_area))


def _parse_polygon(raw: str) -> Optional[list]:
    """
    Parse polygon from a string like "[[x1, y1, x2, y2, ...]]" or "[x1, y1, ...]".
    Returns flat list of (x, y) tuples or None.
    """
    nums = re.findall(r"[-+]?\d*\.?\d+", str(raw))
    if len(nums) < 6:
        return None
    coords = [float(n) for n in nums]
    # Pair up as (x, y)
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]
    return points if len(points) >= 3 else None


def _polygon_bbox(points: list) -> list:
    """Get axis-aligned bbox of a polygon."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def compute_region_iou(pred_text: str, gt_bbox_str: str, gt_mask_str: Optional[str] = None) -> float:
    """
    Compute IoU between model prediction and ground truth region.
    Tries mask IoU first (if gt_mask_str available), falls back to bbox IoU.
    Non-parseable prediction → IoU = 0.
    """
    from .normalize import parse_bbox

    pred_bbox = parse_bbox(pred_text)
    if pred_bbox is None:
        return 0.0

    # Try mask IoU: convert mask polygon to bbox and compute bbox IoU
    # (Full pixel-level mask IoU requires image dimensions; bbox approximation is used here)
    if gt_mask_str and str(gt_mask_str).lower() not in ("nan", "none", ""):
        gt_poly = _parse_polygon(gt_mask_str)
        if gt_poly:
            gt_mask_bbox = _polygon_bbox(gt_poly)
            return bbox_iou(pred_bbox, gt_mask_bbox)

    # Fall back to bbox IoU
    gt_bbox = parse_bbox(gt_bbox_str)
    if gt_bbox is None:
        return 0.0
    return bbox_iou(pred_bbox, gt_bbox)
