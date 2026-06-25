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


def compute_region_iou(
    pred_text: str,
    gt_bbox_str: str,
    gt_mask_str: Optional[str] = None,
    img_w: Optional[int] = None,
    img_h: Optional[int] = None,
    pixel_space: bool = False,
) -> float:
    """
    Compute IoU between model prediction and ground truth region.
    gt_bbox_str must be in 0-1000 relative coordinates.
    gt_mask_str contains pixel-coordinate polygon; img_w/img_h are required to
    convert it to 0-1000 relative before comparison.
    Falls back to bbox IoU when mask conversion is not possible.
    Non-parseable prediction → IoU = 0.

    pixel_space — pass True for models that output absolute pixel coordinates
                  (e.g. OmniGen2, UniWorld); img_w/img_h are used to convert.
    """
    from .normalize import parse_bbox

    pred_bbox = parse_bbox(pred_text, img_w=img_w, img_h=img_h, pixel_space=pixel_space)
    if pred_bbox is None:
        return 0.0

    # Try mask IoU: convert pixel-space mask polygon to 0-1000 relative bbox
    if gt_mask_str and str(gt_mask_str).lower() not in ("nan", "none", "") and img_w and img_h:
        gt_poly = _parse_polygon(gt_mask_str)
        if gt_poly:
            gt_mask_bbox_px = _polygon_bbox(gt_poly)
            gt_mask_bbox = [
                gt_mask_bbox_px[0] / img_w * 1000,
                gt_mask_bbox_px[1] / img_h * 1000,
                gt_mask_bbox_px[2] / img_w * 1000,
                gt_mask_bbox_px[3] / img_h * 1000,
            ]
            return bbox_iou(pred_bbox, gt_mask_bbox)

    # Fall back to bbox IoU
    gt_bbox = parse_bbox(gt_bbox_str)
    if gt_bbox is None:
        return 0.0
    return bbox_iou(pred_bbox, gt_bbox)
