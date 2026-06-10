"""UGG (Understanding-Guided Generation) task evaluation."""

import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from common.io import load_csv, success_rows, load_ugg_gt, resolve_path
from common.normalize import parse_bbox, parse_image_path
from common.geometry import compute_region_iou
from common.aggregate import clip, build_task_result, null_task_result
from common.judge import QwenVLPlusJudge


def _append_row(path: str, row: dict):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def evaluate_ugg(
    csv_path: str,
    data_dir: str,
    judge: QwenVLPlusJudge,
    inference_base_dir: str,
    max_workers: int = 8,
    output_csv: str = None,
) -> dict:
    """
    Evaluate UGG task.

    CSV operation_index mapping:
      0 = understanding  → predicted bbox (text)
      1 = editing        → direct edit image path
      2 = unify          → guided edit image path

    Scores per sample:
      understanding_item = IoU(pred_bbox, GT)
      generation_item    = judge.rate_edit(orig, direct_edit, instruction)
      unified_item       = (IoU + judge.rate_edit(orig, guided_edit, ...)) / 2
    """
    if not os.path.exists(csv_path):
        print(f"[UGG] CSV not found: {csv_path}")
        return null_task_result("UGG")

    df = load_csv(csv_path)
    sdf = success_rows(df)

    # Load GT
    gt: dict = {}
    if data_dir and os.path.isdir(data_dir):
        try:
            gt = load_ugg_gt(data_dir)
        except Exception as e:
            print(f"[UGG] Warning: could not load GT data: {e}")

    # --- Parse rows ---
    # bbox_preds[idx] = raw model response text
    # direct_edits[idx] = absolute image path or ""
    # guided_edits[idx] = absolute image path or ""
    bbox_preds: dict = {}
    direct_edits: dict = {}
    guided_edits: dict = {}
    input_data_cache: dict = {}  # idx -> parsed input_data dict

    import json

    for _, row in sdf.iterrows():
        idx = int(row["dialogue_index"])
        op_idx = int(row.get("operation_index", -1))
        resp = str(row.get("model_response", "") or "")

        # Cache input_data for GT resolution
        if idx not in input_data_cache:
            try:
                input_data_cache[idx] = json.loads(str(row.get("input_data", "{}")))
            except Exception:
                input_data_cache[idx] = {}

        if op_idx == 0:  # understanding → bbox prediction
            bbox_preds[idx] = resp

        elif op_idx == 1:  # editing → direct edit
            img = parse_image_path(resp)
            if img:
                direct_edits[idx] = resolve_path(img, inference_base_dir)

        elif op_idx == 2:  # unify → guided edit
            img = parse_image_path(resp)
            if img:
                guided_edits[idx] = resolve_path(img, inference_base_dir)

    all_indices = set(input_data_cache)
    if not all_indices:
        print("[UGG] No data found in CSV.")
        return null_task_result("UGG")

    # --- Collect judge calls ---
    # We need to call rate_edit for:
    #   (a) direct edit: (orig, direct_edit, instruction, operation) for generation_score
    #   (b) guided edit: (orig, guided_edit, instruction, operation) for unified_score
    # Collect all into a flat list and submit concurrently

    # idx -> (orig, edited, instruction, operation, bbox_str)
    judge_tasks_direct: dict = {}
    judge_tasks_guided: dict = {}

    for idx in sorted(all_indices):
        idata = input_data_cache.get(idx, {})

        # Resolve original image
        orig_raw = idata.get("image_path", "")
        orig_path = resolve_path(orig_raw, inference_base_dir)
        if not os.path.exists(orig_path) and idx in gt:
            orig_path = gt[idx].get("image_path", "")

        instruction = idata.get("instruction", "") or (gt.get(idx, {}).get("instruction", ""))
        operation = idata.get("operation", "") or (gt.get(idx, {}).get("operation", ""))
        if not operation or operation.lower() in ("nan", "none"):
            operation = None

        # GT bbox passed to judge so it can annotate the edit region
        bbox = idata.get("bbox", "") or gt.get(idx, {}).get("bbox", "")

        direct_path = direct_edits.get(idx, "")
        guided_path = guided_edits.get(idx, "")

        if orig_path and os.path.exists(orig_path):
            if direct_path and os.path.exists(direct_path):
                judge_tasks_direct[idx] = (orig_path, direct_path, instruction, operation, bbox)
            if guided_path and os.path.exists(guided_path):
                judge_tasks_guided[idx] = (orig_path, guided_path, instruction, operation, bbox)

    total_judge = len(judge_tasks_direct) + len(judge_tasks_guided)
    print(f"[UGG] {len(all_indices)} samples, {total_judge} judge calls queued...")

    direct_scores: dict = {}   # idx -> float
    guided_scores: dict = {}   # idx -> float

    def _rate(args):
        orig, edited, instruction, operation, bbox = args
        return judge.rate_edit(orig, edited, instruction, operation, bbox)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {}
        for idx, args in judge_tasks_direct.items():
            fut_map[ex.submit(_rate, args)] = ("direct", idx)
        for idx, args in judge_tasks_guided.items():
            fut_map[ex.submit(_rate, args)] = ("guided", idx)

        with tqdm(total=total_judge, desc="[UGG] judge", unit="call") as pbar:
            for fut in as_completed(fut_map):
                kind, idx = fut_map[fut]
                try:
                    score = fut.result()
                except Exception:
                    score = 0.0
                if kind == "direct":
                    direct_scores[idx] = score
                else:
                    guided_scores[idx] = score
                pbar.update(1)

    # --- Compute per-sample scores ---
    details = []
    failure_reasons: dict = {}

    for idx in sorted(all_indices):
        idata = input_data_cache.get(idx, {})
        gt_item = gt.get(idx, {})

        gt_bbox = idata.get("bbox", "") or gt_item.get("bbox", "")
        gt_mask = idata.get("mask", "") or gt_item.get("mask", "")
        pred_text = bbox_preds.get(idx, "")

        # Understanding: IoU
        iou = compute_region_iou(pred_text, gt_bbox, gt_mask)
        understanding_item = clip(iou)

        # Generation: judge score on direct edit
        generation_item = clip(direct_scores.get(idx, 0.0))

        # Unified: (IoU + guided_edit_score) / 2
        guided_edit_score = clip(guided_scores.get(idx, 0.0))
        unified_item = clip((understanding_item + guided_edit_score) / 2.0)

        row = {
            "dialogue_index": idx,
            "understanding_item": understanding_item,
            "generation_item": generation_item,
            "unified_item": unified_item,
            "iou": iou,
            "guided_edit_score": guided_edit_score,
        }
        details.append(row)
        if output_csv:
            _append_row(output_csv, row)

    return build_task_result("UGG", details, len(all_indices), failure_reasons)
