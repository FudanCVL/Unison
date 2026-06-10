"""ME (Mutual Enhancement) task evaluation."""

import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from common.io import load_csv, success_rows, load_me_gt, resolve_path


def _append_row(path: str, row: dict):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
from common.normalize import parse_image_path
from common.aggregate import clip, build_task_result, null_task_result
from common.judge import QwenVLPlusJudge


def _last_valid_response(rows_by_round: dict, odd_only: bool = False) -> tuple:
    """
    Given {round_number: model_response}, return (round_num, response) for
    the last round with a non-null response.  If odd_only=True, only consider
    odd rounds (editing rounds in me_u2g or understanding rounds in me_g2u).
    Returns (None, None) if nothing found.
    """
    rounds = sorted(rows_by_round.keys())
    if odd_only:
        rounds = [r for r in rounds if r % 2 == 1]
    for r in reversed(rounds):
        resp = rows_by_round.get(r, "")
        if resp and str(resp).lower() not in ("nan", "none", ""):
            return r, resp
    return None, None


def _round5_or_last(rows_by_round: dict, odd_only: bool = False) -> tuple:
    """Return round-5 response if valid, otherwise fall back to last valid round."""
    r5 = rows_by_round.get(5, "")
    if r5 and str(r5).lower() not in ("nan", "none", ""):
        return 5, r5
    return _last_valid_response(rows_by_round, odd_only=odd_only)


def evaluate_me(
    csv_path: str,
    data_dir: str,
    judge: QwenVLPlusJudge,
    inference_base_dir: str,
    max_workers: int = 8,
    output_csv: str = None,
) -> dict:
    """
    Evaluate ME task.

    CSV operation_index mapping:
      0 = me_u2g (self-refining image editing)
          odd rounds  = editing   → model_response = edited image path
          even rounds = evaluation → model_response = 'Yes' / 'No, ...'
      1 = me_g2u (self-refining multimodal understanding)
          odd rounds  = understanding → model_response = mismatch text
          even rounds = editing        → model_response = edited image path

    Scores per sample:
      generation_item    = judge.rate_edit(orig, final_edit_u2g, instruction)
      understanding_item = judge.score_mismatch_alignment(last_mismatch, operation, final_caption)
      unified_item       = (generation_item + understanding_item) / 2
    """
    if not os.path.exists(csv_path):
        print(f"[ME] CSV not found: {csv_path}")
        return null_task_result("ME")

    df = load_csv(csv_path)
    sdf = success_rows(df)

    # Load GT
    gt: dict = {}
    if data_dir and os.path.isdir(data_dir):
        try:
            gt = load_me_gt(data_dir)
        except Exception as e:
            print(f"[ME] Warning: could not load GT data: {e}")

    import json

    # u2g_rounds[idx][round_num] = model_response
    # g2u_rounds[idx][round_num] = model_response
    u2g_rounds: dict = {}
    g2u_rounds: dict = {}
    input_data_cache: dict = {}

    for _, row in sdf.iterrows():
        idx = int(row["dialogue_index"])
        op_idx_raw = row.get("operation_index", None)
        op_idx = int(op_idx_raw) if op_idx_raw is not None else -1
        round_raw = row.get("round_number", None)
        resp = str(row.get("model_response", "") or "")

        if idx not in input_data_cache:
            try:
                input_data_cache[idx] = json.loads(str(row.get("input_data", "{}")))
            except Exception:
                input_data_cache[idx] = {}

        # Determine round_number
        try:
            round_num = int(float(round_raw)) if round_raw is not None and str(round_raw).lower() not in ("nan", "none", "") else None
        except (TypeError, ValueError):
            round_num = None

        if op_idx == 0:  # me_u2g
            u2g_rounds.setdefault(idx, {})
            if round_num is not None:
                u2g_rounds[idx][round_num] = resp
            else:
                # Flat result (model didn't do multi-round): treat as round 1
                u2g_rounds[idx][1] = resp

        elif op_idx == 1:  # me_g2u
            g2u_rounds.setdefault(idx, {})
            if round_num is not None:
                g2u_rounds[idx][round_num] = resp
            else:
                g2u_rounds[idx][1] = resp

    all_indices = set(input_data_cache)
    if not all_indices:
        print("[ME] No data found in CSV.")
        return null_task_result("ME")

    # --- Collect judge calls ---
    # (a) rate_edit for u2g: (orig, final_edit_image, instruction, operation)
    # (b) score_mismatch for g2u: (last_mismatch_text, operation, final_caption)
    rate_edit_tasks: dict = {}     # idx -> (orig, edited, instruction, operation)
    mismatch_tasks: dict = {}       # idx -> (predicted, gt_operation, final_caption)

    for idx in sorted(all_indices):
        idata = input_data_cache.get(idx, {})
        gt_item = gt.get(idx, {})

        orig_raw = idata.get("image_path", "") or gt_item.get("image_path", "")
        orig_path = resolve_path(orig_raw, inference_base_dir)
        if not os.path.exists(orig_path):
            orig_path = gt_item.get("image_path", "")

        instruction = idata.get("instruction", "") or gt_item.get("instruction", "")
        operation = idata.get("operation", "") or gt_item.get("operation", "")
        final_caption = idata.get("final_caption", "") or gt_item.get("final_caption", "")

        # u2g: prefer round-5 editing response, fallback to last valid odd round
        u2g_data = u2g_rounds.get(idx, {})
        _, final_edit_resp = _round5_or_last(u2g_data, odd_only=True)
        if final_edit_resp:
            img = parse_image_path(final_edit_resp)
            if img:
                edit_path = resolve_path(img, inference_base_dir)
                if orig_path and os.path.exists(orig_path) and os.path.exists(edit_path):
                    rate_edit_tasks[idx] = (orig_path, edit_path, instruction, operation or None)

        # g2u: prefer round-5 understanding response, fallback to last valid odd round
        g2u_data = g2u_rounds.get(idx, {})
        _, last_mismatch = _round5_or_last(g2u_data, odd_only=True)
        if last_mismatch and operation:
            mismatch_tasks[idx] = (last_mismatch, operation, final_caption)

    total_judge = len(rate_edit_tasks) + len(mismatch_tasks)
    print(f"[ME] {len(all_indices)} samples, {total_judge} judge calls queued...")

    gen_scores: dict = {}
    und_scores: dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {}
        for idx, args in rate_edit_tasks.items():
            fut_map[ex.submit(judge.rate_edit, *args)] = ("gen", idx)
        for idx, args in mismatch_tasks.items():
            fut_map[ex.submit(judge.score_mismatch_alignment, *args)] = ("und", idx)

        with tqdm(total=total_judge, desc="[ME] judge", unit="call") as pbar:
            for fut in as_completed(fut_map):
                kind, idx = fut_map[fut]
                try:
                    score = fut.result()
                except Exception:
                    score = 0.0
                if kind == "gen":
                    gen_scores[idx] = score
                else:
                    und_scores[idx] = score
                pbar.update(1)

    # --- Compute per-sample scores ---
    details = []
    failure_reasons: dict = {}

    for idx in sorted(all_indices):
        generation_item = clip(gen_scores.get(idx, 0.0))
        understanding_item = clip(und_scores.get(idx, 0.0))
        unified_item = clip((generation_item + understanding_item) / 2.0)

        row = {
            "dialogue_index": idx,
            "understanding_item": understanding_item,
            "generation_item": generation_item,
            "unified_item": unified_item,
        }
        details.append(row)
        if output_csv:
            _append_row(output_csv, row)

    return build_task_result("ME", details, len(all_indices), failure_reasons)
