"""IC (Internal Consistency) task evaluation."""

import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from tqdm import tqdm

from common.io import load_csv, success_rows, load_ic_gt, resolve_path
from common.normalize import normalize_yes_no, parse_image_path
from common.aggregate import clip, build_task_result, null_task_result
from common.judge import ClosedSourceJudge


def _append_row(path: str, row: dict):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def evaluate_ic(
    csv_path: str,
    data_dir: str,
    judge: ClosedSourceJudge,
    inference_base_dir: str,
    max_workers: int = 8,
    output_csv: str = None,
) -> dict:
    """
    Evaluate IC task.

    Scores per sample:
      understanding_item = (# model 'yes' on original image) / n_questions
      generation_item    = (# judge 'yes' on generated image) / n_questions
      unified_item       = (# both 'yes' on same question) / n_questions
    """
    if not os.path.exists(csv_path):
        print(f"[IC] CSV not found: {csv_path}")
        return null_task_result("IC")

    df = load_csv(csv_path)
    sdf = success_rows(df)

    # --- Load GT (for question texts; CSV also has them but let's be safe) ---
    gt = {}
    if data_dir and os.path.isdir(data_dir):
        try:
            gt = load_ic_gt(data_dir)
        except Exception as e:
            print(f"[IC] Warning: could not load GT data: {e}")

    # --- Parse understanding and generation rows ---
    # und_answers[dialogue_index][question_id] = 'yes'/'no'/'unknown'
    # und_questions[dialogue_index][question_id] = question_text
    # gen_paths[dialogue_index] = absolute image path or ""
    und_answers: dict = {}
    und_questions: dict = {}
    gen_paths: dict = {}

    for _, row in sdf.iterrows():
        idx = int(row["dialogue_index"])
        op = str(row.get("operation_type", ""))

        if op == "understanding":
            qid = str(row.get("question_id", "")).strip()
            if not qid or qid.lower() in ("nan", "none", ""):
                continue
            resp = str(row.get("model_response", "") or "")
            q_text = str(row.get("question_text", "") or "")
            und_answers.setdefault(idx, {})[qid] = normalize_yes_no(resp)
            und_questions.setdefault(idx, {})[qid] = q_text

        elif op == "generation":
            resp = str(row.get("model_response", "") or "")
            img = parse_image_path(resp)
            if img:
                abs_path = resolve_path(img, inference_base_dir)
                gen_paths[idx] = abs_path

    # All dialogue indices seen
    all_indices = set(und_answers) | set(gen_paths)
    if not all_indices:
        print("[IC] No data found in CSV.")
        return null_task_result("IC")

    # --- Collect judge calls: for each (idx, qid) where gen_path exists ---
    judge_tasks = []  # (idx, qid, gen_path, question_text)
    for idx in sorted(all_indices):
        gen_path = gen_paths.get(idx, "")
        if gen_path and os.path.exists(gen_path):
            questions = und_questions.get(idx, {})
            for qid, q_text in questions.items():
                if q_text:
                    judge_tasks.append((idx, qid, gen_path, q_text))

    print(f"[IC] {len(all_indices)} samples, {len(judge_tasks)} judge calls queued...")

    # --- Execute judge calls concurrently ---
    judge_results: dict = {}  # (idx, qid) -> 'yes'/'no'/'unknown'
    if judge_tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(judge.answer_yes_no, gen_path, q_text): (idx, qid)
                for idx, qid, gen_path, q_text in judge_tasks
            }
            with tqdm(total=len(judge_tasks), desc="[IC] judge", unit="call") as pbar:
                for fut in as_completed(futures):
                    key = futures[fut]
                    try:
                        judge_results[key] = fut.result()
                    except Exception:
                        judge_results[key] = "unknown"
                    pbar.update(1)

    # --- Compute per-sample scores ---
    details = []
    failure_reasons: dict = {}

    for idx in sorted(all_indices):
        und_ans = und_answers.get(idx, {})
        questions = und_questions.get(idx, {})
        gen_path = gen_paths.get(idx, "")
        n = len(questions)

        if n == 0:
            failure_reasons[str(idx)] = "no_understanding_rows"
            continue

        und_yes = sum(1 for qid in questions if und_ans.get(qid) == "yes")
        understanding_item = clip(und_yes / n)

        if gen_path and os.path.exists(gen_path):
            gen_yes = sum(
                1 for qid in questions if judge_results.get((idx, qid)) == "yes"
            )
            generation_item = clip(gen_yes / n)
            both_yes = sum(
                1 for qid in questions
                if und_ans.get(qid) == "yes" and judge_results.get((idx, qid)) == "yes"
            )
            unified_item = clip(both_yes / n)
        else:
            generation_item = 0.0
            unified_item = 0.0

        row = {
            "dialogue_index": idx,
            "num_questions": n,
            "understanding_item": understanding_item,
            "generation_item": generation_item,
            "unified_item": unified_item,
            "gen_path": gen_path,
        }
        details.append(row)
        if output_csv:
            _append_row(output_csv, row)

    return build_task_result("IC", details, len(all_indices), failure_reasons)
