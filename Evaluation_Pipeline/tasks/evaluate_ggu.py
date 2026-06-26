"""GGU (Generation-Guided Understanding) task evaluation."""

import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from common.io import load_csv, success_rows, load_ggu_gt, resolve_path
from common.normalize import normalize_option, parse_image_path
from common.aggregate import clip, build_task_result, null_task_result
from common.judge import ClosedSourceJudge


def _append_row(path: str, row: dict):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _generation_validation_score(
    judge: ClosedSourceJudge,
    gen_path: str,
    validate_questions: list,
    description: str,
) -> float:
    """
    Compute generation quality score for one GGU sample.
    Uses validate_questions if available, otherwise description match.
    """
    if not gen_path or not os.path.exists(gen_path):
        return 0.0

    # Generation quality is scored ONLY from the preset per-question validation
    # (per-question yes/no). No direct image-description scoring fallback.
    if validate_questions:
        return judge.validate_generation_with_answers(gen_path, validate_questions)

    return 0.0


def evaluate_ggu(
    csv_path: str,
    data_dir: str,
    judge: ClosedSourceJudge,
    inference_base_dir: str,
    max_workers: int = 8,
    output_csv: str = None,
) -> dict:
    """
    Evaluate GGU task.

    CSV operation_index mapping:
      0 = understanding → direct QA answer (text)
      1 = generation    → intermediate generated image path
      2 = unify         → guided QA answer using orig + gen image (text)

    Scores per sample:
      understanding_item    = 1 if direct_answer == GT else 0
      generation_item       = generation_validation_score(intermediate_image)
      unified_item          = (guided_answer_correct + generation_item) / 2
    """
    if not os.path.exists(csv_path):
        print(f"[GGU] CSV not found: {csv_path}")
        return null_task_result("GGU")

    df = load_csv(csv_path)
    sdf = success_rows(df)

    # Load GT and build question-text → validate_questions lookup
    # (dialogue_index may not align with GT index when subset data was used)
    q_text_to_validate: dict = {}
    if data_dir and os.path.isdir(data_dir):
        try:
            gt = load_ggu_gt(data_dir)
            for gt_item in gt.values():
                q = gt_item.get("question", "").strip()
                if q and gt_item.get("validate_questions"):
                    q_text_to_validate[q] = gt_item["validate_questions"]
        except Exception as e:
            print(f"[GGU] Warning: could not load GT data: {e}")

    import json

    direct_answers: dict = {}   # idx -> model answer text
    gen_paths: dict = {}         # idx -> abs image path
    guided_answers: dict = {}    # idx -> model answer text
    input_data_cache: dict = {}

    for _, row in sdf.iterrows():
        idx = int(row["dialogue_index"])
        op_idx = int(row.get("operation_index", -1))
        resp = str(row.get("model_response", "") or "")

        if idx not in input_data_cache:
            try:
                input_data_cache[idx] = json.loads(str(row.get("input_data", "{}")))
            except Exception:
                input_data_cache[idx] = {}

        if op_idx == 0:  # understanding → direct QA
            direct_answers[idx] = resp

        elif op_idx == 1:  # generation → intermediate image
            img = parse_image_path(resp)
            if img:
                gen_paths[idx] = resolve_path(img, inference_base_dir)

        elif op_idx == 2:  # unify → guided QA
            guided_answers[idx] = resp

    all_indices = set(input_data_cache)
    if not all_indices:
        print("[GGU] No data found in CSV.")
        return null_task_result("GGU")

    # --- Collect generation validation judge calls ---
    # Each call: (gen_path, validate_questions, description)
    gen_val_tasks: dict = {}  # idx -> (gen_path, validate_questions, description)
    for idx in sorted(all_indices):
        gen_path = gen_paths.get(idx, "")
        if not gen_path or not os.path.exists(gen_path):
            continue
        idata = input_data_cache.get(idx, {})
        q_text = idata.get("question", "").strip()
        # Lookup validate_questions by question text (robust across full/subset data)
        validate_qs = q_text_to_validate.get(q_text, [])
        # Also check image_generation_validate in input_data (for complex_relation)
        if not validate_qs:
            igv = idata.get("image_generation_validate", {})
            if isinstance(igv, dict) and igv:
                validate_qs = [{"text": v, "answer": "yes"} for v in igv.values()]
        description = idata.get("description", "")
        gen_val_tasks[idx] = (gen_path, validate_qs, description)

    print(f"[GGU] {len(all_indices)} samples, {len(gen_val_tasks)} generation validation calls...")

    gen_val_scores: dict = {}  # idx -> float

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_generation_validation_score, judge, gen_path, validate_qs, description): idx
            for idx, (gen_path, validate_qs, description) in gen_val_tasks.items()
        }
        with tqdm(total=len(gen_val_tasks), desc="[GGU] judge", unit="call") as pbar:
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    gen_val_scores[idx] = fut.result()
                except Exception:
                    gen_val_scores[idx] = 0.0
                pbar.update(1)

    # --- Compute per-sample scores ---
    details = []
    failure_reasons: dict = {}

    for idx in sorted(all_indices):
        idata = input_data_cache.get(idx, {})

        # All GT fields come from input_data stored in the CSV
        options = idata.get("options", {})
        gt_answer = str(idata.get("answer", "")).strip().upper()

        if not options or not gt_answer:
            failure_reasons[str(idx)] = "missing_options_or_answer"
            continue

        # Understanding: direct answer correctness
        direct_text = direct_answers.get(idx, "")
        direct_pred = normalize_option(direct_text, options)
        understanding_item = clip(1.0 if direct_pred == gt_answer else 0.0)

        # Generation: validation score
        generation_item = clip(gen_val_scores.get(idx, 0.0))

        # Unified: (guided_correct + generation) / 2
        guided_text = guided_answers.get(idx, "")
        guided_pred = normalize_option(guided_text, options)
        guided_correct = 1.0 if guided_pred == gt_answer else 0.0
        unified_item = clip((guided_correct + generation_item) / 2.0)

        row = {
            "dialogue_index": idx,
            "category": idata.get("category", ""),
            "gt_answer": gt_answer,
            "direct_pred": direct_pred,
            "guided_pred": guided_pred,
            "understanding_item": understanding_item,
            "generation_item": generation_item,
            "guided_answer_correct": guided_correct,
            "unified_item": unified_item,
        }
        details.append(row)
        if output_csv:
            _append_row(output_csv, row)

    return build_task_result("GGU", details, len(all_indices), failure_reasons)
