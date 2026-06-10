#!/usr/bin/env python3
"""
Unison evaluation pipeline — all four tasks with Qwen3-VL-Plus as judge.

Usage
-----
python evaluate_unison.py \
    --result-dir  ../Inference_Pipeline/result/BAGEL-7B-MoT \
    --data-dir    ../data \
    --inference-base-dir ../Inference_Pipeline \
    --api-key     sk-xxxx \
    --output      eval_BAGEL-7B-MoT.json

--result-dir should contain subdirectories IC/, UGG/, GGU/, ME/ with
*_results.csv files produced by the Inference_Pipeline.
"""

import argparse
import json
import os
import sys

from common.judge import QwenVLPlusJudge
from tasks.evaluate_ic import evaluate_ic
from tasks.evaluate_ugg import evaluate_ugg
from tasks.evaluate_ggu import evaluate_ggu
from tasks.evaluate_me import evaluate_me

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "evaluation_results")


def _result_csv_path(model_name: str, task_id: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f"{model_name}_{task_id}.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_csv(task_dir: str, task_id: str) -> str:
    """Return path to the *_results.csv inside task_dir, or '' if not found."""
    if not os.path.isdir(task_dir):
        return ""
    for fname in os.listdir(task_dir):
        if fname.endswith("_results.csv") and task_id in fname:
            return os.path.join(task_dir, fname)
    # Fallback: any CSV in the directory
    for fname in os.listdir(task_dir):
        if fname.endswith(".csv"):
            return os.path.join(task_dir, fname)
    return ""


def _overall_score(task_results: dict) -> float | None:
    scores = []
    for task_id, result in task_results.items():
        uni = result["scores"].get("unified_score")
        if uni is not None:
            scores.append(uni)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unison evaluation pipeline")
    p.add_argument(
        "--result-dir", required=True,
        help="Path to model result directory (contains IC/, UGG/, GGU/, ME/ subdirs)",
    )
    p.add_argument(
        "--data-dir", default=None,
        help="Path to benchmark data directory (needed for GT quality questions). "
             "Defaults to <result-dir>/../../data relative to result-dir.",
    )
    p.add_argument(
        "--inference-base-dir", default=None,
        help="Base directory for resolving relative image paths in CSVs "
             "(typically the Inference_Pipeline/ folder). "
             "Defaults to parent of result-dir.",
    )
    p.add_argument(
        "--api-key", default=None,
        help="DashScope API key. Falls back to DASHSCOPE_API_KEY env var.",
    )
    p.add_argument(
        "--model", default="qwen3-vl-plus",
        help="Judge model name (default: qwen3-vl-plus)",
    )
    p.add_argument(
        "--output", default=None,
        help="Output JSON path. Defaults to eval_<model_name>.json in current dir.",
    )
    p.add_argument(
        "--tasks", default="IC,UGG,GGU,ME",
        help="Comma-separated list of tasks to evaluate (default: IC,UGG,GGU,ME)",
    )
    p.add_argument(
        "--max-workers", type=int, default=8,
        help="Max concurrent judge API threads (default: 8)",
    )
    p.add_argument(
        "--max-items", type=int, default=None,
        help="Limit number of items per task (for quick testing)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve directories
    result_dir = os.path.abspath(args.result_dir)
    model_name = os.path.basename(result_dir)

    if args.data_dir:
        data_dir = os.path.abspath(args.data_dir)
    else:
        # Guess: Inference_Pipeline/result/<model>/ → ../../data
        data_dir = os.path.normpath(os.path.join(result_dir, "..", "..", "data"))
        if not os.path.isdir(data_dir):
            data_dir = ""
            print(f"[WARN] Could not auto-detect data-dir; GT data won't be loaded.")

    if args.inference_base_dir:
        inference_base_dir = os.path.abspath(args.inference_base_dir)
    else:
        # Guess parent of result-dir = Inference_Pipeline/
        # result_dir = .../Inference_Pipeline/result/<model>
        inference_base_dir = os.path.normpath(os.path.join(result_dir, "..", ".."))

    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("[ERROR] No DashScope API key. Pass --api-key or set DASHSCOPE_API_KEY.", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or f"eval_{model_name}.json"

    print(f"Model:              {model_name}")
    print(f"Result dir:         {result_dir}")
    print(f"Data dir:           {data_dir}")
    print(f"Inference base dir: {inference_base_dir}")
    print(f"Output:             {output_path}")
    print()

    # Build judge
    judge = QwenVLPlusJudge(api_key=api_key, model=args.model, max_workers=args.max_workers)

    # Run evaluators
    tasks_to_run = [t.strip().upper() for t in args.tasks.split(",")]
    task_data_dirs = {
        "IC":  os.path.join(data_dir, "Internal_Consistency"),
        "UGG": os.path.join(data_dir, "Und_Guided_Gen"),
        "GGU": os.path.join(data_dir, "Gen_Guided_Und"),
        "ME":  os.path.join(data_dir, "Mutual_Enhancement"),
    }
    task_evaluators = {
        "IC":  evaluate_ic,
        "UGG": evaluate_ugg,
        "GGU": evaluate_ggu,
        "ME":  evaluate_me,
    }

    task_results = {}

    for task_id in tasks_to_run:
        if task_id not in task_evaluators:
            print(f"[WARN] Unknown task: {task_id}, skipping.")
            continue

        task_result_dir = os.path.join(result_dir, task_id)
        csv_path = _find_csv(task_result_dir, task_id)
        task_data_dir = task_data_dirs.get(task_id, "")

        print(f"{'='*60}")
        print(f"Task: {task_id}")
        print(f"  CSV:      {csv_path or '(not found)'}")
        print(f"  Data dir: {task_data_dir}")

        evaluator = task_evaluators[task_id]
        out_csv = _result_csv_path(model_name, task_id)
        result = evaluator(
            csv_path=csv_path,
            data_dir=task_data_dir,
            judge=judge,
            inference_base_dir=inference_base_dir,
            max_workers=args.max_workers,
            output_csv=out_csv,
        )
        print(f"  Saved: {out_csv}")
        task_results[task_id] = result

        scores = result["scores"]
        stats = result["stats"]
        print(
            f"  understanding={scores['understanding_score']}, "
            f"generation={scores['generation_score']}, "
            f"unified={scores['unified_score']} "
            f"[{stats['num_scored']}/{stats['num_total']} scored]"
        )

    print(f"{'='*60}")
    overall = _overall_score(task_results)
    print(f"Overall score (mean of task unified scores): {overall}")

    # Build output JSON
    output = {
        "model_name": model_name,
        "summary": {
            "overall_score": overall,
            "tasks": {
                task_id: result["scores"]
                for task_id, result in task_results.items()
            },
        },
        "stats": {
            task_id: result["stats"]
            for task_id, result in task_results.items()
        },
        "details": {
            task_id: result["details"]
            for task_id, result in task_results.items()
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
