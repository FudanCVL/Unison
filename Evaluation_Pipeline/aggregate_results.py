#!/usr/bin/env python3
"""Aggregate all evaluation_results/*.csv files into evaluation_summary.json.

Usage:
    python aggregate_results.py
    python aggregate_results.py --eval-dir /path/to/evaluation_results
"""

import argparse
import csv
import glob
import json
import os
from statistics import mean

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EVAL_DIR = os.path.join(SCRIPT_DIR, "evaluation_results")
TASKS = ["IC", "GGU", "UGG", "ME"]
METRICS = ["understanding_score", "generation_score", "unified_score"]


def load_csv_scores(path: str) -> dict | None:
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return None
    try:
        return {
            "understanding_score": mean(float(r["understanding_item"]) for r in rows),
            "generation_score":    mean(float(r["generation_item"])    for r in rows),
            "unified_score":       mean(float(r["unified_item"])       for r in rows),
            "num_scored": len(rows),
            "num_total":  len(rows),
        }
    except (KeyError, ValueError) as e:
        print(f"  Warning: skipping {os.path.basename(path)}: {e}")
        return None


def discover_results(eval_dir: str) -> dict:
    """Return {model_name: {task: scores_dict}} from evaluation_results/*.csv."""
    models: dict = {}
    for path in sorted(glob.glob(os.path.join(eval_dir, "*.csv"))):
        fname = os.path.basename(path)
        matched = False
        for task in TASKS:
            suffix = f"_{task}.csv"
            if fname.endswith(suffix):
                model = fname[: -len(suffix)]
                scores = load_csv_scores(path)
                if scores is not None:
                    models.setdefault(model, {})[task] = scores
                matched = True
                break
        if not matched:
            print(f"  Skipping unrecognised file: {fname}")
    return models


def build_summary(models: dict) -> dict:
    summary_models = {}
    for model_name, tasks in models.items():
        task_scores = {
            task: {
                "understanding_score": round(s["understanding_score"], 6),
                "generation_score":    round(s["generation_score"],    6),
                "unified_score":       round(s["unified_score"],       6),
                "num_scored": s["num_scored"],
                "num_total":  s["num_total"],
            }
            for task, s in tasks.items()
        }
        unified_vals = [v["unified_score"] for v in task_scores.values()]
        overall = round(mean(unified_vals), 6) if unified_vals else 0.0
        summary_models[model_name] = {"overall_score": overall, "tasks": task_scores}

    leaderboard = {
        task: {
            metric: [
                {"model": m, "score": v["tasks"][task][metric]}
                for m, v in sorted(
                    summary_models.items(),
                    key=lambda kv: kv[1]["tasks"].get(task, {}).get(metric, -1),
                    reverse=True,
                )
                if task in v["tasks"]
            ]
            for metric in METRICS
        }
        for task in TASKS
    }

    return {"models": summary_models, "leaderboard": leaderboard}


def print_table(summary: dict) -> None:
    models = summary["models"]
    if not models:
        return

    # --- compact unified-score table ---
    col = 10
    header = f"{'Model':<24} {'Overall':>8}  " + "  ".join(
        f"{t + ':uni':>{col}}" for t in TASKS
    )
    print(header)
    print("-" * len(header))
    for name in sorted(models):
        v = models[name]
        scores = "  ".join(
            f"{v['tasks'][t]['unified_score']:>{col}.4f}" if t in v["tasks"] else f"{'N/A':>{col}}"
            for t in TASKS
        )
        print(f"{name:<24} {v['overall_score']:>8.4f}  {scores}")

    # --- detailed table ---
    print(f"\n{'Model':<24} {'Task':<5} {'Und':>8} {'Gen':>8} {'Uni':>8} {'n':>6}")
    print("-" * 65)
    for name in sorted(models):
        v = models[name]
        for task in TASKS:
            if task not in v["tasks"]:
                continue
            t = v["tasks"][task]
            print(
                f"{name:<24} {task:<5} "
                f"{t['understanding_score']:>8.4f} "
                f"{t['generation_score']:>8.4f} "
                f"{t['unified_score']:>8.4f} "
                f"{t['num_scored']:>6}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Unison evaluation results.")
    parser.add_argument(
        "--eval-dir",
        default=DEFAULT_EVAL_DIR,
        help="Directory containing {Model}_{Task}.csv files (default: evaluation_results/)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(SCRIPT_DIR, "evaluation_summary.json"),
        help="Output JSON path (default: evaluation_summary.json)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.eval_dir):
        print(f"Directory not found: {args.eval_dir}")
        return

    print(f"Scanning: {args.eval_dir}")
    models = discover_results(args.eval_dir)
    if not models:
        print("No result CSVs found.")
        return

    found = [(m, list(t.keys())) for m, t in sorted(models.items())]
    for model, tasks in found:
        print(f"  {model}: {', '.join(tasks)}")

    summary = build_summary(models)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {args.out}\n")

    print_table(summary)


if __name__ == "__main__":
    main()
