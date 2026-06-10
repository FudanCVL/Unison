#!/usr/bin/env python3
"""Aggregate all eval_*.json files into a single evaluation_summary.json."""

import glob
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS = ["IC", "GGU", "UGG", "ME"]
METRICS = ["understanding_score", "generation_score", "unified_score"]


def load_model_result(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_summary(results: list[dict]) -> dict:
    models = {}
    for r in results:
        name = r["model_name"]
        task_scores = r["summary"]["tasks"]
        stats = r.get("stats", {})

        row = {"overall_score": r["summary"]["overall_score"], "tasks": {}}
        for task in TASKS:
            if task not in task_scores:
                continue
            s = task_scores[task]
            row["tasks"][task] = {
                "understanding_score": round(s.get("understanding_score", 0), 6),
                "generation_score":    round(s.get("generation_score", 0), 6),
                "unified_score":       round(s.get("unified_score", 0), 6),
                "num_scored": stats.get(task, {}).get("num_scored"),
                "num_total":  stats.get(task, {}).get("num_total"),
            }
        models[name] = row

    # Per-metric leaderboard across models
    leaderboard = {}
    for task in TASKS:
        leaderboard[task] = {}
        for metric in METRICS:
            ranking = sorted(
                [(m, v["tasks"][task][metric]) for m, v in models.items() if task in v["tasks"]],
                key=lambda x: x[1], reverse=True,
            )
            leaderboard[task][metric] = [{"model": m, "score": s} for m, s in ranking]

    return {"models": models, "leaderboard": leaderboard}


def main():
    pattern = os.path.join(SCRIPT_DIR, "eval_*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print("No eval_*.json files found.")
        return

    results = [load_model_result(p) for p in paths]
    summary = build_summary(results)

    out_path = os.path.join(SCRIPT_DIR, "evaluation_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")

    # Print table
    print(f"\n{'Model':<20} {'Overall':>8}  " + "  ".join(f"{t+':uni':>12}" for t in TASKS))
    print("-" * 90)
    for name, v in summary["models"].items():
        scores = "  ".join(
            f"{v['tasks'][t]['unified_score']:>12.4f}" if t in v["tasks"] else f"{'N/A':>12}"
            for t in TASKS
        )
        print(f"{name:<20} {v['overall_score']:>8.4f}  {scores}")


if __name__ == "__main__":
    main()
