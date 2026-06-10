#!/usr/bin/env python3
"""
Quick smoke-test: evaluate 3 items per task with the real Qwen3-VL-Plus judge.
Run from Evaluation_Pipeline/ directory.
"""

import json
import sys
import os
import pandas as pd

sys.path.insert(0, ".")

from common.judge import QwenVLPlusJudge
from common.io import load_csv, success_rows

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
INFERENCE_BASE = "../Inference_Pipeline"
DATA_DIR = "../data"
MODEL = "Janus-Pro-7B"
N = 3  # items per task


def limit_csv(csv_path: str, n: int, key_col: str = "dialogue_index") -> str:
    """Write a temp CSV with only the first N dialogue indices."""
    df = load_csv(csv_path)
    top_indices = sorted(df[key_col].dropna().unique())[:n]
    subset = df[df[key_col].isin(top_indices)]
    tmp = csv_path.replace(".csv", f"_test{n}.csv")
    subset.to_csv(tmp, index=False)
    return tmp


def run_ic(judge):
    from tasks.evaluate_ic import evaluate_ic
    csv = f"{INFERENCE_BASE}/result/{MODEL}/IC/IC_{MODEL}_results.csv"
    tmp = limit_csv(csv, N)
    result = evaluate_ic(
        csv_path=tmp, data_dir=f"{DATA_DIR}/Internal_Consistency",
        judge=judge, inference_base_dir=INFERENCE_BASE, max_workers=4,
    )
    os.remove(tmp)
    return result


def run_ugg(judge):
    from tasks.evaluate_ugg import evaluate_ugg
    csv = f"{INFERENCE_BASE}/result/{MODEL}/UGG/UGG_{MODEL}_results.csv"
    tmp = limit_csv(csv, N)
    result = evaluate_ugg(
        csv_path=tmp, data_dir=f"{DATA_DIR}/Und_Guided_Gen",
        judge=judge, inference_base_dir=INFERENCE_BASE, max_workers=4,
    )
    os.remove(tmp)
    return result


def run_ggu(judge):
    from tasks.evaluate_ggu import evaluate_ggu
    csv = f"{INFERENCE_BASE}/result/TokenFlow/GGU/GGU_TokenFlow_results.csv"
    tmp = limit_csv(csv, N)
    result = evaluate_ggu(
        csv_path=tmp, data_dir=f"{DATA_DIR}/Gen_Guided_Und",
        judge=judge, inference_base_dir=INFERENCE_BASE, max_workers=4,
    )
    os.remove(tmp)
    return result


def print_result(task_id: str, result: dict):
    scores = result["scores"]
    stats = result["stats"]
    print(f"\n{'─'*50}")
    print(f"  {task_id}")
    print(f"  understanding : {scores['understanding_score']}")
    print(f"  generation    : {scores['generation_score']}")
    print(f"  unified       : {scores['unified_score']}")
    print(f"  scored/total  : {stats['num_scored']}/{stats['num_total']}")
    if result["details"]:
        print(f"  --- per-item details ---")
        for d in result["details"]:
            idx = d["dialogue_index"]
            u = round(d["understanding_item"], 3)
            g = round(d["generation_item"], 3)
            uni = round(d["unified_item"], 3)
            print(f"    [{idx}] u={u}  g={g}  uni={uni}")


def main():
    judge = QwenVLPlusJudge(api_key=API_KEY, max_workers=6)

    results = {}

    print("Running IC ...")
    results["IC"] = run_ic(judge)
    print_result("IC", results["IC"])

    print("\nRunning UGG ...")
    results["UGG"] = run_ugg(judge)
    print_result("UGG", results["UGG"])

    print("\nRunning GGU ...")
    results["GGU"] = run_ggu(judge)
    print_result("GGU", results["GGU"])

    # Save
    with open("test_eval_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n\nSaved to test_eval_results.json")


if __name__ == "__main__":
    main()
