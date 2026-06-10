#!/usr/bin/env python3
"""Evaluate GGU 2D/3D spatial generated-image quality via judge VLM.

Reads model-inference GGU CSV, extracts (orig_image_path, generated_image_path)
from operation_type == 'generation' rows, loads quality_questions GT from
2d_spatial.json / spatial.json, runs a judge VLM (Qwen3-VL by default) on the
generated image only with each yes/no question, compares against GT (all yes),
writes summary + details JSON.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


SUBTASK_TOKENS = {
    "2D_Spatial": "2D_Spatial",
    "3D_Spatial": "3D_Spatial",
}


def detect_subtask(image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (sub_task, image_path_relative_to_subtask_dir) or (None, None)."""
    norm = image_path.replace("\\", "/")
    for sub, token in SUBTASK_TOKENS.items():
        needle = f"/{token}/"
        idx = norm.find(needle)
        if idx >= 0:
            rel = norm[idx + len(needle):]
            return sub, rel
    return None, None


def load_generation_rows(csv_path: str) -> List[Dict]:
    """Return list of dicts for operation_type=='generation' rows.

    Each dict: {sub_task, image_path_rel, generated_image_path, raw_image_path}.
    Rows with unknown sub_task are skipped.
    """
    df = pd.read_csv(csv_path)
    df = df[df["operation_type"] == "generation"]
    out = []
    for _, row in df.iterrows():
        sub, rel = detect_subtask(str(row["image_path"]))
        if sub is None:
            continue
        out.append({
            "sub_task": sub,
            "image_path_rel": rel,
            "generated_image_path": str(row["model_response"]),
            "raw_image_path": str(row["image_path"]),
        })
    return out


SUBTASK_JSON_FILES = {
    "2D_Spatial": "2d_spatial.json",
    "3D_Spatial": "spatial.json",
}


def normalize_answer(raw: str) -> str:
    """Normalize judge VLM output to 'yes' or 'no'.

    Same rule as evaluate_ic.py: take the first whitespace-stripped token,
    lowercase, strip trailing punctuation. Anything that isn't 'yes' → 'no'.
    """
    if raw is None:
        return "no"
    text = str(raw).strip().lower()
    if not text:
        return "no"
    first = text.split()[0].rstrip(".,!?;:")
    return "yes" if first == "yes" else "no"


def build_gt_index(data_dir: str, subtasks: List[str]) -> Dict[Tuple[str, str], List[Dict]]:
    """Load quality_questions from each enabled subtask's JSON.

    Returns dict keyed by (sub_task, image_path_relative).
    Missing JSON files are warned and skipped (not an error).
    """
    idx: Dict[Tuple[str, str], List[Dict]] = {}
    for sub in subtasks:
        filename = SUBTASK_JSON_FILES.get(sub)
        if filename is None:
            print(f"[warn] Unknown subtask '{sub}', skipping")
            continue
        path = Path(data_dir) / sub / filename
        if not path.exists():
            print(f"[warn] {path} not found, skipping")
            continue
        items = json.loads(path.read_text(encoding="utf-8"))
        for it in items:
            qq = it.get("quality_questions")
            if not qq:
                continue
            idx[(sub, it["image_path"])] = qq
    return idx


from collections import defaultdict


def aggregate_summary(details: List[Dict], errors: Dict[str, int],
                      model_name: str, judge_model: str) -> Dict:
    """Build the summary dict for the result JSON."""
    total_q = 0
    correct_q = 0
    per_sub = defaultdict(lambda: {"samples": 0, "questions": 0, "correct": 0})
    per_type = defaultdict(lambda: {"questions": 0, "correct": 0})

    for d in details:
        sub = d["sub_task"]
        per_sub[sub]["samples"] += 1
        for q in d["questions"]:
            total_q += 1
            per_sub[sub]["questions"] += 1
            per_type[q["type"]]["questions"] += 1
            if q["correct"]:
                correct_q += 1
                per_sub[sub]["correct"] += 1
                per_type[q["type"]]["correct"] += 1

    def _acc(c, n):
        return (c / n) if n > 0 else 0.0

    summary = {
        "model_name": model_name,
        "judge_model": judge_model,
        "total_samples": len(details),
        "total_questions": total_q,
        "overall_accuracy": _acc(correct_q, total_q),
        "per_subtask": {
            sub: {
                "samples":  v["samples"],
                "questions": v["questions"],
                "accuracy": _acc(v["correct"], v["questions"]),
            } for sub, v in per_sub.items()
        },
        "per_question_type": {
            t: {
                "questions": v["questions"],
                "accuracy": _acc(v["correct"], v["questions"]),
            } for t, v in per_type.items()
        },
        "errors": errors,
    }
    return summary


SYSTEM_PROMPT = (
    "You are a precise VQA assistant. Answer each question with only 'yes' or 'no'. "
    "If the information is unknown or cannot be determined from the image, answer 'no'."
)


def build_judge_messages(image_path: str, question_text: str) -> List[Dict]:
    """Construct messages payload for one judge VLM call."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": f"Question: {question_text}"},
        ]},
    ]


def score_sample(sample_id: str, sub_task: str, generated_image: str,
                 quality_questions: List[Dict], predictions: List[str]) -> Dict:
    """Combine GT + predictions into a per-sample detail dict."""
    assert len(quality_questions) == len(predictions), (
        f"Length mismatch for {sample_id}: "
        f"{len(quality_questions)} questions vs {len(predictions)} preds"
    )
    qs_out = []
    n_correct = 0
    for q, pred in zip(quality_questions, predictions):
        norm_pred = normalize_answer(pred)
        correct = (norm_pred == q["answer"])
        if correct:
            n_correct += 1
        qs_out.append({
            "id": q["id"],
            "type": q["type"],
            "gt": q["answer"],
            "pred": norm_pred,
            "correct": correct,
        })
    return {
        "sample_id": sample_id,
        "sub_task": sub_task,
        "generated_image": generated_image,
        "sample_accuracy": (n_correct / len(quality_questions))
                            if quality_questions else 0.0,
        "questions": qs_out,
    }


def run_worker(gpu_id: int, model_path: str, jobs: List[Dict],
               batch_size: int, max_new_tokens: int) -> List[Dict]:
    """Per-GPU worker. Loads model, processes a slice of jobs, returns details.

    Each job: {sample_id, sub_task, generated_image_path, quality_questions}.
    Implementation mirrors evaluate_ic.py's worker structure.
    """
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    details: List[Dict] = []
    for job in jobs:
        img_path = job["generated_image_path"]
        if not os.path.exists(img_path):
            details.append({"_error": "image_missing", "sample_id": job["sample_id"]})
            continue
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            details.append({"_error": "image_missing", "sample_id": job["sample_id"]})
            continue

        qq = job["quality_questions"]
        preds: List[str] = []
        for i in range(0, len(qq), batch_size):
            batch = qq[i:i + batch_size]
            batch_messages = [
                build_judge_messages(img_path, q["text"]) for q in batch
            ]
            # Construct batched inputs (Qwen3-VL chat template)
            texts = [
                processor.apply_chat_template(m, add_generation_prompt=True)
                for m in batch_messages
            ]
            images = [[image]] * len(batch_messages)
            inputs = processor(
                text=texts, images=images, return_tensors="pt", padding=True
            ).to("cuda:0")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            input_lens = inputs["input_ids"].shape[1]
            decoded = processor.batch_decode(
                out[:, input_lens:], skip_special_tokens=True
            )
            preds.extend(decoded)

        details.append(score_sample(
            sample_id=job["sample_id"],
            sub_task=job["sub_task"],
            generated_image=img_path,
            quality_questions=qq,
            predictions=preds,
        ))
    return details


def prepare_jobs(rows: List[Dict],
                 gt_index: Dict[Tuple[str, str], List[Dict]]
                 ) -> Tuple[List[Dict], Dict[str, int]]:
    """Pair CSV rows with GT; classify rows that have no GT."""
    jobs: List[Dict] = []
    errors = {
        "skipped_no_generation": 0,
        "image_missing": 0,
        "gt_missing": 0,
        "timeout": 0,
    }
    for row in rows:
        key = (row["sub_task"], row["image_path_rel"])
        qq = gt_index.get(key)
        if qq is None:
            errors["gt_missing"] += 1
            continue
        jobs.append({
            "sample_id": f"{row['sub_task']}/{row['image_path_rel']}",
            "sub_task": row["sub_task"],
            "generated_image_path": row["generated_image_path"],
            "quality_questions": qq,
        })
    return jobs, errors


def write_results(path: str, summary: Dict, details: List[Dict]) -> None:
    out_path = Path(path)
    out_path.write_text(
        json.dumps({"summary": summary, "details": details},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def split_jobs(jobs: List[Dict], n: int) -> List[List[Dict]]:
    """Round-robin split into n shards (more even than contiguous slices
    when sample sizes vary)."""
    shards: List[List[Dict]] = [[] for _ in range(n)]
    for i, j in enumerate(jobs):
        shards[i % n].append(j)
    return shards


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate GGU generated-image quality")
    p.add_argument("--model-path", required=True)
    p.add_argument("--csv-path", required=True)
    p.add_argument("--data-dir", default="../data/Gen_Guided_Und")
    p.add_argument("--output", default="evaluate_ggu_quality_results.json")
    p.add_argument("--subtasks", default="2D_Spatial,3D_Spatial",
                   help="Comma-separated subtask names (matches data dir names)")
    p.add_argument("--gpu-ids", default=None,
                   help="Comma-separated CUDA IDs, e.g. 0,1,2,3")
    p.add_argument("--num-gpus", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--model-name", default=None,
                   help="Override model name in output (defaults to CSV parent dir)")
    return p.parse_args()


def _infer_model_name(csv_path: str, override: Optional[str]) -> str:
    if override:
        return override
    # CSV path .../result/<Model>/GGU/GGU_<Model>_results.csv
    parts = Path(csv_path).resolve().parts
    if "result" in parts:
        i = parts.index("result")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "Unknown"


def main():
    args = parse_args()

    subtasks = [s.strip() for s in args.subtasks.split(",") if s.strip()]
    if args.gpu_ids:
        gpu_ids = [int(g) for g in args.gpu_ids.split(",") if g.strip()]
    else:
        gpu_ids = list(range(args.num_gpus))
    num_gpus = len(gpu_ids)

    print(f"[setup] subtasks={subtasks}  gpu_ids={gpu_ids}")
    print(f"[setup] judge model: {args.model_path}")

    rows = load_generation_rows(args.csv_path)
    # Filter rows to enabled subtasks (in case --subtasks excluded some)
    rows = [r for r in rows if r["sub_task"] in subtasks]
    print(f"[load] {len(rows)} generation rows from CSV")

    gt_index = build_gt_index(args.data_dir, subtasks)
    print(f"[load] {len(gt_index)} GT entries from JSON files")

    jobs, errors = prepare_jobs(rows, gt_index)
    print(f"[plan] {len(jobs)} jobs ready, errors={errors}")

    shards = split_jobs(jobs, num_gpus)

    # Dispatch
    import torch.multiprocessing as torch_mp
    torch_mp.set_start_method("spawn", force=True)
    from concurrent.futures import ProcessPoolExecutor

    details: List[Dict] = []
    with ProcessPoolExecutor(
        max_workers=num_gpus, mp_context=torch_mp.get_context("spawn"),
    ) as pool:
        futures = []
        for gpu_id, shard in zip(gpu_ids, shards):
            futures.append(pool.submit(
                run_worker, gpu_id, args.model_path, shard,
                args.batch_size, args.max_new_tokens,
            ))
        for f in futures:
            for d in f.result():
                if d.get("_error") == "image_missing":
                    errors["image_missing"] += 1
                    continue
                details.append(d)

    model_name = _infer_model_name(args.csv_path, args.model_name)
    summary = aggregate_summary(
        details, errors, model_name=model_name, judge_model=args.model_path,
    )
    write_results(args.output, summary, details)
    print(f"[done] summary written to {args.output}")
    print(f"[done] overall_accuracy={summary['overall_accuracy']:.4f}")


if __name__ == "__main__":
    main()
