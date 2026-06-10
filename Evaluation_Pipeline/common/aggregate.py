"""Score aggregation helpers."""

from statistics import mean as _mean
from typing import List, Optional


def clip(score: float) -> float:
    return min(max(float(score), 0.0), 1.0)


def safe_mean(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return clip(_mean(valid))


def null_task_result(task_id: str) -> dict:
    return {
        "task": task_id,
        "scores": {
            "understanding_score": None,
            "generation_score": None,
            "unified_score": None,
        },
        "stats": {
            "num_total": 0,
            "num_scored": 0,
            "num_failed": 0,
            "failure_reasons": {},
        },
        "details": [],
    }


def build_task_result(
    task_id: str,
    details: list,
    num_total: int,
    failure_reasons: dict,
) -> dict:
    num_scored = len(details)
    num_failed = num_total - num_scored

    und_scores = [d["understanding_item"] for d in details]
    gen_scores = [d["generation_item"] for d in details]
    uni_scores = [d["unified_item"] for d in details]

    return {
        "task": task_id,
        "scores": {
            "understanding_score": safe_mean(und_scores),
            "generation_score": safe_mean(gen_scores),
            "unified_score": safe_mean(uni_scores),
        },
        "stats": {
            "num_total": num_total,
            "num_scored": num_scored,
            "num_failed": num_failed,
            "failure_reasons": failure_reasons,
        },
        "details": details,
    }
