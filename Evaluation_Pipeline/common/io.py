"""
I/O utilities: load inference CSVs and ground-truth data files.

Image path resolution
---------------------
Inference CSVs store paths relative to Inference_Pipeline/.
Call resolve_path(path, inference_base_dir) to get an absolute path.
"""

import json
import os
from typing import Dict, List, Optional

import pandas as pd


def resolve_path(path: str, base_dir: Optional[str]) -> str:
    """
    Resolve a possibly-relative path against base_dir.
    Returns absolute path; existence is NOT checked here.
    """
    if not path or str(path).lower() in ("nan", "none", ""):
        return ""
    path = str(path).strip()
    if os.path.isabs(path):
        return path
    if base_dir:
        return os.path.normpath(os.path.join(base_dir, path))
    return path


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load an inference CSV and return the full DataFrame."""
    df = pd.read_csv(csv_path)
    # Normalise string-None values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].where(df[col].notna(), None)
    return df


def success_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rows where status == 'success'."""
    return df[df["status"] == "success"].copy()


# ---------------------------------------------------------------------------
# Ground-truth loaders
# ---------------------------------------------------------------------------

def load_ic_gt(data_dir: str) -> Dict[int, dict]:
    """
    Load IC ground-truth data.
    Returns {0-based_index: {prompt, questions, image_path}}
    """
    prompts_file = os.path.join(data_dir, "prompts.txt")
    questions_file = os.path.join(data_dir, "questions.json")
    images_dir = os.path.join(data_dir, "images")

    with open(prompts_file, encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    with open(questions_file, encoding="utf-8") as f:
        qs_data = json.load(f)
    questions_map = qs_data.get("questions", {})

    result = {}
    for i, prompt in enumerate(prompts):
        key = f"prompt_{i}"
        qs = questions_map.get(key, {})
        # Try .png then .jpg
        img_path = ""
        for ext in ("png", "jpg", "jpeg"):
            candidate = os.path.join(images_dir, f"{i + 1}.{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break
        result[i] = {"prompt": prompt, "questions": qs, "image_path": img_path}
    return result


def load_ugg_gt(data_dir: str) -> Dict[int, dict]:
    """
    Load UGG ground-truth data from UGG.csv.
    Returns {0-based_index: row_dict}
    """
    csv_file = os.path.join(data_dir, "UGG.csv")
    df = pd.read_csv(csv_file)
    result = {}
    for i, row in df.iterrows():
        img_rel = str(row.get("image_path", ""))
        img_path = os.path.join(data_dir, img_rel) if img_rel else ""
        result[i] = {
            "image_path": img_path,
            "instruction": str(row.get("instruction", "")),
            "operation": str(row.get("operation", "")),
            "bbox": str(row.get("bbox", "")),
            "mask": str(row.get("mask", "")),
            "object": str(row.get("object", "")),
        }
    return result


def load_ggu_gt(data_dir: str) -> Dict[int, dict]:
    """
    Load GGU ground-truth data from all three subtask files.
    Returns {0-based_global_index: row_dict} with 'category' field.
    quality_questions is a list of {text, answer} for 2d/3d spatial.
    image_generation_validate is a dict {id: question} for complex_relation.
    """
    sub_tasks = [
        ("2D_Spatial", "2d_spatial.json"),
        ("3D_Spatial", "spatial.json"),
        ("Complex_Relation", "complex_relation.json"),
    ]
    result = {}
    global_idx = 0

    for category_dir, filename in sub_tasks:
        json_file = os.path.join(data_dir, category_dir, filename)
        if not os.path.exists(json_file):
            continue
        with open(json_file, encoding="utf-8") as f:
            items = json.load(f)
        category = category_dir.lower()

        for item in items:
            img_rel = item.get("image_path", "")
            if img_rel:
                img_path = os.path.join(data_dir, category_dir, img_rel)
                if not os.path.exists(img_path):
                    global_idx += 1
                    continue
            else:
                img_path = ""

            # Build validation questions list for this category
            validate_questions: List[dict] = []
            if "quality_questions" in item:
                for q in item["quality_questions"]:
                    validate_questions.append({
                        "text": q.get("text", ""),
                        "answer": q.get("answer", "yes"),
                    })
            elif "image_generation_validate" in item:
                igv = item["image_generation_validate"]
                if isinstance(igv, dict):
                    for k, v in igv.items():
                        validate_questions.append({"text": v, "answer": "yes"})

            result[global_idx] = {
                "image_path": img_path,
                "category": category,
                "question": item.get("question", ""),
                "options": item.get("options", {}),
                "answer": item.get("answer", ""),
                "description": item.get("description", ""),
                "validate_questions": validate_questions,
            }
            global_idx += 1

    return result


def load_me_gt(data_dir: str) -> Dict[int, dict]:
    """
    Load ME ground-truth from ME.csv.
    Returns {0-based_index: row_dict}
    """
    csv_file = os.path.join(data_dir, "ME.csv")
    df = pd.read_csv(csv_file)
    result = {}
    for i, row in df.iterrows():
        img_rel = str(row.get("image_path", ""))
        img_path = os.path.join(data_dir, img_rel)
        if not os.path.exists(img_path):
            img_path = os.path.join(data_dir, "images", img_rel)
        result[i] = {
            "image_path": img_path if os.path.exists(img_path) else "",
            "operation": str(row.get("operation", "")),
            "instruction": str(row.get("instruction", "")),
            "caption": str(row.get("caption", "")),
            "final_caption": str(row.get("final_caption", "")),
        }
    return result
