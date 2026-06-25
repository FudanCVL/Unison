#!/usr/bin/env python3
"""
Generic inference code - supports multi-model, multi-task inference
"""
from __future__ import annotations
import csv
import json
import os
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
import yaml
import argparse
import pandas as pd
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
import re
import inspect
from tqdm import tqdm
import torch
import random
import numpy as np
# AutoModelForImageTextToText / AutoProcessor are only used by the qwen3vl fallback path; lazy-loaded to avoid errors on older transformers versions
from PIL import Image
import multiprocessing
import math
import traceback
import copy
import time
import threading
from datetime import datetime, timedelta

# Customizable list of models
MODELS = [
    # "config/Qwen3-VL-30B-A3B-Instruct.json",
    "config/BAGEL-7B-MoT.json",
]


def load_config(config_path: str = "tasks.yaml") -> Dict[str, Any]:
    """Load the task configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model_config(config_path: str) -> Dict[str, Any]:
    """Load a model configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_ic_task_data(data_dir: str) -> List[Dict[str, Any]]:
    """Load IC task data"""
    data = []
    prompts_file = os.path.join(data_dir, "prompts.txt")
    questions_file = os.path.join(data_dir, "questions.json")
    images_dir = os.path.join(data_dir, "images")

    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines()]

    with open(questions_file, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)

    total_prompts = len(prompts)
    skipped_count = 0

    for idx, prompt in enumerate(prompts):
        prompt_key = f"prompt_{idx}"
        if prompt_key in questions_data.get("questions", {}):
            questions = questions_data["questions"][prompt_key]
            image_num = idx + 1
            png_path = os.path.join(images_dir, f"{image_num}.png")
            jpg_path = os.path.join(images_dir, f"{image_num}.jpg")

            if os.path.exists(png_path):
                image_path = png_path
            elif os.path.exists(jpg_path):
                image_path = jpg_path
            else:
                image_path = None

            if image_path:
                data.append({
                    "index": idx,
                    "image_path": image_path,
                    "prompt": prompt,
                    "questions": questions,
                })
            else:
                skipped_count += 1
        else:
            skipped_count += 1

    if skipped_count > 0:
        print(f"IC task data loading: {len(data)}/{total_prompts} items loaded successfully, {skipped_count} skipped")
    else:
        print(f"IC task data loading: {len(data)}/{total_prompts} items loaded successfully")

    return data


def load_ugg_task_data(data_dir: str) -> List[Dict[str, Any]]:
    """Load UGG task data"""
    csv_file = os.path.join(data_dir, "UGG.csv")
    df = pd.read_csv(csv_file)
    total_rows = len(df)
    data = []
    skipped_count = 0

    for idx, row in df.iterrows():
        image_path = os.path.join(data_dir, str(row["image_path"]))
        if os.path.exists(image_path):
            data.append({
                "index": idx,
                "image_path": image_path,
                "object": str(row.get("object", "")),
                "instruction": str(row.get("instruction", "")),
                "operation": str(row.get("operation", "")),
                "bbox": str(row.get("bbox", "")),
                "mask": str(row.get("mask", "")),
            })
        else:
            skipped_count += 1

    if skipped_count > 0:
        print(f"UGG task data loading: {len(data)}/{total_rows} items loaded successfully, {skipped_count} skipped (image not found)")
    else:
        print(f"UGG task data loading: {len(data)}/{total_rows} items loaded successfully")

    return data


def load_ggu_task_data(data_dir: str) -> List[Dict[str, Any]]:
    """Load GGU task data (merges the three sub-tasks: 2D_Spatial / 3D_Spatial / Complex_Relation)"""
    sub_tasks = [
        ("2D_Spatial",       "2d_spatial.json"),
        ("3D_Spatial",       "spatial.json"),
        ("Complex_Relation", "complex_relation.json"),
    ]
    data = []
    skipped_count = 0
    total_loaded = 0

    for category_dir, filename in sub_tasks:
        json_file = os.path.join(data_dir, category_dir, filename)
        if not os.path.exists(json_file):
            print(f"GGU: {json_file} not found, skipping")
            continue

        with open(json_file, 'r', encoding='utf-8') as f:
            items = json.load(f)

        category = category_dir.lower()  # 2d_spatial / 3d_spatial / complex_relation
        total_loaded += len(items)

        for item in items:
            image_path_raw = item.get("image_path", "")
            if image_path_raw:
                # Image paths are relative to their respective sub-directories (2D_Spatial/, 3D_Spatial/)
                image_path = os.path.join(data_dir, category_dir, image_path_raw)
                if not os.path.exists(image_path):
                    skipped_count += 1
                    continue
            else:
                # Complex_Relation samples may have no source image (an intermediate image is generated from the description)
                image_path = ""

            data.append({
                "index": len(data),
                "image_path": image_path,
                "category": category,
                "question": item.get("question", ""),
                "options": item.get("options", {}),
                "answer": item.get("answer", ""),
                "description": item.get("description", ""),
                "image_generation_validate": item.get("image_generation_validate", {}),
            })

    if skipped_count > 0:
        print(f"GGU task data loading: {len(data)}/{total_loaded} items loaded successfully, {skipped_count} skipped (image not found)")
    else:
        print(f"GGU task data loading: {len(data)} items loaded successfully")

    return data


def load_me_task_data(data_dir: str) -> List[Dict[str, Any]]:
    """Load ME task data"""
    csv_file = os.path.join(data_dir, "ME.csv")
    df = pd.read_csv(csv_file)
    total_rows = len(df)
    data = []
    skipped_count = 0
    skipped_paths = []

    for idx, row in df.iterrows():
        rel_path = str(row["image_path"])
        image_path = os.path.join(data_dir, rel_path)
        if not os.path.exists(image_path):
            # The image may be stored under the images/ sub-directory
            image_path = os.path.join(data_dir, "images", rel_path)
        if os.path.exists(image_path):
            data.append({
                "index": idx,
                "image_path": image_path,
                "operation": str(row.get("operation", "")),
                "instruction": str(row.get("instruction", "")),
                "caption": str(row.get("caption", "")),
                "final_caption": str(row.get("final_caption", "")),
            })
        else:
            skipped_count += 1
            if len(skipped_paths) < 5:
                skipped_paths.append(image_path)

    if skipped_count > 0:
        print(f"ME task data loading: {len(data)}/{total_rows} items loaded successfully, {skipped_count} skipped (image not found)")
        if skipped_paths:
            print(f"  Example skipped paths (first {len(skipped_paths)}):")
            for path in skipped_paths:
                print(f"    - {path}")
        if skipped_count > total_rows * 0.5:
            print(f"  Warning: More than 50% of data items were skipped. Please check image paths in ME.csv")
    else:
        print(f"ME task data loading: {len(data)}/{total_rows} items loaded successfully")

    return data


def load_task_data(task: Dict[str, Any], base_data_dir: str) -> List[Dict[str, Any]]:
    """Load the corresponding data based on the task ID"""
    task_id = task["task_id"]
    task_data_dir_rel = task.get("data_dir", "")
    if task_data_dir_rel.startswith("data/"):
        task_data_dir_rel = task_data_dir_rel[5:]
    task_data_dir = os.path.join(base_data_dir, task_data_dir_rel)

    if task_id == "IC":
        return load_ic_task_data(task_data_dir)
    elif task_id == "UGG":
        return load_ugg_task_data(task_data_dir)
    elif task_id == "GGU":
        return load_ggu_task_data(task_data_dir)
    elif task_id == "ME":
        return load_me_task_data(task_data_dir)
    else:
        raise ValueError(f"Unknown task_id: {task_id}")


def get_task_total_count(task: Dict[str, Any], base_data_dir: str) -> int:
    """Get the total number of data items for a task (without actually loading them)"""
    task_id = task["task_id"]
    task_data_dir_rel = task.get("data_dir", "")
    if task_data_dir_rel.startswith("data/"):
        task_data_dir_rel = task_data_dir_rel[5:]
    task_data_dir = os.path.join(base_data_dir, task_data_dir_rel)

    if task_id == "IC":
        prompts_file = os.path.join(task_data_dir, "prompts.txt")
        if os.path.exists(prompts_file):
            with open(prompts_file, 'r', encoding='utf-8') as f:
                return len([line for line in f.readlines() if line.strip()])
        return 0
    elif task_id == "UGG":
        csv_file = os.path.join(task_data_dir, "UGG.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            return len(df)
        return 0
    elif task_id == "GGU":
        # Sum the counts of the three sub-tasks
        sub_tasks = [
            ("2D_Spatial",       "2d_spatial.json"),
            ("3D_Spatial",       "spatial.json"),
            ("Complex_Relation", "complex_relation.json"),
        ]
        total = 0
        for category_dir, filename in sub_tasks:
            json_file = os.path.join(task_data_dir, category_dir, filename)
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    items = json.load(f)
                total += len(items)
        return total
    elif task_id == "ME":
        csv_file = os.path.join(task_data_dir, "ME.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            return len(df)
        return 0
    else:
        return 0


def validate_task_data(task: Dict[str, Any], base_data_dir: str,
                      skip_threshold: float = 0.5) -> Dict[str, Any]:
    """Validate that the task data loads correctly"""
    task_id = task["task_id"]
    result = {
        "valid": True,
        "task_id": task_id,
        "total_count": 0,
        "loaded_count": 0,
        "skipped_count": 0,
        "missing_images": [],
        "errors": []
    }

    try:
        total_count = get_task_total_count(task, base_data_dir)
        result["total_count"] = total_count

        if total_count == 0:
            result["valid"] = False
            result["errors"].append(f"No data found for task {task_id}")
            return result

        task_data = load_task_data(task, base_data_dir)
        loaded_count = len(task_data)
        result["loaded_count"] = loaded_count
        result["skipped_count"] = total_count - loaded_count

        if loaded_count == 0:
            result["valid"] = False
            result["errors"].append(f"No data items loaded for task {task_id}")

        inference_operations = task.get("inference_operations", [])
        has_generation = any(
            op.get("operation_type") == "generation"
            for op in inference_operations
        )

        if not has_generation:
            missing_count = 0
            for data_item in task_data:
                image_path = data_item.get("image_path", "")
                if image_path and not os.path.exists(image_path):
                    missing_count += 1
                    if len(result["missing_images"]) < 10:
                        result["missing_images"].append(image_path)

            if missing_count > 0:
                result["errors"].append(
                    f"{missing_count} loaded data items have missing image files"
                )

        if total_count > 0:
            skip_ratio = result["skipped_count"] / total_count
            if skip_ratio > skip_threshold:
                result["valid"] = False
                result["errors"].append(
                    f"{skip_ratio*100:.1f}% of data skipped, exceeds threshold {skip_threshold*100:.1f}%"
                )

    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Error loading task data: {str(e)}")

    return result


def validate_all_tasks_data(tasks: List[Dict[str, Any]],
                           base_data_dir: str,
                           skip_threshold: float = 0.5,
                           continue_on_error: bool = False,
                           logger: Optional['InferenceLogger'] = None) -> bool:
    """Validate data loading for all tasks"""
    all_passed = True

    msg = "\nValidating all tasks data..."
    if logger:
        logger.info(msg)
    else:
        print(msg)

    for task in tasks:
        task_id = task["task_id"]
        msg = f"\nValidating task: {task_id}"
        if logger:
            logger.info(msg)
        else:
            print(msg)

        validation_result = validate_task_data(
            task, base_data_dir, skip_threshold
        )

        loaded_count = validation_result["loaded_count"]
        total_count = validation_result["total_count"]
        skipped_count = validation_result["skipped_count"]

        if validation_result["valid"]:
            status_symbol = "✓"
            status_text = "PASSED"
            if skipped_count > 0:
                msg = f"  {status_symbol} {task_id}: {loaded_count}/{total_count} items loaded successfully, {skipped_count} skipped"
            else:
                msg = f"  {status_symbol} {task_id}: {loaded_count}/{total_count} items loaded successfully"
            if logger:
                logger.info(msg)
            else:
                print(msg)
            msg = f"  Status: {status_text}"
            if logger:
                logger.info(msg)
            else:
                print(msg)
        else:
            status_symbol = "✗"
            status_text = "FAILED"
            all_passed = False
            msg = f"  {status_symbol} {task_id}: {loaded_count}/{total_count} items loaded successfully, {skipped_count} skipped"
            if logger:
                logger.info(msg)
            else:
                print(msg)

            if validation_result["missing_images"]:
                msg = f"  Missing images (first {len(validation_result['missing_images'])}):"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
                for img_path in validation_result["missing_images"]:
                    msg = f"    - {img_path}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)

            if validation_result["errors"]:
                for error in validation_result["errors"]:
                    msg = f"  Error: {error}"
                    if logger:
                        logger.error(msg)
                    else:
                        print(msg)

            if total_count > 0:
                skip_ratio = skipped_count / total_count
                if skip_ratio > skip_threshold:
                    msg = f"  Status: {status_text} ({skip_ratio*100:.1f}% skipped, exceeds threshold {skip_threshold*100:.1f}%)"
                else:
                    msg = f"  Status: {status_text}"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)

    return all_passed


class ModelInference:
    """Model inference class (for generic transformers models such as Qwen3-VL)"""

    def __init__(self, model_config: Dict[str, Any]):
        self.config = model_config
        self.model = None
        self.processor = None
        self.gpu_id = model_config.get("gpu_id", 0)

        seed = model_config.get("seed", 666)
        if seed > 0:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def load_model(self):
        """Load the model (only used by the generic transformers path)"""
        model_path = self.config.get("model_path")
        if not model_path:
            raise ValueError("model_path not found in config")

        print(f"Loading model {self.config.get('model_name')} on GPU {self.gpu_id}")
        torch.cuda.set_device(self.gpu_id)

        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype="auto",
            device_map=f"cuda:{self.gpu_id}"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def inference(self, image_path: Union[str, List[str]], prompt: str, index: Optional[int] = None,
                  operation_type: Optional[str] = None) -> str:
        """Run generic transformers inference (understanding models such as Qwen3-VL)"""
        if operation_type:
            self.config["inference_mode"] = operation_type

        if isinstance(image_path, list):
            single_image_path = image_path[0] if image_path else None
        else:
            single_image_path = image_path

        if self.model is None:
            self.load_model()

        messages_content = []

        if single_image_path and os.path.exists(single_image_path):
            messages_content.append({"type": "image", "image": single_image_path})

        prompt_template = self.config.get("prompt_template", "{description}")
        if "{description}" in prompt_template:
            full_text = prompt_template.format(description=prompt)
        else:
            full_text = prompt
        messages_content.append({"type": "text", "text": full_text})

        messages = [{"role": "user", "content": messages_content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        max_new_tokens = self.config.get("max_new_tokens", 512)
        temperature = self.config.get("temperature", 0.6)
        do_sample = self.config.get("do_sample", False)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return generated_text[0].strip()


# ---------------------------------------------------------------------------
# Model dispatch map: api_type -> (module_name, function_name)
# ---------------------------------------------------------------------------
_DISPATCH_MAP: Dict[str, tuple] = {
    "bagel":     ("model_inference.bagel_inference",    "bagel_inference_function"),
    "janus":     ("model_inference.janus_inference",    "janus_inference_function"),
    "seed_x":    ("model_inference.seed_x_inference",   "seed_x_inference_function"),
    "tokenflow": ("model_inference.tokenflow_inference","tokenflow_inference_function"),
    "showo":     ("model_inference.showo_inference",    "showo_inference_function"),
    "showo2":    ("model_inference.showo2_inference",   "showo2_inference_function"),
    "uniworld":  ("model_inference.uniworld_inference", "uniworld_inference_function"),
    "omnigen2":  ("model_inference.omnigen2_inference", "omnigen2_inference_function"),
    "illume":    ("model_inference.illume_inference",   "illume_inference_function"),
    "ddit":      ("model_inference.ddit_inference",     "ddit_inference_function"),
}


def dispatch_inference(
    api_type: str,
    image_paths: Optional[Union[str, List[str]]],
    prompt: str,
    model_config: Dict[str, Any],
    inference_obj: ModelInference,
    index: Optional[int] = None,
    round_number: Optional[int] = None,
    operation_type: Optional[str] = None,
) -> str:
    """
    Unified dispatch for all model inference calls.

    - Known api_type (present in _DISPATCH_MAP) -> dynamically import and call the corresponding adapter.
    - Unknown api_type -> fall back to generic transformers inference (Qwen3-VL, etc.).
    """
    if api_type in _DISPATCH_MAP:
        module_name, func_name = _DISPATCH_MAP[api_type]
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            # Fall back to a relative import (when running directly from Inference_Pipeline/)
            short_name = module_name.split(".")[-1]
            module = importlib.import_module(short_name)
        func = getattr(module, func_name)
        return func(image_paths, prompt, model_config, index=index, round_number=round_number)
    else:
        # Default: generic transformers (Qwen3-VL, etc.)
        if inference_obj.model is None:
            inference_obj.load_model()
        return inference_obj.inference(image_paths, prompt, index=index, operation_type=operation_type)


def load_template_function(template_path: str) -> Callable[[Dict[str, Any]], str]:
    """Load a template function from a module path"""
    if '.' not in template_path:
        raise ValueError(f"Invalid template path: {template_path}. Expected format: 'module.function_name'")

    module_name, function_name = template_path.rsplit('.', 1)

    try:
        module = importlib.import_module(module_name)
        template_func = getattr(module, function_name)

        if not callable(template_func):
            raise ValueError(f"{template_path} is not callable")

        return template_func
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_name}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Function '{function_name}' not found in module '{module_name}': {e}")


def get_data_field_value(data_item: Dict[str, Any], field_name: str) -> Optional[str]:
    """Get a given field's value from a data item and format it"""
    if field_name not in data_item:
        return None

    value = data_item[field_name]

    if isinstance(value, dict):
        if field_name == "questions":
            return "\n".join([f"{k}. {v}" for k, v in value.items()])
        elif field_name == "options":
            return "\n".join([f"{k}: {v}" for k, v in value.items()])
        else:
            return json.dumps(value, ensure_ascii=False)
    elif isinstance(value, (list, tuple)):
        return "\n".join([str(item) for item in value])
    else:
        return str(value) if value is not None else ""


def build_data_format_dict(data_item: Dict[str, Any]) -> Dict[str, str]:
    """Build a format dictionary from a data item, used for placeholder substitution"""
    format_dict = {}
    for key in data_item.keys():
        value = get_data_field_value(data_item, key)
        if value is not None:
            format_dict[key] = value
    return format_dict


def format_prompt_template(template: str, data_item: Dict[str, Any]) -> str:
    """Format a prompt_template, supporting all fields in data_item"""
    format_dict = build_data_format_dict(data_item)

    try:
        return template.format_map(format_dict)
    except KeyError as e:
        raise ValueError(f"Missing field in data_item for template '{template}': {e}")


def is_dialogue_template(result: Any) -> bool:
    """Detect whether the result is in dialogue-template format"""
    if isinstance(result, list) and len(result) > 0:
        first_item = result[0]
        if isinstance(first_item, dict) and "round_number" in first_item:
            return True
    return False


def extract_combined_instructions(output: str) -> str:
    """Extract multiple instructions from an understanding-round output and join them"""
    pattern = r'\d+\.\[[^\]]+\]:\[([^\]]+)\]'
    matches = re.findall(pattern, output)
    if matches:
        return ', '.join(matches)
    if output.lower().startswith('no'):
        rest = output[2:].strip()
        while rest and rest[0] in ".,:;":
            rest = rest[1:].strip()
        if rest:
            return rest
    return output


def parse_evaluation_output(output: str) -> tuple[bool, Optional[str], bool]:
    """Parse an evaluation-round output, decide whether to stop the dialogue, and extract the next round's instruction"""
    output = output.strip()
    output_lower = output.lower()

    yes_pattern = r'\byes\b'
    no_pattern = r'\bno\b'

    yes_match = re.search(yes_pattern, output_lower)
    no_match = re.search(no_pattern, output_lower)

    if yes_match and no_match:
        if yes_match.start() < no_match.start():
            return True, None, True
        else:
            extracted_instructions = extract_combined_instructions(output)
            if extracted_instructions != output and extracted_instructions:
                return False, extracted_instructions, True

            no_start = no_match.start()
            rest = output[no_start + 2:].strip()
            while rest and rest[0] in ".,:;":
                rest = rest[1:].strip()
            return False, rest if rest else None, True
    elif yes_match:
        return True, None, True
    elif no_match:
        extracted_instructions = extract_combined_instructions(output)
        if extracted_instructions != output and extracted_instructions:
            return False, extracted_instructions, True

        no_start = no_match.start()
        rest = output[no_start + 2:].strip()
        while rest and rest[0] in ".,:;":
            rest = rest[1:].strip()
        return False, rest if rest else None, True
    else:
        return False, None, False


def escape_braces(text: str) -> str:
    """Escape curly braces in the text"""
    return text.replace("{", "{{").replace("}", "}}")


def replace_dialogue_placeholders(
    template: str,
    dialogue_history: Dict[int, str],
    data_item: Dict[str, Any]
) -> tuple[str, List[str]]:
    """Replace placeholders in a dialogue template"""
    result = template
    extracted_image_paths = []

    if "{original_image}" in result:
        original_image_path = data_item.get("image_path", "")
        if original_image_path and os.path.exists(original_image_path):
            extracted_image_paths.append(original_image_path)
            result = result.replace("{original_image}", "")
        else:
            print(f"Warning: {{original_image}} placeholder found but original image path is not valid. "
                  f"Path: {original_image_path[:100] if original_image_path else 'None'}...")
            result = result.replace("{original_image}", "")

    for match in re.finditer(r"\{round_(\d+)_image\}", result):
        round_num = int(match.group(1))
        if round_num in dialogue_history:
            potential_image_path = dialogue_history[round_num]
            if potential_image_path and os.path.exists(potential_image_path):
                extracted_image_paths.append(potential_image_path)
                result = result.replace(match.group(0), "")
            else:
                print(f"Warning: {match.group(0)} placeholder found but round {round_num} output is not a valid image path. "
                      f"Output: {potential_image_path[:100] if potential_image_path else 'None'}...")
                result = result.replace(match.group(0), "")
        else:
            print(f"Warning: {match.group(0)} placeholder found but round {round_num} output not available in dialogue history. "
                  f"Available rounds: {sorted(dialogue_history.keys()) if dialogue_history else 'none'}")
            result = result.replace(match.group(0), "")

    if "{previous_output}" in result:
        if dialogue_history:
            last_round = max(dialogue_history.keys())
            previous_output = escape_braces(dialogue_history[last_round])
            result = result.replace("{previous_output}", previous_output)
        else:
            print("Warning: {previous_output} placeholder found but dialogue history is empty")
            result = result.replace("{previous_output}", "")

    for match in re.finditer(r"\{round_(\d+)_output\}", result):
        round_num = int(match.group(1))
        if round_num in dialogue_history:
            escaped_output = escape_braces(dialogue_history[round_num])
            result = result.replace(match.group(0), escaped_output)
        else:
            print(f"Warning: {match.group(0)} placeholder found but round {round_num} output not available in dialogue history. "
                  f"Available rounds: {sorted(dialogue_history.keys()) if dialogue_history else 'none'}")
            result = result.replace(match.group(0), "")

    if "{next_instruction}" in result:
        if "next_instruction" in dialogue_history:
            escaped_instruction = escape_braces(dialogue_history["next_instruction"])
            result = result.replace("{next_instruction}", escaped_instruction)
        else:
            if dialogue_history:
                int_rounds = [k for k in dialogue_history.keys() if isinstance(k, int)]
                if int_rounds:
                    last_round = max(int_rounds)
                    fallback_instruction = escape_braces(dialogue_history[last_round])
                    result = result.replace("{next_instruction}", fallback_instruction)
                else:
                    print("Warning: {next_instruction} placeholder found but no round outputs in dialogue history")
                    result = result.replace("{next_instruction}", "")
            else:
                print("Warning: {next_instruction} placeholder found but no next_instruction or dialogue history available")
                result = result.replace("{next_instruction}", "")

    if "{combined_instructions}" in result:
        int_rounds = [k for k in dialogue_history.keys() if isinstance(k, int)]
        if int_rounds:
            odd_rounds = [r for r in int_rounds if r % 2 == 1]
            if odd_rounds:
                understanding_round = max(odd_rounds)
                understanding_output = dialogue_history[understanding_round]
                combined_instructions = extract_combined_instructions(understanding_output)
                escaped_instructions = escape_braces(combined_instructions)
                result = result.replace("{combined_instructions}", escaped_instructions)
            else:
                print("Warning: {combined_instructions} placeholder found but no understanding round (odd round) in dialogue history")
                result = result.replace("{combined_instructions}", "")
        else:
            print("Warning: {combined_instructions} placeholder found but no round outputs in dialogue history")
            result = result.replace("{combined_instructions}", "")

    format_dict = build_data_format_dict(data_item)

    for key in format_dict:
        format_dict[key] = escape_braces(format_dict[key])

    placeholder_pattern = r'\{([^{}]+)\}'

    def replace_placeholder(match):
        key = match.group(1)
        if key in format_dict:
            return str(format_dict[key])
        else:
            return match.group(0)

    result = re.sub(placeholder_pattern, replace_placeholder, result)

    return result.strip(), extracted_image_paths


def build_prompt_for_operation(prompt_template: Optional[str], task_id: str, data_item: Dict[str, Any], **extra_kwargs) -> Union[str, List[str], List[Dict[str, Any]]]:
    """Build a prompt based on prompt_template or the task type"""
    if prompt_template:
        if '.' in prompt_template:
            try:
                template_func = load_template_function(prompt_template)
                sig = inspect.signature(template_func)
                valid_kwargs = {k: v for k, v in extra_kwargs.items() if k in sig.parameters}
                result = template_func(data_item, **valid_kwargs)
                return result
            except (ValueError, ImportError, AttributeError) as e:
                raise ValueError(f"Failed to load template function '{prompt_template}': {e}")
        else:
            return format_prompt_template(prompt_template, data_item)

    if task_id == "IC":
        prompt = data_item["prompt"]
        questions = data_item["questions"]
        questions_text = "\n".join([f"{k}. {v}" for k, v in questions.items()])
        return f"{prompt}\n\nQuestions:\n{questions_text}"
    elif task_id == "UGG":
        return data_item.get("instruction", "")
    elif task_id == "GGU":
        question = data_item.get("question", "")
        options = data_item.get("options", {})
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        return f"{question}\n\nOptions:\n{options_text}"
    elif task_id == "ME":
        return data_item.get("instruction", "")

    return ""


def build_prompt_for_task(task_id: str, data_item: Dict[str, Any]) -> str:
    """Build a prompt based on the task type (backward-compatibility function)"""
    return build_prompt_for_operation(None, task_id, data_item)


class InferenceLogger:
    """Inference logging system; writes to both a log file and stdout"""

    def __init__(self, log_file_path: str, append: bool = True):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        self.log_file_path = log_file_path

        mode = 'a' if append else 'w'
        file_exists = os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0

        self.log_file = open(log_file_path, mode, encoding='utf-8')
        self.lock = threading.Lock()

        if append and file_exists:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                separator = f"\n{'='*80}\n"
                separator += f"NEW RUN STARTED AT: {timestamp}\n"
                separator += f"{'='*80}\n"
                self.log_file.write(separator)
                self.log_file.flush()
            except Exception:
                pass

    def get_log_file_path(self) -> str:
        return self.log_file_path

    def info(self, msg: str):
        with self.lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = f"[{timestamp}] {msg}\n"
            self.log_file.write(log_msg)
            self.log_file.flush()

    def progress(self, msg: str):
        with self.lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = f"[{timestamp}] {msg}\n"
            self.log_file.write(log_msg)
            self.log_file.flush()

    def error(self, msg: str):
        with self.lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = f"[{timestamp}] ERROR: {msg}\n"
            self.log_file.write(log_msg)
            self.log_file.flush()

    def close(self):
        with self.lock:
            if self.log_file:
                self.log_file.close()
                self.log_file = None


class ProgressTracker:
    """Progress tracking system; uses shared variables to track progress across processes"""

    def __init__(self, manager: multiprocessing.Manager, total_items: int):
        self.total_items = manager.Value('i', total_items)
        self.completed_items = manager.Value('i', 0)
        self.start_time = manager.Value('d', time.time())
        self.lock = manager.Lock()

    def update(self, increment: int = 1):
        with self.lock:
            self.completed_items.value += increment

    def get_progress(self) -> Dict[str, Any]:
        with self.lock:
            completed = self.completed_items.value
            total = self.total_items.value
            elapsed = time.time() - self.start_time.value

            if total == 0:
                percentage = 0.0
                remaining_time = None
            else:
                percentage = (completed / total) * 100.0
                if completed > 0:
                    remaining_time = (elapsed / completed) * (total - completed)
                else:
                    remaining_time = None

            return {
                "percentage": percentage,
                "completed": completed,
                "total": total,
                "elapsed_time": elapsed,
                "remaining_time": remaining_time
            }

    def get_elapsed_time(self) -> float:
        return time.time() - self.start_time.value


def split_data_for_gpus(data: List[Dict[str, Any]], num_gpus: int) -> List[List[Dict[str, Any]]]:
    """Evenly distribute a data list across N GPUs"""
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got {num_gpus}")

    if len(data) == 0:
        return [[] for _ in range(num_gpus)]

    chunk_size = math.ceil(len(data) / num_gpus)

    splits = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(data))
        splits.append(data[start_idx:end_idx])

    return splits


def run_inference_on_gpu(gpu_id: int, model_config: Dict[str, Any], task: Dict[str, Any],
                         args: argparse.Namespace, task_data_subset: List[Dict[str, Any]],
                         log_file_path: Optional[str] = None,
                         progress_tracker: Optional['ProgressTracker'] = None) -> None:
    """Run inference on a single GPU"""
    # By default each worker process inherits all CPU cores as the PyTorch/OpenMP thread-pool size.
    # When running multiple GPUs in parallel, the combined thread pools saturate the CPU; capping at 4 is enough.
    import os as _os
    _os.environ.setdefault("OMP_NUM_THREADS", "4")
    _os.environ.setdefault("MKL_NUM_THREADS", "4")
    import torch as _torch
    _torch.set_num_threads(4)

    logger = None
    if log_file_path:
        logger = InferenceLogger(log_file_path)

    me_dialogue_logger = None
    _csv_fh = None

    try:
        model_config["gpu_id"] = gpu_id

        task_id = task["task_id"]
        model_name = model_config.get("model_name", "unknown")
        safe_model_name = model_name.replace("/", "_").replace(" ", "_")

        if task_id == "ME":
            output_dir = os.path.join(args.result_dir, safe_model_name, task_id)
            me_dialogue_log_path = os.path.join(output_dir, f"ME_dialogue_log_gpu{gpu_id}.log")
            os.makedirs(output_dir, exist_ok=True)
            me_dialogue_logger = InferenceLogger(me_dialogue_log_path)

        log_msg = f"\n[GPU {gpu_id}] ========================================\n"
        log_msg += f"[GPU {gpu_id}] Starting inference for task {task_id}\n"
        log_msg += f"[GPU {gpu_id}] Data items: {len(task_data_subset)}\n"
        log_msg += f"[GPU {gpu_id}] ========================================"
        if logger:
            logger.info(log_msg)
        else:
            print(log_msg)

        inference = ModelInference(model_config)
        api_type = model_config.get("api_type", "qwen3vl")

        inference_operations = task.get("inference_operations", None)

        if inference_operations is None or len(inference_operations) == 0:
            inference_operations = [{
                "operation_type": "understanding",
                "prompt_template": None
            }]

        # Cross-operation output tracking: (operation_type, dialogue_index) -> model_response
        # Used so GGU unify can reference the image produced by the generation operation
        operation_outputs: Dict[tuple, str] = {}

        output_dir = os.path.join(args.result_dir, safe_model_name, task_id)
        temp_result_file = os.path.join(output_dir, f"{task_id}_{safe_model_name}_gpu{gpu_id}_results.csv")
        processed_indices = set()
        processed_operation_indices = set()

        if os.path.exists(temp_result_file):
            existing_df = pd.read_csv(temp_result_file)
            if len(existing_df) > 0:
                # Pre-populate operation_outputs so the unify operation can look up the prior generation output
                if "operation_type" in existing_df.columns and "dialogue_index" in existing_df.columns:
                    for _, row in existing_df.iterrows():
                        op_type = row.get("operation_type", "")
                        d_idx = row.get("dialogue_index")
                        response = row.get("model_response", "")
                        if op_type and pd.notna(d_idx) and pd.notna(response) and response != "":
                            operation_outputs[(str(op_type), int(d_idx))] = str(response)

                if "round_number" in existing_df.columns:
                    # Dialogue-template tasks: keyed by (dialogue_index, operation_index, round_number)
                    # operation_index must be included, otherwise the rounds of ME's two unify ops would collide
                    for _, row in existing_df.iterrows():
                        idx = row.get("dialogue_index")
                        op_index = row.get("operation_index", 0)
                        round_num = row.get("round_number", "")
                        if pd.notna(round_num) and round_num != "":
                            processed_operation_indices.add((idx, int(op_index) if pd.notna(op_index) else 0, str(round_num)))
                elif "operation_type" in existing_df.columns:
                    for _, row in existing_df.iterrows():
                        idx = row.get("dialogue_index")
                        op_type = row.get("operation_type", "understanding")
                        processed_operation_indices.add((idx, op_type))
                else:
                    dialogue_index_col = existing_df.get("dialogue_index", pd.Series(dtype=int))
                    if len(dialogue_index_col) > 0:
                        processed_indices = set(dialogue_index_col.tolist())
                if logger:
                    logger.info(f"[GPU {gpu_id}] Found existing temporary results, will skip processed items")
                else:
                    print(f"[GPU {gpu_id}] Found existing temporary results, will skip processed items")

        # Open the per-GPU CSV (append mode; each inference result is written immediately)
        _CSV_FIELDS = [
            "dialogue_index", "image_path", "input_data", "prompt_used",
            "model_response", "status", "task_id", "model_name", "task_name",
            "operation_type", "operation_index", "round_number",
            "question_id", "question_text", "output_suffix",
        ]
        os.makedirs(output_dir, exist_ok=True)
        _file_is_new = not (os.path.exists(temp_result_file) and os.path.getsize(temp_result_file) > 0)
        _csv_fh = open(temp_result_file, "a", newline="", encoding="utf-8")
        _csv_writer = csv.DictWriter(_csv_fh, fieldnames=_CSV_FIELDS, extrasaction="ignore", restval="")
        if _file_is_new:
            _csv_writer.writeheader()
            _csv_fh.flush()
        results_count = 0

        for op_idx, operation in enumerate(inference_operations):
            operation_type = operation.get("operation_type", "understanding")
            prompt_template = operation.get("prompt_template", None)
            output_suffix = operation.get("output_suffix", None)

            if logger:
                logger.info(f"[GPU {gpu_id}] Operation {op_idx + 1}/{len(inference_operations)}: {operation_type}")
            else:
                print(f"[GPU {gpu_id}] Operation {op_idx + 1}/{len(inference_operations)}: {operation_type}")

            for data_item in task_data_subset:
                idx = data_item["index"]
                image_path = data_item["image_path"]

                if operation_type != "generation" and (not image_path or not os.path.exists(image_path)):
                    # GGU Complex_Relation samples may have no source image; allow them through
                    if not (task_id == "GGU" and data_item.get("category") == "complex_relation"):
                        if logger:
                            logger.info(f"[GPU {gpu_id}] Image not found: {image_path}")
                        else:
                            print(f"[GPU {gpu_id}] Image not found: {image_path}")
                        continue

                # UGG unify: use the predicted bbox produced by the understanding operation, falling back to the GT bbox
                effective_data_item = data_item
                if task_id == "UGG" and operation_type == "unify":
                    predicted_bbox = operation_outputs.get(("understanding", idx), data_item.get("bbox", ""))
                    effective_data_item = dict(data_item)
                    effective_data_item["bbox"] = predicted_bbox

                prompts_result = build_prompt_for_operation(
                    prompt_template, task_id, effective_data_item,
                    max_iterations=task.get("max_iterations", 5)
                )

                if not is_dialogue_template(prompts_result):
                    if processed_operation_indices:
                        if (idx, operation_type) in processed_operation_indices:
                            continue
                    elif idx in processed_indices:
                        if op_idx == 0:
                            continue

                if is_dialogue_template(prompts_result):
                    # -------------------------------------------------------
                    # Dialogue-template handling
                    # -------------------------------------------------------
                    dialogue_history = {}
                    dialogue_rounds = prompts_result
                    should_stop_dialogue = False

                    for round_def in dialogue_rounds:
                        round_num = round_def["round_number"]

                        if processed_operation_indices:
                            round_key = (idx, op_idx, str(round_num))
                            if round_key in processed_operation_indices:
                                if os.path.exists(temp_result_file):
                                    existing_df = pd.read_csv(temp_result_file)
                                    matching_rows = existing_df[
                                        (existing_df["dialogue_index"] == idx) &
                                        (existing_df["operation_index"] == op_idx) &
                                        (existing_df["round_number"] == round_num)
                                    ]
                                    if len(matching_rows) > 0:
                                        last_row = matching_rows.iloc[-1]
                                        dialogue_history[round_num] = last_row.get("model_response", "")
                                continue

                        if should_stop_dialogue:
                            if logger:
                                logger.info(f"[GPU {gpu_id}] Stopping dialogue early due to 'Yes' response in evaluation round")
                            else:
                                print(f"[GPU {gpu_id}] Stopping dialogue early due to 'Yes' response in evaluation round")
                            break

                        round_num = round_def["round_number"]
                        prompt_template_str = round_def["prompt_template"]
                        round_operation_type = round_def.get("operation_type", operation_type)

                        prompt, extracted_image_paths = replace_dialogue_placeholders(
                            prompt_template_str, dialogue_history, data_item
                        )

                        if extracted_image_paths:
                            current_image_path = extracted_image_paths[0] if len(extracted_image_paths) == 1 else extracted_image_paths
                        elif round_operation_type != "generation":
                            current_image_path = image_path
                        else:
                            current_image_path = None

                        if me_dialogue_logger:
                            log_separator = "=" * 50
                            me_dialogue_logger.info(log_separator)
                            me_dialogue_logger.info(f"Dialogue Index: {idx}, Round: {round_num}")
                            if isinstance(current_image_path, list):
                                me_dialogue_logger.info(f"Image Paths: {', '.join(current_image_path)}")
                            else:
                                me_dialogue_logger.info(f"Image Path: {current_image_path if current_image_path else 'None'}")
                            me_dialogue_logger.info(f"Operation Type: {round_operation_type}")
                            me_dialogue_logger.info(f"Prompt: {prompt}")
                            me_dialogue_logger.info(log_separator)

                        try:
                            # Set the current inference mode
                            inference.config["task_id"] = task_id
                            inference.config["inference_mode"] = round_operation_type
                            model_response = dispatch_inference(
                                api_type, current_image_path, prompt, inference.config,
                                inference, index=idx, round_number=round_num,
                                operation_type=round_operation_type
                            )
                            status = "success"
                        except Exception as e:
                            error_msg = f"[GPU {gpu_id}] Error during inference (round {round_num}): {e}"
                            if logger:
                                logger.error(error_msg)
                            else:
                                print(error_msg)
                            model_response = ""
                            status = f"error: {str(e)}"

                        dialogue_history[round_num] = model_response

                        if task_id == "ME" and round_operation_type == "understanding":
                            should_stop, next_instruction, format_valid = parse_evaluation_output(model_response)
                            prompt_template_name = operation.get("prompt_template", "")
                            is_u2g = "u2g" in prompt_template_name.lower()
                            eval_tag = "u2g" if is_u2g else "g2u"

                            def _me_log(msg):
                                if me_dialogue_logger:
                                    me_dialogue_logger.info(msg)
                                if logger:
                                    logger.info(msg)

                            if should_stop:
                                # Case A: explicit "Yes" output — all edits are satisfied
                                should_stop_dialogue = True
                                _me_log(f"[GPU {gpu_id}] Round {round_num} ({eval_tag}): 'Yes' → stopping dialogue")

                            elif format_valid and next_instruction:
                                # Case B: normal "No, [instruction]"
                                if is_u2g:
                                    dialogue_history["next_instruction"] = next_instruction
                                _me_log(f"[GPU {gpu_id}] Round {round_num} ({eval_tag}): 'No' → instruction: {next_instruction[:100]}...")

                            elif format_valid and not next_instruction:
                                # Case C: "No" found but no extractable instruction follows it
                                if is_u2g:
                                    fallback = dialogue_history.get("next_instruction") or data_item.get("instruction", "")
                                    if fallback:
                                        dialogue_history["next_instruction"] = fallback
                                        _me_log(f"[GPU {gpu_id}] Round {round_num} ({eval_tag}): 'No' without instruction → fallback to previous: {fallback[:100]}...")
                                    else:
                                        should_stop_dialogue = True
                                        _me_log(f"[GPU {gpu_id}] Round {round_num} ({eval_tag}): 'No' without instruction, no fallback → stopping")
                                else:
                                    # g2u: "No" but no instruction -> stop per the documented termination condition
                                    should_stop_dialogue = True
                                    _me_log(f"[GPU {gpu_id}] Round {round_num} ({eval_tag}): 'No' without structured instructions → stopping")

                            elif not format_valid and model_response.strip():
                                # Case D: output has content but no yes/no — treat it as an implicit "No, [content]"
                                content = model_response.strip()
                                if is_u2g:
                                    dialogue_history["next_instruction"] = content
                                # g2u: dialogue_history[round_num] is already the content; extract_combined_instructions will handle it
                                _me_log(f"[GPU {gpu_id}] Round {round_num} ({eval_tag}): no yes/no marker, treating output as instruction: {content[:100]}...")

                            else:
                                # Case E: output is completely empty (inference failed)
                                if is_u2g:
                                    fallback = dialogue_history.get("next_instruction") or data_item.get("instruction", "")
                                    if fallback:
                                        dialogue_history["next_instruction"] = fallback
                                        _me_log(f"[GPU {gpu_id}] Round {round_num} ({eval_tag}): empty output → fallback to previous instruction: {fallback[:100]}...")
                                    else:
                                        should_stop_dialogue = True
                                        _me_log(f"[GPU {gpu_id}] Round {round_num} ({eval_tag}): empty output, no fallback → stopping")
                                else:
                                    # g2u: overwrite the current empty value with the previous understanding round's output
                                    prev_und_rounds = [k for k in dialogue_history if isinstance(k, int) and k < round_num and k % 2 == 1]
                                    if prev_und_rounds:
                                        prev_und = dialogue_history[max(prev_und_rounds)]
                                        dialogue_history[round_num] = prev_und
                                        _me_log(f"[GPU {gpu_id}] Round {round_num} ({eval_tag}): empty output → reusing previous understanding output as fallback")
                                    else:
                                        should_stop_dialogue = True
                                        _me_log(f"[GPU {gpu_id}] Round {round_num} ({eval_tag}): empty output, no previous understanding → stopping")

                        if progress_tracker:
                            progress_tracker.update(1)

                        result = {
                            "dialogue_index": idx,
                            "image_path": image_path if image_path else "",
                            "input_data": json.dumps(data_item, ensure_ascii=False),
                            "prompt_used": prompt,
                            "model_response": model_response,
                            "status": status,
                            "task_id": task_id,
                            "model_name": model_name,
                            "task_name": task.get("task_name", task_id),
                            "operation_type": round_operation_type,
                            "operation_index": op_idx,
                            "round_number": round_num
                        }

                        if task_id == "IC":
                            questions = data_item.get("questions", {})
                            if questions and str(round_num) in questions:
                                result["question_id"] = str(round_num)
                                result["question_text"] = questions[str(round_num)]
                            else:
                                result["question_id"] = ""
                                result["question_text"] = ""
                        else:
                            result["question_id"] = ""
                            result["question_text"] = ""

                        if output_suffix:
                            result["output_suffix"] = output_suffix
                        _csv_writer.writerow(result)
                        _csv_fh.flush()
                        results_count += 1

                else:
                    # -------------------------------------------------------
                    # Plain prompt (a string or a list of strings)
                    # -------------------------------------------------------
                    if isinstance(prompts_result, str):
                        prompts_list = [prompts_result]
                        question_info_list = [None]
                    else:
                        prompts_list = prompts_result
                        questions = data_item.get("questions", {})
                        if questions:
                            question_info_list = []
                            for q_id, question_text in questions.items():
                                question_info_list.append({
                                    "question_id": q_id,
                                    "question_text": question_text
                                })
                        else:
                            question_info_list = [{"question_id": str(i), "question_text": ""}
                                                 for i in range(len(prompts_list))]

                    for prompt_idx, prompt in enumerate(prompts_list):
                        question_info = question_info_list[prompt_idx] if prompt_idx < len(question_info_list) else None

                        if task_id == "GGU" and operation_type == "unify":
                            # GGU unify: pass both the source image and the reference image produced by the earlier generation operation
                            gen_image = operation_outputs.get(("generation", idx), "")
                            has_original = bool(image_path) and os.path.exists(image_path)
                            has_generated = bool(gen_image) and os.path.exists(gen_image)
                            if has_original and has_generated:
                                current_image_path = [image_path, gen_image]
                            elif has_generated:
                                current_image_path = gen_image
                            elif has_original:
                                current_image_path = image_path
                            else:
                                current_image_path = None
                        elif operation_type != "generation":
                            current_image_path = image_path
                        else:
                            current_image_path = None

                        try:
                            inference.config["task_id"] = task_id
                            inference.config["inference_mode"] = operation_type
                            model_response = dispatch_inference(
                                api_type, current_image_path, prompt, inference.config,
                                inference, index=idx, operation_type=operation_type
                            )
                            status = "success"
                        except Exception as e:
                            error_msg = f"[GPU {gpu_id}] Error during inference: {e}"
                            if logger:
                                logger.error(error_msg)
                            else:
                                print(error_msg)
                            model_response = ""
                            status = f"error: {str(e)}"

                        # Write to the cross-operation reference cache (successful results only)
                        if status == "success" and model_response:
                            operation_outputs[(operation_type, idx)] = model_response

                        if progress_tracker:
                            progress_tracker.update(1)

                        result = {
                            "dialogue_index": idx,
                            "image_path": image_path if image_path else "",
                            "input_data": json.dumps(data_item, ensure_ascii=False),
                            "prompt_used": prompt,
                            "model_response": model_response,
                            "status": status,
                            "task_id": task_id,
                            "model_name": model_name,
                            "task_name": task.get("task_name", task_id),
                            "operation_type": operation_type,
                            "operation_index": op_idx,
                            "round_number": ""
                        }

                        if question_info:
                            result["question_id"] = question_info.get("question_id", "")
                            result["question_text"] = question_info.get("question_text", "")
                        else:
                            result["question_id"] = ""
                            result["question_text"] = ""

                        if output_suffix:
                            result["output_suffix"] = output_suffix
                        _csv_writer.writerow(result)
                        _csv_fh.flush()
                        results_count += 1

        if results_count > 0:
            if logger:
                logger.info(f"[GPU {gpu_id}] Saved {results_count} new results to: {temp_result_file}")
            else:
                print(f"[GPU {gpu_id}] Saved {results_count} new results to: {temp_result_file}")
        else:
            if logger:
                logger.info(f"[GPU {gpu_id}] No new results to save")
            else:
                print(f"[GPU {gpu_id}] No new results to save")

        log_msg = f"\n[GPU {gpu_id}] ========================================\n"
        log_msg += f"[GPU {gpu_id}] Completed inference for task {task_id}\n"
        log_msg += f"[GPU {gpu_id}] Total results saved: {results_count}\n"
        log_msg += f"[GPU {gpu_id}] ========================================"
        if logger:
            logger.info(log_msg)
        else:
            print(log_msg)

    except Exception as e:
        error_msg = f"\n[GPU {gpu_id}] ========================================\n"
        error_msg += f"[GPU {gpu_id}] FATAL ERROR in inference for task {task_id}\n"
        error_msg += f"[GPU {gpu_id}] Error: {str(e)}\n"
        error_msg += f"[GPU {gpu_id}] Traceback:\n{traceback.format_exc()}\n"
        error_msg += f"[GPU {gpu_id}] ========================================\n"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        raise RuntimeError(error_msg) from e
    finally:
        if _csv_fh is not None:
            _csv_fh.close()
        if logger:
            logger.close()
        if me_dialogue_logger:
            me_dialogue_logger.close()


def _apply_test_limit(task_data: List[Dict[str, Any]], task_id: str,
                      test_mode: bool, test_n: int, num_gpus: int) -> List[Dict[str, Any]]:
    """Truncate the dataset according to the test mode"""
    if test_n > 0:
        if task_id == "GGU":
            by_cat: Dict[str, List] = {}
            for item in task_data:
                cat = item.get("category", "")
                by_cat.setdefault(cat, []).append(item)
            result = []
            for cat_items in by_cat.values():
                result.extend(cat_items[:test_n])
            return result
        else:
            return task_data[:test_n]
    elif test_mode:
        return task_data[:2 * num_gpus]
    return task_data


def calculate_total_items(tasks: List[Dict[str, Any]], model_configs: List[str],
                          base_data_dir: str, test_mode: bool = False,
                          num_gpus: int = 1, test_n: int = 0) -> int:
    """Compute the total number of data items"""
    total = 0

    for model_config_path in model_configs:
        for task in tasks:
            task_id = task["task_id"]

            try:
                task_data = load_task_data(task, base_data_dir)
                task_data = _apply_test_limit(task_data, task_id, test_mode, test_n, num_gpus)

                inference_operations = task.get("inference_operations", None)
                if inference_operations is None or len(inference_operations) == 0:
                    inference_operations = [{"operation_type": "understanding"}]

                for data_item in task_data:
                    prompts_result = build_prompt_for_operation(
                        inference_operations[0].get("prompt_template", None) if inference_operations else None,
                        task_id, data_item,
                        max_iterations=task.get("max_iterations", 5)
                    )

                    if is_dialogue_template(prompts_result):
                        total += len(prompts_result)
                    else:
                        if isinstance(prompts_result, str):
                            total += len(inference_operations)
                        else:
                            total += len(prompts_result) * len(inference_operations)
            except Exception:
                continue

    return total


def merge_gpu_results(result_dir: str, model_name: str, task_id: str, gpu_ids: List[int],
                      logger: Optional['InferenceLogger'] = None) -> None:
    """Merge the result files from all GPUs"""
    safe_model_name = model_name.replace("/", "_").replace(" ", "_")
    output_dir = os.path.join(result_dir, safe_model_name, task_id)
    final_result_file = os.path.join(output_dir, f"{task_id}_{safe_model_name}_results.csv")

    all_dfs = []
    temp_files = []

    for gpu_id in gpu_ids:
        temp_file = os.path.join(output_dir, f"{task_id}_{safe_model_name}_gpu{gpu_id}_results.csv")
        if os.path.exists(temp_file):
            df = pd.read_csv(temp_file)
            if len(df) > 0:
                all_dfs.append(df)
                temp_files.append(temp_file)
            else:
                os.remove(temp_file)

    if not all_dfs:
        msg = f"  No results to merge for task {task_id}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)

    if len(merged_df) > 0:
        if "round_number" in merged_df.columns:
            merged_df["round_number"] = merged_df["round_number"].fillna("")
        if "question_id" in merged_df.columns:
            merged_df["question_id"] = merged_df["question_id"].fillna("")

        if task_id == "IC" and "operation_type" in merged_df.columns:
            duplicates_found = False
            duplicate_rows_list = []
            duplicate_count_total = 0

            understanding_df = merged_df[merged_df["operation_type"] == "understanding"]
            if len(understanding_df) > 0:
                understanding_unique_keys = ["dialogue_index", "question_id"]
                understanding_duplicates = understanding_df.duplicated(subset=understanding_unique_keys, keep=False)
                if understanding_duplicates.any():
                    duplicates_found = True
                    duplicate_count = understanding_duplicates.sum()
                    duplicate_count_total += duplicate_count
                    duplicate_rows_list.append({
                        "operation_type": "understanding",
                        "unique_keys": understanding_unique_keys,
                        "duplicates": understanding_df[understanding_duplicates],
                        "count": duplicate_count
                    })

            generation_df = merged_df[merged_df["operation_type"] == "generation"]
            if len(generation_df) > 0:
                generation_unique_keys = ["dialogue_index"]
                generation_duplicates = generation_df.duplicated(subset=generation_unique_keys, keep=False)
                if generation_duplicates.any():
                    duplicates_found = True
                    duplicate_count = generation_duplicates.sum()
                    duplicate_count_total += duplicate_count
                    duplicate_rows_list.append({
                        "operation_type": "generation",
                        "unique_keys": generation_unique_keys,
                        "duplicates": generation_df[generation_duplicates],
                        "count": duplicate_count
                    })

            if duplicates_found:
                error_msg = f"\n  ERROR: Found {duplicate_count_total} duplicate results in task {task_id}\n"
                for dup_info in duplicate_rows_list:
                    error_msg += f"  Operation '{dup_info['operation_type']}' duplicates ({dup_info['count']} entries):\n"
                    error_msg += f"    Unique key fields: {', '.join(dup_info['unique_keys'])}\n"
                    error_msg += f"    Duplicate entries:\n"
                    for idx, row in dup_info['duplicates'].head(20).iterrows():
                        error_msg += f"      - "
                        key_parts = []
                        for field in dup_info['unique_keys']:
                            value = row.get(field, 'N/A')
                            key_parts.append(f"{field}={value}")
                        error_msg += ", ".join(key_parts) + "\n"
                    if dup_info['count'] > 20:
                        error_msg += f"      ... and {dup_info['count'] - 20} more duplicates\n"
                if logger:
                    logger.error(error_msg)
                else:
                    print(error_msg)
                raise ValueError(f"Duplicate results found in task {task_id}.")
        else:
            unique_key_fields = ["dialogue_index"]

            # operation_index must be added before operation_type:
            # ME has two ops with operation_type="unify", both with round_number in 1-10,
            # so without operation_index the results of the two ops would be mistaken for duplicates.
            if "operation_index" in merged_df.columns:
                unique_key_fields.append("operation_index")

            if "operation_type" in merged_df.columns:
                unique_key_fields.append("operation_type")

            if "round_number" in merged_df.columns:
                unique_key_fields.append("round_number")

            if "question_id" in merged_df.columns:
                unique_key_fields.append("question_id")

            duplicates = merged_df.duplicated(subset=unique_key_fields, keep=False)

            if duplicates.any():
                duplicate_count = duplicates.sum()
                duplicate_rows = merged_df[duplicates]
                error_msg = f"\n  ERROR: Found {duplicate_count} duplicate results in task {task_id}\n"
                error_msg += f"  Unique key fields: {', '.join(unique_key_fields)}\n"
                error_msg += f"  Duplicate entries:\n"
                for idx, row in duplicate_rows.head(20).iterrows():
                    error_msg += f"    - "
                    key_parts = []
                    for field in unique_key_fields:
                        value = row.get(field, 'N/A')
                        key_parts.append(f"{field}={value}")
                    error_msg += ", ".join(key_parts) + "\n"
                if duplicate_count > 20:
                    error_msg += f"    ... and {duplicate_count - 20} more duplicates\n"
                if logger:
                    logger.error(error_msg)
                else:
                    print(error_msg)
                raise ValueError(f"Duplicate results found in task {task_id}. "
                               f"Each ({', '.join(unique_key_fields)}) should be unique.")

    if "dialogue_index" in merged_df.columns:
        merged_df = merged_df.sort_values("dialogue_index").reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    merged_df.to_csv(final_result_file, index=False, encoding='utf-8')
    msg = f"  Merged {len(merged_df)} results from {len(temp_files)} GPU(s) to {final_result_file}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except Exception as e:
            warning_msg = f"  Warning: Failed to remove temporary file {temp_file}: {e}"
            if logger:
                logger.info(warning_msg)
            else:
                print(warning_msg)


def main():
    parser = argparse.ArgumentParser(description='Generic inference script (supports multi-GPU parallel inference)')
    parser.add_argument('--tasks', type=str, default=None, help='Comma-separated list of tasks (e.g. IC,UGG)')
    parser.add_argument('--models', type=str, default=None, help='Comma-separated list of models')
    parser.add_argument('--config-dir', type=str, default='config', help='Configuration directory')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--result-dir', type=str, default='result', help='Results directory')
    parser.add_argument('--gpus', type=str, required=True, help='Comma-separated GPU IDs (required)')
    parser.add_argument('--test', action='store_true', help='Test mode: process only 2 data items per task')
    parser.add_argument('--test-n', type=int, default=0, dest='test_n',
                        help='Test mode: process N data items per task (N per sub-task for GGU); takes precedence over --test')

    args = parser.parse_args()

    gpu_list = [int(g.strip()) for g in args.gpus.split(",")]
    if len(gpu_list) == 0:
        raise ValueError("--gpus cannot be empty; please specify at least one GPU ID")

    log_file_path = os.path.join(args.result_dir, "inference.log")
    logger = InferenceLogger(log_file_path)

    logger.info("\n" + "="*60)
    logger.info("MULTI-GPU INFERENCE MODE")
    logger.info("="*60)
    logger.info(f"GPU IDs: {gpu_list}")
    logger.info(f"Number of GPUs: {len(gpu_list)}")
    logger.info("="*60 + "\n")

    if args.test:
        logger.info("\n" + "="*60)
        logger.info("TEST MODE ENABLED: Each task will process only 2 data items")
        logger.info("="*60 + "\n")

    config = load_config("tasks.yaml")
    tasks = config.get("tasks", [])

    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",")]
        tasks = [t for t in tasks if t["task_id"] in task_ids]

    if args.models:
        model_configs = [m.strip() for m in args.models.split(",")]
    else:
        model_configs = MODELS

    logger.info("\n" + "="*60)
    logger.info("DATA VALIDATION PHASE")
    logger.info("="*60)

    validation_passed = validate_all_tasks_data(
        tasks=tasks,
        base_data_dir=args.data_dir,
        skip_threshold=0.5,
        continue_on_error=False,
        logger=logger
    )

    if not validation_passed:
        logger.error("\n" + "="*60)
        logger.error("DATA VALIDATION FAILED: Please fix the data issues before proceeding")
        logger.error("="*60)
        logger.close()
        return

    logger.info("\n" + "="*60)
    logger.info("DATA VALIDATION PASSED: All tasks data loaded successfully")
    logger.info("="*60 + "\n")

    total_items = calculate_total_items(tasks, model_configs, args.data_dir, args.test,
                                        num_gpus=len(gpu_list), test_n=args.test_n)
    logger.info(f"Total data items to process: {total_items}")

    manager = multiprocessing.Manager()
    progress_tracker = ProgressTracker(manager, total_items)

    pbar = tqdm(total=total_items, desc="Overall Progress", unit="item",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    def update_progress_bar():
        cached_completed = 0
        cached_total = total_items
        cached_percentage = 0.0
        connection_failed = False
        normal_interval = 0.5

        while True:
            try:
                progress_info = progress_tracker.get_progress()
                completed = progress_info["completed"]
                total = progress_info["total"]
                percentage = progress_info["percentage"]
                remaining_time = progress_info["remaining_time"]

                cached_completed = completed
                cached_total = total
                cached_percentage = percentage
                connection_failed = False

                pbar.n = completed

                if remaining_time is not None:
                    remaining_str = str(timedelta(seconds=int(remaining_time)))
                    pbar.set_description(f"Overall Progress ({percentage:.1f}%, ETA: {remaining_str})")
                else:
                    pbar.set_description(f"Overall Progress ({percentage:.1f}%)")

                pbar.refresh()

                if completed >= total:
                    break

                time.sleep(normal_interval)

            except (ConnectionResetError, BrokenPipeError, OSError):
                if not connection_failed:
                    connection_failed = True
                    if cached_total > 0:
                        pbar.n = cached_completed
                        pbar.set_description(f"Overall Progress ({cached_percentage:.1f}%, connection lost)")
                        pbar.refresh()

                if cached_completed >= cached_total:
                    break

                time.sleep(normal_interval)
                continue

            except Exception:
                if cached_total > 0:
                    pbar.n = cached_completed
                    pbar.set_description(f"Overall Progress ({cached_percentage:.1f}%)")
                    pbar.refresh()

                if cached_completed >= cached_total:
                    break

                time.sleep(normal_interval)
                continue

    progress_thread = threading.Thread(target=update_progress_bar, daemon=True)
    progress_thread.start()

    for model_idx, model_config_path in enumerate(model_configs):
        model_config = load_model_config(model_config_path)
        model_name = model_config.get("model_name", Path(model_config_path).stem)
        safe_model_name = model_name.replace("/", "_").replace(" ", "_")

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing with model: {model_name}")
        logger.info(f"{'='*60}")

        excluded_tasks = model_config.get("excluded_tasks", [])

        for task in tasks:
            task_id = task["task_id"]
            if task_id in excluded_tasks:
                logger.info(f"\nSkipping task {task_id} for {model_name} (excluded_tasks)")
                continue
            logger.info(f"\nProcessing task: {task_id}")

            try:
                task_data = load_task_data(task, args.data_dir)
                original_count = len(task_data)

                task_data = _apply_test_limit(task_data, task_id, args.test, args.test_n, len(gpu_list))
                if args.test_n > 0:
                    logger.info(f"Loaded {len(task_data)} data items (--test-n {args.test_n}, limited from {original_count} items)")
                elif args.test:
                    logger.info(f"Loaded {len(task_data)} data items (test mode: {len(gpu_list)} GPUs × 2 = {len(task_data)}, limited from {original_count} items)")
                else:
                    logger.info(f"Loaded {len(task_data)} data items")
            except Exception as e:
                logger.error(f"Error loading task data: {e}")
                continue

            data_splits = split_data_for_gpus(task_data, len(gpu_list))
            logger.info(f"\nData distribution across {len(gpu_list)} GPU(s):")
            for gpu_idx, gpu_id in enumerate(gpu_list):
                logger.info(f"  GPU {gpu_id}: {len(data_splits[gpu_idx])} data items")

            processes = []

            for gpu_idx, gpu_id in enumerate(gpu_list):
                if len(data_splits[gpu_idx]) == 0:
                    logger.info(f"  Skipping GPU {gpu_id}: no data assigned")
                    continue

                process_model_config = copy.deepcopy(model_config)

                log_file_path_for_gpu = logger.get_log_file_path() if logger else None
                p = multiprocessing.Process(
                    target=run_inference_on_gpu,
                    args=(gpu_id, process_model_config, task, args, data_splits[gpu_idx], log_file_path_for_gpu, progress_tracker)
                )
                p.start()
                processes.append((gpu_id, p))
                logger.info(f"  Started process for GPU {gpu_id} (PID: {p.pid})")

            if not processes:
                logger.info(f"  WARNING: No processes started for task {task_id} (no data assigned to any GPU)")
                continue

            logger.info(f"\nWaiting for all {len(processes)} process(es) to complete...")
            failed_processes = []
            successful_processes = []

            for gpu_id, p in processes:
                p.join()
                if p.exitcode != 0:
                    failed_processes.append(gpu_id)
                    logger.error(f"  ERROR: Process for GPU {gpu_id} (PID: {p.pid}) exited with code {p.exitcode}")
                else:
                    successful_processes.append(gpu_id)
                    logger.info(f"  GPU {gpu_id} process completed successfully")

            if failed_processes:
                logger.info(f"\n  WARNING: {len(failed_processes)} process(es) failed: {failed_processes}")
                if successful_processes:
                    logger.info(f"  {len(successful_processes)} process(es) completed successfully: {successful_processes}")
                    logger.info(f"  Attempting to merge results from successful processes...")
                else:
                    logger.error(f"  ERROR: All processes failed for task {task_id}")
                    logger.error(f"  Skipping result merge for this task")
                    continue

            try:
                logger.info(f"\nMerging results from all GPUs...")
                merge_gpu_results(args.result_dir, model_name, task_id, gpu_list, logger)
                logger.info(f"  Successfully merged results for task {task_id}")
            except ValueError as e:
                logger.error(f"\n  ERROR: {e}")
                if failed_processes:
                    logger.error(f"  This may be due to failed processes on GPU(s): {failed_processes}")
                raise
            except Exception as e:
                logger.error(f"\n  ERROR: Failed to merge results for task {task_id}: {e}")
                logger.error(f"  Traceback: {traceback.format_exc()}")
                if failed_processes:
                    logger.error(f"  This may be due to failed processes on GPU(s): {failed_processes}")
                logger.info(f"  Continuing with next task...")

    pbar.close()

    logger.info("\n" + "="*60)
    logger.info("All tasks completed!")
    logger.info("="*60)

    logger.close()


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError as e:
        import sys
        print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}", file=sys.stderr)
        print("This may cause CUDA initialization errors in child processes.", file=sys.stderr)

    main()
