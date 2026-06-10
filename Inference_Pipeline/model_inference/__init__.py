"""
model_inference package — shared utility functions for the per-model inference adapters
"""
import os
import random
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    """Set random seeds to ensure reproducibility"""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_output_path(
    model_config: Dict[str, Any],
    index: Optional[int] = None,
    mode: str = "generation",
    round_number: Optional[int] = None,
) -> str:
    """
    Build the output image path (used uniformly by all adapters).

    Args:
        model_config: model config dict, containing output_image_dir / model_name / task_id
        index: data item index
        mode: operation mode (generation / editing / unify)
        round_number: dialogue round number

    Returns:
        Full path string of the output image
    """
    output_dir = model_config.get("output_image_dir", "result/images")
    model_name = model_config.get("model_name", "model")
    task_id = model_config.get("task_id", "unknown")

    safe_model_name = model_name.replace("/", "_").replace(" ", "_")

    if "{model_name}" in output_dir:
        output_dir = output_dir.format(model_name=safe_model_name)

    full_output_dir = os.path.join(output_dir, task_id)
    os.makedirs(full_output_dir, exist_ok=True)

    if index is not None:
        if round_number is not None:
            filename = f"{index:06d}_round{round_number}_{mode}.png"
        else:
            filename = f"{index:06d}_{mode}.png"
    else:
        if round_number is not None:
            filename = f"{int(time.time())}_round{round_number}_{mode}.png"
        else:
            filename = f"{int(time.time())}_{mode}.png"

    return os.path.join(full_output_dir, filename)


def load_pil_image(image_path: str) -> Image.Image:
    """Load a PIL image and convert it to RGB"""
    return Image.open(image_path).convert("RGB")
