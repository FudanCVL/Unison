#!/usr/bin/env python3
"""Evaluate internal consistency using data from Inference_Pipeline CSV results."""
import argparse
import json
import os
import re
from typing import Dict, List, Sequence, Tuple
import torch
import torch.multiprocessing as torch_mp
from transformers import AutoModelForImageTextToText, AutoProcessor
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import pandas as pd


torch_mp.set_start_method('spawn', force=True)


DEFAULT_CSV_PATH = "../Inference_Pipeline/result/BAGEL-7B-MoT/IC/IC_BAGEL-7B-MoT_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BAGEL internal consistency using CSV data")
    parser.add_argument("--model-path", required=True, help="Path or repo id of Qwen3-VL model")
    parser.add_argument(
        "--csv-path",
        default=DEFAULT_CSV_PATH,
        help="Path to IC results CSV file from Inference_Pipeline",
    )
    parser.add_argument("--output", default="evaluate_ic_results.json", help="Where to store detailed results")
    parser.add_argument("--max-items", type=int, default=None, help="Optional limit on number of BAGEL pairs")
    parser.add_argument("--batch-size", type=int, default=16, help="Generation batch size per modality")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Tokens generated per answer")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 => greedy)")
    parser.add_argument("--device", default="auto", help="Device spec (auto, cpu, cuda:0, etc)")
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated GPU ids (e.g. 0,1) to set CUDA_VISIBLE_DEVICES for multi-card inference",
    )
    parser.add_argument(
        "--caption-prefix",
        default="Caption: ",
        help="Prefix placed before caption text when querying the model",
    )
    parser.add_argument(
        "--question-prefix",
        default="Question: ",
        help="Prefix placed before each question when querying the model",
    )
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a precise VQA assistant. Answer each question with only 'yes' or 'no'. "
            "If the information is unknown, answer 'no'."
        ),
        help="System prompt to steer short yes/no outputs",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPUs to use for parallel processing",
    )
    parser.add_argument(
        "--images-base-dir",
        default=None,
        help="Base directory for resolving image paths (if relative paths in CSV)",
    )
    return parser.parse_args()


def load_data_from_csv(csv_path: str, images_base_dir: str = None) -> List[Tuple[str, List[Tuple[str, str]], str, str]]:
    """
    Load data from CSV file.
    Returns: List of (prompt_id, question_items, image_path, caption_text) tuples
    """
    df = pd.read_csv(csv_path)
    
    # Filter only understanding operations to get questions
    understanding_df = df[df['operation_type'] == 'understanding'].copy()
    
    # Group by dialogue_index to get all questions for each prompt
    grouped = understanding_df.groupby('dialogue_index')
    
    all_data = []
    for dialogue_idx, group in grouped:
        # Get the first row to extract common data
        first_row = group.iloc[0]
        
        # Parse input_data JSON
        try:
            input_data = json.loads(first_row['input_data'])
        except (json.JSONDecodeError, TypeError):
            if isinstance(first_row['input_data'], dict):
                input_data = first_row['input_data']
            else:
                print(f"Warning: Could not parse input_data for dialogue_index {dialogue_idx}, skipping")
                continue
        
        # Extract image path
        image_path = first_row['image_path']
        
        # Try to resolve image path
        # First, try the path as-is
        if os.path.exists(image_path):
            image_path = os.path.abspath(image_path)
        elif images_base_dir and not os.path.isabs(image_path):
            # Try with base directory
            image_path = os.path.join(images_base_dir, image_path)
            if os.path.exists(image_path):
                image_path = os.path.abspath(image_path)
            else:
                image_path = None
        else:
            image_path = None
        
        # If still not found, try common locations relative to CSV
        if image_path is None or not os.path.exists(image_path):
            csv_dir = os.path.dirname(csv_path)
            # Try original image paths (from input_data or common locations)
            original_img_path = input_data.get('image_path', '')
            if original_img_path:
                # Extract image filename (e.g., "1.png" from "data/Internal_Consistency/images/1.png")
                img_filename = os.path.basename(original_img_path)
                # Extract index from filename (e.g., "1" from "1.png")
                try:
                    img_index = int(os.path.splitext(img_filename)[0])
                except ValueError:
                    img_index = None
                
                possible_paths = [
                    original_img_path,  # Try as-is
                    os.path.join(csv_dir, "..", "data", "Internal_Consistency", "images", img_filename),
                    os.path.join(csv_dir, "..", "..", "data_test", "Internal_Consistency", "images", img_filename),
                    os.path.join(csv_dir, "..", "..", "Dataset", "Internal_Consistency", "images", img_filename),
                ]
                # If we have image index, also try with index-based paths
                if img_index is not None:
                    possible_paths.extend([
                        os.path.join(csv_dir, "..", "data", "Internal_Consistency", "images", f"{img_index}.png"),
                        os.path.join(csv_dir, "..", "data", "Internal_Consistency", "images", f"{img_index:03d}.png"),
                    ])
                
                for pp in possible_paths:
                    if os.path.exists(pp):
                        image_path = os.path.abspath(pp)
                        break
            
            # If still not found, try result directory images (generated images)
            if image_path is None or not os.path.exists(image_path):
                possible_paths = [
                    os.path.join(csv_dir, "..", "images", "IC", f"{dialogue_idx:06d}_generation.png"),
                    os.path.join(csv_dir, "..", "images", f"{dialogue_idx:06d}.png"),
                    os.path.join(csv_dir, "..", "images", f"{dialogue_idx}.png"),
                ]
                for pp in possible_paths:
                    if os.path.exists(pp):
                        image_path = os.path.abspath(pp)
                        break
        
        if image_path is None or not os.path.exists(image_path):
            print(f"Warning: Image not found for dialogue_index {dialogue_idx} (tried: {first_row['image_path']}), skipping")
            continue
        
        # Extract caption (prompt)
        caption_text = input_data.get('prompt', '').strip()
        if not caption_text:
            print(f"Warning: No caption found for dialogue_index {dialogue_idx}, skipping")
            continue
        
        # Extract questions
        questions_dict = input_data.get('questions', {})
        if not questions_dict:
            print(f"Warning: No questions found for dialogue_index {dialogue_idx}, skipping")
            continue
        
        # Convert questions to sorted list of (question_id, question_text) tuples
        question_items = sorted(
            questions_dict.items(),
            key=lambda kv: int(kv[0]) if kv[0].isdigit() else float('inf')
        )
        
        # Create prompt_id (using dialogue_index)
        prompt_id = f"prompt_{dialogue_idx}"
        
        all_data.append((prompt_id, question_items, os.path.abspath(image_path), caption_text))
    
    return all_data


def normalize_answer(text: str) -> str:
    cleaned = text.strip().lower()
    if cleaned in {"yes", "no"}:
        return cleaned
    tokens = re.split(r"[^a-z]+", cleaned)
    for token in tokens:
        if token in {"yes", "no"}:
            return token
    if "yes" in cleaned and "no" not in cleaned:
        return "yes"
    if "no" in cleaned and "yes" not in cleaned:
        return "no"
    return "unknown"


def build_image_message(image_path: str, question: str, system_prompt: str) -> List[Dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},  # Use direct path, not file:// URL
                {"type": "text", "text": question},
            ],
        },
    ]


def build_caption_message(caption_text: str, question: str, system_prompt: str, caption_prefix: str, question_prefix: str) -> List[Dict]:
    user_text = f"{caption_prefix}{caption_text}\n{question_prefix}{question}"
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]


def chunk_list(items: Sequence, chunk_size: int) -> List[Sequence]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def prepare_device_map(device: str, gpu_id: int):
    if device == "auto":
        # return "auto"
        return {"": f"cuda:{gpu_id}"}
    if device == "cpu":
        return "cpu"
    return {"": device}


def batch_generate(
    model,
    processor,
    messages_batch: List[List[Dict]],
    max_new_tokens: int,
    temperature: float,
) -> List[str]:
    if not messages_batch:
        return []
    inputs = processor.apply_chat_template(
        messages_batch,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    inputs = inputs.to(device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
    }
    # transformers ignores temperature when do_sample=False, so remove None
    if generation_kwargs["temperature"] is None:
        generation_kwargs.pop("temperature")
    generated_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [t.strip() for t in texts]


def evaluate_prompt_with_progress(
    model,
    processor,
    prompt_id: str,
    question_items: List[Tuple[str, str]],
    image_path: str,
    caption_text: str,
    args: argparse.Namespace,
    pbar=None  # progress bar object
) -> Dict:
    total = len(question_items)
    if total == 0:
        raise ValueError(f"No questions found for {prompt_id}")

    image_messages = [
        build_image_message(image_path, q_text, args.system_prompt)
        for _, q_text in question_items
    ]
    caption_messages = [
        build_caption_message(caption_text, q_text, args.system_prompt, args.caption_prefix, args.question_prefix)
        for _, q_text in question_items
    ]

    image_outputs = []
    caption_outputs = []
    
    # Process image batches
    image_batches = chunk_list(image_messages, args.batch_size)
    for batch in image_batches:
        image_outputs.extend(
            batch_generate(model, processor, batch, args.max_new_tokens, args.temperature)
        )
        if pbar:
            pbar.update(1)  # update progress bar after each batch
    
    # Process text batches
    caption_batches = chunk_list(caption_messages, args.batch_size)
    for batch in caption_batches:
        caption_outputs.extend(
            batch_generate(model, processor, batch, args.max_new_tokens, args.temperature)
        )
        if pbar:
            pbar.update(1)  # update progress bar after each batch

    image_answers = {}
    caption_answers = {}
    image_outs = {}
    caption_outs = {}
    image_yes = 0
    caption_yes = 0
    both_yes = 0

    for (qid, _), img_out, cap_out in zip(question_items, image_outputs, caption_outputs):
        img_ans = normalize_answer(img_out)
        cap_ans = normalize_answer(cap_out)
        image_answers[qid] = img_ans
        caption_answers[qid] = cap_ans
        image_outs[qid] = img_out
        caption_outs[qid] = cap_out
        is_img_yes = img_ans == "yes"
        is_cap_yes = cap_ans == "yes"
        image_yes += int(is_img_yes)
        caption_yes += int(is_cap_yes)
        both_yes += int(is_img_yes and is_cap_yes)

    image_accuracy = image_yes / total
    caption_accuracy = caption_yes / total
    consistency = both_yes / total

    return {
        "prompt_id": prompt_id,
        "index": int(prompt_id.split("_")[-1]) + 1,
        "image_path": image_path,
        "caption_length": len(caption_text),
        "image_accuracy": image_accuracy,
        "caption_accuracy": caption_accuracy,
        "consistency": consistency,
        "image_yes": image_yes,
        "caption_yes": caption_yes,
        "both_yes": both_yes,
        "total_questions": total,
        "image_answers": image_answers,
        "image_outs": image_outs,
        "caption_answers": caption_answers,
        "caption_outs": caption_outs
    }


def process_batch_worker(args_batch):
    """Worker function for processing a batch of prompts with progress tracking"""
    args_dict, batch_data, worker_id = args_batch
    
    # Load model and processor in each process
    args = argparse.Namespace(**args_dict)
    device_map = prepare_device_map(args.device, worker_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    
    # Calculate total batches for progress bar
    total_batches = 0
    for _, question_items, _, _ in batch_data:
        total_batches += len(chunk_list(question_items, args.batch_size)) * 2  # image + caption
    
    results = []
    with tqdm(total=total_batches, desc=f"GPU {worker_id}", position=worker_id, leave=True) as pbar:
        for prompt_id, question_items, image_path, caption_text in batch_data:
            result = evaluate_prompt_with_progress(
                model,
                processor,
                prompt_id,
                question_items,
                image_path,
                caption_text,
                args,
                pbar=pbar,
            )
            results.append(result)
    
    return results


def main():
    args = parse_args()
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"Using GPUs: {args.gpu_ids}")
    
    # Load data from CSV
    print(f"Loading data from CSV: {args.csv_path}")
    all_data = load_data_from_csv(args.csv_path, args.images_base_dir)
    
    if args.max_items is not None:
        all_data = all_data[:args.max_items]
    
    if len(all_data) == 0:
        raise RuntimeError("No data loaded from CSV. Check CSV path and format.")
    
    print(f"Loaded {len(all_data)} prompts from CSV")

    # Split data into chunks for each GPU
    num_gpus = args.num_gpus
    chunk_size = len(all_data) // num_gpus
    if len(all_data) % num_gpus != 0:
        chunk_size += 1  # Make sure all items are included
    
    data_chunks = [all_data[i:i + chunk_size] for i in range(0, len(all_data), chunk_size)]
    
    # Prepare arguments for each process (including worker ID)
    args_dict = {
        "model_path": args.model_path,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "device": args.device,
        "caption_prefix": args.caption_prefix,
        "question_prefix": args.question_prefix,
        "system_prompt": args.system_prompt
    }
    
    args_batches = [(args_dict, chunk, i) for i, chunk in enumerate(data_chunks)]
    
    # Process in parallel
    results = []
    print(f"Starting processing with {num_gpus} GPUs...")
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(process_batch_worker, args_batch) for args_batch in args_batches]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing GPU batches"):
            results.extend(future.result())
    
    # Sort results by original order to maintain sequence
    results.sort(key=lambda x: int(x["prompt_id"].split("_")[-1]))
    
    if len(results) == 0:
        raise RuntimeError("No prompts processed successfully. Check input paths.")

    # Calculate summary statistics
    image_acc_sum = sum(result["image_accuracy"] for result in results)
    caption_acc_sum = sum(result["caption_accuracy"] for result in results)
    consistency_sum = sum(result["consistency"] for result in results)
    
    count = len(results)
    summary = {
        "items": count,
        "average_image_accuracy": image_acc_sum / count,
        "average_caption_accuracy": caption_acc_sum / count,
        "average_consistency": consistency_sum / count,
    }

    output_payload = {"summary": summary, "details": results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Detailed results stored at: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
