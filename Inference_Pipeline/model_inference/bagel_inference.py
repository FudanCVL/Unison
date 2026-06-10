#!/usr/bin/env python3
"""
Bagel model inference function module
Used to invoke the Bagel model within the generic inference pipeline
"""
import sys
import os
import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from PIL import Image

# Global variables holding loaded models (to avoid reloading)
_loaded_inferencers = {}
_failed_inferencers = {}
_bagel_path_setup = False


def setup_bagel_path(bagel_project_path: str):
    """Add the Bagel project path to sys.path"""
    global _bagel_path_setup
    
    if _bagel_path_setup:
        return
    
    # Resolve the path: supports both relative and absolute paths
    if os.path.isabs(bagel_project_path):
        bagel_path = Path(bagel_project_path)
    else:
        # Relative to the directory of the current script
        script_dir = Path(__file__).parent.resolve()
        bagel_path = (script_dir / bagel_project_path).resolve()
    
    if not bagel_path.exists():
        raise ValueError(f"Bagel project path does not exist: {bagel_path}")
    
    bagel_path_str = str(bagel_path)
    if bagel_path_str not in sys.path:
        sys.path.insert(0, bagel_path_str)
    
    _bagel_path_setup = True
    print(f"Bagel project path added to sys.path: {bagel_path_str}")


def set_seed(seed):
    """Set random seeds for reproducibility"""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model_and_inferencer(model_path: str, mode: int, gpu_id: int = 0):
    """Loads the BAGEL model and inferencer components."""
    # Dynamically import Bagel-related modules
    from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
    from data.data_utils import add_special_tokens, pil_img2rgb
    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel import (
        BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
        SiglipVisionConfig, SiglipVisionModel
    )
    from modeling.qwen2 import Qwen2Tokenizer
    
    # Handle HuggingFace model path
    if not os.path.exists(model_path):
        # May be a HuggingFace model name that needs to be downloaded
        from huggingface_hub import snapshot_download
        print(f"Model path {model_path} not found locally, downloading from HuggingFace...")
        model_path = snapshot_download(repo_id=model_path, local_dir_use_symlinks=False)
        print(f"Model downloaded to: {model_path}")
    
    # Load Configs
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    # Initialize Model (Empty Weights)
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # Load Tokenizer and Special Tokens
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Setup Transforms
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # Determine Device Map
    device_map = infer_auto_device_map(
        model,
        max_memory={gpu_id: "80GiB"},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]

    # All modules should go to the specified GPU
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = gpu_id
        else:
            device_map[k] = gpu_id

    # Load Model Weights based on mode
    if mode == 1:
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            offload_folder="offload",
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
    elif mode == 2:  # NF4 Quantization
        from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4"
        )
        model = load_and_quantize_model(
            model,
            weights_location=os.path.join(model_path, "ema.safetensors"),
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    elif mode == 3:  # INT8 Quantization
        from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_8bit=True,
            torch_dtype=torch.float32
        )
        model = load_and_quantize_model(
            model,
            weights_location=os.path.join(model_path, "ema.safetensors"),
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    else:
        raise NotImplementedError(f"Mode {mode} not implemented.")

    # Prepare Inferencer
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    print(f"Bagel model and inferencer loaded successfully on GPU {gpu_id}")
    return inferencer


def image_understanding(inferencer, image_paths: Union[str, List[str]], prompt: str, seed: int = 666, 
                        do_sample: bool = False, text_temperature: float = 0.3, 
                        max_new_tokens: int = 8192) -> str:
    """
    Performs image understanding inference.
    
    Args:
        inferencer: Bagel inferencer object
        image_paths: image path (string) or list of image paths (for multiple images)
        prompt: text prompt
        seed: random seed
        do_sample: whether to sample
        text_temperature: text generation temperature
        max_new_tokens: maximum number of tokens to generate

    Returns:
        text response
    """
    from data.data_utils import pil_img2rgb
    
    set_seed(seed)

    # Handle single or multiple images; None / "" / [] all indicate text-only inference
    if not image_paths:
        image_paths = []
    elif isinstance(image_paths, str):
        image_paths = [image_paths]

    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            img = pil_img2rgb(img)
            images.append(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return ""

    # Text-only understanding (e.g. GGU Complex_Relation has no reference image)
    if not images:
        inference_hyper = dict(
            do_sample=do_sample,
            text_temperature=text_temperature,
            max_think_token_n=max_new_tokens,
        )
        result = inferencer(text=prompt, think=False, understanding_output=True, **inference_hyper)
        return result.get("text", "")

    # If there are multiple images, concatenate them horizontally
    if len(images) > 1:
        # Get the maximum height across all images
        max_height = max(img.height for img in images)
        # Resize all images to the same height (preserving aspect ratio)
        resized_images = []
        total_width = 0
        for img in images:
            ratio = max_height / img.height
            new_width = int(img.width * ratio)
            resized_img = img.resize((new_width, max_height), Image.Resampling.LANCZOS)
            resized_images.append(resized_img)
            total_width += new_width
        
        # Create the concatenated image
        concatenated_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in resized_images:
            concatenated_image.paste(img, (x_offset, 0))
            x_offset += img.width
        image = concatenated_image
    else:
        image = images[0]

    # Set hyperparameters
    inference_hyper = dict(
        do_sample=do_sample,
        text_temperature=text_temperature,
        max_think_token_n=max_new_tokens,
    )

    # Perform inference
    result = inferencer(
        image=image,
        text=prompt,
        think=False,
        understanding_output=True,
        **inference_hyper
    )
    response_text = result.get("text", "")
    return response_text


def image_generation(inferencer, prompt: str, output_path: str,
                     image_shapes: tuple = (512, 512),
                     cfg_text_scale: float = 4.0,
                     cfg_interval: list = [0.4, 1.0],
                     num_timesteps: int = 50,
                     timestep_shift: float = 3.0,
                     cfg_renorm_min: float = 0.0,
                     cfg_renorm_type: str = "global",
                     seed: int = 666,
                     do_sample: bool = False,
                     text_temperature: float = 0.3,
                     max_think_tokens: int = 8192) -> str:
    """
    Generate an image from text

    Args:
        inferencer: Bagel inferencer object
        prompt: text prompt
        output_path: output image path
        image_shapes: image dimensions (H, W)
        cfg_text_scale: text guidance strength
        cfg_interval: CFG interval [start, end]
        num_timesteps: number of diffusion steps
        timestep_shift: timestep shift
        cfg_renorm_min: CFG renormalization minimum
        cfg_renorm_type: CFG renormalization type
        seed: random seed
        do_sample: whether to sample
        text_temperature: text generation temperature

    Returns:
        path to the generated image
    """
    set_seed(seed)
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Set hyperparameters
    inference_hyper = dict(
        max_think_token_n=max_think_tokens,
        do_sample=do_sample,
        text_temperature=text_temperature,
        cfg_text_scale=cfg_text_scale,
        cfg_interval=cfg_interval,
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
        image_shapes=image_shapes,
    )
    
    # Perform inference
    try:
        result = inferencer(text=prompt, think=False, **inference_hyper)
        generated_image = result.get("image")
        
        if generated_image is not None:
            generated_image.save(output_path)
            print(f"Generated image saved to: {output_path}")
            return output_path
        else:
            raise ValueError("Failed to generate image")
    except Exception as e:
        print(f"Error during image generation: {e}")
        raise


def image_editing(inferencer, image_paths: Union[str, List[str]], prompt: str, output_path: str,
                  cfg_text_scale: float = 4.0,
                  cfg_img_scale: float = 1.5,
                  cfg_interval: list = [0.4, 1.0],
                  num_timesteps: int = 50,
                  timestep_shift: float = 3.0,
                  cfg_renorm_min: float = 0.0,
                  cfg_renorm_type: str = "global",
                  seed: int = 666,
                  do_sample: bool = False,
                  text_temperature: float = 0.3,
                  max_think_tokens: int = 8192) -> str:
    """
    Edit an image

    Args:
        inferencer: Bagel inferencer object
        image_paths: image path (string) or list of image paths (when multiple, the first is used as the edit target)
        prompt: editing instruction
        output_path: output image path
        cfg_text_scale: text guidance strength
        cfg_img_scale: image guidance strength
        cfg_interval: CFG interval [start, end]
        num_timesteps: number of diffusion steps
        timestep_shift: timestep shift
        cfg_renorm_min: CFG renormalization minimum
        cfg_renorm_type: CFG renormalization type
        seed: random seed
        do_sample: whether to sample
        text_temperature: text generation temperature

    Returns:
        path to the edited image
    """
    from data.data_utils import pil_img2rgb
    
    set_seed(seed)
    
    # Handle single or multiple images: the editing task uses the first image (the original)
    if isinstance(image_paths, list):
        if len(image_paths) == 0:
            raise ValueError("image_paths list is empty")
        image_path = image_paths[0]  # Use the first image (the original) for editing
    else:
        image_path = image_paths

    # Check that the original image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Load the original image
    try:
        image = Image.open(image_path).convert('RGB')
        image = pil_img2rgb(image)
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Set hyperparameters
    inference_hyper = dict(
        max_think_token_n=max_think_tokens,
        do_sample=do_sample,
        text_temperature=text_temperature,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=cfg_interval,
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
    )
    
    # Perform inference
    try:
        result = inferencer(image=image, text=prompt, think=False, **inference_hyper)
        generated_image = result.get("image")
        
        if generated_image is not None:
            generated_image.save(output_path)
            print(f"Edited image saved to: {output_path}")
            return output_path
        else:
            raise ValueError("Failed to edit image")
    except Exception as e:
        print(f"Error during image editing: {e}")
        raise


def generate_output_path(model_config: Dict[str, Any], index: Optional[int] = None, 
                         mode: str = "generation", round_number: Optional[int] = None) -> str:
    """
    Generate the output image path

    Args:
        model_config: model configuration dictionary
        index: data item index (optional)
        mode: mode (generation/editing)
        round_number: round number (optional, used for multi-turn dialogue scenarios)

    Returns:
        output image path
    """
    output_dir = model_config.get("output_image_dir", "result/images")
    model_name = model_config.get("model_name", "bagel")
    task_id = model_config.get("task_id", "unknown")
    
    # Sanitize the model name for use in paths
    safe_model_name = model_name.replace("/", "_").replace(" ", "_")

    # If output_dir contains a {model_name} placeholder, perform string substitution
    if "{model_name}" in output_dir:
        output_dir = output_dir.format(model_name=safe_model_name)

    # Build the full path
    full_output_dir = os.path.join(output_dir, task_id)
    os.makedirs(full_output_dir, exist_ok=True)

    # Generate the filename
    if index is not None:
        if round_number is not None:
            filename = f"{index:06d}_round{round_number}_{mode}.png"
        else:
            filename = f"{index:06d}_{mode}.png"
    else:
        import time
        if round_number is not None:
            filename = f"{int(time.time())}_round{round_number}_{mode}.png"
        else:
            filename = f"{int(time.time())}_{mode}.png"
    
    return os.path.join(full_output_dir, filename)


def parse_image_shapes(image_shapes_config: Any) -> tuple:
    """
    Parse the image dimensions configuration

    Args:
        image_shapes_config: may be a string "(1024, 1024)", a list [1024, 1024], or a tuple

    Returns:
        image dimensions tuple (H, W)
    """
    if isinstance(image_shapes_config, str):
        # Try parsing the string format "(1024, 1024)"
        import ast
        try:
            shapes = ast.literal_eval(image_shapes_config)
            if isinstance(shapes, (list, tuple)) and len(shapes) == 2:
                return tuple(shapes)
        except:
            pass
        # Try parsing the ratio format "1:1"
        if ":" in image_shapes_config:
            ratio_map = {
                "1:1": (512, 512),
                "4:3": (384, 512),
                "3:4": (512, 384),
                "16:9": (288, 512),
                "9:16": (512, 288),
            }
            return ratio_map.get(image_shapes_config, (512, 512))
    elif isinstance(image_shapes_config, (list, tuple)) and len(image_shapes_config) == 2:
        return tuple(image_shapes_config)
    
    return (512, 512)  # default value


def bagel_inference_function(image_paths: Optional[Union[str, List[str]]], prompt: str, 
                            model_config: Dict[str, Any], index: Optional[int] = None, 
                            round_number: Optional[int] = None) -> str:
    """
    Bagel model inference function (supports understanding/generation/editing)

    Args:
        image_paths: image path (string) or list of image paths (required for understanding/editing modes; may be None for generation mode)
        prompt: text prompt
        model_config: model configuration dictionary
        index: data item index (used to generate the output path)
        round_number: round number (optional, used for multi-turn dialogue scenarios)

    Returns:
        understanding mode: returns a text response
        generation/editing mode: returns an image path
    """
    # 1. Set up the Bagel project path
    bagel_project_path = model_config.get("bagel_project_path", "../../Unified_Models/Bagel")
    try:
        setup_bagel_path(bagel_project_path)
    except ValueError as e:
        error_msg = (
            f"Failed to setup Bagel project path: {e}\n"
            f"Please check the 'bagel_project_path' in your model config file.\n"
            f"Current path: {bagel_project_path}\n"
            f"Tip: You can set 'bagel_project_path' in the model config JSON file to the correct path."
        )
        raise ValueError(error_msg) from e
    
    # 2. Get configuration parameters
    model_path = model_config.get("model_path")
    if not model_path:
        raise ValueError("model_path not found in config")
    
    mode = model_config.get("mode", 1)
    gpu_id = model_config.get("gpu_id", 0)
    
    # 3. Check whether the model is already loaded (use the cache)
    cache_key = f"{model_path}_{mode}_{gpu_id}"

    if cache_key in _failed_inferencers:
        raise _failed_inferencers[cache_key]

    if cache_key not in _loaded_inferencers:
        print(f"Loading Bagel model: {model_path} (mode={mode}, gpu_id={gpu_id})")
        torch.cuda.set_device(gpu_id)
        try:
            inferencer = load_model_and_inferencer(model_path, mode, gpu_id)
        except Exception as e:
            _failed_inferencers[cache_key] = e
            raise
        _loaded_inferencers[cache_key] = inferencer
    else:
        inferencer = _loaded_inferencers[cache_key]
    
    # 4. Get the inference mode (prefer operation_type from the task config; otherwise use inference_mode from the model config)
    inference_mode = model_config.get("inference_mode", "understanding")
    # If inference_mode is explicitly specified in the model config, use it (the task config sets it via ModelInference.inference())

    # 5. Get common inference parameters
    seed = model_config.get("seed", 666)
    do_sample = model_config.get("do_sample", False)
    text_temperature = model_config.get("temperature", model_config.get("text_temperature", 0.3))
    
    # 6. The prompt has already been formatted in the task config, so use it directly
    formatted_prompt = prompt

    # 7. Run inference according to the mode
    try:
        if inference_mode == "understanding":
            # Image understanding mode; falls back to text-only understanding when image_paths is empty (e.g., GGU Complex_Relation)
            max_new_tokens = model_config.get("max_new_tokens", 8192)
            response = image_understanding(
                inferencer=inferencer,
                image_paths=image_paths,
                prompt=formatted_prompt,
                seed=seed,
                do_sample=do_sample,
                text_temperature=text_temperature,
                max_new_tokens=max_new_tokens
            )
            return response
            
        elif inference_mode == "generation":
            # Image generation mode
            output_path = generate_output_path(model_config, index, "generation", round_number)

            # Get generation-related parameters
            image_shapes_config = model_config.get("image_shapes", [512, 512])
            image_shapes = parse_image_shapes(image_shapes_config)
            cfg_text_scale = model_config.get("cfg_text_scale", 4.0)
            cfg_interval = model_config.get("cfg_interval", [0.4, 1.0])
            num_timesteps = model_config.get("num_timesteps", 50)
            timestep_shift = model_config.get("timestep_shift", 3.0)
            cfg_renorm_min = model_config.get("cfg_renorm_min", 0.0)
            cfg_renorm_type = model_config.get("cfg_renorm_type", "global")
            max_think_tokens = model_config.get("max_think_tokens", 8192)
            
            response = image_generation(
                inferencer=inferencer,
                prompt=formatted_prompt,
                output_path=output_path,
                image_shapes=image_shapes,
                cfg_text_scale=cfg_text_scale,
                cfg_interval=cfg_interval,
                num_timesteps=num_timesteps,
                timestep_shift=timestep_shift,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                seed=seed,
                do_sample=do_sample,
                text_temperature=text_temperature,
                max_think_tokens=max_think_tokens
            )
            return response
            
        elif inference_mode == "editing":
            # Image editing mode
            if not image_paths:
                raise ValueError("image_paths is required for editing mode")

            output_path = generate_output_path(model_config, index, "editing", round_number)

            # Get editing-related parameters
            cfg_text_scale = model_config.get("cfg_text_scale", 4.0)
            cfg_img_scale = model_config.get("cfg_img_scale", 1.5)
            cfg_interval = model_config.get("cfg_interval", [0.4, 1.0])
            num_timesteps = model_config.get("num_timesteps", 50)
            timestep_shift = model_config.get("timestep_shift", 3.0)
            cfg_renorm_min = model_config.get("cfg_renorm_min", 0.0)
            cfg_renorm_type = model_config.get("cfg_renorm_type", "global")
            max_think_tokens = model_config.get("max_think_tokens", 8192)

            response = image_editing(
                inferencer=inferencer,
                image_paths=image_paths,  # Pass the list of image paths; the function uses the first one (the original)
                prompt=formatted_prompt,
                output_path=output_path,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                num_timesteps=num_timesteps,
                timestep_shift=timestep_shift,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                seed=seed,
                do_sample=do_sample,
                text_temperature=text_temperature,
                max_think_tokens=max_think_tokens
            )
            return response
            
        elif inference_mode == "unify":
            # Unify mode (similar to editing mode: requires an input image and outputs the edited image)
            if not image_paths:
                raise ValueError("image_paths is required for unify mode")

            output_path = generate_output_path(model_config, index, "unify", round_number)

            # Get editing-related parameters (unify mode uses the same parameters as editing)
            cfg_text_scale = model_config.get("cfg_text_scale", 4.0)
            cfg_img_scale = model_config.get("cfg_img_scale", 1.5)
            cfg_interval = model_config.get("cfg_interval", [0.4, 1.0])
            num_timesteps = model_config.get("num_timesteps", 50)
            timestep_shift = model_config.get("timestep_shift", 3.0)
            cfg_renorm_min = model_config.get("cfg_renorm_min", 0.0)
            cfg_renorm_type = model_config.get("cfg_renorm_type", "global")
            max_think_tokens = model_config.get("max_think_tokens", 8192)
            
            response = image_editing(
                inferencer=inferencer,
                image_paths=image_paths,  # Pass the list of image paths; the function uses the first one (the original)
                prompt=formatted_prompt,
                output_path=output_path,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                num_timesteps=num_timesteps,
                timestep_shift=timestep_shift,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                seed=seed,
                do_sample=do_sample,
                text_temperature=text_temperature,
                max_think_tokens=max_think_tokens
            )
            return response

        else:
            raise ValueError(f"Unknown inference_mode: {inference_mode}. "
                           f"Supported modes: understanding, generation, editing, unify")
            
    except Exception as e:
        print(f"Error during Bagel inference ({inference_mode} mode): {e}")
        raise
