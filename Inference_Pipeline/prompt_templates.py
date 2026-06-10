#!/usr/bin/env python3
"""
Prompt template function definitions
Supports defining prompt templates using full Python syntax
"""
from typing import Dict, Any, List, Union


def ic_understanding_template(data_item: Dict[str, Any]) -> List[str]:
    """
    IC task understanding template (backward compatible, returns a list of strings)
    """
    # image_path = data_item.get("image_path", "")
    questions = data_item.get("questions", {})

    prompts = []
    for q_id, question in questions.items():
        # Build a separate prompt for each question
        single_prompt = f"""{question} Answer ONLY yes or no.""".strip()
        prompts.append(single_prompt)
    
    return prompts

def ic_generation_template(data_item: Dict[str, Any]) -> str:
    """
    IC task generation template

    Args:
        data_item: data item containing fields such as image_path

    Returns:
        the formatted prompt string
    """
    prompt = data_item.get("prompt", "")
    return f"""{prompt}""".strip()



def ugg_understanding_template(data_item: Dict[str, Any]) -> str:
    instruction = data_item.get("instruction", "")
    return (
        f"Based on the image editing instruction: {instruction}, "
        "please output the target object location. "
        "Output only the bbox in the format [x_min, y_min, x_max, y_max]"
    )


def ugg_1_round_template(data_item: Dict[str, Any]) -> str:
    instruction = data_item.get("instruction", "")
    bbox = data_item.get("bbox", "")
    return f"Based on the target object location: {bbox} and instruction: {instruction}, please perform image editing task."


def ugg_unify_template(data_item: Dict[str, Any]) -> str:
    instruction = data_item.get("instruction", "")
    bbox = data_item.get("bbox", "")
    return f"Based on the target object location: {bbox} and instruction: {instruction}, please perform image editing task."



def ggu_understanding_template(data_item: Dict[str, Any]) -> str:
    question = data_item.get("question", "")
    options = data_item.get("options", {})
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    return f"""{question}\n\nOptions:\n{options_text}\nAnswer ONLY option""".strip()


def ggu_generation_template(data_item: Dict[str, Any]) -> str:
    description = data_item.get("description", "")
    return f"{description}".strip()


def ggu_unify_template(data_item: Dict[str, Any]) -> str:
    question = data_item.get("question", "")
    options = data_item.get("options", {})
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    return (
        f"The second image is a reference generated based on the spatial requirements of the question. "
        f"Reason both the original scene image and this reference image, answer the following question:\n\n"
        f"{question}\nOptions:\n{options_text}\nAnswer ONLY option"
    )


def me_u2g_template(data_item: Dict[str, Any], max_iterations: int = 5) -> List[Dict[str, Any]]:
    """
    ME task understanding-guided generation template (up to max_iterations iterations, max_iterations*2 dialogue rounds total)
    Each iteration contains: an editing task (odd rounds) and an evaluation task (even rounds)

    Args:
        data_item: data item containing fields such as instruction
        max_iterations: number of iterations; each iteration has 2 rounds (editing + evaluation), value 1-5 (default 5, i.e. up to 10 rounds)

    Returns:
        a list of dialogue rounds, each element defining one round
    """
    instruction = data_item.get("instruction", "")
    max_iterations = max(1, min(5, int(max_iterations)))
    rounds = []

    # Evaluation template
    self_refine_template = "{original_image}Based on the original image, edited image{edited_image} and instruction: {}\n Carefully compare the original image and the edited image. You must check each operation mentioned in the instruction one by one to verify if it is perfectly and completely satisfied in the edited image.\n If ALL operations are perfectly satisfied (no missing, incomplete, or incorrect parts), output 'Yes'.\n If ANY operation is not perfectly satisfied, incomplete, or incorrectly executed, you MUST output 'No, [edit instruction]' describing what needs to be fixed based on the unsatisfied part(s)."

    for i in range(max_iterations):
        # Editing round (odd rounds: 1, 3, 5, 7, 9)
        round_num = i * 2 + 1
        if i == 0:
            # Round 1 uses the original image and the original instruction (use the instruction text directly)
            editing_prompt = "{original_image}" + instruction
        else:
            # Other rounds use the previous round's edited image, via the {next_instruction} placeholder
            prev_editing_round = (i - 1) * 2 + 1  # round number of the previous editing round
            editing_prompt = f"{{round_{prev_editing_round}_image}}" + "{next_instruction}"
        
        rounds.append({
            "round_number": round_num,
            "prompt_template": editing_prompt,
            "operation_type": "editing"
        })
        
        # Evaluation round (even rounds: 2, 4, 6, 8, 10)
        round_num += 1
        # Includes placeholders for both the original image and the edited image
        # {original_image} is replaced with the original image path in replace_dialogue_placeholders
        # {edited_image} is replaced with {round_{round_num-1}_image}, then replaced with the edited image path in replace_dialogue_placeholders
        edited_image_placeholder = f"{{round_{round_num-1}_image}}"
        # First replace the {edited_image} placeholder, then the instruction placeholder (the last {})
        temp_template = self_refine_template.replace("{edited_image}", edited_image_placeholder)
        # Find the last {} (the instruction placeholder) and replace it
        last_brace_idx = temp_template.rfind("{}")
        if last_brace_idx != -1:
            self_refine_prompt = temp_template[:last_brace_idx] + instruction + temp_template[last_brace_idx+2:]
        else:
            self_refine_prompt = temp_template
        rounds.append({
            "round_number": round_num,
            "prompt_template": self_refine_prompt,
            "operation_type": "understanding"
        })
    
    return rounds


def me_g2u_template(data_item: Dict[str, Any], max_iterations: int = 5) -> List[Dict[str, Any]]:
    """
    ME task generation-guided understanding template (up to max_iterations iterations, max_iterations*2 dialogue rounds total)
    Each iteration contains: an understanding task (odd rounds) and an editing task (even rounds)
    Task: identify where the image and caption are inconsistent, then edit the image based on those inconsistencies

    Args:
        data_item: data item containing fields such as final_caption
        max_iterations: number of iterations; each iteration has 2 rounds (understanding + editing), value 1-5 (default 5, i.e. up to 10 rounds)

    Returns:
        a list of dialogue rounds, each element defining one round
    """
    final_caption = data_item.get("final_caption", "")
    max_iterations = max(1, min(5, int(max_iterations)))
    rounds = []

    # Understanding and editing templates
    understanding_template = "{}Based on the image and caption: {}\n Identify any mismatches between the image and caption. Output 'Yes' if there are no mismatches. Output 'No, 1.[mismatch1]:[edit instruction1], 2.[mismatch2]:[edit instruction2],...' If there are mismatches."
    editing_template = "Based on the instruction: {combined_instructions}, please perform image editing task."

    for i in range(max_iterations):
        # Understanding round (odd rounds: 1, 3, 5, 7, 9)
        round_num = i * 2 + 1
        if i == 0:
            # Round 1 uses the original image (no image placeholder specified; the system uses the original image_path)
            understanding_prompt = understanding_template.format("", final_caption)
        else:
            # Other rounds use the previous round's edited image
            understanding_prompt = understanding_template.format(f"{{round_{round_num-1}_image}}", final_caption)
        
        rounds.append({
            "round_number": round_num,
            "prompt_template": understanding_prompt,
            "operation_type": "understanding"
        })
        
        # Editing round (even rounds: 2, 4, 6, 8, 10)
        round_num += 1
        if i == 0:
            # The first editing round (round 2) uses the original image, no placeholder
            editing_prompt = editing_template
        else:
            # Other editing rounds use the previous round's edited image (round_num-2 is the previous editing round)
            editing_prompt = f"{{round_{round_num-2}_image}}" + editing_template
        rounds.append({
            "round_number": round_num,
            "prompt_template": editing_prompt,
            "operation_type": "editing"
        })
    
    return rounds
