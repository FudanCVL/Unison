# Summary of Inference Formats and Logic for the Four Tasks

---

## Task 1: IC (Internal Consistency)

### Data Sources

| File | Content |
|------|------|
| `prompts.txt` | Text descriptions, one per line (556 total) |
| `questions.json` | `{"questions": {"prompt_0": {"1": "question?", ...}, "prompt_1": {...}}}` |
| `images/N.png` | Reference image for each prompt (1.png, 2.png, ...) |

### A Single Loaded data_item

```python
{
    "index": 0,
    "image_path": "data/Internal_Consistency/images/1.png",
    "prompt": "a bright orange rectangular dining table...",
    "questions": {"1": "Is there a dining table?", "2": "Is there a fire hydrant?", ...}
}
```

### Inference Flow (2 operations)

| Order | operation_type | Image input | Prompt | Output |
|------|---------------|---------|--------|------|
| 1 | `understanding` | Original reference image | `"{question} Answer ONLY yes or no."` **called per question**, one result row per question | "yes" / "no" text |
| 2 | `generation` | No image | `"{prompt}"` (the description from prompts.txt) | Generated image path |

Special columns in the result CSV: `question_id`, `question_text` (one row per question)

### Logic

First record the model's baseline VQA answers on the reference image, then have the model generate an image. During evaluation (`evaluate_ic.py`), Qwen3-VL is asked the same questions about the generated image, and the consistency between the two sets of answers is compared.

---

## Task 2: UGG (Understanding Guided Generation)

### Data Sources

`UGG.csv` (80 entries)

### A Single Loaded data_item

```python
{
    "index": 0,
    "image_path": "data/Und_Guided_Gen/images/COCO_train2014_000000162375.jpg",
    "object": "chair",
    "instruction": "Alter the color of the wooden chair... to dark yellow.",
    "operation": "chair | change color to dark yellow",
    "bbox": "[459.4, 10.58, 639.81, 280.09]",
    "mask": "[[...]]"
}
```

### Inference Flow (3 operations)


#### Operation 1 (understanding) - Understanding

| Image input | Prompt | Output |
|---------|--------|------|
| Original image | `"Based on the image editing instruction: {instruction}, please output the target object location. Output only the bbox in the format [x_min, y_min, x_max, y_max]"` | bbox [x_min, y_min, x_max, y_max] |


#### Operation 2 (editing) - Generation

bbox uses the human-annotated one
| Image input | Prompt | Output |
|---------|--------|------|
| Original image | `"Based on the target object location: {bbox} and instruction: {instruction}, please perform image editing task."` | Edited image path |

#### Operation 3 (unify) - unify

bbox uses the one the model itself output in operation 0
| Image input | Prompt | Output |
|---------|--------|------|
| Original image | `"Based on the target object location: {bbox} and instruction: {instruction}, please perform image editing task."` | Edited image path |

Score: average of the scores from operation 1 and operation 2



---

## Task 3: GGU (Generation Guided Understanding)

### Data Sources

Merged from three sub-task JSON files (542 entries total)

| Sub-task | File | Count | Has original image |
|--------|------|------|---------|
| `2D_Spatial` | `2d_spatial.json` | 182 | ✓ under the `matrices/` directory |
| `3D_Spatial` | `spatial.json` | 180 | ✓ under the `cubes/` directory |
| `Complex_Relation` | `complex_relation.json` | 180 | ✗ (no `image_path` field) |

### A Single Loaded data_item

```python
{
    "index": 0,
    "image_path": "data/Gen_Guided_Und/2D_Spatial/matrices/matrix_3x3_01.jpg",
    "category": "2d_spatial",   # or "3d_spatial" / "complex_relation"
    "question": "Given a matrix, where each square contains a number...",
    "options": {"A": "(2,1)", "B": "...", ...},
    "answer": "B",
    "description": "Based on the given checkerboard matrix image, generate the rotated...",
    "image_generation_validate": {"1": "Is there a matrix?", ...}  # Complex_Relation only
}
```

### Inference Flow (3 operations, non-dialogue)

| Order | operation_type | Image input | Prompt | Output |
|------|---------------|---------|--------|------|
| 1 | `understanding` | Original image (empty for Complex_Relation) | `"{question}\n\nOptions:\n{A: ...}\n{B: ...}...  Answer ONLY option"` | Option text (e.g. "B") |
| 2 | `generation` | No image | `"{description}"` | Generated image path |
| 3 | `unify` | [Original image + image generated in operation 2] (Complex_Relation has only the generated image) | `"The second image is a reference generated based on the spatial requirements of the question. Reason both the original scene image and this reference image, answer the following question:\n\n{question}\nOptions:\n...  Answer ONLY option"` | Option text |

**Special case (Complex_Relation)**: no original image, so `unify` passes only the generated image; for `understanding`, processing continues even without an image (pure text reasoning).



---

## Task 4: ME (Mutual Enhancement)

### Data Sources

`ME.csv` (55 entries)

### A Single Loaded data_item

```python
{
    "index": 0,
    "image_path": "data/Mutual_Enhancement/dsg/data/laion_images/3620.png",
    "operation": "1: Add, A small white dog...; 2: Remove, The person in the white dress...; 3: Alter, Change the color...",
    "instruction": "Add a small white dog..., remove the person..., and change the color of the table to red.",
    "final_caption": "Under a rustic canopy... a small white dog sits calmly..."
}
```

### Inference Flow (2 operations, both multi-turn dialogues, up to 10 rounds, may terminate early)

---

#### Operation 0 (unify / me_u2g - understanding evaluation guiding the editing iteration)

Each iteration = an editing round + an evaluation round, up to 5 iterations (one iteration means one editing plus one understanding):

| Round | op_type | Image input | Prompt logic | Output |
|------|---------|---------|------------|------|
| 1 (editing) | `editing` | Original image | `"{instruction}"` | Edited image 1 path |
| 2 (understanding) | `understanding` | Original image + edited image 1 | Compare the original and edited images, judge whether every editing operation satisfies the instruction: "Yes" or "No, [new instruction]" (if not satisfied, have the model output in the format No, [new instruction]) | Text |
| 3 (editing) | `editing` | Edited image 1 | Use `{next_instruction}` (the correction instruction extracted from round 2) | Edited image 2 path |
| 4 (understanding) | `understanding` | Original image + edited image 2 | Same evaluation as above, still using the instruction from round 1 for the model's judgment | Text |
| ... | ... | ... | ... | ... |

**Termination conditions**: the understanding round outputs "Yes" / a malformed output / outputs "No" but with no follow-up instruction.

**`{next_instruction}` extraction logic**: `parse_evaluation_output()` parses "No, [instruction]" → extracts the latter part as the editing instruction for the next round.

---

#### Operation 1 (unify / me_g2u - generation/editing guiding the understanding comparison iteration)

Each iteration = an understanding round + an editing round, up to 5 iterations:

| Round | op_type | Image input | Prompt logic | Output |
|------|---------|---------|------------|------|
| 1 (understanding) | `understanding` | Original image | Compare the image with `final_caption`, output "Yes" or "No, 1.[issue 1]:[edit instruction 1], 2.[issue 2]:[edit instruction 2]..." | Text |
| 2 (editing) | `editing` | Original image | `"Based on the instruction: {combined_instructions}, please perform image editing task."` | Edited image 1 path |
| 3 (understanding) | `understanding` | Edited image 1 | Compare edited image 1 with `final_caption` | Text |
| 4 (editing) | `editing` | Edited image 1 | Use the latest `{combined_instructions}` | Edited image 2 path |
| ... | ... | ... | ... | ... |

**Termination conditions**: same as u2g.

**`{combined_instructions}` extraction logic**: `extract_combined_instructions()` extracts and merges all editing instructions from "No, 1.[m1]:[e1], 2.[m2]:[e2]...".

---

## Key Notes

1. **IC's understanding answers questions about the reference image**, not the generated image; the VQA on the generated image is run separately by `evaluate_ic.py`.
2. **UGG's two operations are fully independent and parallel** (operation_index 0 and 1 both process the same batch of data).
3. **GGU Complex_Relation has no original image**, so the understanding operation uses image-free pure text reasoning, and unify passes only the generated image.
4. **ME's early-termination mechanism**: if any understanding round outputs "Yes" or has a format error, the dialogue for the current operation stops.
5. **ME's two operations (u2g and g2u) are independent**, both run on the same batch of data, and results are distinguished by `operation_index`.
