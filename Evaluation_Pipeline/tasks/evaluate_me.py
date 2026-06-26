"""ME (Mutual Enhancement) task evaluation."""

import csv
import difflib
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from common.io import load_csv, success_rows, load_me_gt, resolve_path
from common.normalize import parse_image_path
from common.aggregate import clip, build_task_result, null_task_result
from common.judge import ClosedSourceJudge


def _append_row(path: str, row: dict):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _last_valid_response(rows_by_round: dict, odd_only: bool = False) -> tuple:
    """
    Given {round_number: model_response}, return (round_num, response) for
    the last round with a non-null response.  If odd_only=True, only consider
    odd rounds (editing rounds in me_u2g or understanding rounds in me_g2u).
    Returns (None, None) if nothing found.
    """
    rounds = sorted(rows_by_round.keys())
    if odd_only:
        rounds = [r for r in rounds if r % 2 == 1]
    for r in reversed(rounds):
        resp = rows_by_round.get(r, "")
        if resp and str(resp).lower() not in ("nan", "none", ""):
            return r, resp
    return None, None


def _strip_no_prefix(text: str) -> str:
    """Drop a leading 'No,' / 'No.' prefix from a g2u understanding response.

    Mirrors the inference-side extract_combined_instructions: the model answers
    "No, 1.[mismatch]:[edit instruction], ..." — only the listed mismatches are
    the model's actual understanding output.
    """
    t = (text or "").strip()
    if re.match(r"no\b", t.lower()):
        rest = t[2:].strip()
        while rest and rest[0] in ".,:;":
            rest = rest[1:].strip()
        return rest
    return t


# Numbered-list marker: "1.", "2)", "1.[", "1.text" — a 1-2 digit ordinal (not preceded
# by a digit, so it can't be the tail of a decimal/bbox number), optional space, then '.'
# or ')', NOT followed by another digit. The "not a digit" lookahead is what keeps "3.5"
# and bbox coords like "[217, 58, 482, 173]" out while still matching an enumerator that
# runs straight into its text with no space (e.g. OmniGen2's "1.jester hats:...").
_ENUM_MARKER_PAT = r"(?<!\d)\b\d{1,2}\s*[.\)](?!\d)"
_ENUM_MARKER = re.compile(_ENUM_MARKER_PAT)


def _split_mismatch_items(text: str) -> list:
    """Split one g2u understanding response into discrete mismatch items.

    The model answers "No, 1.[mismatch]:[edit instruction], 2.[mismatch]:[...], ..."
    (the trailing :[edit] is optional). We split before each numbered "N." marker —
    whether or not it is followed by a '[' bracket — so each reported mismatch becomes
    one item; the leading "N." enumerator is stripped but the rest (bracketed or plain
    mismatch+edit text) is kept (it all describes one discrepancy). Any preamble before
    the first enumerator (e.g. "The mismatched elements are:") is discarded so it does
    not count as a spurious predicted item.

    Earlier this only split the bracketed "N.[" form, so models that enumerate as
    "No, 1. … 2. …" (no brackets, e.g. OmniGen2) had all their mismatches collapsed
    into a single item — under-counting predictions and inconsistently inflating
    precision vs. bracket-format models. The marker now matches both forms.

    Falls back to ';'-splitting / the whole string when the response isn't enumerated.
    'No' / 'Yes' / empty responses yield []. Items shorter than 3 chars are dropped.
    """
    if str(text or "").strip().lower() in ("nan", "none", ""):
        return []
    t = _strip_no_prefix(str(text))
    if t.strip().lower() in ("yes", ""):
        return []

    m = _ENUM_MARKER.search(t)
    if m:
        # Enumerated "1. … 2. …" / "1.[…] 2.[…]" format: drop any preamble before the
        # first marker, then split before each numbered marker.
        t = t[m.start():]
        items = []
        for part in re.split(rf"(?={_ENUM_MARKER_PAT})", t):
            part = part.strip().strip(",;").strip()
            part = re.sub(r"^\d{1,2}\s*[.\)]\s*", "", part).strip()  # drop "1." / "2)" enumerator
            if len(part) >= 3:
                items.append(part)
        return items

    # Free-form (no enumeration): split on ';' if it yields multiple clauses, else one
    # item. We deliberately do NOT comma-split here — natural-language prose uses commas
    # mid-clause, so comma-splitting would shatter a single mismatch into fragments.
    alt = [s.strip() for s in re.split(r";\s*", t) if len(s.strip()) >= 3]
    if len(alt) > 1:
        return alt
    return [t.strip()] if len(t.strip()) >= 3 else []


# Boilerplate words shared by almost every mismatch sentence. They are stripped before
# the word-overlap dedup so two phrasings of the *same* discrepancy are compared on their
# distinctive content (objects, attributes, positions: "flower crown", "left", "red
# shirt") rather than the scaffolding ("the caption mentions ... but the image shows").
# Deliberately keeps position/object/attribute words (left/right/center/group/colour/etc.).
_DEDUP_STOP = frozenset("""
a an the this that these those is are was were be been being has have had do does did done
in on of to and or but not no with without for from at as by it its he she they them him
her his their our your my me we you i image images picture photo caption text scene
mention mentions mentioned describe describes described state states stated say says
show shows shown showing depict depicts depicted appear appears appeared seen visible
present still now there here which while instead however whereas wearing wears wear worn
has have one some any only both each all more less than then so such very much many few
mismatch mismatches discrepancy discrepancies difference differences between also being
""".split())


def _dedup_key_tokens(s: str) -> frozenset:
    """Distinctive content words of a mismatch string (lowercased alpha tokens ≥3 chars,
    minus the boilerplate above), used for the Jaccard arm of the dedup."""
    return frozenset(w for w in re.findall(r"[a-z]+", s.lower())
                     if len(w) >= 3 and w not in _DEDUP_STOP)


def _same_mismatch(a_lower: str, a_tokens: frozenset, b_lower: str, b_tokens: frozenset,
                   ratio_threshold: float, jaccard_threshold: float) -> bool:
    """True if two mismatch strings describe the same discrepancy. Two signals:
      • near-verbatim character similarity (difflib > ratio_threshold) — catches a round
        copying the previous round almost word-for-word; and
      • content-word overlap (Jaccard ≥ jaccard_threshold) — catches the *re-wordings*
        the char ratio misses (e.g. "the woman on the left is wearing a flower crown" vs
        "the caption describes the woman on the left as wearing a flower crown").
    """
    if difflib.SequenceMatcher(None, a_lower, b_lower).ratio() > ratio_threshold:
        return True
    if a_tokens and b_tokens:
        inter = len(a_tokens & b_tokens)
        if inter and inter / len(a_tokens | b_tokens) >= jaccard_threshold:
            return True
    return False


def _collect_mismatch_items(rows_by_round: dict, sim_threshold: float = 0.85,
                            jaccard_threshold: float = 0.6) -> list:
    """Union of discrete mismatch items across ALL understanding rounds (odd: 1,3,5,7,9),
    deduplicated by both character similarity and content-word overlap.

    In me_g2u the model re-examines the (progressively edited) image each understanding
    round and reports the mismatches it still sees, so one discrepancy is typically
    restated every round in slightly different words. Taking the union keeps a mismatch
    the model only voiced in a later round (recall), but the restatements MUST be merged
    or they inflate the predicted-item count and crush precision. The old dedup used
    character similarity alone (ratio > 0.85), which left cross-round re-wordings
    unmerged (~8-9 items/sample); adding the content-word Jaccard arm collapses those
    re-wordings while keeping genuinely distinct mismatches (crown / shirt / dress)
    separate. The first occurrence (earliest round) is the one kept.

    Note: the union still includes mismatches that only appear in later rounds because an
    edit newly broke something on the *edited* image, so n_pred stays above the GT op
    count for models that never converge to 'Yes' — that is inherent to scoring every
    round. Lower jaccard_threshold to merge more aggressively.
    """
    out: list = []
    out_lower: list = []
    out_tokens: list = []
    for r in sorted(rows_by_round.keys()):
        if r % 2 != 1:
            continue
        resp = rows_by_round.get(r, "")
        if not resp or str(resp).strip().lower() in ("nan", "none", ""):
            continue
        for item in _split_mismatch_items(str(resp)):
            key = item.lower()
            toks = _dedup_key_tokens(item)
            if any(_same_mismatch(key, toks, s, st, sim_threshold, jaccard_threshold)
                   for s, st in zip(out_lower, out_tokens)):
                continue
            out.append(item)
            out_lower.append(key)
            out_tokens.append(toks)
    return out



def evaluate_me(
    csv_path: str,
    data_dir: str,
    judge: ClosedSourceJudge,
    inference_base_dir: str,
    max_workers: int = 8,
    output_csv: str = None,
) -> dict:
    """
    Evaluate ME task.

    CSV operation_index mapping:
      0 = me_u2g (self-refining image editing)
          odd rounds  = editing   → model_response = edited image path
          even rounds = evaluation → model_response = 'Yes' / 'No, ...'
      1 = me_g2u (self-refining multimodal understanding)
          odd rounds  = understanding → model_response = mismatch text
          even rounds = editing        → model_response = edited image path

    Scores per sample:
      generation_item    = judge.rate_edit(orig, final_edit_u2g, instruction)
      understanding_item = judge.score_mismatch_prf(pred_mismatch_items, operation).f1
      unified_item       = (generation_item + understanding_item) / 2

    The understanding score is a soft precision/recall/F1 over the model's discrete
    mismatch items vs the GT sub-operations (see judge.score_mismatch_prf): recall
    rewards covering every GT op, precision penalises hallucinated/extra mismatches, and
    F1 (the headline number) balances the two. Precision and recall are also written to
    the per-sample CSV as diagnostics.
    """
    if not os.path.exists(csv_path):
        print(f"[ME] CSV not found: {csv_path}")
        return null_task_result("ME")

    df = load_csv(csv_path)
    sdf = success_rows(df)

    # Load GT
    gt: dict = {}
    if data_dir and os.path.isdir(data_dir):
        try:
            gt = load_me_gt(data_dir)
        except Exception as e:
            print(f"[ME] Warning: could not load GT data: {e}")

    import json

    # u2g_rounds[idx][round_num] = model_response
    # g2u_rounds[idx][round_num] = model_response
    u2g_rounds: dict = {}
    g2u_rounds: dict = {}
    input_data_cache: dict = {}

    for _, row in sdf.iterrows():
        idx = int(row["dialogue_index"])
        op_idx_raw = row.get("operation_index", None)
        op_idx = int(op_idx_raw) if op_idx_raw is not None else -1
        round_raw = row.get("round_number", None)
        resp = str(row.get("model_response", "") or "")

        if idx not in input_data_cache:
            try:
                input_data_cache[idx] = json.loads(str(row.get("input_data", "{}")))
            except Exception:
                input_data_cache[idx] = {}

        # Determine round_number
        try:
            round_num = int(float(round_raw)) if round_raw is not None and str(round_raw).lower() not in ("nan", "none", "") else None
        except (TypeError, ValueError):
            round_num = None

        if op_idx == 0:  # me_u2g
            u2g_rounds.setdefault(idx, {})
            if round_num is not None:
                u2g_rounds[idx][round_num] = resp
            else:
                # Flat result (model didn't do multi-round): treat as round 1
                u2g_rounds[idx][1] = resp

        elif op_idx == 1:  # me_g2u
            g2u_rounds.setdefault(idx, {})
            if round_num is not None:
                g2u_rounds[idx][round_num] = resp
            else:
                g2u_rounds[idx][1] = resp

    all_indices = set(input_data_cache)
    if not all_indices:
        print("[ME] No data found in CSV.")
        return null_task_result("ME")

    # --- Collect judge calls ---
    # (a) rate_edit for u2g: (orig, final_edit_image, instruction, operation)
    # (b) score_mismatch_prf for g2u: (predicted_items, gt_operation)
    rate_edit_tasks: dict = {}     # idx -> (orig, edited, instruction, operation)
    mismatch_tasks: dict = {}       # idx -> (predicted_items, gt_operation)

    for idx in sorted(all_indices):
        idata = input_data_cache.get(idx, {})
        gt_item = gt.get(idx, {})

        orig_raw = idata.get("image_path", "") or gt_item.get("image_path", "")
        orig_path = resolve_path(orig_raw, inference_base_dir)
        if not os.path.exists(orig_path):
            orig_path = gt_item.get("image_path", "")

        instruction = idata.get("instruction", "") or gt_item.get("instruction", "")
        # GT operation has bboxes normalized to [0, 1000] by load_me_gt; GT takes priority
        operation = gt_item.get("operation", "") or idata.get("operation", "")

        # u2g: last valid odd round (editing rounds: 1, 3, 5, 7, 9)
        u2g_data = u2g_rounds.get(idx, {})
        _, final_edit_resp = _last_valid_response(u2g_data, odd_only=True)
        if final_edit_resp:
            img = parse_image_path(final_edit_resp)
            if img:
                edit_path = resolve_path(img, inference_base_dir)
                if orig_path and os.path.exists(orig_path) and os.path.exists(edit_path):
                    rate_edit_tasks[idx] = (orig_path, edit_path, instruction, operation or None)

        # g2u: discrete mismatch items, union across ALL understanding rounds
        # (1, 3, 5, 7, 9), deduped by char-similarity + content-word overlap so
        # cross-round restatements don't inflate the count (see _collect_mismatch_items).
        # score_mismatch_prf then scores each predicted item against each GT sub-op
        # (P×G matrix) → soft precision/recall/F1, so genuine over-reporting is penalised
        # (precision) while recall still credits a mismatch first voiced in a later round.
        g2u_data = g2u_rounds.get(idx, {})
        pred_items = _collect_mismatch_items(g2u_data)
        if pred_items and operation:
            mismatch_tasks[idx] = (pred_items, operation)

    # Failure visibility: which samples produced no usable output on each axis. They
    # still score 0 (folded into the means below), but we surface the counts instead of
    # letting num_failed silently read 0. A sample can fail one axis and pass the other:
    #   no_valid_final_edit → no parsable final edited image in u2g          → generation_item = 0
    #   no_parsed_mismatch  → no mismatch parsed from any g2u understand round → understanding_item = 0
    no_valid_edit = sorted(set(all_indices) - set(rate_edit_tasks))
    no_parsed_mismatch = sorted(set(all_indices) - set(mismatch_tasks))

    total_judge = len(rate_edit_tasks) + len(mismatch_tasks)
    print(f"[ME] {len(all_indices)} samples, {total_judge} judge calls queued "
          f"({len(no_valid_edit)} w/o usable final edit → gen=0, "
          f"{len(no_parsed_mismatch)} w/o parsed mismatch → und=0)...")

    gen_scores: dict = {}
    und_scores: dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {}
        for idx, args in rate_edit_tasks.items():
            fut_map[ex.submit(judge.rate_edit, *args)] = ("gen", idx)
        for idx, args in mismatch_tasks.items():
            fut_map[ex.submit(judge.score_mismatch_prf, *args)] = ("und", idx)

        with tqdm(total=total_judge, desc="[ME] judge", unit="call") as pbar:
            for fut in as_completed(fut_map):
                kind, idx = fut_map[fut]
                try:
                    score = fut.result()
                except Exception:
                    score = 0.0 if kind == "gen" else None
                if kind == "gen":
                    gen_scores[idx] = score
                else:
                    und_scores[idx] = score  # dict {precision, recall, f1, ...} or None
                pbar.update(1)

    # --- Compute per-sample scores ---
    details = []
    failure_reasons: dict = {}
    if no_valid_edit:
        failure_reasons["no_valid_final_edit"] = len(no_valid_edit)
    if no_parsed_mismatch:
        failure_reasons["no_parsed_mismatch"] = len(no_parsed_mismatch)

    for idx in sorted(all_indices):
        generation_item = clip(gen_scores.get(idx, 0.0))
        prf = und_scores.get(idx) or {}
        understanding_item = clip(prf.get("f1", 0.0))  # F1 is the headline understanding score
        unified_item = clip((generation_item + understanding_item) / 2.0)

        row = {
            "dialogue_index": idx,
            "understanding_item": understanding_item,  # == F1
            "understanding_precision": round(float(prf.get("precision", 0.0)), 4),
            "understanding_recall": round(float(prf.get("recall", 0.0)), 4),
            "n_pred_mismatch": int(prf.get("n_pred", 0)),
            "n_gt_op": int(prf.get("n_gt", 0)),
            "n_extra_mismatch": int(prf.get("n_extra", 0)),
            "n_missed_op": int(prf.get("n_missed", 0)),
            "n_dropped_mismatch": int(prf.get("n_dropped", 0)),
            "generation_item": generation_item,
            "unified_item": unified_item,
        }
        details.append(row)
        if output_csv:
            _append_row(output_csv, row)

    # No-silent-truncation: report samples whose predicted mismatches hit the cap.
    capped = [d for d in details if d.get("n_dropped_mismatch", 0) > 0]
    if capped:
        dropped = sum(d["n_dropped_mismatch"] for d in capped)
        print(f"[ME] {len(capped)} sample(s) exceeded the predicted-mismatch cap "
              f"({dropped} item(s) dropped); their precision is slightly optimistic.")

    # Failure visibility: these samples scored 0 on the named axis (not excluded).
    n = len(all_indices)
    if no_valid_edit:
        print(f"[ME] {len(no_valid_edit)}/{n} sample(s) had no usable final edit "
              f"(generation_item=0).")
    if no_parsed_mismatch:
        print(f"[ME] {len(no_parsed_mismatch)}/{n} sample(s) had no parsed mismatch in "
              f"any g2u understanding round (understanding_item=0).")

    return build_task_result("ME", details, len(all_indices), failure_reasons)
