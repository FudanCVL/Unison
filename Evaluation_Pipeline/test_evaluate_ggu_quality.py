"""Unit tests for evaluate_ggu_quality (pure-function pieces)."""
import sys, os, csv
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import evaluate_ggu_quality as E


def test_detect_subtask_2d():
    p = "../data/Gen_Guided_Und/2D_Spatial/matrices/matrix_3x3_01.jpg"
    sub, rel = E.detect_subtask(p)
    assert sub == "2D_Spatial"
    assert rel == "matrices/matrix_3x3_01.jpg"


def test_detect_subtask_3d():
    p = "../data/Gen_Guided_Und/3D_Spatial/cubes/cube_1_1.jpg"
    sub, rel = E.detect_subtask(p)
    assert sub == "3D_Spatial"
    assert rel == "cubes/cube_1_1.jpg"


def test_detect_subtask_complex_relation_returns_none():
    p = "../data/Gen_Guided_Und/complex_relation/foo/bar.jpg"
    sub, rel = E.detect_subtask(p)
    assert sub is None
    assert rel is None


def test_load_generation_rows_filters_to_generation_only(tmp_path):
    csv_path = tmp_path / "ggu.csv"
    rows = [
        # operation_type, image_path, model_response
        ("understanding", "../data/Gen_Guided_Und/2D_Spatial/m/a.jpg", "A"),
        ("generation",    "../data/Gen_Guided_Und/2D_Spatial/m/a.jpg",
                          "result/Model/images/GGU/000000_generation.png"),
        ("unify",         "../data/Gen_Guided_Und/2D_Spatial/m/a.jpg",
                          "result/Model/images/GGU/000001_generation.png"),
        ("generation",    "../data/Gen_Guided_Und/3D_Spatial/c/x.jpg",
                          "result/Model/images/GGU/000002_generation.png"),
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["operation_type", "image_path", "model_response"])
        for r in rows:
            w.writerow(r)

    out = E.load_generation_rows(str(csv_path))
    assert len(out) == 2
    assert out[0]["sub_task"] == "2D_Spatial"
    assert out[0]["image_path_rel"] == "m/a.jpg"
    assert out[0]["generated_image_path"] == "result/Model/images/GGU/000000_generation.png"
    assert out[1]["sub_task"] == "3D_Spatial"


import json


def test_normalize_answer_yes_variants():
    assert E.normalize_answer("yes") == "yes"
    assert E.normalize_answer("Yes.") == "yes"
    assert E.normalize_answer("  YES \n") == "yes"
    assert E.normalize_answer("yes, the matrix is 3 by 3") == "yes"


def test_normalize_answer_no_variants():
    assert E.normalize_answer("no") == "no"
    assert E.normalize_answer("No.") == "no"
    assert E.normalize_answer("NO") == "no"


def test_normalize_answer_ambiguous_to_no():
    assert E.normalize_answer("maybe") == "no"
    assert E.normalize_answer("I cannot tell") == "no"
    assert E.normalize_answer("") == "no"
    assert E.normalize_answer("unknown") == "no"


def test_build_gt_index(tmp_path):
    data_dir = tmp_path / "Gen_Guided_Und"
    (data_dir / "2D_Spatial").mkdir(parents=True)
    (data_dir / "3D_Spatial").mkdir(parents=True)

    (data_dir / "2D_Spatial" / "2d_spatial.json").write_text(json.dumps([
        {"image_path": "matrices/a.jpg",
         "quality_questions": [{"id": "q01", "type": "dimension",
                                 "text": "Q?", "answer": "yes"}]}
    ]), encoding="utf-8")
    (data_dir / "3D_Spatial" / "spatial.json").write_text(json.dumps([
        {"image_path": "cubes/c.jpg",
         "quality_questions": [{"id": "q01", "type": "dominant_color",
                                 "text": "Q?", "answer": "yes"}]}
    ]), encoding="utf-8")

    idx = E.build_gt_index(str(data_dir), subtasks=["2D_Spatial", "3D_Spatial"])
    assert ("2D_Spatial", "matrices/a.jpg") in idx
    assert ("3D_Spatial", "cubes/c.jpg") in idx
    assert idx[("2D_Spatial", "matrices/a.jpg")][0]["id"] == "q01"


def test_build_gt_index_filtered_subtasks(tmp_path):
    data_dir = tmp_path / "Gen_Guided_Und"
    (data_dir / "2D_Spatial").mkdir(parents=True)
    (data_dir / "2D_Spatial" / "2d_spatial.json").write_text(json.dumps([
        {"image_path": "matrices/a.jpg",
         "quality_questions": [{"id": "q01", "type": "dimension",
                                 "text": "Q?", "answer": "yes"}]}
    ]), encoding="utf-8")

    idx = E.build_gt_index(str(data_dir), subtasks=["2D_Spatial"])
    assert ("2D_Spatial", "matrices/a.jpg") in idx
    # No 3D file → just absent, not error
    assert all(k[0] == "2D_Spatial" for k in idx)


def _detail(sub_task, sample_id, n_correct, n_total, qtype="position"):
    qs = []
    for i in range(n_total):
        qs.append({
            "id": f"q{i+1:02d}", "type": qtype, "gt": "yes",
            "pred": "yes" if i < n_correct else "no",
            "correct": i < n_correct,
        })
    return {
        "sample_id": f"{sub_task}/{sample_id}",
        "sub_task": sub_task,
        "generated_image": f"gen/{sample_id}",
        "sample_accuracy": (n_correct / n_total) if n_total else 0.0,
        "questions": qs,
    }


def test_aggregate_summary_overall_and_per_subtask():
    details = [
        _detail("2D_Spatial", "a.jpg", n_correct=2, n_total=4),
        _detail("2D_Spatial", "b.jpg", n_correct=4, n_total=4),
        _detail("3D_Spatial", "c.jpg", n_correct=1, n_total=2),
    ]
    errors = {"image_missing": 0, "gt_missing": 0,
              "skipped_no_generation": 0, "timeout": 0}
    s = E.aggregate_summary(details, errors,
                            model_name="M", judge_model="J")
    assert s["model_name"] == "M"
    assert s["total_samples"] == 3
    assert s["total_questions"] == 10
    assert abs(s["overall_accuracy"] - 7 / 10) < 1e-9
    assert s["per_subtask"]["2D_Spatial"]["samples"] == 2
    assert s["per_subtask"]["2D_Spatial"]["questions"] == 8
    assert abs(s["per_subtask"]["2D_Spatial"]["accuracy"] - 6 / 8) < 1e-9
    assert s["per_subtask"]["3D_Spatial"]["samples"] == 1
    assert abs(s["per_subtask"]["3D_Spatial"]["accuracy"] - 1 / 2) < 1e-9


def test_aggregate_summary_per_question_type():
    details = [
        _detail("2D_Spatial", "a.jpg", n_correct=2, n_total=4, qtype="position"),
        _detail("3D_Spatial", "c.jpg", n_correct=1, n_total=2, qtype="dominant_color"),
    ]
    errors = {"image_missing": 0, "gt_missing": 0,
              "skipped_no_generation": 0, "timeout": 0}
    s = E.aggregate_summary(details, errors,
                            model_name="M", judge_model="J")
    assert s["per_question_type"]["position"]["questions"] == 4
    assert abs(s["per_question_type"]["position"]["accuracy"] - 2 / 4) < 1e-9
    assert s["per_question_type"]["dominant_color"]["questions"] == 2
    assert abs(s["per_question_type"]["dominant_color"]["accuracy"] - 1 / 2) < 1e-9


def test_aggregate_summary_empty_details():
    errors = {"image_missing": 0, "gt_missing": 0,
              "skipped_no_generation": 0, "timeout": 0}
    s = E.aggregate_summary([], errors, model_name="M", judge_model="J")
    assert s["total_samples"] == 0
    assert s["total_questions"] == 0
    assert s["overall_accuracy"] == 0.0
    assert s["per_subtask"] == {}
    assert s["per_question_type"] == {}


def test_build_judge_messages():
    msgs = E.build_judge_messages("gen/img.png", "Is the matrix 3 by 3?")
    assert msgs[0]["role"] == "system"
    assert "yes" in msgs[0]["content"].lower()
    assert msgs[1]["role"] == "user"
    contents = msgs[1]["content"]
    assert any(c.get("type") == "image" and c["image"] == "gen/img.png"
               for c in contents)
    assert any(c.get("type") == "text" and "Is the matrix 3 by 3?" in c["text"]
               for c in contents)


def test_score_sample_against_gt_all_yes_pred_yes():
    qq = [
        {"id": "q01", "type": "dimension", "text": "Q?", "answer": "yes"},
        {"id": "q02", "type": "position",  "text": "Q?", "answer": "yes"},
    ]
    preds = ["yes", "yes"]
    detail = E.score_sample("2D_Spatial/a.jpg", "2D_Spatial", "gen/a.png", qq, preds)
    assert detail["sample_accuracy"] == 1.0
    assert all(q["correct"] for q in detail["questions"])


def test_score_sample_against_gt_mixed():
    qq = [
        {"id": "q01", "type": "dimension", "text": "Q?", "answer": "yes"},
        {"id": "q02", "type": "position",  "text": "Q?", "answer": "yes"},
    ]
    preds = ["yes", "no"]
    detail = E.score_sample("2D_Spatial/a.jpg", "2D_Spatial", "gen/a.png", qq, preds)
    assert abs(detail["sample_accuracy"] - 0.5) < 1e-9
    assert detail["questions"][0]["correct"] is True
    assert detail["questions"][1]["correct"] is False


def test_prepare_jobs_matches_gt_and_records_missing():
    rows = [
        {"sub_task": "2D_Spatial", "image_path_rel": "matrices/a.jpg",
         "generated_image_path": "gen/a.png", "raw_image_path": "../data/.../a.jpg"},
        {"sub_task": "3D_Spatial", "image_path_rel": "cubes/c.jpg",
         "generated_image_path": "gen/c.png", "raw_image_path": "../data/.../c.jpg"},
        {"sub_task": "2D_Spatial", "image_path_rel": "matrices/z.jpg",
         "generated_image_path": "gen/z.png", "raw_image_path": "../data/.../z.jpg"},
    ]
    gt_index = {
        ("2D_Spatial", "matrices/a.jpg"): [
            {"id": "q01", "type": "dimension", "text": "Q?", "answer": "yes"}],
        ("3D_Spatial", "cubes/c.jpg"): [
            {"id": "q01", "type": "dominant_color", "text": "Q?", "answer": "yes"}],
    }
    jobs, errors = E.prepare_jobs(rows, gt_index)
    assert len(jobs) == 2
    assert errors["gt_missing"] == 1
    assert errors["skipped_no_generation"] == 0


def test_write_results_round_trip(tmp_path):
    summary = {"model_name": "M", "overall_accuracy": 0.75}
    details = [{"sample_id": "x", "sample_accuracy": 0.75, "questions": []}]
    out = tmp_path / "out.json"
    E.write_results(str(out), summary, details)
    obj = json.loads(out.read_text(encoding="utf-8"))
    assert obj["summary"]["model_name"] == "M"
    assert obj["details"][0]["sample_id"] == "x"
