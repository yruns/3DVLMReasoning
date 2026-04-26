"""Tests for extracting batch30 sample ids from feasibility CSV output."""
from __future__ import annotations

import pytest

from evaluation.scripts.extract_batch30_sample_ids import extract_sample_ids


def test_extract_sample_ids_dedupes_and_preserves_order(tmp_path) -> None:
    csv = tmp_path / "scores.csv"
    csv.write_text(
        "scene_id,target_id,category,visible_frames,input_condition,best_iou,best_proposal_index,failure_tag\n"
        "scene0451_00,72,picture,29,single_frame_recon,0.0,,\n"
        "scene0451_00,72,picture,29,multi_frame_recon,0.1,,\n"
        "scene0451_00,32,curtain,104,single_frame_recon,0.03,7,\n"
        "scene0114_00,5,chair,40,single_frame_recon,0.5,3,\n",
        encoding="utf-8",
    )
    out = extract_sample_ids(csv)
    assert [(s["scene_id"], s["target_id"]) for s in out] == [
        ("scene0451_00", 72),
        ("scene0451_00", 32),
        ("scene0114_00", 5),
    ]
    assert out[0]["sample_id"] == "scene0451_00::72"
    assert out[0]["category"] == "picture"


def test_extract_sample_ids_raises_on_missing_target_id(tmp_path) -> None:
    csv = tmp_path / "scores.csv"
    csv.write_text(
        "scene_id,target_id,category,visible_frames,input_condition,best_iou,best_proposal_index,failure_tag\n"
        "scene0451_00,,picture,29,single_frame_recon,0.0,,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="target_id"):
        extract_sample_ids(csv)


def test_extract_sample_ids_raises_on_missing_required_column(tmp_path) -> None:
    csv = tmp_path / "scores.csv"
    csv.write_text("scene_id,category\nscene0451_00,picture\n", encoding="utf-8")
    with pytest.raises(ValueError, match="target_id"):
        extract_sample_ids(csv)


def test_extract_sample_ids_raises_on_empty_csv(tmp_path) -> None:
    csv = tmp_path / "scores.csv"
    csv.write_text(
        "scene_id,target_id,category,visible_frames,input_condition,best_iou,best_proposal_index,failure_tag\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="empty"):
        extract_sample_ids(csv)
