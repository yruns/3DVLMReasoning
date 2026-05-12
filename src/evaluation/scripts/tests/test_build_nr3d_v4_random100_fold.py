"""Tests for building the frozen NR3D v4 random100 fold."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_builder():
    script = (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "build_nr3d_v4_random100_fold.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_nr3d_v4_random100_fold",
        script,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_subset_is_deterministic_independent_of_input_order() -> None:
    builder = _load_builder()
    rows = [
        {
            "sample_id": "scannet/scene0003_00::3::A3",
            "scene_id": "scene0003_00",
            "target_id": 3,
            "category": "table",
        },
        {
            "sample_id": "scannet/scene0001_00::1::A1",
            "scene_id": "scene0001_00",
            "target_id": 1,
            "category": "chair",
        },
        {
            "sample_id": "scannet/scene0002_00::2::A2",
            "scene_id": "scene0002_00",
            "target_id": 2,
            "category": "sofa",
        },
    ]
    prediction_ids = {row["sample_id"] for row in rows}

    selected = builder.select_subset(rows, prediction_ids, n=2)
    selected_reversed = builder.select_subset(list(reversed(rows)), prediction_ids, n=2)

    expected_ids = sorted(prediction_ids, key=builder.stable_key)[:2]
    assert [row["sample_id"] for row in selected] == expected_ids
    assert selected == selected_reversed


def test_select_subset_ignores_correctness_and_status_when_joining_predictions() -> None:
    builder = _load_builder()
    rows = [
        {
            "sample_id": "scannet/scene0001_00::1::A1",
            "scene_id": "scene0001_00",
            "target_id": 1,
            "category": "chair",
        },
        {
            "sample_id": "scannet/scene0002_00::2::A2",
            "scene_id": "scene0002_00",
            "target_id": 2,
            "category": "sofa",
        },
        {
            "sample_id": "scannet/scene0003_00::3::A3",
            "scene_id": "scene0003_00",
            "target_id": 3,
            "category": "table",
        },
    ]
    side_by_side = {
        "pack_v1": {
            "per_sample": [
                {
                    "sample_id": "scannet/scene0001_00::1::A1",
                    "status": "failed",
                    "correct": False,
                },
                {
                    "sample_id": "scannet/scene0002_00::2::A2",
                    "status": "completed",
                    "correct": True,
                },
                {
                    "sample_id": "scannet/scene0003_00::3::A3",
                    "status": "completed",
                    "correct": False,
                },
            ]
        }
    }

    prediction_ids = {
        row["sample_id"] for row in side_by_side["pack_v1"]["per_sample"]
    }
    selected = builder.select_subset(rows, prediction_ids, n=len(prediction_ids))

    assert {row["sample_id"] for row in selected} == prediction_ids


def test_write_outputs_emits_sample_rows_and_summary(tmp_path: Path) -> None:
    builder = _load_builder()
    selected = [
        {
            "sample_id": "scannet/scene0001_00::1::A1",
            "scene_id": "scene0001_00",
            "target_id": 1,
            "category": "chair",
            "query": "ignored",
        },
        {
            "sample_id": "scannet/scene0002_00::2::A2",
            "scene_id": "scene0002_00",
            "target_id": 2,
            "category": "sofa",
            "status": "ignored",
        },
    ]
    sample_out = tmp_path / "samples.json"
    summary_out = tmp_path / "summary.json"

    builder.write_outputs(
        selected,
        sample_out=sample_out,
        summary_out=summary_out,
        n_candidates=5,
    )

    sample_rows = json.loads(sample_out.read_text(encoding="utf-8"))
    assert [set(row) for row in sample_rows] == [
        {"sample_id", "scene_id", "target_id", "category"},
        {"sample_id", "scene_id", "target_id", "category"},
    ]
    assert sample_rows == [
        {
            "sample_id": "scannet/scene0001_00::1::A1",
            "scene_id": "scene0001_00",
            "target_id": 1,
            "category": "chair",
        },
        {
            "sample_id": "scannet/scene0002_00::2::A2",
            "scene_id": "scene0002_00",
            "target_id": 2,
            "category": "sofa",
        },
    ]

    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    assert summary == {
        "selection_salt": builder.SELECTION_SALT,
        "n_selected": 2,
        "n_candidates": 5,
        "sample_ids_path": str(sample_out),
        "first_10_sample_ids": [
            "scannet/scene0001_00::1::A1",
            "scannet/scene0002_00::2::A2",
        ],
    }
