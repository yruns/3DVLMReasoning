"""Tests for ingest_scanrefer_run.py."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


def _load_ingester():
    script = Path(__file__).resolve().parents[4] / "scripts" / "ingest_scanrefer_run.py"
    spec = importlib.util.spec_from_file_location("ingest_scanrefer_run", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ingest


def _write_outputs(output_dir: Path, side_by_side: dict, leaderboard: dict | None = None) -> None:
    output_dir.mkdir()
    (output_dir / "side_by_side.json").write_text(json.dumps(side_by_side), encoding="utf-8")
    if leaderboard is not None:
        (output_dir / "leaderboard_metrics.json").write_text(
            json.dumps(leaderboard), encoding="utf-8"
        )


def test_runs_table_has_scanrefer_columns(tmp_path: Path) -> None:
    ingest = _load_ingester()
    output_dir = tmp_path / "run"
    _write_outputs(output_dir, {
        "pack_v1": {
            "n": 1, "mean_iou": 0.5, "Acc@0.25": 0.5, "Acc@0.50": 0.5,
            "per_sample": [{
                "sample_id": "scannet/scene_a::5::3", "status": "completed",
                "iou": 0.5, "selected_object_id": 5, "confidence": 0.9,
                "query": "test", "predicted_bbox_3d_9dof": [0]*9,
                "gt_bbox_3d_9dof": [0]*9,
            }],
        },
    }, leaderboard={
        "n_total": 1, "n_unique": 1, "n_multiple": 0,
        "acc25_overall": 1.0, "acc50_overall": 1.0,
        "acc25_unique": 1.0, "acc50_unique": 1.0,
        "acc25_multiple": 0.0, "acc50_multiple": 0.0,
        "mean_iou_overall": 0.5,
        "per_sample": [{
            "sample_id": "scannet/scene_a::5::3", "iou": 0.5,
            "is_unique": True, "acc25": 1, "acc50": 1, "target_id": 5,
            "target": "chair", "status": "completed",
        }],
    })
    db = tmp_path / "runs.sqlite"
    ingest(
        db_path=db, output_dir=output_dir, run_id="test_v1",
        branch="feat/test", commit_hash="abc1234",
        leaderboard_metrics_path=output_dir / "leaderboard_metrics.json",
    )
    conn = sqlite3.connect(str(db))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        expected = {"run_id", "n_total", "n_unique", "n_multiple",
                    "acc25_overall", "acc50_overall",
                    "acc25_unique", "acc50_unique",
                    "acc25_multiple", "acc50_multiple",
                    "mean_iou_overall"}
        assert expected <= cols, sorted(expected - cols)
        sample_cols = {r[1] for r in conn.execute("PRAGMA table_info(samples)")}
        expected_sample = {"sample_id", "scene_id", "target_id", "ann_id",
                           "iou", "acc25", "acc50", "is_unique"}
        assert expected_sample <= sample_cols, sorted(expected_sample - sample_cols)
    finally:
        conn.close()


def test_ingest_populates_per_sample_with_unique(tmp_path: Path) -> None:
    ingest = _load_ingester()
    output_dir = tmp_path / "run"
    _write_outputs(output_dir, {
        "pack_v1": {
            "n": 1, "mean_iou": 1.0, "Acc@0.25": 1.0, "Acc@0.50": 1.0,
            "per_sample": [{
                "sample_id": "scannet/scene_a::5::3", "status": "completed",
                "iou": 1.0, "selected_object_id": 5, "confidence": 0.9,
                "query": "the chair", "predicted_bbox_3d_9dof": [0]*9,
                "gt_bbox_3d_9dof": [0]*9,
            }],
        },
    }, leaderboard={
        "n_total": 1, "n_unique": 1, "n_multiple": 0,
        "acc25_overall": 1.0, "acc50_overall": 1.0,
        "acc25_unique": 1.0, "acc50_unique": 1.0,
        "acc25_multiple": 0.0, "acc50_multiple": 0.0,
        "mean_iou_overall": 1.0,
        "per_sample": [{
            "sample_id": "scannet/scene_a::5::3", "iou": 1.0,
            "is_unique": True, "acc25": 1, "acc50": 1, "target_id": 5,
            "target": "chair", "status": "completed",
        }],
    })
    db = tmp_path / "runs.sqlite"
    ingest(
        db_path=db, output_dir=output_dir, run_id="test_v1",
        branch="feat/test", commit_hash="abc1234",
        leaderboard_metrics_path=output_dir / "leaderboard_metrics.json",
    )
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT acc25_overall, acc50_overall, acc25_unique, acc50_unique, "
            "acc25_multiple, acc50_multiple, n_total, n_unique, n_multiple "
            "FROM runs WHERE run_id=?", ("test_v1",)
        ).fetchone()
        assert row == (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1, 1, 0)
        sample_row = conn.execute(
            "SELECT acc25, acc50, is_unique FROM samples "
            "WHERE run_id=? AND sample_id=?",
            ("test_v1", "scannet/scene_a::5::3")
        ).fetchone()
        assert sample_row == (1, 1, 1)
    finally:
        conn.close()


def test_ingest_requires_per_sample(tmp_path: Path) -> None:
    ingest = _load_ingester()
    output_dir = tmp_path / "run"
    _write_outputs(output_dir, {
        "pack_v1": {"n": 0, "mean_iou": 0, "Acc@0.25": 0, "Acc@0.50": 0},
    })
    with pytest.raises(ValueError, match="per_sample"):
        ingest(
            db_path=tmp_path / "runs.sqlite", output_dir=output_dir,
            run_id="x", branch=None, commit_hash=None,
        )
