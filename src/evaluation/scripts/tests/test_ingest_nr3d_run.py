"""Tests for ingesting NR3D VG side-by-side runs."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


def _load_ingester():
    script = Path(__file__).resolve().parents[4] / "scripts" / "ingest_nr3d_run.py"
    spec = importlib.util.spec_from_file_location("ingest_nr3d_run", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ingest


def _write_side_by_side(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir()
    (output_dir / "side_by_side.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_ingest_nr3d_run_records_samples_and_metrics(tmp_path: Path) -> None:
    ingest = _load_ingester()
    output_dir = tmp_path / "run"
    _write_side_by_side(
        output_dir,
        {
            "pack_v1": {
                "n": 2,
                "mean_iou": 0.5,
                "Acc@0.25": 0.5,
                "Acc@0.50": 0.5,
                "per_sample": [
                    {
                        "sample_id": "scannet/scene0001_00::7::A1",
                        "status": "completed",
                        "iou": 1.0,
                        "selected_object_id": 7,
                        "confidence": 0.91,
                        "query": "the chair",
                    },
                    {
                        "sample_id": "scannet/scene0001_00::8::A2",
                        "status": "completed",
                        "iou": 0.0,
                        "selected_object_id": 9,
                        "confidence": 0.44,
                        "query": "the table",
                    },
                ],
            }
        },
    )

    db = tmp_path / "runs.sqlite"
    ingest(
        db_path=db,
        output_dir=output_dir,
        run_id="smoke",
        branch="feat/test",
        commit_hash="abc1234",
        notes="unit test",
    )

    conn = sqlite3.connect(db)
    try:
        run = conn.execute(
            "SELECT run_id, n, mean_iou, acc25, acc50 FROM runs"
        ).fetchone()
        assert run == ("smoke", 2, 0.5, 0.5, 0.5)

        samples = conn.execute(
            "SELECT sample_id, scene_id, target_id, iou, acc25, selected_object_id "
            "FROM samples ORDER BY sample_id"
        ).fetchall()
        assert samples == [
            ("scannet/scene0001_00::7::A1", "scannet/scene0001_00", 7, 1.0, 1, 7),
            ("scannet/scene0001_00::8::A2", "scannet/scene0001_00", 8, 0.0, 0, 9),
        ]
    finally:
        conn.close()


def test_ingest_nr3d_run_populates_tool_calls_from_trace(tmp_path: Path) -> None:
    ingest = _load_ingester()
    output_dir = tmp_path / "run"
    _write_side_by_side(
        output_dir,
        {
            "pack_v1": {
                "n": 1,
                "mean_iou": 0.0,
                "Acc@0.25": 0.0,
                "Acc@0.50": 0.0,
                "per_sample": [
                    {
                        "sample_id": "scannet/scene0001_00::7::A1",
                        "status": "completed",
                        "iou": 0.0,
                        "selected_object_id": 8,
                        "confidence": 0.5,
                        "query": "the chair",
                        "tool_trace": [
                            {
                                "tool_name": "inspect_proposal",
                                "tool_input": {"proposal_id": 7},
                                "response_text": "proposal 7: chair",
                            },
                            {
                                "tool_name": "select_by_text",
                                "tool_input": {"text": "chair"},
                                "response_text": {"proposal_ids": [7]},
                            },
                        ],
                    }
                ],
            }
        },
    )

    db = tmp_path / "runs.sqlite"
    ingest(
        db_path=db,
        output_dir=output_dir,
        run_id="trace-tools",
        branch="feat/test",
        commit_hash="abc1234",
    )

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT question_id, turn_idx, tool_name, tool_input, response_text "
            "FROM tool_calls WHERE run_id=? ORDER BY turn_idx",
            ("trace-tools",),
        ).fetchall()
        assert rows == [
            (
                "scannet/scene0001_00::7::A1",
                0,
                "inspect_proposal",
                '{"proposal_id": 7}',
                "proposal 7: chair",
            ),
            (
                "scannet/scene0001_00::7::A1",
                1,
                "select_by_text",
                '{"text": "chair"}',
                '{"proposal_ids": [7]}',
            ),
        ]
    finally:
        conn.close()


def test_ingest_nr3d_run_requires_per_sample(tmp_path: Path) -> None:
    ingest = _load_ingester()
    output_dir = tmp_path / "run"
    _write_side_by_side(
        output_dir,
        {
            "pack_v1": {
                "n": 1,
                "mean_iou": 0.5,
                "Acc@0.25": 1.0,
                "Acc@0.50": 0.0,
            }
        },
    )

    with pytest.raises(ValueError, match="missing required key: per_sample"):
        ingest(
            db_path=tmp_path / "runs.sqlite",
            output_dir=output_dir,
            run_id="missing-per-sample",
            branch=None,
            commit_hash=None,
        )


def test_ingest_nr3d_run_requires_metrics(tmp_path: Path) -> None:
    ingest = _load_ingester()
    output_dir = tmp_path / "run"
    _write_side_by_side(
        output_dir,
        {
            "pack_v1": {
                "n": 1,
                "Acc@0.25": 1.0,
                "Acc@0.50": 0.0,
                "per_sample": [{"sample_id": "scannet/scene0001_00::7::A1"}],
            }
        },
    )

    with pytest.raises(ValueError, match="missing required key: mean_iou"):
        ingest(
            db_path=tmp_path / "runs.sqlite",
            output_dir=output_dir,
            run_id="missing-metric",
            branch=None,
            commit_hash=None,
        )


def test_ingest_nr3d_run_rejects_n_mismatch(tmp_path: Path) -> None:
    ingest = _load_ingester()
    output_dir = tmp_path / "run"
    _write_side_by_side(
        output_dir,
        {
            "pack_v1": {
                "n": 2,
                "mean_iou": 0.5,
                "Acc@0.25": 1.0,
                "Acc@0.50": 0.0,
                "per_sample": [{"sample_id": "scannet/scene0001_00::7::A1"}],
            }
        },
    )

    with pytest.raises(ValueError, match="n=2 but per_sample has 1 rows"):
        ingest(
            db_path=tmp_path / "runs.sqlite",
            output_dir=output_dir,
            run_id="n-mismatch",
            branch=None,
            commit_hash=None,
        )


def test_ingest_nr3d_run_rejects_malformed_sample_id(tmp_path: Path) -> None:
    ingest = _load_ingester()
    output_dir = tmp_path / "run"
    _write_side_by_side(
        output_dir,
        {
            "pack_v1": {
                "n": 1,
                "mean_iou": 0.0,
                "Acc@0.25": 0.0,
                "Acc@0.50": 0.0,
                "per_sample": [{"sample_id": "scannet/scene0001_00::not-an-int::A1"}],
            }
        },
    )

    with pytest.raises(ValueError, match="target_id segment is not an int"):
        ingest(
            db_path=tmp_path / "runs.sqlite",
            output_dir=output_dir,
            run_id="bad-sample-id",
            branch=None,
            commit_hash=None,
        )


def test_runs_table_has_leaderboard_columns_after_ingest(tmp_path: Path) -> None:
    """After ingesting, runs table must have the 7 new leaderboard columns."""
    ingest = _load_ingester()
    output_dir = tmp_path / "run"
    _write_side_by_side(
        output_dir,
        {
            "pack_v1": {
                "n": 1,
                "mean_iou": 0.5,
                "Acc@0.25": 0.5,
                "Acc@0.50": 0.5,
                "per_sample": [
                    {
                        "sample_id": "scannet/scene0011_00::5::abc",
                        "status": "completed",
                        "iou": 0.5,
                        "predicted_bbox_3d_9dof": [0] * 9,
                        "gt_bbox_3d_9dof": [0] * 9,
                        "selected_object_id": 5,
                        "confidence": 1.0,
                        "query": "test",
                    }
                ],
            }
        },
    )
    db = tmp_path / "runs.sqlite"
    ingest(
        db_path=db,
        output_dir=output_dir,
        run_id="test_v3",
        branch=None,
        commit_hash=None,
    )
    conn = sqlite3.connect(str(db))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        expected_new = {
            "classification_acc_full",
            "classification_acc_filtered",
            "acc_easy",
            "acc_hard",
            "acc_view_dep",
            "acc_view_indep",
            "n_filtered",
        }
        assert expected_new <= cols, sorted(expected_new - cols)
        sample_cols = {r[1] for r in conn.execute("PRAGMA table_info(samples)")}
        expected_sample_new = {
            "is_easy",
            "is_view_dep",
            "is_filtered_out",
            "classification_correct",
        }
        assert expected_sample_new <= sample_cols, sorted(
            expected_sample_new - sample_cols
        )
    finally:
        conn.close()


def test_ingest_with_leaderboard_metrics_populates_columns(tmp_path: Path) -> None:
    """With --leaderboard-metrics, the new columns must be populated."""
    ingest = _load_ingester()
    output_dir = tmp_path / "run"
    sample_id = "scannet/scene0011_00::5::abc"
    _write_side_by_side(
        output_dir,
        {
            "pack_v1": {
                "n": 1,
                "mean_iou": 1.0,
                "Acc@0.25": 1.0,
                "Acc@0.50": 1.0,
                "per_sample": [
                    {
                        "sample_id": sample_id,
                        "status": "completed",
                        "iou": 1.0,
                        "predicted_bbox_3d_9dof": [0] * 9,
                        "gt_bbox_3d_9dof": [0] * 9,
                        "selected_object_id": 5,
                        "confidence": 1.0,
                        "query": "the chair",
                    }
                ],
            }
        },
    )
    leaderboard_metrics = tmp_path / "leaderboard_metrics.json"
    leaderboard_metrics.write_text(
        json.dumps(
            {
                "n_full": 1,
                "n_filtered": 1,
                "classification_acc_full": 1.0,
                "classification_acc_filtered": 1.0,
                "n_easy": 1,
                "n_hard": 0,
                "n_view_dep": 0,
                "n_view_indep": 1,
                "acc_easy": 1.0,
                "acc_hard": 0.0,
                "acc_view_dep": 0.0,
                "acc_view_indep": 1.0,
                "per_sample": [
                    {
                        "sample_id": sample_id,
                        "selected_object_id": 5,
                        "target_id": 5,
                        "is_correct": True,
                        "is_easy": True,
                        "is_view_dep": False,
                        "is_filtered_out": False,
                        "n_objects": 2,
                        "tokens": ["the", "chair"],
                        "mentions_target_class": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    db = tmp_path / "runs.sqlite"
    ingest(
        db_path=db,
        output_dir=output_dir,
        run_id="test_v3_full",
        branch="feat/nr3d-vg-benchmark",
        commit_hash="abc1234",
        notes="leaderboard track test",
        leaderboard_metrics_path=leaderboard_metrics,
    )
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT classification_acc_filtered, acc_easy, acc_hard, "
            "acc_view_dep, acc_view_indep, n_filtered FROM runs "
            "WHERE run_id=?",
            ("test_v3_full",),
        ).fetchone()
        assert row == (1.0, 1.0, 0.0, 0.0, 1.0, 1)
        sample_row = conn.execute(
            "SELECT is_easy, is_view_dep, is_filtered_out, "
            "classification_correct FROM samples "
            "WHERE run_id=? AND sample_id=?",
            ("test_v3_full", sample_id),
        ).fetchone()
        assert sample_row == (1, 0, 0, 1)
    finally:
        conn.close()


def test_ingest_without_leaderboard_metrics_keeps_new_columns_null(
    tmp_path: Path,
) -> None:
    """Old-style ingest (no --leaderboard-metrics) leaves new columns NULL."""
    ingest = _load_ingester()
    output_dir = tmp_path / "run"
    _write_side_by_side(
        output_dir,
        {
            "pack_v1": {
                "n": 1,
                "mean_iou": 0.5,
                "Acc@0.25": 0.5,
                "Acc@0.50": 0.5,
                "per_sample": [
                    {
                        "sample_id": "scannet/scene0011_00::5::abc",
                        "status": "completed",
                        "iou": 0.5,
                        "predicted_bbox_3d_9dof": [0] * 9,
                        "gt_bbox_3d_9dof": [0] * 9,
                        "selected_object_id": 5,
                        "confidence": 1.0,
                        "query": "test",
                    }
                ],
            }
        },
    )
    db = tmp_path / "runs.sqlite"
    ingest(
        db_path=db,
        output_dir=output_dir,
        run_id="test_v2_old",
        branch=None,
        commit_hash=None,
    )
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT classification_acc_full, classification_acc_filtered, "
            "acc_easy, acc_hard, acc_view_dep, acc_view_indep, n_filtered "
            "FROM runs WHERE run_id=?",
            ("test_v2_old",),
        ).fetchone()
        assert row == (None, None, None, None, None, None, None)
    finally:
        conn.close()
