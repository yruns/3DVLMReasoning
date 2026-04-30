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
