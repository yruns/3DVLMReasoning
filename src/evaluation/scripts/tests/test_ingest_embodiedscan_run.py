"""Tests for ingesting EmbodiedScan VG side-by-side runs."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path


def test_ingest_embodiedscan_run_records_samples_and_metrics(tmp_path) -> None:
    script = (
        Path(__file__).resolve().parents[4] / "scripts" / "ingest_embodiedscan_run.py"
    )
    spec = importlib.util.spec_from_file_location("ingest_embodiedscan_run", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    ingest = module.ingest

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "side_by_side.json").write_text(
        json.dumps(
            {
                "pack_v1": {
                    "n": 2,
                    "mean_iou": 0.5,
                    "Acc@0.25": 0.5,
                    "Acc@0.50": 0.5,
                    "per_sample": [
                        {
                            "sample_id": "scene0001_00::7",
                            "status": "completed",
                            "iou": 1.0,
                            "selected_object_id": 7,
                            "confidence": 0.91,
                            "query": "the chair",
                        },
                        {
                            "sample_id": "scene0001_00::8",
                            "status": "completed",
                            "iou": 0.0,
                            "selected_object_id": 9,
                            "confidence": 0.44,
                            "query": "the table",
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
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
            ("scene0001_00::7", "scene0001_00", 7, 1.0, 1, 7),
            ("scene0001_00::8", "scene0001_00", 8, 0.0, 0, 9),
        ]
    finally:
        conn.close()
