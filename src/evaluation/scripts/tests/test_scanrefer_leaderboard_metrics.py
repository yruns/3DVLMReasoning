"""Tests for ScanRefer leaderboard metrics aggregator."""

from __future__ import annotations

import json
import gzip
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.scripts.scanrefer_leaderboard_metrics import (
    aggregate,
    compute_leaderboard_metrics,
    load_sample_id_filter,
    load_side_by_side_predictions,
)


def _record(
    sid: str,
    iou: float,
    is_unique: bool,
    status: str = "completed",
    target_id: int = 0,
    selected_object_id: int = 0,
) -> dict:
    return {
        "sample_id": sid,
        "iou": iou,
        "status": status,
        "is_unique": is_unique,
        "target_id": target_id,
        "selected_object_id": selected_object_id,
    }


def _meta(sid: str, is_unique: bool, target_id: int = 0) -> dict:
    return {
        "sample_id": sid,
        "is_unique": is_unique,
        "target_id": target_id,
        "target": "chair",
    }


def test_aggregate_all_correct():
    preds = [_record("a", 0.9, True), _record("b", 0.9, False), _record("c", 1.0, True)]
    meta = [_meta("a", True), _meta("b", False), _meta("c", True)]
    m = aggregate(preds, meta)
    assert m["n_total"] == 3
    assert m["acc25_overall"] == 1.0
    assert m["acc50_overall"] == 1.0


def test_aggregate_all_wrong():
    preds = [_record("a", 0.0, True), _record("b", 0.0, False)]
    meta = [_meta("a", True), _meta("b", False)]
    m = aggregate(preds, meta)
    assert m["acc25_overall"] == 0.0
    assert m["acc50_overall"] == 0.0


def test_aggregate_iou_threshold_boundaries():
    """IoU exactly at 0.25 or 0.50 should count as correct (≥ threshold)."""
    preds = [
        _record("a", 0.25, True),
        _record("b", 0.50, False),
        _record("c", 0.49, True),
    ]
    meta = [_meta("a", True), _meta("b", False), _meta("c", True)]
    m = aggregate(preds, meta)
    # acc25: a (0.25 >= 0.25) + b (0.5 >= 0.25) + c (0.49 >= 0.25) → 3/3
    assert m["acc25_overall"] == 1.0
    # acc50: only b (0.5 >= 0.50) → 1/3
    assert m["acc50_overall"] == pytest.approx(1 / 3, abs=1e-6)


def test_aggregate_unique_multiple_partition():
    preds = [
        _record("a", 1.0, True),
        _record("b", 1.0, True),  # unique, both correct
        _record("c", 0.0, False),
        _record("d", 1.0, False),  # multiple, 1 of 2 correct
    ]
    meta = [_meta("a", True), _meta("b", True), _meta("c", False), _meta("d", False)]
    m = aggregate(preds, meta)
    assert m["n_unique"] == 2
    assert m["n_multiple"] == 2
    assert m["acc25_unique"] == 1.0
    assert m["acc50_unique"] == 1.0
    assert m["acc25_multiple"] == 0.5
    assert m["acc50_multiple"] == 0.5


def test_aggregate_failed_sentinel_counts_as_zero():
    preds = [_record("a", 0.0, True, status="failed"), _record("b", 1.0, False)]
    meta = [_meta("a", True), _meta("b", False)]
    m = aggregate(preds, meta)
    assert m["acc25_overall"] == 0.5
    assert m["acc25_unique"] == 0.0
    assert m["acc25_multiple"] == 1.0


def test_aggregate_raises_on_missing_sample_id():
    """Every meta entry must have a matching prediction; otherwise raise."""
    preds = [_record("a", 1.0, True)]
    meta = [_meta("a", True), _meta("b", True)]
    with pytest.raises(ValueError, match="missing"):
        aggregate(preds, meta)


def test_aggregate_invariants():
    """n_unique + n_multiple == n_total."""
    preds = [_record("a", 1.0, True), _record("b", 0.0, False), _record("c", 1.0, True)]
    meta = [_meta("a", True), _meta("b", False), _meta("c", True)]
    m = aggregate(preds, meta)
    assert m["n_unique"] + m["n_multiple"] == m["n_total"] == 3
    assert len(m["per_sample"]) == 3


def test_aggregate_per_sample_carries_flags():
    preds = [_record("a", 0.6, True)]
    meta = [_meta("a", True)]
    m = aggregate(preds, meta)
    s = m["per_sample"][0]
    assert s["sample_id"] == "a"
    assert s["is_unique"] is True
    assert s["acc25"] == 1
    assert s["acc50"] == 1


def test_load_sample_id_filter_accepts_runner_json_shapes(tmp_path):
    p = tmp_path / "sample_ids.json"
    p.write_text(
        '[{"sample_id": "scannet/scene_a::1::0"}, "scannet/scene_b::2::1"]',
        encoding="utf-8",
    )

    assert load_sample_id_filter(p) == {
        "scannet/scene_a::1::0",
        "scannet/scene_b::2::1",
    }


def test_load_side_by_side_predictions_streams_without_read_text(tmp_path, monkeypatch):
    side_by_side = tmp_path / "side_by_side.json"
    side_by_side.write_text(
        json.dumps(
            {
                "pack_v1": {
                    "n": 2,
                    "per_sample": [
                        {
                            "sample_id": "a",
                            "iou": 0.75,
                            "status": "completed",
                            "tool_trace": [{"response_text": "x" * 1000}],
                        },
                        {
                            "sample_id": "b",
                            "iou": None,
                            "status": "failed",
                            "tool_trace": [{"response_text": "y" * 1000}],
                        },
                    ],
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    original_read_text = Path.read_text

    def guarded_read_text(self: Path, *args, **kwargs):
        if self == side_by_side:
            raise AssertionError("side_by_side.json must be streamed")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    assert list(load_side_by_side_predictions(side_by_side, backend="pack_v1")) == [
        {"sample_id": "a", "iou": 0.75, "status": "completed"},
        {"sample_id": "b", "iou": None, "status": "failed"},
    ]


def test_compute_leaderboard_metrics_streams_large_side_by_side(
    tmp_path,
    monkeypatch,
):
    side_by_side = tmp_path / "side_by_side.json"
    sample_ids = tmp_path / "sample_ids.json"
    scanrefer_root = tmp_path / "scanrefer"
    raw_root = scanrefer_root / "raw"
    raw_root.mkdir(parents=True)
    raw_root.joinpath("ScanRefer_filtered_val.json").write_text(
        json.dumps(
            [
                {
                    "scene_id": "scene_a",
                    "object_id": "0",
                    "ann_id": "0",
                    "object_name": "chair",
                    "description": "chair",
                },
                {
                    "scene_id": "scene_b",
                    "object_id": "0",
                    "ann_id": "0",
                    "object_name": "table",
                    "description": "table",
                },
            ]
        ),
        encoding="utf-8",
    )
    # Use canonical ids in this test so the lightweight metadata loader can
    # reconstruct them from ScanRefer rows.
    side_by_side.write_text(
        json.dumps(
            {
                "pack_v1": {
                    "n": 2,
                    "per_sample": [
                        {
                            "sample_id": "scannet/scene_a::0::0",
                            "iou": 0.6,
                            "status": "completed",
                        },
                        {
                            "sample_id": "scannet/scene_b::0::0",
                            "iou": 0.1,
                            "status": "completed",
                        },
                    ],
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    sample_ids.write_text(
        '["scannet/scene_a::0::0", "scannet/scene_b::0::0"]',
        encoding="utf-8",
    )
    phase8_root = tmp_path / "phase8"
    for scene, class_name in [("scene_a", "chair"), ("scene_b", "table")]:
        pkl_path = (
            phase8_root
            / scene
            / "conceptgraph"
            / "pcd_saves"
            / "full_pcd_gt_axisaligned_post.pkl.gz"
        )
        pkl_path.parent.mkdir(parents=True)
        with gzip.open(pkl_path, "wb") as f:
            pickle.dump(
                {"objects": [{"class_name": [class_name], "bbox_np": [[0, 0, 0]] * 8}]},
                f,
            )

    original_read_text = Path.read_text

    def guarded_read_text(self: Path, *args, **kwargs):
        if self == side_by_side:
            raise AssertionError("side_by_side.json must be streamed")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    metrics = compute_leaderboard_metrics(
        side_by_side_path=side_by_side,
        scanrefer_data_root=scanrefer_root,
        phase8_data_root=phase8_root,
        sample_ids_path=sample_ids,
    )
    assert metrics["n_total"] == 2
    assert metrics["acc25_overall"] == 0.5


def test_compute_leaderboard_metrics_uses_light_metadata_loader(
    tmp_path,
    monkeypatch,
):
    side_by_side = tmp_path / "side_by_side.json"
    side_by_side.write_text(
        json.dumps(
            {
                "pack_v1": {
                    "n": 2,
                    "per_sample": [
                        {"sample_id": "scannet/scene0000_00::0::0", "iou": 0.6},
                        {"sample_id": "scannet/scene0000_00::1::0", "iou": 0.1},
                    ],
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    sample_ids = tmp_path / "sample_ids.json"
    sample_ids.write_text(
        '["scannet/scene0000_00::0::0", "scannet/scene0000_00::1::0"]',
        encoding="utf-8",
    )
    scanrefer_root = tmp_path / "scanrefer"
    raw_root = scanrefer_root / "raw"
    raw_root.mkdir(parents=True)
    raw_root.joinpath("ScanRefer_filtered_val.json").write_text(
        json.dumps(
            [
                {
                    "scene_id": "scene0000_00",
                    "object_id": "0",
                    "ann_id": "0",
                    "object_name": "chair",
                    "description": "chair",
                },
                {
                    "scene_id": "scene0000_00",
                    "object_id": "1",
                    "ann_id": "0",
                    "object_name": "table",
                    "description": "table",
                },
            ]
        ),
        encoding="utf-8",
    )
    phase8_root = tmp_path / "phase8"
    pkl_path = (
        phase8_root
        / "scene0000_00"
        / "conceptgraph"
        / "pcd_saves"
        / "full_pcd_gt_axisaligned_post.pkl.gz"
    )
    pkl_path.parent.mkdir(parents=True)
    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "objects": [
                    {"class_name": ["chair"], "bbox_np": [[0, 0, 0]] * 8},
                    {"class_name": ["table"], "bbox_np": [[0, 0, 0]] * 8},
                ]
            },
            f,
        )

    class HeavyDataset:
        @classmethod
        def from_path(cls, **kwargs):
            raise AssertionError("leaderboard metrics must not use heavy dataset")

    monkeypatch.setitem(
        sys.modules,
        "benchmarks.scanrefer_loader",
        SimpleNamespace(ScanRefVGDataset=HeavyDataset),
    )

    metrics = compute_leaderboard_metrics(
        side_by_side_path=side_by_side,
        scanrefer_data_root=scanrefer_root,
        phase8_data_root=phase8_root,
        sample_ids_path=sample_ids,
    )
    assert metrics["n_unique"] == 2
    assert metrics["acc25_unique"] == 0.5
