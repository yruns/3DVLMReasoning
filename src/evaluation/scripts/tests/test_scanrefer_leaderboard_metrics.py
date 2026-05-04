"""Tests for ScanRefer leaderboard metrics aggregator."""

from __future__ import annotations

import pytest

from evaluation.scripts.scanrefer_leaderboard_metrics import (
    aggregate,
    load_sample_id_filter,
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
