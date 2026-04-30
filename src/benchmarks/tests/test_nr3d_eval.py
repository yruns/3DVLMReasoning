"""Unit tests for NR3D visual-grounding evaluation."""

from __future__ import annotations

import math

import pytest

from benchmarks.nr3d_eval import evaluate_vg_predictions
from benchmarks.nr3d_loader import Nr3dVGSample


def _sample(
    sample_id: str,
    target: str = "chair",
    bbox: list[float] | None = None,
) -> Nr3dVGSample:
    return Nr3dVGSample(
        sample_id=sample_id,
        scene_id="scene0001_00",
        query="the chair",
        scan_id="scannet/scene0001_00",
        target_id=1,
        target=target,
        gt_bbox_3d=bbox or [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    )


class TestEvaluateNr3dVgPredictions:
    def test_evaluate_vg_empty_predictions(self) -> None:
        with pytest.raises(ValueError, match="predictions is empty"):
            evaluate_vg_predictions([], [_sample("s1")])

    def test_evaluate_vg_unknown_sample_id(self) -> None:
        with pytest.raises(ValueError, match="not in dataset"):
            evaluate_vg_predictions(
                [{"sample_id": "missing", "bbox_3d": [0, 0, 0, 1, 1, 1]}],
                [_sample("s1")],
            )

    def test_evaluate_vg_duplicate_sample_id(self) -> None:
        samples = [_sample("s1"), _sample("s2")]
        predictions = [
            {"sample_id": "s1", "bbox_3d": [0, 0, 0, 1, 1, 1]},
            {"sample_id": "s1", "bbox_3d": [1, 1, 1, 1, 1, 1]},
        ]
        with pytest.raises(ValueError, match="Duplicate prediction sample_id 's1'"):
            evaluate_vg_predictions(predictions, samples)

    def test_evaluate_vg_overall_metrics(self) -> None:
        samples = [
            _sample("s1"),
            _sample("s2"),
            _sample("s3"),
        ]
        predictions = [
            {"sample_id": "s1", "bbox_3d": [0, 0, 0, 1, 1, 1, 0, 0, 0]},
            {"sample_id": "s2", "bbox_3d": [0.5, 0, 0, 1, 1, 1, 0, 0, 0]},
            {"sample_id": "s3", "bbox_3d": [5, 0, 0, 1, 1, 1, 0, 0, 0]},
        ]
        metrics = evaluate_vg_predictions(predictions, samples)
        assert metrics["num_samples"] == 3
        assert metrics["acc_025"] == pytest.approx(2 / 3)
        assert metrics["acc_050"] == pytest.approx(1 / 3)
        assert metrics["mean_iou"] == pytest.approx(4 / 9, abs=0.02)

    def test_evaluate_vg_per_category(self) -> None:
        samples = [_sample("s1", "chair"), _sample("s2", "table")]
        predictions = [
            {"sample_id": "s1", "bbox_3d": [0, 0, 0, 1, 1, 1, 0, 0, 0]},
            {"sample_id": "s2", "bbox_3d": [5, 0, 0, 1, 1, 1, 0, 0, 0]},
        ]
        metrics = evaluate_vg_predictions(predictions, samples)
        assert metrics["per_category"]["chair"]["count"] == 1
        assert metrics["per_category"]["chair"]["acc_050"] == 1.0
        assert metrics["per_category"]["table"]["count"] == 1
        assert metrics["per_category"]["table"]["acc_025"] == 0.0

    def test_evaluate_vg_none_pred(self) -> None:
        metrics = evaluate_vg_predictions(
            [{"sample_id": "s1", "bbox_3d": None}],
            [_sample("s1")],
        )
        assert metrics["mean_iou"] == 0.0
        assert metrics["acc_025"] == 0.0

    def test_evaluate_vg_short_bbox(self) -> None:
        metrics = evaluate_vg_predictions(
            [{"sample_id": "s1", "bbox_3d": [0, 0, 0, 1, 1, 1]}],
            [_sample("s1")],
        )
        assert metrics["mean_iou"] == pytest.approx(1.0)
        assert metrics["acc_050"] == 1.0

    def test_evaluate_vg_invalid_bbox(self) -> None:
        metrics = evaluate_vg_predictions(
            [{"sample_id": "s1", "bbox_3d": [math.nan, 0, 0, 1, 1, 1]}],
            [_sample("s1")],
        )
        assert metrics["mean_iou"] == 0.0
        assert metrics["acc_050"] == 0.0
