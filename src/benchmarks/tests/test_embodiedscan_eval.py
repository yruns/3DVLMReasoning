"""Unit tests for EmbodiedScan evaluation (3D oriented IoU + VG metrics)."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.embodiedscan_eval import (
    compute_oriented_iou_3d,
    euler_to_rotation_matrix,
    evaluate_vg_predictions,
    oriented_bbox_to_corners,
)
from benchmarks.embodiedscan_loader import EmbodiedScanVGSample


# ---------------------------------------------------------------------------
# Tests: euler_to_rotation_matrix
# ---------------------------------------------------------------------------


class TestEulerToRotationMatrix:
    """Tests for ZXY Euler angle to rotation matrix conversion."""

    def test_identity(self) -> None:
        """Zero angles produce identity rotation."""
        R = euler_to_rotation_matrix(0.0, 0.0, 0.0)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_orthogonal(self) -> None:
        """Result is always a valid rotation matrix (R^T R = I, det = 1)."""
        for a, b, g in [(0.5, 0.3, 0.1), (1.57, 0.0, 0.0), (0, 0.5, 0.5)]:
            R = euler_to_rotation_matrix(a, b, g)
            np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=10)
            assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)

    def test_90_deg_alpha(self) -> None:
        """90-degree rotation around Z axis (alpha only)."""
        R = euler_to_rotation_matrix(np.pi / 2, 0.0, 0.0)
        # Z-rotation by pi/2: x→y, y→-x, z→z
        point = np.array([1.0, 0.0, 0.0])
        rotated = R @ point
        np.testing.assert_array_almost_equal(rotated, [0.0, 1.0, 0.0], decimal=10)

    def test_180_deg_alpha(self) -> None:
        """180-degree rotation around Z axis."""
        R = euler_to_rotation_matrix(np.pi, 0.0, 0.0)
        point = np.array([1.0, 0.0, 0.0])
        rotated = R @ point
        np.testing.assert_array_almost_equal(rotated, [-1.0, 0.0, 0.0], decimal=10)


# ---------------------------------------------------------------------------
# Tests: oriented_bbox_to_corners
# ---------------------------------------------------------------------------


class TestOrientedBboxToCorners:
    """Tests for 9-DOF bbox to 8-corner conversion."""

    def test_shape(self) -> None:
        bbox = [0, 0, 0, 1, 1, 1, 0, 0, 0]
        corners = oriented_bbox_to_corners(bbox)
        assert corners.shape == (8, 3)

    def test_axis_aligned_unit_cube(self) -> None:
        """Unit cube at origin with no rotation."""
        bbox = [0, 0, 0, 1, 1, 1, 0, 0, 0]
        corners = oriented_bbox_to_corners(bbox)
        # All corners should be at ±0.5
        assert corners.min() == pytest.approx(-0.5)
        assert corners.max() == pytest.approx(0.5)

    def test_translated_box(self) -> None:
        """Box center translation shifts all corners."""
        bbox = [10, 20, 30, 1, 1, 1, 0, 0, 0]
        corners = oriented_bbox_to_corners(bbox)
        center = corners.mean(axis=0)
        np.testing.assert_array_almost_equal(center, [10, 20, 30])

    def test_scaled_box(self) -> None:
        """Different dimensions produce correct corner spread."""
        bbox = [0, 0, 0, 2, 4, 6, 0, 0, 0]
        corners = oriented_bbox_to_corners(bbox)
        # Extent along each axis = dimension
        extent_x = corners[:, 0].max() - corners[:, 0].min()
        extent_y = corners[:, 1].max() - corners[:, 1].min()
        extent_z = corners[:, 2].max() - corners[:, 2].min()
        assert extent_x == pytest.approx(2.0)
        assert extent_y == pytest.approx(4.0)
        assert extent_z == pytest.approx(6.0)

    def test_rotated_box_preserves_center(self) -> None:
        """Rotation doesn't change center of corners."""
        bbox = [5, 5, 5, 2, 3, 4, 1.0, 0.5, 0.3]
        corners = oriented_bbox_to_corners(bbox)
        center = corners.mean(axis=0)
        np.testing.assert_array_almost_equal(center, [5, 5, 5], decimal=10)


# ---------------------------------------------------------------------------
# Tests: compute_oriented_iou_3d
# ---------------------------------------------------------------------------


class TestComputeOrientedIou3d:
    """Tests for oriented 3D IoU computation."""

    def test_identical_boxes(self) -> None:
        """Identical boxes should have IoU = 1.0."""
        bbox = [0, 0, 0, 1, 1, 1, 0, 0, 0]
        iou = compute_oriented_iou_3d(bbox, bbox)
        assert iou == pytest.approx(1.0, abs=0.01)

    def test_identical_rotated_boxes(self) -> None:
        """Identical rotated boxes should have IoU = 1.0."""
        bbox = [0, 0, 0, 2, 3, 4, 1.57, 0, 0]
        iou = compute_oriented_iou_3d(bbox, bbox)
        assert iou == pytest.approx(1.0, abs=0.01)

    def test_no_overlap(self) -> None:
        """Non-overlapping boxes should have IoU = 0.0."""
        bbox1 = [0, 0, 0, 1, 1, 1, 0, 0, 0]
        bbox2 = [10, 10, 10, 1, 1, 1, 0, 0, 0]
        iou = compute_oriented_iou_3d(bbox1, bbox2)
        assert iou == pytest.approx(0.0, abs=0.01)

    def test_half_overlap_axis_aligned(self) -> None:
        """Two axis-aligned boxes overlapping by half along X axis."""
        bbox1 = [0, 0, 0, 2, 2, 2, 0, 0, 0]
        bbox2 = [1, 0, 0, 2, 2, 2, 0, 0, 0]
        # Overlap region: x=[0,1], y=[-1,1], z=[-1,1] → volume = 1*2*2 = 4
        # Union: 8 + 8 - 4 = 12
        # IoU = 4/12 = 1/3
        iou = compute_oriented_iou_3d(bbox1, bbox2)
        assert iou == pytest.approx(1.0 / 3.0, abs=0.02)

    def test_contained_box(self) -> None:
        """Smaller box fully inside larger box."""
        bbox_large = [0, 0, 0, 4, 4, 4, 0, 0, 0]
        bbox_small = [0, 0, 0, 1, 1, 1, 0, 0, 0]
        # intersection = 1, union = 64 + 1 - 1 = 64
        iou = compute_oriented_iou_3d(bbox_large, bbox_small)
        assert iou == pytest.approx(1.0 / 64.0, abs=0.02)

    def test_symmetric(self) -> None:
        """IoU(A, B) == IoU(B, A)."""
        bbox1 = [0, 0, 0, 2, 3, 1, 0.5, 0.0, 0.0]
        bbox2 = [1, 0, 0, 2, 2, 2, 0.3, 0.0, 0.0]
        iou_ab = compute_oriented_iou_3d(bbox1, bbox2)
        iou_ba = compute_oriented_iou_3d(bbox2, bbox1)
        assert iou_ab == pytest.approx(iou_ba, abs=0.01)

    def test_touching_boxes(self) -> None:
        """Boxes that just touch at a face have IoU ~= 0."""
        bbox1 = [0, 0, 0, 2, 2, 2, 0, 0, 0]
        bbox2 = [2, 0, 0, 2, 2, 2, 0, 0, 0]
        iou = compute_oriented_iou_3d(bbox1, bbox2)
        assert iou == pytest.approx(0.0, abs=0.02)

    def test_rotated_overlap(self) -> None:
        """Rotated box overlapping with axis-aligned box should have 0 < IoU < 1."""
        bbox1 = [0, 0, 0, 2, 2, 2, 0, 0, 0]
        bbox2 = [0, 0, 0, 2, 2, 2, np.pi / 4, 0, 0]  # 45-deg yaw
        iou = compute_oriented_iou_3d(bbox1, bbox2)
        assert 0.0 < iou < 1.0

    def test_real_embodiedscan_bbox_format(self) -> None:
        """Test with realistic 9-DOF bboxes from EmbodiedScan data."""
        bbox1 = [0.37, -0.82, 0.91, 0.26, 0.26, 0.15, 1.57, 0.0, 0.0]
        bbox2 = [-0.48, -0.50, 0.92, 0.21, 0.32, 0.14, 1.57, 0.0, 0.0]
        iou = compute_oriented_iou_3d(bbox1, bbox2)
        assert 0.0 <= iou <= 1.0


# ---------------------------------------------------------------------------
# Tests: evaluate_vg_predictions
# ---------------------------------------------------------------------------


class TestEvaluateVgPredictions:
    """Tests for VG evaluation metrics."""

    @staticmethod
    def _make_sample(
        sample_id: str,
        target: str = "chair",
        gt_bbox: list[float] | None = None,
    ) -> EmbodiedScanVGSample:
        if gt_bbox is None:
            gt_bbox = [0, 0, 0, 1, 1, 1, 0, 0, 0]
        return EmbodiedScanVGSample(
            sample_id=sample_id,
            scene_id="scene0001_00",
            query="the chair",
            target=target,
            gt_bbox_3d=gt_bbox,
        )

    def test_perfect_predictions(self) -> None:
        """All predictions match GT exactly."""
        samples = [
            self._make_sample("s1", "chair"),
            self._make_sample("s2", "table"),
        ]
        predictions = [
            {"sample_id": "s1", "bbox_3d": [0, 0, 0, 1, 1, 1, 0, 0, 0]},
            {"sample_id": "s2", "bbox_3d": [0, 0, 0, 1, 1, 1, 0, 0, 0]},
        ]
        metrics = evaluate_vg_predictions(predictions, samples)
        assert metrics["acc_025"] == pytest.approx(1.0)
        assert metrics["acc_050"] == pytest.approx(1.0)
        assert metrics["num_samples"] == 2

    def test_all_wrong_predictions(self) -> None:
        """All predictions are far from GT."""
        samples = [self._make_sample("s1")]
        predictions = [
            {"sample_id": "s1", "bbox_3d": [100, 100, 100, 1, 1, 1, 0, 0, 0]},
        ]
        metrics = evaluate_vg_predictions(predictions, samples)
        assert metrics["acc_025"] == pytest.approx(0.0)
        assert metrics["acc_050"] == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Prediction partially overlaps GT (above 0.25 but below 0.50)."""
        samples = [self._make_sample("s1")]
        # Half-overlap along x → IoU ~ 1/3 ≈ 0.33
        predictions = [
            {"sample_id": "s1", "bbox_3d": [0.5, 0, 0, 1, 1, 1, 0, 0, 0]},
        ]
        metrics = evaluate_vg_predictions(predictions, samples)
        assert metrics["acc_025"] == pytest.approx(1.0)
        assert metrics["acc_050"] == pytest.approx(0.0)

    def test_per_category_breakdown(self) -> None:
        """Per-category metrics are computed correctly."""
        samples = [
            self._make_sample("s1", "chair"),
            self._make_sample("s2", "table"),
        ]
        predictions = [
            {"sample_id": "s1", "bbox_3d": [0, 0, 0, 1, 1, 1, 0, 0, 0]},
            {"sample_id": "s2", "bbox_3d": [100, 0, 0, 1, 1, 1, 0, 0, 0]},
        ]
        metrics = evaluate_vg_predictions(predictions, samples)
        assert "per_category" in metrics
        assert metrics["per_category"]["chair"]["acc_025"] == pytest.approx(1.0)
        assert metrics["per_category"]["table"]["acc_025"] == pytest.approx(0.0)

    def test_missing_prediction_bbox(self) -> None:
        """Prediction with None bbox_3d counts as zero IoU."""
        samples = [self._make_sample("s1")]
        predictions = [{"sample_id": "s1", "bbox_3d": None}]
        metrics = evaluate_vg_predictions(predictions, samples)
        assert metrics["acc_025"] == pytest.approx(0.0)

    def test_empty_inputs(self) -> None:
        metrics = evaluate_vg_predictions([], [])
        assert metrics["num_samples"] == 0
        assert metrics["acc_025"] == pytest.approx(0.0)

    def test_mean_iou(self) -> None:
        """mean_iou is the average IoU across all samples."""
        samples = [
            self._make_sample("s1"),
            self._make_sample("s2"),
        ]
        predictions = [
            {"sample_id": "s1", "bbox_3d": [0, 0, 0, 1, 1, 1, 0, 0, 0]},  # IoU=1
            {"sample_id": "s2", "bbox_3d": [100, 0, 0, 1, 1, 1, 0, 0, 0]},  # IoU=0
        ]
        metrics = evaluate_vg_predictions(predictions, samples)
        assert metrics["mean_iou"] == pytest.approx(0.5, abs=0.02)
