"""Tests for ScanRefer Mask3D-CG converter pure helpers."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.build_scanrefer_mask3d_cg import (
    BACKGROUND_LABELS,
    axis_aligned_corners_from_pcd,
    build_object_dict,
    is_background_label,
    load_scannet200_class_index,
    scannet200_class_id,
)


def test_axis_aligned_corners_from_unit_cube_pcd():
    """Unit cube with corners at (0,0,0)-(1,1,1) → 8 corners spanning that AABB."""
    pcd = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    corners = axis_aligned_corners_from_pcd(pcd)
    assert corners.shape == (8, 3)
    assert corners.min(axis=0).tolist() == [0.0, 0.0, 0.0]
    assert corners.max(axis=0).tolist() == [1.0, 1.0, 1.0]


def test_axis_aligned_corners_uses_first_three_columns_only():
    """If pcd is (N, 6) [xyz+rgb], rgb is ignored."""
    pcd = np.array(
        [
            [0.0, 0.0, 0.0, 100.0, 100.0, 100.0],
            [2.0, 3.0, 4.0, 200.0, 200.0, 200.0],
        ],
        dtype=np.float32,
    )
    corners = axis_aligned_corners_from_pcd(pcd)
    assert corners.shape == (8, 3)
    assert corners.min(axis=0).tolist() == [0.0, 0.0, 0.0]
    assert corners.max(axis=0).tolist() == [2.0, 3.0, 4.0]


def test_axis_aligned_corners_raises_on_empty_pcd():
    with pytest.raises(ValueError, match="empty"):
        axis_aligned_corners_from_pcd(np.zeros((0, 3), dtype=np.float32))


def test_background_labels_set():
    assert BACKGROUND_LABELS == frozenset({"wall", "floor", "ceiling"})


def test_is_background_label_case_insensitive():
    assert is_background_label("wall") is True
    assert is_background_label("WALL") is True
    assert is_background_label("Floor") is True
    assert is_background_label("chair") is False
    assert is_background_label("") is False


def test_load_scannet200_class_index_returns_lowercased_label_to_idx_dict(tmp_path):
    """Each line of the canonical file is one class name; index is line number (0-based)."""
    txt = tmp_path / "scannet200_classes.txt"
    txt.write_text("alarm clock\narmchair\nchair\n", encoding="utf-8")
    idx = load_scannet200_class_index(txt)
    assert idx == {"alarm clock": 0, "armchair": 1, "chair": 2}


def test_scannet200_class_id_known_label():
    idx = {"chair": 2, "table": 5}
    assert scannet200_class_id("chair", idx) == 2
    assert scannet200_class_id("CHAIR", idx) == 2  # case-insensitive


def test_scannet200_class_id_unknown_returns_minus_one():
    idx = {"chair": 2}
    assert scannet200_class_id("desk", idx) == -1


def test_build_object_dict_minimal_schema():
    """Build a Phase-8-shaped object dict from one Mask3D instance."""
    pcd = np.array(
        [[0.0, 0.0, 0.0, 50.0, 60.0, 70.0],
         [1.0, 1.0, 1.0, 80.0, 90.0, 100.0]],
        dtype=np.float32,
    )
    obj = build_object_dict(
        pcd_with_color=pcd,
        label="chair",
        class_idx=2,
        confidence=0.95,
    )
    assert obj["class_name"] == ["chair"]
    assert obj["class_id"] == [2]
    assert obj["is_background"] == 0
    assert obj["num_detections"] == 1
    assert obj["n_points"] == [2]
    assert obj["conf"] == [pytest.approx(0.95)]
    assert obj["bbox_np"].shape == (8, 3)
    assert obj["pcd_np"].shape == (2, 3)
    assert obj["pcd_color_np"].shape == (2, 3)


def test_build_object_dict_pcd_no_color():
    """Mask3D distribution always emits 6-D, but be defensive: 3-D (xyz only) ok."""
    pcd = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    obj = build_object_dict(
        pcd_with_color=pcd,
        label="chair",
        class_idx=2,
        confidence=0.95,
    )
    assert obj["pcd_color_np"] is None
