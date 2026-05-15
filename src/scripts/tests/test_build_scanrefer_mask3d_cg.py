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


def test_load_scannet200_class_index_skips_blank_lines_without_offset(tmp_path):
    """Blank lines in the canonical file must not advance the canonical index counter."""
    txt = tmp_path / "scannet200_classes.txt"
    txt.write_text("alarm clock\n\nchair\n\n\narmchair\n", encoding="utf-8")
    idx = load_scannet200_class_index(txt)
    assert idx == {"alarm clock": 0, "chair": 1, "armchair": 2}


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


def test_build_one_scene_smoke(tmp_path):
    """Smoke test: synthetic mask3d npz + minimal raw dir → produced pkl + visibility."""
    import json

    from scripts.build_scanrefer_mask3d_cg import build_mask3d_cg_for_scene

    # Synthetic Mask3D npz: 2 instances (1 chair, 1 wall — wall should be filtered)
    npz_path = tmp_path / "scene_test.npz"
    pcd_chair = np.random.rand(100, 6).astype(np.float32) * np.array([1, 1, 1, 255, 255, 255])
    pcd_wall = np.random.rand(50, 6).astype(np.float32) * np.array([3, 3, 3, 255, 255, 255])
    np.savez_compressed(
        npz_path,
        ins_pcds=np.array([pcd_chair, pcd_wall], dtype=object),
        ins_labels=np.array(["chair", "wall"], dtype="<U16"),
        ins_scores=np.array([0.9, 0.7], dtype=np.float32),
    )

    # Synthetic raw dir with 1 frame: identity pose, identity intrinsic, far-wall depth.
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    np.savetxt(raw_dir / "intrinsic_color.txt",
               np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64))
    np.savetxt(raw_dir / "000000.txt", np.eye(4, dtype=np.float64))
    # Synthetic depth: 480x640 uint16 PNG with all pixels = 60000 (=60m at depth_scale=1000).
    # The visibility index requires a real depth map; "far wall" semantics mean every
    # projected object centroid is treated as closer than the depth → visible.
    import cv2
    depth_arr = np.full((480, 640), 60000, dtype=np.uint16)
    cv2.imwrite(str(raw_dir / "000000-depth.png"), depth_arr)
    # scene_info.json with kept_frame_ids=[0]
    (raw_dir / "scene_info.json").write_text(
        json.dumps({"kept_frame_ids": [0]}), encoding="utf-8"
    )

    # Outputs
    pkl_out = tmp_path / "out.pkl.gz"
    vis_out = tmp_path / "vis.pkl"
    info_out = tmp_path / "scene_info.json"

    summary = build_mask3d_cg_for_scene(
        scene_id="scene_test",
        mask3d_npz_path=npz_path,
        raw_dir=raw_dir,
        output_pkl=pkl_out,
        output_visibility=vis_out,
        output_scene_info=info_out,
        scannet200_taxonomy=tmp_path / "scannet200_classes.txt",
        drop_background=True,
    )
    assert summary["n_kept"] == 1   # chair kept, wall dropped
    assert summary["n_dropped"] == 1
    assert pkl_out.exists()
    assert vis_out.exists()
    assert info_out.exists()


def test_load_scannet200_taxonomy_uses_default_taxonomy(tmp_path, monkeypatch):
    """If --scannet200-taxonomy isn't provided, code should default to repo file."""
    from scripts.build_scanrefer_mask3d_cg import DEFAULT_SCANNET200_TAXONOMY
    assert DEFAULT_SCANNET200_TAXONOMY.name == "scannet200_classes.txt"
