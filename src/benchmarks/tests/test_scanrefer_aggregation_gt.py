"""Tests for the aggregation-based GT bbox loader.

Synthetic .ply + segs.json + aggregation.json + axis_align matrix in
tmp_path. Verifies the loader returns the expected AABB for each
objectId and handles edge cases (mismatch, missing file, axis-align off).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from benchmarks.scanrefer_aggregation_gt import (
    _aabb_to_9dof,
    _read_axis_alignment_matrix,
    load_aggregation_gt_bboxes,
)


def _write_ply(path: Path, vertices: np.ndarray) -> None:
    """Write a minimal ASCII .ply that open3d can read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(vertices)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "element face 0",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for v in vertices:
        lines.append(f"{v[0]} {v[1]} {v[2]}")
    path.write_text("\n".join(lines) + "\n")


def _build_synthetic_scene(tmp_path: Path, scene_id: str = "scene_test_00") -> tuple[Path, Path]:
    """Build aux + mesh dirs with 8 vertices split into 2 segments → 2 objectIds."""
    aux_dir = tmp_path / "scannet_aux" / scene_id
    aux_dir.mkdir(parents=True)
    mesh_dir = tmp_path / "scannet_aux_meshes" / scene_id
    mesh_dir.mkdir(parents=True)

    # 8 vertices: 4 forming objectId=0 cube at origin, 4 forming objectId=1 cube shifted
    verts = np.array(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0], [4.0, 3.0, 3.0], [3.0, 4.0, 3.0], [4.0, 4.0, 4.0],
        ],
        dtype=np.float32,
    )
    _write_ply(mesh_dir / f"{scene_id}_vh_clean_2.ply", verts)

    # segIndices: vertices 0-3 → seg_id=10, vertices 4-7 → seg_id=20
    (aux_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json").write_text(
        json.dumps({"segIndices": [10, 10, 10, 10, 20, 20, 20, 20]})
    )

    # aggregation: objectId 0 ← seg 10, objectId 1 ← seg 20
    (aux_dir / f"{scene_id}.aggregation.json").write_text(
        json.dumps(
            {
                "sceneId": scene_id,
                "segGroups": [
                    {"id": 0, "objectId": 0, "label": "thing0", "segments": [10]},
                    {"id": 1, "objectId": 1, "label": "thing1", "segments": [20]},
                ],
            }
        )
    )

    # Identity axis alignment (no transform)
    identity = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
    (aux_dir / f"{scene_id}.txt").write_text(f"axisAlignment = {identity}\n")

    return tmp_path / "scannet_aux", tmp_path / "scannet_aux_meshes"


def test_load_returns_one_bbox_per_object(tmp_path):
    aux_root, mesh_root = _build_synthetic_scene(tmp_path)
    out = load_aggregation_gt_bboxes(
        "scene_test_00",
        scannet_aux_root=aux_root,
        mesh_root=mesh_root,
    )
    assert set(out) == {0, 1}


def test_object_0_bbox_is_unit_cube(tmp_path):
    aux_root, mesh_root = _build_synthetic_scene(tmp_path)
    out = load_aggregation_gt_bboxes(
        "scene_test_00",
        scannet_aux_root=aux_root,
        mesh_root=mesh_root,
    )
    bb = out[0]
    # 4 verts at [(0,0,0), (1,0,0), (0,1,0), (1,1,1)] → AABB [(0,0,0),(1,1,1)]
    assert bb[0] == pytest.approx(0.5)
    assert bb[1] == pytest.approx(0.5)
    assert bb[2] == pytest.approx(0.5)
    assert bb[3] == pytest.approx(1.0)
    assert bb[4] == pytest.approx(1.0)
    assert bb[5] == pytest.approx(1.0)
    assert bb[6:] == [0.0, 0.0, 0.0]


def test_object_1_bbox_is_shifted_cube(tmp_path):
    aux_root, mesh_root = _build_synthetic_scene(tmp_path)
    out = load_aggregation_gt_bboxes(
        "scene_test_00",
        scannet_aux_root=aux_root,
        mesh_root=mesh_root,
    )
    bb = out[1]
    # AABB [(3,3,3),(4,4,4)] → center 3.5, size 1
    assert bb[:3] == [pytest.approx(3.5)] * 3
    assert bb[3:6] == [pytest.approx(1.0)] * 3


def test_axis_alignment_is_applied(tmp_path):
    """Replace identity axisAlignment with a 90deg rotation around z; bbox should rotate."""
    aux_root, mesh_root = _build_synthetic_scene(tmp_path)
    # Override axis-align to swap x ↔ y and shift z by 100
    align = "0 -1 0 0   1 0 0 0   0 0 1 100   0 0 0 1"
    (aux_root / "scene_test_00" / "scene_test_00.txt").write_text(
        f"axisAlignment = {align}\n"
    )
    out = load_aggregation_gt_bboxes(
        "scene_test_00",
        scannet_aux_root=aux_root,
        mesh_root=mesh_root,
        apply_axis_alignment=True,
    )
    bb = out[0]
    # Old AABB was [(0,0,0),(1,1,1)]; under rotation Rz(90deg)+T(0,0,100):
    # (x, y, z) → (-y, x, z+100). 4 corners become x∈[-1, 0], y∈[0, 1], z∈[100, 101]
    assert bb[:3] == [pytest.approx(-0.5), pytest.approx(0.5), pytest.approx(100.5)]


def test_apply_axis_alignment_false_returns_unrotated(tmp_path):
    aux_root, mesh_root = _build_synthetic_scene(tmp_path)
    align = "0 -1 0 0   1 0 0 0   0 0 1 100   0 0 0 1"
    (aux_root / "scene_test_00" / "scene_test_00.txt").write_text(
        f"axisAlignment = {align}\n"
    )
    out = load_aggregation_gt_bboxes(
        "scene_test_00",
        scannet_aux_root=aux_root,
        mesh_root=mesh_root,
        apply_axis_alignment=False,
    )
    # Without alignment, bbox stays at unit cube [(0,0,0),(1,1,1)]
    assert out[0][:3] == [pytest.approx(0.5)] * 3


def test_missing_mesh_raises_file_not_found(tmp_path):
    aux_root, mesh_root = _build_synthetic_scene(tmp_path)
    (mesh_root / "scene_test_00" / "scene_test_00_vh_clean_2.ply").unlink()
    with pytest.raises(FileNotFoundError, match=r"_vh_clean_2\.ply"):
        load_aggregation_gt_bboxes(
            "scene_test_00",
            scannet_aux_root=aux_root,
            mesh_root=mesh_root,
        )


def test_segs_vertex_count_mismatch_raises(tmp_path):
    aux_root, mesh_root = _build_synthetic_scene(tmp_path)
    # Truncate segIndices (now 4, but mesh has 8 verts)
    (aux_root / "scene_test_00" / "scene_test_00_vh_clean_2.0.010000.segs.json").write_text(
        json.dumps({"segIndices": [10, 10, 10, 10]})
    )
    with pytest.raises(ValueError, match="seg_indices mismatch"):
        load_aggregation_gt_bboxes(
            "scene_test_00",
            scannet_aux_root=aux_root,
            mesh_root=mesh_root,
        )


def test_empty_segGroup_silently_skipped(tmp_path):
    aux_root, mesh_root = _build_synthetic_scene(tmp_path)
    # Add a 3rd segGroup that references a seg id that no vertex has
    (aux_root / "scene_test_00" / "scene_test_00.aggregation.json").write_text(
        json.dumps(
            {
                "sceneId": "scene_test_00",
                "segGroups": [
                    {"id": 0, "objectId": 0, "label": "thing0", "segments": [10]},
                    {"id": 1, "objectId": 1, "label": "thing1", "segments": [20]},
                    {"id": 2, "objectId": 99, "label": "ghost", "segments": [9999]},
                ],
            }
        )
    )
    out = load_aggregation_gt_bboxes(
        "scene_test_00",
        scannet_aux_root=aux_root,
        mesh_root=mesh_root,
    )
    assert 99 not in out  # silently skipped (0 verts < 3)


def test_aabb_to_9dof_unit_cube_helper():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.float64)
    bb = _aabb_to_9dof(pts)
    assert bb == [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_read_axis_alignment_matrix_helper(tmp_path):
    p = tmp_path / "scene.txt"
    p.write_text("colorHeight = 968\n"
                 "axisAlignment = 1 0 0 5 0 1 0 6 0 0 1 7 0 0 0 1\n"
                 "colorWidth = 1296\n")
    mat = _read_axis_alignment_matrix(p)
    assert mat.shape == (4, 4)
    assert mat[0, 3] == 5.0
    assert mat[1, 3] == 6.0
    assert mat[2, 3] == 7.0
