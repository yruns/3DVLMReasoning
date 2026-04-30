from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from scripts.nr3d_gt_conceptgraph import (
    CONCEPTGRAPH_OBJECT_FIELDS,
    GtObject,
    RawScene,
    _add_projection_fallback_views,
    build_cfg,
    build_gt_objects_for_scene,
    build_object_dict,
    build_payload,
    load_scene_ids,
    parse_axis_alignment,
    stable_class_colors,
)


def _write_ascii_ply(path: Path, vertices: list[tuple[float, float, float, int, int, int]]) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property uchar alpha",
        "element face 0",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend(
        f"{x} {y} {z} {r} {g} {b} 255" for x, y, z, r, g, b in vertices
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cube_vertices() -> list[tuple[float, float, float, int, int, int]]:
    coords = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ]
    return [(x, y, z, 10 + i, 20 + i, 30 + i) for i, (x, y, z) in enumerate(coords)]


def test_load_scene_ids_requires_json_array_without_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "test_scans.txt"
    path.write_text(json.dumps(["scene0001_00", "scene0002_00"]), encoding="utf-8")

    assert load_scene_ids(path) == ["scene0001_00", "scene0002_00"]

    path.write_text(json.dumps(["scene0001_00", "scene0001_00"]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_scene_ids(path)


def test_parse_axis_alignment_reads_4x4_matrix(tmp_path: Path) -> None:
    path = tmp_path / "scene0001_00.txt"
    path.write_text(
        "axisAlignment = 1 0 0 2 0 1 0 3 0 0 1 4 0 0 0 1\n"
        "colorWidth = 1296\n",
        encoding="utf-8",
    )

    matrix = parse_axis_alignment(path)

    assert matrix.shape == (4, 4)
    np.testing.assert_allclose(matrix[:3, 3], [2.0, 3.0, 4.0])


def test_build_gt_objects_for_scene_uses_segments_and_axis_alignment(tmp_path: Path) -> None:
    aux_root = tmp_path / "aux"
    scene_aux = aux_root / "scene0001_00"
    scene_aux.mkdir(parents=True)
    scannet_root = tmp_path / "ScanNet"
    scene_scan = scannet_root / "scans" / "scene0001_00"
    scene_scan.mkdir(parents=True)

    (scene_aux / "scene0001_00.txt").write_text(
        "axisAlignment = 1 0 0 10 0 1 0 20 0 0 1 30 0 0 0 1\n",
        encoding="utf-8",
    )
    (scene_aux / "scene0001_00.aggregation.json").write_text(
        json.dumps(
            {
                "sceneId": "scannet.scene0001_00",
                "segGroups": [
                    {"id": 7, "objectId": 42, "segments": [5], "label": "chair"}
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_aux / "scene0001_00_vh_clean_2.0.010000.segs.json").write_text(
        json.dumps({"sceneId": "scene0001_00", "segIndices": [5] * 8}),
        encoding="utf-8",
    )
    _write_ascii_ply(scene_scan / "scene0001_00_vh_clean_2.ply", _cube_vertices())

    objects = build_gt_objects_for_scene(
        "scene0001_00",
        aux_root=aux_root,
        scannet_root=scannet_root,
        class_to_idx={"chair": 3},
    )

    assert len(objects) == 1
    obj = objects[0]
    assert obj.object_id == 42
    assert obj.label == "chair"
    assert obj.class_id == 3
    assert obj.is_background == 0
    assert obj.bbox_np.shape == (8, 3)
    assert obj.bbox_np.dtype == np.float64
    np.testing.assert_allclose(obj.pcd_np.min(axis=0), [10.0, 20.0, 30.0])
    np.testing.assert_allclose(obj.pcd_np.max(axis=0), [11.0, 21.0, 31.0])
    np.testing.assert_allclose(obj.pcd_color_np[0], [10 / 255, 20 / 255, 30 / 255])


def test_build_object_dict_populates_exact_conceptgraph_fields(tmp_path: Path) -> None:
    rgb = tmp_path / "000000-rgb.png"
    rgb.write_bytes(b"not used by this helper")
    pcd = np.array(
        [
            [-0.5, -0.5, 4.5],
            [0.5, -0.5, 4.5],
            [-0.5, 0.5, 4.5],
            [0.5, 0.5, 5.5],
        ],
        dtype=np.float64,
    )
    bbox = np.array(
        [
            [-0.5, -0.5, 4.5],
            [0.5, -0.5, 4.5],
            [-0.5, 0.5, 4.5],
            [0.5, 0.5, 4.5],
            [-0.5, -0.5, 5.5],
            [0.5, -0.5, 5.5],
            [-0.5, 0.5, 5.5],
            [0.5, 0.5, 5.5],
        ],
        dtype=np.float64,
    )
    gt = GtObject(
        object_id=9,
        label="chair",
        class_id=2,
        bbox_np=bbox,
        pcd_np=pcd,
        pcd_color_np=np.ones((4, 3), dtype=np.float64),
        is_background=0,
    )
    intrinsic = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    obj = build_object_dict(
        gt,
        visible_views=[(0, 0.75)],
        poses=[np.eye(4)],
        intrinsic=intrinsic,
        image_size=(100, 80),
        raw_rgb_paths=[rgb],
        inst_color=[0.1, 0.2, 0.3],
        clip_ft=np.ones(1024, dtype=np.float32),
        text_ft=np.full(1024, 2.0, dtype=np.float32),
    )

    assert set(obj.keys()) == CONCEPTGRAPH_OBJECT_FIELDS
    assert obj["image_idx"] == [0]
    assert obj["class_name"] == ["chair"]
    assert obj["class_id"] == [2]
    assert obj["num_detections"] == 1
    assert obj["mask"][0].shape == (80, 100)
    assert obj["mask"][0].dtype == np.bool_
    assert obj["mask"][0].any()
    assert obj["xyxy"][0].dtype == np.float32
    assert obj["clip_ft"].shape == (1024,)
    assert obj["text_ft"].shape == (1024,)


def test_build_payload_round_trips_with_open_eqa_top_level_shape(tmp_path: Path) -> None:
    cfg = build_cfg(scene_id="scene0001_00", image_width=100, image_height=80)
    class_names = ["chair"]
    class_colors = stable_class_colors(class_names)
    obj = {
        field: None for field in CONCEPTGRAPH_OBJECT_FIELDS
    }
    obj.update(
        {
            "bbox_np": np.zeros((8, 3), dtype=np.float64),
            "pcd_np": np.zeros((1, 3), dtype=np.float64),
            "pcd_color_np": np.zeros((1, 3), dtype=np.float64),
            "clip_ft": np.ones(1024, dtype=np.float32),
            "text_ft": np.ones(1024, dtype=np.float32),
        }
    )

    payload = build_payload([obj], cfg, class_names, class_colors)
    output = tmp_path / "full_pcd_gt_axisaligned_post.pkl.gz"
    with gzip.open(output, "wb") as handle:
        pickle.dump(payload, handle)
    with gzip.open(output, "rb") as handle:
        loaded = pickle.load(handle)

    assert list(loaded.keys()) == ["objects", "bg_objects", "cfg", "class_names", "class_colors"]
    assert loaded["bg_objects"] is None
    assert loaded["cfg"].gsa_variant == "gt"
    assert loaded["class_names"] == ["chair"]
    assert loaded["class_colors"] == {"0": class_colors["0"]}


def test_projection_fallback_adds_view_for_zero_visibility_object(
    tmp_path: Path,
) -> None:
    gt = GtObject(
        object_id=25,
        label="ceiling",
        class_id=0,
        bbox_np=np.array(
            [
                [-0.5, -0.5, 3.0],
                [0.5, -0.5, 3.0],
                [-0.5, 0.5, 3.0],
                [0.5, 0.5, 3.0],
                [-0.5, -0.5, 3.2],
                [0.5, -0.5, 3.2],
                [-0.5, 0.5, 3.2],
                [0.5, 0.5, 3.2],
            ],
            dtype=np.float64,
        ),
        pcd_np=np.array([[0.0, 0.0, 3.1]], dtype=np.float64),
        pcd_color_np=np.ones((1, 3), dtype=np.float64),
        is_background=1,
    )
    raw_scene = RawScene(
        scene_id="scene0001_00",
        raw_dir=tmp_path,
        rgb_paths=[],
        depth_paths=[],
        pose_paths=[],
        poses=[np.eye(4, dtype=np.float64)],
        intrinsic_color=np.array(
            [
                [100.0, 0.0, 50.0, 0.0],
                [0.0, 100.0, 40.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        intrinsic_depth=np.eye(4, dtype=np.float64),
        image_width=100,
        image_height=80,
        depth_width=100,
        depth_height=80,
    )
    object_to_views = {0: []}
    view_to_objects: dict[int, list[tuple[int, float]]] = {}

    fallbacks, unobserved = _add_projection_fallback_views(
        gt_objects=[gt],
        raw_scene=raw_scene,
        object_to_views=object_to_views,
        view_to_objects=view_to_objects,
    )

    assert unobserved == []
    assert fallbacks[0]["object_id"] == 25
    assert object_to_views[0][0][0] == 0
    assert view_to_objects[0][0][0] == 0


def test_projection_fallback_allows_unobserved_background_object(
    tmp_path: Path,
) -> None:
    gt = GtObject(
        object_id=28,
        label="ceiling",
        class_id=0,
        bbox_np=np.array(
            [
                [10.0, 10.0, -3.2],
                [11.0, 10.0, -3.2],
                [10.0, 11.0, -3.2],
                [11.0, 11.0, -3.2],
                [10.0, 10.0, -3.0],
                [11.0, 10.0, -3.0],
                [10.0, 11.0, -3.0],
                [11.0, 11.0, -3.0],
            ],
            dtype=np.float64,
        ),
        pcd_np=np.array([[10.5, 10.5, -3.1]], dtype=np.float64),
        pcd_color_np=np.ones((1, 3), dtype=np.float64),
        is_background=1,
    )
    raw_scene = RawScene(
        scene_id="scene0001_00",
        raw_dir=tmp_path,
        rgb_paths=[],
        depth_paths=[],
        pose_paths=[],
        poses=[np.eye(4, dtype=np.float64)],
        intrinsic_color=np.eye(4, dtype=np.float64),
        intrinsic_depth=np.eye(4, dtype=np.float64),
        image_width=100,
        image_height=80,
        depth_width=100,
        depth_height=80,
    )

    fallbacks, unobserved = _add_projection_fallback_views(
        gt_objects=[gt],
        raw_scene=raw_scene,
        object_to_views={0: []},
        view_to_objects={},
    )

    assert fallbacks == []
    assert unobserved[0]["object_id"] == 28
