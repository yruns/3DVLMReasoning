"""Hermetic tests for NR3D pack-v1 input preparation."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image


def _corners(
    center: tuple[float, float, float] = (0.0, 0.0, 5.0),
    size: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    cx, cy, cz = center
    dx, dy, dz = size
    mins = np.array([cx - dx / 2, cy - dy / 2, cz - dz / 2], dtype=np.float64)
    maxs = np.array([cx + dx / 2, cy + dy / 2, cz + dz / 2], dtype=np.float64)
    return np.array(
        [
            [x, y, z]
            for x in (mins[0], maxs[0])
            for y in (mins[1], maxs[1])
            for z in (mins[2], maxs[2])
        ],
        dtype=np.float64,
    )


def _write_phase8_tree(
    data_root: Path,
    *,
    scene_id: str = "scene0001_00",
    objects: list[dict[str, Any]] | None = None,
    visibility: dict[str, Any] | None = None,
) -> Path:
    scene_root = data_root / scene_id
    raw = scene_root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for frame_id in (0, 10):
        Image.new("RGB", (100, 100), color="white").save(
            raw / f"{frame_id:06d}-rgb.png"
        )
        np.savetxt(raw / f"{frame_id:06d}.txt", np.eye(4))
    np.savetxt(
        raw / "intrinsic_color.txt",
        np.array(
            [[50, 0, 50, 0], [0, 50, 50, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
        ),
    )
    (raw / "scene_info.json").write_text(
        json.dumps(
            {"scene_id": scene_id, "frame_stride": 10, "kept_frame_ids": [0, 10]}
        ),
        encoding="utf-8",
    )

    cg = scene_root / "conceptgraph"
    pkl_path = cg / "pcd_saves" / "full_pcd_gt_axisaligned_post.pkl.gz"
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "objects": objects
                or [
                    {
                        "bbox_np": _corners(center=(0.0, 0.0, 5.0)),
                        "class_name": ["chair"],
                        "class_id": [7],
                    },
                    {
                        "bbox_np": _corners(center=(1.0, 0.0, 5.0)),
                        "class_name": ["table"],
                        "class_id": [12],
                    },
                ],
                "bg_objects": None,
                "cfg": {},
                "class_names": [],
                "class_colors": {},
            },
            f,
        )

    vis_path = cg / "indices" / "visibility_index.pkl"
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vis_path, "wb") as f:
        pickle.dump(
            visibility
            or {
                "object_to_views": {0: [(0, 0.9), (1, 0.8)], 1: [(1, 0.7)]},
                "view_to_objects": {0: [(0, 0.9)], 1: [(0, 0.8), (1, 0.7)]},
                "metadata": {},
            },
            f,
        )
    return scene_root


def _write_sample_ids(
    path: Path,
    sample_id: str = "scannet/scene0001_00::0::A1",
) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "sample_id": sample_id,
                    "scene_id": "scene0001_00",
                    "target_id": 0,
                    "category": "chair",
                }
            ]
        ),
        encoding="utf-8",
    )


def test_prepare_pack_v1_inputs_nr3d_smoke(tmp_path, monkeypatch) -> None:
    from evaluation.scripts import prepare_pack_v1_inputs_nr3d as prep

    data_root = tmp_path / "scannet"
    _write_phase8_tree(data_root)
    sample_ids = tmp_path / "sample_ids.json"
    _write_sample_ids(sample_ids)
    sample = SimpleNamespace(
        sample_id="scannet/scene0001_00::0::A1",
        scene_id="scene0001_00",
        scan_id="scannet/scene0001_00",
        target_id=0,
        target="chair",
        query="the chair by the table",
        gt_bbox_3d=[0.0, 0.0, 5.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    )
    monkeypatch.setattr(
        prep,
        "load_sample_lookup",
        lambda *, nr3d_root, phase8_data_root, split, sample_ids=None: (
            SimpleNamespace(),
            {sample.sample_id: sample},
        ),
    )

    written = prep.prepare_pack_v1_inputs_nr3d(
        sample_ids_path=sample_ids,
        data_root=data_root,
        pack_name="pack_nr3d_v1",
        split="test",
    )

    scene_dir = data_root / "scene0001_00" / "pack_nr3d_v1"
    sample_json = scene_dir / "samples" / "scannet__scene0001_00__0__A1.json"
    assert written == [sample_json]
    proposals_payload = json.loads((scene_dir / "proposals.jsonl").read_text())
    assert proposals_payload["source"] == "gt"
    assert [p["id"] for p in proposals_payload["proposals"]] == [0, 1]
    assert proposals_payload["proposals"][0]["label"] == "chair"
    assert json.loads((scene_dir / "visibility.json").read_text()) == {
        "0": [0],
        "1": [0, 1],
    }
    assert (scene_dir / "annotated" / "frame_0.png").exists()
    payload = json.loads(sample_json.read_text())
    assert payload["sample_id"] == "scannet/scene0001_00::0::A1"
    assert payload["query"] == "the chair by the table"
    assert payload["keyframes"] == [
        {
            "keyframe_idx": 0,
            "image_path": str(scene_dir / "annotated" / "frame_0.png"),
            "frame_id": 0,
        },
        {
            "keyframe_idx": 1,
            "image_path": str(scene_dir / "annotated" / "frame_1.png"),
            "frame_id": 1,
        },
    ]


def test_sample_artifact_path_sanitizes_scannet_sample_id(tmp_path) -> None:
    from evaluation.scripts.prepare_pack_v1_inputs_nr3d import (
        SampleRequest,
        sample_artifact_path,
    )

    request = SampleRequest(
        sample_id="scannet/scene0001_00::3::assignment-7",
        scene_id="scene0001_00",
        target_id=3,
        category="chair",
    )

    assert sample_artifact_path(
        tmp_path,
        request,
        pack_name="pack_nr3d_v1",
    ) == (
        tmp_path
        / "scene0001_00"
        / "pack_nr3d_v1"
        / "samples"
        / "scannet__scene0001_00__3__assignment-7.json"
    )


def test_select_keyframes_uses_phase8_object_to_views(tmp_path) -> None:
    from evaluation.scripts.prepare_pack_v1_inputs_nr3d import (
        load_phase8_visibility_index,
        select_keyframes_for_sample,
    )

    data_root = tmp_path / "scannet"
    _write_phase8_tree(
        data_root,
        visibility={
            "object_to_views": {0: [(4, 0.9), (2, 0.8), (9, 0.7), (1, 0.6)]},
            "view_to_objects": {
                1: [(0, 0.6)],
                2: [(0, 0.8)],
                4: [(0, 0.9)],
                9: [(0, 0.7)],
            },
            "metadata": {},
        },
    )
    raw = data_root / "scene0001_00" / "raw"
    for frame_id in (20, 40, 90):
        Image.new("RGB", (100, 100), color="white").save(
            raw / f"{frame_id:06d}-rgb.png"
        )
        np.savetxt(raw / f"{frame_id:06d}.txt", np.eye(4))
    (raw / "scene_info.json").write_text(
        json.dumps(
            {
                "scene_id": "scene0001_00",
                "frame_stride": 10,
                "kept_frame_ids": list(range(0, 100, 10)),
            }
        ),
        encoding="utf-8",
    )
    visibility = load_phase8_visibility_index(data_root / "scene0001_00")

    keyframes = select_keyframes_for_sample(
        scene_root=data_root / "scene0001_00",
        target_id=0,
        visibility=visibility,
        k=3,
    )

    assert [item["frame_id"] for item in keyframes] == [4, 2, 9]
    assert keyframes[0]["image_path"].endswith("000040-rgb.png")


def test_prepare_raises_for_unknown_requested_sample(tmp_path, monkeypatch) -> None:
    from evaluation.scripts import prepare_pack_v1_inputs_nr3d as prep

    data_root = tmp_path / "scannet"
    _write_phase8_tree(data_root)
    sample_ids = tmp_path / "sample_ids.json"
    _write_sample_ids(sample_ids)
    monkeypatch.setattr(
        prep,
        "load_sample_lookup",
        lambda *, nr3d_root, phase8_data_root, split, sample_ids=None: (
            SimpleNamespace(),
            {},
        ),
    )

    with pytest.raises(ValueError, match="No NR3D sample"):
        prep.prepare_pack_v1_inputs_nr3d(
            sample_ids_path=sample_ids,
            data_root=data_root,
            pack_name="pack_nr3d_v1",
            split="test",
        )


def test_build_proposals_rejects_bad_bbox_shape(tmp_path) -> None:
    from evaluation.scripts.prepare_pack_v1_inputs_nr3d import (
        build_proposals_from_phase8_objects,
    )

    with pytest.raises(ValueError, match=r"bbox_np shape is \(7, 3\)"):
        build_proposals_from_phase8_objects(
            objects=[{"bbox_np": np.zeros((7, 3)), "class_name": ["chair"]}],
            scene_id="scene0001_00",
        )
