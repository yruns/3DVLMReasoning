"""Smoke test for offline pack-v1 input preparation (GT-pool path)."""
from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


@pytest.mark.integration
def test_prepare_pack_v1_inputs_smoke(tmp_path, monkeypatch) -> None:
    from evaluation.scripts import prepare_pack_v1_inputs as prep

    scene_id = "scene0001_00"
    target_id = 72
    sample_ids = tmp_path / "frozen_sample_ids.json"
    sample_ids.write_text(
        json.dumps(
            [
                {
                    "sample_id": f"{scene_id}::{target_id}",
                    "scene_id": scene_id,
                    "target_id": target_id,
                    "category": "picture",
                }
            ]
        ),
        encoding="utf-8",
    )

    data_root = tmp_path / "embodiedscan"
    rgb_path = data_root / scene_id / "raw" / "000010-rgb.jpg"
    rgb_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), color="white").save(rgb_path)

    scene_info = {
        "sample_idx": f"scannet/{scene_id}",
        "cam2img": [[50, 0, 50], [0, 50, 50], [0, 0, 1]],
        "axis_align_matrix": np.eye(4).tolist(),
        "instances": [
            {
                "bbox_id": target_id,
                "bbox_3d": [0, 0, 5, 1, 1, 1, 0, 0, 0],
                "bbox_label_3d": 7,
            },
            {
                "bbox_id": 99,
                "bbox_3d": [10, 10, 5, 1, 1, 1, 0, 0, 0],
                "bbox_label_3d": 12,
            },
        ],
        "images": [
            {
                "frame_id": 10,
                "img_path": str(rgb_path.relative_to(data_root)),
                "cam2global": np.eye(4).tolist(),
                "visible_instance_ids": [target_id],
            }
        ],
    }
    sample = SimpleNamespace(
        sample_id=f"{scene_id}::{target_id}",
        scene_id=scene_id,
        scan_id=f"scannet/{scene_id}",
        target_id=target_id,
        target="picture",
        query="the picture on the wall",
        gt_bbox_3d=[0, 0, 5, 1, 1, 1, 0, 0, 0],
    )
    adapter = SimpleNamespace(
        dataset=SimpleNamespace(
            get_scene_info=lambda scan_id: scene_info,
            label_to_name={7: "picture", 12: "table"},
        ),
        get_axis_align_matrix=lambda scan_id: np.eye(4),
    )
    monkeypatch.setattr(
        prep,
        "load_sample_lookup",
        lambda data_root, split: (adapter, {(scene_id, target_id): sample}),
    )
    monkeypatch.setattr(
        prep,
        "select_keyframes_for_sample",
        lambda *args, **kwargs: [
            {"keyframe_idx": 0, "image_path": str(rgb_path), "frame_id": 10}
        ],
    )

    written = prep.prepare_pack_v1_inputs(
        sample_ids_path=sample_ids,
        data_root=data_root,
        split="val",
        max_samples=None,
    )

    scene_dir = data_root / scene_id / "pack_v1"
    proposals_jsonl = scene_dir / "proposals.jsonl"
    visibility_json = scene_dir / "visibility.json"
    annotated = scene_dir / "annotated" / "frame_10.png"
    sample_json = scene_dir / "samples" / f"{target_id}.json"
    assert written == [sample_json]
    assert proposals_jsonl.exists()

    proposals_payload = json.loads(proposals_jsonl.read_text())
    assert proposals_payload["source"] == "gt"
    assert proposals_payload["scene_id"] == scene_id
    proposals = proposals_payload["proposals"]
    assert len(proposals) == 2
    by_id = {p["id"]: p for p in proposals}
    assert by_id[target_id]["label"] == "picture"
    assert by_id[target_id]["score"] == 1.0
    assert by_id[99]["label"] == "table"

    visibility = json.loads(visibility_json.read_text())
    # Visibility comes from the pkl `visible_instance_ids`, restricted to the
    # GT instance set. Only target_id is in the fixture's visible list.
    assert visibility == {"10": [target_id]}
    assert annotated.exists()

    sample_payload = json.loads(sample_json.read_text())
    assert sample_payload["sample_id"] == f"{scene_id}::{target_id}"
    assert sample_payload["query"] == "the picture on the wall"
    assert sample_payload["source"] == "gt"
    assert sample_payload["gt_bbox_3d_9dof"] == [0.0, 0.0, 5.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert sample_payload["scene_artifacts_dir"] == str(scene_dir)
    assert sample_payload["keyframes"] == [
        {"keyframe_idx": 0, "image_path": str(annotated), "frame_id": 10}
    ]

    import agents.packs.vg_embodiedscan
    from agents.core.agent_config import Stage2TaskType
    from agents.skills import PACKS
    from agents.skills.validate import validate_packs

    if Stage2TaskType.VISUAL_GROUNDING not in PACKS:
        importlib.reload(agents.packs.vg_embodiedscan)

    from agents.examples.embodiedscan_vg_pack_v1_pilot import build_pack_v1_bundle

    bundle = build_pack_v1_bundle(
        proposals_jsonl=proposals_jsonl,
        source="gt",
        annotated_image_dir=scene_dir / "annotated",
        frame_visibility={10: [target_id]},
        keyframes=[(0, str(annotated), 10)],
        scene_id=scene_id,
    )
    validate_packs(Stage2TaskType.VISUAL_GROUNDING, bundle)
