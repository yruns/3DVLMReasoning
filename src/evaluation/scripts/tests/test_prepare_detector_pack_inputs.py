"""Synthetic tests for V-DETR detector-record to pack conversion."""

from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


def test_prepare_detector_pack_inputs_smoke(tmp_path, monkeypatch) -> None:
    from agents.packs.vg_embodiedscan.proposal_pool import build_vg_proposal_pool
    from evaluation.scripts import prepare_detector_pack_inputs as prep

    scene_id = "scene0001_00"
    target_id = 72
    data_root, sample_ids, rgb_paths, scene_info, sample, adapter = (
        _write_scene_fixture(tmp_path, scene_id=scene_id, target_id=target_id)
    )
    detector_records = tmp_path / "detector_records.jsonl"
    detector_records.write_text(
        _detector_record_jsonl(
            scene_id=scene_id,
            proposals=[
                _proposal(
                    bbox=[1, 0, 5, 1, 1, 1, 0, 0, 0],
                    score=0.25,
                    class_id=4,
                ),
                _proposal(
                    bbox=[0, 0, 5, 1, 1, 1, 0, 0, 0],
                    score=0.95,
                    class_id=2,
                ),
            ],
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        prep,
        "load_sample_lookup",
        lambda data_root, split, infos_pkl=None, vg_json=None, requested_keys=None, detector_record_scene_ids=None: (
            adapter,
            {(scene_id, target_id): sample},
        ),
    )
    monkeypatch.setattr(
        prep,
        "select_keyframes_for_sample",
        lambda *args, **kwargs: [
            {"keyframe_idx": 0, "image_path": str(rgb_paths[0]), "frame_id": 0}
        ],
    )

    written = prep.prepare_detector_pack_inputs(
        detector_records_path=detector_records,
        sample_ids_path=sample_ids,
        data_root=data_root,
        split="val",
        pack_name="pack_vdetr",
        visibility_min_area=32.0,
        max_proposals_per_scene=256,
    )

    scene_dir = data_root / scene_id / "pack_vdetr"
    proposals_jsonl = scene_dir / "proposals.jsonl"
    visibility_json = scene_dir / "visibility.json"
    annotated = scene_dir / "annotated" / "frame_0.png"
    sample_json = scene_dir / "samples" / f"{target_id}.json"

    assert written == [sample_json]
    assert proposals_jsonl.exists()
    assert visibility_json.exists()
    assert annotated.exists()
    assert sample_json.exists()

    proposals_payload = json.loads(proposals_jsonl.read_text(encoding="utf-8"))
    assert proposals_payload["source"] == "vdetr"
    assert proposals_payload["scene_id"] == scene_id
    proposals = proposals_payload["proposals"]
    assert [p["id"] for p in proposals] == [0, 1]
    assert [p["score"] for p in proposals] == [0.95, 0.25]
    assert proposals[0]["label"] == "chair"
    assert proposals[0]["metadata"]["class_id"] == 2
    assert proposals[0]["metadata"]["detector"] == "V-DETR"
    assert proposals[0]["metadata"]["raw_corners"] == _raw_corners(2.0)
    assert "0" in proposals[0]["frame_views"]
    assert proposals[0]["frame_views"]["0"]["raw_rgb_path"] == str(rgb_paths[0])
    assert len(proposals[0]["frame_views"]["0"]["bbox_2d"]) == 4

    visibility = {
        int(k): [int(x) for x in v]
        for k, v in json.loads(visibility_json.read_text(encoding="utf-8")).items()
    }
    pool = build_vg_proposal_pool(
        proposals_jsonl=proposals_jsonl,
        source="vdetr",
        annotated_image_dir=scene_dir / "annotated",
        frame_visibility=visibility,
        axis_align_matrix=None,
    )
    assert pool["source"] == "vdetr"
    assert pool["proposals"][0]["id"] == 0

    sample_payload = json.loads(sample_json.read_text(encoding="utf-8"))
    assert sample_payload["sample_id"] == f"{scene_id}::{target_id}"
    assert sample_payload["source"] == "vdetr"
    assert sample_payload["scene_artifacts_dir"] == str(scene_dir)
    assert sample_payload["gt_bbox_3d_9dof"] == [
        0.0,
        0.0,
        5.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert sample_payload["keyframes"] == [
        {"keyframe_idx": 0, "image_path": str(rgb_paths[0]), "frame_id": 0}
    ]
    assert sample_payload["proposals"][0]["id"] == 0
    assert sample_payload["proposals"][0]["metadata"]["raw_corners"] == _raw_corners(
        2.0
    )


def test_build_proposals_requires_detector_metadata() -> None:
    from benchmarks.embodiedscan_bbox_feasibility.models import ProposalRecord
    from evaluation.scripts import prepare_detector_pack_inputs as prep

    raw = json.loads(
        _detector_record_jsonl(
            scene_id="scene_missing_detector",
            proposals=[
                _proposal(
                    bbox=[0, 0, 5, 1, 1, 1, 0, 0, 0],
                    score=0.9,
                    class_id=2,
                    detector=None,
                )
            ],
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            r"proposal\[0\] in scene scene_missing_detector is missing "
            r"required metadata\.detector"
        ),
    ):
        prep.build_proposals_from_detector_record(
            ProposalRecord.model_validate(raw),
            max_proposals_per_scene=256,
        )


def test_build_proposals_requires_valid_raw_corners_metadata() -> None:
    from benchmarks.embodiedscan_bbox_feasibility.models import ProposalRecord
    from evaluation.scripts import prepare_detector_pack_inputs as prep

    raw = json.loads(
        _detector_record_jsonl(
            scene_id="scene_bad_corners",
            proposals=[
                _proposal(
                    bbox=[0, 0, 5, 1, 1, 1, 0, 0, 0],
                    score=0.9,
                    class_id=2,
                    raw_corners=[[1.0, 2.0] for _ in range(8)],
                )
            ],
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            r"proposal\[0\] in scene scene_bad_corners has invalid "
            r"metadata\.raw_corners"
        ),
    ):
        prep.build_proposals_from_detector_record(
            ProposalRecord.model_validate(raw),
            max_proposals_per_scene=256,
        )


def test_load_sample_lookup_raises_on_vg_row_missing_scene(tmp_path) -> None:
    from evaluation.scripts import prepare_detector_pack_inputs as prep

    data_root = _write_annotation_files(
        tmp_path,
        valid_scene_id="scene_valid",
        valid_target_id=1,
        vg_entries=[
            {
                "scan_id": "scannet/scene9999_99",
                "target_id": 1,
                "target": "chair",
                "text": "missing scene",
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            r"VG row 0 references scan_id='scannet/scene9999_99' not present "
            r"in .*embodiedscan_infos_val\.pkl"
        ),
    ):
        prep.load_sample_lookup(data_root, "val")


def test_sample_ids_filter_before_vg_row_validation(tmp_path) -> None:
    from evaluation.scripts import prepare_detector_pack_inputs as prep

    data_root = _write_annotation_files(
        tmp_path,
        valid_scene_id="scene_valid",
        valid_target_id=1,
        vg_entries=[
            {
                "scan_id": "scannet/scene_valid",
                "target_id": 1,
                "target": "chair",
                "text": "valid chair",
            },
            {
                "scan_id": "scannet/scene9999_99",
                "target_id": 2,
                "target": "table",
                "text": "excluded missing scene",
            },
        ],
    )
    sample_ids = tmp_path / "sample_ids.json"
    sample_ids.write_text(json.dumps(["scene_valid::1"]), encoding="utf-8")

    requests = prep.load_sample_requests(sample_ids)
    _adapter, lookup = prep.load_sample_lookup(
        data_root,
        "val",
        requested_keys={(request.scene_id, request.target_id) for request in requests},
    )

    assert sorted(lookup) == [("scene_valid", 1)]


def test_prepare_detector_pack_inputs_skips_bad_vg_rows_outside_detector_scenes(
    tmp_path,
    monkeypatch,
) -> None:
    from evaluation.scripts import prepare_detector_pack_inputs as prep

    scene_a = "scene0001_00"
    scene_b = "scene0002_00"
    scene_c = "scene0003_00"
    data_root = _write_multiscene_annotation_files(
        tmp_path,
        scenes=[
            _scene_spec(scene_a, 1),
            _scene_spec(scene_b, 2),
            _scene_spec(scene_c, 2),
        ],
        vg_entries=[
            _vg_entry(scene_a, 1, text="valid A"),
            _vg_entry(scene_b, 99, text="bad B"),
            _vg_entry(scene_c, 2, text="valid C"),
        ],
    )
    detector_records = tmp_path / "detector_records.jsonl"
    detector_records.write_text(
        _detector_record_jsonl(
            scene_id=scene_a,
            proposals=[
                _proposal(
                    bbox=[0, 0, 5, 1, 1, 1, 0, 0, 0],
                    score=0.9,
                    class_id=2,
                )
            ],
        )
        + _detector_record_jsonl(
            scene_id=scene_c,
            proposals=[
                _proposal(
                    bbox=[0, 0, 5, 1, 1, 1, 0, 0, 0],
                    score=0.8,
                    class_id=2,
                )
            ],
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        prep,
        "select_keyframes_for_sample",
        lambda sample, adapter, data_root, **kwargs: [
            {
                "keyframe_idx": 0,
                "image_path": str(
                    data_root / sample.scene_id / "raw" / "000000-rgb.jpg"
                ),
                "frame_id": 0,
            }
        ],
    )

    written = prep.prepare_detector_pack_inputs(
        detector_records_path=detector_records,
        data_root=data_root,
        split="val",
        pack_name="pack_vdetr_scope",
        visibility_min_area=32.0,
        max_proposals_per_scene=256,
    )

    assert sorted(path.parent.parent.parent.name for path in written) == [
        scene_a,
        scene_c,
    ]


def test_prepare_detector_pack_inputs_raises_for_bad_vg_row_inside_detector_scenes(
    tmp_path,
) -> None:
    from evaluation.scripts import prepare_detector_pack_inputs as prep

    scene_a = "scene0001_00"
    scene_b = "scene0002_00"
    data_root = _write_multiscene_annotation_files(
        tmp_path,
        scenes=[
            _scene_spec(scene_a, 1),
            _scene_spec(scene_b, 2),
        ],
        vg_entries=[
            _vg_entry(scene_a, 1, text="valid A"),
            _vg_entry(scene_b, 99, text="bad B"),
        ],
    )
    detector_records = tmp_path / "detector_records.jsonl"
    detector_records.write_text(
        _detector_record_jsonl(
            scene_id=scene_a,
            proposals=[
                _proposal(
                    bbox=[0, 0, 5, 1, 1, 1, 0, 0, 0],
                    score=0.9,
                    class_id=2,
                )
            ],
        )
        + _detector_record_jsonl(
            scene_id=scene_b,
            proposals=[
                _proposal(
                    bbox=[0, 0, 5, 1, 1, 1, 0, 0, 0],
                    score=0.8,
                    class_id=2,
                )
            ],
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"VG row 1 references target_id=99 in "
            r"scan_id='scannet/scene0002_00' without a unique GT bbox"
        ),
    ):
        prep.prepare_detector_pack_inputs(
            detector_records_path=detector_records,
            data_root=data_root,
            split="val",
            pack_name="pack_vdetr_scope",
            visibility_min_area=32.0,
            max_proposals_per_scene=256,
        )


def test_prepare_detector_pack_inputs_filters_visibility_by_projection(
    tmp_path,
    monkeypatch,
) -> None:
    from evaluation.scripts import prepare_detector_pack_inputs as prep

    scene_id = "scene_visible"
    target_id = 9
    data_root, sample_ids, rgb_paths, scene_info, sample, adapter = (
        _write_scene_fixture(tmp_path, scene_id=scene_id, target_id=target_id)
    )
    scene_info["images"][1]["world_to_cam"] = [
        [1, 0, 0, 100],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    detector_records = tmp_path / "detector_records.jsonl"
    detector_records.write_text(
        _detector_record_jsonl(
            scene_id=scene_id,
            proposals=[
                _proposal(
                    bbox=[0, 0, 5, 1, 1, 1, 0, 0, 0],
                    score=0.8,
                    class_id=2,
                )
            ],
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        prep,
        "load_sample_lookup",
        lambda data_root, split, infos_pkl=None, vg_json=None, requested_keys=None, detector_record_scene_ids=None: (
            adapter,
            {(scene_id, target_id): sample},
        ),
    )
    monkeypatch.setattr(
        prep,
        "select_keyframes_for_sample",
        lambda *args, **kwargs: [
            {"keyframe_idx": 0, "image_path": str(rgb_paths[0]), "frame_id": 0},
            {"keyframe_idx": 1, "image_path": str(rgb_paths[1]), "frame_id": 1},
        ],
    )

    prep.prepare_detector_pack_inputs(
        detector_records_path=detector_records,
        sample_ids_path=sample_ids,
        data_root=data_root,
        split="val",
        pack_name="pack_vdetr",
        visibility_min_area=32.0,
        max_proposals_per_scene=256,
    )

    visibility = json.loads(
        (data_root / scene_id / "pack_vdetr" / "visibility.json").read_text(
            encoding="utf-8"
        )
    )
    assert visibility == {"0": [0], "1": []}


def test_prepare_detector_pack_inputs_assigns_stable_ids(
    tmp_path,
    monkeypatch,
) -> None:
    from evaluation.scripts import prepare_detector_pack_inputs as prep

    scene_id = "scene_stable"
    target_id = 5
    data_root, sample_ids, rgb_paths, scene_info, sample, adapter = (
        _write_scene_fixture(tmp_path, scene_id=scene_id, target_id=target_id)
    )
    detector_records = tmp_path / "detector_records.jsonl"
    detector_records.write_text(
        _detector_record_jsonl(
            scene_id=scene_id,
            proposals=[
                _proposal(
                    bbox=[2, 0, 5, 1, 1, 1, 0, 0, 0],
                    score=0.2,
                    class_id=4,
                ),
                _proposal(
                    bbox=[0, 0, 5, 1, 1, 1, 0, 0, 0],
                    score=0.9,
                    class_id=2,
                ),
                _proposal(
                    bbox=[1, 0, 5, 1, 1, 1, 0, 0, 0],
                    score=0.5,
                    class_id=3,
                ),
            ],
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        prep,
        "load_sample_lookup",
        lambda data_root, split, infos_pkl=None, vg_json=None, requested_keys=None, detector_record_scene_ids=None: (
            adapter,
            {(scene_id, target_id): sample},
        ),
    )
    monkeypatch.setattr(
        prep,
        "select_keyframes_for_sample",
        lambda *args, **kwargs: [
            {"keyframe_idx": 0, "image_path": str(rgb_paths[0]), "frame_id": 0}
        ],
    )

    observed = []
    for _ in range(2):
        prep.prepare_detector_pack_inputs(
            detector_records_path=detector_records,
            sample_ids_path=sample_ids,
            data_root=data_root,
            split="val",
            pack_name="pack_vdetr",
            visibility_min_area=32.0,
            max_proposals_per_scene=256,
        )
        payload = json.loads(
            (data_root / scene_id / "pack_vdetr" / "proposals.jsonl").read_text(
                encoding="utf-8"
            )
        )
        observed.append(
            [
                (proposal["id"], proposal["score"], proposal["metadata"]["class_id"])
                for proposal in payload["proposals"]
            ]
        )

    assert observed == [
        [(0, 0.9, 2), (1, 0.5, 3), (2, 0.2, 4)],
        [(0, 0.9, 2), (1, 0.5, 3), (2, 0.2, 4)],
    ]


def _write_scene_fixture(tmp_path, *, scene_id: str, target_id: int):
    data_root = tmp_path / "embodiedscan"
    rgb_paths = []
    for frame_id in (0, 1):
        rgb_path = data_root / scene_id / "raw" / f"{frame_id:06d}-rgb.jpg"
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), color="white").save(rgb_path)
        rgb_paths.append(rgb_path)

    sample_ids = tmp_path / "sample_ids.json"
    sample_ids.write_text(
        json.dumps(
            [
                {
                    "sample_id": f"{scene_id}::{target_id}",
                    "scene_id": scene_id,
                    "target_id": target_id,
                    "category": "chair",
                }
            ]
        ),
        encoding="utf-8",
    )
    scene_info = {
        "sample_idx": f"scannet/{scene_id}",
        "cam2img": [[50, 0, 50], [0, 50, 50], [0, 0, 1]],
        "axis_align_matrix": np.eye(4).tolist(),
        "instances": [
            {
                "bbox_id": target_id,
                "bbox_3d": [0, 0, 5, 1, 1, 1, 0, 0, 0],
                "bbox_label_3d": 2,
            }
        ],
        "images": [
            {
                "frame_id": 0,
                "img_path": str(rgb_paths[0].relative_to(data_root)),
                "world_to_cam": np.eye(4).tolist(),
                "visible_instance_ids": [0],
            },
            {
                "frame_id": 1,
                "img_path": str(rgb_paths[1].relative_to(data_root)),
                "world_to_cam": np.eye(4).tolist(),
                "visible_instance_ids": [0],
            },
        ],
    }
    sample = SimpleNamespace(
        sample_id=f"{scene_id}::{target_id}",
        scene_id=scene_id,
        scan_id=f"scannet/{scene_id}",
        target_id=target_id,
        target="chair",
        query="the chair near the table",
        gt_bbox_3d=[0, 0, 5, 1, 1, 1, 0, 0, 0],
    )
    adapter = SimpleNamespace(
        dataset=SimpleNamespace(
            get_scene_info=lambda scan_id: scene_info,
            label_to_name={2: "chair"},
        )
    )
    return data_root, sample_ids, rgb_paths, scene_info, sample, adapter


def _detector_record_jsonl(*, scene_id: str, proposals: list[dict]) -> str:
    return (
        json.dumps(
            {
                "scene_id": scene_id,
                "scan_id": f"scannet/{scene_id}",
                "target_id": None,
                "method": "3d-vdetr",
                "input_condition": "scannet_full",
                "proposals": proposals,
                "failure_tag": None,
                "metadata": {},
            }
        )
        + "\n"
    )


def _write_annotation_files(
    tmp_path,
    *,
    valid_scene_id: str,
    valid_target_id: int,
    vg_entries: list[dict],
):
    data_root = tmp_path / "embodiedscan_annotations"
    data_root.mkdir()
    scene_info = {
        "sample_idx": f"scannet/{valid_scene_id}",
        "instances": [
            {
                "bbox_id": valid_target_id,
                "bbox_3d": [0, 0, 5, 1, 1, 1, 0, 0, 0],
                "bbox_label_3d": 2,
            }
        ],
        "images": [],
    }
    with (data_root / "embodiedscan_infos_val.pkl").open("wb") as handle:
        pickle.dump(
            {
                "metainfo": {"categories": {"chair": 2}},
                "data_list": [scene_info],
            },
            handle,
        )
    (data_root / "embodiedscan_val_vg.json").write_text(
        json.dumps(vg_entries),
        encoding="utf-8",
    )
    return data_root


def _write_multiscene_annotation_files(
    tmp_path,
    *,
    scenes: list[dict],
    vg_entries: list[dict],
):
    data_root = tmp_path / "embodiedscan_multiscene"
    data_root.mkdir()
    for scene in scenes:
        scene_id = scene["scene_id"]
        rgb_path = data_root / scene_id / "raw" / "000000-rgb.jpg"
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), color="white").save(rgb_path)
    with (data_root / "embodiedscan_infos_val.pkl").open("wb") as handle:
        pickle.dump(
            {
                "metainfo": {"categories": {"chair": 2}},
                "data_list": [
                    _scene_info_for_annotation(
                        scene["scene_id"],
                        scene["target_id"],
                    )
                    for scene in scenes
                ],
            },
            handle,
        )
    (data_root / "embodiedscan_val_vg.json").write_text(
        json.dumps(vg_entries),
        encoding="utf-8",
    )
    return data_root


def _scene_spec(scene_id: str, target_id: int) -> dict:
    return {"scene_id": scene_id, "target_id": target_id}


def _scene_info_for_annotation(scene_id: str, target_id: int) -> dict:
    return {
        "sample_idx": f"scannet/{scene_id}",
        "cam2img": [[50, 0, 50], [0, 50, 50], [0, 0, 1]],
        "axis_align_matrix": np.eye(4).tolist(),
        "instances": [
            {
                "bbox_id": target_id,
                "bbox_3d": [0, 0, 5, 1, 1, 1, 0, 0, 0],
                "bbox_label_3d": 2,
            }
        ],
        "images": [
            {
                "frame_id": 0,
                "img_path": f"{scene_id}/raw/000000-rgb.jpg",
                "world_to_cam": np.eye(4).tolist(),
                "visible_instance_ids": [0],
            }
        ],
    }


def _vg_entry(scene_id: str, target_id: int, *, text: str) -> dict:
    return {
        "scan_id": f"scannet/{scene_id}",
        "target_id": target_id,
        "target": "chair",
        "text": text,
    }


def _proposal(
    *,
    bbox: list[float],
    score: float,
    class_id: int,
    detector: str | None = "V-DETR",
    raw_corners: list[list[float]] | None = None,
) -> dict:
    metadata = {
        "class_id": class_id,
        "raw_corners": (
            raw_corners if raw_corners is not None else _raw_corners(class_id)
        ),
    }
    if detector is not None:
        metadata["detector"] = detector
    return {
        "bbox_3d": bbox,
        "score": score,
        "source": "detector",
        "metadata": metadata,
    }


def _raw_corners(offset: float) -> list[list[float]]:
    return [
        [offset + float(i), offset + float(i) + 0.25, offset + float(i) + 0.5]
        for i in range(8)
    ]
