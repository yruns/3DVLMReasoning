"""Smoke + unit tests for ScanRefer pack-v1 prep."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_parse_sample_id_canonical():
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import parse_sample_id

    scene, target_id, ann_id = parse_sample_id("scannet/scene0088_00::5::3")
    assert scene == "scene0088_00"
    assert target_id == 5
    assert ann_id == "3"


def test_parse_sample_id_rejects_malformed():
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import parse_sample_id

    with pytest.raises(ValueError, match="format"):
        parse_sample_id("scannet/scene_x::5")


def test_safe_sample_id_normalizes_separators():
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import safe_sample_id

    assert safe_sample_id("scannet/scene0088_00::5::3") == "scannet__scene0088_00__5__3"


def test_load_mask3d_visibility_index_rejects_projection_only_metadata(
    tmp_path: Path,
) -> None:
    import pickle

    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import (
        load_mask3d_visibility_index,
    )

    path = tmp_path / "scene0001_00" / "conceptgraph" / "indices"
    path.mkdir(parents=True)
    with (path / "visibility_index.pkl").open("wb") as f:
        pickle.dump(
            {
                "object_to_views": {0: [(0, 0.9)]},
                "view_to_objects": {0: [(0, 0.9)]},
                "metadata": {"use_depth": False},
            },
            f,
        )

    with pytest.raises(ValueError, match="projection-only"):
        load_mask3d_visibility_index(tmp_path / "scene0001_00")


def test_load_sample_requests_validates_per_row(tmp_path: Path):
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import load_sample_requests

    p = tmp_path / "ids.json"
    p.write_text(
        json.dumps(
            [
                {
                    "sample_id": "scannet/scene_a::0::0",
                    "scene_id": "scene_a",
                    "target_id": 0,
                    "ann_id": "0",
                    "category": "chair",
                },
            ]
        )
    )
    reqs = load_sample_requests(p)
    assert len(reqs) == 1
    assert reqs[0].scene_id == "scene_a"
    assert reqs[0].target_id == 0
    assert reqs[0].ann_id == "0"


# -- compute_proposal_frame_views (CVRA M2b pack-prep contract) -------------


def _build_scene_for_cvra(tmp_path: Path):
    """Build a minimal raw-frames scene at tmp_path/raw_root/scene_xx that
    `_resolve_raw_rgb_path` can resolve. Returns (raw_root, scene_id)."""
    raw_root = tmp_path / "raw_root"
    scene_id = "scene_test"
    scene_root = raw_root / scene_id
    raw_dir = scene_root / "raw"
    raw_dir.mkdir(parents=True)
    # scene_info.json maps the local view_id → raw frame integer id
    (raw_dir / "scene_info.json").write_text(
        json.dumps({"kept_frame_ids": [10, 20]}),
    )
    for raw_id in (10, 20):
        (raw_dir / f"{raw_id:06d}-rgb.png").write_bytes(b"")
    return raw_root, scene_id


def test_compute_proposal_frame_views_emits_per_frame_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Core CVRA M2b contract: for each (proposal, frame) where the
    proposal is visible, emit bbox_2d + raw_rgb_path + visibility_weight."""
    import numpy as np

    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import (
        SceneFrame,
        compute_proposal_frame_views,
    )

    raw_root, scene_id = _build_scene_for_cvra(tmp_path)

    # Stub the projection: return deterministic rects so we don't need a
    # real intrinsic / extrinsic. Format is (x1, y1, x2, y2).
    def _stub_project(bbox_3d, intrinsic, extrinsic, image_size):
        # return distinct rects per proposal so tests can verify routing
        cx = float(bbox_3d[0])
        return (int(cx) * 10, int(cx) * 10, int(cx) * 10 + 50, int(cx) * 10 + 30)

    monkeypatch.setattr(
        "evaluation.scripts.prepare_pack_v1_inputs_scanrefer.project_visible_bbox_3d_to_2d",
        _stub_project,
    )

    proposals_by_id = {
        0: {"id": 0, "bbox_3d": [1.0] * 9, "label": "chair"},
        1: {"id": 1, "bbox_3d": [2.0] * 9, "label": "table"},
    }
    frame_by_id = {
        0: SceneFrame(
            frame_id=0,
            raw_frame_id=10,
            rgb_path=raw_root / scene_id / "raw" / "000010-rgb.png",
            extrinsic_world_to_cam=np.eye(4),
        ),
        1: SceneFrame(
            frame_id=1,
            raw_frame_id=20,
            rgb_path=raw_root / scene_id / "raw" / "000020-rgb.png",
            extrinsic_world_to_cam=np.eye(4),
        ),
    }
    frame_visibility = {0: [0, 1], 1: [1]}
    view_to_objects = {
        0: [(0, 0.8), (1, 0.5)],
        1: [(1, 0.6)],
    }

    out = compute_proposal_frame_views(
        proposal_by_id=proposals_by_id,
        frame_visibility=frame_visibility,
        frame_by_id=frame_by_id,
        intrinsic=np.eye(3),
        image_size=(640, 480),
        view_to_objects=view_to_objects,
        raw_frames_root=raw_root,
        scene_id=scene_id,
    )

    # Both proposals show up; proposal 1 in two frames, proposal 0 in one
    assert set(out.keys()) == {0, 1}
    assert set(out[0].keys()) == {"0"}  # proposal 0 only in frame 0
    assert set(out[1].keys()) == {"0", "1"}  # proposal 1 in both frames

    # Schema per the M2b contract: bbox_2d (4 ints, ordered), raw_rgb_path
    # (project-root-relative-ish string), visibility_weight (float).
    e = out[0]["0"]
    assert e["bbox_2d"] == [10, 10, 60, 40]
    assert e["raw_rgb_path"].endswith("000010-rgb.png")
    assert e["visibility_weight"] == 0.8

    # proposal 1 in frame 1 should pull weight from view_to_objects[1]
    assert out[1]["1"]["visibility_weight"] == 0.6
    assert out[1]["1"]["raw_rgb_path"].endswith("000020-rgb.png")


def test_compute_proposal_frame_views_skips_collapsed_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """If project_bbox_3d_to_2d returns None or zero-area, omit the (proposal,
    frame) entry. Don't emit a stub or a NaN."""
    import numpy as np

    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import (
        SceneFrame,
        compute_proposal_frame_views,
    )

    raw_root, scene_id = _build_scene_for_cvra(tmp_path)

    monkeypatch.setattr(
        "evaluation.scripts.prepare_pack_v1_inputs_scanrefer.project_visible_bbox_3d_to_2d",
        lambda *a, **kw: None,
    )

    out = compute_proposal_frame_views(
        proposal_by_id={0: {"id": 0, "bbox_3d": [1.0] * 9, "label": "chair"}},
        frame_visibility={0: [0]},
        frame_by_id={
            0: SceneFrame(
                frame_id=0,
                raw_frame_id=10,
                rgb_path=raw_root / scene_id / "raw" / "000010-rgb.png",
                extrinsic_world_to_cam=np.eye(4),
            ),
        },
        intrinsic=np.eye(3),
        image_size=(640, 480),
        view_to_objects={0: [(0, 0.8)]},
        raw_frames_root=raw_root,
        scene_id=scene_id,
    )
    assert out == {0: {}}  # proposal exists but no frame view emitted


def test_compute_proposal_frame_views_normalizes_swapped_coords(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """If projection returns x2<x1 or y2<y1, normalize so x1<=x2 and y1<=y2."""
    import numpy as np

    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import (
        SceneFrame,
        compute_proposal_frame_views,
    )

    raw_root, scene_id = _build_scene_for_cvra(tmp_path)

    monkeypatch.setattr(
        "evaluation.scripts.prepare_pack_v1_inputs_scanrefer.project_visible_bbox_3d_to_2d",
        lambda *a, **kw: (200, 100, 50, 40),  # x2<x1 and y2<y1
    )

    out = compute_proposal_frame_views(
        proposal_by_id={0: {"id": 0, "bbox_3d": [1.0] * 9, "label": "chair"}},
        frame_visibility={0: [0]},
        frame_by_id={
            0: SceneFrame(
                frame_id=0,
                raw_frame_id=10,
                rgb_path=raw_root / scene_id / "raw" / "000010-rgb.png",
                extrinsic_world_to_cam=np.eye(4),
            ),
        },
        intrinsic=np.eye(3),
        image_size=(640, 480),
        view_to_objects={0: [(0, 0.8)]},
        raw_frames_root=raw_root,
        scene_id=scene_id,
    )
    bbox = out[0]["0"]["bbox_2d"]
    assert bbox[0] <= bbox[2]
    assert bbox[1] <= bbox[3]
    assert bbox == [50, 40, 200, 100]


def test_compute_proposal_frame_views_omits_visibility_weight_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """If view_to_objects has no entry for a (proposal, frame) but the
    proposal is in frame_visibility, emit the entry without
    visibility_weight rather than failing."""
    import numpy as np

    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import (
        SceneFrame,
        compute_proposal_frame_views,
    )

    raw_root, scene_id = _build_scene_for_cvra(tmp_path)
    monkeypatch.setattr(
        "evaluation.scripts.prepare_pack_v1_inputs_scanrefer.project_visible_bbox_3d_to_2d",
        lambda *a, **kw: (10, 20, 100, 200),
    )

    out = compute_proposal_frame_views(
        proposal_by_id={0: {"id": 0, "bbox_3d": [1.0] * 9, "label": "chair"}},
        frame_visibility={0: [0]},
        frame_by_id={
            0: SceneFrame(
                frame_id=0,
                raw_frame_id=10,
                rgb_path=raw_root / scene_id / "raw" / "000010-rgb.png",
                extrinsic_world_to_cam=np.eye(4),
            ),
        },
        intrinsic=np.eye(3),
        image_size=(640, 480),
        view_to_objects={},  # missing for (0, 0)
        raw_frames_root=raw_root,
        scene_id=scene_id,
    )
    e = out[0]["0"]
    assert e["bbox_2d"] == [10, 20, 100, 200]
    assert "visibility_weight" not in e


def test_compute_proposal_frame_views_unknown_proposal_id_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """frame_visibility lists an id not in proposal_by_id → fail loud."""
    import numpy as np

    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import (
        SceneFrame,
        compute_proposal_frame_views,
    )

    raw_root, scene_id = _build_scene_for_cvra(tmp_path)

    with pytest.raises(ValueError, match="unknown proposal_id"):
        compute_proposal_frame_views(
            proposal_by_id={0: {"id": 0, "bbox_3d": [1.0] * 9, "label": "chair"}},
            frame_visibility={0: [99]},  # 99 not in proposal_by_id
            frame_by_id={
                0: SceneFrame(
                    frame_id=0,
                    raw_frame_id=10,
                    rgb_path=raw_root / scene_id / "raw" / "000010-rgb.png",
                    extrinsic_world_to_cam=np.eye(4),
                ),
            },
            intrinsic=np.eye(3),
            image_size=(640, 480),
            view_to_objects={0: [(99, 0.5)]},
            raw_frames_root=raw_root,
            scene_id=scene_id,
        )
