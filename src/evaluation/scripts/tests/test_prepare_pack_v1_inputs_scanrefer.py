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


def test_load_sample_requests_validates_per_row(tmp_path: Path):
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import load_sample_requests
    p = tmp_path / "ids.json"
    p.write_text(json.dumps([
        {"sample_id": "scannet/scene_a::0::0", "scene_id": "scene_a",
         "target_id": 0, "ann_id": "0", "category": "chair"},
    ]))
    reqs = load_sample_requests(p)
    assert len(reqs) == 1
    assert reqs[0].scene_id == "scene_a"
    assert reqs[0].target_id == 0
    assert reqs[0].ann_id == "0"


def test_normalize_category_tokens_drops_articles_and_underscore():
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import (
        _normalize_category_tokens,
    )
    assert _normalize_category_tokens("the office chair") == {"office", "chair"}
    assert _normalize_category_tokens("trash_can") == {"trash", "can"}
    assert _normalize_category_tokens("UNKNOW") == {"unknow"}
    assert _normalize_category_tokens("") == set()
    assert _normalize_category_tokens("the") == set()


def test_matching_proposal_ids_exact_label():
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import (
        _matching_proposal_ids,
    )
    proposal_labels = {0: "chair", 1: "table", 2: "lamp"}
    assert _matching_proposal_ids(["chair"], proposal_labels) == {0}
    assert _matching_proposal_ids(["table"], proposal_labels) == {1}


def test_matching_proposal_ids_underscore_to_space():
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import (
        _matching_proposal_ids,
    )
    # ScanNet200 label "trash can" should match parser "trash_can"
    proposal_labels = {0: "trash can", 1: "chair"}
    assert _matching_proposal_ids(["trash_can"], proposal_labels) == {0}


def test_matching_proposal_ids_token_overlap():
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import (
        _matching_proposal_ids,
    )
    # parser says "office chair", Mask3D label says "chair"
    proposal_labels = {0: "chair", 1: "office desk", 2: "lamp"}
    assert _matching_proposal_ids(["office chair"], proposal_labels) == {0, 1}


def test_matching_proposal_ids_multi_categories():
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import (
        _matching_proposal_ids,
    )
    proposal_labels = {0: "chair", 1: "sofa", 2: "lamp", 3: "kitchen counter"}
    matched = _matching_proposal_ids(["pillow", "sofa", "kitchen"], proposal_labels)
    assert matched == {1, 3}


def test_matching_proposal_ids_no_match():
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import (
        _matching_proposal_ids,
    )
    proposal_labels = {0: "chair", 1: "table"}
    assert _matching_proposal_ids(["fridge"], proposal_labels) == set()


def test_matching_proposal_ids_skips_unknow():
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import (
        _matching_proposal_ids,
    )
    proposal_labels = {0: "chair"}
    # "UNKNOW" sentinel from parser should never produce false matches
    assert _matching_proposal_ids(["UNKNOW"], proposal_labels) == set()
    assert _matching_proposal_ids(["unknown"], proposal_labels) == set()


def test_select_keyframes_mask3d_query_driven_happy_path(tmp_path: Path):
    """End-to-end with stub QueryParser + synthetic SceneArtifacts."""
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import (
        Mask3dVisibility,
        SceneArtifacts,
        select_keyframes_mask3d_query_driven,
    )
    from query_scene.query_structures import (
        GroundingQuery,
        HypothesisOutputV1,
        QueryNode,
    )

    raw = tmp_path / "raw_root"
    scene_root = raw / "scene_test"
    raw_dir = scene_root / "raw"
    raw_dir.mkdir(parents=True)
    # synthesize scene_info.json mapping view-id 0..2 to raw frame ids
    (raw_dir / "scene_info.json").write_text(
        json.dumps({"kept_frame_ids": [10, 20, 30]}),
    )
    for raw_id in (10, 20, 30):
        (raw_dir / f"{raw_id:06d}-rgb.png").write_bytes(b"")  # presence is enough

    proposal_labels = {0: "chair", 1: "table", 2: "lamp"}
    # frame 0: chair (high) + table; frame 1: table only; frame 2: chair (low) only
    visibility = Mask3dVisibility(
        object_to_views={
            0: [(0, 0.8), (2, 0.2)],
            1: [(0, 0.5), (1, 0.6)],
            2: [],
        },
        view_to_objects={
            0: [(0, 0.8), (1, 0.5)],
            1: [(1, 0.6)],
            2: [(0, 0.2)],
        },
    )
    artifacts = SceneArtifacts(
        scene_dir=tmp_path / "pack",
        proposals_jsonl=tmp_path / "proposals.jsonl",
        visibility_json=tmp_path / "visibility.json",
        annotated_dir=tmp_path / "annotated",
        frame_visibility={0: [0, 1], 1: [1], 2: [0]},
        proposal_ids=[0, 1, 2],
        proposal_labels=proposal_labels,
        mask3d_visibility=visibility,
        scene_categories=["chair", "lamp", "table"],
    )

    class _StubParser:
        def parse(self, query: str) -> HypothesisOutputV1:
            assert query == "the chair near the wall"
            return HypothesisOutputV1.from_direct_query(
                GroundingQuery(
                    raw_query=query,
                    root=QueryNode(categories=["chair"]),
                    expect_unique=True,
                )
            )

    kfs = select_keyframes_mask3d_query_driven(
        scene_id="scene_test",
        query="the chair near the wall",
        scene_artifacts=artifacts,
        raw_frames_root=raw,
        query_parser=_StubParser(),
        k=2,
    )
    # frame 0 has highest "chair" weight (0.8), frame 2 is the only other chair frame (0.2)
    assert [k["frame_id"] for k in kfs] == [0, 2]
    assert kfs[0]["keyframe_idx"] == 0


def test_select_keyframes_mask3d_query_driven_returns_empty_on_no_match(
    tmp_path: Path,
):
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import (
        Mask3dVisibility,
        SceneArtifacts,
        select_keyframes_mask3d_query_driven,
    )
    from query_scene.query_structures import (
        GroundingQuery,
        HypothesisOutputV1,
        QueryNode,
    )

    artifacts = SceneArtifacts(
        scene_dir=tmp_path / "pack",
        proposals_jsonl=tmp_path / "proposals.jsonl",
        visibility_json=tmp_path / "visibility.json",
        annotated_dir=tmp_path / "annotated",
        frame_visibility={0: [0]},
        proposal_ids=[0],
        proposal_labels={0: "chair"},
        mask3d_visibility=Mask3dVisibility(
            object_to_views={0: [(0, 0.8)]},
            view_to_objects={0: [(0, 0.8)]},
        ),
        scene_categories=["chair"],
    )

    class _StubParser:
        def parse(self, query: str) -> HypothesisOutputV1:
            return HypothesisOutputV1.from_direct_query(
                GroundingQuery(
                    raw_query=query,
                    root=QueryNode(categories=["fridge"]),  # not in scene
                    expect_unique=True,
                )
            )

    out = select_keyframes_mask3d_query_driven(
        scene_id="scene_test",
        query="the fridge in the kitchen",
        scene_artifacts=artifacts,
        raw_frames_root=tmp_path / "raw",
        query_parser=_StubParser(),
        k=3,
    )
    assert out == []


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
        "evaluation.scripts.prepare_pack_v1_inputs_scanrefer.project_bbox_3d_to_2d",
        _stub_project,
    )

    proposals_by_id = {
        0: {"id": 0, "bbox_3d": [1.0] * 9, "label": "chair"},
        1: {"id": 1, "bbox_3d": [2.0] * 9, "label": "table"},
    }
    frame_by_id = {
        0: SceneFrame(
            frame_id=0, raw_frame_id=10,
            rgb_path=raw_root / scene_id / "raw" / "000010-rgb.png",
            extrinsic_world_to_cam=np.eye(4),
        ),
        1: SceneFrame(
            frame_id=1, raw_frame_id=20,
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
    assert set(out[0].keys()) == {"0"}            # proposal 0 only in frame 0
    assert set(out[1].keys()) == {"0", "1"}        # proposal 1 in both frames

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
        "evaluation.scripts.prepare_pack_v1_inputs_scanrefer.project_bbox_3d_to_2d",
        lambda *a, **kw: None,
    )

    out = compute_proposal_frame_views(
        proposal_by_id={0: {"id": 0, "bbox_3d": [1.0] * 9, "label": "chair"}},
        frame_visibility={0: [0]},
        frame_by_id={
            0: SceneFrame(
                frame_id=0, raw_frame_id=10,
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """If projection returns x2<x1 or y2<y1, normalize so x1<=x2 and y1<=y2."""
    import numpy as np

    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import (
        SceneFrame,
        compute_proposal_frame_views,
    )

    raw_root, scene_id = _build_scene_for_cvra(tmp_path)

    monkeypatch.setattr(
        "evaluation.scripts.prepare_pack_v1_inputs_scanrefer.project_bbox_3d_to_2d",
        lambda *a, **kw: (200, 100, 50, 40),  # x2<x1 and y2<y1
    )

    out = compute_proposal_frame_views(
        proposal_by_id={0: {"id": 0, "bbox_3d": [1.0] * 9, "label": "chair"}},
        frame_visibility={0: [0]},
        frame_by_id={
            0: SceneFrame(
                frame_id=0, raw_frame_id=10,
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
        "evaluation.scripts.prepare_pack_v1_inputs_scanrefer.project_bbox_3d_to_2d",
        lambda *a, **kw: (10, 20, 100, 200),
    )

    out = compute_proposal_frame_views(
        proposal_by_id={0: {"id": 0, "bbox_3d": [1.0] * 9, "label": "chair"}},
        frame_visibility={0: [0]},
        frame_by_id={
            0: SceneFrame(
                frame_id=0, raw_frame_id=10,
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
                    frame_id=0, raw_frame_id=10,
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
