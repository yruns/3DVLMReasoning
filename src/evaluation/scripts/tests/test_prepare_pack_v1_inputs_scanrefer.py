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
