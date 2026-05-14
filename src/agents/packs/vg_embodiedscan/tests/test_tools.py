"""VG pack tools: per-tool tests + FAIL-LOUD gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from agents.core.agent_config import Stage2DeepAgentConfig, Stage2TaskType
from agents.core.task_types import KeyframeEvidence, Stage2EvidenceBundle
from agents.packs.vg_embodiedscan.ctx import (
    Proposal,
    ProposalFrameView,
    VgEmbodiedScanCtx,
)
from agents.packs.vg_embodiedscan.tools import build_vg_tools
from agents.runtime.base import Stage2RuntimeState


def _runtime(tmp_path: Path) -> Stage2RuntimeState:
    annotated = tmp_path / "ann"
    annotated.mkdir()
    from PIL import Image

    Image.new("RGB", (160, 80), color=(220, 220, 220)).save(
        tmp_path / "raw10.png", format="PNG"
    )
    Image.new("RGB", (160, 80), color=(220, 220, 220)).save(
        tmp_path / "raw11.png", format="PNG"
    )
    bundle = Stage2EvidenceBundle(
        keyframes=[
            KeyframeEvidence(keyframe_idx=0, image_path="a.png", frame_id=10),
            KeyframeEvidence(keyframe_idx=1, image_path="b.png", frame_id=11),
        ]
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.task_ctx = VgEmbodiedScanCtx(
        proposal_pool_source="vdetr",
        proposals=[
            Proposal(
                id=0,
                bbox_3d_9dof=[0] * 9,
                category="chair",
                score=0.9,
                frame_views={
                    10: ProposalFrameView(
                        proposal_id=0,
                        frame_id=10,
                        bbox_2d=(10, 20, 30, 40),
                        raw_rgb_path=tmp_path / "raw10.png",
                    )
                },
            ),
            Proposal(
                id=1,
                bbox_3d_9dof=[1] * 9,
                category="desk",
                score=0.8,
                frame_views={
                    10: ProposalFrameView(
                        proposal_id=1,
                        frame_id=10,
                        bbox_2d=(80, 20, 120, 40),
                        raw_rgb_path=tmp_path / "raw10.png",
                    ),
                    11: ProposalFrameView(
                        proposal_id=1,
                        frame_id=11,
                        bbox_2d=(30, 20, 60, 40),
                        raw_rgb_path=tmp_path / "raw11.png",
                    ),
                },
            ),
            Proposal(
                id=2,
                bbox_3d_9dof=[2] * 9,
                category="chair",
                score=0.7,
                frame_views={
                    11: ProposalFrameView(
                        proposal_id=2,
                        frame_id=11,
                        bbox_2d=(90, 20, 120, 40),
                        raw_rgb_path=tmp_path / "raw11.png",
                    )
                },
            ),
        ],
        frame_index={10: [0, 1], 11: [1, 2]},
        proposal_index={0: [10], 1: [10, 11], 2: [11]},
        annotated_image_dir=annotated,
    )
    return rs


def test_view_keyframe_marked_returns_image_content(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    # create a fake marked image
    marked = rs.task_ctx.annotated_image_dir / "frame_10.png"
    marked.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header

    tool = next(t for t in build_vg_tools(rs) if t.name == "view_keyframe_marked")
    response = tool.invoke({"frame_id": 10})
    assert "frame_10.png" in response
    assert "visible_proposals" in response
    assert "[0, 1]" in response or "0, 1" in response
    assert "left_to_right" in response
    assert "0:chair@x=20.0" in response
    assert "1:desk@x=100.0" in response
    assert "boxes_2d={0: [10, 20, 30, 40], 1: [80, 20, 120, 40]}" in response


def test_list_frame_proposals_returns_frame_inventory(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "list_frame_proposals")
    payload = json.loads(tool.invoke({"frame_id": 10}))
    assert payload["frame_id"] == 10
    assert payload["visible_proposal_ids"] == [0, 1]
    assert payload["left_to_right"] == ["#0 chair", "#1 desk"]
    assert payload["boxes_2d"] == {"0": [10, 20, 30, 40], "1": [80, 20, 120, 40]}


def test_view_keyframe_marked_filters_by_category_and_renders_dynamic_image(
    tmp_path: Path,
) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "view_keyframe_marked")
    response = tool.invoke({"frame_id": 10, "categories": ["chair"]})
    assert "filtered marked image" in response
    assert "filtered_by={'categories': ['chair'], 'proposal_ids': []}" in response
    assert "visible_proposals=[0]" in response
    assert "categories=['chair']" in response
    assert "boxes_2d={0: [10, 20, 30, 40]}" in response
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert len(pending) == 1
    assert "filtered_marks" in pending[0]
    assert Path(pending[0]).exists()


def test_view_keyframe_marked_filters_by_proposal_ids(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "view_keyframe_marked")
    response = tool.invoke({"frame_id": 10, "proposal_ids": [1]})
    assert "filtered marked image" in response
    assert "visible_proposals=[1]" in response
    assert "categories=['desk']" in response
    assert "boxes_2d={1: [80, 20, 120, 40]}" in response


def test_view_keyframe_marked_accepts_single_value_filters(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "view_keyframe_marked")

    category_response = tool.invoke({"frame_id": 10, "categories": "chair"})
    assert "filtered marked image" in category_response
    assert "visible_proposals=[0]" in category_response

    id_response = tool.invoke({"frame_id": 10, "proposal_ids": "1"})
    assert "filtered marked image" in id_response
    assert "visible_proposals=[1]" in id_response


def test_view_keyframe_marked_filter_without_matches_errors(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "view_keyframe_marked")
    response = tool.invoke({"frame_id": 10, "categories": ["lamp"]})
    assert response.startswith("ERROR")
    assert "no visible proposals matched filters" in response
    assert "vg_pending_images" not in rs.bundle.extra_metadata


def test_view_keyframe_marked_unknown_frame_errors(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "view_keyframe_marked")
    response = tool.invoke({"frame_id": 999})
    assert response.startswith("ERROR")


def test_view_keyframe_marked_image_drained_into_evidence_update(
    tmp_path: Path,
) -> None:
    """End-to-end: vg_pending_images queued by view_keyframe_marked must be
    drained into the chassis's next-turn user message via
    build_evidence_update_message."""

    from agents.core.agent_config import Stage2DeepAgentConfig
    from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime

    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    # Pre-mark all bundle keyframe images as already-seen so the only
    # NEW image picked up is the marked one queued by view_keyframe_marked.
    for kf in rs.bundle.keyframes:
        rs.seen_image_paths.add(kf.image_path)

    # write a real PNG so chassis image_to_data_url can decode it
    from PIL import Image

    marked = rs.task_ctx.annotated_image_dir / "frame_10.png"
    Image.new("RGB", (4, 4), color=(0, 0, 0)).save(marked, format="PNG")

    tool = next(t for t in build_vg_tools(rs) if t.name == "view_keyframe_marked")
    tool.invoke({"frame_id": 10})

    # The queue must have been populated.
    assert str(marked) in rs.bundle.extra_metadata["vg_pending_images"]

    # Now drain via the chassis injector.
    runtime = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    msg = runtime.build_evidence_update_message(rs)
    assert msg is not None, "expected an evidence-update message but got None"
    # The path should be present in either the rendered text or the
    # multimodal image_url block; check both representations.
    serialized = str(msg.content)
    assert str(marked) in serialized or any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for part in (msg.content if isinstance(msg.content, list) else [])
    )
    # And the queue must be drained so we don't re-inject next turn.
    assert rs.bundle.extra_metadata["vg_pending_images"] == []


def test_inspect_proposal_returns_metadata_and_frames(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "inspect_proposal")
    payload = json.loads(tool.invoke({"proposal_id": 1}))
    assert payload["proposal_id"] == 1
    assert payload["category"] == "desk"
    assert payload["score"] == 0.8
    assert payload["frames_appeared"] == [10, 11]
    assert payload["bbox_3d_9dof"] == [1] * 9


def test_inspect_proposal_unknown_id_errors(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "inspect_proposal")
    response = tool.invoke({"proposal_id": 99})
    assert response.startswith("ERROR")


def test_compare_proposals_spatial_closest_to(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    # set proposal centers far apart so order is deterministic
    rs.task_ctx.proposals = [
        Proposal(
            id=0, bbox_3d_9dof=[0, 0, 0, 1, 1, 1, 0, 0, 0], category="chair", score=0.9
        ),
        Proposal(
            id=1, bbox_3d_9dof=[5, 5, 5, 1, 1, 1, 0, 0, 0], category="chair", score=0.7
        ),
        Proposal(
            id=2,
            bbox_3d_9dof=[10, 10, 10, 1, 1, 1, 0, 0, 0],
            category="desk",
            score=0.8,
        ),
    ]
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")
    payload = json.loads(
        tool.invoke(
            {
                "candidate_ids": [0, 1],
                "anchor_id": 2,
                "relation": "closest_to",
            }
        )
    )
    assert payload["ranked_ids"] == [1, 0]


def test_compare_proposals_spatial_farthest_from(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    rs.task_ctx.proposals = [
        Proposal(
            id=0, bbox_3d_9dof=[0, 0, 0, 1, 1, 1, 0, 0, 0], category="chair", score=0.9
        ),
        Proposal(
            id=1, bbox_3d_9dof=[5, 5, 5, 1, 1, 1, 0, 0, 0], category="chair", score=0.7
        ),
        Proposal(
            id=2,
            bbox_3d_9dof=[10, 10, 10, 1, 1, 1, 0, 0, 0],
            category="desk",
            score=0.8,
        ),
    ]
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")
    payload = json.loads(
        tool.invoke(
            {
                "candidate_ids": [0, 1],
                "anchor_id": 2,
                "relation": "farthest_from",
            }
        )
    )
    assert payload["ranked_ids"] == [0, 1]


def test_compare_proposals_spatial_above_uses_z_axis(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    rs.task_ctx.proposals = [
        Proposal(
            id=0,
            bbox_3d_9dof=[0, 0, 3, 1, 1, 1, 0, 0, 0],
            category="cabinet",
            score=0.9,
        ),
        Proposal(
            id=1,
            bbox_3d_9dof=[0, 0, 1, 1, 1, 1, 0, 0, 0],
            category="cabinet",
            score=0.7,
        ),
        Proposal(
            id=2,
            bbox_3d_9dof=[0, 0, 0, 1, 1, 1, 0, 0, 0],
            category="refrigerator",
            score=0.8,
        ),
    ]
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")
    payload = json.loads(
        tool.invoke(
            {
                "candidate_ids": [0, 1],
                "anchor_id": 2,
                "relation": "above",
            }
        )
    )
    assert payload["ranked_ids"] == [0, 1]
    assert payload["vertical_offsets"] == [3.0, 1.0]


def test_compare_proposals_spatial_below_uses_z_axis(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    rs.task_ctx.proposals = [
        Proposal(
            id=0, bbox_3d_9dof=[0, 0, -2, 1, 1, 1, 0, 0, 0], category="box", score=0.9
        ),
        Proposal(
            id=1, bbox_3d_9dof=[0, 0, 2, 1, 1, 1, 0, 0, 0], category="box", score=0.7
        ),
        Proposal(
            id=2, bbox_3d_9dof=[0, 0, 0, 1, 1, 1, 0, 0, 0], category="table", score=0.8
        ),
    ]
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")
    payload = json.loads(
        tool.invoke(
            {
                "candidate_ids": [0, 1],
                "anchor_id": 2,
                "relation": "below",
            }
        )
    )
    assert payload["ranked_ids"] == [0, 1]
    assert payload["vertical_offsets"] == [-2.0, 2.0]


def test_compare_proposals_spatial_left_right_use_coviewed_2d_geometry(
    tmp_path: Path,
) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")
    left_payload = json.loads(
        tool.invoke(
            {
                "candidate_ids": [0, 2],
                "anchor_id": 1,
                "relation": "left_of",
            }
        )
    )
    right_payload = json.loads(
        tool.invoke(
            {
                "candidate_ids": [0, 2],
                "anchor_id": 1,
                "relation": "right_of",
            }
        )
    )

    assert left_payload["ranked_ids"] == [0, 2]
    assert left_payload["mean_2d_center_offsets_x"] == [-80.0, 60.0]
    assert left_payload["supporting_frame_counts"] == [1, 0]
    assert left_payload["contradicting_frame_counts"] == [0, 1]

    assert right_payload["ranked_ids"] == [2, 0]
    assert right_payload["mean_2d_center_offsets_x"] == [60.0, -80.0]
    assert right_payload["supporting_frame_counts"] == [1, 0]
    assert right_payload["contradicting_frame_counts"] == [0, 1]


def test_compare_proposals_spatial_next_to_and_near_use_floor_distance(
    tmp_path: Path,
) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    rs.task_ctx.proposals = [
        Proposal(
            id=0,
            bbox_3d_9dof=[0.5, 0, 5, 1, 1, 1, 0, 0, 0],
            category="box",
            score=0.9,
        ),
        Proposal(
            id=1,
            bbox_3d_9dof=[2, 0, 0, 1, 1, 1, 0, 0, 0],
            category="box",
            score=0.7,
        ),
        Proposal(
            id=2,
            bbox_3d_9dof=[0, 0, 0, 1, 1, 1, 0, 0, 0],
            category="table",
            score=0.8,
        ),
    ]
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")
    for relation in ("next_to", "near"):
        payload = json.loads(
            tool.invoke(
                {
                    "candidate_ids": [0, 1],
                    "anchor_id": 2,
                    "relation": relation,
                }
            )
        )
        assert payload["ranked_ids"] == [0, 1]
        assert payload["horizontal_distances"] == [0.5, 2.0]


def test_compare_proposals_spatial_unknown_relation_errors(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")
    response = tool.invoke(
        {
            "candidate_ids": [0, 1],
            "anchor_id": 2,
            "relation": "diagonal_to",
        }
    )
    assert response.startswith("ERROR")


