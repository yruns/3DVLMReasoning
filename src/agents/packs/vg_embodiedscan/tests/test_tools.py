"""VG pack tools: per-tool tests + FAIL-LOUD gate."""

from __future__ import annotations

import json
from pathlib import Path

from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
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
    bundle = Stage2EvidenceBundle()
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


def test_list_frame_proposals_returns_frame_inventory(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "list_frame_proposals")
    payload = json.loads(tool.invoke({"frame_id": 10}))
    assert payload["frame_id"] == 10
    assert payload["visible_proposal_ids"] == [0, 1]
    assert payload["left_to_right"] == ["#0 chair", "#1 desk"]
    assert payload["boxes_2d"] == {"0": [10, 20, 30, 40], "1": [80, 20, 120, 40]}


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


def test_compare_proposals_spatial_returns_stable_evidence_id(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")

    payload0 = json.loads(
        tool.invoke({"candidate_ids": [0, 2], "anchor_id": 1, "relation": "left_of"})
    )
    trace0 = rs.tool_trace[-1]
    payload1 = json.loads(
        tool.invoke({"candidate_ids": [2], "anchor_id": 1, "relation": "right_of"})
    )
    trace1 = rs.tool_trace[-1]

    assert payload0["evidence_id"] == "compare_proposals_spatial:0"
    assert trace0.tool_input["evidence_id"] == payload0["evidence_id"]
    assert payload0["candidate_ids"] == [0, 2]
    assert payload0["anchor_id"] == 1
    assert payload0["relation"] == "left_of"

    assert payload1["evidence_id"] == "compare_proposals_spatial:1"
    assert trace1.tool_input["evidence_id"] == payload1["evidence_id"]
    assert payload1["candidate_ids"] == [2]
    assert payload1["anchor_id"] == 1
    assert payload1["relation"] == "right_of"


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


def test_compare_candidates_to_anchors_reports_anchor_disagreement(
    tmp_path: Path,
) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    rs.task_ctx.proposals = [
        Proposal(
            id=19,
            bbox_3d_9dof=[0, 0, 0, 1, 1, 1, 0, 0, 0],
            category="pillow",
            score=0.9,
        ),
        Proposal(
            id=20,
            bbox_3d_9dof=[10, 0, 0, 1, 1, 1, 0, 0, 0],
            category="pillow",
            score=0.8,
        ),
        Proposal(
            id=3,
            bbox_3d_9dof=[-1, 0, 0, 1, 1, 1, 0, 0, 0],
            category="door",
            score=0.7,
        ),
        Proposal(
            id=24,
            bbox_3d_9dof=[11, 0, 0, 1, 1, 1, 0, 0, 0],
            category="door",
            score=0.7,
        ),
    ]
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_candidates_to_anchors")

    payload = json.loads(
        tool.invoke(
            {
                "candidate_ids": [19, 20],
                "anchor_ids": [3, 24],
                "relation": "farthest_from",
            }
        )
    )

    assert payload["evidence_id"] == "compare_candidates_to_anchors:0"
    assert rs.tool_trace[-1].tool_input["evidence_id"] == payload["evidence_id"]
    assert payload["candidate_ids"] == [19, 20]
    assert payload["anchor_ids"] == [3, 24]
    assert payload["relation"] == "farthest_from"
    assert payload["per_anchor"] == [
        {
            "anchor_id": 3,
            "ranked_ids": [20, 19],
            "distances": [11.0, 1.0],
            "horizontal_distances": [11.0, 1.0],
        },
        {
            "anchor_id": 24,
            "ranked_ids": [19, 20],
            "distances": [11.0, 1.0],
            "horizontal_distances": [11.0, 1.0],
        },
    ]
    assert payload["top1_by_anchor"] == {"3": 20, "24": 19}
    assert payload["anchor_disagreement"] is True
    assert payload["globally_consistent_top1"] is None


def test_compare_proposals_spatial_accepts_common_relation_aliases(
    tmp_path: Path,
) -> None:
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

    for alias in ("closer_to", "nearest_to", "nearest", "closest"):
        payload = json.loads(
            tool.invoke(
                {
                    "candidate_ids": [0, 1],
                    "anchor_id": 2,
                    "relation": alias,
                }
            )
        )
        assert payload["relation"] == "closest_to"
        assert payload["requested_relation"] == alias
        assert payload["ranked_ids"] == [1, 0]

    for alias in ("farther_from", "further_from", "furthest_from", "furthest"):
        payload = json.loads(
            tool.invoke(
                {
                    "candidate_ids": [0, 1],
                    "anchor_id": 2,
                    "relation": alias,
                }
            )
        )
        assert payload["relation"] == "farthest_from"
        assert payload["requested_relation"] == alias
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


def test_compare_proposals_spatial_below_prefers_horizontal_alignment(
    tmp_path: Path,
) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    rs.task_ctx.proposals = [
        Proposal(
            id=0, bbox_3d_9dof=[0, 0, -2, 1, 1, 1, 0, 0, 0], category="printer", score=0.9
        ),
        Proposal(
            id=1, bbox_3d_9dof=[5, 0, -3, 1, 1, 1, 0, 0, 0], category="printer", score=0.9
        ),
        Proposal(
            id=2,
            bbox_3d_9dof=[0, 0, 0, 1, 1, 1, 0, 0, 0],
            category="window",
            score=0.8,
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
    assert payload["horizontal_distances"] == [0.0, 5.0]
    assert payload["vertical_offsets"] == [-2.0, -3.0]


def test_compare_proposals_spatial_above_prefers_horizontal_alignment(
    tmp_path: Path,
) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    rs.task_ctx.proposals = [
        Proposal(
            id=0, bbox_3d_9dof=[0, 0, 2, 1, 1, 1, 0, 0, 0], category="cabinet", score=0.9
        ),
        Proposal(
            id=1, bbox_3d_9dof=[5, 0, 3, 1, 1, 1, 0, 0, 0], category="cabinet", score=0.9
        ),
        Proposal(
            id=2,
            bbox_3d_9dof=[0, 0, 0, 1, 1, 1, 0, 0, 0],
            category="counter",
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
    assert payload["horizontal_distances"] == [0.0, 5.0]
    assert payload["vertical_offsets"] == [2.0, 3.0]


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
