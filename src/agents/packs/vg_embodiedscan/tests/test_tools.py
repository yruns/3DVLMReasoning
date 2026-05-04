"""VG pack tools: per-tool tests + FAIL-LOUD gate."""

from __future__ import annotations

import json
from pathlib import Path

from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import KeyframeEvidence, Stage2EvidenceBundle
from agents.packs.vg_embodiedscan.ctx import (
    Proposal,
    VgEmbodiedScanCtx,
)
from agents.packs.vg_embodiedscan.tools import build_vg_tools
from agents.runtime.base import Stage2RuntimeState


def _runtime(tmp_path: Path) -> Stage2RuntimeState:
    annotated = tmp_path / "ann"
    annotated.mkdir()
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
            Proposal(id=0, bbox_3d_9dof=[0] * 9, category="chair", score=0.9),
            Proposal(id=1, bbox_3d_9dof=[1] * 9, category="desk", score=0.8),
            Proposal(id=2, bbox_3d_9dof=[2] * 9, category="chair", score=0.7),
        ],
        frame_index={10: [0, 1], 11: [1, 2]},
        proposal_index={0: [10], 1: [10, 11], 2: [11]},
        annotated_image_dir=annotated,
    )
    return rs


def test_list_keyframes_with_proposals_gates_on_skill(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    tool = next(
        t for t in build_vg_tools(rs) if t.name == "list_keyframes_with_proposals"
    )
    response = tool.invoke({})
    assert response.startswith("ERROR")
    assert "vg-grounding-playbook" in response


def test_list_keyframes_with_proposals_returns_structured(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(
        t for t in build_vg_tools(rs) if t.name == "list_keyframes_with_proposals"
    )
    payload = json.loads(tool.invoke({}))
    assert len(payload) == 2
    assert payload[0]["frame_id"] == 10
    assert payload[0]["visible_proposal_ids"] == [0, 1]
    assert payload[0]["annotated_image"].endswith("/ann/frame_10.png")


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


def test_find_proposals_by_category_lists_ids(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "find_proposals_by_category")
    payload = json.loads(tool.invoke({"category": "chair"}))
    assert payload["proposal_ids"] == [0, 2]


def test_find_proposals_by_category_unknown_returns_empty(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "find_proposals_by_category")
    payload = json.loads(tool.invoke({"category": "spaceship"}))
    assert payload["proposal_ids"] == []
    assert "available_categories" in payload


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


def test_compare_proposals_spatial_unknown_relation_errors(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")
    response = tool.invoke(
        {
            "candidate_ids": [0, 1],
            "anchor_id": 2,
            "relation": "left_of",
        }
    )
    assert response.startswith("ERROR")


# -- select_among_proposals --------------------------------------------------


def _wire_vlm(rs: Stage2RuntimeState, response_text: str) -> list[list]:
    """Attach a stub vlm_judge that records calls and returns response_text."""
    captured: list[list] = []

    def _judge(messages):
        captured.append(messages)
        return response_text

    rs.vlm_judge = _judge
    rs.image_to_data_url = lambda p: f"data:image/jpeg;base64,FAKE({p})"
    return captured


def _annotate_for(rs: Stage2RuntimeState, frame_ids: list[int]) -> None:
    for fid in frame_ids:
        (rs.task_ctx.annotated_image_dir / f"frame_{fid}.png").write_bytes(
            b"\x89PNG\r\n\x1a\n"
        )


def test_select_among_proposals_gates_on_skill(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    tool = next(t for t in build_vg_tools(rs) if t.name == "select_among_proposals")
    response = tool.invoke({"candidate_ids": [0, 2], "description": "the chair"})
    assert response.startswith("ERROR")
    assert "vg-grounding-playbook" in response


def test_select_among_proposals_requires_two_candidates(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    _wire_vlm(rs, "")
    tool = next(t for t in build_vg_tools(rs) if t.name == "select_among_proposals")
    response = tool.invoke({"candidate_ids": [0], "description": "the chair"})
    assert response.startswith("ERROR")
    assert "length >= 2" in response


def test_select_among_proposals_validates_pool_membership(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    _wire_vlm(rs, "")
    tool = next(t for t in build_vg_tools(rs) if t.name == "select_among_proposals")
    response = tool.invoke({"candidate_ids": [0, 999], "description": "the chair"})
    assert response.startswith("ERROR")
    assert "999" in response


def test_select_among_proposals_requires_vlm_hooks(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    # do not wire vlm_judge / image_to_data_url
    tool = next(t for t in build_vg_tools(rs) if t.name == "select_among_proposals")
    response = tool.invoke({"candidate_ids": [0, 2], "description": "the chair"})
    assert response.startswith("ERROR")
    assert "VLM hooks" in response


def test_select_among_proposals_calls_vlm_with_one_image_per_candidate(
    tmp_path: Path,
) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    _annotate_for(rs, [10, 11])
    captured = _wire_vlm(
        rs, '{"selected_proposal_id": 2, "reasoning": "left chair"}'
    )

    tool = next(t for t in build_vg_tools(rs) if t.name == "select_among_proposals")
    response = tool.invoke(
        {"candidate_ids": [0, 2], "description": "the chair near the wall"}
    )
    payload = json.loads(response)
    assert payload["selected_proposal_id"] == 2
    assert payload["reasoning"] == "left chair"
    assert payload["candidate_ids"] == [0, 2]

    # one VLM call, system + human, with two image_url parts (one per candidate)
    assert len(captured) == 1
    messages = captured[0]
    assert len(messages) == 2  # SystemMessage + HumanMessage
    human_content = messages[1].content
    image_parts = [c for c in human_content if c.get("type") == "image_url"]
    assert len(image_parts) == 2
    assert all("FAKE(" in p["image_url"]["url"] for p in image_parts)


def test_select_among_proposals_strips_json_fence(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    _annotate_for(rs, [10, 11])
    fenced = (
        "```json\n"
        '{"selected_proposal_id": 0, "reasoning": "front chair"}\n'
        "```"
    )
    _wire_vlm(rs, fenced)
    tool = next(t for t in build_vg_tools(rs) if t.name == "select_among_proposals")
    payload = json.loads(
        tool.invoke({"candidate_ids": [0, 2], "description": "the front chair"})
    )
    assert payload["selected_proposal_id"] == 0


def test_select_among_proposals_rejects_invalid_vlm_choice(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    _annotate_for(rs, [10, 11])
    _wire_vlm(rs, '{"selected_proposal_id": 99, "reasoning": "out of pool"}')
    tool = next(t for t in build_vg_tools(rs) if t.name == "select_among_proposals")
    response = tool.invoke({"candidate_ids": [0, 2], "description": "the chair"})
    assert response.startswith("ERROR")
    assert "99" in response


def test_select_among_proposals_rejects_unparseable_response(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    _annotate_for(rs, [10, 11])
    _wire_vlm(rs, "I think it is the second one.")
    tool = next(t for t in build_vg_tools(rs) if t.name == "select_among_proposals")
    response = tool.invoke({"candidate_ids": [0, 2], "description": "the chair"})
    assert response.startswith("ERROR")
    assert "could not parse VLM JSON" in response


def test_select_among_proposals_errors_on_missing_annotation(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    _wire_vlm(rs, '{"selected_proposal_id": 0, "reasoning": "x"}')
    # do not write annotated PNGs — the tool should error
    tool = next(t for t in build_vg_tools(rs) if t.name == "select_among_proposals")
    response = tool.invoke({"candidate_ids": [0, 2], "description": "the chair"})
    assert response.startswith("ERROR")
    assert "annotated image missing" in response


def test_attach_vlm_hooks_wires_state_for_tool(tmp_path: Path) -> None:
    """BaseStage2Runtime.attach_vlm_hooks should make state-level hooks available."""
    from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime

    rt = DeepAgentsStage2Runtime()
    rs = _runtime(tmp_path)
    assert rs.vlm_judge is None
    rt.attach_vlm_hooks(rs)
    assert callable(rs.vlm_judge)
    assert callable(rs.image_to_data_url)
