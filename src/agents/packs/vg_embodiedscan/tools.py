"""EmbodiedScan VG tools. All bodies FAIL-LOUD on missing primary skill."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool

PRIMARY_SKILL = "vg-grounding-playbook"


def _gate(runtime: Any) -> str | None:
    """Return ERROR string if skill not loaded, else None."""
    if PRIMARY_SKILL not in runtime.skills_loaded:
        return f"ERROR: load_skill({PRIMARY_SKILL!r}) before calling this tool."
    return None


def build_vg_tools(runtime: Any) -> list[BaseTool]:
    ctx = runtime.task_ctx  # VgEmbodiedScanCtx

    @tool
    def list_keyframes_with_proposals() -> str:
        """VG tool. Detailed usage in skill 'vg-grounding-playbook'."""
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("list_keyframes_with_proposals", {}, gate)
            return gate
        items = []
        for kf in runtime.bundle.keyframes:
            fid = kf.frame_id
            visible = ctx.frame_index.get(fid, []) if fid is not None else []
            items.append(
                {
                    "keyframe_idx": kf.keyframe_idx,
                    "frame_id": fid,
                    "visible_proposal_ids": visible,
                    "n_proposals": len(visible),
                    "annotated_image": str(ctx.annotated_image_dir / f"frame_{fid}.png"),
                }
            )
        text = json.dumps(items, ensure_ascii=False)
        runtime.record("list_keyframes_with_proposals", {}, text)
        return text

    @tool
    def view_keyframe_marked(frame_id: int) -> str:
        """VG tool. Detailed usage in skill 'vg-grounding-playbook'."""
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("view_keyframe_marked", {"frame_id": frame_id}, gate)
            return gate
        if frame_id not in ctx.frame_index:
            err = (
                f"ERROR: frame_id={frame_id} not in proposal index; "
                f"available: {sorted(ctx.frame_index.keys())[:20]}"
            )
            runtime.record("view_keyframe_marked", {"frame_id": frame_id}, err)
            return err
        marked_path = ctx.annotated_image_dir / f"frame_{frame_id}.png"
        if not marked_path.exists():
            err = f"ERROR: annotated image not found: {marked_path}"
            runtime.record("view_keyframe_marked", {"frame_id": frame_id}, err)
            return err
        visible = ctx.frame_index[frame_id]
        # Mark the path as a fresh image to inject into the next user message
        runtime.bundle.extra_metadata = dict(runtime.bundle.extra_metadata or {})
        runtime.bundle.extra_metadata.setdefault("vg_pending_images", []).append(str(marked_path))
        runtime.mark_evidence_updated()
        body = (
            f"frame_id={frame_id} marked image at {marked_path}; "
            f"visible_proposals={visible}; "
            f"categories={[next((p.category for p in ctx.proposals if p.id == pid), '?') for pid in visible]}"
        )
        runtime.record("view_keyframe_marked", {"frame_id": frame_id}, body)
        return body

    @tool
    def inspect_proposal(proposal_id: int) -> str:
        """VG tool. Detailed usage in skill 'vg-grounding-playbook'."""
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("inspect_proposal", {"proposal_id": proposal_id}, gate)
            return gate
        proposal = next((p for p in ctx.proposals if p.id == proposal_id), None)
        if proposal is None:
            err = (
                f"ERROR: proposal_id={proposal_id} not in pool; "
                f"available count={len(ctx.proposals)}"
            )
            runtime.record("inspect_proposal", {"proposal_id": proposal_id}, err)
            return err
        payload = {
            "proposal_id": proposal.id,
            "category": proposal.category,
            "score": proposal.score,
            "bbox_3d_9dof": list(proposal.bbox_3d_9dof),
            "frames_appeared": ctx.proposal_index.get(proposal_id, []),
            "source": ctx.proposal_pool_source,
        }
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("inspect_proposal", {"proposal_id": proposal_id}, text)
        return text

    @tool
    def find_proposals_by_category(category: str) -> str:
        """VG tool. Detailed usage in skill 'vg-grounding-playbook'."""
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("find_proposals_by_category", {"category": category}, gate)
            return gate
        ids = [p.id for p in ctx.proposals if p.category.strip().lower() == category.strip().lower()]
        payload = {
            "category": category,
            "proposal_ids": ids,
            "available_categories": sorted({p.category for p in ctx.proposals if p.category}),
        }
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("find_proposals_by_category", {"category": category}, text)
        return text

    return [
        list_keyframes_with_proposals,
        view_keyframe_marked,
        inspect_proposal,
        find_proposals_by_category,
    ]


__all__ = ["build_vg_tools", "PRIMARY_SKILL"]
