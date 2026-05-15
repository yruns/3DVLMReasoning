"""v9.1 mark_frame_with_bbox tool: high-contrast annotated single-frame view."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, tool

from agents.catalog import SceneProposal
from agents.runtime.scene_runtime import get_scene_catalog
from agents.tools.scene_perception import _gate


def _norm_category(category: str) -> str:
    return " ".join(str(category).strip().lower().split())


def _filter_visible(
    proposals: list[SceneProposal],
    frame_id: int,
    labels: list[str],
    ids: list[int],
) -> list[SceneProposal]:
    wanted_cat = {_norm_category(c) for c in labels if c}
    wanted_ids = {int(i) for i in ids}
    out: list[SceneProposal] = []
    for p in proposals:
        if frame_id not in p.frame_views:
            continue
        cat_match = bool(wanted_cat) and _norm_category(p.category) in wanted_cat
        id_match = p.proposal_id in wanted_ids
        if cat_match or id_match:
            out.append(p)
    return out


def build_mark_frame_with_bbox_tool(runtime: Any) -> BaseTool:
    @tool
    def mark_frame_with_bbox(
        frame_id: int,
        labels: list[str] | None = None,
        ids: list[int] | None = None,
    ) -> str:
        """Render one first-person frame with high-contrast bboxes for the labels / ids you name.

        Requires at least one of `labels` (category names) or `ids` (proposal ids).
        Detailed usage in 'scene-exploration-playbook'.
        """
        labels_in = [str(c) for c in (labels or []) if c]
        ids_in = [int(i) for i in (ids or [])]
        request = {"frame_id": int(frame_id), "labels": labels_in, "ids": ids_in}
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("mark_frame_with_bbox", request, gate)
            return gate
        if not labels_in and not ids_in:
            err = (
                "ERROR: mark_frame_with_bbox requires at least one of {labels, ids}; "
                "to view plain RGB, request the frame via a selector"
            )
            runtime.record("mark_frame_with_bbox", request, err)
            return err
        catalog = get_scene_catalog(runtime)
        if int(frame_id) not in {int(f) for f in catalog.valid_frame_ids}:
            err = (
                f"ERROR: frame_id={frame_id} not in valid_frame_ids; "
                f"available[:20]={sorted(int(f) for f in catalog.valid_frame_ids)[:20]}"
            )
            runtime.record("mark_frame_with_bbox", request, err)
            return err
        proposals_here = [p for p in catalog.proposals if int(frame_id) in p.frame_views]
        visible = _filter_visible(proposals_here, int(frame_id), labels_in, ids_in)
        if not visible:
            err = (
                f"ERROR: no visible proposals matched filters for frame_id={frame_id}; "
                f"filtered_by={{'labels': {labels_in}, 'ids': {ids_in}}}; "
                f"visible_proposals={[p.proposal_id for p in proposals_here]}"
            )
            runtime.record("mark_frame_with_bbox", request, err)
            return err
        # Rendering implementation comes in Task 3-4. For now, return a placeholder
        # that still meets the response schema so downstream guards parse it.
        placeholder = f"frame_id={frame_id} mark image PLACEHOLDER; visible_proposals=[]; categories=[]; left_to_right=[]; boxes_2d={{}}"
        runtime.record("mark_frame_with_bbox", request, placeholder)
        return placeholder

    return mark_frame_with_bbox


__all__ = ["build_mark_frame_with_bbox_tool"]
