"""v9 scene-perception tools: view_bev / list_scene_proposals / inspect_proposal.

All tools share the `scene-exploration-playbook` skill gate.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool

from agents.runtime.scene_runtime import get_scene_catalog

SCENE_EXPLORATION_SKILL = "scene-exploration-playbook"


def _gate(runtime: Any) -> str | None:
    if SCENE_EXPLORATION_SKILL not in runtime.skills_loaded:
        return f"ERROR: load_skill({SCENE_EXPLORATION_SKILL!r}) before calling this tool."
    return None


def _in_bev_box(position_3d: tuple[float, float, float], box: list[float]) -> bool:
    xmin, ymin, xmax, ymax = box
    return xmin <= position_3d[0] <= xmax and ymin <= position_3d[1] <= ymax


def build_scene_perception_tools(runtime: Any) -> list[BaseTool]:
    @tool
    def list_scene_proposals(
        category: str | None = None,
        region_bev: list[float] | None = None,
        limit: int | None = None,
    ) -> str:
        """Scene-scoped proposal list. Detailed usage in 'scene-exploration-playbook'."""
        request = {"category": category, "region_bev": region_bev, "limit": limit}
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("list_scene_proposals", request, gate)
            return gate
        catalog = get_scene_catalog(runtime)
        proposals = list(catalog.proposals)
        if category is not None:
            cat_norm = str(category).strip().lower()
            proposals = [p for p in proposals if p.category.strip().lower() == cat_norm]
        if region_bev is not None:
            if len(region_bev) != 4:
                err = "ERROR: region_bev must be [xmin, ymin, xmax, ymax]"
                runtime.record("list_scene_proposals", request, err)
                return err
            proposals = [p for p in proposals if _in_bev_box(p.position_3d, list(region_bev))]
        if limit is not None and limit >= 0:
            proposals = proposals[:limit]
        payload = {
            "count": len(proposals),
            "proposals": [
                {
                    "proposal_id": p.proposal_id,
                    "category": p.category,
                    "position_3d": list(p.position_3d),
                    "frame_count": len(p.frame_views),
                }
                for p in proposals
            ],
        }
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("list_scene_proposals", request, text)
        return text

    @tool
    def inspect_proposal(proposal_id: int) -> str:
        """Return position, 9dof bbox, source, and frames-appeared for one proposal."""
        request = {"proposal_id": int(proposal_id)}
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("inspect_proposal", request, gate)
            return gate
        catalog = get_scene_catalog(runtime)
        proposal = catalog.proposal_by_id(int(proposal_id))
        if proposal is None:
            err = (
                f"ERROR: proposal_id={proposal_id} not in catalog; "
                f"available count={len(catalog.proposals)}"
            )
            runtime.record("inspect_proposal", request, err)
            return err
        payload = {
            "proposal_id": proposal.proposal_id,
            "category": proposal.category,
            "position_3d": list(proposal.position_3d),
            "bbox_3d_9dof": list(proposal.bbox_3d_9dof) if proposal.bbox_3d_9dof else None,
            "frames_appeared": sorted(proposal.frame_views.keys()),
            "source": proposal.source,
        }
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("inspect_proposal", request, text)
        return text

    return [list_scene_proposals, inspect_proposal]


__all__ = [
    "SCENE_EXPLORATION_SKILL",
    "build_scene_perception_tools",
]
