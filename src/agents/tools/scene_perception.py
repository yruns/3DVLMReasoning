"""v9 scene-perception tools: view_bev / list_scene_proposals / inspect_proposal.

All tools share the `scene-exploration-playbook` skill gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from agents.runtime.scene_runtime import get_scene_catalog, make_tool_image_ref

SCENE_EXPLORATION_SKILL = "scene-exploration-playbook"


def _gate(runtime: Any) -> str | None:
    if SCENE_EXPLORATION_SKILL not in runtime.skills_loaded:
        return (
            f"ERROR: load_skill({SCENE_EXPLORATION_SKILL!r}) before calling this tool."
        )
    return None


def _in_bev_box(position_3d: tuple[float, float, float], box: list[float]) -> bool:
    xmin, ymin, xmax, ymax = box
    return xmin <= position_3d[0] <= xmax and ymin <= position_3d[1] <= ymax


def _render_highlighted_bev(
    catalog, highlight_ids: list[int], output_path: Path
) -> Path:
    """Render a highlight overlay using the SAME perspective view params as the base BEV.

    Reads the persisted ``view_params`` sidecar (``*.view.json``) that
    ``build_with_labels`` wrote next to the base BEV PNG, loads the cached
    PNG as the backdrop, and re-runs ``_overlay_proposal_labels`` through
    the perspective branch of ``_project_centroid``. No mesh re-render —
    this is purely a label refresh that lands on the same pixels as the
    original BEV. See Task 18 of the v9.1 plan.
    """
    import cv2

    from query_scene.scene_bev_builder import (
        ScanNetSceneBEVBuilderBase,
        SceneBEVConfig,
        _load_view_params,
        _view_params_path,
    )

    base_png = Path(catalog.bev_image_path)
    view_path = _view_params_path(base_png)
    if not view_path.exists():
        raise FileNotFoundError(
            f"view_params sidecar missing for base BEV {base_png}; "
            "rerun pack-prep so build_with_labels writes the .view.json sidecar"
        )
    view = _load_view_params(view_path)

    class _Backdrop(ScanNetSceneBEVBuilderBase):
        benchmark = "backdrop"

        def resolve_paths(self, scene_id, data_root):
            raise NotImplementedError

    base = cv2.imread(str(base_png))
    if base is None:
        raise FileNotFoundError(f"backing BEV image not readable: {base_png}")
    img = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
    builder = _Backdrop(config=SceneBEVConfig(image_size=img.shape[1]))
    out = builder._overlay_proposal_labels(
        img, catalog.proposals, view, list(highlight_ids)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return output_path


# Renderer-version tag baked into the highlight cache filename so that any
# change to the label rendering (font, colours, anchoring, etc.) invalidates
# stale on-disk overlays automatically. Bump this whenever
# ``_overlay_proposal_labels`` or its inputs change in a user-visible way.
# v9.3 (2026-05-17): top-left default, focused-labels, sharper text.
_HIGHLIGHT_BEV_RENDERER_VERSION: str = "v93"


def _resolve_highlight_ids(
    catalog,
    highlight: list[int] | None,
    categories: list[str] | None,
) -> tuple[list[int], list[str]]:
    """Combine `highlight` (explicit proposal_ids) with `categories` (category
    name filters) into a single, deduplicated, sorted highlight list.

    Returns (resolved_ids, missing_categories). ``missing_categories`` lists
    requested category strings that matched zero catalog entries, so the tool
    can surface a helpful message to the agent.
    """
    ids: set[int] = set()
    if highlight:
        ids.update(int(i) for i in highlight)
    missing: list[str] = []
    if categories:
        norm_to_proposals: dict[str, list[int]] = {}
        for p in catalog.proposals:
            norm_to_proposals.setdefault(str(p.category).strip().lower(), []).append(
                int(p.proposal_id)
            )
        for raw in categories:
            key = str(raw).strip().lower()
            if not key:
                continue
            hits = norm_to_proposals.get(key, [])
            if not hits:
                missing.append(str(raw))
            ids.update(hits)
    return sorted(ids), missing


def build_scene_perception_tools(runtime: Any) -> list[BaseTool]:
    @tool
    def view_bev(
        highlight: list[int] | None = None,
        categories: list[str] | None = None,
    ) -> str:
        """Inject the scene BEV image with optional focused labels.

        v9.3 (current) behavior:

        - ``view_bev()`` (no args) → clean mesh + trajectory + small unlabeled
          dot at each proposal centroid. No text labels by default, so the BEV
          stays readable even in scenes with 50+ proposals. Use this as your
          orientation map.
        - ``view_bev(highlight=[#a, #b])`` → re-renders with text labels
          ("#id category") drawn ONLY on those proposals. Highlighted dots
          enlarge and switch to red.
        - ``view_bev(categories=["chair", "table"])`` → text labels on every
          proposal whose category matches (case-insensitive exact match
          against the catalog category strings). Use this when you don't have
          IDs yet but know the category of interest. Mirrors how
          `mark_frame_with_bbox` adds focused annotations to a frame.
        - Both args may be combined; the union is labeled.

        Detailed usage in 'scene-exploration-playbook'.
        """
        request = {
            "highlight": list(highlight) if highlight else None,
            "categories": list(categories) if categories else None,
        }
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("view_bev", request, gate)
            return gate
        catalog = get_scene_catalog(runtime)
        resolved_ids, missing_cats = _resolve_highlight_ids(
            catalog,
            highlight,
            categories,
        )
        if not resolved_ids:
            image_ref = make_tool_image_ref(
                runtime,
                catalog.bev_image_path,
                metadata={
                    "source_tool": "view_bev",
                    "selected_because": "default BEV view",
                },
            )
            suffix = ""
            if missing_cats:
                suffix = (
                    f"; categories with no matches: {missing_cats!r} "
                    "(check spelling or use list_scene_proposals to discover "
                    "what categories exist)"
                )
            text = (
                f"bev image at {catalog.bev_image_path}; "
                f"default view (mesh + trajectory + small dots, no text labels"
                f"{suffix})"
            )
            runtime.record("view_bev", request, text, image_metadata=[image_ref])
            return text
        cache_dir = Path(catalog.bev_image_path).parent / "highlights"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ids_token = "_".join(str(i) for i in resolved_ids)
        # Renderer version baked into the cache key — any future change to
        # ``_overlay_proposal_labels`` (font / colours / anchoring) should
        # bump _HIGHLIGHT_BEV_RENDERER_VERSION so stale overlays from older
        # renderers are not silently reused.
        out_path = cache_dir / (
            f"bev_h_{ids_token}_{_HIGHLIGHT_BEV_RENDERER_VERSION}.png"
        )
        if not out_path.exists():
            _render_highlighted_bev(catalog, resolved_ids, out_path)
        image_ref = make_tool_image_ref(
            runtime,
            str(out_path),
            metadata={
                "source_tool": "view_bev",
                "selected_because": f"highlight={resolved_ids}",
            },
        )
        parts = [f"bev image at {out_path}", f"highlight={resolved_ids}"]
        if categories:
            parts.append(f"resolved_from_categories={list(categories)!r}")
        if missing_cats:
            parts.append(f"categories_with_no_matches={missing_cats!r}")
        text = "; ".join(parts)
        runtime.record("view_bev", request, text, image_metadata=[image_ref])
        return text

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
            proposals = [
                p for p in proposals if _in_bev_box(p.position_3d, list(region_bev))
            ]
        if limit is not None and limit >= 0:
            proposals = proposals[:limit]
        rows = []
        for p in proposals:
            row = {
                "proposal_id": p.proposal_id,
                "category": p.category,
                "position_3d": list(p.position_3d),
                "frame_count": len(p.frame_views),
            }
            if p.enriched_category is not None:
                row["enriched_category"] = p.enriched_category
            if p.compact_note is not None:
                row["compact_note"] = p.compact_note
            rows.append(row)
        payload = {
            "count": len(proposals),
            "proposals": rows,
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
            "bbox_3d_9dof": (
                list(proposal.bbox_3d_9dof) if proposal.bbox_3d_9dof else None
            ),
            "frames_appeared": sorted(proposal.frame_views.keys()),
            "source": proposal.source,
        }
        if proposal.enriched_category is not None:
            payload["enriched_category"] = proposal.enriched_category
        if proposal.compact_note is not None:
            payload["compact_note"] = proposal.compact_note
        if proposal.enrichment is not None:
            payload["enrichment"] = proposal.enrichment
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("inspect_proposal", request, text)
        return text

    return [view_bev, list_scene_proposals, inspect_proposal]


__all__ = [
    "SCENE_EXPLORATION_SKILL",
    "build_scene_perception_tools",
]
