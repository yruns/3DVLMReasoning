"""Adapters that build a SceneCatalog from benchmark-specific assets."""

from __future__ import annotations

from typing import Any, Literal

from agents.catalog.models import FrameView, SceneCatalog, SceneProposal

VgPoolSource = Literal["mask3d", "vdetr", "gt", "conceptgraph"]


def _build_frame_view(raw: dict) -> FrameView:
    return FrameView(
        frame_id=int(raw["frame_id"]),
        bbox_2d=tuple(int(v) for v in raw["bbox_2d"]),  # type: ignore[arg-type]
        raw_rgb_path=str(raw["raw_rgb_path"]),
        visibility_weight=(
            float(raw["visibility_weight"])
            if raw.get("visibility_weight") is not None
            else None
        ),
    )


def _frame_views_from_raw(raw: Any) -> dict[int, FrameView]:
    if raw is None:
        return {}
    out: dict[int, FrameView] = {}
    if isinstance(raw, dict):
        for fid, value in raw.items():
            item = dict(value)
            item.setdefault("frame_id", int(fid))
            view = _build_frame_view(item)
            out[view.frame_id] = view
        return out
    if isinstance(raw, list):
        for value in raw:
            view = _build_frame_view(dict(value))
            out[view.frame_id] = view
        return out
    raise ValueError(f"frame_views must be dict or list, got {type(raw).__name__}")


def from_vg_proposal_pool(
    *,
    pool: dict[str, Any],
    scene_id: str,
    bev_image_path: str,
    scene_category: str | None,
    axis_align_matrix: list[list[float]] | None,
    valid_frame_ids: list[int],
) -> SceneCatalog:
    """Convert a VG proposal pool (NR3D / ScanRefer / EmbodiedScan) into SceneCatalog."""
    if not valid_frame_ids:
        raise ValueError("from_vg_proposal_pool: valid_frame_ids must be non-empty")
    source: VgPoolSource = pool.get("source", "mask3d")
    if source not in ("mask3d", "vdetr", "gt", "conceptgraph"):
        raise ValueError(f"from_vg_proposal_pool: unsupported source {source!r}")

    proposals: list[SceneProposal] = []
    for raw in pool.get("proposals", []) or []:
        bbox = raw.get("bbox_3d_9dof") or raw.get("bbox_3d")
        if bbox is None or len(bbox) != 9:
            raise ValueError(
                f"proposal id={raw.get('id')} bbox_3d_9dof must have 9 elements"
            )
        bbox9 = tuple(float(v) for v in bbox)
        position = (float(bbox9[0]), float(bbox9[1]), float(bbox9[2]))
        proposals.append(
            SceneProposal(
                proposal_id=int(raw["id"]),
                category=str(raw.get("category") or raw.get("label") or ""),
                position_3d=position,
                bbox_3d_9dof=bbox9,
                frame_views=_frame_views_from_raw(raw.get("frame_views")),
                source=source,
            )
        )

    sorted_frames = sorted(int(fid) for fid in valid_frame_ids)
    catalog = SceneCatalog(
        scene_id=scene_id,
        scene_category=scene_category,
        proposals=proposals,
        total_frames=len(sorted_frames),
        frame_id_range=(sorted_frames[0], sorted_frames[-1]),
        valid_frame_ids=sorted_frames,
        bev_image_path=bev_image_path,
        axis_align_matrix=axis_align_matrix,
    )
    return catalog


__all__ = ["from_vg_proposal_pool"]
