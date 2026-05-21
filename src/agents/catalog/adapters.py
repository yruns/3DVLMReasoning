"""Adapters that build a SceneCatalog from benchmark-specific assets."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
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
                enriched_category=raw.get("enriched_category"),
                compact_note=raw.get("compact_note"),
                enrichment=raw.get("enrichment"),
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


def _load_conceptgraph_objects(pcd_saves_dir: Path) -> list[dict]:
    candidates = sorted(pcd_saves_dir.glob("full_pcd*.pkl.gz"))
    if not candidates:
        raise FileNotFoundError(
            f"no full_pcd*.pkl.gz under {pcd_saves_dir}; cannot build QA catalog"
        )
    with gzip.open(candidates[-1], "rb") as fh:
        payload = pickle.load(fh)
    objects = payload.get("objects") if isinstance(payload, dict) else None
    if not isinstance(objects, list):
        raise ValueError(
            f"conceptgraph payload at {candidates[-1]} missing 'objects' list"
        )
    return objects


def _invert_view_to_objects(
    view_to_objects: dict[int, list[tuple[int, float]]],
) -> dict[int, list[tuple[int, float]]]:
    """Return object_id -> [(frame_id, weight), ...]."""
    out: dict[int, list[tuple[int, float]]] = {}
    for frame_id, entries in view_to_objects.items():
        for obj_id, weight in entries:
            out.setdefault(int(obj_id), []).append((int(frame_id), float(weight)))
    return out


def from_conceptgraph_objects(
    *,
    pcd_saves_dir: Path,
    detections_dir: Path,
    view_to_objects: dict[int, list[tuple[int, float]]],
    scene_id: str,
    bev_image_path: str,
    scene_category: str | None,
    valid_frame_ids: list[int],
    raw_rgb_template: str,
) -> SceneCatalog:
    """Build a SceneCatalog from ConceptGraph 3D-object segmentation output (QA path)."""
    if not valid_frame_ids:
        raise ValueError("from_conceptgraph_objects: valid_frame_ids must be non-empty")
    del detections_dir  # accepted for API parity / future enrichment
    raw_objects = _load_conceptgraph_objects(Path(pcd_saves_dir))
    obj_to_frames = _invert_view_to_objects(view_to_objects)

    proposals: list[SceneProposal] = []
    for raw in raw_objects:
        obj_id = int(raw["id"])
        frames = obj_to_frames.get(obj_id) or []
        if not frames:
            continue
        bbox = raw.get("bbox_3d_9dof") or raw.get("bbox_3d")
        if bbox is None or len(bbox) != 9:
            raise ValueError(
                f"conceptgraph object id={obj_id} bbox_3d_9dof must have 9 elements"
            )
        bbox9 = tuple(float(v) for v in bbox)
        frame_views: dict[int, FrameView] = {}
        for frame_id, weight in frames:
            frame_views[frame_id] = FrameView(
                frame_id=frame_id,
                bbox_2d=(0, 0, 0, 0),  # QA pack does not require per-view 2D bboxes
                raw_rgb_path=raw_rgb_template.format(frame_id=frame_id),
                visibility_weight=weight,
            )
        proposals.append(
            SceneProposal(
                proposal_id=obj_id,
                category=str(raw.get("category") or raw.get("label") or "object"),
                position_3d=(bbox9[0], bbox9[1], bbox9[2]),
                bbox_3d_9dof=bbox9,
                frame_views=frame_views,
                source="conceptgraph",
            )
        )

    sorted_frames = sorted(int(fid) for fid in valid_frame_ids)
    return SceneCatalog(
        scene_id=scene_id,
        scene_category=scene_category,
        proposals=proposals,
        total_frames=len(sorted_frames),
        frame_id_range=(sorted_frames[0], sorted_frames[-1]),
        valid_frame_ids=sorted_frames,
        bev_image_path=bev_image_path,
        axis_align_matrix=None,
    )


def from_gt_embodiedscan(
    *,
    es_annotations: dict[str, Any],
    scene_id: str,
    bev_image_path: str,
    valid_frame_ids: list[int],
    raw_rgb_template: str,
) -> SceneCatalog:
    """Build a SceneCatalog from EmbodiedScan GT annotations (oracle pack)."""
    if not valid_frame_ids:
        raise ValueError("from_gt_embodiedscan: valid_frame_ids must be non-empty")
    instances = es_annotations.get("instances") or []
    obj_to_frames = _invert_view_to_objects(es_annotations.get("view_to_objects") or {})
    proposals: list[SceneProposal] = []
    for inst in instances:
        obj_id = int(inst.get("bbox_id") or inst["id"])
        bbox = inst.get("bbox_3d_9dof") or inst.get("bbox_3d")
        if bbox is None or len(bbox) != 9:
            raise ValueError(
                f"GT instance bbox_id={obj_id} bbox_3d_9dof must have 9 elements"
            )
        bbox9 = tuple(float(v) for v in bbox)
        frame_views: dict[int, FrameView] = {}
        for frame_id, weight in obj_to_frames.get(obj_id, []):
            frame_views[frame_id] = FrameView(
                frame_id=frame_id,
                bbox_2d=(0, 0, 0, 0),
                raw_rgb_path=raw_rgb_template.format(frame_id=frame_id),
                visibility_weight=weight,
            )
        proposals.append(
            SceneProposal(
                proposal_id=obj_id,
                category=str(inst.get("category") or inst.get("label") or "object"),
                position_3d=(bbox9[0], bbox9[1], bbox9[2]),
                bbox_3d_9dof=bbox9,
                frame_views=frame_views,
                source="gt",
            )
        )
    sorted_frames = sorted(int(fid) for fid in valid_frame_ids)
    return SceneCatalog(
        scene_id=scene_id,
        proposals=proposals,
        total_frames=len(sorted_frames),
        frame_id_range=(sorted_frames[0], sorted_frames[-1]),
        valid_frame_ids=sorted_frames,
        bev_image_path=bev_image_path,
    )


__all__ = [
    "from_vg_proposal_pool",
    "from_conceptgraph_objects",
    "from_gt_embodiedscan",
]
