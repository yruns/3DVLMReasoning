"""Backend for the spatial_compare VG tool.

Ranks scene objects of a target category by 3D distance to objects
of an anchor category, supporting closest_to and farthest_from queries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger


def _get_aligned_centroid(
    obj: Any,
    axis_align_matrix: np.ndarray | None,
) -> np.ndarray:
    """Get object centroid in the aligned coordinate frame."""
    centroid = np.array(
        getattr(obj, "centroid", [0, 0, 0]), dtype=np.float64
    )
    if axis_align_matrix is not None:
        ctr_h = np.append(centroid, 1.0)
        centroid = (axis_align_matrix @ ctr_h)[:3]
    return centroid


def _category_matches(obj_category: str, query: str) -> bool:
    """Case-insensitive substring category match."""
    a = obj_category.lower()
    b = query.lower()
    return a == b or b in a or a in b


def find_objects_by_category(
    scene_objects: list[Any], category: str
) -> list[Any]:
    """Find all objects matching a category (case-insensitive, substring)."""
    return [
        obj for obj in scene_objects
        if _category_matches(getattr(obj, "category", ""), category)
        and getattr(obj, "category", "") not in ("wall", "floor", "ceiling")
    ]


def handle_spatial_compare(
    runtime_state: Any,
    target_category: str,
    relation: str,
    anchor_category: str,
) -> str:
    """Handle a spatial_compare tool call.

    Finds target and anchor objects, computes distances, returns
    a ranked list.
    """
    scene_objects = runtime_state.vg_scene_objects
    if scene_objects is None:
        return "ERROR: No scene objects available."

    axis_align = runtime_state.vg_axis_align_matrix

    targets = find_objects_by_category(scene_objects, target_category)
    anchors = find_objects_by_category(scene_objects, anchor_category)

    if not targets:
        all_cats = sorted({
            getattr(o, "category", "?") for o in scene_objects
            if getattr(o, "category", "") not in ("wall", "floor", "ceiling", "")
        })
        return (
            f"ERROR: No objects matching '{target_category}' found.\n"
            f"Available categories: {', '.join(all_cats)}"
        )

    if not anchors:
        all_cats = sorted({
            getattr(o, "category", "?") for o in scene_objects
            if getattr(o, "category", "") not in ("wall", "floor", "ceiling", "")
        })
        return (
            f"ERROR: No anchor objects matching '{anchor_category}' found.\n"
            f"Available categories: {', '.join(all_cats)}"
        )

    # Compute anchor centroids
    anchor_centroids = [_get_aligned_centroid(a, axis_align) for a in anchors]

    # For each target, compute min distance to any anchor
    ranked: list[tuple[Any, float, Any]] = []
    for t in targets:
        t_ctr = _get_aligned_centroid(t, axis_align)
        dists = [np.linalg.norm(t_ctr - ac) for ac in anchor_centroids]
        min_idx = int(np.argmin(dists))
        ranked.append((t, float(dists[min_idx]), anchors[min_idx]))

    # Sort
    reverse = relation == "farthest_from"
    ranked.sort(key=lambda x: x[1], reverse=reverse)

    # Format response
    label = "FARTHEST" if reverse else "CLOSEST"
    lines = [
        f"Found {len(targets)} '{target_category}' object(s), "
        f"{len(anchors)} '{anchor_category}' anchor(s).",
        f"Ranked by {relation.replace('_', ' ')} {anchor_category}:",
    ]
    for i, (t, dist, anchor) in enumerate(ranked):
        t_id = getattr(t, "obj_id", "?")
        t_cat = getattr(t, "category", "?")
        a_id = getattr(anchor, "obj_id", "?")
        marker = f" <- {label}" if i == 0 else ""
        lines.append(
            f"  {i + 1}. [ID={t_id}] {t_cat} "
            f"({dist:.2f}m from {anchor_category} [ID={a_id}]){marker}"
        )

    logger.info(
        "[spatial_compare] {} {} {} → {} results",
        target_category, relation, anchor_category, len(ranked),
    )

    return "\n".join(lines)
