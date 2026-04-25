"""Backend for the select_object VG tool.

Looks up an object by ID from the scene graph, computes its precise
3D bounding box from point cloud data + axis_align_matrix, and stores
the result in the runtime state for extraction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger


def find_object_by_id(
    scene_objects: list[Any], object_id: int
) -> Any | None:
    """Find a scene object by its obj_id."""
    for obj in scene_objects:
        if getattr(obj, "obj_id", None) == object_id:
            return obj
    return None


def compute_bbox_3d(
    obj: Any,
    axis_align_matrix: np.ndarray | None = None,
) -> list[float]:
    """Compute precise 9-DOF bbox from an object's point cloud.

    Transforms all points to the aligned frame, then computes
    centroid and axis-aligned extent.

    FAIL-LOUD: raises ValueError if the object has no point cloud.
    Per the project's strict no-fallback rule, we do not substitute a
    default extent when geometry is missing.

    Returns:
        [cx, cy, cz, dx, dy, dz, 0, 0, 0]

    Raises:
        ValueError: if obj.pcd_np is None or empty.
    """
    pcd = getattr(obj, "pcd_np", None)
    if pcd is None or len(pcd) == 0:
        obj_id = getattr(obj, "obj_id", "?")
        raise ValueError(
            f"object {obj_id} has no pcd; cannot compute 9-DOF bbox"
        )

    pts = np.array(pcd, dtype=np.float64)
    if axis_align_matrix is not None:
        ones = np.ones((len(pts), 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])
        pts = (axis_align_matrix @ pts_h.T).T[:, :3]
    centroid = pts.mean(axis=0)
    extent = pts.max(axis=0) - pts.min(axis=0)

    return [
        float(centroid[0]), float(centroid[1]), float(centroid[2]),
        float(extent[0]), float(extent[1]), float(extent[2]),
        0.0, 0.0, 0.0,
    ]


def handle_select_object(
    runtime_state: Any,
    object_id: int,
    rationale: str,
) -> str:
    """Handle a select_object tool call.

    Looks up the object, computes bbox, stores in runtime state,
    and returns a confirmation string.
    """
    scene_objects = runtime_state.vg_scene_objects
    if scene_objects is None:
        return "ERROR: No scene objects available for selection."

    obj = find_object_by_id(scene_objects, object_id)
    if obj is None:
        available = [
            f"[ID={getattr(o, 'obj_id', '?')}] {getattr(o, 'category', '?')}"
            for o in scene_objects
            if getattr(o, "category", "") not in ("wall", "floor", "ceiling")
        ]
        return (
            f"ERROR: Object ID={object_id} not found in scene graph.\n"
            f"Available objects:\n" + "\n".join(available[:20])
        )

    try:
        bbox_3d = compute_bbox_3d(obj, runtime_state.vg_axis_align_matrix)
    except ValueError as exc:
        return f"ERROR: {exc}"
    category = getattr(obj, "category", "unknown")

    # Store in runtime state
    runtime_state.vg_selected_object_id = object_id
    runtime_state.vg_selected_bbox_3d = bbox_3d
    runtime_state.vg_selection_rationale = rationale

    logger.info(
        "[select_object] Selected obj_id={} ({}), bbox=({:.2f},{:.2f},{:.2f})",
        object_id, category, bbox_3d[0], bbox_3d[1], bbox_3d[2],
    )

    return (
        f"Object selected: [ID={object_id}] {category}\n"
        f"3D bbox (aligned): center=({bbox_3d[0]:.3f}, {bbox_3d[1]:.3f}, {bbox_3d[2]:.3f}), "
        f"extent=({bbox_3d[3]:.3f}, {bbox_3d[4]:.3f}, {bbox_3d[5]:.3f})\n"
        f"Rationale: {rationale}\n"
        f"bbox_3d has been auto-filled in your output payload."
    )
