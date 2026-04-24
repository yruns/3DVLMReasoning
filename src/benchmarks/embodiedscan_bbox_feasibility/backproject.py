from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import (
    aabb_from_points,
    backproject_depth,
    is_non_degenerate_bbox,
    transform_points,
)
from .models import BBox3DProposal


def proposal_from_depth_mask(
    *,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    source: str,
    score: float | None = None,
    min_points: int = 5,
    metadata: dict[str, Any] | None = None,
) -> BBox3DProposal | None:
    cam_points = backproject_depth(depth, intrinsic, mask=mask)
    if len(cam_points) < min_points:
        return None

    world_points = transform_points(cam_points, camera_to_world)
    bbox = aabb_from_points(world_points)
    if not is_non_degenerate_bbox(bbox):
        return None

    return BBox3DProposal(
        bbox_3d=bbox,
        score=score,
        source=source,
        metadata={**(metadata or {}), "num_points": int(len(world_points))},
    )
