"""EmbodiedScan 3D visual grounding evaluation.

Computes 3D IoU for 9-DOF oriented bounding boxes using the ZXY Euler
angle convention, and evaluates VG predictions with Acc@0.25 / Acc@0.50.

EmbodiedScan bbox format: [cx, cy, cz, dx, dy, dz, alpha, beta, gamma]
  - (cx, cy, cz): box center
  - (dx, dy, dz): box dimensions (size along each axis)
  - (alpha, beta, gamma): ZXY Euler angles (radians)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from scipy.spatial import ConvexHull

from .embodiedscan_loader import EmbodiedScanVGSample

# Standard 8-corner offsets for a unit box centered at origin.
# Each row is (sx, sy, sz) where s ∈ {-0.5, +0.5}.
_UNIT_CORNERS = np.array(
    [
        [-0.5, -0.5, -0.5],
        [+0.5, -0.5, -0.5],
        [+0.5, +0.5, -0.5],
        [-0.5, +0.5, -0.5],
        [-0.5, -0.5, +0.5],
        [+0.5, -0.5, +0.5],
        [+0.5, +0.5, +0.5],
        [-0.5, +0.5, +0.5],
    ],
    dtype=np.float64,
)

# 12 edges of a box, each as (start_corner_idx, end_corner_idx).
_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
]

# 6 faces of a box, each as 4 corner indices (counter-clockwise outward).
_BOX_FACES = [
    (0, 3, 2, 1),  # bottom  (z = -0.5)
    (4, 5, 6, 7),  # top     (z = +0.5)
    (0, 1, 5, 4),  # front   (y = -0.5)
    (2, 3, 7, 6),  # back    (y = +0.5)
    (0, 4, 7, 3),  # left    (x = -0.5)
    (1, 2, 6, 5),  # right   (x = +0.5)
]


def euler_to_rotation_matrix(
    alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Convert ZXY Euler angles to a 3x3 rotation matrix.

    Follows the same convention as EmbodiedScan / pytorch3d:
    R = Rz(alpha) @ Rx(beta) @ Ry(gamma)

    Args:
        alpha: Rotation around Z axis (radians).
        beta: Rotation around X axis (radians).
        gamma: Rotation around Y axis (radians).

    Returns:
        3x3 rotation matrix.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    # Rz @ Rx @ Ry
    R = np.array(
        [
            [
                ca * cg - sa * sb * sg,
                -sa * cb,
                ca * sg + sa * sb * cg,
            ],
            [
                sa * cg + ca * sb * sg,
                ca * cb,
                sa * sg - ca * sb * cg,
            ],
            [
                -cb * sg,
                sb,
                cb * cg,
            ],
        ],
        dtype=np.float64,
    )
    return R


def oriented_bbox_to_corners(bbox_9dof: list[float]) -> np.ndarray:
    """Convert a 9-DOF oriented bounding box to 8 corner coordinates.

    Args:
        bbox_9dof: [cx, cy, cz, dx, dy, dz, alpha, beta, gamma].

    Returns:
        (8, 3) array of corner coordinates.
    """
    cx, cy, cz = bbox_9dof[0], bbox_9dof[1], bbox_9dof[2]
    dx, dy, dz = bbox_9dof[3], bbox_9dof[4], bbox_9dof[5]
    alpha, beta, gamma = bbox_9dof[6], bbox_9dof[7], bbox_9dof[8]

    # Scale unit corners by dimensions
    corners = _UNIT_CORNERS * np.array([dx, dy, dz])

    # Rotate
    R = euler_to_rotation_matrix(alpha, beta, gamma)
    corners = (R @ corners.T).T

    # Translate
    corners += np.array([cx, cy, cz])

    return corners


def compute_oriented_iou_3d(
    bbox1: list[float], bbox2: list[float]
) -> float:
    """Compute 3D IoU between two 9-DOF oriented bounding boxes.

    Uses vertex enumeration: collects all vertices of the intersection
    polytope (corners of each box inside the other + edge-face
    intersection points), then computes the convex hull volume.

    Args:
        bbox1: First box [cx,cy,cz,dx,dy,dz,alpha,beta,gamma].
        bbox2: Second box [cx,cy,cz,dx,dy,dz,alpha,beta,gamma].

    Returns:
        IoU value in [0, 1].
    """
    corners1 = oriented_bbox_to_corners(bbox1)
    corners2 = oriented_bbox_to_corners(bbox2)

    vol1 = abs(bbox1[3] * bbox1[4] * bbox1[5])
    vol2 = abs(bbox2[3] * bbox2[4] * bbox2[5])

    if vol1 < 1e-12 or vol2 < 1e-12:
        return 0.0

    intersection_pts = _find_intersection_vertices(corners1, corners2)

    if len(intersection_pts) < 4:
        return 0.0

    inter_vol = _convex_hull_volume(intersection_pts)
    if inter_vol < 1e-12:
        return 0.0

    union = vol1 + vol2 - inter_vol
    return float(inter_vol / max(union, 1e-12))


def evaluate_vg_predictions(
    predictions: list[dict[str, Any]],
    samples: list[EmbodiedScanVGSample],
) -> dict[str, Any]:
    """Evaluate VG predictions against ground truth.

    Args:
        predictions: List of {"sample_id": str, "bbox_3d": [9 floats] | None}.
        samples: Corresponding ground truth samples.

    Returns:
        Dictionary with keys: acc_025, acc_050, mean_iou,
        per_category, num_samples.
    """
    if not samples:
        return {
            "acc_025": 0.0,
            "acc_050": 0.0,
            "mean_iou": 0.0,
            "per_category": {},
            "num_samples": 0,
        }

    # Index predictions by sample_id
    pred_map = {p["sample_id"]: p for p in predictions}

    ious: list[float] = []
    cat_ious: dict[str, list[float]] = {}

    for sample in samples:
        pred = pred_map.get(sample.sample_id)
        pred_bbox = pred.get("bbox_3d") if pred else None
        gt_bbox = sample.gt_bbox_3d

        if pred_bbox is None or gt_bbox is None:
            iou = 0.0
        else:
            try:
                # Guard against malformed bbox values that slipped past parsing
                pred_bbox = [float(v) for v in pred_bbox[:9]]
                gt_bbox = [float(v) for v in gt_bbox[:9]]
                if len(pred_bbox) < 6 or len(gt_bbox) < 6:
                    iou = 0.0
                else:
                    # Pad missing euler angles
                    while len(pred_bbox) < 9:
                        pred_bbox.append(0.0)
                    while len(gt_bbox) < 9:
                        gt_bbox.append(0.0)
                    iou = compute_oriented_iou_3d(pred_bbox, gt_bbox)
            except (TypeError, ValueError, IndexError):
                logger.warning(
                    "Invalid bbox for {}: pred={}, gt={}",
                    sample.sample_id, pred_bbox, gt_bbox,
                )
                iou = 0.0

        ious.append(iou)
        cat_ious.setdefault(sample.target, []).append(iou)

    ious_arr = np.array(ious)
    acc_025 = float((ious_arr >= 0.25).mean())
    acc_050 = float((ious_arr >= 0.50).mean())
    mean_iou = float(ious_arr.mean())

    # Per-category breakdown
    per_category: dict[str, dict[str, float]] = {}
    for cat, cat_iou_list in sorted(cat_ious.items()):
        arr = np.array(cat_iou_list)
        per_category[cat] = {
            "acc_025": float((arr >= 0.25).mean()),
            "acc_050": float((arr >= 0.50).mean()),
            "mean_iou": float(arr.mean()),
            "count": len(cat_iou_list),
        }

    return {
        "acc_025": acc_025,
        "acc_050": acc_050,
        "mean_iou": mean_iou,
        "per_category": per_category,
        "num_samples": len(samples),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_intersection_vertices(
    corners1: np.ndarray, corners2: np.ndarray
) -> np.ndarray:
    """Find all vertices of the intersection polytope of two oriented boxes.

    Vertices come from three sources:
    1. Corners of box1 that lie inside box2
    2. Corners of box2 that lie inside box1
    3. Points where edges of one box intersect faces of the other
    """
    faces1 = _box_face_planes(corners1)
    faces2 = _box_face_planes(corners2)

    pts: list[np.ndarray] = []

    # Corners of box1 inside box2
    for c in corners1:
        if _point_inside_halfspaces(c, faces2):
            pts.append(c)

    # Corners of box2 inside box1
    for c in corners2:
        if _point_inside_halfspaces(c, faces1):
            pts.append(c)

    # Edge-face intersections: edges of box1 vs faces of box2
    for i, j in _BOX_EDGES:
        for normal, offset in faces2:
            pt = _edge_plane_intersection(
                corners1[i], corners1[j], normal, offset
            )
            if pt is not None and _point_inside_halfspaces(pt, faces2):
                pts.append(pt)

    # Edge-face intersections: edges of box2 vs faces of box1
    for i, j in _BOX_EDGES:
        for normal, offset in faces1:
            pt = _edge_plane_intersection(
                corners2[i], corners2[j], normal, offset
            )
            if pt is not None and _point_inside_halfspaces(pt, faces1):
                pts.append(pt)

    if len(pts) == 0:
        return np.empty((0, 3))

    return np.array(pts)


def _box_face_planes(
    corners: np.ndarray,
) -> list[tuple[np.ndarray, float]]:
    """Compute outward-facing half-space planes for a box's 6 faces.

    Returns list of (normal, offset) where normal points outward and
    a point p is inside the box iff normal @ p <= offset for all planes.
    """
    center = corners.mean(axis=0)
    planes: list[tuple[np.ndarray, float]] = []

    for face_indices in _BOX_FACES:
        p0 = corners[face_indices[0]]
        p1 = corners[face_indices[1]]
        p2 = corners[face_indices[2]]

        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        offset = np.dot(normal, p0)

        # Ensure normal points outward (away from center).
        # Center must satisfy n @ center <= offset if normal is outward.
        if np.dot(normal, center) > offset:
            normal = -normal
            offset = -offset

        planes.append((normal, offset))

    return planes


def _point_inside_halfspaces(
    point: np.ndarray,
    planes: list[tuple[np.ndarray, float]],
    eps: float = 1e-6,
) -> bool:
    """Check if a point lies inside all half-spaces (n @ p <= d + eps)."""
    for normal, offset in planes:
        if np.dot(normal, point) > offset + eps:
            return False
    return True


def _edge_plane_intersection(
    p0: np.ndarray,
    p1: np.ndarray,
    normal: np.ndarray,
    offset: float,
    eps: float = 1e-8,
) -> np.ndarray | None:
    """Find intersection of line segment [p0, p1] with plane n @ p = d.

    Returns the intersection point if it lies strictly on the segment,
    or None otherwise.
    """
    d0 = np.dot(normal, p0) - offset
    d1 = np.dot(normal, p1) - offset

    # Both on same side → no intersection
    if d0 * d1 > eps:
        return None

    denom = d0 - d1
    if abs(denom) < eps:
        return None

    t = d0 / denom
    if t < -eps or t > 1.0 + eps:
        return None

    return p0 + t * (p1 - p0)


def _convex_hull_volume(points: np.ndarray) -> float:
    """Compute volume of the convex hull of a point set.

    Returns 0 for degenerate cases (collinear/coplanar points).
    """
    if len(points) < 4:
        return 0.0

    # Check if points are nearly coplanar
    centered = points - points.mean(axis=0)
    if np.linalg.matrix_rank(centered, tol=1e-8) < 3:
        return 0.0

    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0
