"""Viewer-frame geometry for viewpoint-aware spatial constraints.

This module implements the viewer-pose resolution and viewer-frame
directional relation checks that the executor consults when a constraint
sets ``reference_frame=ReferenceFrame.VIEWER``.

Design:

- ``resolve_viewer_pose`` takes a list of anchor objects (already resolved
  by the executor's category lookup) and optionally a camera trajectory
  (frame poses) plus a visibility index. It returns a viewer position and
  forward direction.

  - Trajectory mode (preferred): pick the camera frame with maximum
    visibility coverage of the anchor objects, return that frame's
    world position and forward axis.
  - Geometric mode (fallback): use the room centroid as a proxy for the
    speaker's position, anchored 1.5 m back from the anchor centroid.

- ``viewer_axes`` constructs an orthonormal (forward, right, up) basis
  with +Z up and right-handed cross product. ``right = forward x up`` so
  that for forward = +X, right = -Y (typical ScanNet axis-aligned scenes).

- ``viewer_frame_relation_score`` returns a continuous satisfaction score
  in [0, 1] for the four directional relations, evaluated in the viewer
  frame relative to a single anchor centroid. Score 0 means the candidate
  is on the wrong side; positive scores grow with the lateral / depth
  projection toward the named side.

Strict no-fallback: callers are responsible for deciding what to do when
``resolve_viewer_pose`` returns ``None``. The executor downgrades to the
SOFT / RANK_ONLY behavior already defined by the constraint's
``execution_policy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class ViewerPose:
    """A resolved viewer pose with orthonormal viewer-frame axes.

    All vectors live in world coordinates. ``forward`` is the speaker's
    facing direction (unit), ``right`` is the speaker's right-hand axis
    (unit), ``up`` is +Z by convention.
    """

    position: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    source: str  # "trajectory" or "geometric"


class _HasCentroid(Protocol):
    centroid: np.ndarray
    obj_id: int


def _to_xyz(value: object) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        return None
    return arr


def viewer_axes(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (forward, right, up) orthonormal basis with +Z up.

    Returns the input forward (renormalised), the right axis, and the up
    axis (+Z). If forward is degenerate (zero or vertical), raises
    ``ValueError`` - the caller is expected to fall back to a non-geometric
    execution policy in that case.
    """
    f = np.asarray(forward, dtype=np.float64)
    # Project to horizontal plane; viewer-frame directionals are about
    # left/right/front/behind and never care about pitch.
    f = np.array([f[0], f[1], 0.0], dtype=np.float64)
    n = float(np.linalg.norm(f))
    if n < 1e-6:
        raise ValueError("viewer forward vector is degenerate after horizontal projection")
    f = f / n
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(f, up)
    rn = float(np.linalg.norm(right))
    if rn < 1e-6:
        raise ValueError("viewer right vector is degenerate")
    right = right / rn
    return f, right, up


def _camera_visibility_score(
    cam_world_pose: np.ndarray,
    anchor_centroids: np.ndarray,
) -> float:
    """Heuristic: prefer frames whose forward axis points at the anchors.

    ``cam_world_pose`` is a 4x4 world_T_cam matrix (camera-to-world).
    Camera convention: -Z is forward, +Y is up. We compare the camera's
    forward direction to (mean_anchor_centroid - camera_position).

    The score is the mean cosine between the camera forward and each
    anchor direction. No depth check - we assume the visibility index has
    already pruned occluded frames if it was used to build the trajectory.
    """
    cam_pos = cam_world_pose[:3, 3]
    # Camera looks down -Z in its own frame; transform to world.
    cam_forward = -cam_world_pose[:3, 2]
    cam_forward = cam_forward / max(1e-6, float(np.linalg.norm(cam_forward)))

    cosines = []
    for centroid in anchor_centroids:
        delta = centroid - cam_pos
        n = float(np.linalg.norm(delta))
        if n < 1e-3:
            continue
        cosines.append(float(np.dot(delta / n, cam_forward)))
    if not cosines:
        return float("-inf")
    return float(np.mean(cosines))


def resolve_viewer_pose(
    anchor_objects: list[_HasCentroid],
    *,
    all_object_centroids: np.ndarray | None = None,
    camera_poses: list[np.ndarray] | None = None,
    standoff: float = 1.5,
    allow_geometric_fallback: bool = True,
) -> ViewerPose | None:
    """Resolve a viewer pose from a set of facing-anchor objects.

    Strategy:
    1. Trajectory mode (when ``camera_poses`` is non-empty): pick the frame
       whose forward axis best aligns with the mean anchor direction.
       Returns that frame's pose with ``source="trajectory"``.
    2. Geometric fallback (only when ``allow_geometric_fallback=True``):
       viewer is placed ``standoff`` metres back from the anchor centroid
       along the line from the room centroid to the anchor centroid.
       Returns a pose with ``source="geometric"``.

    Returns None when:
    - No anchor objects / centroids,
    - Trajectory mode fails (no 4x4 frame meets the alignment threshold)
      AND ``allow_geometric_fallback=False``,
    - The geometric fallback is allowed but degenerates (anchor at room
      centroid, vertical forward, etc.).

    The ``allow_geometric_fallback`` flag is a strict-no-fallback knob.
    The ``QueryExecutor`` MUST pass ``allow_geometric_fallback=False`` so a
    bad / empty / degenerate trajectory cannot silently fabricate a viewer
    pose. Pure unit tests that exercise geometric mode opt in with True.
    """
    if not anchor_objects:
        return None

    centroids = []
    for obj in anchor_objects:
        c = _to_xyz(getattr(obj, "centroid", None))
        if c is not None:
            centroids.append(c)
    if not centroids:
        return None
    anchor_centroids = np.stack(centroids)
    mean_anchor = anchor_centroids.mean(axis=0)

    if camera_poses:
        best_pose: np.ndarray | None = None
        best_score = float("-inf")
        for pose in camera_poses:
            pose = np.asarray(pose, dtype=np.float64)
            if pose.shape != (4, 4):
                continue
            score = _camera_visibility_score(pose, anchor_centroids)
            if score > best_score:
                best_score = score
                best_pose = pose
        if best_pose is not None and best_score > -1.0:
            position = best_pose[:3, 3]
            forward = -best_pose[:3, 2]
            try:
                f, right, up = viewer_axes(forward)
            except ValueError:
                pass
            else:
                return ViewerPose(
                    position=position.astype(np.float64),
                    forward=f,
                    right=right,
                    up=up,
                    source="trajectory",
                )

    # Trajectory mode failed (no poses, all degenerate, or no frame faces
    # the anchor). Strict no-fallback path: refuse to synthesize a pose.
    if not allow_geometric_fallback:
        return None

    if all_object_centroids is None or len(all_object_centroids) == 0:
        room_center = anchor_centroids.mean(axis=0)
    else:
        room_center = np.asarray(all_object_centroids, dtype=np.float64).mean(axis=0)

    forward_world = mean_anchor - room_center
    forward_world[2] = 0.0
    if float(np.linalg.norm(forward_world)) < 1e-3:
        return None
    try:
        f, right, up = viewer_axes(forward_world)
    except ValueError:
        return None

    position = mean_anchor - standoff * f
    return ViewerPose(
        position=position.astype(np.float64),
        forward=f,
        right=right,
        up=up,
        source="geometric",
    )


def viewer_frame_relation_score(
    relation: str,
    candidate_centroid: np.ndarray,
    anchor_centroid: np.ndarray,
    pose: ViewerPose,
) -> float:
    """Continuous satisfaction score in [0, 1] for a directional relation.

    The relation is interpreted as "is candidate on the named side of
    anchor, as seen by the viewer?". We project ``candidate - anchor``
    into the viewer's (right, forward) basis and score by the signed
    projection along the named axis, normalised by a soft denominator
    that also rewards candidates that are not too far behind the anchor
    relative to the viewer.
    """
    delta = np.asarray(candidate_centroid, dtype=np.float64) - np.asarray(
        anchor_centroid, dtype=np.float64
    )
    right_proj = float(np.dot(delta, pose.right))
    forward_proj = float(np.dot(delta, pose.forward))

    rel = relation.lower().replace(" ", "_")
    sign = 0
    axis_proj = 0.0
    off_proj = 0.0
    if rel == "right_of":
        sign = 1
        axis_proj = right_proj
        off_proj = forward_proj
    elif rel == "left_of":
        sign = -1
        axis_proj = right_proj
        off_proj = forward_proj
    elif rel == "in_front_of":
        # candidate sits between viewer and anchor -> closer to viewer than anchor
        sign = -1
        axis_proj = forward_proj
        off_proj = right_proj
    elif rel == "behind":
        sign = 1
        axis_proj = forward_proj
        off_proj = right_proj
    else:
        return 0.0

    signed = sign * axis_proj
    if signed <= 0:
        return 0.0

    # Reward axis alignment, soft penalty on off-axis distance.
    score = signed / (signed + abs(off_proj) + 0.5)
    if not np.isfinite(score):
        return 0.0
    return float(np.clip(score, 0.0, 1.0))


def viewer_frame_axis_value(
    centroid: np.ndarray,
    pose: ViewerPose,
    axis: str,
) -> float:
    """Project a centroid onto a viewer-frame axis for SelectConstraint sorting.

    ``axis`` is ``"right"`` (positive = right of viewer) or ``"forward"``
    (positive = ahead of viewer). Used by SelectConstraint with metric
    ``x_position`` when ``reference_frame=VIEWER``.
    """
    c = np.asarray(centroid, dtype=np.float64)
    if axis == "right":
        return float(np.dot(c, pose.right))
    if axis == "forward":
        return float(np.dot(c, pose.forward))
    raise ValueError(f"unknown viewer axis {axis!r}; expected 'right' or 'forward'")
