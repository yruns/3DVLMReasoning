"""
Spatial Relation Checker.

This module implements geometric checks for common spatial relations
between 3D objects in a scene. It supports approximately 15 core relations
that cover the vast majority of natural language spatial queries.

Supported Relations:
- Vertical: on_top_of, above, below
- Horizontal: next_to, near, beside
- Directional: in_front_of, behind, left_of, right_of
- Containment: inside
- Multi-object: between

Usage:
    checker = SpatialRelationChecker()
    satisfies, score = checker.check(target_obj, anchor_obj, "on")
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .keyframe_selector import SceneObject


# Relation aliases mapping natural language variants to canonical names
RELATION_ALIASES: dict[str, str] = {
    # on_top_of variants
    "on": "on_top_of",
    "upon": "on_top_of",
    "atop": "on_top_of",
    "on_top": "on_top_of",
    "on_top_of": "on_top_of",
    "resting_on": "on_top_of",
    # above variants
    "above": "above",
    "over": "above",
    "higher_than": "above",
    # below variants
    "below": "below",
    "under": "below",
    "beneath": "below",
    "underneath": "below",
    "lower_than": "below",
    # next_to / near variants
    "next_to": "next_to",
    "beside": "next_to",
    "near": "near",
    "by": "near",
    "close_to": "near",
    "adjacent_to": "next_to",
    "next": "next_to",
    # in_front_of variants
    "in_front_of": "in_front_of",
    "front": "in_front_of",
    "facing": "in_front_of",
    "before": "in_front_of",
    # behind variants
    "behind": "behind",
    "back": "behind",
    "back_of": "behind",
    "in_back_of": "behind",
    # left_of variants
    "left_of": "left_of",
    "left": "left_of",
    "to_the_left_of": "left_of",
    "on_the_left_of": "left_of",
    # right_of variants
    "right_of": "right_of",
    "right": "right_of",
    "to_the_right_of": "right_of",
    "on_the_right_of": "right_of",
    # inside variants
    "inside": "inside",
    "in": "inside",
    "within": "inside",
    "contained_in": "inside",
    # between
    "between": "between",
    "in_between": "between",
    # leaning/against
    "against": "against",
    "leaning_on": "against",
    "leaning_against": "against",
    # around
    "around": "around",
    "surrounding": "around",
}


@dataclass
class RelationResult:
    """Result of a spatial relation check."""

    satisfies: bool
    score: float  # 0.0 to 1.0, higher means stronger relation
    details: dict[str, Any] | None = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class SpatialRelationChecker:
    """
    Checks spatial relations between 3D objects.

    Supports approximately 15 core spatial relations that cover
    the vast majority of natural language spatial queries.

    Attributes:
        default_thresholds: Default distance/angle thresholds for each relation
    """

    # Default thresholds (in meters)
    # Note: Adjusted thresholds for better handling of typical indoor scenes
    DEFAULT_THRESHOLDS = {
        "on_top_of": {"max_horizontal": 0.5, "min_vertical": 0.0, "max_vertical": 1.0},
        "above": {"max_horizontal": 1.0, "min_vertical": 0.1},
        "below": {"max_horizontal": 1.0, "max_vertical": -0.1},
        "next_to": {"max_distance": 1.5},  # Increased from 1.0 for better coverage
        "near": {"max_distance": 3.0},  # Increased from 2.0 for typical room layouts
        "inside": {"margin": 0.1},
        "between": {"max_distance_ratio": 0.3},  # max distance from line / line length
    }

    def __init__(self, thresholds: dict | None = None):
        """
        Initialize the spatial relation checker.

        Args:
            thresholds: Optional custom thresholds to override defaults
        """
        self.thresholds = {**self.DEFAULT_THRESHOLDS}
        if thresholds:
            for key, value in thresholds.items():
                if key in self.thresholds:
                    self.thresholds[key].update(value)
                else:
                    self.thresholds[key] = value

    def check(
        self,
        target: SceneObject,
        anchor: SceneObject | list[SceneObject],
        relation: str,
    ) -> RelationResult:
        """
        Check if target satisfies the spatial relation with anchor.

        Args:
            target: The target object
            anchor: The reference object(s). List for "between" relation.
            relation: Spatial relation (e.g., "on", "near", "between")

        Returns:
            RelationResult with satisfies (bool) and score (float)
        """
        # Normalize relation name
        canonical = RELATION_ALIASES.get(
            relation.lower().replace(" ", "_"), relation.lower()
        )

        # Get the appropriate check function
        check_func = getattr(self, f"is_{canonical}", None)

        if check_func is None:
            # Unknown relation - fall back to proximity check
            return self.is_near(target, anchor)

        # Handle multi-anchor relations
        if canonical == "between":
            if isinstance(anchor, list) and len(anchor) >= 2:
                best = RelationResult(satisfies=False, score=0.0)
                best_unsatisfied = RelationResult(satisfies=False, score=0.0)
                for anchor1, anchor2 in combinations(anchor, 2):
                    result = check_func(target, anchor1, anchor2)
                    pair = (
                        int(getattr(anchor1, "obj_id", -1)),
                        int(getattr(anchor2, "obj_id", -1)),
                    )
                    result.details = dict(result.details)
                    result.details["anchor_pair"] = pair
                    if result.satisfies and result.score >= best.score:
                        best = result
                    elif result.score >= best_unsatisfied.score:
                        best_unsatisfied = result
                return best if best.satisfies else best_unsatisfied
            else:
                return RelationResult(satisfies=False, score=0.0)

        # Single anchor relations
        if isinstance(anchor, list):
            anchor = anchor[0] if anchor else None

        if anchor is None:
            return RelationResult(satisfies=False, score=0.0)

        return check_func(target, anchor)

    def _get_centroid(self, obj: SceneObject) -> np.ndarray:
        """Get object centroid as numpy array."""
        if hasattr(obj, "centroid") and obj.centroid is not None:
            return np.asarray(obj.centroid, dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    def _coerce_bbox_minmax(
        self, bbox: Any
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Coerce common bbox representations into (min_xyz, max_xyz)."""
        if bbox is None:
            return None

        if hasattr(bbox, "min_point") and hasattr(bbox, "max_point"):
            min_point = np.asarray(bbox.min_point, dtype=np.float32).reshape(-1)[:3]
            max_point = np.asarray(bbox.max_point, dtype=np.float32).reshape(-1)[:3]
            if min_point.size == 3 and max_point.size == 3:
                return np.minimum(min_point, max_point), np.maximum(
                    min_point, max_point
                )
            return None

        try:
            arr = np.asarray(bbox, dtype=np.float32)
        except (TypeError, ValueError):
            return None

        if arr.size < 6:
            return None

        if arr.ndim == 1 and arr.size == 6:
            points = arr.reshape(2, 3)
        elif arr.ndim >= 2 and arr.shape[-1] >= 3:
            points = arr.reshape(-1, arr.shape[-1])[:, :3]
        elif arr.ndim == 1 and arr.size % 3 == 0:
            points = arr.reshape(-1, 3)
        else:
            return None

        if len(points) < 2 or not np.isfinite(points).all():
            return None

        min_point = np.min(points, axis=0).astype(np.float32)
        max_point = np.max(points, axis=0).astype(np.float32)
        return min_point, max_point

    def _get_bbox(self, obj: SceneObject) -> tuple[np.ndarray, np.ndarray] | None:
        """Get object bounding box as (min_point, max_point)."""
        for attr_name in ("bbox_3d", "bbox_np", "bbox"):
            if hasattr(obj, attr_name):
                bbox = self._coerce_bbox_minmax(getattr(obj, attr_name))
                if bbox is not None:
                    return bbox
        return None

    def _bbox_center(self, bbox: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return bbox center."""
        min_point, max_point = bbox
        return (min_point + max_point) / 2

    def _bbox_xy_gap(
        self,
        bbox1: tuple[np.ndarray, np.ndarray],
        bbox2: tuple[np.ndarray, np.ndarray],
    ) -> float:
        """Return the XY distance between two bbox footprints."""
        min1, max1 = bbox1
        min2, max2 = bbox2
        dx = max(float(min2[0] - max1[0]), float(min1[0] - max2[0]), 0.0)
        dy = max(float(min2[1] - max1[1]), float(min1[1] - max2[1]), 0.0)
        return float(np.hypot(dx, dy))

    def _bbox_3d_gap(
        self,
        bbox1: tuple[np.ndarray, np.ndarray],
        bbox2: tuple[np.ndarray, np.ndarray],
    ) -> float:
        """Return the Euclidean gap between two 3D bboxes."""
        min1, max1 = bbox1
        min2, max2 = bbox2
        gaps = np.maximum(np.maximum(min2 - max1, min1 - max2), 0.0)
        return float(np.linalg.norm(gaps))

    def _bbox_xy_overlap_ratio(
        self,
        bbox1: tuple[np.ndarray, np.ndarray],
        bbox2: tuple[np.ndarray, np.ndarray],
    ) -> float:
        """Return XY intersection over the smaller footprint area."""
        min1, max1 = bbox1
        min2, max2 = bbox2
        overlap_x = max(0.0, float(min(max1[0], max2[0]) - max(min1[0], min2[0])))
        overlap_y = max(0.0, float(min(max1[1], max2[1]) - max(min1[1], min2[1])))
        intersection = overlap_x * overlap_y
        area1 = max(0.0, float((max1[0] - min1[0]) * (max1[1] - min1[1])))
        area2 = max(0.0, float((max2[0] - min2[0]) * (max2[1] - min2[1])))
        denom = min(area1, area2)
        if denom <= 1e-6:
            return 0.0
        return float(intersection / denom)

    def _axis_overlap_ratio(
        self,
        bbox1: tuple[np.ndarray, np.ndarray],
        bbox2: tuple[np.ndarray, np.ndarray],
        axis: int,
    ) -> float:
        """Return axis overlap over bbox1 extent."""
        min1, max1 = bbox1
        min2, max2 = bbox2
        overlap = max(0.0, float(min(max1[axis], max2[axis]) - max(min1[axis], min2[axis])))
        extent = max(1e-6, float(max1[axis] - min1[axis]))
        return float(overlap / extent)

    def _category_text(self, obj: SceneObject) -> str:
        """Return lower-cased category text from common object label fields."""
        parts: list[str] = []
        for attr in ("category", "object_tag", "summary"):
            value = getattr(obj, attr, None)
            if isinstance(value, str):
                parts.append(value)
        class_name = getattr(obj, "class_name", None)
        if isinstance(class_name, (list, tuple)):
            parts.extend(str(v) for v in class_name if isinstance(v, str))
        return " ".join(parts).lower()

    def _is_vertical_surface_anchor(self, obj: SceneObject) -> bool:
        """Return True for anchors whose natural-language ``on`` means attached-to."""
        text = self._category_text(obj)
        vertical_terms = {
            "wall",
            "partition",
            "door",
            "whiteboard",
            "blackboard",
            "chalkboard",
            "bulletin board",
            "board",
            "mirror",
            "window",
        }
        return any(term in text for term in vertical_terms)

    def _check_vertical_surface_attachment(
        self,
        target: SceneObject,
        anchor: SceneObject,
        target_bbox: tuple[np.ndarray, np.ndarray],
        anchor_bbox: tuple[np.ndarray, np.ndarray],
    ) -> RelationResult:
        """Check ``target on anchor`` where anchor is a vertical surface."""
        xy_gap = self._bbox_xy_gap(target_bbox, anchor_bbox)
        overlap_ratio = self._bbox_xy_overlap_ratio(target_bbox, anchor_bbox)
        z_overlap_ratio = self._axis_overlap_ratio(target_bbox, anchor_bbox, axis=2)
        max_surface_gap = 0.75
        has_surface_contact = overlap_ratio > 0.0 or xy_gap <= max_surface_gap
        has_vertical_overlap = z_overlap_ratio >= 0.25
        details = {
            "mode": "bbox_vertical_surface_attachment",
            "bbox_used": True,
            "xy_gap": xy_gap,
            "xy_overlap_ratio": overlap_ratio,
            "z_overlap_ratio": z_overlap_ratio,
            "max_surface_gap": max_surface_gap,
        }

        if not has_surface_contact:
            details["reason"] = "surface_proximity"
            return RelationResult(satisfies=False, score=0.0, details=details)
        if not has_vertical_overlap:
            details["reason"] = "vertical_overlap"
            return RelationResult(satisfies=False, score=0.0, details=details)

        proximity_score = (
            min(1.0, overlap_ratio)
            if overlap_ratio > 0.0
            else max(0.0, 1.0 - xy_gap / (max_surface_gap + 1e-6))
        )
        score = 0.55 * proximity_score + 0.45 * min(1.0, z_overlap_ratio)
        return RelationResult(satisfies=True, score=float(score), details=details)

    # ========== Vertical Relations ==========

    def is_on_top_of(
        self,
        target: SceneObject,
        anchor: SceneObject,
    ) -> RelationResult:
        """
        Check if target is on top of anchor.

        Criteria:
        - Target is above anchor (positive z difference)
        - Horizontal distance is small (within anchor's footprint)
        """
        t_bbox = self._get_bbox(target)
        a_bbox = self._get_bbox(anchor)
        if t_bbox is not None and a_bbox is not None:
            if self._is_vertical_surface_anchor(anchor):
                return self._check_vertical_surface_attachment(
                    target, anchor, t_bbox, a_bbox
                )

            t_min, _t_max = t_bbox
            a_min, a_max = a_bbox
            thres = self.thresholds["on_top_of"]
            xy_gap = self._bbox_xy_gap(t_bbox, a_bbox)
            overlap_ratio = self._bbox_xy_overlap_ratio(t_bbox, a_bbox)
            vertical_gap = float(t_min[2] - a_max[2])
            contact_tolerance = float(thres.get("contact_tolerance", 0.2))
            anchor_xy_diag = float(np.linalg.norm(a_max[:2] - a_min[:2]))
            max_horizontal = max(
                float(thres["max_horizontal"]),
                min(1.0, 0.35 * anchor_xy_diag),
            )
            anchor_height = max(0.0, float(a_max[2] - a_min[2]))
            penetration_tolerance = (
                max(contact_tolerance, min(1.75, 0.85 * anchor_height))
                if overlap_ratio > 0.0
                else max(contact_tolerance, min(0.45, 0.45 * anchor_height))
            )
            max_vertical = float(thres["max_vertical"])

            vertical_ok = (
                vertical_gap >= float(thres["min_vertical"]) - penetration_tolerance
                and vertical_gap <= max_vertical
            )
            supported = overlap_ratio > 0.0 or xy_gap <= max_horizontal
            details = {
                "mode": "bbox_horizontal_support",
                "bbox_used": True,
                "xy_gap": xy_gap,
                "xy_overlap_ratio": overlap_ratio,
                "vertical_gap": vertical_gap,
                "max_horizontal": max_horizontal,
                "penetration_tolerance": penetration_tolerance,
            }

            if not supported:
                details["reason"] = "horizontal_support"
                return RelationResult(satisfies=False, score=0.0, details=details)
            if not vertical_ok:
                details["reason"] = "vertical_gap"
                return RelationResult(satisfies=False, score=0.0, details=details)

            gap_score = max(0.0, 1.0 - xy_gap / (max_horizontal + 1e-6))
            support_score = max(gap_score, min(1.0, overlap_ratio))
            if vertical_gap < 0.0:
                vertical_score = max(
                    0.0,
                    1.0 - abs(vertical_gap) / (penetration_tolerance + 1e-6),
                )
            else:
                vertical_score = max(
                    0.0,
                    1.0 - abs(vertical_gap) / (max_vertical + 1e-6),
                )
            score = 0.65 * support_score + 0.35 * vertical_score
            return RelationResult(satisfies=True, score=float(score), details=details)

        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)
        diff = t_pos - a_pos

        thres = self.thresholds["on_top_of"]
        horizontal_dist = np.linalg.norm(diff[:2])
        vertical_diff = diff[2]

        # Must be above
        if vertical_diff < thres["min_vertical"]:
            return RelationResult(satisfies=False, score=0.0)

        # Must be within horizontal threshold
        if horizontal_dist > thres["max_horizontal"]:
            return RelationResult(satisfies=False, score=0.0)

        # Score: higher when closer horizontally and moderately above
        h_score = max(0, 1 - horizontal_dist / thres["max_horizontal"])
        v_score = min(1.0, vertical_diff / 0.3)  # Peak at 0.3m above
        score = 0.6 * h_score + 0.4 * v_score

        return RelationResult(
            satisfies=True,
            score=float(score),
            details={
                "horizontal_dist": float(horizontal_dist),
                "vertical_diff": float(vertical_diff),
            },
        )

    def is_above(
        self,
        target: SceneObject,
        anchor: SceneObject,
    ) -> RelationResult:
        """
        Check if target is above anchor (not necessarily touching).
        """
        t_bbox = self._get_bbox(target)
        a_bbox = self._get_bbox(anchor)
        if t_bbox is not None and a_bbox is not None:
            t_center = self._bbox_center(t_bbox)
            a_center = self._bbox_center(a_bbox)
            thres = self.thresholds["above"]
            vertical_diff = float(t_center[2] - a_center[2])
            xy_gap = self._bbox_xy_gap(t_bbox, a_bbox)
            overlap_ratio = self._bbox_xy_overlap_ratio(t_bbox, a_bbox)
            max_horizontal = float(thres["max_horizontal"])
            details = {
                "mode": "bbox_vertical_ordering",
                "bbox_used": True,
                "xy_gap": xy_gap,
                "xy_overlap_ratio": overlap_ratio,
                "vertical_diff": vertical_diff,
            }

            if vertical_diff < float(thres["min_vertical"]):
                details["reason"] = "vertical_order"
                return RelationResult(satisfies=False, score=0.0, details=details)
            if xy_gap > max_horizontal and overlap_ratio <= 0.0:
                details["reason"] = "horizontal_support"
                return RelationResult(satisfies=False, score=0.0, details=details)

            h_score = (
                min(1.0, overlap_ratio)
                if overlap_ratio > 0
                else max(0.0, 1.0 - xy_gap / (max_horizontal + 1e-6))
            )
            v_score = min(1.0, vertical_diff / 1.0)
            score = 0.5 * h_score + 0.5 * v_score
            return RelationResult(satisfies=True, score=float(score), details=details)

        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)
        diff = t_pos - a_pos

        thres = self.thresholds["above"]
        horizontal_dist = np.linalg.norm(diff[:2])
        vertical_diff = diff[2]

        if vertical_diff < thres["min_vertical"]:
            return RelationResult(satisfies=False, score=0.0)

        if horizontal_dist > thres["max_horizontal"]:
            return RelationResult(satisfies=False, score=0.0)

        # Score based on how directly above
        h_score = max(0, 1 - horizontal_dist / thres["max_horizontal"])
        v_score = min(1.0, vertical_diff / 1.0)
        score = 0.5 * h_score + 0.5 * v_score

        return RelationResult(satisfies=True, score=float(score))

    def is_below(
        self,
        target: SceneObject,
        anchor: SceneObject,
    ) -> RelationResult:
        """
        Check if target is below anchor.
        """
        t_bbox = self._get_bbox(target)
        a_bbox = self._get_bbox(anchor)
        if t_bbox is not None and a_bbox is not None:
            t_center = self._bbox_center(t_bbox)
            a_center = self._bbox_center(a_bbox)
            thres = self.thresholds["below"]
            vertical_diff = float(t_center[2] - a_center[2])
            xy_gap = self._bbox_xy_gap(t_bbox, a_bbox)
            overlap_ratio = self._bbox_xy_overlap_ratio(t_bbox, a_bbox)
            max_horizontal = float(thres["max_horizontal"])
            details = {
                "mode": "bbox_vertical_ordering",
                "bbox_used": True,
                "xy_gap": xy_gap,
                "xy_overlap_ratio": overlap_ratio,
                "vertical_diff": vertical_diff,
            }

            if vertical_diff > float(thres["max_vertical"]):
                details["reason"] = "vertical_order"
                return RelationResult(satisfies=False, score=0.0, details=details)
            if xy_gap > max_horizontal and overlap_ratio <= 0.0:
                details["reason"] = "horizontal_support"
                return RelationResult(satisfies=False, score=0.0, details=details)

            h_score = (
                min(1.0, overlap_ratio)
                if overlap_ratio > 0
                else max(0.0, 1.0 - xy_gap / (max_horizontal + 1e-6))
            )
            v_score = min(1.0, abs(vertical_diff) / 1.0)
            score = 0.5 * h_score + 0.5 * v_score
            return RelationResult(satisfies=True, score=float(score), details=details)

        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)
        diff = t_pos - a_pos

        thres = self.thresholds["below"]
        horizontal_dist = np.linalg.norm(diff[:2])
        vertical_diff = diff[2]

        if vertical_diff > thres["max_vertical"]:
            return RelationResult(satisfies=False, score=0.0)

        if horizontal_dist > thres["max_horizontal"]:
            return RelationResult(satisfies=False, score=0.0)

        h_score = max(0, 1 - horizontal_dist / thres["max_horizontal"])
        v_score = min(1.0, abs(vertical_diff) / 1.0)
        score = 0.5 * h_score + 0.5 * v_score

        return RelationResult(satisfies=True, score=float(score))

    # ========== Horizontal Distance Relations ==========

    def is_next_to(
        self,
        target: SceneObject,
        anchor: SceneObject,
    ) -> RelationResult:
        """
        Check if target is next to anchor (close proximity).
        """
        t_bbox = self._get_bbox(target)
        a_bbox = self._get_bbox(anchor)
        if t_bbox is not None and a_bbox is not None:
            thres = self.thresholds["next_to"]
            distance = self._bbox_xy_gap(t_bbox, a_bbox)
            max_distance = float(thres["max_distance"])
            details = {
                "distance": distance,
                "distance_mode": "bbox_xy_gap",
                "bbox_used": True,
            }
            if distance > max_distance:
                return RelationResult(satisfies=False, score=0.0, details=details)
            score = max(0.0, 1.0 - distance / (max_distance + 1e-6))
            return RelationResult(satisfies=True, score=float(score), details=details)

        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)

        thres = self.thresholds["next_to"]
        distance = float(np.linalg.norm(t_pos - a_pos))

        if distance > thres["max_distance"]:
            return RelationResult(satisfies=False, score=0.0)

        score = max(0, 1 - distance / thres["max_distance"])
        return RelationResult(
            satisfies=True, score=score, details={"distance": distance}
        )

    def is_near(
        self,
        target: SceneObject,
        anchor: SceneObject,
    ) -> RelationResult:
        """
        Check if target is near anchor (looser than next_to).
        """
        t_bbox = self._get_bbox(target)
        a_bbox = self._get_bbox(anchor)
        if t_bbox is not None and a_bbox is not None:
            thres = self.thresholds["near"]
            distance = self._bbox_3d_gap(t_bbox, a_bbox)
            max_distance = float(thres["max_distance"])
            details = {
                "distance": distance,
                "distance_mode": "bbox_3d_gap",
                "bbox_used": True,
            }
            if distance > max_distance:
                return RelationResult(satisfies=False, score=0.0, details=details)
            score = max(0.0, 1.0 - distance / (max_distance + 1e-6))
            return RelationResult(satisfies=True, score=float(score), details=details)

        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)

        thres = self.thresholds["near"]
        distance = float(np.linalg.norm(t_pos - a_pos))

        if distance > thres["max_distance"]:
            return RelationResult(satisfies=False, score=0.0)

        score = max(0, 1 - distance / thres["max_distance"])
        return RelationResult(
            satisfies=True, score=score, details={"distance": distance}
        )

    # ========== Directional Relations ==========
    # Note: These require a reference direction. We use the global coordinate system.
    # Typically: +X = right, +Y = forward, +Z = up (may need adjustment per scene)

    def is_in_front_of(
        self,
        target: SceneObject,
        anchor: SceneObject,
    ) -> RelationResult:
        """
        Check if target is in front of anchor.

        Uses +Y as the forward direction (scene-dependent).
        """
        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)
        diff = t_pos - a_pos

        # Target should be in positive Y direction from anchor
        if diff[1] <= 0:
            return RelationResult(satisfies=False, score=0.0)

        # Also check horizontal distance isn't too far
        lateral_dist = abs(diff[0])
        forward_dist = diff[1]

        if lateral_dist > forward_dist:
            # More to the side than in front
            return RelationResult(satisfies=False, score=0.0)

        score = min(1.0, forward_dist / 2.0) * max(0, 1 - lateral_dist / forward_dist)
        return RelationResult(satisfies=True, score=float(score))

    def is_behind(
        self,
        target: SceneObject,
        anchor: SceneObject,
    ) -> RelationResult:
        """
        Check if target is behind anchor.
        """
        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)
        diff = t_pos - a_pos

        if diff[1] >= 0:
            return RelationResult(satisfies=False, score=0.0)

        lateral_dist = abs(diff[0])
        backward_dist = abs(diff[1])

        if lateral_dist > backward_dist:
            return RelationResult(satisfies=False, score=0.0)

        score = min(1.0, backward_dist / 2.0) * max(0, 1 - lateral_dist / backward_dist)
        return RelationResult(satisfies=True, score=float(score))

    def is_left_of(
        self,
        target: SceneObject,
        anchor: SceneObject,
    ) -> RelationResult:
        """
        Check if target is to the left of anchor.

        Uses -X as the left direction.
        """
        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)
        diff = t_pos - a_pos

        if diff[0] >= 0:
            return RelationResult(satisfies=False, score=0.0)

        lateral_dist = abs(diff[0])
        forward_dist = abs(diff[1])

        if forward_dist > lateral_dist:
            return RelationResult(satisfies=False, score=0.0)

        score = min(1.0, lateral_dist / 2.0)
        return RelationResult(satisfies=True, score=float(score))

    def is_right_of(
        self,
        target: SceneObject,
        anchor: SceneObject,
    ) -> RelationResult:
        """
        Check if target is to the right of anchor.
        """
        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)
        diff = t_pos - a_pos

        if diff[0] <= 0:
            return RelationResult(satisfies=False, score=0.0)

        lateral_dist = abs(diff[0])
        forward_dist = abs(diff[1])

        if forward_dist > lateral_dist:
            return RelationResult(satisfies=False, score=0.0)

        score = min(1.0, lateral_dist / 2.0)
        return RelationResult(satisfies=True, score=float(score))

    # ========== Containment Relations ==========

    def is_inside(
        self,
        target: SceneObject,
        anchor: SceneObject,
    ) -> RelationResult:
        """
        Check if target is inside anchor.

        Requires bounding box information for accurate check.
        Falls back to proximity if no bbox available.
        """
        t_pos = self._get_centroid(target)
        t_bbox = self._get_bbox(target)
        a_bbox = self._get_bbox(anchor)

        if a_bbox is None:
            # Cannot determine containment without bounding box
            # Do NOT fall back to proximity - "inside" requires bbox data
            return RelationResult(
                satisfies=False,
                score=0.0,
                details={"bbox_used": False, "reason": "missing_anchor_bbox"},
            )

        a_min, a_max = a_bbox
        margin = self.thresholds["inside"]["margin"]

        if t_bbox is not None:
            t_min, t_max = t_bbox
            inside = np.all(t_min >= a_min - margin) and np.all(
                t_max <= a_max + margin
            )
            target_center = self._bbox_center(t_bbox)
            bbox_used = True
        else:
            inside = np.all(t_pos >= a_min - margin) and np.all(t_pos <= a_max + margin)
            target_center = t_pos
            bbox_used = False

        if not inside:
            return RelationResult(
                satisfies=False,
                score=0.0,
                details={"bbox_used": bbox_used, "reason": "outside_anchor_bbox"},
            )

        # Score based on how centered within the bbox
        a_center = (a_min + a_max) / 2
        a_size = a_max - a_min
        normalized_dist = np.abs(target_center - a_center) / (a_size / 2 + 1e-6)
        score = float(1 - np.mean(normalized_dist))

        return RelationResult(
            satisfies=True,
            score=max(0, score),
            details={"bbox_used": bbox_used},
        )

    # ========== Multi-Object Relations ==========

    def is_between(
        self,
        target: SceneObject,
        anchor1: SceneObject,
        anchor2: SceneObject,
    ) -> RelationResult:
        """
        Check if target is between anchor1 and anchor2.
        """
        t_pos = self._get_centroid(target)
        a1_pos = self._get_centroid(anchor1)
        a2_pos = self._get_centroid(anchor2)

        # Vector from anchor1 to anchor2
        line_vec = a2_pos - a1_pos
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-6:
            return RelationResult(satisfies=False, score=0.0)

        line_dir = line_vec / line_len

        # Project target onto the line
        t_vec = t_pos - a1_pos
        proj_len = np.dot(t_vec, line_dir)

        # Check if projection is between the two anchors
        if proj_len < 0 or proj_len > line_len:
            return RelationResult(satisfies=False, score=0.0)

        # Distance from target to the line
        proj_point = a1_pos + proj_len * line_dir
        dist_to_line = np.linalg.norm(t_pos - proj_point)

        # Check if close enough to the line
        thres = self.thresholds["between"]
        max_dist = line_len * thres["max_distance_ratio"]

        if dist_to_line > max_dist:
            return RelationResult(satisfies=False, score=0.0)

        # Score: higher when more centered and closer to line
        center_score = 1 - abs(proj_len - line_len / 2) / (line_len / 2)
        line_score = 1 - dist_to_line / max_dist if max_dist > 0 else 1.0
        score = 0.5 * center_score + 0.5 * line_score

        return RelationResult(
            satisfies=True,
            score=float(score),
            details={
                "proj_len": proj_len,
                "line_len": line_len,
                "dist_to_line": dist_to_line,
            },
        )

    # ========== Contact Relations ==========

    def is_against(
        self,
        target: SceneObject,
        anchor: SceneObject,
    ) -> RelationResult:
        """
        Check if target is against (leaning on) anchor.

        Similar to next_to but with stricter distance threshold.
        """
        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)

        distance = float(np.linalg.norm(t_pos - a_pos))

        # Very close proximity for "against"
        max_dist = 0.5
        if distance > max_dist:
            return RelationResult(satisfies=False, score=0.0)

        score = max(0, 1 - distance / max_dist)
        return RelationResult(satisfies=True, score=score)

    def is_around(
        self,
        target: SceneObject,
        anchor: SceneObject,
    ) -> RelationResult:
        """
        Check if target is around anchor.

        This is essentially the same as "near" but can be used for
        collections of objects surrounding something.
        """
        return self.is_near(target, anchor)


# Convenience function
def check_relation(
    target: SceneObject,
    anchor: SceneObject | list[SceneObject],
    relation: str,
    checker: SpatialRelationChecker | None = None,
) -> tuple[bool, float]:
    """
    Check if target satisfies the spatial relation with anchor.

    Args:
        target: Target object
        anchor: Reference object(s)
        relation: Spatial relation string
        checker: Optional pre-initialized checker

    Returns:
        Tuple of (satisfies, score)
    """
    if checker is None:
        checker = SpatialRelationChecker()

    result = checker.check(target, anchor, relation)
    return result.satisfies, result.score


def get_canonical_relation(relation: str) -> str:
    """Get the canonical relation name for a given relation string."""
    return RELATION_ALIASES.get(relation.lower().replace(" ", "_"), relation.lower())
