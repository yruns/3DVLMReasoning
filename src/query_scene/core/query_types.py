"""Core query types and data structures.

This module defines the fundamental data types used throughout the query_scene
package for representing queries, objects, views, and grounding results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class QueryType(str, Enum):
    """Query type classification."""

    SIMPLE = "simple"  # Direct object reference (e.g., "the chair")
    SPATIAL = "spatial"  # Spatial relation (e.g., "lamp on table")
    SUPERLATIVE = "superlative"  # Selection constraint (e.g., "nearest door")
    COMPOUND = "compound"  # Multiple constraints
    UNKNOWN = "unknown"


@dataclass
class BoundingBox3D:
    """3D axis-aligned bounding box.

    Attributes:
        min_corner: (x, y, z) minimum coordinates
        max_corner: (x, y, z) maximum coordinates
    """

    min_corner: np.ndarray
    max_corner: np.ndarray

    def __post_init__(self):
        self.min_corner = np.asarray(self.min_corner, dtype=np.float32)
        self.max_corner = np.asarray(self.max_corner, dtype=np.float32)

    @property
    def center(self) -> np.ndarray:
        """Compute the center of the bounding box."""
        return (self.min_corner + self.max_corner) / 2

    @property
    def size(self) -> np.ndarray:
        """Compute the size (extent) of the bounding box."""
        return self.max_corner - self.min_corner

    @property
    def volume(self) -> float:
        """Compute the volume of the bounding box."""
        s = self.size
        return float(s[0] * s[1] * s[2])

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "min_corner": self.min_corner.tolist(),
            "max_corner": self.max_corner.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BoundingBox3D":
        return cls(
            min_corner=np.array(data["min_corner"]),
            max_corner=np.array(data["max_corner"]),
        )

    @classmethod
    def from_points(cls, points: np.ndarray) -> "BoundingBox3D":
        """Create bounding box from point cloud."""
        return cls(
            min_corner=points.min(axis=0),
            max_corner=points.max(axis=0),
        )

    def contains(self, point: np.ndarray) -> bool:
        """Check if a point is inside the bounding box."""
        point = np.asarray(point)
        return bool(
            np.all(point >= self.min_corner) and np.all(point <= self.max_corner)
        )

    def intersection_volume(self, other: "BoundingBox3D") -> float:
        """Compute intersection volume with another bounding box."""
        inter_min = np.maximum(self.min_corner, other.min_corner)
        inter_max = np.minimum(self.max_corner, other.max_corner)
        inter_size = np.maximum(0, inter_max - inter_min)
        return float(np.prod(inter_size))

    def iou(self, other: "BoundingBox3D") -> float:
        """Compute Intersection over Union with another bounding box."""
        inter = self.intersection_volume(other)
        union = self.volume + other.volume - inter
        return inter / union if union > 0 else 0.0


@dataclass
class ObjectDescriptions:
    """Multi-source object descriptions.

    Attributes:
        clip_description: CLIP-based object description
        vlm_description: VLM-generated description
        llm_description: LLM-generated description
    """

    clip_description: str = ""
    vlm_description: str = ""
    llm_description: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "clip_description": self.clip_description,
            "vlm_description": self.vlm_description,
            "llm_description": self.llm_description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "ObjectDescriptions":
        return cls(
            clip_description=data.get("clip_description", ""),
            vlm_description=data.get("vlm_description", ""),
            llm_description=data.get("llm_description", ""),
        )


@dataclass
class ObjectNode:
    """A node in the scene graph representing an object.

    Attributes:
        object_id: Unique identifier
        category: Semantic category (e.g., "chair", "table")
        centroid: 3D centroid position
        bbox: 3D bounding box
        mask_id: Optional segmentation mask ID
        clip_feature: Optional CLIP embedding
        confidence: Detection confidence score
        descriptions: Multi-source descriptions
        attributes: Additional attributes (color, material, etc.)
        visible_from_frames: Frame indices where object is visible
    """

    object_id: int
    category: str
    centroid: np.ndarray
    bbox: BoundingBox3D | None = None
    mask_id: int | None = None
    clip_feature: np.ndarray | None = None
    confidence: float = 1.0
    descriptions: ObjectDescriptions | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    visible_from_frames: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.centroid = np.asarray(self.centroid, dtype=np.float32)
        if self.clip_feature is not None:
            self.clip_feature = np.asarray(self.clip_feature, dtype=np.float32)

    def to_dict(self, include_features: bool = False) -> dict[str, Any]:
        result = {
            "object_id": self.object_id,
            "category": self.category,
            "centroid": self.centroid.tolist(),
            "confidence": self.confidence,
            "attributes": self.attributes,
        }
        if self.bbox is not None:
            result["bbox"] = self.bbox.to_dict()
        if self.mask_id is not None:
            result["mask_id"] = self.mask_id
        if self.descriptions is not None:
            result["descriptions"] = self.descriptions.to_dict()
        if include_features and self.clip_feature is not None:
            result["clip_feature"] = self.clip_feature.tolist()
        if self.visible_from_frames:
            result["visible_from_frames"] = self.visible_from_frames
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectNode":
        bbox = None
        if "bbox" in data:
            bbox = BoundingBox3D.from_dict(data["bbox"])

        descriptions = None
        if "descriptions" in data:
            descriptions = ObjectDescriptions.from_dict(data["descriptions"])

        clip_feature = None
        if "clip_feature" in data:
            clip_feature = np.array(data["clip_feature"])

        return cls(
            object_id=data["object_id"],
            category=data["category"],
            centroid=np.array(data["centroid"]),
            bbox=bbox,
            mask_id=data.get("mask_id"),
            clip_feature=clip_feature,
            confidence=data.get("confidence", 1.0),
            descriptions=descriptions,
            attributes=data.get("attributes", {}),
            visible_from_frames=data.get("visible_from_frames", []),
        )

    def __repr__(self) -> str:
        return f"ObjectNode(id={self.object_id}, category='{self.category}')"


@dataclass
class RegionNode:
    """A region in the scene (room, area, zone).

    Attributes:
        region_id: Unique identifier
        region_type: Type of region (e.g., "room", "area")
        name: Human-readable name
        centroid: 3D centroid position
        bbox: 3D bounding box
        object_ids: IDs of objects contained in this region
    """

    region_id: int
    region_type: str
    name: str
    centroid: np.ndarray
    bbox: BoundingBox3D | None = None
    object_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "region_id": self.region_id,
            "region_type": self.region_type,
            "name": self.name,
            "centroid": self.centroid.tolist(),
            "object_ids": self.object_ids,
        }
        if self.bbox is not None:
            result["bbox"] = self.bbox.to_dict()
        return result


@dataclass
class ViewScore:
    """Scores for a view (frame) of an object.

    Attributes:
        frame_id: Frame index
        visibility: Visibility score (0-1)
        distance: Distance from camera
        angle: View angle score
        occlusion: Occlusion score (0 = fully occluded, 1 = unoccluded)
        semantic_relevance: Semantic relevance to query
    """

    frame_id: int
    visibility: float = 0.0
    distance: float = 0.0
    angle: float = 0.0
    occlusion: float = 1.0
    semantic_relevance: float = 0.0

    def get_composite_score(self, weights: dict[str, float] | None = None) -> float:
        """Compute weighted composite score."""
        default_weights = {
            "visibility": 0.3,
            "distance": 0.2,
            "angle": 0.2,
            "occlusion": 0.15,
            "semantic_relevance": 0.15,
        }
        weights = weights or default_weights

        return (
            weights.get("visibility", 0.3) * self.visibility
            + weights.get("distance", 0.2) * (1 / (1 + self.distance))
            + weights.get("angle", 0.2) * self.angle
            + weights.get("occlusion", 0.15) * self.occlusion
            + weights.get("semantic_relevance", 0.15) * self.semantic_relevance
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "visibility": self.visibility,
            "distance": self.distance,
            "angle": self.angle,
            "occlusion": self.occlusion,
            "semantic_relevance": self.semantic_relevance,
        }


@dataclass
class QueryInfo:
    """Metadata about a query.

    Attributes:
        raw_query: Original query string
        query_type: Classified query type
        parsed_categories: Extracted object categories
        confidence: Parser confidence
    """

    raw_query: str
    query_type: QueryType = QueryType.UNKNOWN
    parsed_categories: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "query_type": self.query_type.value,
            "parsed_categories": self.parsed_categories,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryInfo":
        return cls(
            raw_query=data["raw_query"],
            query_type=QueryType(data.get("query_type", "unknown")),
            parsed_categories=data.get("parsed_categories", []),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class GroundingResult:
    """Result of a grounding query.

    Attributes:
        success: Whether grounding succeeded
        object_node: The grounded object (if success)
        score: Confidence score
        reason: Explanation or failure reason
        alternatives: Alternative candidates
        query_info: Query metadata
    """

    success: bool = False
    object_node: ObjectNode | None = None
    score: float = 0.0
    reason: str = ""
    alternatives: list[ObjectNode] = field(default_factory=list)
    query_info: QueryInfo | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "success": self.success,
            "score": self.score,
            "reason": self.reason,
        }
        if self.object_node is not None:
            result["object_node"] = self.object_node.to_dict()
        if self.alternatives:
            result["alternatives"] = [a.to_dict() for a in self.alternatives]
        if self.query_info is not None:
            result["query_info"] = self.query_info.to_dict()
        return result

    @classmethod
    def failure(cls, reason: str) -> "GroundingResult":
        return cls(success=False, reason=reason)

    @classmethod
    def from_object(
        cls,
        obj: ObjectNode,
        score: float = 1.0,
        query_info: QueryInfo | None = None,
    ) -> "GroundingResult":
        return cls(
            success=True,
            object_node=obj,
            score=score,
            query_info=query_info,
        )


@dataclass
class CameraPose:
    """Camera pose with intrinsics.

    Attributes:
        pose: 4x4 camera-to-world transformation matrix
        intrinsics: 3x3 or 4x4 intrinsic matrix
        frame_id: Optional frame index
        timestamp: Optional timestamp
    """

    pose: np.ndarray
    intrinsics: np.ndarray
    frame_id: int | None = None
    timestamp: float | None = None

    def __post_init__(self):
        self.pose = np.asarray(self.pose, dtype=np.float64)
        self.intrinsics = np.asarray(self.intrinsics, dtype=np.float64)

    @property
    def fx(self) -> float:
        return float(self.intrinsics[0, 0])

    @property
    def fy(self) -> float:
        return float(self.intrinsics[1, 1])

    @property
    def cx(self) -> float:
        return float(self.intrinsics[0, 2])

    @property
    def cy(self) -> float:
        return float(self.intrinsics[1, 2])

    @property
    def position(self) -> np.ndarray:
        """Camera position in world coordinates."""
        return self.pose[:3, 3]

    @property
    def rotation(self) -> np.ndarray:
        """Camera rotation matrix (3x3)."""
        return self.pose[:3, :3]

    @property
    def forward(self) -> np.ndarray:
        """Camera forward direction (viewing direction)."""
        return -self.rotation[:, 2]  # Negative Z in camera frame
