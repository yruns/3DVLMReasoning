"""Core query_scene types and data structures.

This module exposes the fundamental types from the core subpackage.
"""

from .hypotheses import (
    DIRECTIONAL_RELATIONS,
    SUPPORTED_RELATIONS,
    SUPPORTED_RELATIONS_STR,
    ConstraintType,
    ExecutionPolicy,
    GroundingQuery,
    HypothesisKind,
    HypothesisOutputV1,
    ParseMode,
    QueryHypothesis,
    QueryNode,
    ReferenceFrame,
    SelectConstraint,
    SpatialConstraint,
    SpatialRelation,
    ViewpointContext,
    ViewpointKind,
    simple_query,
    spatial_query,
    superlative_query,
)
from .query_types import (
    BoundingBox3D,
    CameraPose,
    GroundingResult,
    ObjectDescriptions,
    ObjectNode,
    QueryInfo,
    QueryType,
    RegionNode,
    ViewScore,
)
from .results import (
    EvidenceBundle,
    ExecutionResult,
    KeyframeResult,
)

__all__ = [
    # Query types
    "QueryType",
    "BoundingBox3D",
    "ObjectDescriptions",
    "ObjectNode",
    "RegionNode",
    "ViewScore",
    "QueryInfo",
    "GroundingResult",
    "CameraPose",
    # Hypotheses
    "ConstraintType",
    "SpatialRelation",
    "HypothesisKind",
    "ParseMode",
    "QueryNode",
    "SpatialConstraint",
    "SelectConstraint",
    "GroundingQuery",
    "QueryHypothesis",
    "HypothesisOutputV1",
    "ReferenceFrame",
    "ViewpointKind",
    "ExecutionPolicy",
    "ViewpointContext",
    "DIRECTIONAL_RELATIONS",
    "SUPPORTED_RELATIONS",
    "SUPPORTED_RELATIONS_STR",
    "simple_query",
    "spatial_query",
    "superlative_query",
    # Results
    "KeyframeResult",
    "ExecutionResult",
    "EvidenceBundle",
]
