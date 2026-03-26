"""
QueryScene: Query-Driven Scene Representation for VLM Reasoning
===============================================================

A novel scene representation optimized for VLM inference, featuring:
- Multi-granularity CLIP indexing (region -> object -> point)
- View-object bidirectional index with semantic scoring
- Query-adaptive VLM input construction
- Structured output parsing
- Nested spatial query support (e.g., "pillow on sofa nearest door")

Example:
    >>> from query_scene import QueryScenePipeline
    >>> pipeline = QueryScenePipeline.from_scene("/path/to/scene")
    >>> result = pipeline.query("沙发旁边的台灯")
    >>> print(result.object_node.category, result.centroid)

    # For nested queries:
    >>> from query_scene import QueryParser, QueryExecutor
    >>> parser = QueryParser(llm_model="gpt-4o", scene_categories=["pillow", "sofa", "door"])
    >>> query = parser.parse("the pillow on the sofa nearest the door")
"""

# Query parsing (nested spatial queries)
from .core import (
    SUPPORTED_RELATIONS,
    SUPPORTED_RELATIONS_STR,
    BoundingBox3D,
    CameraPose,
    ConstraintType,
    EvidenceBundle,
    ExecutionResult,
    GroundingQuery,
    GroundingResult,
    HypothesisKind,
    HypothesisOutputV1,
    KeyframeResult,
    ObjectDescriptions,
    ObjectNode,
    ParseMode,
    QueryHypothesis,
    QueryInfo,
    QueryNode,
    QueryType,
    RegionNode,
    SelectConstraint,
    SpatialConstraint,
    SpatialRelation,
    ViewScore,
    simple_query,
    spatial_query,
    superlative_query,
)
from .description_generator import DescriptionGenerator, generate_descriptions

# Import parser from parsing subpackage
from .parsing import QueryParser, parse_query
from .point_feature_extractor import (
    PointFeatureConfig,
    PointFeatureExtractor,
    PointFeatureIndex,
    compute_scene_point_features,
)
from .query_executor import QueryExecutor, execute_query
from .query_pipeline import QueryScenePipeline, run_query, run_query_with_dataset
from .quick_filters import (
    QUICK_FILTER_CONFIGS,
    AttributeFilter,
    FilterConfig,
    FilterType,
    QuickFilters,
    get_supported_quick_filters,
    has_quick_filter,
    quick_filter,
)

# Import from retrieval subpackage
from .retrieval import (
    CLIPIndex,
    KeyframeSelector,  # noqa: F401
    PointLevelIndex,
    RegionIndex,
    RelationResult,
    SceneIndices,
    SceneObject,  # noqa: F401
    SpatialIndex,
    SpatialRelationChecker,
    VisibilityIndex,
)
from .retrieval import (
    KeyframeResult as RetrievalKeyframeResult,  # noqa: F401
)
from .scene_representation import QuerySceneRepresentation
from .spatial_relations import (
    RELATION_ALIASES,
    check_relation,
    get_canonical_relation,
)
from .utils import (
    annotate_bev_with_distances,
    annotate_image_with_objects,
    crop_object_from_image,
    project_3d_bbox_to_2d,
    project_point_to_image,
)
from .vlm_interface import STRATEGY_MAP, VLMInput, VLMInputConstructor, VLMOutputParser

__version__ = "0.1.0"
__all__ = [
    # Core data structures
    "ObjectNode",
    "ObjectDescriptions",
    "RegionNode",
    "ViewScore",
    "GroundingResult",
    "QueryInfo",
    "QueryType",
    "BoundingBox3D",
    "CameraPose",
    # Core results
    "KeyframeResult",
    "ExecutionResult",
    "EvidenceBundle",
    # Scene
    "QuerySceneRepresentation",
    # Pipeline
    "QueryScenePipeline",
    "run_query",
    "run_query_with_dataset",
    # Query structures (nested spatial queries)
    "GroundingQuery",
    "QueryNode",
    "SpatialConstraint",
    "SelectConstraint",
    "ConstraintType",
    "SpatialRelation",
    "HypothesisKind",
    "ParseMode",
    "QueryHypothesis",
    "HypothesisOutputV1",
    "SUPPORTED_RELATIONS",
    "SUPPORTED_RELATIONS_STR",
    "simple_query",
    "spatial_query",
    "superlative_query",
    # Query parser
    "QueryParser",
    "parse_query",
    # Query executor
    "QueryExecutor",
    "ExecutionResult",
    "execute_query",
    # Spatial relations
    "SpatialRelationChecker",
    "RelationResult",
    "RELATION_ALIASES",
    "check_relation",
    "get_canonical_relation",
    # Quick filters
    "QuickFilters",
    "AttributeFilter",
    "FilterType",
    "FilterConfig",
    "QUICK_FILTER_CONFIGS",
    "quick_filter",
    "has_quick_filter",
    "get_supported_quick_filters",
    # Indices (hierarchical)
    "CLIPIndex",
    "VisibilityIndex",
    "SpatialIndex",
    "RegionIndex",
    "PointLevelIndex",
    "SceneIndices",
    # OpenScene-style point features
    "PointFeatureExtractor",
    "PointFeatureIndex",
    "PointFeatureConfig",
    "compute_scene_point_features",
    # LSeg dense features
    "LSegFeatureExtractor",
    "DensePointFeatureExtractor",
    "LSegConfig",
    "extract_dense_scene_features",
    # VLM
    "VLMInputConstructor",
    "VLMOutputParser",
    "VLMInput",
    "STRATEGY_MAP",
    # Description generation
    "DescriptionGenerator",
    "generate_descriptions",
    # Utils
    "project_point_to_image",
    "project_3d_bbox_to_2d",
    "crop_object_from_image",
    "annotate_image_with_objects",
    "annotate_bev_with_distances",
]
