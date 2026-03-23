"""Query parsing module.

This module provides a complete query parsing system for converting natural
language spatial queries into structured executable representations.

Components:
- structures: AST-like data structures and prompt templates
- parser: QueryParser class for actual parsing using LLM

Usage:
    from query_scene.parsing import QueryParser, HypothesisOutputV1

    parser = QueryParser(
        llm_model="gpt-5.2-2025-12-11",
        scene_categories=["sofa", "pillow", "door"]
    )
    result = parser.parse("the pillow on the sofa")
"""

from .parser import QueryParser, parse_query
from .structures import (
    SUPPORTED_RELATIONS,
    SUPPORTED_RELATIONS_STR,
    ConstraintType,
    GroundingQuery,
    HypothesisKind,
    HypothesisOutputV1,
    ParseMode,
    QueryHypothesis,
    QueryNode,
    SelectConstraint,
    SpatialConstraint,
    SpatialRelation,
    get_few_shot_examples,
    get_system_prompt,
    validate_parsed_output,
)

__all__ = [
    # Parser
    "QueryParser",
    "parse_query",
    # Structures
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
    "SUPPORTED_RELATIONS",
    "SUPPORTED_RELATIONS_STR",
    # Utilities
    "get_system_prompt",
    "get_few_shot_examples",
    "validate_parsed_output",
]
