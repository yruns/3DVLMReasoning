"""Query parsing structures and prompt templates.

This module provides AST-like structures and prompt templates for converting
natural language queries into structured executable query representations.
"""

from __future__ import annotations

# Re-export core structures from core.hypotheses
from ..core.hypotheses import (
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
)


def get_system_prompt() -> str:
    """
    Get the system prompt for query parsing.

    This prompt instructs the LLM to parse natural language queries into
    the structured HypothesisOutputV1 format.

    Returns:
        System prompt string with full parsing instructions
    """
    return f"""You are a spatial query parser for 3D scene understanding.
Your task is to parse natural language queries into a structured HypothesisOutputV1 JSON format.

The output must be a valid HypothesisOutputV1 with the following structure:
- format_version: Always "hypothesis_output_v1"
- parse_mode: "single" or "multi" (see decision rules below)
- hypotheses: List of QueryHypothesis objects (1-3 hypotheses)

Each QueryHypothesis has:
- kind: "direct", "proxy", or "context"
- rank: 1-based priority (1=highest priority)
- grounding_query: A GroundingQuery object
- lexical_hints: List of free-form hints (synonyms, paraphrases) from the query

GroundingQuery structure:
- raw_query: The original query text
- root: A QueryNode representing the target object
- expect_unique: True if the query uses "the" (singular), False otherwise

Each QueryNode has:
- categories: LIST of object types (MUST be EXACT strings from SCENE CATEGORIES, or ["UNKNOW"] if no match)
- attributes: List of adjective attributes like "red", "large", "wooden"
- spatial_constraints: List of spatial relations to other objects (filter phase, AND logic)
- select_constraint: Optional selection like "nearest", "largest", "second" (select phase)

SpatialConstraint structure:
- relation: PREFERRED to be one of these predefined values: {SUPPORTED_RELATIONS_STR}
  (Map synonyms: "on top of"→"on", "under"→"below", "close to"→"near")
- anchors: List of reference QueryNode objects (1 for most relations, 2 for "between")

SelectConstraint structure (for superlative/ordinal):
- constraint_type: "superlative" or "ordinal"
- metric: "distance", "size", "height", "x_position", etc.
- order: "min" (nearest/smallest), "max" (farthest/largest), "asc", "desc"
- reference: QueryNode for distance reference (e.g., "nearest the door" -> door)
- position: Integer for ordinal (1=first, 2=second, etc.)

=== PARSE MODE DECISION RULES ===

Use parse_mode="single" when:
1. Target AND all anchors/references exist in SCENE CATEGORIES
2. Even with semantic expansion (pillow → [pillow, throw_pillow]), if all categories are in scene

Use parse_mode="multi" when:
1. Target category is NOT in scene → output ["UNKNOW"] and add PROXY/CONTEXT fallback
2. Any anchor/reference category is NOT in scene → output ["UNKNOW"] in that anchor and add PROXY fallback
3. Query is ambiguous and may need fallback strategies

=== HYPOTHESIS KIND RULES ===

DIRECT hypothesis (always rank=1):
- Parse the query literally
- Use ["UNKNOW"] for any category not found in scene
- Keep original spatial constraints and select constraints

PROXY hypothesis (rank=2, only in multi mode):
- Created when target or anchor is UNKNOW
- For missing ANCHOR: replace UNKNOW anchor with semantically similar categories from scene
  (e.g., if "bed" is missing but target is "pillow", try "sofa" or "armchair" as proxy anchor)
- For missing TARGET: replace UNKNOW target with related categories from scene

CONTEXT hypothesis (rank=3, used as last resort):
- Created when both direct and proxy may fail
- Remove all spatial constraints and select constraints
- Replace target with the anchor categories (find the context/scene objects)
- Set expect_unique=false

=== CATEGORY RULES ===

1. SEMANTIC EXPANSION (CRITICAL): The `categories` field is a LIST. When the user mentions a general term,
   include ALL semantically related categories from SCENE CATEGORIES:
   - Query "a pillow" with scene [door, pillow, throw_pillow, sofa] → categories: ["pillow", "throw_pillow"]
   - Query "the lamp" with scene [floor_lamp, table_lamp, sofa] → categories: ["floor_lamp", "table_lamp"]

2. Every category MUST be EXACT string from SCENE CATEGORIES (case-sensitive, keep underscores).

3. If no suitable category exists in SCENE CATEGORIES, output ["UNKNOW"].

4. This applies to ALL QueryNode objects: root, anchors in spatial_constraints, references in select_constraint.

5. Map common relation synonyms: "on top of"→"on", "under"/"beneath"→"below", "close to"→"near"

6. "nearest/closest X" uses SelectConstraint with metric="distance", order="min", reference=X

7. "largest/biggest" uses SelectConstraint with metric="size", order="max", reference=null

=== VISUAL CONTEXT (if image provided) ===

When a Bird's Eye View (BEV) image is provided:
- Each object is shown as a labeled circle at its centroid position
- Labels follow format "NNN: category" (e.g., "001: sofa", "002: pillow")
- Object IDs correspond to the SCENE CATEGORIES list
- Use this visual context to understand spatial relationships
- Reference the image to verify "near", "on", "between" relationships
- Use object positions to resolve ambiguous references
- Note: In the BEV, Y-axis increases downward (image coordinates)"""


def get_few_shot_examples() -> str:
    """
    Get few-shot examples for the parser.

    Returns:
        String containing formatted examples for in-context learning
    """
    return """
EXAMPLES:

=== EXAMPLE 1: Simple query - target and anchor both exist (SINGLE mode) ===
Query: "the pillow on the sofa" (scene has: pillow, throw_pillow, sofa, door)
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "single",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the pillow on the sofa",
        "root": {
          "categories": ["pillow", "throw_pillow"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["pillow", "sofa"]
    }
  ]
}

=== EXAMPLE 2: Superlative with reference (SINGLE mode) ===
Query: "the sofa nearest the door" (scene has: sofa, door, window)
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "single",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the sofa nearest the door",
        "root": {
          "categories": ["sofa"],
          "attributes": [],
          "spatial_constraints": [],
          "select_constraint": {
            "constraint_type": "superlative",
            "metric": "distance",
            "order": "min",
            "reference": {"categories": ["door"], "attributes": [], "spatial_constraints": [], "select_constraint": null},
            "position": null
          }
        },
        "expect_unique": true
      },
      "lexical_hints": ["sofa", "door", "nearest"]
    }
  ]
}

=== EXAMPLE 3: Missing anchor - "bed" not in scene (MULTI mode with PROXY) ===
Query: "the pillow on the bed" (scene has: pillow, throw_pillow, sofa, armchair, door - NO bed)
NOTE: "bed" is NOT in scene, so anchor uses ["UNKNOW"]. Add PROXY hypothesis with "sofa" as proxy anchor.
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "multi",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the pillow on the bed",
        "root": {
          "categories": ["pillow", "throw_pillow"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["UNKNOW"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["pillow", "bed"]
    },
    {
      "kind": "proxy",
      "rank": 2,
      "grounding_query": {
        "raw_query": "proxy for: the pillow on the bed",
        "root": {
          "categories": ["pillow", "throw_pillow"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["sofa", "armchair"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["proxy_anchor"]
    }
  ]
}

=== EXAMPLE 4: Missing target - "laptop" not in scene (MULTI mode with PROXY and CONTEXT) ===
Query: "the laptop on the table" (scene has: book, cup, side_table, coffee_table, chair - NO laptop)
NOTE: "laptop" is NOT in scene. PROXY tries related objects like "book". CONTEXT falls back to anchor.
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "multi",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the laptop on the table",
        "root": {
          "categories": ["UNKNOW"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["side_table", "coffee_table"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["laptop", "table"]
    },
    {
      "kind": "proxy",
      "rank": 2,
      "grounding_query": {
        "raw_query": "proxy for: the laptop on the table",
        "root": {
          "categories": ["book", "cup"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["side_table", "coffee_table"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["proxy"]
    },
    {
      "kind": "context",
      "rank": 3,
      "grounding_query": {
        "raw_query": "context for: the laptop on the table",
        "root": {
          "categories": ["side_table", "coffee_table"],
          "attributes": [],
          "spatial_constraints": [],
          "select_constraint": null
        },
        "expect_unique": false
      },
      "lexical_hints": ["context"]
    }
  ]
}

=== EXAMPLE 5: Semantic expansion only, all exist (SINGLE mode) ===
Query: "the cushion on the couch" (scene has: sofa, sofa_seat_cushion, pillow, throw_pillow, door)
NOTE: "cushion" expands to all cushion-like categories; "couch" maps to "sofa". All exist → SINGLE mode.
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "single",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the cushion on the couch",
        "root": {
          "categories": ["sofa_seat_cushion", "pillow", "throw_pillow"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["cushion", "couch"]
    }
  ]
}
"""


def validate_parsed_output(
    output: HypothesisOutputV1,
    scene_categories: list[str] | None = None,
) -> list[str]:
    """
    Validate a parsed output for consistency.

    This validates the structural integrity of the parsed output and
    optionally checks that all categories exist in the scene.

    Args:
        output: The parsed output to validate
        scene_categories: Optional list of valid categories (excluding "UNKNOW")

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if not output.hypotheses:
        errors.append("No hypotheses generated")
        return errors

    for i, hyp in enumerate(output.hypotheses):
        # Check rank consistency
        if hyp.rank < 1:
            errors.append(f"Hypothesis {i}: rank must be >= 1")

        # Check that direct hypothesis is rank 1
        if hyp.kind == "direct" and hyp.rank != 1:
            errors.append(f"Hypothesis {i}: DIRECT hypothesis must be rank=1")

        # Check parse mode consistency
        if output.parse_mode == "single" and hyp.kind != "direct":
            errors.append(
                f"Hypothesis {i}: SINGLE mode should only have DIRECT hypotheses"
            )

        if output.parse_mode == "multi" and hyp.kind == "direct" and hyp.rank != 1:
            errors.append(
                f"Hypothesis {i}: MULTI mode DIRECT hypothesis must be rank=1"
            )

        # Check grounding_query.root is not None
        if hyp.grounding_query.root is None:
            errors.append(f"Hypothesis {i}: grounding_query.root is None")
            continue

        # Validate categories if provided
        if scene_categories:
            all_cats = _collect_all_categories(hyp.grounding_query.root)
            for cat in all_cats:
                if cat != "UNKNOW" and cat not in scene_categories:
                    errors.append(
                        f"Hypothesis {i}: Unknown category '{cat}' not in scene"
                    )

    return errors


def _collect_all_categories(node: QueryNode) -> list[str]:
    """Recursively collect all categories from a QueryNode tree."""
    cats = list(node.categories)

    for constraint in node.spatial_constraints:
        for anchor in constraint.anchors:
            cats.extend(_collect_all_categories(anchor))

    if node.select_constraint and node.select_constraint.reference:
        cats.extend(_collect_all_categories(node.select_constraint.reference))

    return cats


__all__ = [
    # Structures (re-exported from core)
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
    # Prompt templates
    "get_system_prompt",
    "get_few_shot_examples",
    # Validation
    "validate_parsed_output",
]
