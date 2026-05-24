"""
Nested Spatial Query Structures.

This module defines Pydantic models for representing nested spatial queries
that support arbitrary depth of spatial constraints and selection operations.

Example query: "the pillow on the sofa nearest the door"
Parsed structure:
    QueryNode(
        category="pillow",
        spatial_constraints=[
            SpatialConstraint(
                relation="on",
                anchors=[
                    QueryNode(
                        category="sofa",
                        select_constraint=SelectConstraint(
                            constraint_type=ConstraintType.SUPERLATIVE,
                            metric="distance",
                            order="min",
                            reference=QueryNode(category="door")
                        )
                    )
                ]
            )
        ]
    )
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ConstraintType(str, Enum):
    """Type of selection constraint."""

    # Superlative: nearest, largest, highest, smallest
    SUPERLATIVE = "superlative"

    # Comparative: closer than, larger than
    COMPARATIVE = "comparative"

    # Ordinal: first, second, third
    ORDINAL = "ordinal"


class SpatialRelation(str, Enum):
    """
    Predefined spatial relations for scene understanding.

    Relations are categorized by whether they support quick coordinate-based filtering:

    VIEW-INDEPENDENT (supports quick filtering):
    =============================================
    These relations can be evaluated using simple coordinate comparisons
    because they don't depend on the observer's viewpoint.

    Vertical Relations (Z-axis) - gravity defines "up":
    - ON: target.z > anchor.z (target is on top of anchor)
    - ABOVE: target.z > anchor.z (target is above anchor)
    - BELOW: target.z < anchor.z (target is below anchor)

    Distance Relations - Euclidean distance is view-independent:
    - NEAR: distance(target, anchor) < threshold
    - NEXT_TO: distance(target, anchor) < threshold (stricter)
    - BESIDE: similar to NEXT_TO

    VIEW-DEPENDENT (requires full spatial reasoning):
    =================================================
    These relations depend on the observer's viewpoint and CANNOT be
    filtered using simple world-coordinate comparisons.

    Horizontal Relations - depend on viewing direction:
    - LEFT_OF: requires knowing observer's facing direction
    - RIGHT_OF: requires knowing observer's facing direction
    - IN_FRONT_OF: requires knowing observer's position
    - BEHIND: requires knowing observer's position

    Complex Relations - require geometric reasoning:
    - INSIDE: target is within anchor's bounding box
    - BETWEEN: target is between two anchors
    """

    # ===== VIEW-INDEPENDENT: Vertical relations (Z-axis) =====
    # Safe for quick filtering - gravity defines "up" universally
    ON = "on"
    ABOVE = "above"
    BELOW = "below"

    # ===== VIEW-INDEPENDENT: Distance relations =====
    # Safe for quick filtering - Euclidean distance is view-independent
    NEAR = "near"
    NEXT_TO = "next_to"
    BESIDE = "beside"

    # ===== VIEW-DEPENDENT: Horizontal relations =====
    # NOT safe for quick filtering - require observer viewpoint
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"

    # ===== COMPLEX: Containment/Multi-object =====
    # NOT safe for quick filtering - require full geometric reasoning
    INSIDE = "inside"
    BETWEEN = "between"

    @classmethod
    def from_string(cls, s: str) -> SpatialRelation | None:
        """
        Convert a string to SpatialRelation, handling aliases.

        Returns None if the relation is not in the predefined list,
        meaning quick filtering cannot be applied.

        Examples:
            SpatialRelation.from_string("on top of") -> SpatialRelation.ON
            SpatialRelation.from_string("under") -> SpatialRelation.BELOW
            SpatialRelation.from_string("hanging from") -> None  # Not predefined
        """
        if not s:
            return None

        # Normalize
        s_lower = s.lower().strip().replace(" ", "_")

        # Direct match
        try:
            return cls(s_lower)
        except ValueError:
            pass

        # Alias mapping
        aliases = {
            # ON variants
            "on_top_of": cls.ON,
            "upon": cls.ON,
            "atop": cls.ON,
            "resting_on": cls.ON,
            # ABOVE variants
            "over": cls.ABOVE,
            "higher_than": cls.ABOVE,
            # BELOW variants
            "under": cls.BELOW,
            "beneath": cls.BELOW,
            "underneath": cls.BELOW,
            "lower_than": cls.BELOW,
            # LEFT_OF variants
            "left": cls.LEFT_OF,
            "to_the_left_of": cls.LEFT_OF,
            # RIGHT_OF variants
            "right": cls.RIGHT_OF,
            "to_the_right_of": cls.RIGHT_OF,
            # IN_FRONT_OF variants
            "front": cls.IN_FRONT_OF,
            "facing": cls.IN_FRONT_OF,
            # BEHIND variants
            "back": cls.BEHIND,
            "back_of": cls.BEHIND,
            "in_back_of": cls.BEHIND,
            # NEAR variants
            "close_to": cls.NEAR,
            "nearby": cls.NEAR,
            # NEXT_TO variants
            "adjacent_to": cls.NEXT_TO,
            "adjacent": cls.NEXT_TO,
            # INSIDE variants
            "in": cls.INSIDE,
            "within": cls.INSIDE,
            "contained_in": cls.INSIDE,
            # BETWEEN variants
            "in_between": cls.BETWEEN,
        }

        if s_lower in aliases:
            return aliases[s_lower]

        # Unknown relation - return None (no quick filter available)
        return None

    def is_view_dependent(self) -> bool:
        """
        Check if this relation is view-dependent.

        View-dependent relations (left, right, front, behind) require
        knowing the observer's viewpoint and cannot be filtered using
        simple world-coordinate comparisons.
        """
        return self in [
            SpatialRelation.LEFT_OF,
            SpatialRelation.RIGHT_OF,
            SpatialRelation.IN_FRONT_OF,
            SpatialRelation.BEHIND,
        ]

    def supports_quick_filter(self) -> bool:
        """
        Check if this relation supports quick coordinate-based filtering.

        Only VIEW-INDEPENDENT relations support quick filtering:
        - Vertical (on, above, below): gravity defines "up"
        - Distance (near, next_to, beside): Euclidean distance is view-independent

        View-dependent relations (left, right, front, behind) and
        complex relations (inside, between) require full spatial reasoning.
        """
        # Only view-independent relations support quick filtering
        return self in [
            # Vertical relations
            SpatialRelation.ON,
            SpatialRelation.ABOVE,
            SpatialRelation.BELOW,
            # Distance relations
            SpatialRelation.NEAR,
            SpatialRelation.NEXT_TO,
            SpatialRelation.BESIDE,
        ]

    def get_filter_type(self) -> str | None:
        """
        Get the type of quick filter for this relation.

        Returns None for view-dependent or complex relations that
        don't support quick filtering.
        """
        if self in [SpatialRelation.ON, SpatialRelation.ABOVE, SpatialRelation.BELOW]:
            return "vertical"
        elif self in [
            SpatialRelation.NEAR,
            SpatialRelation.NEXT_TO,
            SpatialRelation.BESIDE,
        ]:
            return "distance"
        # View-dependent and complex relations don't have quick filters
        # LEFT_OF, RIGHT_OF, IN_FRONT_OF, BEHIND, INSIDE, BETWEEN
        return None


# List of supported relations for prompt
SUPPORTED_RELATIONS = [r.value for r in SpatialRelation]
SUPPORTED_RELATIONS_STR = ", ".join(SUPPORTED_RELATIONS)


# Relations whose semantics depend on the speaker / observer / object frame.
# When the parser routes one of these into a non-world reference_frame, the
# executor MUST honour the configured execution_policy (see ExecutionPolicy).
DIRECTIONAL_RELATIONS: frozenset[str] = frozenset(
    {
        SpatialRelation.LEFT_OF.value,
        SpatialRelation.RIGHT_OF.value,
        SpatialRelation.IN_FRONT_OF.value,
        SpatialRelation.BEHIND.value,
    }
)


class ReferenceFrame(str, Enum):
    """Reference frame in which a spatial / select constraint is evaluated.

    Defaults to WORLD which reproduces the pre-viewpoint global X/Y/Z
    semantics. VIEWER and OBJECT_LOCAL require a viewpoint context id.
    AMBIGUOUS is set by the parser-normalize layer when a directional
    relation is detected without an explicit observer cue, so the executor
    can route it to a non-filtering policy without inventing a viewer pose.
    """

    WORLD = "world"
    VIEWER = "viewer"
    OBJECT_LOCAL = "object_local"
    AMBIGUOUS = "ambiguous"


class ViewpointKind(str, Enum):
    """Linguistic frame of a viewpoint context.

    - FACING_ANCHOR / FACING_ANCHOR_SET: "facing the X", "facing the windows"
    - ENTERING_FROM: "entering from the door"
    - STANDING_AT: "standing at the foot of the bed"
    - OBJECT_LOCAL: object_A facing object_B (no human observer involved)
    """

    FACING_ANCHOR = "facing_anchor"
    FACING_ANCHOR_SET = "facing_anchor_set"
    ENTERING_FROM = "entering_from"
    STANDING_AT = "standing_at"
    OBJECT_LOCAL = "object_local"


class ExecutionPolicy(str, Enum):
    """How the executor treats a failing constraint.

    - HARD: legacy behavior; if filter empties the candidates, return []
    - SOFT: keep all candidates, multiply per-candidate score by satisfaction
    - RANK_ONLY: never filter; record per-candidate score in metadata as
      ``soft_match_score`` for downstream rerankers (e.g. joint_coverage)
    """

    HARD = "hard"
    SOFT = "soft"
    RANK_ONLY = "rank_only"


class ViewpointContext(BaseModel):
    """A named viewpoint description referenced by spatial / select constraints.

    Anchors are full QueryNode objects so the executor can resolve them with
    the same category-expansion / multilabel index logic as any other anchor.
    Exactly one of ``facing_anchor``, ``origin_anchor`` and ``subject_anchor``
    should be populated for a well-formed context, but no hard pydantic check
    is enforced - the parser-normalize layer downgrades unresolved contexts
    to a non-filtering policy rather than refusing to validate.
    """

    id: str = Field(
        ..., min_length=1, description="Unique id referenced by *_context_id fields"
    )
    kind: ViewpointKind = Field(..., description="Linguistic frame this context represents")
    facing_anchor: QueryNode | None = Field(
        default=None,
        description="Anchor for FACING_ANCHOR / FACING_ANCHOR_SET kinds",
    )
    origin_anchor: QueryNode | None = Field(
        default=None,
        description="Anchor for ENTERING_FROM / STANDING_AT kinds",
    )
    subject_anchor: QueryNode | None = Field(
        default=None,
        description="Subject object for OBJECT_LOCAL kind (e.g. armchair in 'armchair facing couch')",
    )
    raw_phrase: str = Field(
        default="",
        description="Original natural-language fragment, for trace / debug only",
    )
    confidence: Literal["explicit", "inferred", "ambiguous"] = Field(
        default="explicit",
        description="explicit = literal cue in query; inferred = parser deduced; ambiguous = parser undecided",
    )


class QueryNode(BaseModel):

    categories: list[str] = Field(
        ...,
        min_length=1,
        description="Object categories to search for. Include ALL semantically related "
        "categories from scene. E.g., for 'pillow' query with scene containing "
        "[pillow, throw_pillow], return ['pillow', 'throw_pillow']",
    )

    attributes: list[str] = Field(
        default_factory=list,
        description="Attribute filters like 'red', 'large', 'wooden'",
    )

    spatial_constraints: list[SpatialConstraint] = Field(
        default_factory=list,
        description="List of spatial constraints (AND logic between them)",
    )

    select_constraint: SelectConstraint | None = Field(
        default=None,
        description="Selection constraint like 'nearest', 'largest', 'second'",
    )

    open_ended: bool = Field(
        default=False,
        description="True when the query asks 'what is near/behind/on X?' with "
        "an unknown target. When open_ended=True, categories=['UNKNOW'] and "
        "spatial_constraints define the anchor. The executor returns ALL objects "
        "matching the spatial relation, not a specific category.",
    )

    node_id: str = Field(
        default="", description="Unique identifier for tracking during execution"
    )

    @property
    def category(self) -> str:
        """Primary category (first in list). For backward compatibility."""
        return self.categories[0] if self.categories else ""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "categories": ["pillow", "throw_pillow"],
                    "attributes": ["red"],
                    "spatial_constraints": [],
                    "select_constraint": None,
                    "node_id": "target_pillow",
                }
            ]
        }
    }


class SpatialConstraint(BaseModel):

    relation: str = Field(
        ..., description=f"Spatial relation. MUST be one of: {SUPPORTED_RELATIONS_STR}"
    )

    anchors: list[QueryNode] = Field(
        ..., description="Reference objects. Usually 1, can be 2 for 'between'"
    )

    reference_frame: ReferenceFrame = Field(
        default=ReferenceFrame.WORLD,
        description="Frame in which this relation is evaluated. Default world reproduces legacy behavior.",
    )

    viewpoint_context_id: str | None = Field(
        default=None,
        description="Id of a ViewpointContext on the enclosing GroundingQuery. Required when "
        "reference_frame is VIEWER or OBJECT_LOCAL with a subject anchor.",
    )

    execution_policy: ExecutionPolicy = Field(
        default=ExecutionPolicy.HARD,
        description="How a failing constraint is handled. HARD = legacy filter-to-empty; "
        "SOFT keeps candidates with scaled score; RANK_ONLY records score in metadata.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"relation": "on", "anchors": [{"category": "table"}]}]
        }
    }

    @property
    def relation_enum(self) -> SpatialRelation | None:
        """
        Get the normalized SpatialRelation enum.

        Converts LLM output like "on top of" to SpatialRelation.ON.
        Returns None if the relation is not in the predefined list,
        meaning this constraint cannot use quick coordinate-based filtering.

        Examples:
            "on top of" -> SpatialRelation.ON
            "near" -> SpatialRelation.NEAR
            "hanging from" -> None (not predefined, no quick filter)
        """
        return SpatialRelation.from_string(self.relation)

    @property
    def supports_quick_filter(self) -> bool:
        """
        Check if this relation supports quick coordinate-based filtering.

        Returns True only if the relation is in the predefined list.
        For unknown/custom relations, returns False and the system
        will skip quick filtering and use full spatial relation checking.
        """
        rel = self.relation_enum
        return rel is not None and rel.supports_quick_filter()

    @property
    def filter_type(self) -> str | None:
        """
        Get the filter type (vertical, horizontal, distance, containment).

        Returns None if the relation is not predefined.
        """
        rel = self.relation_enum
        return rel.get_filter_type() if rel is not None else None


class SelectConstraint(BaseModel):

    constraint_type: ConstraintType = Field(
        ...,
        description="Type: superlative (nearest/largest), comparative, or ordinal (first/second)",
    )

    metric: str = Field(
        ...,
        description="Metric to compare: 'distance', 'size', 'height', 'x_position', 'y_position'",
    )

    order: str = Field(
        ...,
        description="Order: 'min' (nearest/smallest), 'max' (farthest/largest), 'asc', 'desc'",
    )

    reference: QueryNode | None = Field(
        default=None,
        description="Reference object for distance comparisons (e.g., door for 'nearest the door')",
    )

    position: int | None = Field(
        default=None,
        description="Position for ordinal selection: 1=first, 2=second, etc.",
    )

    reference_frame: ReferenceFrame = Field(
        default=ReferenceFrame.WORLD,
        description="Frame in which the metric axis is evaluated. Default world reproduces "
        "legacy semantics (x_position = global X, etc.).",
    )

    viewpoint_context_id: str | None = Field(
        default=None,
        description="Id of a ViewpointContext on the enclosing GroundingQuery. Required when "
        "reference_frame is VIEWER or OBJECT_LOCAL.",
    )

    execution_policy: ExecutionPolicy = Field(
        default=ExecutionPolicy.HARD,
        description="How a failing selection is handled. HARD reproduces legacy behavior.",
    )

    @model_validator(mode="after")
    def validate_constraint(self) -> SelectConstraint:
        """Validate that ordinal constraints have position set."""
        if self.constraint_type == ConstraintType.ORDINAL and self.position is None:
            raise ValueError("Ordinal constraints require 'position' to be set")
        if (
            self.constraint_type == ConstraintType.SUPERLATIVE
            and self.metric == "distance"
            and self.reference is None
        ):
            # This is actually OK - could be "nearest" without explicit reference
            pass
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "constraint_type": "superlative",
                    "metric": "distance",
                    "order": "min",
                    "reference": {"category": "door"},
                    "position": None,
                }
            ]
        }
    }


class GroundingQuery(BaseModel):
    """
    Complete grounding query representation.

    This is the top-level structure returned by the query parser.

    Attributes:
        raw_query: Original natural language query
        root: Root query node (the target object to find)
        expect_unique: Whether to expect a single result (True for "the X", False for "X" or "Xs")
    """

    raw_query: str = Field(default="", description="Original natural language query")

    root: QueryNode = Field(
        ..., description="Root query node representing the target object"
    )

    expect_unique: bool = Field(
        default=True,
        description="True for 'the X' (expect single result), False for 'X' or 'Xs'",
    )

    viewpoint_contexts: list[ViewpointContext] = Field(
        default_factory=list,
        description="Viewpoint contexts referenced by constraints in this query. Empty list "
        "preserves legacy world-frame behavior.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "raw_query": "the pillow on the sofa",
                    "root": {
                        "categories": ["pillow", "throw_pillow"],
                        "spatial_constraints": [
                            {"relation": "on", "anchors": [{"categories": ["sofa"]}]}
                        ],
                    },
                    "expect_unique": True,
                }
            ]
        }
    }

    def get_all_categories(self) -> list[str]:
        """Extract all object categories mentioned in the query."""
        categories: list[str] = []
        self._collect_categories(self.root, categories)
        for vc in self.viewpoint_contexts:
            for anchor in (vc.facing_anchor, vc.origin_anchor, vc.subject_anchor):
                if anchor is not None:
                    self._collect_categories(anchor, categories)
        return categories

    def _collect_categories(self, node: QueryNode, categories: list[str]) -> None:
        """Recursively collect categories from a node."""
        categories.extend(node.categories)

        for constraint in node.spatial_constraints:
            for anchor in constraint.anchors:
                self._collect_categories(anchor, categories)

        if node.select_constraint and node.select_constraint.reference:
            self._collect_categories(node.select_constraint.reference, categories)

    def iter_constraints(
        self,
    ) -> Iterable[tuple[QueryNode, SpatialConstraint | SelectConstraint]]:
        """Yield every constraint paired with its owning node (for executor / linters)."""
        stack: list[QueryNode] = [self.root]
        while stack:
            node = stack.pop()
            for sc in node.spatial_constraints:
                yield node, sc
                for anchor in sc.anchors:
                    stack.append(anchor)
            if node.select_constraint is not None:
                yield node, node.select_constraint
                if node.select_constraint.reference is not None:
                    stack.append(node.select_constraint.reference)

    @model_validator(mode="after")
    def validate_viewpoint_context_references(self) -> GroundingQuery:
        """Every viewpoint_context_id referenced by a constraint must resolve.

        Also requires that VIEWER / OBJECT_LOCAL reference_frame come with a
        non-null viewpoint_context_id. AMBIGUOUS / WORLD allow null.
        """
        known_ids = {vc.id for vc in self.viewpoint_contexts}
        if len({vc.id for vc in self.viewpoint_contexts}) != len(self.viewpoint_contexts):
            raise ValueError("viewpoint_contexts must have unique ids")

        non_world_frames = {ReferenceFrame.VIEWER, ReferenceFrame.OBJECT_LOCAL}

        for _node, constraint in self.iter_constraints():
            frame = constraint.reference_frame
            ctx_id = constraint.viewpoint_context_id

            if ctx_id is not None and ctx_id not in known_ids:
                raise ValueError(
                    f"viewpoint_context_id {ctx_id!r} is not declared in viewpoint_contexts"
                )

            if frame in non_world_frames and ctx_id is None:
                raise ValueError(
                    f"reference_frame={frame.value} requires a viewpoint_context_id"
                )

        return self


class HypothesisKind(str, Enum):
    """Type of hypothesis for open-world query parsing."""

    DIRECT = "direct"
    PROXY = "proxy"
    CONTEXT = "context"


class ParseMode(str, Enum):
    """Parsing mode for hypothesis output."""

    SINGLE = "single"
    MULTI = "multi"


class QueryHypothesis(BaseModel):
    """One executable hypothesis parsed from user query."""

    kind: HypothesisKind = Field(...)
    rank: int = Field(..., ge=1, description="1-based priority rank")
    grounding_query: GroundingQuery = Field(...)
    lexical_hints: list[str] = Field(
        default_factory=list,
        description="Free-form lexical hints (synonyms/paraphrases), not executable categories",
    )


class HypothesisOutputV1(BaseModel):
    """
    Unified query parsing output for keyframe selection.

    This is the canonical structured format consumed by KeyframeSelector.
    It supports both deterministic single-result parsing and multi-hypothesis
    parsing for open-world fallback.
    """

    format_version: Literal["hypothesis_output_v1"] = "hypothesis_output_v1"
    parse_mode: ParseMode = Field(...)
    hypotheses: list[QueryHypothesis] = Field(..., min_length=1, max_length=3)

    @model_validator(mode="after")
    def validate_hypothesis_output(self) -> HypothesisOutputV1:
        """Validate rank uniqueness and parse-mode consistency."""
        ranks = [h.rank for h in self.hypotheses]
        if len(ranks) != len(set(ranks)):
            raise ValueError("Hypothesis ranks must be unique")
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ValueError("Hypothesis ranks must be contiguous and start from 1")

        if self.parse_mode == ParseMode.SINGLE:
            if len(self.hypotheses) != 1:
                raise ValueError("parse_mode='single' requires exactly one hypothesis")
            if self.hypotheses[0].kind != HypothesisKind.DIRECT:
                raise ValueError(
                    "parse_mode='single' requires hypothesis kind='direct'"
                )

        return self

    def ordered_hypotheses(self) -> list[QueryHypothesis]:
        """Return hypotheses sorted by rank ascending."""
        return sorted(self.hypotheses, key=lambda x: x.rank)

    def validate_categories(self, scene_categories: Iterable[str]) -> None:
        """
        Validate that every executable category is in scene categories or UNKNOW.
        """
        scene_set = set(scene_categories)
        for hypothesis in self.hypotheses:
            for cat in hypothesis.grounding_query.get_all_categories():
                if cat != "UNKNOW" and cat not in scene_set:
                    raise ValueError(
                        f"Category '{cat}' is not in scene categories and is not UNKNOW"
                    )

    def validate_no_mask_leak(self, hidden_categories: Iterable[str]) -> None:
        """
        Validate that hidden categories do not appear in executable hypotheses.
        """
        hidden_set = set(hidden_categories)
        if not hidden_set:
            return
        for hypothesis in self.hypotheses:
            for cat in hypothesis.grounding_query.get_all_categories():
                if cat in hidden_set:
                    raise ValueError(f"Masked category leak detected: '{cat}'")

    @classmethod
    def from_direct_query(cls, grounding_query: GroundingQuery) -> HypothesisOutputV1:
        """Build a single-direct output from one grounding query."""
        return cls(
            parse_mode=ParseMode.SINGLE,
            hypotheses=[
                QueryHypothesis(
                    kind=HypothesisKind.DIRECT,
                    rank=1,
                    grounding_query=grounding_query,
                )
            ],
        )


# Rebuild models to resolve forward references
QueryNode.model_rebuild()
SpatialConstraint.model_rebuild()
SelectConstraint.model_rebuild()
ViewpointContext.model_rebuild()
GroundingQuery.model_rebuild()
QueryHypothesis.model_rebuild()
HypothesisOutputV1.model_rebuild()


# Convenience functions for creating queries programmatically
def simple_query(category: str, attributes: list[str] | None = None) -> GroundingQuery:
    """Create a simple query for a single object category."""
    return GroundingQuery(
        raw_query=category,
        root=QueryNode(categories=[category], attributes=attributes or []),
    )


def spatial_query(
    target: str,
    relation: str,
    anchor: str,
    target_attributes: list[str] | None = None,
    anchor_attributes: list[str] | None = None,
) -> GroundingQuery:
    """Create a simple spatial query: target [relation] anchor."""
    return GroundingQuery(
        raw_query=f"{target} {relation} {anchor}",
        root=QueryNode(
            categories=[target],
            attributes=target_attributes or [],
            spatial_constraints=[
                SpatialConstraint(
                    relation=relation,
                    anchors=[
                        QueryNode(
                            categories=[anchor], attributes=anchor_attributes or []
                        )
                    ],
                )
            ],
        ),
    )


def superlative_query(
    target: str,
    metric: str,
    order: str,
    reference: str | None = None,
) -> GroundingQuery:
    """Create a superlative query: e.g., 'the nearest chair to the door'."""
    ref_node = QueryNode(categories=[reference]) if reference else None

    return GroundingQuery(
        raw_query=f"{order} {target}" + (f" to {reference}" if reference else ""),
        root=QueryNode(
            categories=[target],
            select_constraint=SelectConstraint(
                constraint_type=ConstraintType.SUPERLATIVE,
                metric=metric,
                order=order,
                reference=ref_node,
            ),
        ),
    )
