"""
Query Executor for Nested Spatial Queries.

This module implements a recursive query executor that evaluates
GroundingQuery structures against a set of scene objects.

The execution follows a bottom-up approach:
1. Start from the innermost (leaf) nodes
2. Evaluate spatial constraints to filter candidates
3. Apply select constraints to choose final results
4. Propagate results up to parent nodes

Usage:
    executor = QueryExecutor(objects, relation_checker)
    results = executor.execute(grounding_query)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from .core import (
    DIRECTIONAL_RELATIONS,
    ConstraintType,
    ExecutionPolicy,
    GroundingQuery,
    QueryNode,
    ReferenceFrame,
    SelectConstraint,
    SpatialConstraint,
    ViewpointContext,
)
from .quick_filters import AttributeFilter, QuickFilters
from .retrieval.spatial_checker import SpatialRelationChecker
from .viewpoint_geometry import (
    ViewerPose,
    resolve_viewer_pose,
    viewer_frame_axis_value,
    viewer_frame_relation_score,
)

# Score floor for SOFT policy when no satisfaction is found. Multiplying by
# this value keeps a candidate alive (does not zero it out) while still
# letting properly-satisfying constraints dominate the ranking.
SOFT_POLICY_SCORE_FLOOR: float = 1e-3

if TYPE_CHECKING:
    from .retrieval import SceneObject

UNKNOWN_CATEGORY_SENTINELS = frozenset(
    {"", "unknow", "unknown", "none", "null", "n/a", "na"}
)


class ExecutionMode(str, Enum):
    """Controls whether execution is strict filtering or recall-biased."""

    STRICT = "strict"
    RECALL = "recall"


@dataclass
class ExecutionResult:
    """Result of executing a query node."""

    node_id: str
    matched_objects: list[SceneObject]
    scores: dict[int, float] = field(default_factory=dict)  # obj_id -> score
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return len(self.matched_objects) == 0

    @property
    def best_object(self) -> SceneObject | None:
        if not self.matched_objects:
            return None
        if self.scores:
            best_id = max(self.scores.keys(), key=lambda x: self.scores[x])
            for obj in self.matched_objects:
                if obj.obj_id == best_id:
                    return obj
        return self.matched_objects[0]


class QueryExecutor:
    """
    Recursive executor for nested spatial queries.

    Evaluates GroundingQuery structures by:
    1. Finding objects matching each node's category
    2. Applying spatial constraints as filters
    3. Applying select constraints for final selection

    Attributes:
        objects: List of scene objects to search
        relation_checker: Spatial relation checker instance
        category_index: Index mapping categories to objects
    """

    def __init__(
        self,
        objects: list[SceneObject],
        relation_checker: SpatialRelationChecker | None = None,
        clip_features: np.ndarray | None = None,
        clip_encoder: Any | None = None,
        use_quick_filters: bool = True,
        camera_poses: list[np.ndarray] | None = None,
    ):
        """
        Initialize the query executor.

        Args:
            objects: List of scene objects
            relation_checker: Optional pre-configured relation checker
            clip_features: Optional pre-computed CLIP features for objects
            clip_encoder: Optional CLIP text encoder for semantic matching
            use_quick_filters: Whether to use quick filters for pre-filtering
            camera_poses: Optional list of 4x4 world_T_cam matrices used by
                viewer-frame resolution to pick the most-aligned frame for
                a ``FACING_ANCHOR`` viewpoint context. When None, viewer-pose
                resolution falls back to the geometric room-centroid heuristic.
        """
        self.objects = objects
        self.relation_checker = relation_checker or SpatialRelationChecker()
        self.clip_features = clip_features
        self.clip_encoder = clip_encoder
        self.use_quick_filters = use_quick_filters
        self.camera_poses = camera_poses or []

        # Set per-query in execute(); read by _resolve_viewer_pose_for_constraint.
        self._viewpoint_contexts: dict[str, ViewpointContext] = {}
        self._viewer_pose_cache: dict[str, ViewerPose | None] = {}
        self._execution_trace: list[dict[str, Any]] = []
        self._execution_mode = ExecutionMode.STRICT

        # Quick filters for fast pre-filtering
        self._quick_filters = QuickFilters() if use_quick_filters else None
        self._attribute_filter = AttributeFilter() if use_quick_filters else None

        # Build category index (primary category only)
        self._category_index: dict[str, list[SceneObject]] = {}
        for obj in objects:
            category = self._get_category(obj).lower()
            if category not in self._category_index:
                self._category_index[category] = []
            self._category_index[category].append(obj)

        # Build multi-label index: maps minority detection class names to objects.
        # An object detected as ["sink"x6, "lamp"x4] has primary="sink" in
        # _category_index, but "lamp" only appears here — preventing the 65%
        # class loss from majority-vote.
        self._multilabel_index: dict[str, list[SceneObject]] = {}
        for obj in objects:
            if not hasattr(obj, "class_name") or not obj.class_name:
                continue
            primary = self._get_category(obj).lower()
            counts = Counter(obj.class_name)
            for cls, cnt in counts.items():
                cls_lower = cls.lower() if cls else ""
                # Skip primary (already in _category_index), empty, or low-count
                if not cls_lower or cls_lower == primary or cnt < 2:
                    continue
                if cls_lower not in self._multilabel_index:
                    self._multilabel_index[cls_lower] = []
                self._multilabel_index[cls_lower].append(obj)

        # Execution cache for memoization
        self._cache: dict[str, ExecutionResult] = {}

    def _get_category(self, obj: SceneObject) -> str:
        """Get the category string for an object."""
        if hasattr(obj, "object_tag") and obj.object_tag:
            return obj.object_tag
        return obj.category

    def _get_centroid(self, obj: SceneObject) -> np.ndarray:
        """Get object centroid."""
        if hasattr(obj, "centroid") and obj.centroid is not None:
            return np.asarray(obj.centroid, dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    @staticmethod
    def _object_ids(objects: list[SceneObject]) -> list[int]:
        """Return stable object ids for trace metadata."""
        return [int(getattr(obj, "obj_id", -1)) for obj in objects]

    @staticmethod
    def _neutral_select_for_policy(
        candidates: list[SceneObject],
        scores: dict[int, float],
        policy: ExecutionPolicy,
    ) -> tuple[list[SceneObject], dict[int, float]]:
        """Apply a SelectConstraint policy with zero metric evidence.

        Used when a viewer-frame select metric (e.g. x_position in viewer
        frame) cannot be evaluated because the pose did not resolve. As
        with ``_neutral_evidence_for_policy``, we refuse to substitute
        world coordinates for viewer coordinates - that would silently
        change the meaning of the parsed select.

        - HARD: return empty (no selection possible).
        - SOFT / RANK_ONLY: keep all candidates with their incoming
          multiplicative score unchanged (no metric reranking).
        """
        if policy == ExecutionPolicy.HARD:
            return [], {}
        kept_scores = {c.obj_id: scores.get(c.obj_id, 1.0) for c in candidates}
        return list(candidates), kept_scores

    @staticmethod
    def _neutral_evidence_for_policy(
        candidates: list[SceneObject],
        policy: ExecutionPolicy,
    ) -> tuple[list[SceneObject], dict[int, float], dict[int, float]]:
        """Apply ``policy`` with zero geometric evidence.

        Used when a viewer-frame spatial constraint requested a pose that
        could not be resolved. We must NOT fall back to world-frame
        evaluation (that would silently fabricate evidence), so the executor
        instead applies the policy as if no candidate satisfies any anchor:

        - HARD: return empty (no fabricated matches).
        - SOFT: keep all candidates with the SOFT score floor so downstream
          ranking still sees them, but with the lowest possible score.
        - RANK_ONLY: keep all candidates with mult=1.0 and soft=0.0 so the
          joint-coverage tie-breaker still sees them but the directional
          relation contributes nothing.
        """
        if policy == ExecutionPolicy.HARD:
            return [], {}, {}
        if policy == ExecutionPolicy.SOFT:
            return (
                list(candidates),
                {c.obj_id: SOFT_POLICY_SCORE_FLOOR for c in candidates},
                {c.obj_id: 0.0 for c in candidates},
            )
        # RANK_ONLY
        return (
            list(candidates),
            {c.obj_id: 1.0 for c in candidates},
            {c.obj_id: 0.0 for c in candidates},
        )

    def _resolve_viewer_pose_for_constraint(
        self, constraint: SpatialConstraint | SelectConstraint
    ) -> ViewerPose | None:
        """Look up (and cache) the ViewerPose for a viewer-frame constraint.

        Returns None when:
        - constraint.reference_frame != VIEWER (caller's responsibility to check first)
        - viewpoint_context_id not set or unknown to this query
        - facing/origin anchor objects do not resolve to scene objects
        - trajectory mode in ``resolve_viewer_pose`` cannot pick an aligned
          frame (no camera_poses, all degenerate, or none face the anchor).
          The executor explicitly passes ``allow_geometric_fallback=False``
          so the geometric room-centroid synthesis is DISABLED on this path -
          there is no silent fabrication of a viewer pose.

        On None, the executor MUST apply the constraint's declared policy
        via ``_neutral_evidence_for_policy`` / ``_neutral_select_for_policy``
        (HARD -> empty; SOFT / RANK_ONLY keep all with neutral scores).
        World-frame fallthrough is forbidden.
        """
        ctx_id = constraint.viewpoint_context_id
        if not ctx_id:
            return None
        if ctx_id in self._viewer_pose_cache:
            return self._viewer_pose_cache[ctx_id]

        ctx = self._viewpoint_contexts.get(ctx_id)
        if ctx is None:
            self._viewer_pose_cache[ctx_id] = None
            return None

        # Prefer facing_anchor; fall back to origin/subject for other kinds.
        anchor_node = ctx.facing_anchor or ctx.origin_anchor or ctx.subject_anchor
        if anchor_node is None:
            self._viewer_pose_cache[ctx_id] = None
            return None

        anchor_result = self._execute_node(anchor_node)
        if not anchor_result.matched_objects:
            self._viewer_pose_cache[ctx_id] = None
            return None

        all_centroids = None
        if self.objects:
            all_centroids = np.stack([self._get_centroid(o) for o in self.objects])

        # Strict no-fallback: refuse to synthesize a viewer pose when the
        # camera trajectory is missing or degenerate. The executor MUST NOT
        # invoke the geometric fallback path inside resolve_viewer_pose -
        # the caller already accepted that responsibility upstream (either
        # by loading a trajectory or by accepting None and downgrading to
        # the constraint's declared execution policy via
        # _neutral_evidence_for_policy / _neutral_select_for_policy).
        pose = resolve_viewer_pose(
            anchor_result.matched_objects,
            all_object_centroids=all_centroids,
            camera_poses=self.camera_poses if self.camera_poses else None,
            allow_geometric_fallback=False,
        )
        self._viewer_pose_cache[ctx_id] = pose
        return pose

    def execute(
        self,
        query: GroundingQuery,
        mode: ExecutionMode | str = ExecutionMode.STRICT,
    ) -> ExecutionResult:
        """
        Execute a grounding query.

        Args:
            query: GroundingQuery to execute

        Returns:
            ExecutionResult with matched objects
        """
        if not isinstance(mode, ExecutionMode):
            mode = ExecutionMode(mode)
        logger.info(
            f"[QueryExecutor] Executing query: '{query.raw_query}' "
            f"(mode={mode.value})"
        )

        # Clear caches for new query
        self._cache.clear()
        self._viewpoint_contexts = {vc.id: vc for vc in query.viewpoint_contexts}
        self._viewer_pose_cache = {}
        self._execution_trace = []
        self._execution_mode = mode

        # Execute from root
        result = self._execute_node(query.root)

        # If expect_unique, keep only the best result
        if query.expect_unique and len(result.matched_objects) > 1:
            best = result.best_object
            if best:
                result = ExecutionResult(
                    node_id=result.node_id,
                    matched_objects=[best],
                    scores={best.obj_id: result.scores.get(best.obj_id, 1.0)},
                    metadata=result.metadata,
                )

        if self._execution_trace:
            result.metadata = dict(result.metadata)
            result.metadata["execution_trace"] = list(self._execution_trace)

        logger.info(f"[QueryExecutor] Found {len(result.matched_objects)} objects")
        return result

    def _execute_node(self, node: QueryNode) -> ExecutionResult:
        """
        Recursively execute a query node.

        Execution order:
        1. Find candidates by category
        2. Filter by attributes
        3. Apply spatial constraints (filter phase)
        4. Apply select constraint (selection phase)
        """
        # Check cache
        if node.node_id and node.node_id in self._cache:
            return self._cache[node.node_id]

        # Determine nesting depth from node_id (e.g., "root_sc0_a0_sc0_a0" has depth 2)
        depth = node.node_id.count("_sc") if node.node_id else 0
        indent = "  " * depth
        logger.debug(
            f"[QueryExecutor]{indent} Executing node: categories={node.categories} (depth={depth})"
        )

        # Step 1: Find candidates by categories (supports semantic expansion)
        if node.open_ended and node.spatial_constraints:
            # Open-ended query: return ALL objects, let spatial constraints filter
            candidates = list(self.objects)
            logger.info(
                f"[QueryExecutor] Open-ended query: starting with all "
                f"{len(candidates)} objects for spatial filtering"
            )
        else:
            candidates = self._find_by_categories(node.categories)
            logger.debug(
                f"[QueryExecutor] Found {len(candidates)} candidates for categories {node.categories}"
            )

        if not candidates:
            result = ExecutionResult(node_id=node.node_id, matched_objects=[])
            self._cache[node.node_id] = result
            return result

        # Step 2: Filter by attributes
        if node.attributes:
            candidates = self._filter_by_attributes(candidates, node.attributes)
            logger.debug(
                f"[QueryExecutor] After attribute filter: {len(candidates)} candidates"
            )

        # Step 3: Apply spatial constraints (AND logic)
        scores = {obj.obj_id: 1.0 for obj in candidates}
        soft_match_scores: dict[int, dict[str, float]] = {}

        for sc_idx, constraint in enumerate(node.spatial_constraints):
            candidates, constraint_scores, constraint_soft = (
                self._apply_spatial_constraint(
                    candidates,
                    constraint,
                    node_id=node.node_id,
                    node_categories=node.categories,
                    constraint_index=sc_idx,
                )
            )
            # Combine multiplicative scores
            for obj_id, score in constraint_scores.items():
                if obj_id in scores:
                    scores[obj_id] *= score

            # Aggregate soft satisfaction per constraint for read-only consumers
            if constraint_soft:
                key = f"sc{sc_idx}:{constraint.relation}:{constraint.execution_policy.value}"
                for obj_id, soft_score in constraint_soft.items():
                    soft_match_scores.setdefault(obj_id, {})[key] = soft_score

            logger.debug(
                f"[QueryExecutor] After '{constraint.relation}' constraint "
                f"(policy={constraint.execution_policy.value}): "
                f"{len(candidates)} candidates"
            )

            if not candidates:
                break

        # Step 4: Apply select constraint
        if candidates and node.select_constraint:
            candidates, scores = self._apply_select_constraint(
                candidates, scores, node.select_constraint
            )
            logger.debug(
                f"[QueryExecutor] After select constraint: {len(candidates)} candidates"
            )

        metadata: dict[str, Any] = {
            "categories": node.categories,
            "category": node.category,
        }
        if soft_match_scores:
            metadata["soft_match_scores"] = soft_match_scores

        result = ExecutionResult(
            node_id=node.node_id,
            matched_objects=candidates,
            scores=scores,
            metadata=metadata,
        )

        if node.node_id:
            self._cache[node.node_id] = result

        return result

    def _find_by_categories(self, categories: list[str]) -> list[SceneObject]:
        """
        Find objects matching any of the given categories.

        This method supports semantic expansion - when the LLM returns multiple
        related categories (e.g., ["pillow", "throw_pillow"]), all matching
        objects are returned.

        Args:
            categories: List of category strings to search for

        Returns:
            List of matched scene objects (deduplicated)
        """
        matches = []
        seen_ids = set()
        search_categories = [
            category
            for category in categories
            if category.strip().lower() not in UNKNOWN_CATEGORY_SENTINELS
        ]
        if not search_categories:
            logger.warning(
                f"[QueryExecutor] No searchable category in sentinel-only categories {categories}"
            )
            return []

        for category in search_categories:
            category_lower = category.lower()

            # Exact match
            if category_lower in self._category_index:
                for obj in self._category_index[category_lower]:
                    if obj.obj_id not in seen_ids:
                        matches.append(obj)
                        seen_ids.add(obj.obj_id)

        # If we found matches, return them
        if matches:
            logger.debug(
                f"[QueryExecutor] Exact match for categories {categories}: {len(matches)} objects"
            )
            return matches

        # Fallback 1: substring matching on primary categories
        for category in search_categories:
            category_lower = category.lower()
            for cat, objs in self._category_index.items():
                if category_lower in cat or cat in category_lower:
                    for obj in objs:
                        if obj.obj_id not in seen_ids:
                            matches.append(obj)
                            seen_ids.add(obj.obj_id)

        if matches:
            logger.debug(
                f"[QueryExecutor] Substring match for categories {categories}: {len(matches)} objects"
            )
            return matches

        # Fallback 2: multi-label exact match (minority detection classes)
        for category in search_categories:
            category_lower = category.lower()
            if category_lower in self._multilabel_index:
                for obj in self._multilabel_index[category_lower]:
                    if obj.obj_id not in seen_ids:
                        matches.append(obj)
                        seen_ids.add(obj.obj_id)

        if matches:
            logger.debug(
                f"[QueryExecutor] Multi-label match for categories {categories}: {len(matches)} objects"
            )
            return matches

        # Fallback 3: CLIP similarity (if available) - use first category
        if (
            self.clip_features is not None
            and self.clip_encoder is not None
            and search_categories
        ):
            return self._find_by_clip_similarity(search_categories[0])

        # No matches found
        logger.warning(f"[QueryExecutor] No match for categories {categories}")
        return []

    def _find_by_category(self, category: str) -> list[SceneObject]:
        """
        Find objects matching a category.

        Uses exact match first, then substring match, then CLIP similarity.

        Note: This method is kept for backward compatibility.
        New code should use _find_by_categories() instead.
        """
        return self._find_by_categories([category])

    def _find_by_clip_similarity(
        self, category: str, top_k: int = 10, min_similarity: float = 0.2
    ) -> list[SceneObject]:
        """Find objects by CLIP text-image similarity.

        Returns empty list when CLIP encoder unavailable (open_clip not installed).
        Raises on encoding failure when CLIP IS loaded.
        """
        text_feature = self.clip_encoder(category)
        if text_feature is None:
            return []

        # Compute similarities
        similarities = self.clip_features @ text_feature
        top_indices = np.argsort(-similarities)[:top_k]

        matches = [
            self.objects[i] for i in top_indices if similarities[i] > min_similarity
        ]

        if matches:
            logger.info(
                f"[QueryExecutor] CLIP matched '{category}' -> "
                f"{[(self._get_category(m), f'{similarities[self.objects.index(m)]:.2f}') for m in matches[:3]]}"
            )

        return matches

    def _filter_by_attributes(
        self,
        candidates: list[SceneObject],
        attributes: list[str],
    ) -> list[SceneObject]:
        """Filter candidates by attributes (color, size, etc.).

        Uses AttributeFilter for color and size filtering.
        """
        if not attributes or not self._attribute_filter:
            return candidates

        filtered = candidates
        for attr in attributes:
            attr_lower = attr.lower()

            # Try color filtering
            if self._attribute_filter._color_lookup.get(attr_lower):
                filtered = self._attribute_filter.filter_by_color(filtered, attr_lower)
                logger.debug(
                    f"[QueryExecutor] After color filter '{attr}': {len(filtered)} candidates"
                )

            # Note: size filtering (largest, smallest) is handled by select_constraint

        return filtered

    def _apply_spatial_constraint(
        self,
        candidates: list[SceneObject],
        constraint: SpatialConstraint,
        node_id: str | None = None,
        node_categories: list[str] | None = None,
        constraint_index: int | None = None,
    ) -> tuple[list[SceneObject], dict[int, float], dict[int, float]]:
        """
        Apply a spatial constraint to filter candidates.

        Uses a two-phase approach:
        1. Quick filter: Fast pre-filtering using simple coordinate comparisons
        2. Full check: Accurate spatial relation checking for remaining candidates

        The behavior of a failing constraint is governed by
        ``constraint.execution_policy``:

        - HARD: legacy filter-on-fail. If every candidate fails, returns ``[], {}, {}``.
        - SOFT: keep all candidates. Multiplicative scores are scaled by the
          satisfaction score (with a floor of SOFT_POLICY_SCORE_FLOOR for
          unsatisfied candidates) so downstream ranking still discriminates.
        - RANK_ONLY: keep all candidates with multiplicative score unchanged.
          Per-candidate satisfaction is written into the third return value
          (soft_match_scores) for read-only consumers like joint_coverage.

        Returns:
            Tuple of (candidates_out, multiplicative_scores, soft_match_scores)
        """
        # Default behavior reproduced even on legacy callers that build a
        # SpatialConstraint without the new fields - pydantic supplies HARD/world.
        policy = constraint.execution_policy
        trace: dict[str, Any] = {
            "node_id": node_id,
            "node_categories": list(node_categories or []),
            "constraint_index": constraint_index,
            "relation": constraint.relation,
            "reference_frame": constraint.reference_frame.value,
            "policy": policy.value,
            "candidate_ids_before": self._object_ids(candidates),
            "anchor_ids": [],
            "quick_filter_applied": False,
            "quick_filter_candidate_ids": self._object_ids(candidates),
            "quick_filter_would_empty": False,
            "candidate_relation_scores": {},
            "candidate_relation_details": {},
            "output_candidate_ids": [],
        }

        # Execute anchor nodes to get reference objects
        anchor_objects = []
        for anchor_node in constraint.anchors:
            logger.debug(
                f"[QueryExecutor] Resolving anchor {anchor_node.categories} "
                f"(has {len(anchor_node.spatial_constraints)} nested constraints)"
            )
            anchor_result = self._execute_node(anchor_node)
            logger.debug(
                f"[QueryExecutor] Anchor {anchor_node.categories} resolved to "
                f"{len(anchor_result.matched_objects)} objects: "
                f"{[self._get_category(o) for o in anchor_result.matched_objects[:3]]}"
                f"{'...' if len(anchor_result.matched_objects) > 3 else ''}"
            )
            anchor_objects.extend(anchor_result.matched_objects)

        trace["anchor_ids"] = self._object_ids(anchor_objects)

        if not anchor_objects:
            logger.warning(
                f"[QueryExecutor] No anchor objects found for relation "
                f"'{constraint.relation}' (policy={policy.value}). "
                "Spatial constraint cannot be evaluated geometrically."
            )
            if policy == ExecutionPolicy.HARD:
                trace["anchor_resolution_empty"] = True
                trace["output_candidate_ids"] = []
                self._execution_trace.append(trace)
                return [], {}, {}
            # SOFT / RANK_ONLY: no information to filter on, keep everything.
            scores = {c.obj_id: 1.0 for c in candidates}
            trace["anchor_resolution_empty"] = True
            trace["output_candidate_ids"] = self._object_ids(candidates)
            self._execution_trace.append(trace)
            return list(candidates), scores, {}

        # Phase 1: Quick filter (if available). Only run for HARD policy - SOFT
        # and RANK_ONLY must never reduce the candidate pool here.
        pre_filtered = candidates
        if policy == ExecutionPolicy.HARD and (
            self._quick_filters and self._quick_filters.has_filter(constraint.relation)
        ):
            quick_filtered = self._quick_filters.filter_candidates(
                candidates, anchor_objects, constraint.relation
            )
            trace["quick_filter_applied"] = True
            trace["quick_filter_candidate_ids"] = self._object_ids(quick_filtered)
            logger.debug(
                f"[QueryExecutor] Quick filter '{constraint.relation}': "
                f"{len(candidates)} -> {len(quick_filtered)} candidates"
            )

            if not quick_filtered:
                trace["quick_filter_would_empty"] = True
                logger.warning(
                    f"[QueryExecutor] Quick filter '{constraint.relation}' eliminated all "
                    f"{len(candidates)} candidates. Falling through to full checker."
                )
                pre_filtered = candidates
            else:
                pre_filtered = quick_filtered

        # Phase 2: Full spatial relation check. Viewer-frame branch when the
        # constraint asks for it AND the pose resolves; otherwise use the
        # world-frame SpatialRelationChecker. Critical: when viewer is
        # requested but pose cannot resolve, we do NOT fall back to the
        # world-frame relation checker - that would silently fabricate
        # world-frame evidence. Instead we apply the declared policy with
        # neutral evidence (HARD = empty, SOFT/RANK_ONLY = keep candidates).
        wants_viewer_frame = (
            constraint.reference_frame == ReferenceFrame.VIEWER
            and constraint.relation.lower().replace(" ", "_") in DIRECTIONAL_RELATIONS
        )
        viewer_pose: ViewerPose | None = None
        if wants_viewer_frame:
            viewer_pose = self._resolve_viewer_pose_for_constraint(constraint)
            if viewer_pose is None:
                logger.warning(
                    f"[QueryExecutor] viewer-frame requested for relation "
                    f"'{constraint.relation}' but pose unresolved (policy={policy.value}); "
                    "applying declared policy with neutral evidence (no world-frame fallback)."
                )
                out, out_scores, out_soft = self._neutral_evidence_for_policy(
                    pre_filtered, policy
                )
                trace["viewer_pose_unresolved"] = True
                trace["output_candidate_ids"] = self._object_ids(out)
                self._execution_trace.append(trace)
                return out, out_scores, out_soft

        satisfied_set: set[int] = set()
        sat_scores: dict[int, float] = {}
        relation_scores: dict[int, float] = {}
        relation_details: dict[int, dict[str, Any]] = {}

        for cand in pre_filtered:
            best_score = 0.0
            best_satisfying_score = 0.0
            satisfies_any = False
            best_details: dict[str, Any] = {}

            if wants_viewer_frame and viewer_pose is not None:
                cand_centroid = self._get_centroid(cand)
                for anchor in anchor_objects:
                    score = viewer_frame_relation_score(
                        constraint.relation,
                        cand_centroid,
                        self._get_centroid(anchor),
                        viewer_pose,
                    )
                    if score > 0:
                        satisfies_any = True
                        best_score = max(best_score, score)
                        best_satisfying_score = max(best_satisfying_score, score)
            elif constraint.relation.lower() == "between" and len(anchor_objects) >= 2:
                # For "between", let the relation checker search all anchor pairs.
                result = self.relation_checker.check(
                    cand, anchor_objects, constraint.relation
                )
                best_score = result.score
                best_details = dict(result.details)
                if result.satisfies:
                    satisfies_any = True
                    best_satisfying_score = result.score
            else:
                # For other relations, check against each anchor
                for anchor in anchor_objects:
                    result = self.relation_checker.check(
                        cand, anchor, constraint.relation
                    )
                    if result.score >= best_score:
                        best_score = result.score
                        best_details = dict(result.details)
                    if result.satisfies:
                        satisfies_any = True
                        best_satisfying_score = max(
                            best_satisfying_score, result.score
                        )

            relation_scores[cand.obj_id] = float(best_score)
            if best_details:
                relation_details[cand.obj_id] = best_details
            if satisfies_any:
                satisfied_set.add(cand.obj_id)
                sat_scores[cand.obj_id] = best_satisfying_score

        trace["candidate_relation_scores"] = relation_scores
        trace["candidate_relation_details"] = relation_details
        trace["full_check_satisfied_ids"] = sorted(satisfied_set)

        if policy == ExecutionPolicy.HARD:
            filtered = [c for c in pre_filtered if c.obj_id in satisfied_set]
            trace["full_check_emptied"] = bool(pre_filtered and not filtered)

            relation_norm = constraint.relation.lower().replace(" ", "_")
            if (
                self._execution_mode == ExecutionMode.RECALL
                and trace["full_check_emptied"]
                and relation_norm
                in {"on", "on_top", "on_top_of", "upon", "atop", "resting_on"}
            ):
                keep_scores = {c.obj_id: 1.0 for c in pre_filtered}
                soft_scores = {
                    c.obj_id: sat_scores.get(c.obj_id, 0.0) for c in pre_filtered
                }
                trace["hard_on_recall_fallback"] = True
                trace["effective_policy"] = ExecutionPolicy.RANK_ONLY.value
                trace["output_candidate_ids"] = self._object_ids(pre_filtered)
                self._execution_trace.append(trace)
                return list(pre_filtered), keep_scores, soft_scores

            trace["hard_on_recall_fallback"] = False
            trace["output_candidate_ids"] = self._object_ids(filtered)
            self._execution_trace.append(trace)
            return filtered, dict(sat_scores), {}

        if policy == ExecutionPolicy.SOFT:
            out_scores: dict[int, float] = {}
            for c in pre_filtered:
                s = sat_scores.get(c.obj_id, 0.0)
                out_scores[c.obj_id] = s if s > 0 else SOFT_POLICY_SCORE_FLOOR
            trace["full_check_emptied"] = False
            trace["output_candidate_ids"] = self._object_ids(pre_filtered)
            self._execution_trace.append(trace)
            return list(pre_filtered), out_scores, dict(sat_scores)

        # RANK_ONLY: keep all candidates, multiplicative score untouched (1.0),
        # surface satisfaction in soft_match_scores for downstream rerankers.
        keep_scores: dict[int, float] = {c.obj_id: 1.0 for c in pre_filtered}
        soft_scores = {c.obj_id: sat_scores.get(c.obj_id, 0.0) for c in pre_filtered}
        trace["full_check_emptied"] = False
        trace["output_candidate_ids"] = self._object_ids(pre_filtered)
        self._execution_trace.append(trace)
        return list(pre_filtered), keep_scores, soft_scores

    def _apply_select_constraint(
        self,
        candidates: list[SceneObject],
        scores: dict[int, float],
        constraint: SelectConstraint,
    ) -> tuple[list[SceneObject], dict[int, float]]:
        """
        Apply a select constraint (superlative/ordinal).

        Honours ``constraint.execution_policy``:
        - HARD (default): legacy behavior, collapses to the chosen candidate.
        - SOFT: keep all candidates but rerank scores by metric so the chosen
          candidate has the highest score and others remain present.
        - RANK_ONLY: keep all candidates and scores unchanged, but record the
          metric ordering in the candidate scores by ordinal rank (so a
          downstream consumer that picks argmax(scores) still gets the
          select-favored candidate).

        Args:
            candidates: Current candidate objects
            scores: Current scores
            constraint: Select constraint to apply

        Returns:
            Tuple of (candidates, updated scores)
        """
        if not candidates:
            return [], {}

        policy = constraint.execution_policy

        if constraint.constraint_type == ConstraintType.SUPERLATIVE:
            return self._apply_superlative(candidates, scores, constraint, policy)
        elif constraint.constraint_type == ConstraintType.ORDINAL:
            return self._apply_ordinal(candidates, scores, constraint, policy)
        elif constraint.constraint_type == ConstraintType.COMPARATIVE:
            return self._apply_comparative(candidates, scores, constraint)
        else:
            return candidates, scores

    def _apply_superlative(
        self,
        candidates: list[SceneObject],
        scores: dict[int, float],
        constraint: SelectConstraint,
        policy: ExecutionPolicy = ExecutionPolicy.HARD,
    ) -> tuple[list[SceneObject], dict[int, float]]:
        """Apply superlative constraint (nearest, largest, etc.)."""
        metric = constraint.metric.lower()
        order = constraint.order.lower()

        # Viewer-frame projection: x_position becomes signed projection onto
        # the viewer's right axis. If the constraint requests viewer frame
        # but the pose cannot be resolved, we MUST NOT silently fall through
        # to world x/y - that would fabricate evidence the parser explicitly
        # said is not in world frame. Apply the declared policy with neutral
        # evidence instead.
        viewer_pose: ViewerPose | None = None
        wants_viewer_select = (
            constraint.reference_frame == ReferenceFrame.VIEWER
            and metric in ("x_position", "x", "y_position", "y")
        )
        if wants_viewer_select:
            viewer_pose = self._resolve_viewer_pose_for_constraint(constraint)
            if viewer_pose is None:
                logger.warning(
                    f"[QueryExecutor] viewer-frame select '{metric}' requested "
                    f"but pose unresolved (policy={policy.value}); applying "
                    "declared policy with neutral evidence (no world-frame fallback)."
                )
                return self._neutral_select_for_policy(
                    candidates, scores, policy
                )

        # Get reference objects if needed
        ref_objects = []
        if constraint.reference:
            ref_result = self._execute_node(constraint.reference)
            ref_objects = ref_result.matched_objects

        # Compute metric values for each candidate
        values = []
        for cand in candidates:
            if metric == "distance" and ref_objects:
                # Distance to nearest reference object
                cand_pos = self._get_centroid(cand)
                min_dist = float("inf")
                for ref in ref_objects:
                    ref_pos = self._get_centroid(ref)
                    dist = float(np.linalg.norm(cand_pos - ref_pos))
                    min_dist = min(min_dist, dist)
                values.append(min_dist)

            elif metric == "size":
                # Use bounding box volume (not point count)
                if hasattr(cand, "bbox_3d") and cand.bbox_3d is not None:
                    size = np.prod(cand.bbox_3d.size)
                elif (
                    hasattr(cand, "pcd_np")
                    and cand.pcd_np is not None
                    and len(cand.pcd_np) > 0
                ):
                    # Compute volume from point cloud bounding box
                    pts = np.asarray(cand.pcd_np)
                    bbox_size = pts.max(axis=0) - pts.min(axis=0)
                    size = np.prod(bbox_size)
                elif hasattr(cand, "point_cloud") and cand.point_cloud is not None:
                    pts = np.asarray(cand.point_cloud)
                    if len(pts) > 0:
                        bbox_size = pts.max(axis=0) - pts.min(axis=0)
                        size = np.prod(bbox_size)
                    else:
                        size = 0.0
                else:
                    size = 0.0
                values.append(size)

            elif metric == "height":
                # Z coordinate
                pos = self._get_centroid(cand)
                values.append(pos[2])

            elif metric in ["x_position", "x"]:
                pos = self._get_centroid(cand)
                if viewer_pose is not None:
                    values.append(viewer_frame_axis_value(pos, viewer_pose, "right"))
                else:
                    values.append(pos[0])

            elif metric in ["y_position", "y"]:
                pos = self._get_centroid(cand)
                if viewer_pose is not None:
                    values.append(viewer_frame_axis_value(pos, viewer_pose, "forward"))
                else:
                    values.append(pos[1])

            else:
                # Default: use existing score
                values.append(scores.get(cand.obj_id, 0.0))

        # Sort and select
        indexed = list(zip(candidates, values, strict=False))

        if order == "min":
            indexed.sort(key=lambda x: x[1])
        else:  # max
            indexed.sort(key=lambda x: x[1], reverse=True)

        if policy == ExecutionPolicy.HARD:
            best_cand, best_value = indexed[0]
            new_scores = {best_cand.obj_id: 1.0}
            logger.debug(
                f"[QueryExecutor] Superlative '{order} {metric}': "
                f"selected {self._get_category(best_cand)} with value {best_value:.3f}"
            )
            return [best_cand], new_scores

        # SOFT / RANK_ONLY: keep all candidates, but rerank by metric so the
        # best candidate has the highest score. Use a monotonically decreasing
        # ramp; for SOFT this multiplies into existing scores, for RANK_ONLY
        # it replaces them but does not drop candidates.
        n = len(indexed)
        new_scores = dict(scores) if policy == ExecutionPolicy.SOFT else {}
        ordered_candidates = [c for c, _ in indexed]
        for rank, (c, _v) in enumerate(indexed):
            rank_score = max(SOFT_POLICY_SCORE_FLOOR, 1.0 - (rank / max(1, n)))
            if policy == ExecutionPolicy.SOFT:
                base = new_scores.get(c.obj_id, 1.0)
                new_scores[c.obj_id] = base * rank_score
            else:
                new_scores[c.obj_id] = rank_score
        logger.debug(
            f"[QueryExecutor] Superlative '{order} {metric}' (policy={policy.value}): "
            f"kept {n} candidates, top={self._get_category(ordered_candidates[0])}"
        )
        return ordered_candidates, new_scores

    def _apply_ordinal(
        self,
        candidates: list[SceneObject],
        scores: dict[int, float],
        constraint: SelectConstraint,
        policy: ExecutionPolicy = ExecutionPolicy.HARD,
    ) -> tuple[list[SceneObject], dict[int, float]]:
        """Apply ordinal constraint (first, second, etc.)."""
        if constraint.position is None:
            return candidates, scores

        position = constraint.position  # 1-indexed
        metric = constraint.metric.lower()
        order = constraint.order.lower()

        # Viewer-frame x/y: same no-world-fabrication rule as
        # _apply_superlative. When the pose resolves, the sort key MUST use
        # viewer-frame projection; if the pose does not resolve, we apply
        # the declared policy with neutral evidence rather than silently
        # falling back to world x/y.
        viewer_pose: ViewerPose | None = None
        wants_viewer_axis = (
            constraint.reference_frame == ReferenceFrame.VIEWER
            and metric in ("x_position", "x", "y_position", "y")
        )
        if wants_viewer_axis:
            viewer_pose = self._resolve_viewer_pose_for_constraint(constraint)
            if viewer_pose is None:
                logger.warning(
                    f"[QueryExecutor] viewer-frame ordinal '{metric}' requested "
                    f"but pose unresolved (policy={policy.value}); applying "
                    "declared policy with neutral evidence (no world-frame fallback)."
                )
                return self._neutral_select_for_policy(candidates, scores, policy)

        # Sort candidates by metric. When viewer-frame is requested AND the
        # pose resolved, x_position / y_position become signed projections
        # onto the viewer-right / viewer-forward axes (mirrors superlative).
        def get_value(cand):
            pos = self._get_centroid(cand)
            if metric in ["x_position", "x"]:
                if viewer_pose is not None:
                    return viewer_frame_axis_value(pos, viewer_pose, "right")
                return pos[0]
            elif metric in ["y_position", "y"]:
                if viewer_pose is not None:
                    return viewer_frame_axis_value(pos, viewer_pose, "forward")
                return pos[1]
            elif metric == "height":
                return pos[2]
            elif metric == "size":
                # Use same size logic as superlative
                if hasattr(cand, "bbox_3d") and cand.bbox_3d is not None:
                    return float(np.prod(cand.bbox_3d.size))
                if (
                    hasattr(cand, "pcd_np")
                    and cand.pcd_np is not None
                    and len(cand.pcd_np) > 0
                ):
                    pts = np.asarray(cand.pcd_np)
                    bbox_size = pts.max(axis=0) - pts.min(axis=0)
                    return float(np.prod(bbox_size))
                if hasattr(cand, "point_cloud") and cand.point_cloud is not None:
                    pts = np.asarray(cand.point_cloud)
                    if len(pts) > 0:
                        bbox_size = pts.max(axis=0) - pts.min(axis=0)
                        return float(np.prod(bbox_size))
                return 0.0
            else:
                return scores.get(cand.obj_id, 0.0)

        sorted_candidates = sorted(candidates, key=get_value, reverse=(order == "desc"))

        if position <= 0 or position > len(sorted_candidates):
            logger.warning(
                f"[QueryExecutor] Ordinal position {position} out of range "
                f"(have {len(sorted_candidates)} candidates)"
            )
            if policy == ExecutionPolicy.HARD:
                return [], {}
            return list(candidates), dict(scores)

        if policy == ExecutionPolicy.HARD:
            selected = sorted_candidates[position - 1]
            return [selected], {selected.obj_id: 1.0}

        n = len(sorted_candidates)
        new_scores = dict(scores) if policy == ExecutionPolicy.SOFT else {}
        for rank, c in enumerate(sorted_candidates):
            distance = abs(rank - (position - 1))
            rank_score = max(SOFT_POLICY_SCORE_FLOOR, 1.0 - distance / max(1, n))
            if policy == ExecutionPolicy.SOFT:
                base = new_scores.get(c.obj_id, 1.0)
                new_scores[c.obj_id] = base * rank_score
            else:
                new_scores[c.obj_id] = rank_score
        return sorted_candidates, new_scores

    def _apply_comparative(
        self,
        candidates: list[SceneObject],
        scores: dict[int, float],
        constraint: SelectConstraint,
    ) -> tuple[list[SceneObject], dict[int, float]]:
        """Apply comparative constraint (closer than, larger than)."""
        # Comparative requires a reference object to compare against
        # For now, this is similar to superlative but keeps multiple results

        # Get reference objects
        ref_objects = []
        if constraint.reference:
            ref_result = self._execute_node(constraint.reference)
            ref_objects = ref_result.matched_objects

        if not ref_objects:
            return candidates, scores

        constraint.metric.lower()
        constraint.order.lower()

        # Filter candidates that satisfy the comparison
        # This would require knowing what to compare against
        # For now, return all candidates

        return candidates, scores


# Convenience function
def execute_query(
    query: GroundingQuery,
    objects: list[SceneObject],
    relation_checker: SpatialRelationChecker | None = None,
) -> ExecutionResult:
    """
    Execute a grounding query against scene objects.

    Args:
        query: GroundingQuery to execute
        objects: List of scene objects
        relation_checker: Optional spatial relation checker

    Returns:
        ExecutionResult with matched objects
    """
    executor = QueryExecutor(objects, relation_checker)
    return executor.execute(query)
