"""Phase-1 tests: execution policy branches and the parse-normalize step.

Three guarantees pinned here:
1. HARD policy reproduces legacy behavior exactly.
2. SOFT policy keeps all candidates with score reshaping; never empties.
3. RANK_ONLY policy keeps all candidates with mult-score=1.0 and records
   per-candidate satisfaction in ``ExecutionResult.metadata['soft_match_scores']``.

Plus a normalize-step regression: a directional relation without viewpoint
context auto-demotes to ``frame=ambiguous`` + ``policy=rank_only`` so the
executor cannot drive the candidate set to empty.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from query_scene.core import (
    ExecutionPolicy,
    GroundingQuery,
    QueryNode,
    ReferenceFrame,
    SelectConstraint,
    SpatialConstraint,
    ViewpointContext,
    ViewpointKind,
)
from query_scene.query_executor import (
    SOFT_POLICY_SCORE_FLOOR,
    ExecutionMode,
    QueryExecutor,
)
from query_scene.retrieval.spatial_checker import RelationResult


def _mk_obj(obj_id: int, category: str, xyz: tuple[float, float, float]) -> object:
    return SimpleNamespace(
        obj_id=obj_id,
        category=category,
        object_tag=category,
        centroid=np.array(xyz, dtype=np.float32),
        bbox_3d=None,
        n_points=10,
        class_name=[category],
        summary=category,
    )


def _scene_left_right() -> tuple[list[object], list[object], object]:
    """Anchor 'door' at x=-2; cabinets at x=0,1,2. All cabinets are right_of door."""
    cabinets = [_mk_obj(i, "cabinet", (float(i), 0.0, 0.0)) for i in range(3)]
    door = _mk_obj(99, "door", (-2.0, 0.0, 0.0))
    return cabinets + [door], cabinets, door


def _sc(
    relation: str,
    anchor_categories: list[str],
    *,
    policy: ExecutionPolicy = ExecutionPolicy.HARD,
    frame: ReferenceFrame = ReferenceFrame.WORLD,
    ctx_id: str | None = None,
) -> SpatialConstraint:
    return SpatialConstraint(
        relation=relation,
        anchors=[QueryNode(categories=anchor_categories, node_id="anchor")],
        reference_frame=frame,
        viewpoint_context_id=ctx_id,
        execution_policy=policy,
    )


class TestHardPolicyMatchesLegacy(unittest.TestCase):
    def test_right_of_hard_keeps_satisfying_candidates(self) -> None:
        objs, cabinets, _door = _scene_left_right()
        executor = QueryExecutor(objects=objs)
        sc = _sc("right_of", ["door"])

        out, scores, soft = executor._apply_spatial_constraint(cabinets, sc)

        self.assertEqual({o.obj_id for o in out}, {0, 1, 2})
        self.assertTrue(all(scores[o.obj_id] > 0 for o in out))
        self.assertEqual(soft, {})

    def test_left_of_hard_empties_when_none_satisfy(self) -> None:
        objs, cabinets, _door = _scene_left_right()
        executor = QueryExecutor(objects=objs)
        sc = _sc("left_of", ["door"])

        out, scores, soft = executor._apply_spatial_constraint(cabinets, sc)

        self.assertEqual(out, [])
        self.assertEqual(scores, {})
        self.assertEqual(soft, {})

    def test_hard_no_anchor_returns_empty(self) -> None:
        cabinets = [_mk_obj(i, "cabinet", (float(i), 0.0, 0.0)) for i in range(2)]
        executor = QueryExecutor(objects=cabinets)
        sc = _sc("right_of", ["door"])

        out, scores, soft = executor._apply_spatial_constraint(cabinets, sc)

        self.assertEqual(out, [])
        self.assertEqual(scores, {})
        self.assertEqual(soft, {})


class _AlwaysSatisfyChecker:
    def check(self, target: object, anchor: object, relation: str) -> RelationResult:
        return RelationResult(
            satisfies=True,
            score=0.9,
            details={"source": "full_checker", "relation": relation},
        )


class _NeverSatisfyChecker:
    def check(self, target: object, anchor: object, relation: str) -> RelationResult:
        return RelationResult(
            satisfies=False,
            score=0.0,
            details={"source": "full_checker", "reason": "forced_false"},
        )


class TestHardQuickFilterSafety(unittest.TestCase):
    def test_hard_quick_filter_empty_falls_through_to_full_checker(self) -> None:
        box = _mk_obj(1, "box", (0.0, 0.0, 0.0))
        table = _mk_obj(2, "table", (0.0, 0.0, 1.0))
        executor = QueryExecutor(
            objects=[box, table], relation_checker=_AlwaysSatisfyChecker()
        )
        sc = _sc("on", ["table"], policy=ExecutionPolicy.HARD)

        out, scores, soft = executor._apply_spatial_constraint([box], sc)

        self.assertEqual([o.obj_id for o in out], [1])
        self.assertEqual(scores[1], 0.9)
        self.assertEqual(soft, {})

    def test_execution_trace_records_quick_filter_and_full_check(self) -> None:
        box = _mk_obj(1, "box", (0.0, 0.0, 0.0))
        table = _mk_obj(2, "table", (0.0, 0.0, 1.0))
        executor = QueryExecutor(
            objects=[box, table], relation_checker=_AlwaysSatisfyChecker()
        )
        query = GroundingQuery(
            raw_query="box on table",
            root=QueryNode(
                categories=["box"],
                spatial_constraints=[_sc("on", ["table"], policy=ExecutionPolicy.HARD)],
                node_id="root",
            ),
            expect_unique=False,
        )

        result = executor.execute(query)

        self.assertEqual([o.obj_id for o in result.matched_objects], [1])
        trace = result.metadata["execution_trace"]
        self.assertEqual(len(trace), 1)
        entry = trace[0]
        self.assertEqual(entry["node_id"], "root")
        self.assertEqual(entry["relation"], "on")
        self.assertEqual(entry["policy"], "hard")
        self.assertEqual(entry["candidate_ids_before"], [1])
        self.assertEqual(entry["anchor_ids"], [2])
        self.assertEqual(entry["quick_filter_candidate_ids"], [])
        self.assertTrue(entry["quick_filter_would_empty"])
        self.assertEqual(entry["output_candidate_ids"], [1])
        self.assertEqual(entry["candidate_relation_scores"][1], 0.9)

    def test_strict_mode_keeps_hard_on_empty(self) -> None:
        box = _mk_obj(1, "box", (0.0, 0.0, 0.0))
        table = _mk_obj(2, "table", (0.0, 0.0, 1.0))
        executor = QueryExecutor(
            objects=[box, table], relation_checker=_NeverSatisfyChecker()
        )
        query = GroundingQuery(
            raw_query="box on table",
            root=QueryNode(
                categories=["box"],
                spatial_constraints=[_sc("on", ["table"], policy=ExecutionPolicy.HARD)],
                node_id="root",
            ),
            expect_unique=False,
        )

        result = executor.execute(query, mode=ExecutionMode.STRICT)

        self.assertEqual(result.matched_objects, [])
        trace = result.metadata["execution_trace"]
        self.assertFalse(trace[0].get("hard_on_recall_fallback", False))

    def test_recall_mode_demotes_empty_hard_on_to_rank_only(self) -> None:
        box = _mk_obj(1, "box", (0.0, 0.0, 0.0))
        table = _mk_obj(2, "table", (0.0, 0.0, 1.0))
        executor = QueryExecutor(
            objects=[box, table], relation_checker=_NeverSatisfyChecker()
        )
        query = GroundingQuery(
            raw_query="box on table",
            root=QueryNode(
                categories=["box"],
                spatial_constraints=[_sc("on", ["table"], policy=ExecutionPolicy.HARD)],
                node_id="root",
            ),
            expect_unique=False,
        )

        result = executor.execute(query, mode=ExecutionMode.RECALL)

        self.assertEqual([o.obj_id for o in result.matched_objects], [1])
        self.assertEqual(result.scores[1], 1.0)
        self.assertEqual(
            next(iter(result.metadata["soft_match_scores"][1].values())),
            0.0,
        )
        trace = result.metadata["execution_trace"]
        self.assertTrue(trace[0]["hard_on_recall_fallback"])
        self.assertEqual(trace[0]["output_candidate_ids"], [1])


class TestSoftPolicyKeepsCandidates(unittest.TestCase):
    def test_left_of_soft_keeps_all_with_score_floor(self) -> None:
        objs, cabinets, _door = _scene_left_right()
        executor = QueryExecutor(objects=objs)
        sc = _sc("left_of", ["door"], policy=ExecutionPolicy.SOFT)

        out, scores, _soft = executor._apply_spatial_constraint(cabinets, sc)

        self.assertEqual({o.obj_id for o in out}, {0, 1, 2})
        for cab in out:
            self.assertEqual(scores[cab.obj_id], SOFT_POLICY_SCORE_FLOOR)

    def test_right_of_soft_keeps_all_satisfied_scored_above_floor(self) -> None:
        objs, cabinets, _door = _scene_left_right()
        executor = QueryExecutor(objects=objs)
        sc = _sc("right_of", ["door"], policy=ExecutionPolicy.SOFT)

        out, scores, _soft = executor._apply_spatial_constraint(cabinets, sc)

        self.assertEqual(len(out), 3)
        for cab in out:
            self.assertGreater(scores[cab.obj_id], SOFT_POLICY_SCORE_FLOOR)

    def test_soft_no_anchor_keeps_all_with_unit_score(self) -> None:
        cabinets = [_mk_obj(i, "cabinet", (float(i), 0.0, 0.0)) for i in range(2)]
        executor = QueryExecutor(objects=cabinets)
        sc = _sc("right_of", ["door"], policy=ExecutionPolicy.SOFT)

        out, scores, _soft = executor._apply_spatial_constraint(cabinets, sc)

        self.assertEqual({o.obj_id for o in out}, {0, 1})
        for cab in out:
            self.assertEqual(scores[cab.obj_id], 1.0)


class TestRankOnlyPolicyKeepsAllAndSurfacesSoftScore(unittest.TestCase):
    def test_left_of_rank_only_keeps_all_with_unit_mult_score(self) -> None:
        objs, cabinets, _door = _scene_left_right()
        executor = QueryExecutor(objects=objs)
        sc = _sc("left_of", ["door"], policy=ExecutionPolicy.RANK_ONLY)

        out, scores, soft = executor._apply_spatial_constraint(cabinets, sc)

        self.assertEqual({o.obj_id for o in out}, {0, 1, 2})
        for cab in out:
            self.assertEqual(scores[cab.obj_id], 1.0)
            self.assertEqual(soft[cab.obj_id], 0.0)

    def test_right_of_rank_only_surfaces_positive_soft_scores(self) -> None:
        objs, cabinets, _door = _scene_left_right()
        executor = QueryExecutor(objects=objs)
        sc = _sc("right_of", ["door"], policy=ExecutionPolicy.RANK_ONLY)

        out, scores, soft = executor._apply_spatial_constraint(cabinets, sc)

        self.assertEqual(len(out), 3)
        for cab in out:
            self.assertEqual(scores[cab.obj_id], 1.0)
            self.assertGreater(soft[cab.obj_id], 0.0)


class TestExecutorAggregatesSoftScoresIntoMetadata(unittest.TestCase):
    def test_rank_only_constraint_records_soft_scores_on_result(self) -> None:
        objs, cabinets, _door = _scene_left_right()
        executor = QueryExecutor(objects=objs)

        gq = GroundingQuery(
            raw_query="cabinet right_of door (rank_only)",
            root=QueryNode(
                categories=["cabinet"],
                spatial_constraints=[
                    _sc("right_of", ["door"], policy=ExecutionPolicy.RANK_ONLY)
                ],
                node_id="root",
            ),
            expect_unique=False,
        )
        result = executor.execute(gq)

        self.assertEqual({o.obj_id for o in result.matched_objects}, {0, 1, 2})
        self.assertIn("soft_match_scores", result.metadata)
        soft = result.metadata["soft_match_scores"]
        self.assertTrue(set(soft.keys()).issuperset({0, 1, 2}))


class TestSelectConstraintPolicy(unittest.TestCase):
    def test_superlative_hard_collapses_to_one(self) -> None:
        objs, cabinets, _door = _scene_left_right()
        executor = QueryExecutor(objects=objs)

        sel = SelectConstraint(
            constraint_type="superlative",
            metric="x_position",
            order="max",
            reference=None,
            position=None,
        )
        gq = GroundingQuery(
            raw_query="rightmost cabinet",
            root=QueryNode(
                categories=["cabinet"],
                select_constraint=sel,
                node_id="root",
            ),
            expect_unique=False,
        )
        result = executor.execute(gq)
        self.assertEqual(len(result.matched_objects), 1)
        self.assertEqual(result.matched_objects[0].obj_id, 2)

    def test_superlative_soft_keeps_all_with_descending_scores(self) -> None:
        objs, cabinets, _door = _scene_left_right()
        executor = QueryExecutor(objects=objs)

        sel = SelectConstraint(
            constraint_type="superlative",
            metric="x_position",
            order="max",
            reference=None,
            position=None,
            execution_policy=ExecutionPolicy.SOFT,
        )
        gq = GroundingQuery(
            raw_query="rightmost cabinet soft",
            root=QueryNode(
                categories=["cabinet"],
                select_constraint=sel,
                node_id="root",
            ),
            expect_unique=False,
        )
        result = executor.execute(gq)

        self.assertEqual(len(result.matched_objects), 3)
        self.assertEqual(result.matched_objects[0].obj_id, 2)
        score_seq = [result.scores[o.obj_id] for o in result.matched_objects]
        self.assertEqual(score_seq, sorted(score_seq, reverse=True))


class TestNormalizeViewpointPolicies(unittest.TestCase):
    """Selector-level demotion: directional w/o context -> rank_only."""

    def _make_selector(self):
        # Minimal mock selector exposing only the helper under test
        from query_scene.keyframe_selector import KeyframeSelector

        class _Stub(KeyframeSelector):
            def __init__(self) -> None:
                self.scene_categories = ["cabinet", "door", "loveseat"]

        return _Stub()

    def test_directional_without_context_demoted(self) -> None:
        selector = self._make_selector()
        gq = GroundingQuery(
            raw_query="cabinet on the right",
            root=QueryNode(
                categories=["cabinet"],
                spatial_constraints=[
                    _sc("right_of", ["door"]),
                ],
                node_id="root",
            ),
            expect_unique=True,
        )
        out = selector._normalize_viewpoint_policies(gq)
        sc = out.root.spatial_constraints[0]
        self.assertEqual(sc.reference_frame, ReferenceFrame.AMBIGUOUS)
        self.assertEqual(sc.execution_policy, ExecutionPolicy.RANK_ONLY)
        self.assertIsNone(sc.viewpoint_context_id)

    def test_world_non_directional_not_touched(self) -> None:
        selector = self._make_selector()
        gq = GroundingQuery(
            raw_query="cabinet near door",
            root=QueryNode(
                categories=["cabinet"],
                spatial_constraints=[_sc("near", ["door"])],
                node_id="root",
            ),
            expect_unique=True,
        )
        out = selector._normalize_viewpoint_policies(gq)
        sc = out.root.spatial_constraints[0]
        self.assertEqual(sc.reference_frame, ReferenceFrame.WORLD)
        self.assertEqual(sc.execution_policy, ExecutionPolicy.HARD)

    def test_world_distance_select_constraint_not_touched(self) -> None:
        selector = self._make_selector()
        sel = SelectConstraint(
            constraint_type="superlative",
            metric="distance",
            order="min",
            reference=QueryNode(categories=["door"], node_id="sel_ref"),
            position=None,
        )
        gq = GroundingQuery(
            raw_query="cabinet closest to the door",
            root=QueryNode(
                categories=["cabinet"],
                select_constraint=sel,
                node_id="root",
            ),
            expect_unique=True,
        )
        out = selector._normalize_viewpoint_policies(gq)
        out_sel = out.root.select_constraint
        self.assertIsNotNone(out_sel)
        self.assertEqual(out_sel.reference_frame, ReferenceFrame.WORLD)
        self.assertEqual(out_sel.execution_policy, ExecutionPolicy.HARD)

    def test_viewer_with_resolved_context_not_touched(self) -> None:
        selector = self._make_selector()
        ctx = ViewpointContext(
            id="vp1",
            kind=ViewpointKind.FACING_ANCHOR,
            facing_anchor=QueryNode(categories=["door"], node_id="vp_facing"),
            raw_phrase="facing the door",
        )
        gq = GroundingQuery(
            raw_query="facing the door, cabinet on the right",
            root=QueryNode(
                categories=["cabinet"],
                spatial_constraints=[
                    _sc(
                        "right_of",
                        ["door"],
                        policy=ExecutionPolicy.SOFT,
                        frame=ReferenceFrame.VIEWER,
                        ctx_id="vp1",
                    )
                ],
                node_id="root",
            ),
            expect_unique=True,
            viewpoint_contexts=[ctx],
        )
        out = selector._normalize_viewpoint_policies(gq)
        sc = out.root.spatial_constraints[0]
        self.assertEqual(sc.reference_frame, ReferenceFrame.VIEWER)
        self.assertEqual(sc.execution_policy, ExecutionPolicy.SOFT)
        self.assertEqual(sc.viewpoint_context_id, "vp1")

    def test_viewer_with_ungrounded_context_demoted(self) -> None:
        selector = self._make_selector()
        # facing_anchor's QueryNode is UNKNOW-only: ungrounded context
        ctx = ViewpointContext(
            id="vp1",
            kind=ViewpointKind.FACING_ANCHOR,
            facing_anchor=QueryNode(categories=["UNKNOW"], node_id="vp_facing"),
            raw_phrase="facing the mystery thing",
        )
        gq = GroundingQuery(
            raw_query="facing mystery, cabinet on the right",
            root=QueryNode(
                categories=["cabinet"],
                spatial_constraints=[
                    _sc(
                        "right_of",
                        ["door"],
                        policy=ExecutionPolicy.SOFT,
                        frame=ReferenceFrame.VIEWER,
                        ctx_id="vp1",
                    )
                ],
                node_id="root",
            ),
            expect_unique=True,
            viewpoint_contexts=[ctx],
        )
        out = selector._normalize_viewpoint_policies(gq)
        sc = out.root.spatial_constraints[0]
        self.assertEqual(sc.reference_frame, ReferenceFrame.AMBIGUOUS)
        self.assertEqual(sc.execution_policy, ExecutionPolicy.RANK_ONLY)
        self.assertIsNone(sc.viewpoint_context_id)


class TestRankTargetObjectsByScore(unittest.TestCase):
    """Joint coverage consumer: target ordering reflects scores + soft scores."""

    def _make_selector(self):
        from query_scene.keyframe_selector import KeyframeSelector

        class _Stub(KeyframeSelector):
            def __init__(self) -> None:
                pass

        return _Stub()

    def test_legacy_uniform_scores_preserve_order(self) -> None:
        selector = self._make_selector()
        objs = [_mk_obj(i, "cabinet", (float(i), 0, 0)) for i in range(3)]
        from query_scene.query_executor import ExecutionResult

        result = ExecutionResult(
            node_id="r",
            matched_objects=objs,
            scores={o.obj_id: 1.0 for o in objs},
        )
        ranked = selector._rank_target_objects_by_score(result)
        self.assertEqual([o.obj_id for o in ranked], [0, 1, 2])

    def test_soft_scores_reorder_by_score_desc(self) -> None:
        selector = self._make_selector()
        objs = [_mk_obj(i, "cabinet", (float(i), 0, 0)) for i in range(3)]
        from query_scene.query_executor import ExecutionResult

        result = ExecutionResult(
            node_id="r",
            matched_objects=objs,
            scores={0: 0.001, 1: 0.5, 2: 1.0},
        )
        ranked = selector._rank_target_objects_by_score(result)
        self.assertEqual([o.obj_id for o in ranked], [2, 1, 0])

    def test_rank_only_uses_soft_score_as_tiebreaker(self) -> None:
        selector = self._make_selector()
        objs = [_mk_obj(i, "cabinet", (float(i), 0, 0)) for i in range(3)]
        from query_scene.query_executor import ExecutionResult

        result = ExecutionResult(
            node_id="r",
            matched_objects=objs,
            scores={o.obj_id: 1.0 for o in objs},
            metadata={
                "soft_match_scores": {
                    0: {"sc0:right_of:rank_only": 0.0},
                    1: {"sc0:right_of:rank_only": 0.3},
                    2: {"sc0:right_of:rank_only": 0.9},
                }
            },
        )
        ranked = selector._rank_target_objects_by_score(result)
        self.assertEqual([o.obj_id for o in ranked], [2, 1, 0])


class TestParseQueryHypothesesViewpointNormalizeOptIn(unittest.TestCase):
    """Fix-1 regression: default parse_query_hypotheses must NOT silently
    apply the viewpoint policy floor. ``select_keyframes_v2(viewpoint_aware
    =False)`` is the supported no-drift production path; that path must
    NOT mutate world/hard directional constraints into ambiguous/rank_only.
    """

    def _make_selector(self):
        from query_scene.keyframe_selector import KeyframeSelector

        class _Stub(KeyframeSelector):
            def __init__(self) -> None:
                self.scene_categories = ["cabinet", "door"]

        return _Stub()

    def _normalize_via_parse_helper(
        self, apply_viewpoint_normalize: bool
    ) -> SpatialConstraint:
        # Build a grounding query as if the parser had emitted a legacy
        # world/hard directional constraint, then run it through the part of
        # parse_query_hypotheses that handles per-hypothesis post-processing.
        # We do not call parse_query_hypotheses end-to-end because it would
        # require a real LLM round-trip.
        selector = self._make_selector()
        gq = GroundingQuery(
            raw_query="cabinet to the right of the door",
            root=QueryNode(
                categories=["cabinet"],
                spatial_constraints=[_sc("right_of", ["door"])],
                node_id="root",
            ),
            expect_unique=True,
        )
        sanitized = selector._sanitize_grounding_query_categories(gq)
        if apply_viewpoint_normalize:
            sanitized = selector._normalize_viewpoint_policies(sanitized)
        return sanitized.root.spatial_constraints[0]

    def test_default_no_normalize_preserves_world_hard_directional(self) -> None:
        sc = self._normalize_via_parse_helper(apply_viewpoint_normalize=False)
        self.assertEqual(sc.reference_frame, ReferenceFrame.WORLD)
        self.assertEqual(sc.execution_policy, ExecutionPolicy.HARD)

    def test_explicit_normalize_demotes_world_hard_directional(self) -> None:
        sc = self._normalize_via_parse_helper(apply_viewpoint_normalize=True)
        self.assertEqual(sc.reference_frame, ReferenceFrame.AMBIGUOUS)
        self.assertEqual(sc.execution_policy, ExecutionPolicy.RANK_ONLY)


class TestViewerFrameUnresolvedPoseNoWorldFabrication(unittest.TestCase):
    """Fix-3: when viewer-frame is requested but the pose does not resolve,
    the executor MUST NOT silently fall through to world-frame evaluation.
    Instead it applies the constraint's declared policy with neutral
    evidence.
    """

    def _build_scene_and_query(
        self, policy: ExecutionPolicy
    ) -> tuple[QueryExecutor, GroundingQuery]:
        cabinets = [_mk_obj(i, "cabinet", (float(i), 0.0, 0.0)) for i in range(3)]
        # Note: NO door in the scene, so the viewpoint context's facing
        # anchor cannot resolve and the viewer pose will be None.
        executor = QueryExecutor(objects=cabinets)

        ctx = ViewpointContext(
            id="vp_missing",
            kind=ViewpointKind.FACING_ANCHOR,
            facing_anchor=QueryNode(
                categories=["UNKNOW"], node_id="vp_missing_facing"
            ),
            raw_phrase="facing the missing thing",
        )
        sc = SpatialConstraint(
            relation="right_of",
            anchors=[QueryNode(categories=["cabinet"], node_id="anchor")],
            reference_frame=ReferenceFrame.VIEWER,
            viewpoint_context_id="vp_missing",
            execution_policy=policy,
        )
        gq = GroundingQuery(
            raw_query="facing missing, cabinet on the right of the cabinet",
            root=QueryNode(
                categories=["cabinet"],
                spatial_constraints=[sc],
                node_id="root",
            ),
            expect_unique=False,
            viewpoint_contexts=[ctx],
        )
        return executor, gq

    def test_spatial_viewer_hard_unresolved_returns_empty_no_world_fallback(
        self,
    ) -> None:
        executor, gq = self._build_scene_and_query(ExecutionPolicy.HARD)
        result = executor.execute(gq)
        # World-frame fallback would have returned at least one cabinet that
        # is right_of another cabinet. We require empty.
        self.assertEqual(result.matched_objects, [])

    def test_spatial_viewer_soft_unresolved_keeps_all_with_floor_scores(
        self,
    ) -> None:
        executor, gq = self._build_scene_and_query(ExecutionPolicy.SOFT)
        result = executor.execute(gq)
        self.assertEqual({o.obj_id for o in result.matched_objects}, {0, 1, 2})
        # SOFT floor (no satisfaction) means every score is the floor.
        for obj in result.matched_objects:
            self.assertAlmostEqual(
                result.scores[obj.obj_id], 1e-3, places=6
            )

    def test_spatial_viewer_rank_only_unresolved_keeps_all_with_zero_soft(
        self,
    ) -> None:
        executor, gq = self._build_scene_and_query(ExecutionPolicy.RANK_ONLY)
        result = executor.execute(gq)
        self.assertEqual({o.obj_id for o in result.matched_objects}, {0, 1, 2})
        for obj in result.matched_objects:
            self.assertEqual(result.scores[obj.obj_id], 1.0)
        soft = result.metadata.get("soft_match_scores", {})
        for obj in result.matched_objects:
            entries = soft.get(obj.obj_id, {})
            self.assertTrue(
                all(v == 0.0 for v in entries.values()),
                f"expected 0.0 soft scores for obj {obj.obj_id}, got {entries}",
            )


class TestSelectViewerFrameUnresolvedPoseNoWorldFabrication(unittest.TestCase):
    def _build_scene_and_query(
        self, policy: ExecutionPolicy
    ) -> tuple[QueryExecutor, GroundingQuery]:
        # Three cabinets at strictly increasing world X; if we silently fell
        # back to world x_position with order=max we would pick obj_id=2.
        # Viewer pose is unresolvable because the facing anchor uses UNKNOW.
        cabinets = [_mk_obj(i, "cabinet", (float(i), 0.0, 0.0)) for i in range(3)]
        executor = QueryExecutor(objects=cabinets)

        ctx = ViewpointContext(
            id="vp_missing",
            kind=ViewpointKind.FACING_ANCHOR,
            facing_anchor=QueryNode(
                categories=["UNKNOW"], node_id="vp_missing_facing"
            ),
            raw_phrase="facing the missing thing",
        )
        sel = SelectConstraint(
            constraint_type="superlative",
            metric="x_position",
            order="max",
            reference=None,
            position=None,
            reference_frame=ReferenceFrame.VIEWER,
            viewpoint_context_id="vp_missing",
            execution_policy=policy,
        )
        gq = GroundingQuery(
            raw_query="rightmost cabinet in viewer frame",
            root=QueryNode(
                categories=["cabinet"],
                select_constraint=sel,
                node_id="root",
            ),
            # expect_unique=False so the executor doesn't collapse to a
            # single best object via expect_unique handling; we want to
            # verify the select stage itself keeps all candidates when the
            # viewer pose is unresolvable.
            expect_unique=False,
            viewpoint_contexts=[ctx],
        )
        return executor, gq

    def test_select_viewer_hard_unresolved_returns_empty_no_world_fallback(
        self,
    ) -> None:
        executor, gq = self._build_scene_and_query(ExecutionPolicy.HARD)
        result = executor.execute(gq)
        self.assertEqual(result.matched_objects, [])

    def test_select_viewer_soft_unresolved_keeps_all_without_metric_rerank(
        self,
    ) -> None:
        executor, gq = self._build_scene_and_query(ExecutionPolicy.SOFT)
        result = executor.execute(gq)
        # SOFT keeps all candidates and leaves their incoming mult-score
        # untouched (no fabricated viewer-axis rank).
        self.assertEqual({o.obj_id for o in result.matched_objects}, {0, 1, 2})
        # Incoming score for every candidate was 1.0 (no spatial constraints).
        # Neutral select must preserve that.
        for obj in result.matched_objects:
            self.assertEqual(result.scores[obj.obj_id], 1.0)

    def test_select_viewer_rank_only_unresolved_keeps_all_with_unit_scores(
        self,
    ) -> None:
        executor, gq = self._build_scene_and_query(ExecutionPolicy.RANK_ONLY)
        result = executor.execute(gq)
        self.assertEqual({o.obj_id for o in result.matched_objects}, {0, 1, 2})
        for obj in result.matched_objects:
            self.assertEqual(result.scores[obj.obj_id], 1.0)


if __name__ == "__main__":
    unittest.main()
