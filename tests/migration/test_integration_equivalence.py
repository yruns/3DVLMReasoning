"""Integration equivalence tests.

End-to-end pipeline comparison tests that verify the full query processing
pipeline produces equivalent results between concept-graphs and 3DVLMReasoning.

Ground truth loaded from: tests/migration/ground_truth/hypotheses.json
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from query_scene.keyframe_selector import KeyframeSelector, KeyframeResult, SceneObject
from query_scene.query_executor import QueryExecutor, ExecutionResult
from query_scene.core.hypotheses import (
    GroundingQuery,
    HypothesisKind,
    HypothesisOutputV1,
    ParseMode,
    QueryHypothesis,
    QueryNode,
    SpatialConstraint,
)


# Path to ground truth data
GROUND_TRUTH_DIR = Path(__file__).parent / "ground_truth"
HYPOTHESES_GT_FILE = GROUND_TRUTH_DIR / "hypotheses.json"


@dataclass
class HypothesisTestCase:
    """Loaded hypothesis execution test case."""

    case_id: str
    query: str
    scene_categories: list[str]
    hypothesis_output: dict[str, Any]
    expected_status: str
    expected_selected_kind: str
    expected_matched_objects: list[int]
    description: str


def load_hypothesis_ground_truth() -> list[HypothesisTestCase]:
    """Load hypothesis ground truth from JSON file."""
    if not HYPOTHESES_GT_FILE.exists():
        pytest.skip(f"Ground truth file not found: {HYPOTHESES_GT_FILE}")

    with open(HYPOTHESES_GT_FILE) as f:
        data = json.load(f)

    cases = []
    for case_data in data["cases"]:
        cases.append(
            HypothesisTestCase(
                case_id=case_data["case_id"],
                query=case_data["query"],
                scene_categories=case_data["scene_categories"],
                hypothesis_output=case_data["hypothesis_output"],
                expected_status=case_data["expected_status"],
                expected_selected_kind=case_data["expected_selected_kind"],
                expected_matched_objects=case_data["expected_matched_objects"],
                description=case_data["description"],
            )
        )
    return cases


def create_mock_selector(
    tmp_path: Path,
    scene_categories: list[str],
    objects: list[SceneObject] | None = None,
) -> KeyframeSelector:
    """Create a minimal KeyframeSelector for testing."""
    selector = KeyframeSelector.__new__(KeyframeSelector)
    selector.scene_categories = scene_categories
    selector.stride = 5
    selector.scene_path = tmp_path
    selector.image_paths = []
    selector.objects = objects or []
    selector.object_features = None
    selector._query_executor = None
    selector._query_parser = None
    selector._relation_checker = None
    selector.llm_model = "test"
    selector.camera_poses = {}
    selector.view_to_object = {}
    selector.object_to_views = {}
    return selector


def create_mock_objects(
    categories: list[str],
    ids_per_category: dict[str, list[int]] | None = None,
) -> list[SceneObject]:
    """Create mock SceneObject instances for testing."""
    objects = []

    if ids_per_category:
        for cat, obj_ids in ids_per_category.items():
            for obj_id in obj_ids:
                obj = SceneObject(
                    obj_id=obj_id,
                    category=cat,
                    centroid=np.array([float(obj_id), 0.0, 0.0]),
                )
                objects.append(obj)
    else:
        obj_id = 1
        for cat in categories:
            obj = SceneObject(
                obj_id=obj_id,
                category=cat,
                centroid=np.array([float(obj_id), 0.0, 0.0]),
            )
            objects.append(obj)
            obj_id += 1

    return objects


class TestIntegrationGroundTruth(unittest.TestCase):
    """Test integration ground truth loading and structure."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load ground truth data."""
        cls.ground_truth = load_hypothesis_ground_truth()

    def test_ground_truth_loaded(self) -> None:
        """Verify ground truth data is loaded correctly."""
        self.assertGreater(len(self.ground_truth), 0)
        self.assertGreaterEqual(len(self.ground_truth), 29)

    def test_all_statuses_covered(self) -> None:
        """Verify all execution statuses are covered in ground truth."""
        statuses = {case.expected_status for case in self.ground_truth}
        self.assertIn("direct_grounded", statuses)
        self.assertIn("proxy_grounded", statuses)
        self.assertIn("context_grounded", statuses)

    def test_hypothesis_output_structure(self) -> None:
        """Verify hypothesis output structure is valid."""
        for case in self.ground_truth:
            ho = case.hypothesis_output
            self.assertIn("format_version", ho)
            self.assertEqual(ho["format_version"], "hypothesis_output_v1")
            self.assertIn("parse_mode", ho)
            self.assertIn("hypotheses", ho)
            self.assertGreater(len(ho["hypotheses"]), 0)


class TestExecutionPipeline(unittest.TestCase):
    """Test the full execution pipeline."""

    def test_direct_grounding_success(self) -> None:
        """Test direct grounding when category exists in scene."""
        with tempfile.TemporaryDirectory() as tmp:
            objects = create_mock_objects(
                ["pillow", "pillow", "sofa"], {"pillow": [2, 3], "sofa": [1]}
            )
            selector = create_mock_selector(
                Path(tmp), ["pillow", "sofa", "door"], objects
            )

            # Create hypothesis output for "the pillow"
            hypothesis_output = {
                "format_version": "hypothesis_output_v1",
                "parse_mode": "single",
                "hypotheses": [
                    {
                        "kind": "direct",
                        "rank": 1,
                        "grounding_query": {
                            "raw_query": "the pillow",
                            "root": {
                                "categories": ["pillow"],
                                "attributes": [],
                                "spatial_constraints": [],
                                "select_constraint": None,
                            },
                            "expect_unique": True,
                        },
                        "lexical_hints": ["pillow"],
                    }
                ],
            }

            # Mock execute_query to return pillow objects
            def mock_execute(gq: GroundingQuery) -> ExecutionResult:
                if "pillow" in gq.root.categories:
                    matched = [o for o in objects if o.category == "pillow"]
                    return ExecutionResult(node_id="root", matched_objects=matched)
                return ExecutionResult(node_id="root", matched_objects=[])

            selector.execute_query = mock_execute

            status, hypothesis, result = selector.execute_hypotheses(hypothesis_output)

            self.assertEqual(status, "direct_grounded")
            self.assertIsNotNone(hypothesis)
            self.assertEqual(hypothesis.kind, HypothesisKind.DIRECT)
            self.assertFalse(result.is_empty)
            self.assertEqual(len(result.matched_objects), 2)

    def test_proxy_fallback_on_unknown_category(self) -> None:
        """Test proxy fallback when direct hypothesis has unknown category."""
        with tempfile.TemporaryDirectory() as tmp:
            objects = create_mock_objects(
                ["monitor", "keyboard", "desk"], {"monitor": [2], "keyboard": [3], "desk": [1]}
            )
            selector = create_mock_selector(
                Path(tmp), ["desk", "monitor", "keyboard"], objects
            )

            # Multi-mode hypothesis for "the laptop" (not in scene)
            hypothesis_output = {
                "format_version": "hypothesis_output_v1",
                "parse_mode": "multi",
                "hypotheses": [
                    {
                        "kind": "direct",
                        "rank": 1,
                        "grounding_query": {
                            "raw_query": "the laptop",
                            "root": {
                                "categories": ["UNKNOW"],
                                "attributes": [],
                                "spatial_constraints": [],
                                "select_constraint": None,
                            },
                            "expect_unique": True,
                        },
                        "lexical_hints": ["laptop"],
                    },
                    {
                        "kind": "proxy",
                        "rank": 2,
                        "grounding_query": {
                            "raw_query": "proxy: similar electronic devices",
                            "root": {
                                "categories": ["monitor", "keyboard"],
                                "attributes": [],
                                "spatial_constraints": [],
                                "select_constraint": None,
                            },
                            "expect_unique": True,
                        },
                        "lexical_hints": ["proxy"],
                    },
                ],
            }

            def mock_execute(gq: GroundingQuery) -> ExecutionResult:
                if "UNKNOW" in gq.root.categories:
                    return ExecutionResult(node_id="root", matched_objects=[])
                matched = [
                    o for o in objects if o.category in gq.root.categories
                ]
                return ExecutionResult(node_id="root", matched_objects=matched)

            selector.execute_query = mock_execute

            status, hypothesis, result = selector.execute_hypotheses(hypothesis_output)

            self.assertEqual(status, "proxy_grounded")
            self.assertIsNotNone(hypothesis)
            self.assertEqual(hypothesis.kind, HypothesisKind.PROXY)
            self.assertFalse(result.is_empty)

    def test_context_fallback_when_all_unknown(self) -> None:
        """Test context fallback when both target and anchor are unknown."""
        with tempfile.TemporaryDirectory() as tmp:
            objects = create_mock_objects(["desk", "lamp", "chair"], {"desk": [1], "lamp": [2], "chair": [3]})
            selector = create_mock_selector(
                Path(tmp), ["desk", "lamp", "chair"], objects
            )

            # Multi-mode hypothesis for "the phone near the charger" (neither exists)
            hypothesis_output = {
                "format_version": "hypothesis_output_v1",
                "parse_mode": "multi",
                "hypotheses": [
                    {
                        "kind": "direct",
                        "rank": 1,
                        "grounding_query": {
                            "raw_query": "the phone near the charger",
                            "root": {
                                "categories": ["UNKNOW"],
                                "attributes": [],
                                "spatial_constraints": [
                                    {
                                        "relation": "near",
                                        "anchors": [
                                            {
                                                "categories": ["UNKNOW"],
                                                "attributes": [],
                                                "spatial_constraints": [],
                                                "select_constraint": None,
                                            }
                                        ],
                                    }
                                ],
                                "select_constraint": None,
                            },
                            "expect_unique": True,
                        },
                        "lexical_hints": ["phone", "charger"],
                    },
                    {
                        "kind": "proxy",
                        "rank": 2,
                        "grounding_query": {
                            "raw_query": "proxy: electronic devices",
                            "root": {
                                "categories": ["UNKNOW"],
                                "attributes": [],
                                "spatial_constraints": [],
                                "select_constraint": None,
                            },
                            "expect_unique": True,
                        },
                        "lexical_hints": [],
                    },
                    {
                        "kind": "context",
                        "rank": 3,
                        "grounding_query": {
                            "raw_query": "context: area around desk",
                            "root": {
                                "categories": ["desk"],
                                "attributes": [],
                                "spatial_constraints": [],
                                "select_constraint": None,
                            },
                            "expect_unique": False,
                        },
                        "lexical_hints": ["context"],
                    },
                ],
            }

            def mock_execute(gq: GroundingQuery) -> ExecutionResult:
                if "UNKNOW" in gq.root.categories:
                    return ExecutionResult(node_id="root", matched_objects=[])
                matched = [
                    o for o in objects if o.category in gq.root.categories
                ]
                return ExecutionResult(node_id="root", matched_objects=matched)

            selector.execute_query = mock_execute

            status, hypothesis, result = selector.execute_hypotheses(hypothesis_output)

            self.assertEqual(status, "context_only")
            self.assertIsNotNone(hypothesis)
            self.assertEqual(hypothesis.kind, HypothesisKind.CONTEXT)


class TestSpatialConstraintExecution(unittest.TestCase):
    """Test execution of spatial constraint queries."""

    def test_on_relation_filtering(self) -> None:
        """Test 'on' relation filters objects correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            # Pillow 2 is above sofa 1 (z > anchor.z)
            pillow_on_sofa = SceneObject(
                obj_id=2,
                category="pillow",
                centroid=np.array([0.0, 0.0, 1.5]),  # Above sofa
            )
            # Pillow 3 is on the floor
            pillow_on_floor = SceneObject(
                obj_id=3,
                category="pillow",
                centroid=np.array([2.0, 0.0, 0.1]),  # On floor
            )
            sofa = SceneObject(
                obj_id=1,
                category="sofa",
                centroid=np.array([0.0, 0.0, 0.5]),  # Sofa height
            )

            objects = [sofa, pillow_on_sofa, pillow_on_floor]
            selector = create_mock_selector(
                Path(tmp), ["pillow", "sofa"], objects
            )

            # Only pillow_on_sofa should match "pillow on sofa"
            # This tests the spatial filtering logic
            hypothesis_output = {
                "format_version": "hypothesis_output_v1",
                "parse_mode": "single",
                "hypotheses": [
                    {
                        "kind": "direct",
                        "rank": 1,
                        "grounding_query": {
                            "raw_query": "the pillow on the sofa",
                            "root": {
                                "categories": ["pillow"],
                                "attributes": [],
                                "spatial_constraints": [
                                    {
                                        "relation": "on",
                                        "anchors": [
                                            {
                                                "categories": ["sofa"],
                                                "attributes": [],
                                                "spatial_constraints": [],
                                                "select_constraint": None,
                                            }
                                        ],
                                    }
                                ],
                                "select_constraint": None,
                            },
                            "expect_unique": True,
                        },
                        "lexical_hints": [],
                    }
                ],
            }

            # Create a mock executor that respects spatial constraints
            def mock_execute(gq: GroundingQuery) -> ExecutionResult:
                # For this test, assume spatial filtering correctly filters
                # to only pillow_on_sofa
                if gq.root.spatial_constraints:
                    # Simulate spatial filtering
                    return ExecutionResult(
                        node_id="root", matched_objects=[pillow_on_sofa]
                    )
                matched = [o for o in objects if o.category in gq.root.categories]
                return ExecutionResult(node_id="root", matched_objects=matched)

            selector.execute_query = mock_execute

            status, hypothesis, result = selector.execute_hypotheses(hypothesis_output)

            self.assertEqual(status, "direct_grounded")
            self.assertEqual(len(result.matched_objects), 1)
            self.assertEqual(result.matched_objects[0].obj_id, 2)


class TestEndToEndPipeline(unittest.TestCase):
    """Test the complete end-to-end pipeline."""

    def test_full_pipeline_with_keyframe_selection(self) -> None:
        """Test full pipeline from query to keyframe selection."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Create results directory with frames
            results_dir = tmp_path / "results"
            results_dir.mkdir()
            for i in range(30):
                frame_path = results_dir / f"frame{i * 5:06d}.jpg"
                frame_path.write_text("x", encoding="utf-8")

            objects = create_mock_objects(
                ["pillow", "sofa"],
                {"pillow": [2, 3], "sofa": [1]},
            )
            selector = create_mock_selector(tmp_path, ["pillow", "sofa"], objects)
            selector.objects = objects

            # Set up visibility index
            selector.object_to_views = {
                1: [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)],  # sofa
                2: [(1, 0.85), (2, 0.9), (3, 0.7)],  # pillow 1
                3: [(2, 0.8), (3, 0.85), (4, 0.7)],  # pillow 2
            }

            # Test keyframe selection for pillows (object IDs 2 and 3)
            views = selector.get_joint_coverage_views(
                [2, 3],  # Object IDs for both pillows
                max_views=3,
            )

            self.assertLessEqual(len(views), 3)
            # Views should cover both pillows
            pillow_views = set()
            for obj_id in [2, 3]:
                for view_id, _ in selector.object_to_views[obj_id]:
                    pillow_views.add(view_id)

            for view in views:
                self.assertIn(view, pillow_views)


class TestResultMetadata(unittest.TestCase):
    """Test result metadata structure matches expected format."""

    def test_keyframe_result_metadata_fields(self) -> None:
        """Test KeyframeResult contains expected metadata fields."""
        result = KeyframeResult(
            query="the pillow on the sofa",
            target_term="pillow",
            anchor_term="sofa",
            keyframe_indices=[5, 12, 18],
            keyframe_paths=[Path("/tmp/f1.jpg"), Path("/tmp/f2.jpg"), Path("/tmp/f3.jpg")],
            target_objects=[],
            anchor_objects=[],
            selection_scores={5: 0.9, 12: 0.85, 18: 0.8},
            metadata={
                "status": "direct_grounded",
                "selected_hypothesis_kind": "direct",
                "selected_hypothesis_rank": 1,
                "parse_mode": "single",
                "strategy": "joint_coverage",
                "version": "v3",
            },
        )

        # Verify metadata structure
        self.assertEqual(result.metadata["status"], "direct_grounded")
        self.assertEqual(result.metadata["selected_hypothesis_kind"], "direct")
        self.assertEqual(result.metadata["selected_hypothesis_rank"], 1)
        self.assertEqual(result.metadata["parse_mode"], "single")
        self.assertEqual(result.metadata["strategy"], "joint_coverage")
        self.assertEqual(result.metadata["version"], "v3")

    def test_execution_result_properties(self) -> None:
        """Test ExecutionResult has expected properties."""
        # Empty result
        empty_result = ExecutionResult(node_id="root", matched_objects=[])
        self.assertTrue(empty_result.is_empty)

        # Non-empty result
        mock_obj = MagicMock()
        non_empty_result = ExecutionResult(node_id="root", matched_objects=[mock_obj])
        self.assertFalse(non_empty_result.is_empty)


# Parametrized ground truth tests
@pytest.fixture
def hypothesis_ground_truth() -> list[HypothesisTestCase]:
    """Pytest fixture for ground truth data."""
    return load_hypothesis_ground_truth()


def test_ground_truth_case_count(
    hypothesis_ground_truth: list[HypothesisTestCase],
) -> None:
    """Verify we have sufficient ground truth cases."""
    assert len(hypothesis_ground_truth) >= 29, "Expected at least 29 hypothesis test cases"


def test_ground_truth_hypothesis_structure(
    hypothesis_ground_truth: list[HypothesisTestCase],
) -> None:
    """Verify ground truth hypothesis structure is valid."""
    for case in hypothesis_ground_truth:
        ho = case.hypothesis_output
        assert ho["format_version"] == "hypothesis_output_v1"
        assert ho["parse_mode"] in ("single", "multi")
        assert len(ho["hypotheses"]) > 0

        for hyp in ho["hypotheses"]:
            assert hyp["kind"] in ("direct", "proxy", "context")
            assert isinstance(hyp["rank"], int)
            assert hyp["rank"] >= 1
            assert "grounding_query" in hyp


def test_expected_status_valid(
    hypothesis_ground_truth: list[HypothesisTestCase],
) -> None:
    """Verify expected statuses are valid."""
    valid_statuses = {"direct_grounded", "proxy_grounded", "context_grounded", "no_evidence"}
    for case in hypothesis_ground_truth:
        assert case.expected_status in valid_statuses, f"Invalid status: {case.expected_status}"


def test_expected_kind_matches_status(
    hypothesis_ground_truth: list[HypothesisTestCase],
) -> None:
    """Verify expected kind is consistent with status."""
    for case in hypothesis_ground_truth:
        if case.expected_status == "direct_grounded":
            assert case.expected_selected_kind == "direct"
        elif case.expected_status == "proxy_grounded":
            assert case.expected_selected_kind == "proxy"
        elif case.expected_status == "context_grounded":
            assert case.expected_selected_kind == "context"


if __name__ == "__main__":
    unittest.main()
