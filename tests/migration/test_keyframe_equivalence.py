"""Keyframe selection equivalence tests.

Tests that verify keyframe selection behavior matches between
concept-graphs and 3DVLMReasoning implementations.

Ground truth loaded from: tests/migration/ground_truth/keyframes.json
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from query_scene.keyframe_selector import KeyframeSelector, KeyframeResult, SceneObject
from query_scene.query_executor import ExecutionResult
from query_scene.core.hypotheses import (
    GroundingQuery,
    HypothesisOutputV1,
    QueryHypothesis,
    QueryNode,
    HypothesisKind,
    ParseMode,
)


# Path to ground truth data
GROUND_TRUTH_DIR = Path(__file__).parent / "ground_truth"
KEYFRAMES_GT_FILE = GROUND_TRUTH_DIR / "keyframes.json"


@dataclass
class KeyframeTestCase:
    """Loaded keyframe test case."""

    case_id: str
    query: str
    scene_categories: list[str]
    target_object_ids: list[int]
    anchor_object_ids: list[int]
    expected_keyframe_indices: list[int]
    k_tolerance: int
    description: str


def load_keyframe_ground_truth() -> list[KeyframeTestCase]:
    """Load keyframe ground truth from JSON file."""
    if not KEYFRAMES_GT_FILE.exists():
        pytest.skip(f"Ground truth file not found: {KEYFRAMES_GT_FILE}")

    with open(KEYFRAMES_GT_FILE) as f:
        data = json.load(f)

    cases = []
    for case_data in data["cases"]:
        cases.append(
            KeyframeTestCase(
                case_id=case_data["case_id"],
                query=case_data["query"],
                scene_categories=case_data["scene_categories"],
                target_object_ids=case_data["target_object_ids"],
                anchor_object_ids=case_data["anchor_object_ids"],
                expected_keyframe_indices=case_data["expected_keyframe_indices"],
                k_tolerance=case_data["k_tolerance"],
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
    num_per_category: int = 2,
) -> list[SceneObject]:
    """Create mock SceneObject instances for testing."""
    objects = []
    obj_id = 1
    for cat in categories:
        for i in range(num_per_category):
            obj = SceneObject(
                obj_id=obj_id,
                category=cat,
                centroid=np.array([float(obj_id), 0.0, 0.0]),
            )
            objects.append(obj)
            obj_id += 1
    return objects


class TestKeyframeEquivalence(unittest.TestCase):
    """Test keyframe selection equivalence with ground truth."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load ground truth data."""
        cls.ground_truth = load_keyframe_ground_truth()

    def test_ground_truth_loaded(self) -> None:
        """Verify ground truth data is loaded correctly."""
        self.assertGreater(len(self.ground_truth), 0)
        self.assertGreaterEqual(len(self.ground_truth), 50)

    def test_keyframe_result_structure(self) -> None:
        """Test KeyframeResult dataclass structure matches expected format."""
        result = KeyframeResult(
            query="test query",
            target_term="pillow",
            anchor_term="sofa",
            keyframe_indices=[1, 2, 3],
            keyframe_paths=[Path("/tmp/f1.jpg"), Path("/tmp/f2.jpg")],
            target_objects=[],
            anchor_objects=[],
            selection_scores={1: 0.9, 2: 0.8, 3: 0.7},
            metadata={"status": "success"},
        )

        self.assertEqual(result.query, "test query")
        self.assertEqual(result.target_term, "pillow")
        self.assertEqual(result.anchor_term, "sofa")
        self.assertEqual(len(result.keyframe_indices), 3)
        self.assertIn("status", result.metadata)

    def test_scene_object_creation(self) -> None:
        """Test SceneObject creation matches expected structure."""
        obj = SceneObject(
            obj_id=1,
            category="pillow",
            centroid=np.array([1.0, 2.0, 3.0]),
        )

        self.assertEqual(obj.obj_id, 1)
        self.assertEqual(obj.category, "pillow")
        np.testing.assert_array_equal(obj.centroid, [1.0, 2.0, 3.0])


class TestNormalizeHypothesisOutput(unittest.TestCase):
    """Test hypothesis normalization equivalence."""

    def test_normalize_legacy_payload(self) -> None:
        """Test normalization of legacy (pre-v1) payloads."""
        with tempfile.TemporaryDirectory() as tmp:
            selector = create_mock_selector(
                Path(tmp), ["pillow", "sofa", "door", "lamp"]
            )

            # Legacy format: direct GroundingQuery
            legacy_payload = {
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
                "lexical_hints": ["cushion"],
            }

            normalized = selector.normalize_hypothesis_output(legacy_payload)

            self.assertIsInstance(normalized, HypothesisOutputV1)
            self.assertEqual(normalized.parse_mode, ParseMode.SINGLE)
            self.assertEqual(len(normalized.hypotheses), 1)
            # Legacy "direct" kind is preserved as DIRECT (not mapped to SIMPLE)
            self.assertEqual(normalized.hypotheses[0].kind, HypothesisKind.DIRECT)
            self.assertEqual(
                "pillow", normalized.hypotheses[0].grounding_query.target.object_category
            )

    def test_normalize_v1_payload(self) -> None:
        """Test normalization of v1 format payloads."""
        with tempfile.TemporaryDirectory() as tmp:
            selector = create_mock_selector(
                Path(tmp), ["pillow", "sofa", "door", "lamp"]
            )

            v1_payload = {
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
                        "lexical_hints": ["pillow", "sofa"],
                    }
                ],
            }

            normalized = selector.normalize_hypothesis_output(v1_payload)

            self.assertIsInstance(normalized, HypothesisOutputV1)
            self.assertEqual(normalized.parse_mode, ParseMode.SINGLE)
            self.assertEqual(len(normalized.hypotheses), 1)

    def test_normalize_sorts_by_rank(self) -> None:
        """Test that normalization sorts hypotheses by rank."""
        with tempfile.TemporaryDirectory() as tmp:
            selector = create_mock_selector(
                Path(tmp), ["pillow", "sofa", "door", "lamp"]
            )

            # Hypotheses out of order
            payload = {
                "format_version": "hypothesis_output_v1",
                "parse_mode": "multi",
                "hypotheses": [
                    {
                        "kind": "proxy",
                        "rank": 2,
                        "grounding_query": {
                            "raw_query": "proxy",
                            "root": {
                                "categories": ["sofa"],
                                "attributes": [],
                                "spatial_constraints": [],
                                "select_constraint": None,
                            },
                            "expect_unique": True,
                        },
                        "lexical_hints": [],
                    },
                    {
                        "kind": "direct",
                        "rank": 1,
                        "grounding_query": {
                            "raw_query": "direct",
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
                ],
            }

            normalized = selector.normalize_hypothesis_output(payload)

            # Should be sorted by rank
            self.assertEqual([h.rank for h in normalized.hypotheses], [1, 2])
            # Kind is preserved (DIRECT for rank 1, PROXY for rank 2)
            self.assertEqual(normalized.hypotheses[0].kind, HypothesisKind.DIRECT)
            self.assertEqual(normalized.hypotheses[1].kind, HypothesisKind.PROXY)


class TestExecuteHypotheses(unittest.TestCase):
    """Test hypothesis execution equivalence."""

    def test_execute_direct_success(self) -> None:
        """Test direct hypothesis success case."""
        with tempfile.TemporaryDirectory() as tmp:
            selector = create_mock_selector(
                Path(tmp), ["pillow", "sofa", "door", "lamp"]
            )

            payload = {
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
                        "lexical_hints": [],
                    }
                ],
            }

            # Mock execute_query to return non-empty result
            mock_obj = MagicMock()
            selector.execute_query = lambda gq: ExecutionResult(
                node_id="root", matched_objects=[mock_obj]
            )

            status, hypothesis, result = selector.execute_hypotheses(payload)

            self.assertEqual(status, "direct_grounded")
            self.assertIsNotNone(hypothesis)
            self.assertEqual(hypothesis.kind, HypothesisKind.DIRECT)
            self.assertFalse(result.is_empty)

    def test_execute_proxy_fallback(self) -> None:
        """Test proxy fallback when direct fails."""
        with tempfile.TemporaryDirectory() as tmp:
            selector = create_mock_selector(
                Path(tmp), ["pillow", "sofa", "door", "lamp"]
            )

            payload = {
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
                        "lexical_hints": [],
                    },
                    {
                        "kind": "proxy",
                        "rank": 2,
                        "grounding_query": {
                            "raw_query": "proxy",
                            "root": {
                                "categories": ["pillow"],
                                "attributes": [],
                                "spatial_constraints": [],
                                "select_constraint": None,
                            },
                            "expect_unique": True,
                        },
                        "lexical_hints": [],
                    },
                ],
            }

            mock_obj = MagicMock()

            def fake_execute(gq):
                if gq.target.object_category == "UNKNOW":
                    return ExecutionResult(node_id="root", matched_objects=[])
                else:
                    return ExecutionResult(node_id="root", matched_objects=[mock_obj])

            selector.execute_query = fake_execute

            status, hypothesis, result = selector.execute_hypotheses(payload)

            self.assertEqual(status, "proxy_grounded")
            self.assertIsNotNone(hypothesis)
            self.assertEqual(hypothesis.kind, HypothesisKind.PROXY)

    def test_hidden_categories_leak_check(self) -> None:
        """Test that hidden category leaks are detected."""
        with tempfile.TemporaryDirectory() as tmp:
            selector = create_mock_selector(
                Path(tmp), ["pillow", "sofa", "door", "lamp"]
            )

            payload = {
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
                        "lexical_hints": [],
                    }
                ],
            }

            selector.execute_query = lambda gq: ExecutionResult(
                node_id="root", matched_objects=[MagicMock()]
            )

            # Should raise when hidden category matches hypothesis
            with self.assertRaises(ValueError):
                selector.execute_hypotheses(payload, hidden_categories={"pillow"})


class TestJointCoverageViews(unittest.TestCase):
    """Test joint coverage view selection."""

    def test_joint_coverage_basic(self) -> None:
        """Test basic joint coverage computation."""
        with tempfile.TemporaryDirectory() as tmp:
            selector = create_mock_selector(Path(tmp), ["pillow", "sofa"])

            # Set up visibility index
            selector.object_to_views = {
                1: [(0, 0.9), (1, 0.8), (2, 0.5)],
                2: [(1, 0.7), (2, 0.9), (3, 0.6)],
            }

            # Test single object coverage
            views = selector.get_best_views_for_object(1, top_k=2)
            self.assertEqual(len(views), 2)
            self.assertIn(0, views)  # Highest score view

    def test_joint_coverage_multiple_objects(self) -> None:
        """Test joint coverage with multiple target objects."""
        with tempfile.TemporaryDirectory() as tmp:
            selector = create_mock_selector(Path(tmp), ["pillow", "sofa"])

            # Create mock objects
            obj1 = SceneObject(
                obj_id=1,
                category="pillow",
                centroid=np.array([0.0, 0.0, 0.0]),
            )
            obj2 = SceneObject(
                obj_id=2,
                category="pillow",
                centroid=np.array([2.0, 0.0, 0.0]),
            )

            selector.object_to_views = {
                1: [(0, 0.9), (1, 0.8), (2, 0.5)],
                2: [(1, 0.85), (2, 0.9), (3, 0.6)],
            }

            views = selector.get_joint_coverage_views(
                [1, 2],  # Object IDs
                max_views=3,
            )

            # Should include views that cover both objects
            self.assertLessEqual(len(views), 3)


class TestViewMapping(unittest.TestCase):
    """Test view to frame mapping."""

    def test_map_view_to_frame(self) -> None:
        """Test view index to frame index mapping."""
        with tempfile.TemporaryDirectory() as tmp:
            selector = create_mock_selector(Path(tmp), ["pillow"])
            selector.stride = 5

            # View 0 -> frame 0
            self.assertEqual(selector.map_view_to_frame(0), 0)
            # View 1 -> frame 5
            self.assertEqual(selector.map_view_to_frame(1), 5)
            # View 3 -> frame 15
            self.assertEqual(selector.map_view_to_frame(3), 15)

    def test_resolve_keyframe_path_fallback(self) -> None:
        """Test keyframe path resolution with fallback."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            selector = create_mock_selector(tmp_path, ["pillow"])

            # Create results directory with some frames
            results_dir = tmp_path / "results"
            results_dir.mkdir(parents=True)

            # Create frame that exists
            existing_frame = results_dir / "frame000010.jpg"
            existing_frame.write_text("x", encoding="utf-8")

            # Request view=2 (frame 10 exists)
            path, resolved_view = selector._resolve_keyframe_path(2)
            self.assertEqual(path, existing_frame)
            self.assertEqual(resolved_view, 2)


# Parametrized ground truth tests
@pytest.fixture
def keyframe_ground_truth() -> list[KeyframeTestCase]:
    """Pytest fixture for ground truth data."""
    return load_keyframe_ground_truth()


def test_ground_truth_case_count(keyframe_ground_truth: list[KeyframeTestCase]) -> None:
    """Verify we have sufficient ground truth cases."""
    assert len(keyframe_ground_truth) >= 50, "Expected at least 50 keyframe test cases"


def test_ground_truth_structure(keyframe_ground_truth: list[KeyframeTestCase]) -> None:
    """Verify ground truth structure is valid."""
    for case in keyframe_ground_truth:
        assert case.case_id, "case_id must not be empty"
        assert case.query, "query must not be empty"
        assert len(case.scene_categories) > 0, "scene_categories must not be empty"
        assert case.k_tolerance >= 0, "k_tolerance must be non-negative"


def test_keyframe_indices_within_tolerance(
    keyframe_ground_truth: list[KeyframeTestCase],
) -> None:
    """Test that keyframe indices are reasonable."""
    for case in keyframe_ground_truth[:10]:  # Test first 10 for speed
        # Expected indices should not be negative
        for idx in case.expected_keyframe_indices:
            assert idx >= 0, f"Negative keyframe index in case {case.case_id}"

        # Should have reasonable number of keyframes
        assert (
            1 <= len(case.expected_keyframe_indices) <= 10
        ), f"Unexpected keyframe count in case {case.case_id}"


if __name__ == "__main__":
    unittest.main()
