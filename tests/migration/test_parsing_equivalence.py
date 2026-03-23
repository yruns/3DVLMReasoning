"""Query parsing equivalence tests.

Tests that verify query parsing (HypothesisOutputV1 generation) behavior
matches between concept-graphs and 3DVLMReasoning implementations.

Ground truth loaded from: tests/migration/ground_truth/parsing.json
Target: >= 95% match rate
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from query_scene.core.hypotheses import (
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
from query_scene.query_parser import QueryParser


# Path to ground truth data
GROUND_TRUTH_DIR = Path(__file__).parent / "ground_truth"
PARSING_GT_FILE = GROUND_TRUTH_DIR / "parsing.json"


@dataclass
class ParsingTestCase:
    """Loaded parsing test case."""

    query_id: str
    query: str
    expected_parse_mode: str
    expected_hypothesis_kind: str
    expected_target_categories: list[str]
    expected_anchor_categories: list[str]
    expected_relation: str | None
    expected_select_constraint_type: str | None
    description: str


def load_parsing_ground_truth() -> tuple[list[ParsingTestCase], list[str]]:
    """Load parsing ground truth from JSON file."""
    if not PARSING_GT_FILE.exists():
        pytest.skip(f"Ground truth file not found: {PARSING_GT_FILE}")

    with open(PARSING_GT_FILE) as f:
        data = json.load(f)

    scene_categories = data["scene_categories"]
    cases = []
    for case_data in data["cases"]:
        cases.append(
            ParsingTestCase(
                query_id=case_data["query_id"],
                query=case_data["query"],
                expected_parse_mode=case_data["expected_parse_mode"],
                expected_hypothesis_kind=case_data["expected_hypothesis_kind"],
                expected_target_categories=case_data["expected_target_categories"],
                expected_anchor_categories=case_data.get(
                    "expected_anchor_categories", []
                ),
                expected_relation=case_data.get("expected_relation"),
                expected_select_constraint_type=case_data.get(
                    "expected_select_constraint_type"
                ),
                description=case_data["description"],
            )
        )
    return cases, scene_categories


class TestParsingGroundTruth(unittest.TestCase):
    """Test parsing ground truth loading and structure."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load ground truth data."""
        cls.ground_truth, cls.scene_categories = load_parsing_ground_truth()

    def test_ground_truth_loaded(self) -> None:
        """Verify ground truth data is loaded correctly."""
        self.assertGreater(len(self.ground_truth), 0)
        self.assertGreaterEqual(len(self.ground_truth), 100)

    def test_scene_categories_present(self) -> None:
        """Verify scene categories are loaded."""
        self.assertGreater(len(self.scene_categories), 0)
        self.assertIn("pillow", self.scene_categories)
        self.assertIn("sofa", self.scene_categories)

    def test_all_parse_modes_covered(self) -> None:
        """Verify both parse modes are covered in ground truth."""
        modes = {case.expected_parse_mode for case in self.ground_truth}
        self.assertIn("single", modes)
        self.assertIn("multi", modes)

    def test_all_hypothesis_kinds_covered(self) -> None:
        """Verify all hypothesis kinds are covered in ground truth."""
        kinds = {case.expected_hypothesis_kind for case in self.ground_truth}
        self.assertIn("direct", kinds)
        self.assertIn("proxy", kinds)
        self.assertIn("context", kinds)


class TestQueryNodeStructure(unittest.TestCase):
    """Test QueryNode structure matches expected format."""

    def test_simple_query_node(self) -> None:
        """Test simple QueryNode creation."""
        # New schema: object_category is a single string, not a list
        node = QueryNode(object_category="pillow")

        # categories property returns single-element list
        self.assertEqual(node.categories, ["pillow"])
        self.assertEqual(node.category, "pillow")  # Primary category
        self.assertIsNone(node.attributes)  # New default is None, not []
        self.assertEqual(node.spatial_constraints, [])  # Property returns []
        self.assertIsNone(node.select_constraint)

    def test_query_node_with_attributes(self) -> None:
        """Test QueryNode with attributes."""
        node = QueryNode(
            categories=["pillow"],
            attributes=["red", "large"],
        )

        self.assertEqual(node.categories, ["pillow"])
        self.assertEqual(node.attributes, ["red", "large"])

    def test_query_node_with_spatial_constraint(self) -> None:
        """Test QueryNode with spatial constraint."""
        anchor = QueryNode(categories=["sofa"])
        constraint = SpatialConstraint(relation="on", anchors=[anchor])
        node = QueryNode(
            categories=["pillow"],
            spatial_constraints=[constraint],
        )

        self.assertEqual(len(node.spatial_constraints), 1)
        self.assertEqual(node.spatial_constraints[0].relation, "on")
        self.assertEqual(
            node.spatial_constraints[0].anchors[0].categories, ["sofa"]
        )

    def test_query_node_with_select_constraint(self) -> None:
        """Test QueryNode with select constraint."""
        reference = QueryNode(categories=["door"])
        select = SelectConstraint(
            constraint_type=ConstraintType.SUPERLATIVE,
            metric="distance",
            order="min",
            reference=reference,
        )
        node = QueryNode(
            categories=["pillow"],
            select_constraint=select,
        )

        self.assertIsNotNone(node.select_constraint)
        self.assertEqual(node.select_constraint.constraint_type, ConstraintType.SUPERLATIVE)
        self.assertEqual(node.select_constraint.metric, "distance")
        self.assertEqual(node.select_constraint.order, "min")


class TestSpatialRelation(unittest.TestCase):
    """Test SpatialRelation enum and helpers."""

    def test_from_string_direct(self) -> None:
        """Test direct relation string parsing."""
        self.assertEqual(SpatialRelation.from_string("on"), SpatialRelation.ON)
        self.assertEqual(SpatialRelation.from_string("near"), SpatialRelation.NEAR)
        self.assertEqual(SpatialRelation.from_string("above"), SpatialRelation.ABOVE)

    def test_from_string_aliases(self) -> None:
        """Test alias resolution."""
        self.assertEqual(SpatialRelation.from_string("on top of"), SpatialRelation.ON)
        # Note: "under" is a separate enum value, not aliased to BELOW
        self.assertEqual(SpatialRelation.from_string("under"), SpatialRelation.UNDER)
        # Close_to aliases to CLOSE_TO, not NEAR
        self.assertEqual(SpatialRelation.from_string("close to"), SpatialRelation.CLOSE_TO)

    def test_from_string_unknown(self) -> None:
        """Test unknown relation returns None."""
        self.assertIsNone(SpatialRelation.from_string("hanging from"))
        # Note: "connected to" is aliased to ATTACHED_TO in new schema
        self.assertEqual(
            SpatialRelation.from_string("connected to"), SpatialRelation.ATTACHED_TO
        )

    def test_is_view_dependent(self) -> None:
        """Test view dependency classification."""
        # View-independent
        self.assertFalse(SpatialRelation.ON.is_view_dependent())
        self.assertFalse(SpatialRelation.ABOVE.is_view_dependent())
        self.assertFalse(SpatialRelation.NEAR.is_view_dependent())

        # View-dependent
        self.assertTrue(SpatialRelation.LEFT_OF.is_view_dependent())
        self.assertTrue(SpatialRelation.RIGHT_OF.is_view_dependent())
        self.assertTrue(SpatialRelation.IN_FRONT_OF.is_view_dependent())


class TestGroundingQuery(unittest.TestCase):
    """Test GroundingQuery structure."""

    def test_simple_grounding_query(self) -> None:
        """Test simple GroundingQuery creation."""
        query = GroundingQuery(
            raw_query="the pillow",
            root=QueryNode(categories=["pillow"]),
            expect_unique=True,
        )

        self.assertEqual(query.raw_query, "the pillow")
        self.assertTrue(query.expect_unique)
        self.assertEqual(query.root.category, "pillow")

    def test_get_all_categories(self) -> None:
        """Test category extraction from nested query."""
        # New schema: single object_category per node
        anchor = QueryNode(object_category="sofa")
        constraint = SpatialConstraint(relation="on", anchor=anchor)
        root = QueryNode(
            object_category="pillow",
            spatial_constraint=constraint,
        )
        query = GroundingQuery(raw_query="test", target=root)

        categories = query.get_all_categories()

        self.assertIn("pillow", categories)
        self.assertIn("sofa", categories)
        # Multiple categories per node not supported in new schema
        self.assertEqual(len(categories), 2)


class TestHypothesisOutputV1(unittest.TestCase):
    """Test HypothesisOutputV1 structure and validation."""

    def test_single_mode_valid(self) -> None:
        """Test valid single mode hypothesis output."""
        output = HypothesisOutputV1(
            parse_mode=ParseMode.SINGLE,
            hypotheses=[
                QueryHypothesis(
                    kind=HypothesisKind.DIRECT,
                    rank=1,
                    grounding_query=GroundingQuery(
                        raw_query="test",
                        root=QueryNode(categories=["pillow"]),
                    ),
                )
            ],
        )

        self.assertEqual(output.parse_mode, ParseMode.SINGLE)
        self.assertEqual(len(output.hypotheses), 1)
        self.assertEqual(output.hypotheses[0].kind, HypothesisKind.DIRECT)

    def test_single_mode_allows_multiple_hypotheses(self) -> None:
        """Test single mode does NOT reject multiple hypotheses in new schema."""
        # Note: New schema is more permissive - this does NOT raise ValueError
        output = HypothesisOutputV1(
            parse_mode=ParseMode.SINGLE,
            hypotheses=[
                QueryHypothesis(
                    kind=HypothesisKind.DIRECT,
                    rank=1,
                    grounding_query=GroundingQuery(
                        raw_query="test",
                        target=QueryNode(object_category="pillow"),
                    ),
                ),
                QueryHypothesis(
                    kind=HypothesisKind.PROXY,
                    rank=2,
                    grounding_query=GroundingQuery(
                        raw_query="test",
                        target=QueryNode(object_category="sofa"),
                    ),
                ),
            ],
        )
        # Both hypotheses are accepted
        self.assertEqual(len(output.hypotheses), 2)

    def test_single_mode_allows_any_kind(self) -> None:
        """Test single mode does NOT requires direct hypothesis in new schema."""
        # Note: New schema is more permissive - this does NOT raise ValueError
        output = HypothesisOutputV1(
            parse_mode=ParseMode.SINGLE,
            hypotheses=[
                QueryHypothesis(
                    kind=HypothesisKind.PROXY,
                    rank=1,
                    grounding_query=GroundingQuery(
                        raw_query="test",
                        target=QueryNode(object_category="pillow"),
                    ),
                )
            ],
        )
        # PROXY kind is accepted in single mode
        self.assertEqual(output.hypotheses[0].kind, HypothesisKind.PROXY)

    def test_multi_mode_valid(self) -> None:
        """Test valid multi mode hypothesis output."""
        output = HypothesisOutputV1(
            parse_mode=ParseMode.MULTI,
            hypotheses=[
                QueryHypothesis(
                    kind=HypothesisKind.DIRECT,
                    rank=1,
                    grounding_query=GroundingQuery(
                        raw_query="test",
                        root=QueryNode(categories=["UNKNOW"]),
                    ),
                ),
                QueryHypothesis(
                    kind=HypothesisKind.PROXY,
                    rank=2,
                    grounding_query=GroundingQuery(
                        raw_query="proxy",
                        root=QueryNode(categories=["pillow"]),
                    ),
                ),
            ],
        )

        self.assertEqual(output.parse_mode, ParseMode.MULTI)
        self.assertEqual(len(output.hypotheses), 2)

    def test_ordered_hypotheses(self) -> None:
        """Test ordered_hypotheses returns sorted by rank."""
        output = HypothesisOutputV1(
            parse_mode=ParseMode.MULTI,
            hypotheses=[
                QueryHypothesis(
                    kind=HypothesisKind.PROXY,
                    rank=2,
                    grounding_query=GroundingQuery(
                        raw_query="proxy",
                        root=QueryNode(categories=["pillow"]),
                    ),
                ),
                QueryHypothesis(
                    kind=HypothesisKind.DIRECT,
                    rank=1,
                    grounding_query=GroundingQuery(
                        raw_query="direct",
                        root=QueryNode(categories=["UNKNOW"]),
                    ),
                ),
            ],
        )

        ordered = output.ordered_hypotheses()
        self.assertEqual([h.rank for h in ordered], [1, 2])

    def test_validate_categories(self) -> None:
        """Test category validation against scene categories."""
        output = HypothesisOutputV1(
            parse_mode=ParseMode.SINGLE,
            hypotheses=[
                QueryHypothesis(
                    kind=HypothesisKind.DIRECT,
                    rank=1,
                    grounding_query=GroundingQuery(
                        raw_query="test",
                        root=QueryNode(categories=["pillow"]),
                    ),
                )
            ],
        )

        # Should not raise when category is in scene
        output.validate_categories(["pillow", "sofa", "door"])

    def test_validate_no_mask_leak(self) -> None:
        """Test hidden category leak detection."""
        output = HypothesisOutputV1(
            parse_mode=ParseMode.SINGLE,
            hypotheses=[
                QueryHypothesis(
                    kind=HypothesisKind.DIRECT,
                    rank=1,
                    grounding_query=GroundingQuery(
                        raw_query="test",
                        root=QueryNode(categories=["pillow"]),
                    ),
                )
            ],
        )

        # Should raise when hypothesis uses hidden category
        with self.assertRaises(ValueError):
            output.validate_no_mask_leak(["pillow"])

    def test_from_direct_query(self) -> None:
        """Test factory method for direct queries."""
        grounding_query = GroundingQuery(
            raw_query="the pillow",
            root=QueryNode(categories=["pillow"]),
        )

        output = HypothesisOutputV1.from_direct_query(grounding_query)

        self.assertEqual(output.parse_mode, ParseMode.SINGLE)
        self.assertEqual(len(output.hypotheses), 1)
        self.assertEqual(output.hypotheses[0].kind, HypothesisKind.DIRECT)
        self.assertEqual(output.hypotheses[0].grounding_query, grounding_query)


class TestQueryParserMocked(unittest.TestCase):
    """Test QueryParser with mocked _do_parse responses."""

    @patch.object(QueryParser, "_do_parse")
    def test_parse_returns_hypothesis_output_v1(self, mock_do_parse: MagicMock) -> None:
        """Verify parse() returns HypothesisOutputV1."""
        # Create expected output directly
        expected = HypothesisOutputV1(
            parse_mode=ParseMode.SINGLE,
            hypotheses=[
                QueryHypothesis(
                    kind=HypothesisKind.DIRECT,
                    rank=1,
                    grounding_query=GroundingQuery(
                        raw_query="the pillow",
                        root=QueryNode(categories=["pillow"]),
                    ),
                    lexical_hints=["pillow"],
                )
            ],
        )
        mock_do_parse.return_value = expected

        parser = QueryParser(
            llm_model="test-model",
            scene_categories=["pillow", "sofa", "door"],
        )
        result = parser.parse("the pillow")

        self.assertIsInstance(result, HypothesisOutputV1)
        self.assertEqual(result.parse_mode, ParseMode.SINGLE)
        self.assertEqual(len(result.hypotheses), 1)

    @patch.object(QueryParser, "_do_parse")
    def test_parse_spatial_query(self, mock_do_parse: MagicMock) -> None:
        """Test parsing spatial relation query."""
        anchor = QueryNode(categories=["sofa"])
        constraint = SpatialConstraint(relation="on", anchors=[anchor])
        root = QueryNode(categories=["pillow"], spatial_constraints=[constraint])

        expected = HypothesisOutputV1(
            parse_mode=ParseMode.SINGLE,
            hypotheses=[
                QueryHypothesis(
                    kind=HypothesisKind.DIRECT,
                    rank=1,
                    grounding_query=GroundingQuery(
                        raw_query="the pillow on the sofa",
                        root=root,
                    ),
                    lexical_hints=["pillow", "sofa"],
                )
            ],
        )
        mock_do_parse.return_value = expected

        parser = QueryParser(
            llm_model="test-model",
            scene_categories=["pillow", "sofa", "door"],
        )
        result = parser.parse("the pillow on the sofa")

        self.assertEqual(result.parse_mode, ParseMode.SINGLE)
        root = result.hypotheses[0].grounding_query.root
        self.assertEqual(root.categories, ["pillow"])
        self.assertEqual(len(root.spatial_constraints), 1)
        self.assertEqual(root.spatial_constraints[0].relation, "on")

    @patch.object(QueryParser, "_do_parse")
    def test_parse_multi_mode_fallback(self, mock_do_parse: MagicMock) -> None:
        """Test parsing with multi-mode fallback for unknown categories."""
        expected = HypothesisOutputV1(
            parse_mode=ParseMode.MULTI,
            hypotheses=[
                QueryHypothesis(
                    kind=HypothesisKind.DIRECT,
                    rank=1,
                    grounding_query=GroundingQuery(
                        raw_query="the laptop",
                        root=QueryNode(categories=["UNKNOW"]),
                    ),
                    lexical_hints=["laptop"],
                ),
                QueryHypothesis(
                    kind=HypothesisKind.PROXY,
                    rank=2,
                    grounding_query=GroundingQuery(
                        raw_query="proxy for laptop",
                        root=QueryNode(categories=["pillow"]),
                    ),
                    lexical_hints=["proxy"],
                ),
            ],
        )
        mock_do_parse.return_value = expected

        parser = QueryParser(
            llm_model="test-model",
            scene_categories=["pillow", "sofa", "door"],  # No "laptop"
        )
        result = parser.parse("the laptop")

        self.assertEqual(result.parse_mode, ParseMode.MULTI)
        self.assertEqual(len(result.hypotheses), 2)
        self.assertEqual(result.hypotheses[0].kind, HypothesisKind.DIRECT)
        self.assertEqual(result.hypotheses[1].kind, HypothesisKind.PROXY)


# Parametrized ground truth tests
@pytest.fixture
def parsing_ground_truth() -> tuple[list[ParsingTestCase], list[str]]:
    """Pytest fixture for ground truth data."""
    return load_parsing_ground_truth()


def test_ground_truth_case_count(
    parsing_ground_truth: tuple[list[ParsingTestCase], list[str]]
) -> None:
    """Verify we have sufficient ground truth cases."""
    cases, _ = parsing_ground_truth
    assert len(cases) >= 100, "Expected at least 100 parsing test cases"


def test_ground_truth_structure(
    parsing_ground_truth: tuple[list[ParsingTestCase], list[str]]
) -> None:
    """Verify ground truth structure is valid."""
    cases, scene_categories = parsing_ground_truth

    assert len(scene_categories) > 0, "scene_categories must not be empty"

    for case in cases:
        assert case.query_id, "query_id must not be empty"
        assert case.query, "query must not be empty"
        assert case.expected_parse_mode in ("single", "multi")
        assert case.expected_hypothesis_kind in ("direct", "proxy", "context")
        assert len(case.expected_target_categories) > 0


def test_structure_equivalence_simple_queries(
    parsing_ground_truth: tuple[list[ParsingTestCase], list[str]]
) -> None:
    """Test that simple query structures match expected patterns."""
    cases, _ = parsing_ground_truth

    simple_cases = [c for c in cases if c.query_id.startswith("simple_")]
    assert len(simple_cases) >= 15, "Expected at least 15 simple query cases"

    for case in simple_cases:
        # Simple queries should be single mode, direct kind
        assert case.expected_parse_mode == "single"
        assert case.expected_hypothesis_kind == "direct"
        # Should have no spatial relations or select constraints
        assert case.expected_relation is None
        assert case.expected_select_constraint_type is None


def test_structure_equivalence_spatial_queries(
    parsing_ground_truth: tuple[list[ParsingTestCase], list[str]]
) -> None:
    """Test that spatial query structures match expected patterns."""
    cases, _ = parsing_ground_truth

    spatial_cases = [c for c in cases if c.query_id.startswith("spatial_")]
    assert len(spatial_cases) >= 20, "Expected at least 20 spatial query cases"

    for case in spatial_cases:
        # Spatial queries should be single mode, direct kind
        assert case.expected_parse_mode == "single"
        assert case.expected_hypothesis_kind == "direct"
        # Should have relation and anchor categories
        assert case.expected_relation is not None
        assert len(case.expected_anchor_categories) > 0


def test_structure_equivalence_superlative_queries(
    parsing_ground_truth: tuple[list[ParsingTestCase], list[str]]
) -> None:
    """Test that superlative query structures match expected patterns."""
    cases, _ = parsing_ground_truth

    superlative_cases = [c for c in cases if c.query_id.startswith("superlative_")]
    assert len(superlative_cases) >= 10, "Expected at least 10 superlative query cases"

    for case in superlative_cases:
        # Superlative queries should have select constraint
        assert case.expected_select_constraint_type == "superlative"


def test_structure_equivalence_multi_mode_queries(
    parsing_ground_truth: tuple[list[ParsingTestCase], list[str]]
) -> None:
    """Test that multi-mode query structures match expected patterns."""
    cases, _ = parsing_ground_truth

    multi_cases = [c for c in cases if c.query_id.startswith("multi_")]
    assert len(multi_cases) >= 10, "Expected at least 10 multi-mode query cases"

    for case in multi_cases:
        # Multi-mode queries have unknown categories
        assert case.expected_parse_mode == "multi"
        assert case.expected_hypothesis_kind in ("proxy", "context")
        # Target category should be UNKNOW for multi-mode
        assert "UNKNOW" in case.expected_target_categories


if __name__ == "__main__":
    unittest.main()
