"""Phase-0 tests: viewpoint-aware additions to HypothesisOutputV1.

These tests pin the contract that:
- All new fields have defaults that reproduce legacy world-frame behavior, so
  cached pre-viewpoint parser outputs continue to validate unchanged.
- The JSON Schema accepts the new fields and rejects malformed shapes.
- GroundingQuery rejects dangling viewpoint_context_id references and
  non-world reference_frames without a context id.
- GroundingQuery.get_all_categories walks viewpoint context anchors so the
  UNKNOW invariant and hidden-category mask leak guard apply to them.
"""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as jsonschema_validate
from pydantic import ValidationError as PydanticValidationError

from query_scene.core import (
    DIRECTIONAL_RELATIONS,
    ExecutionPolicy,
    GroundingQuery,
    HypothesisOutputV1,
    ReferenceFrame,
    SelectConstraint,
    SpatialConstraint,
    ViewpointContext,
    ViewpointKind,
)

SCHEMA_PATH = (
    Path(__file__).resolve().parents[3] / "schema" / "hypothesis_output_v1.json"
)
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

SCENE_CATEGORIES = ["pillow", "sofa", "door", "cabinet", "loveseat", "window"]


def _node(categories: list[str], node_id: str = "") -> dict:
    return {
        "categories": categories,
        "attributes": [],
        "spatial_constraints": [],
        "select_constraint": None,
        "node_id": node_id,
    }


def _legacy_payload() -> dict:
    """Payload identical to a pre-viewpoint parser output."""
    return {
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
                                "anchors": [_node(["sofa"], node_id="anchor")],
                            }
                        ],
                        "select_constraint": None,
                        "node_id": "root",
                    },
                    "expect_unique": True,
                },
                "lexical_hints": [],
            }
        ],
    }


def _viewpoint_payload(
    *,
    relation: str = "right_of",
    frame: str = "viewer",
    policy: str = "soft",
    ctx_id: str | None = "vp1",
    include_ctx: bool = True,
    facing_categories: list[str] | None = None,
) -> dict:
    facing_categories = facing_categories or ["door"]
    payload = _legacy_payload()
    payload["hypotheses"][0]["grounding_query"]["raw_query"] = (
        "facing the door, the cabinet on the right"
    )
    payload["hypotheses"][0]["grounding_query"]["root"]["categories"] = ["cabinet"]
    payload["hypotheses"][0]["grounding_query"]["root"]["spatial_constraints"] = [
        {
            "relation": relation,
            "anchors": [_node(["door"], node_id="anchor")],
            "reference_frame": frame,
            "viewpoint_context_id": ctx_id,
            "execution_policy": policy,
        }
    ]
    if include_ctx:
        payload["hypotheses"][0]["grounding_query"]["viewpoint_contexts"] = [
            {
                "id": "vp1",
                "kind": "facing_anchor",
                "facing_anchor": _node(facing_categories, node_id="vp_facing"),
                "raw_phrase": "facing the door",
                "confidence": "explicit",
            }
        ]
    return payload


class TestViewpointDefaultsAreLegacy(unittest.TestCase):
    """Defaults on all new fields must reproduce pre-viewpoint behavior."""

    def test_legacy_payload_still_validates_against_schema_and_pydantic(self) -> None:
        payload = _legacy_payload()
        jsonschema_validate(instance=payload, schema=SCHEMA)
        parsed = HypothesisOutputV1.model_validate(payload)
        parsed.validate_categories(SCENE_CATEGORIES)

    def test_spatial_constraint_defaults_reproduce_world_hard(self) -> None:
        sc = SpatialConstraint.model_validate(
            {"relation": "on", "anchors": [_node(["sofa"])]}
        )
        self.assertEqual(sc.reference_frame, ReferenceFrame.WORLD)
        self.assertIsNone(sc.viewpoint_context_id)
        self.assertEqual(sc.execution_policy, ExecutionPolicy.HARD)

    def test_select_constraint_defaults_reproduce_world_hard(self) -> None:
        sel = SelectConstraint.model_validate(
            {
                "constraint_type": "superlative",
                "metric": "x_position",
                "order": "max",
                "reference": None,
                "position": None,
            }
        )
        self.assertEqual(sel.reference_frame, ReferenceFrame.WORLD)
        self.assertIsNone(sel.viewpoint_context_id)
        self.assertEqual(sel.execution_policy, ExecutionPolicy.HARD)

    def test_grounding_query_defaults_empty_viewpoint_contexts(self) -> None:
        parsed = HypothesisOutputV1.model_validate(_legacy_payload())
        gq = parsed.hypotheses[0].grounding_query
        self.assertEqual(gq.viewpoint_contexts, [])


class TestViewpointPayloadValidates(unittest.TestCase):
    def test_viewer_frame_with_context_validates(self) -> None:
        payload = _viewpoint_payload()
        jsonschema_validate(instance=payload, schema=SCHEMA)
        parsed = HypothesisOutputV1.model_validate(payload)
        parsed.validate_categories(SCENE_CATEGORIES)

        gq = parsed.hypotheses[0].grounding_query
        self.assertEqual(len(gq.viewpoint_contexts), 1)
        self.assertEqual(gq.viewpoint_contexts[0].id, "vp1")
        self.assertEqual(gq.viewpoint_contexts[0].kind, ViewpointKind.FACING_ANCHOR)

        sc = gq.root.spatial_constraints[0]
        self.assertEqual(sc.reference_frame, ReferenceFrame.VIEWER)
        self.assertEqual(sc.viewpoint_context_id, "vp1")
        self.assertEqual(sc.execution_policy, ExecutionPolicy.SOFT)

    def test_select_constraint_viewer_axis_validates(self) -> None:
        payload = _legacy_payload()
        payload["hypotheses"][0]["grounding_query"]["root"]["categories"] = ["window"]
        payload["hypotheses"][0]["grounding_query"]["root"]["spatial_constraints"] = []
        payload["hypotheses"][0]["grounding_query"]["root"]["select_constraint"] = {
            "constraint_type": "superlative",
            "metric": "x_position",
            "order": "max",
            "reference": None,
            "position": None,
            "reference_frame": "viewer",
            "viewpoint_context_id": "vp1",
            "execution_policy": "soft",
        }
        payload["hypotheses"][0]["grounding_query"]["viewpoint_contexts"] = [
            {
                "id": "vp1",
                "kind": "facing_anchor_set",
                "facing_anchor": _node(["window"]),
                "raw_phrase": "facing the windows",
                "confidence": "explicit",
            }
        ]
        jsonschema_validate(instance=payload, schema=SCHEMA)
        parsed = HypothesisOutputV1.model_validate(payload)
        parsed.validate_categories(SCENE_CATEGORIES)

    def test_object_local_with_subject_anchor_validates(self) -> None:
        payload = _legacy_payload()
        payload["hypotheses"][0]["grounding_query"]["root"]["categories"] = ["pillow"]
        payload["hypotheses"][0]["grounding_query"]["root"]["spatial_constraints"] = [
            {
                "relation": "in_front_of",
                "anchors": [_node(["sofa"])],
                "reference_frame": "object_local",
                "viewpoint_context_id": "vp_local",
                "execution_policy": "rank_only",
            }
        ]
        payload["hypotheses"][0]["grounding_query"]["viewpoint_contexts"] = [
            {
                "id": "vp_local",
                "kind": "object_local",
                "subject_anchor": _node(["pillow"]),
                "facing_anchor": _node(["sofa"]),
                "raw_phrase": "pillow facing the sofa",
                "confidence": "explicit",
            }
        ]
        jsonschema_validate(instance=payload, schema=SCHEMA)
        parsed = HypothesisOutputV1.model_validate(payload)
        parsed.validate_categories(SCENE_CATEGORIES)

    def test_ambiguous_frame_without_ctx_validates(self) -> None:
        payload = _legacy_payload()
        payload["hypotheses"][0]["grounding_query"]["root"]["categories"] = ["cabinet"]
        payload["hypotheses"][0]["grounding_query"]["root"]["spatial_constraints"] = [
            {
                "relation": "left_of",
                "anchors": [_node(["sofa"])],
                "reference_frame": "ambiguous",
                "viewpoint_context_id": None,
                "execution_policy": "rank_only",
            }
        ]
        jsonschema_validate(instance=payload, schema=SCHEMA)
        parsed = HypothesisOutputV1.model_validate(payload)
        parsed.validate_categories(SCENE_CATEGORIES)


class TestViewpointValidationRejections(unittest.TestCase):
    def test_dangling_context_id_rejected(self) -> None:
        payload = _viewpoint_payload(include_ctx=False)
        with self.assertRaises(PydanticValidationError):
            HypothesisOutputV1.model_validate(payload)

    def test_viewer_frame_without_context_id_rejected(self) -> None:
        payload = _viewpoint_payload(ctx_id=None)
        with self.assertRaises(PydanticValidationError):
            HypothesisOutputV1.model_validate(payload)

    def test_object_local_without_context_id_rejected(self) -> None:
        payload = _viewpoint_payload(frame="object_local", ctx_id=None)
        with self.assertRaises(PydanticValidationError):
            HypothesisOutputV1.model_validate(payload)

    def test_duplicate_context_ids_rejected(self) -> None:
        payload = _viewpoint_payload()
        ctx = payload["hypotheses"][0]["grounding_query"]["viewpoint_contexts"][0]
        payload["hypotheses"][0]["grounding_query"]["viewpoint_contexts"].append(
            deepcopy(ctx)
        )
        with self.assertRaises(PydanticValidationError):
            HypothesisOutputV1.model_validate(payload)

    def test_unknown_reference_frame_rejected_by_schema(self) -> None:
        payload = _viewpoint_payload(frame="upside_down")
        with self.assertRaises(JsonSchemaValidationError):
            jsonschema_validate(instance=payload, schema=SCHEMA)

    def test_unknown_execution_policy_rejected_by_schema(self) -> None:
        payload = _viewpoint_payload(policy="lenient")
        with self.assertRaises(JsonSchemaValidationError):
            jsonschema_validate(instance=payload, schema=SCHEMA)


class TestCategoryWalkCoversViewpointAnchors(unittest.TestCase):
    """UNKNOW invariant and hidden-mask leak guard must reach viewpoint anchors."""

    def test_viewpoint_facing_anchor_unknow_category_caught(self) -> None:
        payload = _viewpoint_payload(facing_categories=["mystery_thing"])
        with self.assertRaises(ValueError):
            HypothesisOutputV1.model_validate(payload).validate_categories(
                SCENE_CATEGORIES
            )

    def test_viewpoint_anchor_hidden_category_caught(self) -> None:
        payload = _viewpoint_payload(facing_categories=["pillow"])
        parsed = HypothesisOutputV1.model_validate(payload)
        with self.assertRaises(ValueError):
            parsed.validate_no_mask_leak({"pillow"})


class TestPublicEnums(unittest.TestCase):
    def test_directional_relations_constant_matches_expected_set(self) -> None:
        self.assertEqual(
            DIRECTIONAL_RELATIONS,
            frozenset({"left_of", "right_of", "in_front_of", "behind"}),
        )

    def test_reference_frame_enum_values(self) -> None:
        self.assertEqual(
            {f.value for f in ReferenceFrame},
            {"world", "viewer", "object_local", "ambiguous"},
        )

    def test_execution_policy_enum_values(self) -> None:
        self.assertEqual(
            {p.value for p in ExecutionPolicy},
            {"hard", "soft", "rank_only"},
        )

    def test_viewpoint_kind_enum_values(self) -> None:
        self.assertEqual(
            {k.value for k in ViewpointKind},
            {
                "facing_anchor",
                "facing_anchor_set",
                "entering_from",
                "standing_at",
                "object_local",
            },
        )


class TestIterConstraintsHelper(unittest.TestCase):
    def test_iter_constraints_yields_spatial_and_select(self) -> None:
        payload = _legacy_payload()
        payload["hypotheses"][0]["grounding_query"]["root"]["select_constraint"] = {
            "constraint_type": "superlative",
            "metric": "size",
            "order": "max",
            "reference": None,
            "position": None,
        }
        parsed = HypothesisOutputV1.model_validate(payload)
        gq = parsed.hypotheses[0].grounding_query
        pairs = list(gq.iter_constraints())
        kinds = sorted(type(c).__name__ for _node, c in pairs)
        self.assertEqual(kinds, ["SelectConstraint", "SpatialConstraint"])


if __name__ == "__main__":
    unittest.main()
