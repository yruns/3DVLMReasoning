"""Tests for the spatial_compare VG tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from agents.tools.spatial_compare import (
    find_objects_by_category,
    handle_spatial_compare,
)


@dataclass
class FakeSceneObject:
    obj_id: int
    category: str
    centroid: list[float] = field(default_factory=lambda: [0, 0, 0])


@dataclass
class FakeRuntimeState:
    vg_scene_objects: list[Any] | None = None
    vg_axis_align_matrix: Any = None


class TestFindObjectsByCategory:
    def test_exact_match(self):
        objs = [
            FakeSceneObject(0, "chair"),
            FakeSceneObject(1, "table"),
            FakeSceneObject(2, "chair"),
        ]
        result = find_objects_by_category(objs, "chair")
        assert len(result) == 2

    def test_substring_match(self):
        objs = [
            FakeSceneObject(0, "office chair"),
            FakeSceneObject(1, "desk"),
        ]
        result = find_objects_by_category(objs, "chair")
        assert len(result) == 1
        assert result[0].category == "office chair"

    def test_case_insensitive(self):
        objs = [FakeSceneObject(0, "Chair")]
        result = find_objects_by_category(objs, "chair")
        assert len(result) == 1

    def test_no_match(self):
        objs = [FakeSceneObject(0, "table")]
        assert find_objects_by_category(objs, "chair") == []

    def test_skips_background(self):
        objs = [FakeSceneObject(0, "wall"), FakeSceneObject(1, "floor")]
        assert find_objects_by_category(objs, "wall") == []


class TestHandleSpatialCompare:
    def _make_state(self):
        objs = [
            FakeSceneObject(0, "chair", [0, 0, 0]),
            FakeSceneObject(1, "chair", [3, 0, 0]),
            FakeSceneObject(2, "chair", [10, 0, 0]),
            FakeSceneObject(3, "desk", [1, 0, 0]),
        ]
        return FakeRuntimeState(vg_scene_objects=objs)

    def test_closest_to(self):
        state = self._make_state()
        result = handle_spatial_compare(
            state, "chair", "closest_to", "desk"
        )
        assert "CLOSEST" in result
        # Chair at (0,0,0) is 1m from desk at (1,0,0) — closest
        assert "[ID=0]" in result.split("CLOSEST")[0]

    def test_farthest_from(self):
        state = self._make_state()
        result = handle_spatial_compare(
            state, "chair", "farthest_from", "desk"
        )
        assert "FARTHEST" in result
        # Chair at (10,0,0) is 9m from desk — farthest
        assert "[ID=2]" in result.split("FARTHEST")[0]

    def test_no_targets(self):
        state = self._make_state()
        result = handle_spatial_compare(
            state, "lamp", "closest_to", "desk"
        )
        assert "ERROR" in result
        assert "lamp" in result

    def test_no_anchors(self):
        state = self._make_state()
        result = handle_spatial_compare(
            state, "chair", "closest_to", "sofa"
        )
        assert "ERROR" in result
        assert "sofa" in result

    def test_no_scene_objects(self):
        state = FakeRuntimeState()
        result = handle_spatial_compare(
            state, "chair", "closest_to", "desk"
        )
        assert "ERROR" in result

    def test_with_axis_align(self):
        objs = [
            FakeSceneObject(0, "chair", [0, 0, 0]),
            FakeSceneObject(1, "desk", [1, 0, 0]),
        ]
        mat = np.eye(4)
        mat[:3, 3] = [10, 20, 30]  # Translation
        state = FakeRuntimeState(
            vg_scene_objects=objs, vg_axis_align_matrix=mat
        )
        result = handle_spatial_compare(
            state, "chair", "closest_to", "desk"
        )
        # Distance should be 1m (translation doesn't affect relative distance)
        assert "1.00m" in result
