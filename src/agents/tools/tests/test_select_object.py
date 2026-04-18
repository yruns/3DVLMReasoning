"""Tests for the select_object VG tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from agents.tools.select_object import (
    compute_bbox_3d,
    find_object_by_id,
    handle_select_object,
)


@dataclass
class FakeSceneObject:
    obj_id: int
    category: str
    centroid: list[float] = field(default_factory=lambda: [0, 0, 0])
    pcd_np: Any = None


@dataclass
class FakeRuntimeState:
    vg_scene_objects: list[Any] | None = None
    vg_axis_align_matrix: Any = None
    vg_selected_object_id: int | None = None
    vg_selected_bbox_3d: list[float] | None = None
    vg_selection_rationale: str = ""


class TestFindObjectById:
    def test_found(self):
        objs = [FakeSceneObject(0, "chair"), FakeSceneObject(5, "table")]
        assert find_object_by_id(objs, 5).category == "table"

    def test_not_found(self):
        objs = [FakeSceneObject(0, "chair")]
        assert find_object_by_id(objs, 99) is None

    def test_empty(self):
        assert find_object_by_id([], 0) is None


class TestComputeBbox3d:
    def test_from_pcd(self):
        pts = np.array([[0, 0, 0], [2, 4, 6]], dtype=np.float64)
        obj = FakeSceneObject(0, "box", pcd_np=pts)
        bbox = compute_bbox_3d(obj)
        assert len(bbox) == 9
        assert bbox[0] == pytest.approx(1.0)  # cx
        assert bbox[1] == pytest.approx(2.0)  # cy
        assert bbox[2] == pytest.approx(3.0)  # cz
        assert bbox[3] == pytest.approx(2.0)  # dx
        assert bbox[4] == pytest.approx(4.0)  # dy
        assert bbox[5] == pytest.approx(6.0)  # dz
        assert bbox[6:] == [0.0, 0.0, 0.0]

    def test_from_centroid_fallback(self):
        obj = FakeSceneObject(0, "lamp", centroid=[1.0, 2.0, 3.0])
        bbox = compute_bbox_3d(obj)
        assert bbox[:3] == pytest.approx([1.0, 2.0, 3.0])
        assert bbox[3:6] == pytest.approx([0.3, 0.3, 0.3])

    def test_with_axis_align_identity(self):
        pts = np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float64)
        obj = FakeSceneObject(0, "box", pcd_np=pts)
        identity = np.eye(4)
        bbox = compute_bbox_3d(obj, axis_align_matrix=identity)
        assert bbox[0] == pytest.approx(2.0)

    def test_with_axis_align_translation(self):
        pts = np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float64)
        obj = FakeSceneObject(0, "box", pcd_np=pts)
        mat = np.eye(4)
        mat[:3, 3] = [10, 20, 30]  # Translation
        bbox = compute_bbox_3d(obj, axis_align_matrix=mat)
        assert bbox[0] == pytest.approx(11.0)  # 1.0 + 10
        assert bbox[1] == pytest.approx(21.0)
        assert bbox[2] == pytest.approx(31.0)
        # Extents unchanged by translation
        assert bbox[3] == pytest.approx(2.0)

    def test_centroid_fallback_with_axis_align(self):
        obj = FakeSceneObject(0, "lamp", centroid=[1.0, 2.0, 3.0])
        mat = np.eye(4)
        mat[:3, 3] = [10, 20, 30]
        bbox = compute_bbox_3d(obj, axis_align_matrix=mat)
        assert bbox[:3] == pytest.approx([11.0, 22.0, 33.0])


class TestHandleSelectObject:
    def test_success(self):
        objs = [FakeSceneObject(5, "chair", centroid=[1, 2, 3])]
        state = FakeRuntimeState(vg_scene_objects=objs)
        result = handle_select_object(state, 5, "it is the only chair")
        assert "Object selected" in result
        assert "[ID=5]" in result
        assert state.vg_selected_object_id == 5
        assert state.vg_selected_bbox_3d is not None
        assert len(state.vg_selected_bbox_3d) == 9

    def test_not_found(self):
        objs = [FakeSceneObject(5, "chair")]
        state = FakeRuntimeState(vg_scene_objects=objs)
        result = handle_select_object(state, 99, "")
        assert "ERROR" in result
        assert state.vg_selected_object_id is None

    def test_no_scene_objects(self):
        state = FakeRuntimeState()
        result = handle_select_object(state, 0, "")
        assert "ERROR" in result
