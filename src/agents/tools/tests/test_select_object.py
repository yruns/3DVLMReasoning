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


class TestHandleSelectObject:
    def test_success(self) -> None:
        obj = FakeSceneObject(
            obj_id=5,
            category="chair",
            centroid=[1, 2, 3],
            pcd_np=np.array([[0.5, 1.5, 2.5], [1.5, 2.5, 3.5]]),
        )
        runtime = FakeRuntimeState(vg_scene_objects=[obj])
        response = handle_select_object(runtime, object_id=5, rationale="test rationale")
        assert "Object selected" in response
        assert "chair" in response
        # bbox center should equal mean of the 2 pcd points = [1.0, 2.0, 3.0]
        assert runtime.vg_selected_object_id == 5
        assert runtime.vg_selected_bbox_3d[:3] == [1.0, 2.0, 3.0]
        assert runtime.vg_selection_rationale == "test rationale"

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


def test_compute_bbox_3d_raises_when_no_pcd() -> None:
    """No-fallback rule: object without point cloud must raise, not default."""
    from agents.tools.select_object import compute_bbox_3d

    obj = FakeSceneObject(obj_id=42, category="lamp", pcd_np=None)
    with pytest.raises(ValueError, match="object 42 has no pcd"):
        compute_bbox_3d(obj)


def test_compute_bbox_3d_raises_on_empty_pcd() -> None:
    from agents.tools.select_object import compute_bbox_3d

    obj = FakeSceneObject(obj_id=7, category="cup", pcd_np=np.zeros((0, 3)))
    with pytest.raises(ValueError, match="object 7 has no pcd"):
        compute_bbox_3d(obj)


def test_handle_select_object_surfaces_no_pcd_error() -> None:
    """Error must propagate to the agent as a tool ERROR string."""
    from agents.tools.select_object import handle_select_object

    obj = FakeSceneObject(obj_id=3, category="picture", pcd_np=None)
    runtime = FakeRuntimeState(vg_scene_objects=[obj])
    response = handle_select_object(runtime, object_id=3, rationale="test")
    assert response.startswith("ERROR:")
    assert "no pcd" in response
