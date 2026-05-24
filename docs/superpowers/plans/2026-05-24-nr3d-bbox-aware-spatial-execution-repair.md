# NR3D BBox-Aware Spatial Execution Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair spatial execution false negatives that clear valid NR3D target candidates before keyframe selection.

**Architecture:** Keep strict execution as the default, but make relation checking bbox-aware and prevent quick filters from being a final empty decision. Add compact per-constraint trace metadata so remaining no-evidence rows can be attributed to parser, anchor, quick-filter, or full-check failure.

**Tech Stack:** Python 3.11 via `.venv`, `unittest`/`pytest`, `numpy`, existing `query_scene` modules.

---

### Task 1: Add Failing BBox-Aware Spatial Relation Tests

**Files:**
- Create: `src/query_scene/tests/test_spatial_checker_bbox.py`
- Test: `src/query_scene/tests/test_spatial_checker_bbox.py`

- [x] **Step 1: Write failing tests for bbox-aware relations**

```python
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from query_scene.retrieval.spatial_checker import SpatialRelationChecker


def _box(min_xyz: tuple[float, float, float], max_xyz: tuple[float, float, float]) -> np.ndarray:
    x0, y0, z0 = min_xyz
    x1, y1, z1 = max_xyz
    return np.array(
        [
            [x0, y0, z0],
            [x0, y0, z1],
            [x0, y1, z0],
            [x0, y1, z1],
            [x1, y0, z0],
            [x1, y0, z1],
            [x1, y1, z0],
            [x1, y1, z1],
        ],
        dtype=np.float32,
    )


def _obj(obj_id: int, category: str, min_xyz: tuple[float, float, float], max_xyz: tuple[float, float, float]):
    bbox = _box(min_xyz, max_xyz)
    return SimpleNamespace(
        obj_id=obj_id,
        category=category,
        object_tag=category,
        centroid=bbox.mean(axis=0),
        bbox_np=bbox,
        bbox_3d=None,
    )


class TestBBoxAwareSpatialChecker(unittest.TestCase):
    def test_inside_uses_bbox_np_anchor(self) -> None:
        checker = SpatialRelationChecker()
        picture = _obj(1, "picture", (1.0, 1.0, 1.0), (1.5, 1.5, 1.5))
        frame = _obj(2, "frame", (0.5, 0.5, 0.5), (2.0, 2.0, 2.0))

        result = checker.check(picture, frame, "inside")

        self.assertTrue(result.satisfies)
        self.assertGreater(result.score, 0.0)
        self.assertTrue(result.details["bbox_used"])

    def test_on_top_of_uses_anchor_footprint_not_centroid_radius(self) -> None:
        checker = SpatialRelationChecker()
        pillow = _obj(1, "pillow", (2.6, 0.2, 0.95), (3.1, 0.7, 1.2))
        bed = _obj(2, "bed", (0.0, 0.0, 0.0), (4.0, 2.0, 1.0))

        result = checker.check(pillow, bed, "on")

        self.assertTrue(result.satisfies)
        self.assertGreater(result.score, 0.0)
        self.assertEqual(result.details["mode"], "bbox_horizontal_support")

    def test_next_to_uses_bbox_gap_not_centroid_distance(self) -> None:
        checker = SpatialRelationChecker()
        window = _obj(1, "window", (2.1, 0.2, 1.2), (2.4, 1.2, 2.5))
        bed = _obj(2, "bed", (0.0, 0.0, 0.0), (2.0, 1.8, 0.7))

        result = checker.check(window, bed, "next_to")

        self.assertTrue(result.satisfies)
        self.assertGreater(result.score, 0.0)
        self.assertEqual(result.details["distance_mode"], "bbox_xy_gap")

    def test_between_searches_all_anchor_pairs(self) -> None:
        checker = SpatialRelationChecker()
        lamp = _obj(1, "lamp", (4.9, 0.0, 0.0), (5.1, 0.2, 0.6))
        wrong_anchor = _obj(2, "bed", (0.0, 5.0, 0.0), (0.5, 5.5, 0.5))
        bed_left = _obj(3, "bed", (0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
        bed_right = _obj(4, "bed", (10.0, 0.0, 0.0), (10.5, 0.5, 0.5))

        result = checker.check(lamp, [wrong_anchor, bed_left, bed_right], "between")

        self.assertTrue(result.satisfies)
        self.assertGreater(result.score, 0.0)
        self.assertEqual(result.details["anchor_pair"], (3, 4))
```

- [x] **Step 2: Run tests and verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest src/query_scene/tests/test_spatial_checker_bbox.py -v`

Expected: at least `inside`, `on_top_of`, `next_to`, and `between` fail under current centroid/bbox behavior.

### Task 2: Implement BBox-Aware Spatial Checker

**Files:**
- Modify: `src/query_scene/retrieval/spatial_checker.py`
- Test: `src/query_scene/tests/test_spatial_checker_bbox.py`

- [x] **Step 1: Add bbox helper methods**

Add helper methods to `SpatialRelationChecker`: `_coerce_bbox_minmax`, `_get_bbox`, `_bbox_xy_gap`, `_bbox_3d_gap`, `_bbox_xy_overlap_ratio`, and `_category_text`.

- [x] **Step 2: Update relation methods**

Update `is_on_top_of`, `is_above`, `is_below`, `is_next_to`, `is_near`, `is_inside`, and `check(..., relation="between")` to use bbox evidence when available and legacy centroid fallback otherwise.

- [x] **Step 3: Run spatial checker tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest src/query_scene/tests/test_spatial_checker_bbox.py -v`

Expected: all tests pass.

### Task 3: Add Quick Filter Safety and Trace Tests

**Files:**
- Modify: `src/query_scene/tests/test_execution_policy.py`
- Test: `src/query_scene/tests/test_execution_policy.py`

- [x] **Step 1: Add failing tests**

Add tests proving a hard quick filter that would empty candidates falls through to the full checker, and that executor metadata contains a compact `execution_trace`.

- [x] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest src/query_scene/tests/test_execution_policy.py -v`

Expected: new tests fail because quick-filter empty currently returns `[]` and no execution trace is emitted.

### Task 4: Implement QueryExecutor Trace and Quick Filter Safety

**Files:**
- Modify: `src/query_scene/query_executor.py`
- Test: `src/query_scene/tests/test_execution_policy.py`

- [x] **Step 1: Add trace state**

Initialize `self._execution_trace` in `execute()`, append compact trace dictionaries in `_apply_spatial_constraint()`, and copy the list into the root `ExecutionResult.metadata`.

- [x] **Step 2: Change quick-filter empty behavior**

When a hard quick filter returns zero candidates, set `quick_filter_would_empty=true`, restore `pre_filtered=candidates`, and continue into the full checker.

- [x] **Step 3: Preserve existing hard semantics after full check**

If full checker still finds no satisfying candidates under hard policy, return `[]` as before.

- [x] **Step 4: Run executor tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest src/query_scene/tests/test_execution_policy.py -v`

Expected: all tests pass.

### Task 5: Regression Verification

**Files:**
- Existing tests only.

- [x] **Step 1: Run focused query_scene tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest src/query_scene/tests/test_spatial_checker_bbox.py src/query_scene/tests/test_execution_policy.py src/query_scene/tests/test_keyframe_selector_hypothesis.py -v`

Expected: all tests pass.

- [x] **Step 2: Run lint on touched files**

Run: `ruff check src/query_scene/retrieval/spatial_checker.py src/query_scene/query_executor.py src/query_scene/tests/test_spatial_checker_bbox.py src/query_scene/tests/test_execution_policy.py`

Expected: no lint errors.

- [ ] **Step 3: Commit implementation**

```bash
git add src/query_scene/retrieval/spatial_checker.py src/query_scene/query_executor.py src/query_scene/tests/test_spatial_checker_bbox.py src/query_scene/tests/test_execution_policy.py docs/superpowers/plans/2026-05-24-nr3d-bbox-aware-spatial-execution-repair.md
git commit -m "Repair bbox-aware spatial execution"
```
