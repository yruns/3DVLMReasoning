"""Tests for ``scripts/build_nr3d_strat600_fold.py``.

These exercise the deterministic stratified-sampling primitives without
hitting the real NR3D data root (which is heavy).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_builder():
    script = (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "build_nr3d_strat600_fold.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_nr3d_strat600_fold",
        script,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stable_key_is_deterministic() -> None:
    builder = _load_builder()
    k1 = builder.stable_key("scannet/scene0001_00::1::A1")
    k2 = builder.stable_key("scannet/scene0001_00::1::A1")
    assert k1 == k2
    k_other = builder.stable_key("scannet/scene0001_00::2::A2")
    assert k1 != k_other


def test_is_easy_boundary() -> None:
    builder = _load_builder()
    assert builder.is_easy(1) is True
    assert builder.is_easy(2) is True
    assert builder.is_easy(3) is False


def test_is_view_dep_token_set() -> None:
    builder = _load_builder()
    assert builder.is_view_dep(["the", "chair", "in", "front"]) is True
    assert builder.is_view_dep(["leftmost", "monitor"]) is True
    assert builder.is_view_dep(["red", "chair"]) is False
    # case-sensitive (referit3d tokens are pre-lowercased)
    assert builder.is_view_dep(["Front"]) is False


def test_largest_remainder_allocation_sums_to_total() -> None:
    builder = _load_builder()
    cell_sizes = {
        (True, False): 2570,
        (True, True):  1203,
        (False, False): 2483,
        (False, True): 1549,
    }
    allocation = builder.largest_remainder_allocation(cell_sizes, total=600)
    assert sum(allocation.values()) == 600
    # Each cell at least floor(quota), at most floor+1
    pool_n = sum(cell_sizes.values())
    for cell, n_cell in allocation.items():
        quota = 600 * cell_sizes[cell] / pool_n
        assert int(quota) <= n_cell <= int(quota) + 1


def test_largest_remainder_allocation_proportions_close_to_pool() -> None:
    builder = _load_builder()
    cell_sizes = {
        (True, False): 2570,
        (True, True):  1203,
        (False, False): 2483,
        (False, True): 1549,
    }
    allocation = builder.largest_remainder_allocation(cell_sizes, total=600)
    pool_n = sum(cell_sizes.values())
    for cell, n_cell in allocation.items():
        pool_pct = cell_sizes[cell] / pool_n
        sub_pct = n_cell / 600
        assert abs(pool_pct - sub_pct) < 0.01  # within 1 pp of pool share


def test_largest_remainder_allocation_rejects_oversized_request() -> None:
    builder = _load_builder()
    import pytest
    with pytest.raises(ValueError):
        builder.largest_remainder_allocation({(True, False): 10}, total=100)


def test_select_subset_deterministic_and_stratified() -> None:
    builder = _load_builder()
    # Build a synthetic pool of 100 rows: 50 easy/vindep, 20 easy/vdep,
    # 20 hard/vindep, 10 hard/vdep. Request 20 → should keep 4-cell ratio.
    rows = []
    for i in range(50):
        rows.append({
            "sample_id": f"scannet/scene_a::E_VI::{i}",
            "scene_id": "scene_a", "target_id": i, "category": "chair",
            "n_objects": 1, "tokens": ["a", "chair"],
            "is_easy": True, "is_view_dep": False,
        })
    for i in range(20):
        rows.append({
            "sample_id": f"scannet/scene_a::E_VD::{i}",
            "scene_id": "scene_a", "target_id": 100 + i, "category": "chair",
            "n_objects": 1, "tokens": ["chair", "front"],
            "is_easy": True, "is_view_dep": True,
        })
    for i in range(20):
        rows.append({
            "sample_id": f"scannet/scene_a::H_VI::{i}",
            "scene_id": "scene_a", "target_id": 200 + i, "category": "chair",
            "n_objects": 5, "tokens": ["a", "chair"],
            "is_easy": False, "is_view_dep": False,
        })
    for i in range(10):
        rows.append({
            "sample_id": f"scannet/scene_a::H_VD::{i}",
            "scene_id": "scene_a", "target_id": 300 + i, "category": "chair",
            "n_objects": 5, "tokens": ["chair", "behind"],
            "is_easy": False, "is_view_dep": True,
        })

    selected = builder.select_subset(rows, n=20)
    assert len(selected) == 20

    # Verify per-cell counts match largest-remainder allocation
    from collections import Counter
    cells = Counter((s["is_easy"], s["is_view_dep"]) for s in selected)
    cell_sizes = {(True, False): 50, (True, True): 20,
                  (False, False): 20, (False, True): 10}
    expected = builder.largest_remainder_allocation(cell_sizes, total=20)
    for cell, n_cell in expected.items():
        assert cells[cell] == n_cell, (
            f"cell {cell}: got {cells[cell]}, expected {n_cell}"
        )

    # Determinism: re-running on shuffled input yields the same selection
    import random
    rng = random.Random(123)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    re_selected = builder.select_subset(shuffled, n=20)
    assert sorted(s["sample_id"] for s in selected) == \
        sorted(s["sample_id"] for s in re_selected)
