# EmbodiedScan 3D BBox Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a proposal-only feasibility harness comparing 2D-lifted, ConceptGraph, reconstructed-point-cloud detector, ScanNet-crop detector, and full-ScanNet detector 3D bbox proposals on EmbodiedScan ScanNet targets.

**Architecture:** Add a focused `benchmarks.embodiedscan_bbox_feasibility` package with small modules for data models, target indexing, observations, geometry, proposal generation, detector adaptation, evaluation, and CLI orchestration. The first working milestone is a smoke pipeline that can evaluate ConceptGraph and JSON-backed detector proposals; external SOTA detector execution is exposed through a strict adapter that either returns proposals or records `model_blocked`.

**Tech Stack:** Python 3.10, Pydantic v2, NumPy, SciPy/OpenCV/Pillow where already available, existing `benchmarks.embodiedscan_loader` and `benchmarks.embodiedscan_eval`, pytest. Run Linux commands inside conda env `conceptgraph`; run long detector/batch jobs in tmux and never use GPU 1.

---

## File Structure

- Create `src/benchmarks/embodiedscan_bbox_feasibility/__init__.py`: package exports.
- Create `src/benchmarks/embodiedscan_bbox_feasibility/models.py`: Pydantic contracts for targets, observations, proposals, metrics, and failure tags.
- Create `src/benchmarks/embodiedscan_bbox_feasibility/targets.py`: load EmbodiedScan samples and deduplicate `(scan_id, target_id)`.
- Create `src/benchmarks/embodiedscan_bbox_feasibility/geometry.py`: 9-DOF bbox helpers, point-cloud bbox fitting, depth backprojection, pose transforms, non-degenerate checks.
- Create `src/benchmarks/embodiedscan_bbox_feasibility/observations.py`: frame path resolution, target-conditioned observation records, and deterministic frame-window selection.
- Create `src/benchmarks/embodiedscan_bbox_feasibility/conceptgraph.py`: direct ConceptGraph PKL proposal generation without `KeyframeSelector`.
- Create `src/benchmarks/embodiedscan_bbox_feasibility/backproject.py`: 2D detection mask/bbox backprojection proposal generators for single-frame and multi-frame conditions.
- Create `src/benchmarks/embodiedscan_bbox_feasibility/detector_adapter.py`: external detector proposal ingestion and strict blocked-result reporting.
- Create `src/benchmarks/embodiedscan_bbox_feasibility/evaluator.py`: class-agnostic best-IoU scorer and aggregate metrics.
- Create `src/benchmarks/embodiedscan_bbox_feasibility/cli.py`: smoke/pilot/full command-line entry point.
- Create `src/benchmarks/tests/test_embodiedscan_bbox_feasibility_*.py`: focused unit tests for each module.

## Task 1: Proposal And Target Data Models

**Files:**
- Create: `src/benchmarks/embodiedscan_bbox_feasibility/__init__.py`
- Create: `src/benchmarks/embodiedscan_bbox_feasibility/models.py`
- Test: `src/benchmarks/tests/test_embodiedscan_bbox_feasibility_models.py`

- [ ] **Step 1: Write failing model tests**

Add `src/benchmarks/tests/test_embodiedscan_bbox_feasibility_models.py`:

```python
import pytest

from benchmarks.embodiedscan_bbox_feasibility.models import (
    BBox3DProposal,
    EmbodiedScanTarget,
    FailureTag,
    ObservationRecord,
    ProposalRecord,
)


def test_bbox_proposal_normalizes_to_nine_floats() -> None:
    proposal = BBox3DProposal(
        bbox_3d=[1, 2, 3, 4, 5, 6],
        score=0.5,
        source="unit",
    )
    assert proposal.bbox_3d == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0]


def test_bbox_proposal_rejects_short_box() -> None:
    with pytest.raises(ValueError, match="at least 6"):
        BBox3DProposal(bbox_3d=[1, 2, 3, 4, 5], score=1.0, source="unit")


def test_proposal_record_keeps_target_conditioned_observation() -> None:
    target = EmbodiedScanTarget(
        sample_ids=["sample-a"],
        scan_id="scannet/scene0001_00",
        scene_id="scene0001_00",
        target_id=7,
        target_category="chair",
        gt_bbox_3d=[0, 0, 0, 1, 1, 1, 0, 0, 0],
    )
    obs = ObservationRecord(
        policy="target_best_visible_centered_window",
        frame_ids=[10, 12, 14],
    )
    record = ProposalRecord(
        scene_id=target.scene_id,
        scan_id=target.scan_id,
        target_id=target.target_id,
        method="3d-mv-recon-detector",
        input_condition="multi_frame_recon_3",
        observation=obs,
        proposals=[],
        failure_tag=FailureTag.NO_PROPOSAL,
    )
    assert record.target_id == 7
    assert record.observation.frame_ids == [10, 12, 14]
    assert record.failure_tag == FailureTag.NO_PROPOSAL
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
pytest src/benchmarks/tests/test_embodiedscan_bbox_feasibility_models.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks.embodiedscan_bbox_feasibility'`.

- [ ] **Step 3: Implement models**

Create `src/benchmarks/embodiedscan_bbox_feasibility/__init__.py`:

```python
"""EmbodiedScan 3D bbox proposal feasibility utilities."""
```

Create `src/benchmarks/embodiedscan_bbox_feasibility/models.py`:

```python
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class FailureTag(str, Enum):
    NO_PROPOSAL = "no_proposal"
    COORD_MISMATCH = "coord_mismatch"
    DEGENERATE_BOX = "degenerate_box"
    VISIBILITY_LIMITED = "visibility_limited"
    OVERMERGE = "overmerge"
    FRAGMENTATION = "fragmentation"
    DETECTOR_OOD = "detector_ood"
    MODEL_BLOCKED = "model_blocked"


class EmbodiedScanTarget(BaseModel):
    sample_ids: list[str] = Field(default_factory=list)
    scan_id: str
    scene_id: str
    target_id: int
    target_category: str = ""
    gt_bbox_3d: list[float]

    @field_validator("gt_bbox_3d")
    @classmethod
    def _validate_gt_bbox(cls, value: list[float]) -> list[float]:
        return _normalize_bbox_9dof(value, field_name="gt_bbox_3d")


class ObservationRecord(BaseModel):
    policy: str
    frame_ids: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BBox3DProposal(BaseModel):
    bbox_3d: list[float]
    score: float | None = None
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bbox_3d")
    @classmethod
    def _validate_bbox(cls, value: list[float]) -> list[float]:
        return _normalize_bbox_9dof(value, field_name="bbox_3d")


class ProposalRecord(BaseModel):
    scene_id: str
    scan_id: str
    target_id: int | None = None
    method: str
    input_condition: str
    observation: ObservationRecord | None = None
    proposals: list[BBox3DProposal] = Field(default_factory=list)
    failure_tag: FailureTag | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TargetScore(BaseModel):
    scan_id: str
    scene_id: str
    target_id: int
    method: str
    input_condition: str
    best_iou: float
    best_proposal_index: int | None = None
    failure_tag: FailureTag | None = None


class AggregateMetrics(BaseModel):
    method: str
    input_condition: str
    num_targets: int
    mean_best_iou: float
    median_best_iou: float
    acc_025: float
    acc_050: float
    mean_proposals_per_record: float
    non_degenerate_box_ratio: float
    failure_counts: dict[str, int] = Field(default_factory=dict)


def _normalize_bbox_9dof(value: list[float], *, field_name: str) -> list[float]:
    if len(value) < 6:
        raise ValueError(f"{field_name} must contain at least 6 values")
    out = [float(v) for v in value[:9]]
    while len(out) < 9:
        out.append(0.0)
    return out
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
pytest src/benchmarks/tests/test_embodiedscan_bbox_feasibility_models.py -v
```

Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/benchmarks/embodiedscan_bbox_feasibility src/benchmarks/tests/test_embodiedscan_bbox_feasibility_models.py
git commit -m "feat(embodiedscan): add bbox feasibility data models"
```

## Task 2: Unique Target Index

**Files:**
- Create: `src/benchmarks/embodiedscan_bbox_feasibility/targets.py`
- Test: `src/benchmarks/tests/test_embodiedscan_bbox_feasibility_targets.py`

- [ ] **Step 1: Write failing target-index tests**

Add `src/benchmarks/tests/test_embodiedscan_bbox_feasibility_targets.py`:

```python
from benchmarks.embodiedscan_bbox_feasibility.targets import deduplicate_targets
from benchmarks.embodiedscan_loader import EmbodiedScanVGSample


def _sample(sample_id: str, target_id: int, query: str) -> EmbodiedScanVGSample:
    return EmbodiedScanVGSample(
        sample_id=sample_id,
        scene_id="scene0001_00",
        query=query,
        scan_id="scannet/scene0001_00",
        target_id=target_id,
        target="chair",
        gt_bbox_3d=[1, 2, 3, 4, 5, 6, 0, 0, 0],
    )


def test_deduplicate_targets_merges_repeated_referring_expressions() -> None:
    targets = deduplicate_targets([
        _sample("a", 7, "the chair"),
        _sample("b", 7, "the wooden chair"),
        _sample("c", 8, "the table"),
    ])
    assert len(targets) == 2
    assert targets[0].target_id == 7
    assert targets[0].sample_ids == ["a", "b"]
    assert targets[1].target_id == 8


def test_deduplicate_targets_skips_samples_without_gt_bbox() -> None:
    sample = _sample("a", 7, "the chair")
    sample.gt_bbox_3d = None
    assert deduplicate_targets([sample]) == []
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
pytest src/benchmarks/tests/test_embodiedscan_bbox_feasibility_targets.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `targets`.

- [ ] **Step 3: Implement target deduplication**

Create `src/benchmarks/embodiedscan_bbox_feasibility/targets.py`:

```python
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

from benchmarks.embodiedscan_loader import EmbodiedScanDataset, EmbodiedScanVGSample

from .models import EmbodiedScanTarget


def load_targets(
    data_root: str,
    *,
    split: str = "val",
    source_filter: str | None = "scannet",
    max_samples: int | None = None,
    mini: bool = False,
) -> list[EmbodiedScanTarget]:
    dataset = EmbodiedScanDataset.from_path(
        data_root,
        split=split,
        source_filter=source_filter,
        max_samples=max_samples,
        mini=mini,
    )
    return deduplicate_targets(dataset)


def deduplicate_targets(
    samples: Iterable[EmbodiedScanVGSample],
) -> list[EmbodiedScanTarget]:
    by_key: OrderedDict[tuple[str, int], EmbodiedScanTarget] = OrderedDict()
    for sample in samples:
        if sample.gt_bbox_3d is None:
            continue
        key = (sample.scan_id, sample.target_id)
        if key not in by_key:
            by_key[key] = EmbodiedScanTarget(
                sample_ids=[sample.sample_id],
                scan_id=sample.scan_id,
                scene_id=sample.scene_id,
                target_id=sample.target_id,
                target_category=sample.target,
                gt_bbox_3d=sample.gt_bbox_3d,
            )
        else:
            by_key[key].sample_ids.append(sample.sample_id)
    return list(by_key.values())
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
pytest src/benchmarks/tests/test_embodiedscan_bbox_feasibility_targets.py -v
```

Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add src/benchmarks/embodiedscan_bbox_feasibility/targets.py src/benchmarks/tests/test_embodiedscan_bbox_feasibility_targets.py
git commit -m "feat(embodiedscan): index unique bbox feasibility targets"
```

## Task 3: Geometry And IoU Utilities

**Files:**
- Create: `src/benchmarks/embodiedscan_bbox_feasibility/geometry.py`
- Test: `src/benchmarks/tests/test_embodiedscan_bbox_feasibility_geometry.py`

- [ ] **Step 1: Write failing geometry tests**

Add `src/benchmarks/tests/test_embodiedscan_bbox_feasibility_geometry.py`:

```python
import numpy as np

from benchmarks.embodiedscan_bbox_feasibility.geometry import (
    aabb_from_points,
    backproject_depth,
    is_non_degenerate_bbox,
    transform_points,
)


def test_aabb_from_points_returns_embodiedscan_9dof() -> None:
    pts = np.array([[0, 0, 0], [2, 4, 6]], dtype=np.float32)
    bbox = aabb_from_points(pts)
    assert bbox == [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0, 0.0, 0.0]
    assert is_non_degenerate_bbox(bbox)


def test_backproject_depth_uses_intrinsics_and_mask() -> None:
    depth = np.array([[1.0, 2.0], [0.0, 4.0]], dtype=np.float32)
    mask = np.array([[True, False], [False, True]])
    intrinsic = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=np.float32)
    pts = backproject_depth(depth, intrinsic, mask=mask)
    assert pts.shape == (2, 3)
    assert np.allclose(pts[0], [0.0, 0.0, 1.0])
    assert np.allclose(pts[1], [4.0, 4.0, 4.0])


def test_transform_points_applies_homogeneous_transform() -> None:
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, 3] = [10.0, 20.0, 30.0]
    out = transform_points(pts, mat)
    assert np.allclose(out, [[11.0, 22.0, 33.0]])
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
pytest src/benchmarks/tests/test_embodiedscan_bbox_feasibility_geometry.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `geometry`.

- [ ] **Step 3: Implement geometry helpers**

Create `src/benchmarks/embodiedscan_bbox_feasibility/geometry.py`:

```python
from __future__ import annotations

import numpy as np


def is_non_degenerate_bbox(bbox_3d: list[float], *, min_extent: float = 1e-4) -> bool:
    if len(bbox_3d) < 6:
        return False
    values = np.asarray(bbox_3d[:6], dtype=np.float64)
    if not np.isfinite(values).all():
        return False
    return bool(np.all(values[3:6] > min_extent))


def aabb_from_points(points: np.ndarray) -> list[float]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
        raise ValueError("points must have shape (N, >=3) with N > 0")
    xyz = pts[:, :3]
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    center = (lo + hi) / 2.0
    extent = hi - lo
    return [
        float(center[0]), float(center[1]), float(center[2]),
        float(extent[0]), float(extent[1]), float(extent[2]),
        0.0, 0.0, 0.0,
    ]


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    mat = np.asarray(transform, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must have shape (N, >=3)")
    if mat.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    ones = np.ones((len(pts), 1), dtype=np.float64)
    pts_h = np.hstack([pts[:, :3], ones])
    return (mat @ pts_h.T).T[:, :3]


def backproject_depth(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    depth_scale: float = 1.0,
    min_depth: float = 1e-6,
) -> np.ndarray:
    depth_arr = np.asarray(depth, dtype=np.float64) / float(depth_scale)
    k = np.asarray(intrinsic, dtype=np.float64)
    if k.shape[0] < 3 or k.shape[1] < 3:
        raise ValueError("intrinsic must be at least 3x3")
    valid = depth_arr > min_depth
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(valid)
    z = depth_arr[ys, xs]
    x = (xs.astype(np.float64) - k[0, 2]) * z / k[0, 0]
    y = (ys.astype(np.float64) - k[1, 2]) * z / k[1, 1]
    return np.stack([x, y, z], axis=1).astype(np.float32)
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
pytest src/benchmarks/tests/test_embodiedscan_bbox_feasibility_geometry.py -v
```

Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/benchmarks/embodiedscan_bbox_feasibility/geometry.py src/benchmarks/tests/test_embodiedscan_bbox_feasibility_geometry.py
git commit -m "feat(embodiedscan): add bbox feasibility geometry helpers"
```

## Task 4: ConceptGraph Proposal Generator

**Files:**
- Create: `src/benchmarks/embodiedscan_bbox_feasibility/conceptgraph.py`
- Test: `src/benchmarks/tests/test_embodiedscan_bbox_feasibility_conceptgraph.py`

- [ ] **Step 1: Write failing ConceptGraph tests**

Add `src/benchmarks/tests/test_embodiedscan_bbox_feasibility_conceptgraph.py`:

```python
import gzip
import pickle
from pathlib import Path

import numpy as np

from benchmarks.embodiedscan_bbox_feasibility.conceptgraph import (
    generate_conceptgraph_proposals,
)


def test_generate_conceptgraph_proposals_reads_pkl_without_keyframe_selector(tmp_path: Path) -> None:
    scene = tmp_path / "scene0001_00" / "conceptgraph"
    pcd_dir = scene / "pcd_saves"
    pcd_dir.mkdir(parents=True)
    pkl_path = pcd_dir / "full_pcd_mock_post.pkl.gz"
    payload = {
        "objects": [
            {
                "pcd_np": np.array([[0, 0, 0], [2, 4, 6]], dtype=np.float32),
                "class_name": ["chair"],
                "conf": [0.9],
            },
            {
                "pcd_np": np.array([[1, 1, 1]], dtype=np.float32),
                "class_name": ["floor"],
            },
        ]
    }
    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(payload, f)

    record = generate_conceptgraph_proposals(
        scene_path=scene,
        scan_id="scannet/scene0001_00",
        scene_id="scene0001_00",
    )

    assert record.method == "2d-cg"
    assert record.target_id is None
    assert len(record.proposals) == 1
    assert record.proposals[0].bbox_3d[:6] == [1.0, 2.0, 3.0, 2.0, 4.0, 6.0]
    assert record.proposals[0].metadata["category"] == "chair"
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
pytest src/benchmarks/tests/test_embodiedscan_bbox_feasibility_conceptgraph.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `conceptgraph`.

- [ ] **Step 3: Implement ConceptGraph generator**

Create `src/benchmarks/embodiedscan_bbox_feasibility/conceptgraph.py`:

```python
from __future__ import annotations

import gzip
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import aabb_from_points, is_non_degenerate_bbox
from .models import BBox3DProposal, FailureTag, ProposalRecord

_BG = {"wall", "floor", "ceiling"}


def generate_conceptgraph_proposals(
    *,
    scene_path: str | Path,
    scan_id: str,
    scene_id: str,
) -> ProposalRecord:
    scene = Path(scene_path)
    pkl_path = _find_pcd_file(scene)
    if pkl_path is None:
        return ProposalRecord(
            scene_id=scene_id,
            scan_id=scan_id,
            method="2d-cg",
            input_condition="conceptgraph_scene",
            proposals=[],
            failure_tag=FailureTag.NO_PROPOSAL,
            metadata={"reason": "no_pcd_file", "scene_path": str(scene)},
        )

    with gzip.open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    proposals: list[BBox3DProposal] = []
    for obj_idx, obj in enumerate(payload.get("objects", [])):
        category = _category(obj)
        if category.lower() in _BG:
            continue
        pcd_np = obj.get("pcd_np")
        if pcd_np is None or len(pcd_np) == 0:
            continue
        bbox = aabb_from_points(np.asarray(pcd_np, dtype=np.float32))
        if not is_non_degenerate_bbox(bbox):
            continue
        score = _score(obj)
        proposals.append(
            BBox3DProposal(
                bbox_3d=bbox,
                score=score,
                source="conceptgraph",
                metadata={
                    "obj_idx": obj_idx,
                    "category": category,
                    "num_points": int(len(pcd_np)),
                    "pkl_path": str(pkl_path),
                },
            )
        )

    return ProposalRecord(
        scene_id=scene_id,
        scan_id=scan_id,
        target_id=None,
        method="2d-cg",
        input_condition="conceptgraph_scene",
        proposals=proposals,
        failure_tag=None if proposals else FailureTag.NO_PROPOSAL,
    )


def _find_pcd_file(scene_path: Path) -> Path | None:
    pcd_dir = scene_path / "pcd_saves"
    for pattern in ("*ram*_post.pkl.gz", "*_post.pkl.gz", "*.pkl.gz"):
        matches = sorted(pcd_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _category(obj: dict[str, Any]) -> str:
    names = [str(n) for n in obj.get("class_name", []) if n]
    valid = [n for n in names if n.lower() not in {"item", "object", "none"}]
    if not valid:
        return "unknown"
    return Counter(valid).most_common(1)[0][0]


def _score(obj: dict[str, Any]) -> float | None:
    conf = obj.get("conf") or []
    if not conf:
        return None
    arr = np.asarray(conf, dtype=np.float64)
    return float(arr.mean())
```

- [ ] **Step 4: Run test and verify pass**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
pytest src/benchmarks/tests/test_embodiedscan_bbox_feasibility_conceptgraph.py -v
```

Expected: PASS, 1 test.

- [ ] **Step 5: Commit**

```bash
git add src/benchmarks/embodiedscan_bbox_feasibility/conceptgraph.py src/benchmarks/tests/test_embodiedscan_bbox_feasibility_conceptgraph.py
git commit -m "feat(embodiedscan): generate conceptgraph bbox proposals"
```

## Task 5: Observation Records And Frame Windows

**Files:**
- Create: `src/benchmarks/embodiedscan_bbox_feasibility/observations.py`
- Test: `src/benchmarks/tests/test_embodiedscan_bbox_feasibility_observations.py`

- [ ] **Step 1: Write failing observation tests**

Add `src/benchmarks/tests/test_embodiedscan_bbox_feasibility_observations.py`:

```python
from benchmarks.embodiedscan_bbox_feasibility.models import EmbodiedScanTarget
from benchmarks.embodiedscan_bbox_feasibility.observations import (
    centered_frame_window,
    make_observation,
)


def test_centered_frame_window_clamps_to_available_frames() -> None:
    assert centered_frame_window(center=2, available=[0, 2, 4, 6, 8], size=3) == [0, 2, 4]
    assert centered_frame_window(center=0, available=[0, 2, 4, 6, 8], size=3) == [0, 2, 4]
    assert centered_frame_window(center=8, available=[0, 2, 4, 6, 8], size=3) == [4, 6, 8]


def test_make_observation_records_target_policy() -> None:
    target = EmbodiedScanTarget(
        sample_ids=["a"],
        scan_id="scannet/scene0001_00",
        scene_id="scene0001_00",
        target_id=4,
        target_category="chair",
        gt_bbox_3d=[0, 0, 0, 1, 1, 1, 0, 0, 0],
    )
    obs = make_observation(target, best_frame_id=10, available_frame_ids=[6, 8, 10, 12, 14], window_size=3)
    assert obs.policy == "target_best_visible_centered_window"
    assert obs.frame_ids == [8, 10, 12]
    assert obs.metadata["target_id"] == 4
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
pytest src/benchmarks/tests/test_embodiedscan_bbox_feasibility_observations.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `observations`.

- [ ] **Step 3: Implement observation helpers**

Create `src/benchmarks/embodiedscan_bbox_feasibility/observations.py`:

```python
from __future__ import annotations

from pathlib import Path

from .models import EmbodiedScanTarget, ObservationRecord


def centered_frame_window(center: int, available: list[int], size: int) -> list[int]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not available:
        return []
    ordered = sorted(int(v) for v in available)
    if center in ordered:
        center_idx = ordered.index(center)
    else:
        center_idx = min(range(len(ordered)), key=lambda i: abs(ordered[i] - center))
    half = size // 2
    start = max(0, center_idx - half)
    end = min(len(ordered), start + size)
    start = max(0, end - size)
    return ordered[start:end]


def make_observation(
    target: EmbodiedScanTarget,
    *,
    best_frame_id: int,
    available_frame_ids: list[int],
    window_size: int,
) -> ObservationRecord:
    return ObservationRecord(
        policy="target_best_visible_centered_window",
        frame_ids=centered_frame_window(best_frame_id, available_frame_ids, window_size),
        metadata={
            "scan_id": target.scan_id,
            "scene_id": target.scene_id,
            "target_id": target.target_id,
            "best_frame_id": int(best_frame_id),
            "window_size": int(window_size),
        },
    )


def list_raw_frame_ids(scene_root: str | Path) -> list[int]:
    root = Path(scene_root)
    raw = root / "raw"
    ids: list[int] = []
    for path in sorted(raw.glob("*-rgb.*")):
        stem = path.name.split("-rgb", 1)[0]
        if stem.isdigit():
            ids.append(int(stem))
    return ids
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
pytest src/benchmarks/tests/test_embodiedscan_bbox_feasibility_observations.py -v
```

Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add src/benchmarks/embodiedscan_bbox_feasibility/observations.py src/benchmarks/tests/test_embodiedscan_bbox_feasibility_observations.py
git commit -m "feat(embodiedscan): add bbox feasibility observation windows"
```

## Task 6: 2D Backprojection Proposal Generator

**Files:**
- Create: `src/benchmarks/embodiedscan_bbox_feasibility/backproject.py`
- Test: `src/benchmarks/tests/test_embodiedscan_bbox_feasibility_backproject.py`

- [ ] **Step 1: Write failing backprojection tests**

Add `src/benchmarks/tests/test_embodiedscan_bbox_feasibility_backproject.py`:

```python
import numpy as np

from benchmarks.embodiedscan_bbox_feasibility.backproject import (
    proposal_from_depth_mask,
)


def test_proposal_from_depth_mask_backprojects_and_transforms() -> None:
    depth = np.ones((2, 2), dtype=np.float32)
    mask = np.array([[True, False], [False, True]])
    intrinsic = np.eye(3, dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [10, 20, 30]

    proposal = proposal_from_depth_mask(
        depth=depth,
        mask=mask,
        intrinsic=intrinsic,
        camera_to_world=pose,
        source="2d-backproject",
        min_points=1,
    )

    assert proposal is not None
    assert proposal.source == "2d-backproject"
    assert proposal.bbox_3d[:3] == [10.5, 20.5, 31.0]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
pytest src/benchmarks/tests/test_embodiedscan_bbox_feasibility_backproject.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `backproject`.

- [ ] **Step 3: Implement core backprojection proposal function**

Create `src/benchmarks/embodiedscan_bbox_feasibility/backproject.py`:

```python
from __future__ import annotations

import numpy as np

from .geometry import aabb_from_points, backproject_depth, is_non_degenerate_bbox, transform_points
from .models import BBox3DProposal


def proposal_from_depth_mask(
    *,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    source: str,
    score: float | None = None,
    min_points: int = 5,
    metadata: dict | None = None,
) -> BBox3DProposal | None:
    cam_points = backproject_depth(depth, intrinsic, mask=mask)
    if len(cam_points) < min_points:
        return None
    world_points = transform_points(cam_points, camera_to_world)
    bbox = aabb_from_points(world_points)
    if not is_non_degenerate_bbox(bbox):
        return None
    return BBox3DProposal(
        bbox_3d=bbox,
        score=score,
        source=source,
        metadata={**(metadata or {}), "num_points": int(len(world_points))},
    )
```

- [ ] **Step 4: Run test and verify pass**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
pytest src/benchmarks/tests/test_embodiedscan_bbox_feasibility_backproject.py -v
```

Expected: PASS, 1 test.

- [ ] **Step 5: Commit**

```bash
git add src/benchmarks/embodiedscan_bbox_feasibility/backproject.py src/benchmarks/tests/test_embodiedscan_bbox_feasibility_backproject.py
git commit -m "feat(embodiedscan): add rgbd backprojection proposals"
```

## Task 7: Detector Adapter With Strict Blocked Reporting

**Files:**
- Create: `src/benchmarks/embodiedscan_bbox_feasibility/detector_adapter.py`
- Test: `src/benchmarks/tests/test_embodiedscan_bbox_feasibility_detector_adapter.py`

- [ ] **Step 1: Write failing detector adapter tests**

Add `src/benchmarks/tests/test_embodiedscan_bbox_feasibility_detector_adapter.py`:

```python
import json
from pathlib import Path

from benchmarks.embodiedscan_bbox_feasibility.detector_adapter import (
    load_detector_proposals_json,
    model_blocked_record,
)
from benchmarks.embodiedscan_bbox_feasibility.models import FailureTag


def test_load_detector_proposals_json(tmp_path: Path) -> None:
    path = tmp_path / "pred.json"
    path.write_text(
        json.dumps({
            "proposals": [
                {"bbox_3d": [0, 0, 0, 1, 1, 1], "score": 0.9, "label": "chair"}
            ]
        }),
        encoding="utf-8",
    )
    record = load_detector_proposals_json(
        path=path,
        scene_id="scene0001_00",
        scan_id="scannet/scene0001_00",
        method="3d-scannet-full-detector",
        input_condition="scannet_full",
    )
    assert len(record.proposals) == 1
    assert record.proposals[0].metadata["label"] == "chair"


def test_model_blocked_record_is_explicit() -> None:
    record = model_blocked_record(
        scene_id="scene0001_00",
        scan_id="scannet/scene0001_00",
        method="3d-sf-recon-dest-vdetr",
        input_condition="single_frame_recon",
        reason="DEST-VDETR checkpoint unavailable",
    )
    assert record.failure_tag == FailureTag.MODEL_BLOCKED
    assert record.metadata["reason"] == "DEST-VDETR checkpoint unavailable"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
pytest src/benchmarks/tests/test_embodiedscan_bbox_feasibility_detector_adapter.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `detector_adapter`.

- [ ] **Step 3: Implement JSON ingestion and blocked record**

Create `src/benchmarks/embodiedscan_bbox_feasibility/detector_adapter.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import BBox3DProposal, FailureTag, ProposalRecord


def load_detector_proposals_json(
    *,
    path: str | Path,
    scene_id: str,
    scan_id: str,
    method: str,
    input_condition: str,
    target_id: int | None = None,
) -> ProposalRecord:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    proposals = []
    for raw in data.get("proposals", []):
        metadata: dict[str, Any] = dict(raw.get("metadata", {}))
        if "label" in raw:
            metadata["label"] = raw["label"]
        proposals.append(
            BBox3DProposal(
                bbox_3d=raw["bbox_3d"],
                score=raw.get("score"),
                source="detector",
                metadata=metadata,
            )
        )
    return ProposalRecord(
        scene_id=scene_id,
        scan_id=scan_id,
        target_id=target_id,
        method=method,
        input_condition=input_condition,
        proposals=proposals,
        failure_tag=None if proposals else FailureTag.NO_PROPOSAL,
        metadata={"path": str(path)},
    )


def model_blocked_record(
    *,
    scene_id: str,
    scan_id: str,
    method: str,
    input_condition: str,
    reason: str,
    target_id: int | None = None,
) -> ProposalRecord:
    return ProposalRecord(
        scene_id=scene_id,
        scan_id=scan_id,
        target_id=target_id,
        method=method,
        input_condition=input_condition,
        proposals=[],
        failure_tag=FailureTag.MODEL_BLOCKED,
        metadata={"reason": reason},
    )
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
pytest src/benchmarks/tests/test_embodiedscan_bbox_feasibility_detector_adapter.py -v
```

Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add src/benchmarks/embodiedscan_bbox_feasibility/detector_adapter.py src/benchmarks/tests/test_embodiedscan_bbox_feasibility_detector_adapter.py
git commit -m "feat(embodiedscan): add detector proposal adapter"
```

## Task 8: Evaluator

**Files:**
- Create: `src/benchmarks/embodiedscan_bbox_feasibility/evaluator.py`
- Test: `src/benchmarks/tests/test_embodiedscan_bbox_feasibility_evaluator.py`

- [ ] **Step 1: Write failing evaluator tests**

Add `src/benchmarks/tests/test_embodiedscan_bbox_feasibility_evaluator.py`:

```python
from benchmarks.embodiedscan_bbox_feasibility.evaluator import (
    evaluate_records,
)
from benchmarks.embodiedscan_bbox_feasibility.models import (
    BBox3DProposal,
    EmbodiedScanTarget,
    ProposalRecord,
)


def test_evaluate_records_uses_best_iou_per_target() -> None:
    target = EmbodiedScanTarget(
        sample_ids=["a"],
        scan_id="scannet/scene0001_00",
        scene_id="scene0001_00",
        target_id=1,
        target_category="chair",
        gt_bbox_3d=[0, 0, 0, 1, 1, 1, 0, 0, 0],
    )
    record = ProposalRecord(
        scene_id="scene0001_00",
        scan_id="scannet/scene0001_00",
        target_id=None,
        method="unit",
        input_condition="scene",
        proposals=[
            BBox3DProposal(bbox_3d=[5, 5, 5, 1, 1, 1], score=0.1, source="unit"),
            BBox3DProposal(bbox_3d=[0, 0, 0, 1, 1, 1], score=0.9, source="unit"),
        ],
    )
    result = evaluate_records([target], [record])
    assert result.metrics.mean_best_iou == 1.0
    assert result.metrics.acc_025 == 1.0
    assert result.scores[0].best_proposal_index == 1
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
pytest src/benchmarks/tests/test_embodiedscan_bbox_feasibility_evaluator.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `evaluator`.

- [ ] **Step 3: Implement evaluator**

Create `src/benchmarks/embodiedscan_bbox_feasibility/evaluator.py`:

```python
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from benchmarks.embodiedscan_eval import compute_oriented_iou_3d

from .geometry import is_non_degenerate_bbox
from .models import AggregateMetrics, EmbodiedScanTarget, ProposalRecord, TargetScore


@dataclass
class EvaluationResult:
    scores: list[TargetScore]
    metrics: AggregateMetrics


def evaluate_records(
    targets: list[EmbodiedScanTarget],
    records: list[ProposalRecord],
) -> EvaluationResult:
    if not records:
        raise ValueError("records must not be empty")
    method = records[0].method
    input_condition = records[0].input_condition
    scores: list[TargetScore] = []
    proposal_counts = []
    valid_box_count = 0
    total_box_count = 0
    failure_counts: Counter[str] = Counter()

    for target in targets:
        record = _find_record(target, records)
        if record is None:
            scores.append(_score_missing(target, method, input_condition))
            failure_counts["no_proposal"] += 1
            continue

        if record.failure_tag is not None:
            failure_counts[record.failure_tag.value] += 1
        proposal_counts.append(len(record.proposals))
        best_iou = 0.0
        best_idx: int | None = None
        for idx, proposal in enumerate(record.proposals):
            total_box_count += 1
            if is_non_degenerate_bbox(proposal.bbox_3d):
                valid_box_count += 1
            iou = compute_oriented_iou_3d(proposal.bbox_3d, target.gt_bbox_3d)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        scores.append(
            TargetScore(
                scan_id=target.scan_id,
                scene_id=target.scene_id,
                target_id=target.target_id,
                method=record.method,
                input_condition=record.input_condition,
                best_iou=float(best_iou),
                best_proposal_index=best_idx,
                failure_tag=record.failure_tag,
            )
        )

    ious = np.asarray([s.best_iou for s in scores], dtype=np.float64)
    metrics = AggregateMetrics(
        method=method,
        input_condition=input_condition,
        num_targets=len(scores),
        mean_best_iou=float(ious.mean()) if len(ious) else 0.0,
        median_best_iou=float(np.median(ious)) if len(ious) else 0.0,
        acc_025=float((ious >= 0.25).mean()) if len(ious) else 0.0,
        acc_050=float((ious >= 0.50).mean()) if len(ious) else 0.0,
        mean_proposals_per_record=float(np.mean(proposal_counts)) if proposal_counts else 0.0,
        non_degenerate_box_ratio=(
            float(valid_box_count / total_box_count) if total_box_count else 0.0
        ),
        failure_counts=dict(failure_counts),
    )
    return EvaluationResult(scores=scores, metrics=metrics)


def _find_record(
    target: EmbodiedScanTarget,
    records: list[ProposalRecord],
) -> ProposalRecord | None:
    exact = [
        r for r in records
        if r.scan_id == target.scan_id and r.target_id == target.target_id
    ]
    if exact:
        return exact[0]
    scene_level = [r for r in records if r.scan_id == target.scan_id and r.target_id is None]
    if scene_level:
        return scene_level[0]
    return None


def _score_missing(
    target: EmbodiedScanTarget,
    method: str,
    input_condition: str,
) -> TargetScore:
    return TargetScore(
        scan_id=target.scan_id,
        scene_id=target.scene_id,
        target_id=target.target_id,
        method=method,
        input_condition=input_condition,
        best_iou=0.0,
        best_proposal_index=None,
    )
```

- [ ] **Step 4: Run test and verify pass**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
pytest src/benchmarks/tests/test_embodiedscan_bbox_feasibility_evaluator.py -v
```

Expected: PASS, 1 test.

- [ ] **Step 5: Commit**

```bash
git add src/benchmarks/embodiedscan_bbox_feasibility/evaluator.py src/benchmarks/tests/test_embodiedscan_bbox_feasibility_evaluator.py
git commit -m "feat(embodiedscan): evaluate bbox proposal upper bounds"
```

## Task 9: Smoke CLI

**Files:**
- Create: `src/benchmarks/embodiedscan_bbox_feasibility/cli.py`
- Test: `src/benchmarks/tests/test_embodiedscan_bbox_feasibility_cli.py`

- [ ] **Step 1: Write failing CLI test**

Add `src/benchmarks/tests/test_embodiedscan_bbox_feasibility_cli.py`:

```python
from benchmarks.embodiedscan_bbox_feasibility.cli import build_parser


def test_parser_accepts_smoke_mode() -> None:
    args = build_parser().parse_args([
        "smoke",
        "--data-root",
        "data/embodiedscan",
        "--scene-data-root",
        "data/embodiedscan/scannet",
        "--max-targets",
        "5",
    ])
    assert args.command == "smoke"
    assert args.max_targets == 5
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
pytest src/benchmarks/tests/test_embodiedscan_bbox_feasibility_cli.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `cli`.

- [ ] **Step 3: Implement parser and smoke command**

Create `src/benchmarks/embodiedscan_bbox_feasibility/cli.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .conceptgraph import generate_conceptgraph_proposals
from .evaluator import evaluate_records
from .targets import load_targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EmbodiedScan 3D bbox feasibility harness")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="Run ConceptGraph smoke evaluation")
    smoke.add_argument("--data-root", type=Path, required=True)
    smoke.add_argument("--scene-data-root", type=Path, required=True)
    smoke.add_argument("--output-dir", type=Path, default=Path("outputs/embodiedscan_bbox_feasibility"))
    smoke.add_argument("--max-targets", type=int, default=50)
    smoke.add_argument("--mini", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        run_smoke(args)


def run_smoke(args: argparse.Namespace) -> None:
    targets = load_targets(
        str(args.data_root),
        split="val",
        source_filter="scannet",
        max_samples=None,
        mini=args.mini,
    )[: args.max_targets]
    records = []
    seen_scan_ids: set[str] = set()
    for target in targets:
        if target.scan_id in seen_scan_ids:
            continue
        seen_scan_ids.add(target.scan_id)
        scene_name = target.scan_id.split("/")[-1]
        scene_path = args.scene_data_root / scene_name / "conceptgraph"
        records.append(
            generate_conceptgraph_proposals(
                scene_path=scene_path,
                scan_id=target.scan_id,
                scene_id=target.scene_id,
            )
        )
    result = evaluate_records(targets, records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "smoke_metrics.json"
    scores_path = args.output_dir / "smoke_scores.jsonl"
    metrics_path.write_text(result.metrics.model_dump_json(indent=2), encoding="utf-8")
    with scores_path.open("w", encoding="utf-8") as f:
        for score in result.scores:
            f.write(json.dumps(score.model_dump(), ensure_ascii=False) + "\n")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {scores_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit test and verify pass**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
pytest src/benchmarks/tests/test_embodiedscan_bbox_feasibility_cli.py -v
```

Expected: PASS, 1 test.

- [ ] **Step 5: Run smoke CLI on local data**

Run in tmux because it touches real data:

```bash
tmux new-session -d -s bbox-smoke "cd /home/ysh/codecase/3DVLMReasoning && source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph && export PYTHONPATH=src:\${PYTHONPATH:-} && python -m benchmarks.embodiedscan_bbox_feasibility.cli smoke --data-root data/embodiedscan --scene-data-root data/embodiedscan/scannet --max-targets 10 --output-dir outputs/embodiedscan_bbox_feasibility/smoke 2>&1 | tee /tmp/bbox_smoke.log"
```

Check:

```bash
tmux capture-pane -t bbox-smoke -p -S -50
```

Expected: command writes `smoke_metrics.json` and `smoke_scores.jsonl`, or fails loudly with a missing data path that must be fixed before scaling.

- [ ] **Step 6: Commit**

```bash
git add src/benchmarks/embodiedscan_bbox_feasibility/cli.py src/benchmarks/tests/test_embodiedscan_bbox_feasibility_cli.py
git commit -m "feat(embodiedscan): add bbox feasibility smoke cli"
```

## Task 10: Final Verification

**Files:**
- Modify only if verification exposes defects in files from Tasks 1-9.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
pytest src/benchmarks/tests/test_embodiedscan_bbox_feasibility_*.py -v
```

Expected: all feasibility tests pass.

- [ ] **Step 2: Run lint on new package**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
ruff check src/benchmarks/embodiedscan_bbox_feasibility src/benchmarks/tests/test_embodiedscan_bbox_feasibility_*.py
```

Expected: no ruff errors.

- [ ] **Step 3: Confirm smoke outputs are parseable**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
python -m json.tool outputs/embodiedscan_bbox_feasibility/smoke/smoke_metrics.json >/tmp/bbox_smoke_metrics.pretty.json
head -n 3 outputs/embodiedscan_bbox_feasibility/smoke/smoke_scores.jsonl
```

Expected: `json.tool` exits 0 and JSONL rows contain `best_iou`.

- [ ] **Step 4: Commit verification fixes if any**

If Step 1-3 required code fixes, commit them:

```bash
git add src/benchmarks/embodiedscan_bbox_feasibility src/benchmarks/tests/test_embodiedscan_bbox_feasibility_*.py
git commit -m "fix(embodiedscan): stabilize bbox feasibility smoke harness"
```

If there were no fixes, do not create an empty commit.

## Plan Self-Review

- Spec coverage: Tasks 1-2 cover unique target indexing; Tasks 3 and 6 cover RGB-D backprojection; Task 4 covers `2D-CG`; Task 7 covers external SOTA detector proposal ingestion plus explicit `model_blocked`; Task 8 covers class-agnostic best-IoU metrics; Task 9 covers the smoke execution path. Full DEST-VDETR environment installation is intentionally outside this first implementation plan because the spec requires strict blocked reporting when the public SOTA path is unavailable.
- Placeholder scan: no placeholder markers or vague "fill in later" instructions remain.
- Type consistency: `EmbodiedScanTarget`, `ObservationRecord`, `BBox3DProposal`, `ProposalRecord`, `FailureTag`, and `AggregateMetrics` are introduced in Task 1 and reused consistently in later tasks.
