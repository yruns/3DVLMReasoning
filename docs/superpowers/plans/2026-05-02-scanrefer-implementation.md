# ScanRefer Detection-Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship v1 ScanRefer detection-mode evaluation on the canonical full val (9508 utterances on 141 scenes) using our zero-shot RGB+VLM agent (gpt-5.4-2026-03-05) against Mask3D-predicted candidate proposals.

**Architecture:** Mirror NR3D v3's pipeline shape with two pkl sources: GT bbox lookup via the existing Phase 8 GT-CG pkl (read-only consumer); proposal pool via a new Mask3D-injected ConceptGraph pkl produced from ZSVG3D's `.npz` distribution. Stage 1 keyframe selector and Stage 2 agent untouched. New modules: 1 producer, 1 loader, 1 pack-prep, 1 runner, 1 aggregator, 1 ingester.

**Tech Stack:** Python 3.12 (uv `.venv`), pytest, numpy, sqlite3, existing `compute_oriented_iou_3d`, existing `build_visibility_index`, `Stage2DeepResearchAgent`. No new external dependencies.

**Context references:**
- Design spec: `docs/superpowers/specs/2026-05-02-scanrefer-design.md` (commit `30354a5`)
- NR3D v3 plan as architectural template: `docs/superpowers/plans/2026-05-01-nr3d-leaderboard-track.md`
- Audit reports: `tmp/scanrefer_zsl_code_audit.md`, `tmp/scanrefer_zsl_paper_survey.md`, `tmp/seeground_vog_audit_{claude,codex}.md`
- Existing helper: `src/scripts/build_visibility_index.py::build_visibility_index`
- ScanNet200 class taxonomy: `conceptgraph/scannet200_classes.txt`
- Branch: `feat/scanrefer-vg-benchmark`, baseline tip: `30354a5`

**Date placeholder:** Use `YYYYMMDD = 20260502` throughout. If implementation lands later, replace with the actual date in file names and run IDs.

---

## File Structure

**New files (creates):**

| File | Purpose |
|---|---|
| `src/scripts/build_scanrefer_mask3d_cg.py` | Mask3D `.npz` → ConceptGraph-shaped pkl + visibility index converter |
| `src/scripts/tests/test_build_scanrefer_mask3d_cg.py` | Pure-helper tests for the converter |
| `src/benchmarks/scanrefer_loader.py` | `ScanRefVGDataset` + `ScanRefVGSample` |
| `src/benchmarks/tests/test_scanrefer_loader.py` | Loader tests |
| `src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py` | Pack-prep mirror with two pkl sources |
| `src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer.py` | Pack-prep tests |
| `src/evaluation/scripts/run_scanrefer_vg_side_by_side.py` | Runner mirror |
| `src/evaluation/scripts/tests/test_run_scanrefer_vg_side_by_side.py` | Runner tests |
| `src/evaluation/scripts/scanrefer_leaderboard_metrics.py` | Aggregator (Unique/Multiple slicing) |
| `src/evaluation/scripts/tests/test_scanrefer_leaderboard_metrics.py` | Aggregator tests |
| `scripts/ingest_scanrefer_run.py` | SQLite ingester |
| `src/evaluation/scripts/tests/test_ingest_scanrefer_run.py` | Ingester tests |
| `docs/benchmark/scanrefer/v1_mask3d_track_20260502.md` | v1 version doc |
| `docs/benchmark/scanrefer/README.md` | Benchmark index |
| `docs/benchmark/scanrefer/leaderboard.md` | Public leaderboard reference |

**Modified files:**

| File | Change |
|---|---|
| `docs/benchmark/README.md` | Add ScanRefer row to Active Benchmarks |

**Untouched (deliberately):**

| File | Reason |
|---|---|
| `src/benchmarks/nr3d_loader.py` | NR3D loader, read-only consumer (Phase 8 GT-CG pkl as GT lookup table) |
| `src/evaluation/scripts/run_nr3d_vg_side_by_side.py` | NR3D runner, used as read-reference template |
| `src/evaluation/scripts/nr3d_leaderboard_metrics.py` | NR3D aggregator, used as read-reference template |
| `scripts/ingest_nr3d_run.py` | NR3D ingester, used as read-reference template |
| `src/agents/**` | Stage 2 agent code, fully reused |
| `data/nr3d/scannet/<scene>/conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz` | Phase 8 GT-CG pkl, read-only consumer |
| `data/scanrefer/Mask3d/scannet200/<scene>.npz` | ZSVG3D Mask3D distribution, read-only input |

---

## Task 1: Mask3D-CG converter pure helpers (RED → GREEN)

**Files:**
- Create: `src/scripts/tests/test_build_scanrefer_mask3d_cg.py`
- Create: `src/scripts/build_scanrefer_mask3d_cg.py` (helpers section only; full converter in Task 2)

This task delivers the **pure** helper functions that don't touch disk. Bbox derivation, class lookup, background filter. The IO + visibility step lands in Task 2.

- [ ] **Step 1.1: Create test scaffolding directory**

```bash
mkdir -p src/scripts/tests
touch src/scripts/tests/__init__.py
```

- [ ] **Step 1.2: Write the failing test file**

Create `src/scripts/tests/test_build_scanrefer_mask3d_cg.py` with this EXACT content:

```python
"""Tests for ScanRefer Mask3D-CG converter pure helpers."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.build_scanrefer_mask3d_cg import (
    BACKGROUND_LABELS,
    axis_aligned_corners_from_pcd,
    build_object_dict,
    is_background_label,
    load_scannet200_class_index,
    scannet200_class_id,
)


def test_axis_aligned_corners_from_unit_cube_pcd():
    """Unit cube with corners at (0,0,0)-(1,1,1) → 8 corners spanning that AABB."""
    pcd = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    corners = axis_aligned_corners_from_pcd(pcd)
    assert corners.shape == (8, 3)
    assert corners.min(axis=0).tolist() == [0.0, 0.0, 0.0]
    assert corners.max(axis=0).tolist() == [1.0, 1.0, 1.0]


def test_axis_aligned_corners_uses_first_three_columns_only():
    """If pcd is (N, 6) [xyz+rgb], rgb is ignored."""
    pcd = np.array(
        [
            [0.0, 0.0, 0.0, 100.0, 100.0, 100.0],
            [2.0, 3.0, 4.0, 200.0, 200.0, 200.0],
        ],
        dtype=np.float32,
    )
    corners = axis_aligned_corners_from_pcd(pcd)
    assert corners.shape == (8, 3)
    assert corners.min(axis=0).tolist() == [0.0, 0.0, 0.0]
    assert corners.max(axis=0).tolist() == [2.0, 3.0, 4.0]


def test_axis_aligned_corners_raises_on_empty_pcd():
    with pytest.raises(ValueError, match="empty"):
        axis_aligned_corners_from_pcd(np.zeros((0, 3), dtype=np.float32))


def test_background_labels_set():
    assert BACKGROUND_LABELS == frozenset({"wall", "floor", "ceiling"})


def test_is_background_label_case_insensitive():
    assert is_background_label("wall") is True
    assert is_background_label("WALL") is True
    assert is_background_label("Floor") is True
    assert is_background_label("chair") is False
    assert is_background_label("") is False


def test_load_scannet200_class_index_returns_lowercased_label_to_idx_dict(tmp_path):
    """Each line of the canonical file is one class name; index is line number (0-based)."""
    txt = tmp_path / "scannet200_classes.txt"
    txt.write_text("alarm clock\narmchair\nchair\n", encoding="utf-8")
    idx = load_scannet200_class_index(txt)
    assert idx == {"alarm clock": 0, "armchair": 1, "chair": 2}


def test_scannet200_class_id_known_label():
    idx = {"chair": 2, "table": 5}
    assert scannet200_class_id("chair", idx) == 2
    assert scannet200_class_id("CHAIR", idx) == 2  # case-insensitive


def test_scannet200_class_id_unknown_returns_minus_one():
    idx = {"chair": 2}
    assert scannet200_class_id("desk", idx) == -1


def test_build_object_dict_minimal_schema():
    """Build a Phase-8-shaped object dict from one Mask3D instance."""
    pcd = np.array(
        [[0.0, 0.0, 0.0, 50.0, 60.0, 70.0],
         [1.0, 1.0, 1.0, 80.0, 90.0, 100.0]],
        dtype=np.float32,
    )
    obj = build_object_dict(
        pcd_with_color=pcd,
        label="chair",
        class_idx=2,
        confidence=0.95,
    )
    assert obj["class_name"] == ["chair"]
    assert obj["class_id"] == [2]
    assert obj["is_background"] == 0
    assert obj["num_detections"] == 1
    assert obj["n_points"] == [2]
    assert obj["conf"] == [pytest.approx(0.95)]
    assert obj["bbox_np"].shape == (8, 3)
    assert obj["pcd_np"].shape == (2, 3)
    assert obj["pcd_color_np"].shape == (2, 3)


def test_build_object_dict_pcd_no_color():
    """Mask3D distribution always emits 6-D, but be defensive: 3-D (xyz only) ok."""
    pcd = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    obj = build_object_dict(
        pcd_with_color=pcd,
        label="chair",
        class_idx=2,
        confidence=0.95,
    )
    assert obj["pcd_color_np"] is None
```

- [ ] **Step 1.3: Run tests to confirm RED**

Run:
```bash
source .venv/bin/activate && PYTHONPATH=src pytest src/scripts/tests/test_build_scanrefer_mask3d_cg.py -v 2>&1 | tail -10
```

Expected: ImportError (`scripts.build_scanrefer_mask3d_cg` does not exist) — all 11 tests fail at collection.

- [ ] **Step 1.4: Implement the helpers**

Create `src/scripts/build_scanrefer_mask3d_cg.py` with this EXACT content (helpers section only; the IO functions land in Task 2):

```python
"""Mask3D `.npz` → ConceptGraph-shaped pkl + visibility index converter.

Source: ZSVG3D's CUHK SharePoint distribution (`Mask3d/scannet200/<scene>.npz`),
which itself is repackaged from upstream `mask3d_inst_seg.zip` (Schult 2022).

This producer mirrors the schema of the NR3D Phase 8 GT-CG pkl so all
NR3D pack-prep / runner / aggregator code can be reused unchanged. The
ScanRefer loader points at this output for the proposal pool.

Source citations:
- ZSVG3D `process_mask3d.ipynb` cell 3 — original .npz packaging
- ZSVG3D `zsvg/loc_interpreters_pred.py:16-30` — bbox derivation = (min+max)/2 center, max-min extent
- conceptgraph/scannet200_classes.txt — class taxonomy used for class_id lookup
- src/scripts/build_visibility_index.py::build_visibility_index — visibility helper reused as-is
"""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import numpy as np

BACKGROUND_LABELS: frozenset[str] = frozenset({"wall", "floor", "ceiling"})


def axis_aligned_corners_from_pcd(pcd: np.ndarray) -> np.ndarray:
    """Return 8 corners of the axis-aligned bbox enclosing ``pcd[:, :3]``.

    Args:
        pcd: (N, 3) or (N, 6+) float array. Only the first 3 columns (XYZ) are used.

    Returns:
        (8, 3) float array, ordered with [min,min,min], [max,min,min], etc.
        Matches the ZSVG3D / Phase 8 GT-CG axis-aligned 8-corner convention.

    Raises:
        ValueError: if pcd is empty.
    """
    if pcd.shape[0] == 0:
        raise ValueError("pcd is empty; cannot derive bbox")
    xyz = np.asarray(pcd[:, :3], dtype=np.float64)
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    corners = np.array(
        [
            [mn[0], mn[1], mn[2]],
            [mx[0], mn[1], mn[2]],
            [mn[0], mx[1], mn[2]],
            [mx[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mx[0], mn[1], mx[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mx[1], mx[2]],
        ],
        dtype=np.float64,
    )
    return corners


def is_background_label(label: str) -> bool:
    """Return True if ``label`` (case-insensitive) names a structural element."""
    if not label:
        return False
    return label.lower() in BACKGROUND_LABELS


def load_scannet200_class_index(taxonomy_file: Path) -> dict[str, int]:
    """Load lower-cased label → integer-index from ScanNet200 class file.

    Args:
        taxonomy_file: Path to ``conceptgraph/scannet200_classes.txt`` —
            one class name per line, in canonical order.

    Returns:
        Dict mapping each lowercased class name to its 0-based line index.
    """
    if not taxonomy_file.exists():
        raise FileNotFoundError(f"ScanNet200 taxonomy not found: {taxonomy_file}")
    out: dict[str, int] = {}
    with open(taxonomy_file, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            label = line.strip()
            if not label:
                continue
            out[label.lower()] = idx
    return out


def scannet200_class_id(label: str, taxonomy: dict[str, int]) -> int:
    """Return canonical class index for ``label``, or -1 if unknown."""
    if not label:
        return -1
    return taxonomy.get(label.lower(), -1)


def build_object_dict(
    *,
    pcd_with_color: np.ndarray,
    label: str,
    class_idx: int,
    confidence: float,
) -> dict:
    """Build a Phase 8 GT-CG-shaped object dict from one Mask3D instance.

    The schema matches what ``build_proposals_from_phase8_objects`` consumes
    (see ``src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py``).

    Args:
        pcd_with_color: (N, 3) XYZ-only or (N, 6+) XYZ+RGB Mask3D points.
        label: ScanNet200 class string from `ins_labels[i]`.
        class_idx: Canonical 0-based index in the ScanNet200 taxonomy, or -1.
        confidence: Mask3D `ins_scores[i]` (recorded only; not consumed by agent).

    Returns:
        Dict with the minimal Phase-8-shaped fields:
        bbox_np (8,3), class_name, class_id, pcd_np, pcd_color_np,
        is_background, num_detections, n_points, conf.
    """
    arr = np.asarray(pcd_with_color, dtype=np.float64)
    bbox_np = axis_aligned_corners_from_pcd(arr)
    xyz = arr[:, :3].copy()
    if arr.shape[1] >= 6:
        rgb = arr[:, 3:6].copy()
    else:
        rgb = None
    return {
        "bbox_np": bbox_np,
        "class_name": [str(label)],
        "class_id": [int(class_idx)],
        "pcd_np": xyz,
        "pcd_color_np": rgb,
        "is_background": 0,
        "num_detections": 1,
        "n_points": [int(len(xyz))],
        "conf": [float(confidence)],
    }
```

- [ ] **Step 1.5: Run tests to confirm GREEN**

Run:
```bash
source .venv/bin/activate && PYTHONPATH=src pytest src/scripts/tests/test_build_scanrefer_mask3d_cg.py -v 2>&1 | tail -20
```

Expected: 11 tests pass.

- [ ] **Step 1.6: Commit**

```bash
git add src/scripts/build_scanrefer_mask3d_cg.py \
        src/scripts/tests/__init__.py \
        src/scripts/tests/test_build_scanrefer_mask3d_cg.py
git commit -m "feat(scanrefer): add Mask3D-CG converter pure helpers

Pure helper functions with no IO: axis_aligned_corners_from_pcd (matches
ZSVG3D's (min+max)/2 derivation), background-label filter, ScanNet200
class taxonomy loader, and Phase-8-shaped object dict builder. Schema
output matches what NR3D pack-prep already consumes, so downstream
infrastructure is reused unchanged.

Companion tests cover bbox derivation on (N,3)/(N,6) inputs, empty-pcd
guard, label case-insensitivity, taxonomy lookup, and full object_dict
schema."
```

---

## Task 2: Mask3D-CG converter IO + CLI

**Files:**
- Modify: `src/scripts/build_scanrefer_mask3d_cg.py` (append IO functions + main)
- Modify: `src/scripts/tests/test_build_scanrefer_mask3d_cg.py` (append IO smoke test)

This task wires the helpers into a per-scene IO pipeline, projects Mask3D points into the existing posed RGB frames to build a visibility index, and emits the per-scene pkl + visibility + scene_info.

- [ ] **Step 2.1: Append IO smoke test**

Append to `src/scripts/tests/test_build_scanrefer_mask3d_cg.py`:

```python
def test_build_one_scene_smoke(tmp_path):
    """Smoke test: synthetic mask3d npz + minimal raw dir → produced pkl + visibility."""
    import json

    from scripts.build_scanrefer_mask3d_cg import build_mask3d_cg_for_scene

    # Synthetic Mask3D npz: 2 instances (1 chair, 1 wall — wall should be filtered)
    npz_path = tmp_path / "scene_test.npz"
    pcd_chair = np.random.rand(100, 6).astype(np.float32) * np.array([1, 1, 1, 255, 255, 255])
    pcd_wall = np.random.rand(50, 6).astype(np.float32) * np.array([3, 3, 3, 255, 255, 255])
    np.savez_compressed(
        npz_path,
        ins_pcds=np.array([pcd_chair, pcd_wall], dtype=object),
        ins_labels=np.array(["chair", "wall"], dtype="<U16"),
        ins_scores=np.array([0.9, 0.7], dtype=np.float32),
    )

    # Synthetic raw dir with 1 frame: identity pose, identity intrinsic
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    np.savetxt(raw_dir / "intrinsic_color.txt",
               np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64))
    np.savetxt(raw_dir / "000000.txt", np.eye(4, dtype=np.float64))
    # scene_info.json with kept_frame_ids=[0]
    (raw_dir / "scene_info.json").write_text(
        json.dumps({"kept_frame_ids": [0]}), encoding="utf-8"
    )

    # Outputs
    pkl_out = tmp_path / "out.pkl.gz"
    vis_out = tmp_path / "vis.pkl"
    info_out = tmp_path / "scene_info.json"

    summary = build_mask3d_cg_for_scene(
        scene_id="scene_test",
        mask3d_npz_path=npz_path,
        raw_dir=raw_dir,
        output_pkl=pkl_out,
        output_visibility=vis_out,
        output_scene_info=info_out,
        scannet200_taxonomy=tmp_path / "scannet200_classes.txt",
        drop_background=True,
    )
    assert summary["n_kept"] == 1   # chair kept, wall dropped
    assert summary["n_dropped"] == 1
    assert pkl_out.exists()
    assert vis_out.exists()
    assert info_out.exists()


def test_load_scannet200_taxonomy_uses_default_taxonomy(tmp_path, monkeypatch):
    """If --scannet200-taxonomy isn't provided, code should default to repo file."""
    from scripts.build_scanrefer_mask3d_cg import DEFAULT_SCANNET200_TAXONOMY
    assert DEFAULT_SCANNET200_TAXONOMY.name == "scannet200_classes.txt"
```

Note: the smoke test creates an empty taxonomy file at `tmp_path / "scannet200_classes.txt"` since the test passes that path explicitly; `chair` not being in that taxonomy would yield `class_id=-1` which is fine.

- [ ] **Step 2.2: Run new test to confirm RED**

Run:
```bash
source .venv/bin/activate && PYTHONPATH=src pytest src/scripts/tests/test_build_scanrefer_mask3d_cg.py::test_build_one_scene_smoke src/scripts/tests/test_build_scanrefer_mask3d_cg.py::test_load_scannet200_taxonomy_uses_default_taxonomy -v 2>&1 | tail -10
```

Expected: ImportError on `build_mask3d_cg_for_scene` and `DEFAULT_SCANNET200_TAXONOMY`.

- [ ] **Step 2.3: Append IO functions + CLI to the converter**

Open `src/scripts/build_scanrefer_mask3d_cg.py` and append (after the helpers from Task 1):

```python


# ---------- IO + Visibility ----------

import argparse
import json

DEFAULT_SCANNET200_TAXONOMY = Path("conceptgraph/scannet200_classes.txt")
DEFAULT_MASK3D_ROOT = Path("data/scanrefer/Mask3d/scannet200")
DEFAULT_RAW_ROOT = Path("data/nr3d/scannet")
DEFAULT_OUTPUT_ROOT = Path("data/scanrefer/scannet")


def _load_camera(raw_dir: Path) -> tuple[np.ndarray, list[np.ndarray], list[Path]]:
    """Load intrinsics + per-frame cam-to-world poses + depth paths from raw dir."""
    intr_path = raw_dir / "intrinsic_color.txt"
    if not intr_path.exists():
        raise FileNotFoundError(f"intrinsic_color.txt missing: {intr_path}")
    intr_mat = np.loadtxt(intr_path)
    if intr_mat.shape == (4, 4):
        intr_mat = intr_mat[:3, :3]

    info_path = raw_dir / "scene_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"scene_info.json missing: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    kept = info.get("kept_frame_ids")
    if not isinstance(kept, list):
        raise ValueError(f"scene_info.json missing kept_frame_ids: {info_path}")

    poses: list[np.ndarray] = []
    depth_paths: list[Path] = []
    for frame_id in kept:
        pose_path = raw_dir / f"{int(frame_id):06d}.txt"
        if not pose_path.exists():
            raise FileNotFoundError(f"pose missing: {pose_path}")
        poses.append(np.loadtxt(pose_path))
        depth = raw_dir / f"{int(frame_id):06d}-depth.png"
        depth_paths.append(depth if depth.exists() else None)
    return intr_mat, poses, depth_paths


def build_mask3d_cg_for_scene(
    *,
    scene_id: str,
    mask3d_npz_path: Path,
    raw_dir: Path,
    output_pkl: Path,
    output_visibility: Path,
    output_scene_info: Path,
    scannet200_taxonomy: Path = DEFAULT_SCANNET200_TAXONOMY,
    drop_background: bool = True,
) -> dict:
    """Convert one scene's Mask3D `.npz` into ConceptGraph-shaped pkl + visibility.

    Returns:
        Summary dict with n_kept, n_dropped, n_visibility_mappings, output paths.

    Raises:
        FileNotFoundError: if any required input is missing.
    """
    if not mask3d_npz_path.exists():
        raise FileNotFoundError(f"Mask3D npz missing: {mask3d_npz_path}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw dir missing: {raw_dir}")
    taxonomy = (
        load_scannet200_class_index(scannet200_taxonomy)
        if scannet200_taxonomy.exists()
        else {}
    )

    data = np.load(mask3d_npz_path, allow_pickle=True)
    ins_pcds = data["ins_pcds"]
    ins_labels = data["ins_labels"]
    ins_scores = data.get("ins_scores")
    if ins_scores is None:
        ins_scores = np.ones(len(ins_pcds), dtype=np.float32)

    objects: list[dict] = []
    n_dropped = 0
    for i in range(len(ins_pcds)):
        label = str(ins_labels[i])
        if drop_background and is_background_label(label):
            n_dropped += 1
            continue
        pcd = ins_pcds[i]
        if pcd is None or len(pcd) == 0:
            n_dropped += 1
            continue
        try:
            obj = build_object_dict(
                pcd_with_color=pcd,
                label=label,
                class_idx=scannet200_class_id(label, taxonomy),
                confidence=float(ins_scores[i]),
            )
        except ValueError:
            n_dropped += 1
            continue
        objects.append(obj)
    n_kept = len(objects)

    # Build visibility index by reusing the existing helper.
    intr, poses, depth_paths = _load_camera(raw_dir)
    from scripts.build_visibility_index import (
        build_visibility_index,
        save_visibility_index,
    )

    object_to_views, view_to_objects = build_visibility_index(
        objects=objects,
        poses=poses,
        depth_paths=depth_paths,
        intrinsics=intr,
        max_distance=5.0,
        use_depth=True,
        stride=1,
    )
    n_visibility_mappings = sum(len(v) for v in view_to_objects.values())

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_pkl, "wb") as f:
        pickle.dump({"objects": objects, "bg_objects": []}, f)

    output_visibility.parent.mkdir(parents=True, exist_ok=True)
    save_visibility_index(
        object_to_views=object_to_views,
        view_to_objects=view_to_objects,
        out_path=output_visibility,
    )

    output_scene_info.parent.mkdir(parents=True, exist_ok=True)
    output_scene_info.write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "source_npz": str(mask3d_npz_path),
                "raw_dir": str(raw_dir),
                "num_objects": n_kept,
                "num_dropped_background": n_dropped,
                "num_rgb_frames": len(poses),
                "num_visibility_mappings": n_visibility_mappings,
                "source": "scanrefer-mask3d-cg-producer",
                "bbox_geometry": "axis-aligned 8 corners from Mask3D pcd (min/max)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "n_kept": n_kept,
        "n_dropped": n_dropped,
        "n_visibility_mappings": n_visibility_mappings,
        "output_pkl": str(output_pkl),
        "output_visibility": str(output_visibility),
        "output_scene_info": str(output_scene_info),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes", nargs="*", default=None,
        help="Scene IDs to process. If omitted, --scene-list is used.",
    )
    parser.add_argument(
        "--scene-list", type=Path,
        default=Path("data/scanrefer/raw/ScanRefer_filtered_val.txt"),
    )
    parser.add_argument("--mask3d-root", type=Path, default=DEFAULT_MASK3D_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scannet200-taxonomy", type=Path,
                        default=DEFAULT_SCANNET200_TAXONOMY)
    parser.add_argument("--no-drop-background", action="store_true")
    parser.add_argument("--report", type=Path,
                        default=Path("tmp/scanrefer_handoff/converter_report.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scenes:
        scene_ids = args.scenes
    else:
        scene_ids = args.scene_list.read_text(encoding="utf-8").split()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    for sc in scene_ids:
        npz = args.mask3d_root / f"{sc}.npz"
        raw = args.raw_root / sc / "raw"
        out_pkl = args.output_root / sc / "conceptgraph" / "pcd_saves" / "full_pcd_mask3d_axisaligned.pkl.gz"
        out_vis = args.output_root / sc / "conceptgraph" / "indices" / "visibility_index.pkl"
        out_info = args.output_root / sc / "conceptgraph" / "scene_info.json"
        summary = build_mask3d_cg_for_scene(
            scene_id=sc,
            mask3d_npz_path=npz,
            raw_dir=raw,
            output_pkl=out_pkl,
            output_visibility=out_vis,
            output_scene_info=out_info,
            scannet200_taxonomy=args.scannet200_taxonomy,
            drop_background=not args.no_drop_background,
        )
        summaries.append({"scene_id": sc, **summary})
        print(
            f"{sc}: kept={summary['n_kept']} dropped={summary['n_dropped']} "
            f"vis_mappings={summary['n_visibility_mappings']}"
        )

    args.report.write_text(
        json.dumps({"scenes": summaries, "n_scenes": len(summaries)}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote summary report to {args.report}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.4: Run tests to confirm GREEN**

Run:
```bash
source .venv/bin/activate && PYTHONPATH=src pytest src/scripts/tests/test_build_scanrefer_mask3d_cg.py -v 2>&1 | tail -20
```

Expected: 13 tests pass.

- [ ] **Step 2.5: Commit**

```bash
git add src/scripts/build_scanrefer_mask3d_cg.py \
        src/scripts/tests/test_build_scanrefer_mask3d_cg.py
git commit -m "feat(scanrefer): Mask3D-CG converter IO + CLI

build_mask3d_cg_for_scene() reads ZSVG3D .npz, builds Phase-8-shaped
object list, projects Mask3D pcds via existing build_visibility_index
into kept frames, writes pkl + visibility + scene_info per scene.
CLI accepts --scenes <ids> or --scene-list, defaults to data/scanrefer
ScanRefer val txt list."
```

---

## Task 3: Run converter on all 141 ScanRefer val scenes

**Files:** none modified (read-only producer run + verification).

- [ ] **Step 3.1: Verify all inputs exist**

Run:
```bash
source .venv/bin/activate && python -c "
from pathlib import Path
val_scans = Path('data/scanrefer/raw/ScanRefer_filtered_val.txt').read_text().split()
miss = []
for sc in val_scans:
    npz = Path(f'data/scanrefer/Mask3d/scannet200/{sc}.npz')
    raw = Path(f'data/nr3d/scannet/{sc}/raw')
    if not npz.exists() or not raw.exists():
        miss.append((sc, npz.exists(), raw.exists()))
print(f'val scenes: {len(val_scans)}, missing inputs: {len(miss)}')
for sc, npz_ok, raw_ok in miss[:5]:
    print(f'  {sc}: npz={npz_ok} raw={raw_ok}')
"
```

Expected: `val scenes: 141, missing inputs: 0`. If any scene is missing, stop and report — the data prep from earlier turns should have covered all 141 (verified post-rsync at the end of the data-prep cycle).

- [ ] **Step 3.2: Run converter on all 141 scenes**

```bash
mkdir -p tmp/scanrefer_handoff
source .venv/bin/activate
PYTHONPATH=src python -m scripts.build_scanrefer_mask3d_cg \
    --scene-list data/scanrefer/raw/ScanRefer_filtered_val.txt \
    --mask3d-root data/scanrefer/Mask3d/scannet200 \
    --raw-root data/nr3d/scannet \
    --output-root data/scanrefer/scannet \
    --scannet200-taxonomy conceptgraph/scannet200_classes.txt \
    --report tmp/scanrefer_handoff/converter_report.json \
    2>&1 | tee tmp/scanrefer_handoff/converter.log
```

Expected: 141 stdout lines like `scene0011_00: kept=42 dropped=8 vis_mappings=873`. Wall: 25-40 min (CPU-only, dominated by visibility projection).

- [ ] **Step 3.3: Verify outputs**

```bash
source .venv/bin/activate && python -c "
import gzip, pickle, json
from pathlib import Path
val_scans = Path('data/scanrefer/raw/ScanRefer_filtered_val.txt').read_text().split()
ok = bad = 0
for sc in val_scans:
    pkl = Path(f'data/scanrefer/scannet/{sc}/conceptgraph/pcd_saves/full_pcd_mask3d_axisaligned.pkl.gz')
    vis = Path(f'data/scanrefer/scannet/{sc}/conceptgraph/indices/visibility_index.pkl')
    info = Path(f'data/scanrefer/scannet/{sc}/conceptgraph/scene_info.json')
    if not (pkl.exists() and vis.exists() and info.exists()):
        bad += 1
        print(f'BAD {sc}: pkl={pkl.exists()} vis={vis.exists()} info={info.exists()}')
        continue
    with gzip.open(pkl, 'rb') as f:
        objs = pickle.load(f)['objects']
    if len(objs) < 5:
        bad += 1
        print(f'BAD {sc}: only {len(objs)} objects')
        continue
    if objs[0]['bbox_np'].shape != (8, 3):
        bad += 1
        print(f'BAD {sc}: bbox_np shape {objs[0][\"bbox_np\"].shape}')
        continue
    ok += 1
print(f'\\n{ok}/{len(val_scans)} scenes OK; {bad} bad')
"
```

Expected: `141/141 scenes OK; 0 bad`.

- [ ] **Step 3.4: No commit needed (data/ is gitignored)**

The converter outputs land under `data/scanrefer/scannet/`, which is git-ignored. Confirm clean working tree:

```bash
git status -s | grep -v scheduled_tasks.lock
```

Expected: only the `tmp/scanrefer_handoff/converter_report.json` and `converter.log` (also gitignored under `tmp/`); no source-file changes.

---

## Task 4: ScanRefer loader (RED → GREEN)

**Files:**
- Create: `src/benchmarks/scanrefer_loader.py`
- Create: `src/benchmarks/tests/test_scanrefer_loader.py`

- [ ] **Step 4.1: Write the failing test file**

Create `src/benchmarks/tests/test_scanrefer_loader.py`:

```python
"""Tests for ScanRefer VG loader."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from benchmarks.scanrefer_loader import (
    ScanRefVGDataset,
    ScanRefVGSample,
    parse_scanrefer_sample_id,
)


@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    """Build a fake ScanRefer-shaped data tree under tmp_path.

    Returns the root that should be passed as `data_root`.
    """
    # ScanRefer JSON
    raw = tmp_path / "scanrefer/raw"
    raw.mkdir(parents=True)
    (raw / "ScanRefer_filtered_val.json").write_text(
        json.dumps([
            {"scene_id": "scene_a", "object_id": "0", "object_name": "chair",
             "ann_id": "0", "description": "the red chair", "token": ["the","red","chair"]},
            {"scene_id": "scene_a", "object_id": "1", "object_name": "chair",
             "ann_id": "1", "description": "the blue chair", "token": ["the","blue","chair"]},
            {"scene_id": "scene_a", "object_id": "2", "object_name": "table",
             "ann_id": "0", "description": "the wooden table", "token": ["the","wooden","table"]},
        ]),
        encoding="utf-8",
    )

    # Phase-8 GT-CG pkl mock for scene_a (3 GT instances: 2 chairs + 1 table)
    cg = tmp_path / "phase8/scene_a/conceptgraph/pcd_saves"
    cg.mkdir(parents=True)
    objs = []
    for i, label in enumerate(["chair", "chair", "table"]):
        corners = np.array([[i, 0, 0],[i+1, 0, 0],[i, 1, 0],[i+1, 1, 0],
                            [i, 0, 1],[i+1, 0, 1],[i, 1, 1],[i+1, 1, 1]], dtype=np.float64)
        objs.append({"bbox_np": corners, "class_name": [label], "class_id": [i]})
    with gzip.open(cg / "full_pcd_gt_axisaligned_post.pkl.gz", "wb") as f:
        pickle.dump({"objects": objs, "bg_objects": []}, f)

    return tmp_path


def test_parse_scanrefer_sample_id_canonical_form():
    parsed = parse_scanrefer_sample_id("scannet/scene0088_00::5::3")
    assert parsed.scan_id == "scannet/scene0088_00"
    assert parsed.scene_id == "scene0088_00"
    assert parsed.target_id == 5
    assert parsed.ann_id == "3"


def test_parse_scanrefer_sample_id_rejects_malformed():
    with pytest.raises(ValueError, match="format"):
        parse_scanrefer_sample_id("scannet/scene0088_00::5")


def test_load_returns_3_samples(fake_dataset: Path):
    ds = ScanRefVGDataset.from_path(
        data_root=fake_dataset / "scanrefer",
        phase8_data_root=fake_dataset / "phase8",
        split="val",
    )
    assert len(ds) == 3


def test_sample_id_format(fake_dataset: Path):
    ds = ScanRefVGDataset.from_path(
        data_root=fake_dataset / "scanrefer",
        phase8_data_root=fake_dataset / "phase8",
        split="val",
    )
    sids = [s.sample_id for s in ds]
    assert "scannet/scene_a::0::0" in sids
    assert "scannet/scene_a::1::1" in sids


def test_is_unique_field_chair_count_equals_two(fake_dataset: Path):
    """Two chairs in scene_a → both chair samples have is_unique=False; table is_unique=True."""
    ds = ScanRefVGDataset.from_path(
        data_root=fake_dataset / "scanrefer",
        phase8_data_root=fake_dataset / "phase8",
        split="val",
    )
    by_target = {s.target: [] for s in ds}
    for s in ds:
        by_target[s.target].append(s.is_unique)
    assert all(u is False for u in by_target["chair"])
    assert by_target["table"] == [True]


def test_gt_bbox_present_and_correct_shape(fake_dataset: Path):
    ds = ScanRefVGDataset.from_path(
        data_root=fake_dataset / "scanrefer",
        phase8_data_root=fake_dataset / "phase8",
        split="val",
    )
    s = ds[0]
    assert s.gt_bbox_3d is not None
    assert len(s.gt_bbox_3d) == 9   # 9-DoF (Euler=0 for axis-aligned)


def test_filter_by_sample_ids(fake_dataset: Path):
    """When sample_ids is provided, loader returns only matching utterances."""
    ds = ScanRefVGDataset.from_path(
        data_root=fake_dataset / "scanrefer",
        phase8_data_root=fake_dataset / "phase8",
        split="val",
        sample_ids={"scannet/scene_a::2::0"},
    )
    assert len(ds) == 1
    assert ds[0].target == "table"


def test_skipped_when_phase8_missing(fake_dataset: Path):
    """If a sample's scene has no Phase 8 pkl, it's skipped (not crashed)."""
    # Add an utterance for a scene without a Phase 8 pkl
    raw = fake_dataset / "scanrefer/raw/ScanRefer_filtered_val.json"
    data = json.loads(raw.read_text())
    data.append({
        "scene_id": "scene_b", "object_id": "0", "object_name": "lamp",
        "ann_id": "0", "description": "the lamp", "token": ["the", "lamp"],
    })
    raw.write_text(json.dumps(data), encoding="utf-8")

    ds = ScanRefVGDataset.from_path(
        data_root=fake_dataset / "scanrefer",
        phase8_data_root=fake_dataset / "phase8",
        split="val",
    )
    # scene_b utterance is skipped
    assert len(ds) == 3
    assert ds.stats["skipped_missing_scene"] == 1
```

- [ ] **Step 4.2: Create __init__ for benchmarks/tests if missing**

```bash
test -f src/benchmarks/tests/__init__.py || touch src/benchmarks/tests/__init__.py
```

- [ ] **Step 4.3: Run tests to confirm RED**

Run:
```bash
source .venv/bin/activate && PYTHONPATH=src pytest src/benchmarks/tests/test_scanrefer_loader.py -v 2>&1 | tail -10
```

Expected: ImportError on `benchmarks.scanrefer_loader` — all tests fail.

- [ ] **Step 4.4: Implement the loader**

Create `src/benchmarks/scanrefer_loader.py`:

```python
"""ScanRefer visual-grounding benchmark loader.

ScanRefer (Chen et al. ECCV 2020): natural-language ScanNet references with
detection-mode evaluation. Each utterance has a target ScanNet `objectId`
and a free-form description.

This loader emits ``ScanRefVGSample`` records with:
- query (the natural-language description)
- target_id (the ScanNet objectId, indexes the GT pool — Phase 8 GT-CG pkl)
- gt_bbox_3d (9-DoF axis-aligned, derived from Phase 8 8-corner bbox)
- is_unique (True iff the scene contains exactly one GT instance whose
  class equals the target's `object_name`)

The proposal pool (Mask3D predictions) is NOT loaded here — it lives in a
separate Mask3D-CG pkl produced by ``scripts/build_scanrefer_mask3d_cg.py``
and is consumed by ``prepare_pack_v1_inputs_scanrefer.py``.
"""

from __future__ import annotations

import gzip
import json
import pickle
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from .base import BenchmarkSample

_PHASE8_PCD_REL = Path("conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz")

_STATS_KEYS = [
    "total_loaded",
    "skipped_missing_scene",
    "skipped_missing_or_ambiguous_bbox",
]


@dataclass(frozen=True)
class ParsedScanRefSampleId:
    scan_id: str
    scene_id: str
    target_id: int
    ann_id: str


def parse_scanrefer_sample_id(sample_id: str) -> ParsedScanRefSampleId:
    """Parse ``scannet/<scene>::<target_id>::<ann_id>`` form."""
    parts = sample_id.split("::")
    if len(parts) != 3:
        raise ValueError(
            "Expected sample_id in '<scan_id>::<target_id>::<ann_id>' "
            f"format, got {sample_id!r}"
        )
    scan_id, target_text, ann_id = parts
    try:
        target_id = int(target_text)
    except ValueError as exc:
        raise ValueError(f"Invalid target_id in sample_id={sample_id!r}") from exc
    return ParsedScanRefSampleId(
        scan_id=scan_id,
        scene_id=scan_id.split("/")[-1],
        target_id=target_id,
        ann_id=ann_id,
    )


@dataclass
class ScanRefVGSample(BenchmarkSample):
    """ScanRefer visual-grounding sample."""

    scan_id: str = ""
    target_id: int = -1
    target: str = ""             # ScanRefer JSON's object_name
    ann_id: str = ""
    description: str = ""        # alias of query, kept for clarity
    is_unique: bool = False
    gt_bbox_3d: list[float] | None = None

    @property
    def text(self) -> str:
        """VG referring expression (alias for query)."""
        return self.query


def _phase8_corners_to_9dof(corners) -> list[float]:
    """Reuse the same conversion NR3D uses (axis-aligned, Euler=0)."""
    import numpy as np
    arr = np.asarray(corners, dtype=np.float64)
    if arr.shape != (8, 3):
        raise ValueError(f"corners shape {arr.shape}, expected (8,3)")
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    cx, cy, cz = ((mn + mx) / 2.0).tolist()
    dx, dy, dz = (mx - mn).tolist()
    return [cx, cy, cz, dx, dy, dz, 0.0, 0.0, 0.0]


class ScanRefVGDataset:
    """ScanRefer VG dataset backed by Phase 8 GT-CG pkl for GT bbox lookup."""

    def __init__(
        self,
        samples: list[ScanRefVGSample],
        stats: dict[str, int],
        split: str,
    ) -> None:
        self._samples = samples
        self._stats = stats
        self._split = split

    @classmethod
    def from_path(
        cls,
        data_root: str | Path,
        phase8_data_root: str | Path,
        split: str = "val",
        sample_ids: set[str] | None = None,
    ) -> ScanRefVGDataset:
        """Load ScanRefer VG samples joined to Phase 8 GT-CG GT bboxes.

        Args:
            data_root: Directory containing ``raw/ScanRefer_filtered_<split>.json``.
            phase8_data_root: Directory containing ``<scene>/conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz``.
            split: ``"val"`` or ``"train"``.
            sample_ids: Optional restriction to specific sample_ids.

        Raises:
            FileNotFoundError: If ScanRefer JSON is missing.
        """
        if split not in {"val", "train"}:
            raise ValueError(f"Unknown ScanRefer split {split!r}")

        data_root = Path(data_root)
        phase8_data_root = Path(phase8_data_root)
        json_path = data_root / "raw" / f"ScanRefer_filtered_{split}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"ScanRefer JSON missing: {json_path}")

        utterances = json.loads(json_path.read_text(encoding="utf-8"))

        # Cache per-scene Phase 8 objects + class counts (for is_unique).
        scene_cache: dict[str, list[dict[str, Any]] | None] = {}
        scene_class_counts: dict[str, Counter] = {}

        def _load_scene(scene: str) -> list[dict[str, Any]] | None:
            if scene in scene_cache:
                return scene_cache[scene]
            pkl_path = phase8_data_root / scene / _PHASE8_PCD_REL
            if not pkl_path.exists():
                scene_cache[scene] = None
                return None
            with gzip.open(pkl_path, "rb") as f:
                objs = pickle.load(f).get("objects") or []
            scene_cache[scene] = objs
            counts: Counter = Counter()
            for o in objs:
                cn = o.get("class_name")
                if isinstance(cn, list) and cn:
                    counts[cn[0].lower()] += 1
            scene_class_counts[scene] = counts
            return objs

        stats = dict.fromkeys(_STATS_KEYS, 0)
        samples: list[ScanRefVGSample] = []
        requested = set(sample_ids) if sample_ids is not None else None

        for row in utterances:
            scene = row["scene_id"]
            target_id = int(row["object_id"])
            ann_id = row["ann_id"]
            target_name = row["object_name"]
            sample_id = f"scannet/{scene}::{target_id}::{ann_id}"
            if requested is not None and sample_id not in requested:
                continue

            objs = _load_scene(scene)
            if objs is None:
                stats["skipped_missing_scene"] += 1
                continue
            if target_id < 0 or target_id >= len(objs):
                stats["skipped_missing_or_ambiguous_bbox"] += 1
                continue

            corners = objs[target_id].get("bbox_np")
            if corners is None:
                stats["skipped_missing_or_ambiguous_bbox"] += 1
                continue
            try:
                gt_bbox = _phase8_corners_to_9dof(corners)
            except ValueError:
                stats["skipped_missing_or_ambiguous_bbox"] += 1
                continue

            counts = scene_class_counts.get(scene, Counter())
            is_unique = counts.get(target_name.lower(), 0) == 1

            description = row["description"]
            samples.append(
                ScanRefVGSample(
                    sample_id=sample_id,
                    scene_id=scene,
                    query=description,
                    scan_id=f"scannet/{scene}",
                    target_id=target_id,
                    target=target_name,
                    ann_id=ann_id,
                    description=description,
                    is_unique=is_unique,
                    gt_bbox_3d=gt_bbox,
                    metadata={"token": list(row.get("token", []))},
                )
            )

        stats["total_loaded"] = len(samples)
        logger.info(
            "Built {} ScanRefer samples (split={}, stats={})",
            len(samples), split, stats,
        )
        return cls(samples=samples, stats=stats, split=split)

    def __iter__(self) -> Iterator[ScanRefVGSample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> ScanRefVGSample:
        return self._samples[idx]

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    @property
    def split(self) -> str:
        return self._split
```

- [ ] **Step 4.5: Run tests to confirm GREEN**

Run:
```bash
source .venv/bin/activate && PYTHONPATH=src pytest src/benchmarks/tests/test_scanrefer_loader.py -v 2>&1 | tail -15
```

Expected: 8 tests pass.

- [ ] **Step 4.6: Commit**

```bash
git add src/benchmarks/scanrefer_loader.py \
        src/benchmarks/tests/test_scanrefer_loader.py
git commit -m "feat(scanrefer): add ScanRefer loader with Phase 8 GT lookup

ScanRefVGDataset joins ScanRefer JSON with Phase 8 GT-CG pkl for GT bbox
derivation (axis-aligned 9-DoF, Euler=0). Computes is_unique from
per-scene same-class instance counts. Sample_id format
scannet/<scene>::<target_id>::<ann_id> mirrors NR3D convention.

Companion tests cover sample_id parsing, fold size, sample_id format,
is_unique computation, gt_bbox shape, sample_ids filtering, and
missing-Phase-8 scene skip."
```

---

## Task 5: Pack-prep mirror

**Files:**
- Create: `src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py`
- Create: `src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer.py`

This task mirrors `prepare_pack_v1_inputs_nr3d.py` with two pkl sources: Mask3D-CG pkl for proposals, Phase 8 GT-CG pkl for GT bbox lookup.

- [ ] **Step 5.1: Read NR3D pack-prep as reference**

Open `src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py` for reference. The ScanRefer version has the same control flow with these specific changes:

| NR3D | ScanRefer |
|---|---|
| `PHASE8_PCD_REL = "conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz"` | `MASK3D_PCD_REL = "conceptgraph/pcd_saves/full_pcd_mask3d_axisaligned.pkl.gz"` |
| `pack_name="pack_nr3d_v1"` | `pack_name="pack_scanrefer_v1"` |
| `Nr3dVGAdapter.from_phase8_test` for samples | `ScanRefVGDataset.from_path` directly |
| `parse_sample_id` returns `(scene, target_id, assignment_id)` | returns `(scene, target_id, ann_id)` |
| GT bbox from `phase8_data_root` (per-scene Phase 8 pkl) | GT bbox from `Nr3dDataset`-equivalent — `scanrefer_loader.ScanRefVGDataset.gt_bbox_3d` (already computed) |
| Proposals from same Phase 8 pkl as GT | Proposals from new Mask3D-CG pkl |

- [ ] **Step 5.2: Write the pack-prep test file**

Create `src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer.py`:

```python
"""Smoke + unit tests for ScanRefer pack-v1 prep."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_parse_sample_id_canonical():
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import parse_sample_id
    scene, target_id, ann_id = parse_sample_id("scannet/scene0088_00::5::3")
    assert scene == "scene0088_00"
    assert target_id == 5
    assert ann_id == "3"


def test_parse_sample_id_rejects_malformed():
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import parse_sample_id
    with pytest.raises(ValueError, match="format"):
        parse_sample_id("scannet/scene_x::5")


def test_safe_sample_id_normalizes_separators():
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import safe_sample_id
    assert safe_sample_id("scannet/scene0088_00::5::3") == "scannet__scene0088_00__5__3"


def test_load_sample_requests_validates_per_row(tmp_path: Path):
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import load_sample_requests
    p = tmp_path / "ids.json"
    p.write_text(json.dumps([
        {"sample_id": "scannet/scene_a::0::0", "scene_id": "scene_a",
         "target_id": 0, "ann_id": "0", "category": "chair"},
    ]))
    reqs = load_sample_requests(p)
    assert len(reqs) == 1
    assert reqs[0].scene_id == "scene_a"
    assert reqs[0].target_id == 0
    assert reqs[0].ann_id == "0"
```

- [ ] **Step 5.3: Implement pack-prep**

Create `src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py` (mirror of NR3D pack-prep with the changes documented in Step 5.1; full code below for unambiguous reference):

```python
"""Prepare offline pack inputs for ScanRefer VG runs from Mask3D-CG output.

Mirrors prepare_pack_v1_inputs_nr3d.py with two pkl sources:
- Proposal pool: Mask3D-CG pkl (full_pcd_mask3d_axisaligned.pkl.gz)
- GT bbox lookup: Phase 8 GT-CG pkl (full_pcd_gt_axisaligned_post.pkl.gz)
  via ScanRefVGDataset (carries gt_bbox_3d on each sample).
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.embodiedscan_bbox_feasibility.render_marks import (
    render_marked_keyframe,
)
from benchmarks.embodiedscan_bbox_feasibility.visibility_index import (
    project_bbox_3d_to_2d,
)
from benchmarks.scanrefer_loader import (
    ScanRefVGDataset,
    ScanRefVGSample,
)
from evaluation.scripts.prepare_pack_v1_inputs import (
    load_image_size,
    normalize_prepared_keyframes,
    validate_bbox_9dof,
    validate_matrix_4x4,
)

MASK3D_PCD_REL = Path("conceptgraph/pcd_saves/full_pcd_mask3d_axisaligned.pkl.gz")
VIS_REL = Path("conceptgraph/indices/visibility_index.pkl")


@dataclass(frozen=True)
class SampleRequest:
    sample_id: str
    scene_id: str
    target_id: int
    ann_id: str
    category: str


@dataclass(frozen=True)
class SceneFrame:
    frame_id: int
    raw_frame_id: int
    rgb_path: Path
    extrinsic_world_to_cam: np.ndarray


@dataclass(frozen=True)
class Mask3dVisibility:
    object_to_views: dict[int, list[tuple[int, float]]]
    view_to_objects: dict[int, list[tuple[int, float]]]


@dataclass(frozen=True)
class SceneArtifacts:
    scene_dir: Path
    proposals_jsonl: Path
    visibility_json: Path
    annotated_dir: Path
    frame_visibility: dict[int, list[int]]
    proposal_ids: list[int]


def parse_sample_id(sample_id: str) -> tuple[str, int, str]:
    parts = sample_id.split("::")
    if len(parts) != 3:
        raise ValueError(
            f"Expected ScanRefer sample_id '<scan_id>::<target_id>::<ann_id>', "
            f"got {sample_id!r}"
        )
    scan_id, target_text, ann_id = parts
    if not scan_id or not ann_id:
        raise ValueError(f"Invalid ScanRefer sample_id={sample_id!r}")
    try:
        target_id = int(target_text)
    except ValueError as exc:
        raise ValueError(f"Invalid target_id in sample_id={sample_id!r}") from exc
    scene = scan_id.split("/")[-1]
    return scene, target_id, ann_id


def safe_sample_id(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("::", "__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-ids", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="ScanRefer scannet root, e.g. data/scanrefer/scannet")
    parser.add_argument("--phase8-data-root", default=Path("data/nr3d/scannet"),
                        type=Path, help="Phase 8 GT-CG root for GT bbox lookup")
    parser.add_argument("--scanrefer-root", default=Path("data/scanrefer"),
                        type=Path, help="Root containing raw/ScanRefer_filtered_*.json")
    parser.add_argument("--raw-frames-root", default=Path("data/nr3d/scannet"),
                        type=Path, help="Root with <scene>/raw/ frames; ScanRefer "
                        "scenes are a superset of NR3D scenes so this is shared.")
    parser.add_argument("--pack-name", default="pack_scanrefer_v1")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def prepare_pack_v1_inputs_scanrefer(
    *,
    sample_ids_path: Path,
    data_root: Path,
    pack_name: str = "pack_scanrefer_v1",
    split: str = "val",
    scanrefer_root: Path = Path("data/scanrefer"),
    phase8_data_root: Path = Path("data/nr3d/scannet"),
    raw_frames_root: Path = Path("data/nr3d/scannet"),
    max_samples: int | None = None,
) -> list[Path]:
    requests = load_sample_requests(sample_ids_path)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        requests = requests[:max_samples]

    requested_sids = {r.sample_id for r in requests}
    ds = ScanRefVGDataset.from_path(
        data_root=scanrefer_root,
        phase8_data_root=phase8_data_root,
        split=split,
        sample_ids=requested_sids,
    )
    sample_lookup: dict[str, ScanRefVGSample] = {s.sample_id: s for s in ds}

    scene_artifacts: dict[str, SceneArtifacts] = {}
    written: list[Path] = []
    for request in requests:
        sample = sample_lookup.get(request.sample_id)
        if sample is None:
            raise ValueError(
                f"No ScanRefer sample for sample_id={request.sample_id!r}"
            )
        if request.scene_id not in scene_artifacts:
            scene_artifacts[request.scene_id] = prepare_scene_artifacts(
                scene_id=request.scene_id,
                data_root=data_root,
                raw_frames_root=raw_frames_root,
                pack_name=pack_name,
            )
        written.append(
            write_sample_artifact(
                request=request,
                sample=sample,
                data_root=data_root,
                raw_frames_root=raw_frames_root,
                scene_artifacts=scene_artifacts[request.scene_id],
            )
        )
    return written


def prepare_scene_artifacts(
    *,
    scene_id: str,
    data_root: Path,
    raw_frames_root: Path,
    pack_name: str = "pack_scanrefer_v1",
) -> SceneArtifacts:
    scene_root = data_root / scene_id
    objects = load_mask3d_objects(scene_root)
    proposals = build_proposals_from_mask3d_objects(objects=objects, scene_id=scene_id)
    if not proposals:
        raise ValueError(f"scene has no Mask3D objects: {scene_id}")
    visibility = load_mask3d_visibility_index(scene_root)
    frame_visibility = {
        frame_id: [obj_id for obj_id, _score in entries]
        for frame_id, entries in visibility.view_to_objects.items()
    }
    proposal_ids = [int(p["id"]) for p in proposals]
    valid_ids = set(proposal_ids)
    for frame_id, ids in frame_visibility.items():
        unknown = sorted(set(ids) - valid_ids)
        if unknown:
            raise ValueError(
                f"{scene_id} visibility frame {frame_id} unknown ids: {unknown}"
            )

    scene_dir = data_root / scene_id / pack_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "proposals.jsonl").write_text(
        json.dumps({
            "source": "mask3d",
            "scene_id": scene_id,
            "axis_align_matrix": None,
            "proposals": proposals,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (scene_dir / "visibility.json").write_text(
        json.dumps(
            {str(k): v for k, v in sorted(frame_visibility.items())},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    raw_scene_root = raw_frames_root / scene_id
    frames = scene_frames(raw_scene_root, sorted(frame_visibility))
    if not frames:
        raise ValueError(f"scene has no visible frames: {scene_id}")
    frame_by_id = {f.frame_id: f for f in frames}
    intrinsic = scene_intrinsic(raw_scene_root)
    image_size = load_image_size(frames[0].rgb_path)
    annotated_dir = scene_dir / "annotated"
    render_annotated_frames(
        proposal_by_id={int(p["id"]): p for p in proposals},
        frame_visibility=frame_visibility,
        frame_by_id=frame_by_id,
        intrinsic=intrinsic,
        image_size=image_size,
        annotated_dir=annotated_dir,
    )
    return SceneArtifacts(
        scene_dir=scene_dir,
        proposals_jsonl=scene_dir / "proposals.jsonl",
        visibility_json=scene_dir / "visibility.json",
        annotated_dir=annotated_dir,
        frame_visibility=frame_visibility,
        proposal_ids=proposal_ids,
    )


def build_proposals_from_mask3d_objects(
    *, objects: list[dict[str, Any]], scene_id: str,
) -> list[dict[str, Any]]:
    """Convert Mask3D-CG object list to proposals.jsonl shape."""
    proposals: list[dict[str, Any]] = []
    for obj_id, obj in enumerate(objects):
        names = obj.get("class_name")
        if not (isinstance(names, list) and names and all(
            isinstance(n, str) and n for n in names
        )):
            continue
        corners = np.asarray(obj["bbox_np"], dtype=np.float64)
        if corners.shape != (8, 3):
            raise ValueError(
                f"{scene_id}::{obj_id} bbox_np shape {corners.shape}, expected (8,3)"
            )
        mn = corners.min(axis=0)
        mx = corners.max(axis=0)
        bbox_9dof = [
            *((mn + mx) / 2.0).tolist(),    # cx, cy, cz
            *(mx - mn).tolist(),            # dx, dy, dz
            0.0, 0.0, 0.0,                  # Euler=0
        ]
        label = Counter(names).most_common(1)[0][0]
        ids = obj.get("class_id") or [-1]
        label_idx = int(Counter(int(v) for v in ids).most_common(1)[0][0])
        proposals.append({
            "id": obj_id,
            "bbox_3d": bbox_9dof,
            "score": 1.0,                   # uniform per Decision 4
            "label": label,
            "label_idx": label_idx,
        })
    return proposals


def write_sample_artifact(
    *, request: SampleRequest, sample: ScanRefVGSample,
    data_root: Path, raw_frames_root: Path, scene_artifacts: SceneArtifacts,
) -> Path:
    visibility = load_mask3d_visibility_index(data_root / request.scene_id)
    keyframes = select_keyframes_from_phase8_target(
        scene_id=request.scene_id,
        target_id=request.target_id,
        raw_frames_root=raw_frames_root,
        k=5,
    )
    normalized = normalize_prepared_keyframes(keyframes, scene_artifacts.annotated_dir)
    gt_bbox = validate_bbox_9dof(sample.gt_bbox_3d, f"{request.sample_id}.gt_bbox_3d_9dof")
    payload = {
        "sample_id": request.sample_id,
        "scene_id": request.scene_id,
        "target_id": request.target_id,
        "ann_id": request.ann_id,
        "category": request.category or sample.target,
        "is_unique": bool(sample.is_unique),
        "query": sample.query,
        "gt_bbox_3d_9dof": gt_bbox,
        "scene_artifacts_dir": str(scene_artifacts.scene_dir),
        "source": "mask3d",
        "keyframes": normalized,
    }
    path = sample_artifact_path(data_root, request,
                                pack_name=scene_artifacts.scene_dir.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def select_keyframes_from_phase8_target(
    *, scene_id: str, target_id: int,
    raw_frames_root: Path, k: int = 5,
) -> list[dict[str, Any]]:
    """Pick top-k frames where the Phase 8 GT target is most visible.

    Uses the Phase 8 visibility index at
    data/nr3d/scannet/<scene>/conceptgraph/indices/visibility_index.pkl
    (note: NOT the Mask3D-CG visibility — keyframe choice is GT-driven so
    the agent sees frames where the target object is actually present).
    """
    phase8_vis_path = raw_frames_root / scene_id / VIS_REL
    if not phase8_vis_path.exists():
        raise FileNotFoundError(f"Phase 8 visibility missing: {phase8_vis_path}")
    with open(phase8_vis_path, "rb") as f:
        payload = pickle.load(f)
    obj_to_views = payload.get("object_to_views") or {}
    views = obj_to_views.get(int(target_id))
    if not views:
        raise ValueError(
            f"no Phase 8-visible frames for target_id={target_id} in {scene_id}"
        )
    keyframes = []
    for kfi, (frame_id, _score) in enumerate(views[:k]):
        keyframes.append({
            "keyframe_idx": kfi,
            "image_path": str(_resolve_raw_rgb_path(
                raw_frames_root / scene_id, int(frame_id)
            )),
            "frame_id": int(frame_id),
        })
    return keyframes


def render_annotated_frames(
    *, proposal_by_id: dict[int, dict[str, Any]],
    frame_visibility: dict[int, list[int]],
    frame_by_id: dict[int, SceneFrame],
    intrinsic: np.ndarray, image_size: tuple[int, int],
    annotated_dir: Path,
) -> None:
    for frame_id, visible_ids in frame_visibility.items():
        if frame_id not in frame_by_id:
            raise ValueError(f"visibility references missing frame_id={frame_id}")
        frame = frame_by_id[frame_id]
        marks = []
        for prop_id in visible_ids:
            prop = proposal_by_id.get(int(prop_id))
            if prop is None:
                raise ValueError(f"unknown proposal_id={prop_id}")
            rect = project_bbox_3d_to_2d(
                prop["bbox_3d"], intrinsic, frame.extrinsic_world_to_cam, image_size,
            )
            if rect is None:
                continue
            marks.append({
                "proposal_id": int(prop_id),
                "label": prop["label"],
                "bbox_2d": rect,
            })
        render_marked_keyframe(
            rgb_path=frame.rgb_path,
            out_path=annotated_dir / f"frame_{frame_id}.png",
            marks=marks,
        )


def scene_frames(scene_root: Path, frame_ids: Sequence[int]) -> list[SceneFrame]:
    frames: list[SceneFrame] = []
    for frame_id in frame_ids:
        raw_id = _raw_frame_id_for_view(scene_root, int(frame_id))
        pose_path = scene_root / "raw" / f"{raw_id:06d}.txt"
        if not pose_path.exists():
            raise FileNotFoundError(f"Missing pose: {pose_path}")
        cam_to_world = validate_matrix_4x4(np.loadtxt(pose_path), field_name=str(pose_path))
        frames.append(SceneFrame(
            frame_id=int(frame_id), raw_frame_id=raw_id,
            rgb_path=_resolve_raw_rgb_path(scene_root, int(frame_id)),
            extrinsic_world_to_cam=np.linalg.inv(cam_to_world),
        ))
    return frames


def scene_intrinsic(scene_root: Path) -> np.ndarray:
    p = scene_root / "raw" / "intrinsic_color.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing intrinsic: {p}")
    mat = np.asarray(np.loadtxt(p), dtype=float)
    if mat.shape == (4, 4):
        return mat[:3, :3]
    if mat.shape == (3, 3):
        return mat
    raise ValueError(f"intrinsic must be 3x3 or 4x4, got {mat.shape}: {p}")


def _resolve_raw_rgb_path(scene_root: Path, frame_id: int) -> Path:
    raw_id = _raw_frame_id_for_view(scene_root, frame_id)
    p = scene_root / "raw" / f"{raw_id:06d}-rgb.png"
    if not p.exists():
        raise FileNotFoundError(f"Missing RGB: {p}")
    return p


def _raw_frame_id_for_view(scene_root: Path, frame_id: int) -> int:
    info_p = scene_root / "raw" / "scene_info.json"
    info = json.loads(info_p.read_text(encoding="utf-8"))
    kept = info.get("kept_frame_ids")
    if not isinstance(kept, list):
        raise ValueError(f"{info_p} missing kept_frame_ids")
    if frame_id < 0 or frame_id >= len(kept):
        raise ValueError(
            f"{scene_root.name} frame_id={frame_id} out of range ({len(kept)} kept)"
        )
    return int(kept[frame_id])


def load_mask3d_objects(scene_root: Path) -> list[dict[str, Any]]:
    p = scene_root / MASK3D_PCD_REL
    if not p.exists():
        raise FileNotFoundError(f"Missing Mask3D-CG pkl: {p}")
    with gzip.open(p, "rb") as f:
        return pickle.load(f)["objects"]


def load_mask3d_visibility_index(scene_root: Path) -> Mask3dVisibility:
    p = scene_root / VIS_REL
    if not p.exists():
        raise FileNotFoundError(f"Missing Mask3D visibility: {p}")
    with open(p, "rb") as f:
        payload = pickle.load(f)
    return Mask3dVisibility(
        object_to_views=_coerce_visibility(payload.get("object_to_views"), p),
        view_to_objects=_coerce_visibility(payload.get("view_to_objects"), p),
    )


def _coerce_visibility(raw: dict[Any, Any] | None,
                       path: Path) -> dict[int, list[tuple[int, float]]]:
    if not isinstance(raw, dict):
        raise ValueError(f"{path} missing visibility map")
    out: dict[int, list[tuple[int, float]]] = {}
    for k, entries in raw.items():
        out[int(k)] = [(int(e[0]), float(e[1])) for e in entries]
    return out


def load_sample_requests(path: Path) -> list[SampleRequest]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"sample ids JSON must be a list: {path}")
    out: list[SampleRequest] = []
    for i, row in enumerate(raw, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"row {i} must be object: {row!r}")
        sid = row.get("sample_id")
        if not isinstance(sid, str):
            raise ValueError(f"row {i} missing sample_id")
        scene, tid, ann_id = parse_sample_id(sid)
        out.append(SampleRequest(
            sample_id=sid,
            scene_id=row.get("scene_id") or scene,
            target_id=int(row.get("target_id", tid)),
            ann_id=str(row.get("ann_id", ann_id)),
            category=str(row.get("category") or ""),
        ))
    return out


def sample_artifact_path(data_root: Path, request: SampleRequest, *,
                         pack_name: str = "pack_scanrefer_v1") -> Path:
    return (data_root / request.scene_id / pack_name / "samples"
            / f"{safe_sample_id(request.sample_id)}.json")


def main() -> None:
    args = parse_args()
    written = prepare_pack_v1_inputs_scanrefer(
        sample_ids_path=args.sample_ids,
        data_root=args.data_root,
        pack_name=args.pack_name,
        split=args.split,
        scanrefer_root=args.scanrefer_root,
        phase8_data_root=args.phase8_data_root,
        raw_frames_root=args.raw_frames_root,
        max_samples=args.max_samples,
    )
    print(f"wrote {len(written)} sample artifacts under "
          f"{args.data_root}/<scene>/{args.pack_name}/")


if __name__ == "__main__":
    main()


__all__ = [
    "SampleRequest",
    "SceneArtifacts",
    "build_proposals_from_mask3d_objects",
    "load_sample_requests",
    "parse_sample_id",
    "prepare_pack_v1_inputs_scanrefer",
    "safe_sample_id",
]
```

- [ ] **Step 5.4: Run tests to confirm GREEN**

Run:
```bash
source .venv/bin/activate && PYTHONPATH=src pytest src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer.py -v 2>&1 | tail -10
```

Expected: 4 tests pass.

- [ ] **Step 5.5: Smoke pack-prep on 5 utts**

Generate a 5-utterance sample-ids JSON and run pack-prep:

```bash
source .venv/bin/activate && PYTHONPATH=src python -c "
import json
from benchmarks.scanrefer_loader import ScanRefVGDataset
ds = ScanRefVGDataset.from_path(
    data_root='data/scanrefer',
    phase8_data_root='data/nr3d/scannet',
    split='val',
)
ids = []
seen_scenes = set()
for s in ds:
    if s.scene_id in seen_scenes:
        continue
    ids.append({'sample_id': s.sample_id, 'scene_id': s.scene_id,
                'target_id': s.target_id, 'ann_id': s.ann_id, 'category': s.target})
    seen_scenes.add(s.scene_id)
    if len(ids) == 5: break
import os
os.makedirs('tmp/scanrefer_artifacts', exist_ok=True)
json.dump(ids, open('tmp/scanrefer_artifacts/smoke5_sample_ids.json', 'w'), indent=2)
print('wrote', len(ids), 'smoke ids')
"

PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py \
    --sample-ids tmp/scanrefer_artifacts/smoke5_sample_ids.json \
    --data-root data/scanrefer/scannet \
    --phase8-data-root data/nr3d/scannet \
    --scanrefer-root data/scanrefer \
    --raw-frames-root data/nr3d/scannet \
    --pack-name pack_scanrefer_v1
```

Expected: ~5 sample artifacts under `data/scanrefer/scannet/<scene>/pack_scanrefer_v1/samples/`.

Verify one sample's payload:

```bash
ls data/scanrefer/scannet/*/pack_scanrefer_v1/samples/*.json | head -1 | xargs cat | python -m json.tool | head -25
```

Should show `sample_id`, `scene_id`, `target_id`, `ann_id`, `is_unique`, `gt_bbox_3d_9dof` (9 floats).

- [ ] **Step 5.6: Commit**

```bash
git add src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py \
        src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer.py
git commit -m "feat(scanrefer): pack-prep with Mask3D pool + Phase 8 GT lookup

Mirrors prepare_pack_v1_inputs_nr3d.py with two pkl sources:
proposal pool from full_pcd_mask3d_axisaligned.pkl.gz (Task 1-3 producer
output), GT bbox lookup via ScanRefVGDataset (Task 4 loader, joins
Phase 8 GT-CG pkl). Keyframe selection uses Phase 8 visibility of the
target GT instance — agent sees frames where the actual referred
object is visible.

Pack output schema: {sample_id, scene_id, target_id, ann_id, category,
is_unique, query, gt_bbox_3d_9dof, keyframes, scene_artifacts_dir}.

Companion tests cover sample_id parsing/safe-encoding, sample-request
loading, and parse_sample_id rejection of malformed input."
```

---

## Task 6: Runner mirror + 20-utt smoke

**Files:**
- Create: `src/evaluation/scripts/run_scanrefer_vg_side_by_side.py`
- Create: `src/evaluation/scripts/tests/test_run_scanrefer_vg_side_by_side.py`

The runner is a near-verbatim mirror of `run_nr3d_vg_side_by_side.py`. Only sample-id parsing and pack-name defaults change.

- [ ] **Step 6.1: Read NR3D runner as reference**

Open `src/evaluation/scripts/run_nr3d_vg_side_by_side.py`. Note these lines / functions that need to change for ScanRefer:

| Function | NR3D | ScanRefer |
|---|---|---|
| `parse_nr3d_sample_id` | parses `<scene>::<target>::<assignment>` | rename to `parse_scanrefer_sample_id`, parses `<scene>::<target>::<ann_id>` |
| `pack_name` default | `pack_nr3d_v1` | `pack_scanrefer_v1` |
| Default `--data-root` help text | references NR3D | references ScanRefer |
| All other logic | unchanged | unchanged |

- [ ] **Step 6.2: Write runner test scaffolding**

Create `src/evaluation/scripts/tests/test_run_scanrefer_vg_side_by_side.py`:

```python
"""Smoke tests for ScanRefer side-by-side runner."""

from __future__ import annotations

import pytest


def test_parse_sample_id_canonical():
    from evaluation.scripts.run_scanrefer_vg_side_by_side import parse_scanrefer_sample_id
    parsed = parse_scanrefer_sample_id("scannet/scene0088_00::5::3")
    assert parsed.scan_id == "scannet/scene0088_00"
    assert parsed.scene_id == "scene0088_00"
    assert parsed.target_id == 5
    assert parsed.ann_id == "3"


def test_parse_sample_id_rejects_malformed():
    from evaluation.scripts.run_scanrefer_vg_side_by_side import parse_scanrefer_sample_id
    with pytest.raises(ValueError, match="format"):
        parse_scanrefer_sample_id("scannet/scene_x::5")


def test_safe_sample_id():
    from evaluation.scripts.run_scanrefer_vg_side_by_side import safe_sample_id
    assert safe_sample_id("scannet/scene_a::5::3") == "scannet__scene_a__5__3"


def test_compare_backends_persists_failed_sentinel_on_sample_exception(tmp_path, monkeypatch):
    """Mirror of NR3D test: a per-sample exception → failed sentinel, not crash."""
    from evaluation.scripts import run_scanrefer_vg_side_by_side as mod

    def fake_run_one_sample(sample_id, backend, **kwargs):
        if sample_id.endswith("::3"):
            raise RuntimeError("simulated upstream 500")
        return {"sample_id": sample_id, "backend": backend, "status": "completed",
                "iou": 1.0, "predicted_bbox_3d_9dof": [0]*9,
                "gt_bbox_3d_9dof": [0]*9, "selected_object_id": 0,
                "confidence": 0.9, "query": "test"}

    monkeypatch.setattr(mod, "run_one_sample", fake_run_one_sample)
    monkeypatch.setattr(mod, "validate_unique_sample_ids", lambda _: None)
    monkeypatch.setattr(mod, "preflight_pack_sample_exists", lambda *a, **kw: None)

    out = tmp_path / "out"
    sample_ids = ["scannet/scene_a::1::0", "scannet/scene_a::2::3"]
    results = mod.compare_backends(
        sample_ids=sample_ids, output_dir=out, data_root=tmp_path,
        pack_name="pack_scanrefer_v1", workers=1,
    )
    per_sample = results["pack_v1"]["per_sample"]
    assert len(per_sample) == 2
    failed = [r for r in per_sample if r["sample_id"].endswith("::3")][0]
    assert failed["status"] == "failed"
    assert failed["iou"] == 0.0
    assert "simulated upstream 500" in failed["error"]
```

- [ ] **Step 6.3: Run tests to confirm RED**

Run:
```bash
source .venv/bin/activate && PYTHONPATH=src pytest src/evaluation/scripts/tests/test_run_scanrefer_vg_side_by_side.py -v 2>&1 | tail -10
```

Expected: ImportError on the runner module.

- [ ] **Step 6.4: Implement the runner**

Copy `src/evaluation/scripts/run_nr3d_vg_side_by_side.py` to `src/evaluation/scripts/run_scanrefer_vg_side_by_side.py`, then make the following minimal edits:

1. Rename class:

```python
@dataclass(frozen=True)
class ParsedScanRefSampleId:
    scan_id: str
    scene_id: str
    target_id: int
    ann_id: str
```

2. Rename function — replace ALL occurrences of `parse_nr3d_sample_id` with `parse_scanrefer_sample_id` and update body:

```python
def parse_scanrefer_sample_id(sample_id: str) -> ParsedScanRefSampleId:
    if not isinstance(sample_id, str):
        raise TypeError(
            "Expected sample_id in '<scene>::<target_id>::<ann_id>' "
            f"string format, got {type(sample_id).__name__}: {sample_id!r}"
        )
    parts = sample_id.split("::")
    if len(parts) != 3:
        raise ValueError(
            "Expected sample_id in '<scene>::<target_id>::<ann_id>' "
            f"format, got {sample_id!r}"
        )
    scan_id, target_text, ann_id = parts
    if not scan_id or not ann_id:
        raise ValueError(f"Invalid ScanRefer sample_id={sample_id!r}")
    try:
        target_id = int(target_text)
    except ValueError as exc:
        raise ValueError(f"Invalid target_id in sample_id={sample_id!r}") from exc
    return ParsedScanRefSampleId(
        scan_id=scan_id,
        scene_id=scan_id.split("/")[-1],
        target_id=target_id,
        ann_id=ann_id,
    )
```

3. Update default pack_name throughout (15 occurrences in the NR3D version):

```python
pack_name: str = "pack_scanrefer_v1"
```

4. Update CLI default:

```python
parser.add_argument("--pack-name", default="pack_scanrefer_v1")
parser.add_argument(
    "--data-root", required=True, type=Path,
    help="ScanRefer scannet root, e.g. data/scanrefer/scannet"
)
```

5. The bundle builder import remains the same:

```python
from agents.examples.embodiedscan_vg_pack_v1_pilot import (
    build_pack_v1_bundle as bundle_builder,
)
```

(no change — pack_v1 builder is benchmark-agnostic).

- [ ] **Step 6.5: Run tests to confirm GREEN**

Run:
```bash
source .venv/bin/activate && PYTHONPATH=src pytest src/evaluation/scripts/tests/test_run_scanrefer_vg_side_by_side.py -v 2>&1 | tail -10
```

Expected: 4 tests pass.

- [ ] **Step 6.6: 20-utt smoke run**

Generate a 20-utt sample-ids JSON, prep packs, then run:

```bash
source .venv/bin/activate && PYTHONPATH=src python -c "
import json
from benchmarks.scanrefer_loader import ScanRefVGDataset
ds = ScanRefVGDataset.from_path(
    data_root='data/scanrefer',
    phase8_data_root='data/nr3d/scannet',
    split='val',
)
seen_scenes = []
out = []
for s in ds:
    if s.scene_id not in seen_scenes:
        seen_scenes.append(s.scene_id)
        if len(seen_scenes) > 4: break
    if len([r for r in out if r['scene_id']==s.scene_id]) < 5:
        out.append({'sample_id': s.sample_id, 'scene_id': s.scene_id,
                    'target_id': s.target_id, 'ann_id': s.ann_id,
                    'category': s.target})
import os
os.makedirs('tmp/scanrefer_artifacts', exist_ok=True)
json.dump(out, open('tmp/scanrefer_artifacts/smoke20_sample_ids.json', 'w'), indent=2)
print('wrote', len(out), 'smoke ids on', len(seen_scenes), 'scenes')
"

# Pack-prep for 20 utts
PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py \
    --sample-ids tmp/scanrefer_artifacts/smoke20_sample_ids.json \
    --data-root data/scanrefer/scannet --pack-name pack_scanrefer_v1

# Runner smoke (workers=1, sample-retries=0 → fail-fast for diagnostics)
tmux new-session -d -s scanrefer-smoke20 "source .venv/bin/activate && PYTHONPATH=src python \
    src/evaluation/scripts/run_scanrefer_vg_side_by_side.py \
    --sample-ids tmp/scanrefer_artifacts/smoke20_sample_ids.json \
    --data-root data/scanrefer/scannet \
    --pack-name pack_scanrefer_v1 \
    --output-dir tmp/scanrefer_eval_smoke20 \
    --workers 1 --sample-retries 0 2>&1 | tee /tmp/scanrefer_smoke20.log"

# Wait for completion
until ! tmux has-session -t scanrefer-smoke20 2>/dev/null; do sleep 30; done

# Inspect output
ls tmp/scanrefer_eval_smoke20/per_sample/pack_scanrefer_v1/ | wc -l   # → 20
python -c "
import json
d = json.load(open('tmp/scanrefer_eval_smoke20/side_by_side.json'))
m = d['pack_v1']
print(f'n={m[\"n\"]} mean_iou={m[\"mean_iou\"]:.4f} acc25={m[\"Acc@0.25\"]:.4f} acc50={m[\"Acc@0.50\"]:.4f}')
print(f'failed: {sum(1 for s in m[\"per_sample\"] if s[\"status\"]==\"failed\")}/20')
"
```

Expected: 20 per-sample checkpoints, mean_iou > 0, ≤2 failures, complete in ~20-30 min.

- [ ] **Step 6.7: Commit**

```bash
git add src/evaluation/scripts/run_scanrefer_vg_side_by_side.py \
        src/evaluation/scripts/tests/test_run_scanrefer_vg_side_by_side.py
git commit -m "feat(scanrefer): runner mirror + 20-utt smoke

Mirrors run_nr3d_vg_side_by_side.py with new sample-id parser
(<scene>::<target_id>::<ann_id>) and pack_scanrefer_v1 defaults.
Same Stage2DeepResearchAgent + same failed-sentinel wrapper.

Smoke run on 20 utts × 4 scenes verifies pipeline end-to-end."
```

---

## Task 7: Aggregator (RED → GREEN)

**Files:**
- Create: `src/evaluation/scripts/scanrefer_leaderboard_metrics.py`
- Create: `src/evaluation/scripts/tests/test_scanrefer_leaderboard_metrics.py`

- [ ] **Step 7.1: Write the failing test file**

Create `src/evaluation/scripts/tests/test_scanrefer_leaderboard_metrics.py`:

```python
"""Tests for ScanRefer leaderboard metrics aggregator."""

from __future__ import annotations

import pytest

from evaluation.scripts.scanrefer_leaderboard_metrics import aggregate


def _record(sid: str, iou: float, is_unique: bool, status: str = "completed",
            target_id: int = 0, selected_object_id: int = 0) -> dict:
    return {
        "sample_id": sid,
        "iou": iou,
        "status": status,
        "is_unique": is_unique,
        "target_id": target_id,
        "selected_object_id": selected_object_id,
    }


def _meta(sid: str, is_unique: bool, target_id: int = 0) -> dict:
    return {"sample_id": sid, "is_unique": is_unique, "target_id": target_id, "target": "chair"}


def test_aggregate_all_correct():
    preds = [_record("a", 0.9, True), _record("b", 0.9, False), _record("c", 1.0, True)]
    meta = [_meta("a", True), _meta("b", False), _meta("c", True)]
    m = aggregate(preds, meta)
    assert m["n_total"] == 3
    assert m["acc25_overall"] == 1.0
    assert m["acc50_overall"] == 1.0


def test_aggregate_all_wrong():
    preds = [_record("a", 0.0, True), _record("b", 0.0, False)]
    meta = [_meta("a", True), _meta("b", False)]
    m = aggregate(preds, meta)
    assert m["acc25_overall"] == 0.0
    assert m["acc50_overall"] == 0.0


def test_aggregate_iou_threshold_boundaries():
    """IoU exactly at 0.25 or 0.50 should count as correct (≥ threshold)."""
    preds = [_record("a", 0.25, True), _record("b", 0.50, False), _record("c", 0.49, True)]
    meta = [_meta("a", True), _meta("b", False), _meta("c", True)]
    m = aggregate(preds, meta)
    # acc25: a (0.25 >= 0.25) + b (0.5 >= 0.25) + c (0.49 >= 0.25) → 3/3
    assert m["acc25_overall"] == 1.0
    # acc50: only b (0.5 >= 0.50) → 1/3
    assert m["acc50_overall"] == pytest.approx(1/3, abs=1e-6)


def test_aggregate_unique_multiple_partition():
    preds = [
        _record("a", 1.0, True), _record("b", 1.0, True),     # unique, both correct
        _record("c", 0.0, False), _record("d", 1.0, False),   # multiple, 1 of 2 correct
    ]
    meta = [_meta("a", True), _meta("b", True), _meta("c", False), _meta("d", False)]
    m = aggregate(preds, meta)
    assert m["n_unique"] == 2
    assert m["n_multiple"] == 2
    assert m["acc25_unique"] == 1.0
    assert m["acc50_unique"] == 1.0
    assert m["acc25_multiple"] == 0.5
    assert m["acc50_multiple"] == 0.5


def test_aggregate_failed_sentinel_counts_as_zero():
    preds = [_record("a", 0.0, True, status="failed"), _record("b", 1.0, False)]
    meta = [_meta("a", True), _meta("b", False)]
    m = aggregate(preds, meta)
    assert m["acc25_overall"] == 0.5
    assert m["acc25_unique"] == 0.0
    assert m["acc25_multiple"] == 1.0


def test_aggregate_raises_on_missing_sample_id():
    """Every meta entry must have a matching prediction; otherwise raise."""
    preds = [_record("a", 1.0, True)]
    meta = [_meta("a", True), _meta("b", True)]
    with pytest.raises(ValueError, match="missing"):
        aggregate(preds, meta)


def test_aggregate_invariants():
    """n_unique + n_multiple == n_total."""
    preds = [_record("a", 1.0, True), _record("b", 0.0, False), _record("c", 1.0, True)]
    meta = [_meta("a", True), _meta("b", False), _meta("c", True)]
    m = aggregate(preds, meta)
    assert m["n_unique"] + m["n_multiple"] == m["n_total"] == 3
    assert len(m["per_sample"]) == 3


def test_aggregate_per_sample_carries_flags():
    preds = [_record("a", 0.6, True)]
    meta = [_meta("a", True)]
    m = aggregate(preds, meta)
    s = m["per_sample"][0]
    assert s["sample_id"] == "a"
    assert s["is_unique"] is True
    assert s["acc25"] == 1
    assert s["acc50"] == 1
```

- [ ] **Step 7.2: Run tests to confirm RED**

Run:
```bash
source .venv/bin/activate && PYTHONPATH=src pytest src/evaluation/scripts/tests/test_scanrefer_leaderboard_metrics.py -v 2>&1 | tail -10
```

Expected: ImportError.

- [ ] **Step 7.3: Implement the aggregator**

Create `src/evaluation/scripts/scanrefer_leaderboard_metrics.py`:

```python
"""ScanRefer leaderboard-track post-aggregator.

Reads side_by_side.json from run_scanrefer_vg_side_by_side.py + sample
metadata from ScanRefVGDataset, emits canonical ScanRefer metrics:
Acc@0.25 / Acc@0.50 × { Unique, Multiple, Overall }.

Source citations:
- ScanRefer paper Table 1 — canonical Unique/Multiple slicing
- ZSVG3D visprog_scanrefer.py:51 — `unique = (...class_ids == target_class_id).sum() == 1`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def aggregate(
    predictions: list[dict[str, Any]],
    sample_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute Unique/Multiple/Overall × @0.25/0.50 from predictions.

    Args:
        predictions: per-sample records from side_by_side.json.
            Each must have ``sample_id``, ``iou``, ``status``.
            ``iou`` may be None for failed sentinels (counted as 0).
        sample_meta: per-sample metadata records from ScanRefVGDataset.
            Each must have ``sample_id``, ``is_unique``.

    Returns:
        Metrics dict with the leaderboard columns and per_sample list.

    Raises:
        ValueError: if any meta sample_id is missing from predictions.
    """
    pred_by_sid = {p["sample_id"]: p for p in predictions}
    meta_sids = {m["sample_id"] for m in sample_meta}
    missing = meta_sids - set(pred_by_sid.keys())
    if missing:
        raise ValueError(
            f"side_by_side missing {len(missing)} sample_ids; "
            f"first 5: {sorted(missing)[:5]}"
        )

    samples: list[dict[str, Any]] = []
    for meta in sample_meta:
        sid = meta["sample_id"]
        rec = pred_by_sid[sid]
        iou_raw = rec.get("iou")
        iou = float(iou_raw) if iou_raw is not None else 0.0
        if str(rec.get("status", "")).lower() == "failed":
            iou = 0.0
        is_unique = bool(meta.get("is_unique", False))
        samples.append({
            "sample_id": sid,
            "iou": iou,
            "is_unique": is_unique,
            "acc25": int(iou >= 0.25),
            "acc50": int(iou >= 0.50),
            "target_id": meta.get("target_id"),
            "target": meta.get("target"),
            "status": rec.get("status"),
        })

    n_total = len(samples)
    unique = [s for s in samples if s["is_unique"]]
    multi = [s for s in samples if not s["is_unique"]]

    def _mean(xs: list[int]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    metrics = {
        "n_total": n_total,
        "n_unique": len(unique),
        "n_multiple": len(multi),
        "acc25_overall": _mean([s["acc25"] for s in samples]),
        "acc50_overall": _mean([s["acc50"] for s in samples]),
        "acc25_unique": _mean([s["acc25"] for s in unique]),
        "acc50_unique": _mean([s["acc50"] for s in unique]),
        "acc25_multiple": _mean([s["acc25"] for s in multi]),
        "acc50_multiple": _mean([s["acc50"] for s in multi]),
        "mean_iou_overall": _mean([s["iou"] for s in samples]) if samples else 0.0,
        "per_sample": samples,
    }
    if metrics["n_unique"] + metrics["n_multiple"] != metrics["n_total"]:
        raise AssertionError(
            f"n_unique + n_multiple != n_total: {metrics['n_unique']} + "
            f"{metrics['n_multiple']} != {metrics['n_total']}"
        )
    return metrics


def compute_leaderboard_metrics(
    side_by_side_path: Path,
    scanrefer_data_root: Path,
    phase8_data_root: Path,
    backend: str = "pack_v1",
) -> dict[str, Any]:
    """IO wrapper: load side_by_side + ScanRefer loader, call aggregate."""
    from benchmarks.scanrefer_loader import ScanRefVGDataset

    payload = json.loads(side_by_side_path.read_text(encoding="utf-8"))
    if backend not in payload:
        raise ValueError(f"side_by_side.json missing backend={backend!r}")
    per_sample = payload[backend].get("per_sample") or []

    ds = ScanRefVGDataset.from_path(
        data_root=str(scanrefer_data_root),
        phase8_data_root=str(phase8_data_root),
        split="val",
    )
    sample_meta = [
        {
            "sample_id": s.sample_id,
            "is_unique": s.is_unique,
            "target_id": s.target_id,
            "target": s.target,
        }
        for s in ds
    ]
    return aggregate(per_sample, sample_meta)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side-by-side", required=True, type=Path)
    p.add_argument("--scanrefer-data-root", default=Path("data/scanrefer"), type=Path)
    p.add_argument("--phase8-data-root", default=Path("data/nr3d/scannet"), type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--backend", default="pack_v1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = compute_leaderboard_metrics(
        side_by_side_path=args.side_by_side,
        scanrefer_data_root=args.scanrefer_data_root,
        phase8_data_root=args.phase8_data_root,
        backend=args.backend,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"n_total={metrics['n_total']} (Unique={metrics['n_unique']}, "
          f"Multiple={metrics['n_multiple']})")
    print(f"acc25  overall={metrics['acc25_overall']:.4f} "
          f"unique={metrics['acc25_unique']:.4f} "
          f"multiple={metrics['acc25_multiple']:.4f}")
    print(f"acc50  overall={metrics['acc50_overall']:.4f} "
          f"unique={metrics['acc50_unique']:.4f} "
          f"multiple={metrics['acc50_multiple']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.4: Run tests to confirm GREEN**

Run:
```bash
source .venv/bin/activate && PYTHONPATH=src pytest src/evaluation/scripts/tests/test_scanrefer_leaderboard_metrics.py -v 2>&1 | tail -15
```

Expected: 8 tests pass.

- [ ] **Step 7.5: Commit**

```bash
git add src/evaluation/scripts/scanrefer_leaderboard_metrics.py \
        src/evaluation/scripts/tests/test_scanrefer_leaderboard_metrics.py
git commit -m "feat(scanrefer): leaderboard aggregator with Unique/Multiple slicing

Pure aggregator + IO wrapper + CLI. Computes Acc@0.25 / Acc@0.50 ×
{Unique, Multiple, Overall} from side_by_side.json + ScanRefVGDataset
metadata. Failed sentinels count as iou=0 (consistent with NR3D v3).
Inner-join completeness asserted (raises on missing sample_id)."
```

---

## Task 8: Ingester (RED → GREEN)

**Files:**
- Create: `scripts/ingest_scanrefer_run.py`
- Create: `src/evaluation/scripts/tests/test_ingest_scanrefer_run.py`

- [ ] **Step 8.1: Write the failing test file**

Create `src/evaluation/scripts/tests/test_ingest_scanrefer_run.py`:

```python
"""Tests for ingest_scanrefer_run.py."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


def _load_ingester():
    script = Path(__file__).resolve().parents[4] / "scripts" / "ingest_scanrefer_run.py"
    spec = importlib.util.spec_from_file_location("ingest_scanrefer_run", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ingest


def _write_outputs(output_dir: Path, side_by_side: dict, leaderboard: dict | None = None) -> None:
    output_dir.mkdir()
    (output_dir / "side_by_side.json").write_text(json.dumps(side_by_side), encoding="utf-8")
    if leaderboard is not None:
        (output_dir / "leaderboard_metrics.json").write_text(
            json.dumps(leaderboard), encoding="utf-8"
        )


def test_runs_table_has_scanrefer_columns(tmp_path: Path) -> None:
    ingest = _load_ingester()
    output_dir = tmp_path / "run"
    _write_outputs(output_dir, {
        "pack_v1": {
            "n": 1, "mean_iou": 0.5, "Acc@0.25": 0.5, "Acc@0.50": 0.5,
            "per_sample": [{
                "sample_id": "scannet/scene_a::5::3", "status": "completed",
                "iou": 0.5, "selected_object_id": 5, "confidence": 0.9,
                "query": "test", "predicted_bbox_3d_9dof": [0]*9,
                "gt_bbox_3d_9dof": [0]*9,
            }],
        },
    }, leaderboard={
        "n_total": 1, "n_unique": 1, "n_multiple": 0,
        "acc25_overall": 1.0, "acc50_overall": 1.0,
        "acc25_unique": 1.0, "acc50_unique": 1.0,
        "acc25_multiple": 0.0, "acc50_multiple": 0.0,
        "mean_iou_overall": 0.5,
        "per_sample": [{
            "sample_id": "scannet/scene_a::5::3", "iou": 0.5,
            "is_unique": True, "acc25": 1, "acc50": 1, "target_id": 5,
            "target": "chair", "status": "completed",
        }],
    })
    db = tmp_path / "runs.sqlite"
    ingest(
        db_path=db, output_dir=output_dir, run_id="test_v1",
        branch="feat/test", commit_hash="abc1234",
        leaderboard_metrics_path=output_dir / "leaderboard_metrics.json",
    )
    conn = sqlite3.connect(str(db))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        expected = {"run_id", "n_total", "n_unique", "n_multiple",
                    "acc25_overall", "acc50_overall",
                    "acc25_unique", "acc50_unique",
                    "acc25_multiple", "acc50_multiple",
                    "mean_iou_overall"}
        assert expected <= cols, sorted(expected - cols)
        sample_cols = {r[1] for r in conn.execute("PRAGMA table_info(samples)")}
        expected_sample = {"sample_id", "scene_id", "target_id", "ann_id",
                           "iou", "acc25", "acc50", "is_unique"}
        assert expected_sample <= sample_cols, sorted(expected_sample - sample_cols)
    finally:
        conn.close()


def test_ingest_populates_per_sample_with_unique(tmp_path: Path) -> None:
    ingest = _load_ingester()
    output_dir = tmp_path / "run"
    _write_outputs(output_dir, {
        "pack_v1": {
            "n": 1, "mean_iou": 1.0, "Acc@0.25": 1.0, "Acc@0.50": 1.0,
            "per_sample": [{
                "sample_id": "scannet/scene_a::5::3", "status": "completed",
                "iou": 1.0, "selected_object_id": 5, "confidence": 0.9,
                "query": "the chair", "predicted_bbox_3d_9dof": [0]*9,
                "gt_bbox_3d_9dof": [0]*9,
            }],
        },
    }, leaderboard={
        "n_total": 1, "n_unique": 1, "n_multiple": 0,
        "acc25_overall": 1.0, "acc50_overall": 1.0,
        "acc25_unique": 1.0, "acc50_unique": 1.0,
        "acc25_multiple": 0.0, "acc50_multiple": 0.0,
        "mean_iou_overall": 1.0,
        "per_sample": [{
            "sample_id": "scannet/scene_a::5::3", "iou": 1.0,
            "is_unique": True, "acc25": 1, "acc50": 1, "target_id": 5,
            "target": "chair", "status": "completed",
        }],
    })
    db = tmp_path / "runs.sqlite"
    ingest(
        db_path=db, output_dir=output_dir, run_id="test_v1",
        branch="feat/test", commit_hash="abc1234",
        leaderboard_metrics_path=output_dir / "leaderboard_metrics.json",
    )
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT acc25_overall, acc50_overall, acc25_unique, acc50_unique, "
            "acc25_multiple, acc50_multiple, n_total, n_unique, n_multiple "
            "FROM runs WHERE run_id=?", ("test_v1",)
        ).fetchone()
        assert row == (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1, 1, 0)
        sample_row = conn.execute(
            "SELECT acc25, acc50, is_unique FROM samples "
            "WHERE run_id=? AND sample_id=?",
            ("test_v1", "scannet/scene_a::5::3")
        ).fetchone()
        assert sample_row == (1, 1, 1)
    finally:
        conn.close()


def test_ingest_requires_per_sample(tmp_path: Path) -> None:
    ingest = _load_ingester()
    output_dir = tmp_path / "run"
    _write_outputs(output_dir, {
        "pack_v1": {"n": 0, "mean_iou": 0, "Acc@0.25": 0, "Acc@0.50": 0},
    })
    with pytest.raises(ValueError, match="per_sample"):
        ingest(
            db_path=tmp_path / "runs.sqlite", output_dir=output_dir,
            run_id="x", branch=None, commit_hash=None,
        )
```

- [ ] **Step 8.2: Run tests to confirm RED**

Run:
```bash
source .venv/bin/activate && PYTHONPATH=src pytest src/evaluation/scripts/tests/test_ingest_scanrefer_run.py -v 2>&1 | tail -10
```

Expected: 3 tests fail (script doesn't exist).

- [ ] **Step 8.3: Implement the ingester**

Create `scripts/ingest_scanrefer_run.py`:

```python
"""Ingest a ScanRefer VG side-by-side run into a SQLite database.

Mirrors scripts/ingest_nr3d_run.py with Unique/Multiple slicing columns
instead of NR3D's Easy/Hard/View-Dep/View-Indep.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    branch          TEXT,
    commit_hash     TEXT,
    output_dir      TEXT NOT NULL,
    backend         TEXT NOT NULL,
    n_total         INTEGER,
    n_unique        INTEGER,
    n_multiple      INTEGER,
    acc25_overall   REAL,
    acc50_overall   REAL,
    acc25_unique    REAL,
    acc50_unique    REAL,
    acc25_multiple  REAL,
    acc50_multiple  REAL,
    mean_iou_overall REAL,
    judge_model     TEXT,
    started_at      REAL,
    ingested_at     REAL NOT NULL,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS samples (
    run_id                  TEXT NOT NULL,
    sample_id               TEXT NOT NULL,
    scene_id                TEXT,
    target_id               INTEGER,
    ann_id                  TEXT,
    description             TEXT,
    status                  TEXT,
    selected_object_id      INTEGER,
    confidence              REAL,
    iou                     REAL,
    acc25                   INTEGER,
    acc50                   INTEGER,
    is_unique               INTEGER,
    predicted_bbox_3d_9dof  TEXT,
    gt_bbox_3d_9dof         TEXT,
    PRIMARY KEY (run_id, sample_id)
);

CREATE INDEX IF NOT EXISTS idx_scanrefer_samples_run ON samples(run_id);
CREATE INDEX IF NOT EXISTS idx_scanrefer_samples_scene ON samples(scene_id);
CREATE INDEX IF NOT EXISTS idx_scanrefer_samples_unique ON samples(is_unique);

CREATE TABLE IF NOT EXISTS tool_calls (
    run_id          TEXT NOT NULL,
    question_id     TEXT NOT NULL,
    turn_idx        INTEGER NOT NULL,
    tool_name       TEXT NOT NULL,
    tool_input      TEXT,
    response_text   TEXT
);
CREATE INDEX IF NOT EXISTS idx_scanrefer_tools_run_qid
    ON tool_calls(run_id, question_id);

CREATE TABLE IF NOT EXISTS llm_calls (
    run_id              TEXT NOT NULL,
    ts                  REAL,
    model               TEXT,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    cached_tokens       INTEGER,
    session_id          TEXT,
    question_id         TEXT
);
CREATE INDEX IF NOT EXISTS idx_scanrefer_llm_run ON llm_calls(run_id);
"""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_sample_id(sample_id: str) -> tuple[str, int, str]:
    parts = sample_id.split("::")
    if len(parts) < 3:
        raise ValueError(
            f"sample_id {sample_id!r} malformed (need <scan_id>::<target_id>::<ann_id>)"
        )
    try:
        target_id = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"sample_id {sample_id!r} target_id segment not int") from exc
    return parts[0], target_id, parts[2]


def _json_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def ingest(
    *, db_path: Path, output_dir: Path, run_id: str,
    branch: str | None, commit_hash: str | None,
    backend: str = "pack_v1",
    judge_model: str | None = None, notes: str | None = None,
    leaderboard_metrics_path: Path | None = None,
) -> None:
    side_by_side_path = output_dir / "side_by_side.json"
    if not side_by_side_path.exists():
        raise FileNotFoundError(f"Missing side_by_side.json: {side_by_side_path}")
    payload = _read_json(side_by_side_path)
    metrics = payload.get(backend)
    if not isinstance(metrics, dict):
        raise ValueError(f"side_by_side.json missing backend={backend!r}")
    per_sample = metrics.get("per_sample")
    if not isinstance(per_sample, list) or not per_sample:
        raise ValueError(
            f"side_by_side.json[{backend}].per_sample must be a non-empty list"
        )

    leaderboard: dict[str, Any] | None = None
    leaderboard_per_sample: dict[str, dict[str, Any]] = {}
    if leaderboard_metrics_path is not None:
        leaderboard = _read_json(Path(leaderboard_metrics_path))
        for entry in leaderboard.get("per_sample", []):
            sid = entry.get("sample_id")
            if isinstance(sid, str):
                leaderboard_per_sample[sid] = entry

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA)
        cur = conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO runs (
                run_id, branch, commit_hash, output_dir, backend,
                n_total, n_unique, n_multiple,
                acc25_overall, acc50_overall,
                acc25_unique, acc50_unique,
                acc25_multiple, acc50_multiple,
                mean_iou_overall,
                judge_model, started_at, ingested_at, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, branch, commit_hash, str(output_dir), backend,
                leaderboard.get("n_total") if leaderboard else None,
                leaderboard.get("n_unique") if leaderboard else None,
                leaderboard.get("n_multiple") if leaderboard else None,
                leaderboard.get("acc25_overall") if leaderboard else None,
                leaderboard.get("acc50_overall") if leaderboard else None,
                leaderboard.get("acc25_unique") if leaderboard else None,
                leaderboard.get("acc50_unique") if leaderboard else None,
                leaderboard.get("acc25_multiple") if leaderboard else None,
                leaderboard.get("acc50_multiple") if leaderboard else None,
                leaderboard.get("mean_iou_overall") if leaderboard else None,
                judge_model, None, time.time(), notes,
            ),
        )
        cur.execute("DELETE FROM samples WHERE run_id=?", (run_id,))
        cur.execute("DELETE FROM tool_calls WHERE run_id=?", (run_id,))
        cur.execute("DELETE FROM llm_calls WHERE run_id=?", (run_id,))

        for item in per_sample:
            sample_id = str(item.get("sample_id") or "")
            if not sample_id:
                raise ValueError(f"per_sample missing sample_id: {item!r}")
            scene_id, target_id, ann_id = _parse_sample_id(sample_id)
            iou_raw = item.get("iou")
            iou = float(iou_raw) if iou_raw is not None else None
            extra = leaderboard_per_sample.get(sample_id)
            cur.execute(
                """INSERT INTO samples (
                    run_id, sample_id, scene_id, target_id, ann_id,
                    description, status, selected_object_id, confidence,
                    iou, acc25, acc50, is_unique,
                    predicted_bbox_3d_9dof, gt_bbox_3d_9dof
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, sample_id, scene_id, target_id, ann_id,
                    item.get("query"),
                    item.get("status"),
                    item.get("selected_object_id"),
                    item.get("confidence"),
                    iou,
                    int(iou is not None and iou >= 0.25),
                    int(iou is not None and iou >= 0.50),
                    int(extra["is_unique"]) if extra and extra.get("is_unique") is not None else None,
                    _json_or_none(item.get("predicted_bbox_3d_9dof")),
                    _json_or_none(item.get("gt_bbox_3d_9dof")),
                ),
            )
        conn.commit()
    finally:
        conn.close()
    print(f"[ingest] run_id={run_id} samples={len(per_sample)} db={db_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--run-id", required=True)
    p.add_argument("--branch", default=None)
    p.add_argument("--commit", default=None, dest="commit_hash")
    p.add_argument("--backend", default="pack_v1")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--db", default=Path("docs/benchmark/scanrefer/runs.sqlite"), type=Path)
    p.add_argument("--notes", default=None)
    p.add_argument("--leaderboard-metrics", default=None, type=Path,
                   help="optional path to leaderboard_metrics.json")
    args = p.parse_args()
    ingest(
        db_path=args.db, output_dir=args.output_dir, run_id=args.run_id,
        branch=args.branch, commit_hash=args.commit_hash, backend=args.backend,
        judge_model=args.judge_model, notes=args.notes,
        leaderboard_metrics_path=args.leaderboard_metrics,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.4: Run tests to confirm GREEN**

Run:
```bash
source .venv/bin/activate && PYTHONPATH=src pytest src/evaluation/scripts/tests/test_ingest_scanrefer_run.py -v 2>&1 | tail -10
```

Expected: 3 tests pass.

- [ ] **Step 8.5: Commit**

```bash
git add scripts/ingest_scanrefer_run.py \
        src/evaluation/scripts/tests/test_ingest_scanrefer_run.py
git commit -m "feat(scanrefer): SQLite ingester with Unique/Multiple slicing

scripts/ingest_scanrefer_run.py mirrors NR3D ingester with ScanRefer
schema: runs table has n_total/n_unique/n_multiple +
acc25/acc50_overall/unique/multiple + mean_iou_overall; samples table
has is_unique boolean + ann_id text. Optional --leaderboard-metrics
flag populates the slicing columns from leaderboard_metrics.json.

Companion tests cover schema columns, full ingest with leaderboard,
and per_sample missing rejection."
```

---

## Task 9: Full agent run on 9508 utts (~16h background)

**Files:** none modified (long-running execution).

- [ ] **Step 9.1: Generate full sample-ids JSON**

```bash
source .venv/bin/activate && PYTHONPATH=src python -c "
import json
from pathlib import Path
from benchmarks.scanrefer_loader import ScanRefVGDataset
ds = ScanRefVGDataset.from_path(
    data_root='data/scanrefer',
    phase8_data_root='data/nr3d/scannet',
    split='val',
)
out = []
for s in ds:
    out.append({
        'sample_id': s.sample_id,
        'scene_id': s.scene_id,
        'target_id': s.target_id,
        'ann_id': s.ann_id,
        'category': s.target,
    })
import os
os.makedirs('tmp/scanrefer_artifacts', exist_ok=True)
json.dump(out, open('tmp/scanrefer_artifacts/full_val_sample_ids.json', 'w'), indent=2)
print(f'wrote {len(out)} sample ids')
"
```

Expected: `wrote 9508 sample ids`.

- [ ] **Step 9.2: Run pack-prep on all 141 scenes**

```bash
source .venv/bin/activate
PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py \
    --sample-ids tmp/scanrefer_artifacts/full_val_sample_ids.json \
    --data-root data/scanrefer/scannet \
    --phase8-data-root data/nr3d/scannet \
    --scanrefer-root data/scanrefer \
    --raw-frames-root data/nr3d/scannet \
    --pack-name pack_scanrefer_v1 \
    2>&1 | tee /tmp/scanrefer_full_prep.log
```

Expected: ~141 scenes prepared, ~9508 sample artifacts. ~30 min wall.

- [ ] **Step 9.3: Launch full agent run in tmux**

```bash
tmux new-session -d -s scanrefer-full \
  "source .venv/bin/activate && PYTHONPATH=src python \
    src/evaluation/scripts/run_scanrefer_vg_side_by_side.py \
    --sample-ids tmp/scanrefer_artifacts/full_val_sample_ids.json \
    --data-root data/scanrefer/scannet \
    --pack-name pack_scanrefer_v1 \
    --output-dir tmp/scanrefer_eval_v1_full \
    --workers 32 --sample-retries 1 2>&1 | tee /tmp/scanrefer_full.log"
echo "Launched. Monitor with: tmux capture-pane -t scanrefer-full -p -S -20"
```

Expected wall: ~16h (mirrors NR3D v2 16h50m on 8584 utts; we have 9508).

- [ ] **Step 9.4: Periodic progress check (every ~30 min)**

```bash
# Visible progress
tmux capture-pane -t scanrefer-full -p -S -30 | tail -20

# Per-sample checkpoint count
ls tmp/scanrefer_eval_v1_full/per_sample/pack_scanrefer_v1/ 2>/dev/null | wc -l

# Rough throughput in last 10 min
ls -lat tmp/scanrefer_eval_v1_full/per_sample/pack_scanrefer_v1/ 2>/dev/null | head -100 | \
  awk '{print $6, $7, $8}' | head
```

When `wc -l` reaches 9508 OR the tmux session ends, proceed to Step 9.5.

- [ ] **Step 9.5: Verify run completion**

```bash
tmux has-session -t scanrefer-full 2>/dev/null && echo "still running" || echo "done"
ls tmp/scanrefer_eval_v1_full/side_by_side.json
python -c "
import json
d = json.load(open('tmp/scanrefer_eval_v1_full/side_by_side.json'))
m = d['pack_v1']
n_failed = sum(1 for s in m['per_sample'] if s.get('status') == 'failed')
print(f'n={m[\"n\"]} mean_iou={m[\"mean_iou\"]:.4f} acc25={m[\"Acc@0.25\"]:.4f} acc50={m[\"Acc@0.50\"]:.4f} failed={n_failed}')
"
```

Expected: `n=9508`, failed < 5% (i.e., < 476).

- [ ] **Step 9.6: No commit needed**

`tmp/` is gitignored. The runner artifacts live there until aggregation+ingest in Task 10.

---

## Task 10: Aggregate + ingest v1

**Files:** none modified; one new SQLite row.

- [ ] **Step 10.1: Run aggregator**

```bash
source .venv/bin/activate
PYTHONPATH=src python src/evaluation/scripts/scanrefer_leaderboard_metrics.py \
    --side-by-side tmp/scanrefer_eval_v1_full/side_by_side.json \
    --scanrefer-data-root data/scanrefer \
    --phase8-data-root data/nr3d/scannet \
    --output tmp/scanrefer_eval_v1_full/leaderboard_metrics.json
```

Expected stdout:
```
n_total=9508 (Unique=NN, Multiple=NN)
acc25  overall=0.XXXX unique=0.XXXX multiple=0.XXXX
acc50  overall=0.XXXX unique=0.XXXX multiple=0.XXXX
```

Capture the numbers — they go in Task 11's v1 doc.

- [ ] **Step 10.2: Sanity-check the aggregator output**

```bash
python -c "
import json
m = json.load(open('tmp/scanrefer_eval_v1_full/leaderboard_metrics.json'))
assert m['n_total'] == 9508
assert m['n_unique'] + m['n_multiple'] == m['n_total']
assert len(m['per_sample']) == 9508
print('all invariants pass')
print(f'n_unique={m[\"n_unique\"]} n_multiple={m[\"n_multiple\"]}')
"
```

Expected: `all invariants pass` + non-zero `n_unique` + non-zero `n_multiple`.

- [ ] **Step 10.3: Save numbers for Task 11**

```bash
python -c "
import json
m = json.load(open('tmp/scanrefer_eval_v1_full/leaderboard_metrics.json'))
out = (
    f'n_total={m[\"n_total\"]}\n'
    f'n_unique={m[\"n_unique\"]}\n'
    f'n_multiple={m[\"n_multiple\"]}\n'
    f'acc25_overall={m[\"acc25_overall\"]:.4f}\n'
    f'acc50_overall={m[\"acc50_overall\"]:.4f}\n'
    f'acc25_unique={m[\"acc25_unique\"]:.4f}\n'
    f'acc50_unique={m[\"acc50_unique\"]:.4f}\n'
    f'acc25_multiple={m[\"acc25_multiple\"]:.4f}\n'
    f'acc50_multiple={m[\"acc50_multiple\"]:.4f}\n'
    f'mean_iou_overall={m[\"mean_iou_overall\"]:.4f}\n'
)
open('/tmp/scanrefer_v1_numbers.txt', 'w').write(out)
print(out)
"
```

- [ ] **Step 10.4: Run ingester**

```bash
PYTHONPATH=src python scripts/ingest_scanrefer_run.py \
    --output-dir tmp/scanrefer_eval_v1_full \
    --run-id v1_mask3d_track_20260502 \
    --branch feat/scanrefer-vg-benchmark \
    --commit "$(git rev-parse --short HEAD)" \
    --backend pack_v1 \
    --judge-model none \
    --notes "v1 ScanRefer detection-mode track on Mask3D pool, gpt-5.4-2026-03-05" \
    --leaderboard-metrics tmp/scanrefer_eval_v1_full/leaderboard_metrics.json \
    --db docs/benchmark/scanrefer/runs.sqlite
```

Expected: `[ingest] run_id=v1_mask3d_track_20260502 samples=9508 db=...`

- [ ] **Step 10.5: Verify SQL row**

```bash
sqlite3 docs/benchmark/scanrefer/runs.sqlite "
SELECT run_id,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', acc25_unique) AS u25,
       printf('%.4f', acc50_unique) AS u50,
       printf('%.4f', acc25_multiple) AS m25,
       printf('%.4f', acc50_multiple) AS m50,
       n_total
FROM runs WHERE run_id='v1_mask3d_track_20260502';"
```

Expected: 7 non-null values + n_total=9508.

- [ ] **Step 10.6: Commit the SQLite db**

```bash
git add docs/benchmark/scanrefer/runs.sqlite
git commit -m "docs(scanrefer): v1 ingest into runs.sqlite

v1_mask3d_track_20260502: full 9508-utterance ScanRefer val run on
Mask3D-pool detection-mode track, ingested with Unique/Multiple
slicing.

Numbers retained in /tmp/scanrefer_v1_numbers.txt; substituted into
the v1 version doc in Task 11."
```

---

## Task 11: v1 doc + index files

**Files:**
- Create: `docs/benchmark/scanrefer/v1_mask3d_track_20260502.md`
- Create: `docs/benchmark/scanrefer/README.md`
- Create: `docs/benchmark/scanrefer/leaderboard.md`
- Modify: `docs/benchmark/README.md`

- [ ] **Step 11.1: Re-read captured numbers**

```bash
cat /tmp/scanrefer_v1_numbers.txt
```

These 10 numbers go into all the docs in this task.

- [ ] **Step 11.2: Write v1 version doc**

Create `docs/benchmark/scanrefer/v1_mask3d_track_20260502.md` with this content (replace bracketed placeholders `[ACC25_OVERALL]`, `[ACC50_OVERALL]`, `[ACC25_UNIQUE]`, `[ACC50_UNIQUE]`, `[ACC25_MULTIPLE]`, `[ACC50_MULTIPLE]`, `[N_UNIQUE]`, `[N_MULTIPLE]`, `[MEAN_IOU]` with measured numbers as `XX.XX` percentages):

```markdown
# v1 Mask3D-Track — 2026-05-02

First ScanRefer detection-mode evaluation on the canonical full val set
(9508 utterances on 141 scenes), using Mask3D ScanNet200 predictions as
the proposal pool — the de-facto shared detector for Camp-A zero-shot
methods (ZSVG3D / SeeGround / CSVG / Z3D / SeqVLM).

## Run Identity

- Branch: `feat/scanrefer-vg-benchmark`
- Tip commit: `<TIP>` (record `git rev-parse --short HEAD` here at run time)
- Internal version: `v1_mask3d_track`
- Run ID: `v1_mask3d_track_20260502`
- Stage 2 backend: `gpt-5.4-2026-03-05` via internal ModelHub
- Workers: 32 (NR3D v3 sweet spot for 3-key API rotation)

## Methodology

ScanRefer is a detection-mode benchmark — methods must produce 3D bboxes,
not pick from a GT pool. We adopt the standard Camp-A protocol used by
ZSVG3D, SeeGround, CSVG, and Z3D:

- **Pool**: Mask3D ScanNet200 predictions distributed by ZSVG3D
  (`data/scanrefer/Mask3d/scannet200/<scene>.npz`), repackaged into a
  ConceptGraph-shaped pkl per scene by `build_scanrefer_mask3d_cg.py`.
- **GT bbox**: Phase 8 GT-CG pkl (functionally equivalent to Vil3dRef's
  `pcd_with_global_alignment` `.pth`; same `(min+max)/2` axis-aligned
  derivation).
- **Agent input**: 5 RGB keyframes per query, selected by Phase 8
  visibility of the target GT instance, with Mask3D candidate boxes
  projected as 2D overlays. Reuses NR3D v3 Stage 1 + Stage 2 unchanged.
- **Metric**: axis-aligned 3D IoU via `compute_oriented_iou_3d`
  with Euler=0; Acc@0.25 / Acc@0.50 × { Unique, Multiple, Overall }.
- **Filtering**: drop wall / floor / ceiling Mask3D instances (matches
  ZSVG3D `keep_background=False`).

## Fold

- Split: ScanRefer val
- Total utterances: 9508
- Scenes: 141 (130 NR3D-overlap + 11 ScanRefer-only built by
  `nr3d_gt_conceptgraph` Linux producer on 2026-05-02 — see
  `docs/benchmark/scanrefer/producer_report_20260502.md`).
- Unique partition: [N_UNIQUE] utts where target's class has exactly 1 GT instance in scene
- Multiple partition: [N_MULTIPLE] utts where target's class has ≥ 2 GT instances

## Headline Metrics

| Metric | Unique | Multiple | Overall |
|---|---:|---:|---:|
| Acc@0.25 | **[ACC25_UNIQUE]** | **[ACC25_MULTIPLE]** | **[ACC25_OVERALL]** |
| Acc@0.50 | **[ACC50_UNIQUE]** | **[ACC50_MULTIPLE]** | **[ACC50_OVERALL]** |

mean IoU (overall): [MEAN_IOU]

## SOTA Comparison (ScanRefer val, detection mode w/ Mask3D pool)

| Method | Setup | Unique@0.25 | Unique@0.50 | Multiple@0.25 | Multiple@0.50 | Overall@0.25 | Overall@0.50 |
|---|---|---:|---:|---:|---:|---:|---:|
| ZSVG3D / GPT-4-Turbo | Mask3D pool, zero-shot | 63.8 | 58.4 | 27.7 | 24.6 | 36.4 | 32.7 |
| CSVG (Mask3D) | Mask3D pool, zero-shot | 68.8 | 61.2 | 38.4 | 27.3 | 49.6 | 39.8 |
| SeeGround / Qwen2-VL-72B | Mask3D pool, zero-shot | 75.7 | 68.9 | 34.0 | 30.0 | 44.1 | 39.4 |
| VLM-Grounder / GPT-4V (250 sub-sample) | 2D-online proj, zero-shot | 66.0 | 29.8 | 48.3 | 33.5 | 51.6 | 32.8 |
| Z3D (Mask3D row) | Mask3D pool, zero-shot | 82.3 | 74.8 | 51.5 | 45.7 | 58.9 | 52.7 |
| **Ours v1 (gpt-5.4)** | Mask3D pool, zero-shot | **[ACC25_UNIQUE]** | **[ACC50_UNIQUE]** | **[ACC25_MULTIPLE]** | **[ACC50_MULTIPLE]** | **[ACC25_OVERALL]** | **[ACC50_OVERALL]** |

For supervised SOTA (test-server top-5 + val Table 1), see `leaderboard.md`.

## Caveats

- **Zero-shot RGB+VLM agent vs trained 3D models** — paradigm difference, not protocol violation.
- **Mask3D pool quality bounds the upper limit** — instances mis-segmented by Mask3D are unrecoverable.
- **Wall/floor/ceiling Mask3D candidates filtered** — matches ZSVG3D `keep_background=False`. Drop count per scene visible in producer report.
- **GT bbox source** = Phase 8 GT-CG pkl (Vil3dRef-equivalent; see `pool_equivalence_log_20260501.md` from NR3D v3 work).
- **Per-LLM-call durability gap** carried forward from NR3D v3 — `tool_calls`/`llm_calls` SQLite tables empty for v1.

## Cross-version comparison (ScanRefer-only)

| Version | Date | n_total | Headline | Notes |
|---|---|---:|---|---|
| **v1_mask3d_track** | 2026-05-02 | 9508 | Acc@0.25 = [ACC25_OVERALL] / Acc@0.50 = [ACC50_OVERALL] | First ScanRefer detection-mode track |

## SQLite Reproduction

```sql
SELECT run_id, n_total, n_unique, n_multiple,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', acc25_unique) AS u25,
       printf('%.4f', acc50_unique) AS u50,
       printf('%.4f', acc25_multiple) AS m25,
       printf('%.4f', acc50_multiple) AS m50
FROM runs WHERE run_id='v1_mask3d_track_20260502';
```

## Raw Artifacts

- ScanRefer JSON: `data/scanrefer/raw/ScanRefer_filtered_val.json`
- Mask3D `.npz` distribution: `data/scanrefer/Mask3d/scannet200/<scene>.npz`
- Mask3D-CG pkl (per scene): `data/scanrefer/scannet/<scene>/conceptgraph/pcd_saves/full_pcd_mask3d_axisaligned.pkl.gz`
- Phase 8 GT lookup (per scene): `data/nr3d/scannet/<scene>/conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz`
- Per-sample checkpoints: `tmp/scanrefer_eval_v1_full/per_sample/pack_scanrefer_v1/*.json`
- Aggregate: `tmp/scanrefer_eval_v1_full/side_by_side.json`
- Leaderboard metrics: `tmp/scanrefer_eval_v1_full/leaderboard_metrics.json`
- SQLite row: `docs/benchmark/scanrefer/runs.sqlite::runs.run_id='v1_mask3d_track_20260502'`

## Reproduction Command

```bash
source .venv/bin/activate

# Step 1 (one-time): convert Mask3D npz → ConceptGraph-shaped pkl
PYTHONPATH=src python -m scripts.build_scanrefer_mask3d_cg \
    --scene-list data/scanrefer/raw/ScanRefer_filtered_val.txt \
    --mask3d-root data/scanrefer/Mask3d/scannet200 \
    --raw-root data/nr3d/scannet \
    --output-root data/scanrefer/scannet

# Step 2: pack-prep
PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py \
    --sample-ids tmp/scanrefer_artifacts/full_val_sample_ids.json \
    --data-root data/scanrefer/scannet \
    --pack-name pack_scanrefer_v1

# Step 3: full agent run (~16h)
PYTHONPATH=src python src/evaluation/scripts/run_scanrefer_vg_side_by_side.py \
    --sample-ids tmp/scanrefer_artifacts/full_val_sample_ids.json \
    --data-root data/scanrefer/scannet \
    --pack-name pack_scanrefer_v1 \
    --output-dir tmp/scanrefer_eval_v1_full \
    --workers 32 --sample-retries 1

# Step 4: aggregate + ingest
PYTHONPATH=src python src/evaluation/scripts/scanrefer_leaderboard_metrics.py \
    --side-by-side tmp/scanrefer_eval_v1_full/side_by_side.json \
    --output tmp/scanrefer_eval_v1_full/leaderboard_metrics.json
PYTHONPATH=src python scripts/ingest_scanrefer_run.py \
    --output-dir tmp/scanrefer_eval_v1_full \
    --run-id v1_mask3d_track_20260502 \
    --branch feat/scanrefer-vg-benchmark --commit "$(git rev-parse --short HEAD)" \
    --leaderboard-metrics tmp/scanrefer_eval_v1_full/leaderboard_metrics.json \
    --db docs/benchmark/scanrefer/runs.sqlite
```

## Next Steps

- v2: SeeGround head-to-head with Qwen2-VL-72B backbone (apples-to-apples backbone match).
- v3: detector ablation (BIP3D / V-DETR / GroupFree3D pool variants).
- v4: SR3D extension.
```

After substituting placeholders, verify:

```bash
grep -nE '\[ACC25_OVERALL\]|\[ACC50_OVERALL\]|\[ACC25_UNIQUE\]|\[ACC50_UNIQUE\]|\[ACC25_MULTIPLE\]|\[ACC50_MULTIPLE\]|\[N_UNIQUE\]|\[N_MULTIPLE\]|\[MEAN_IOU\]|<TIP>' docs/benchmark/scanrefer/v1_mask3d_track_20260502.md
```

Expected: zero matches (all placeholders substituted).

- [ ] **Step 11.3: Write `docs/benchmark/scanrefer/README.md`**

```markdown
# ScanRefer VG Evaluation Results

This directory tracks ScanRefer visual-grounding evaluations for the
Stage-2 task-pack pipeline.

**Benchmark:** ScanRefer (Chen et al. ECCV 2020) — natural-language ScanNet references with detection-mode evaluation
**Metric:** axis-aligned 3D IoU; Acc@0.25 / Acc@0.50 × { Unique, Multiple, Overall }
**Judge:** none (programmatic IoU)

## Version Timeline

| Version | Date | Headline | Eval Scale | Key Change |
|---|---|---|---|---|
| [v1_mask3d_track](v1_mask3d_track_20260502.md) | 2026-05-02 | Acc@0.25 = [ACC25_OVERALL] / Acc@0.50 = [ACC50_OVERALL] | 9508 utts (full canonical val) | First ScanRefer eval. Mask3D ScanNet200 pool from ZSVG3D distribution; gpt-5.4-2026-03-05 backend; full 141 / 141 scene coverage (130 NR3D-overlap + 11 newly-built). |

## Current Interpretation

**v1_mask3d_track, 2026-05-02**: First ScanRefer detection-mode eval on
canonical 9508 utts. Headline **Acc@0.25 = [ACC25_OVERALL]** /
**Acc@0.50 = [ACC50_OVERALL]** with Unique/Multiple decomposition. Pool
is Mask3D ScanNet200 from ZSVG3D's CUHK SharePoint distribution (the
de-facto shared detector for Camp-A zero-shot methods); GT lookup is
the Phase 8 GT-CG pkl (Vil3dRef-equivalent). See
`v1_mask3d_track_20260502.md` for SOTA comparison.

## Reproduction

See the `Reproduction Command` section in
[v1_mask3d_track_20260502.md](v1_mask3d_track_20260502.md).

## SQLite

Canonical DB: `docs/benchmark/scanrefer/runs.sqlite`

```sql
SELECT run_id, n_total,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50
FROM runs;
```

## Caveats

- ScanRefer test split is server-only; we report on val (9508 utts, 141 scenes).
- Mask3D-pool detection-mode is the canonical setup (ZSVG3D / SeeGround / CSVG / Z3D); GT-pool ablation is intentionally NOT pursued (would break benchmark).
- Wall/floor/ceiling Mask3D candidates dropped per Camp-A convention.
```

(Substitute the bracketed numbers from `/tmp/scanrefer_v1_numbers.txt`.)

- [ ] **Step 11.4: Write `docs/benchmark/scanrefer/leaderboard.md`**

```markdown
# ScanRefer Public Leaderboard (reference)

Source: https://kaldir.vc.in.tum.de/scanrefer_benchmark/benchmark_localization
(test-server) and SeeGround Table 1 (val numbers from supervised SOTA).

## Test-server top-5 (official, Acc@0.5IoU Overall)

| Rank | Method | Unique@0.25 | Unique@0.50 | Multiple@0.25 | Multiple@0.50 | Overall@0.25 | Overall@0.50 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | UniVLG | 88.95 | 82.36 | 59.21 | 50.30 | 65.88 | 57.49 |
| 2 | Chat-Scene | 88.87 | 80.05 | 54.21 | 48.61 | 61.98 | 55.66 |
| 3 | ConcreteNet | 86.07 | 79.23 | 47.46 | 40.91 | 56.12 | 49.50 |
| 4 | cus3d | 83.84 | 70.73 | 49.08 | 40.00 | 56.88 | 46.89 |
| 5 | D-LISA | 81.95 | 69.00 | 49.75 | 39.67 | 56.97 | 46.25 |

## Validation top-5 (supervised, SeeGround Table 1)

| Rank | Method | Venue | Unique@0.25 | Unique@0.50 | Multiple@0.25 | Multiple@0.50 | Overall@0.25 | Overall@0.50 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | ConcreteNet | ECCV 2024 | 86.4 | 82.1 | 42.4 | 38.4 | 50.6 | 46.5 |
| 2 | 3D-VisTA | ICCV 2023 | 81.6 | 75.1 | 43.7 | 39.1 | 50.6 | 45.8 |
| 3 | MCLN | ECCV 2024 | 86.9 | 72.7 | 52.0 | 40.8 | 57.2 | 45.7 |
| 4 | G3-LQ | CVPR 2024 | 88.6 | 73.3 | 50.2 | 39.7 | 56.0 | 44.7 |
| 5 | EDA | CVPR 2023 | 85.8 | 68.6 | 49.1 | 37.6 | 54.6 | 42.3 |

## Zero-shot (Mask3D-pool) reference

| Method | Year | Unique@0.25 | Unique@0.50 | Multiple@0.25 | Multiple@0.50 | Overall@0.25 | Overall@0.50 |
|---|---|---:|---:|---:|---:|---:|---:|
| LLM-Grounder (998 sub-sample) | ICRA 2024 | - | - | - | - | 17.1 | 5.3 |
| ZSVG3D / GPT-4-Turbo | CVPR 2024 | 63.8 | 58.4 | 27.7 | 24.6 | 36.4 | 32.7 |
| CSVG (Mask3D) | BMVC 2025 | 68.8 | 61.2 | 38.4 | 27.3 | 49.6 | 39.8 |
| SeeGround / Qwen2-VL-72B | CVPR 2025 | 75.7 | 68.9 | 34.0 | 30.0 | 44.1 | 39.4 |
| VLM-Grounder (250 sub-sample) | CoRL 2024 | 66.0 | 29.8 | 48.3 | 33.5 | 51.6 | 32.8 |
| Z3D (Mask3D row) | 2026 arXiv | 82.3 | 74.8 | 51.5 | 45.7 | 58.9 | 52.7 |
| **Ours v1 (gpt-5.4)** | this work | **[ACC25_UNIQUE]** | **[ACC50_UNIQUE]** | **[ACC25_MULTIPLE]** | **[ACC50_MULTIPLE]** | **[ACC25_OVERALL]** | **[ACC50_OVERALL]** |
```

(Substitute placeholders.)

- [ ] **Step 11.5: Update `docs/benchmark/README.md`**

Open `docs/benchmark/README.md` and add a row to the Active Benchmarks table:

```markdown
| **ScanRefer VG** | live | [`scanrefer/`](scanrefer/) | v1 Mask3D-pool 9508Q val: Overall@0.25 **[ACC25_OVERALL]**, Overall@0.50 **[ACC50_OVERALL]**, Unique/Multiple split (gpt-5.4, zero-shot) |
```

- [ ] **Step 11.6: Commit**

```bash
git add docs/benchmark/scanrefer/v1_mask3d_track_20260502.md \
        docs/benchmark/scanrefer/README.md \
        docs/benchmark/scanrefer/leaderboard.md \
        docs/benchmark/README.md
git commit -m "docs(scanrefer): v1 mask3d-track results + indexes

First ScanRefer detection-mode evaluation: 9508 utts on 141 scenes
canonical val with Mask3D-pool from ZSVG3D distribution and gpt-5.4
agent. Acc@0.25/0.50 × Unique/Multiple/Overall + comparison vs Camp-A
zero-shot SOTA (ZSVG3D / SeeGround / CSVG / Z3D / VLM-Grounder /
LLM-Grounder).

Index files: scanrefer/README.md (timeline + Current Interpretation),
scanrefer/leaderboard.md (test-server + val supervised + zero-shot
ref tables), docs/benchmark/README.md row added."
```

---

## Task 12: Final regression sweep

**Files:** none modified; verification only.

- [ ] **Step 12.1: Run all new test files**

```bash
source .venv/bin/activate && PYTHONPATH=src pytest \
    src/scripts/tests/test_build_scanrefer_mask3d_cg.py \
    src/benchmarks/tests/test_scanrefer_loader.py \
    src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer.py \
    src/evaluation/scripts/tests/test_run_scanrefer_vg_side_by_side.py \
    src/evaluation/scripts/tests/test_scanrefer_leaderboard_metrics.py \
    src/evaluation/scripts/tests/test_ingest_scanrefer_run.py \
    -v 2>&1 | tail -10
```

Expected: all tests pass (~38 tests total).

- [ ] **Step 12.2: Confirm no regression on existing tests**

```bash
source .venv/bin/activate && PYTHONPATH=src pytest \
    src/evaluation/scripts/tests/test_nr3d_leaderboard_metrics.py \
    src/evaluation/scripts/tests/test_ingest_nr3d_run.py \
    src/evaluation/scripts/tests/test_run_nr3d_vg_side_by_side.py \
    -v 2>&1 | tail -10
```

Expected: all pass; no regressions.

- [ ] **Step 12.3: Verify clean working tree**

```bash
git status -s | grep -v scheduled_tasks.lock
```

Expected: empty (only session-only `.claude/scheduled_tasks.lock` is OK).

- [ ] **Step 12.4: Verify SQLite has v1 row**

```bash
sqlite3 docs/benchmark/scanrefer/runs.sqlite "
SELECT run_id, n_total,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50
FROM runs ORDER BY ingested_at;"
```

Expected: 1 row, n_total=9508, non-null floats.

- [ ] **Step 12.5: Push the branch**

```bash
git push origin feat/scanrefer-vg-benchmark
```

Expected: push succeeds; branch upstream is up-to-date.

---

## Summary

After all 12 tasks:

- 6 new modules: converter, loader, pack-prep, runner, aggregator, ingester (~1500 lines source + ~600 lines tests)
- ~38 new tests, all green
- 1 SQLite row in `docs/benchmark/scanrefer/runs.sqlite`
- 1 v1 version doc + 1 README + 1 leaderboard + 1 row in `docs/benchmark/README.md`
- No source changes outside the new ScanRefer files (NR3D pipeline untouched)
- ~7 hours dev wall + ~16 hours full agent run (Tasks 9-10 dominate elapsed time)
