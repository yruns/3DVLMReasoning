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
    canonical_idx = 0
    with open(taxonomy_file, encoding="utf-8") as f:
        for line in f:
            label = line.strip()
            if not label:
                continue
            out[label.lower()] = canonical_idx
            canonical_idx += 1
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
