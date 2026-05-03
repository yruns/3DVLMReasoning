"""ScanNet aggregation-based GT bbox loader for the ScanRefer v2 track.

Each ScanNet scene's `.aggregation.json` lists segGroups (one per `objectId`)
with the segment ids that belong to that instance. The `_vh_clean_2.0.010000.segs.json`
maps each mesh vertex to a segment id. The `_vh_clean_2.ply` mesh provides
the vertex coordinates. The `<scene>.txt` file contains a 4×4 axis-alignment
matrix.

This module joins those four sources to derive an axis-aligned 8-corner GT
bbox per objectId. Output convention is identical to other v1 helpers:
9-DoF [cx, cy, cz, dx, dy, dz, roll, pitch, yaw] with Euler = 0.

This is the GT bbox source ZSVG3D / SeeGround / CSVG / Z3D and Mask3D
training all derive from. See
``docs/benchmark/scanrefer/v1_oracle_analysis_20260503.md`` for why v1's
Phase 8 GT-CG was incompatible with paper-comparable evaluation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from loguru import logger


def _read_axis_alignment_matrix(scene_txt: Path) -> np.ndarray:
    """Parse `axisAlignment = ...` line from `<scene>.txt`.

    Returns:
        (4, 4) float64 matrix.
    """
    if not scene_txt.exists():
        raise FileNotFoundError(f"axis-align txt missing: {scene_txt}")
    text = scene_txt.read_text(encoding="utf-8")
    m = re.search(r"axisAlignment\s*=\s*([-\d.\seE+]+)", text)
    if not m:
        raise ValueError(f"axisAlignment line missing in {scene_txt}")
    vals = [float(x) for x in m.group(1).split()]
    if len(vals) != 16:
        raise ValueError(
            f"axisAlignment must have 16 floats, got {len(vals)} in {scene_txt}"
        )
    return np.array(vals, dtype=np.float64).reshape(4, 4)


def _aabb_to_9dof(corners_xyz: np.ndarray) -> list[float]:
    """Return [cx,cy,cz, dx,dy,dz, 0, 0, 0] AABB-9DoF from a (>=2, 3) point set."""
    arr = np.asarray(corners_xyz, dtype=np.float64)
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    cx, cy, cz = ((mn + mx) / 2.0).tolist()
    dx, dy, dz = (mx - mn).tolist()
    return [cx, cy, cz, dx, dy, dz, 0.0, 0.0, 0.0]


def load_aggregation_gt_bboxes(
    scene_id: str,
    *,
    scannet_aux_root: Path = Path("data/nr3d/scannet_aux"),
    mesh_root: Path = Path("data/nr3d/scannet_aux_meshes"),
    apply_axis_alignment: bool = True,
) -> dict[int, list[float]]:
    """Derive {objectId: bbox_9dof_aabb} from ScanNet aux + mesh.

    Args:
        scene_id: ScanNet scene name (e.g. "scene0011_00").
        scannet_aux_root: Directory holding `<scene>/<scene>.aggregation.json`,
            `<scene>_vh_clean_2.0.010000.segs.json`, and `<scene>.txt`.
        mesh_root: Directory holding `<scene>/<scene>_vh_clean_2.ply`.
        apply_axis_alignment: If True (default), apply the axis-align matrix
            to mesh vertices before deriving the AABB. ZSVG3D / SeeGround /
            CSVG / Z3D / Mask3D-training all assume axis-aligned coords.

    Returns:
        Dict mapping `int(objectId) → 9-DoF AABB list` for every segGroup
        with ≥ 3 mesh vertices. Empty / degenerate segGroups are silently
        skipped (and logged).

    Raises:
        FileNotFoundError: any required input file missing.
        ValueError: vertex count vs segIndices mismatch (mesh-vs-segs file
            mismatch — almost always means the wrong .ply variant).
    """
    aux_dir = Path(scannet_aux_root) / scene_id
    ply_path = Path(mesh_root) / scene_id / f"{scene_id}_vh_clean_2.ply"
    segs_path = aux_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    agg_path = aux_dir / f"{scene_id}.aggregation.json"
    align_path = aux_dir / f"{scene_id}.txt"

    for p in (ply_path, segs_path, agg_path, align_path):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")

    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.shape[0] == 0:
        raise ValueError(f"mesh has no vertices: {ply_path}")

    seg_indices = np.array(json.loads(segs_path.read_text(encoding="utf-8"))["segIndices"])
    if len(seg_indices) != len(verts):
        raise ValueError(
            f"vertex / seg_indices mismatch for {scene_id}: "
            f"verts={len(verts)} segs={len(seg_indices)}"
        )

    if apply_axis_alignment:
        mat = _read_axis_alignment_matrix(align_path)
        verts_h = np.hstack([verts, np.ones((len(verts), 1))])
        verts = (mat @ verts_h.T).T[:, :3]

    aggdata = json.loads(agg_path.read_text(encoding="utf-8"))
    out: dict[int, list[float]] = {}
    skipped = 0
    for sg in aggdata.get("segGroups", []):
        oid = int(sg["objectId"])
        target_segs = set(int(s) for s in sg.get("segments", []))
        if not target_segs:
            skipped += 1
            continue
        mask = np.isin(seg_indices, list(target_segs))
        n = int(mask.sum())
        if n < 3:
            logger.debug(
                "skipping {} objectId={} label={!r}: only {} verts",
                scene_id, oid, sg.get("label"), n,
            )
            skipped += 1
            continue
        out[oid] = _aabb_to_9dof(verts[mask])

    if skipped:
        logger.debug("{}: skipped {} segGroups (< 3 verts)", scene_id, skipped)
    return out


__all__ = ["load_aggregation_gt_bboxes"]
