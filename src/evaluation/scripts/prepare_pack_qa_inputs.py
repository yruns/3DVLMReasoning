"""Pack-v1 prep for QA benchmarks (OpenEQA / SQA3D) under v9 catalog-first design.

Mirrors the per-benchmark VG prep scripts, but builds catalogs from
ConceptGraph 3D-object segmentation output instead of Mask3D / V-DETR proposal
pools. The same script handles OpenEQA and SQA3D via the `benchmark` flag.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from agents.catalog import from_conceptgraph_objects
from query_scene.scene_bev_builder import (
    OpenEqaScanNetBEVBuilder,
    Sqa3dScanNetBEVBuilder,
)

# Tracks the active benchmark for `_render_qa_bev` to pick the right builder.
# write_qa_scene_artifacts mutates this before rendering. Tests can also
# monkey-patch _render_qa_bev directly to avoid needing a real mesh.
_BENCHMARK_FOR_BUILDER: str = "openeqa"


def _render_qa_bev(
    *,
    scene_id: str,
    data_root: Path,
    proposals,
    output_path: Path,
    highlight_ids: list[int] | None,
) -> Path:
    benchmark = _BENCHMARK_FOR_BUILDER
    builder = (
        OpenEqaScanNetBEVBuilder()
        if benchmark == "openeqa"
        else Sqa3dScanNetBEVBuilder()
    )
    return builder.build_with_labels(
        scene_id=scene_id,
        data_root=data_root,
        proposals=proposals,
        output_path=output_path,
        highlight_ids=highlight_ids,
    )


def _build_camera_trajectory_qa(scene_dir: Path) -> dict[int, list[float]]:
    traj_path = scene_dir / "conceptgraph" / "traj.txt"
    if not traj_path.exists():
        raise FileNotFoundError(f"traj.txt missing: {traj_path}")
    raw = np.loadtxt(str(traj_path)).reshape(-1, 4, 4)
    out: dict[int, list[float]] = {}
    for i, pose in enumerate(raw):
        x = float(pose[0, 3])
        y = float(pose[1, 3])
        forward = -pose[:3, 2]
        yaw = float(math.atan2(forward[1], forward[0]))
        out[i] = [x, y, yaw]
    return out


def write_qa_scene_artifacts(
    *,
    benchmark: str,
    clip_id: str,
    data_root: Path,
    pack_name: str,
    view_to_objects: dict[int, list[tuple[int, float]]],
    valid_frame_ids: list[int],
    scene_category: str | None,
) -> dict[str, str]:
    """Emit BEV + scene_catalog.json + camera trajectory for a QA clip."""
    if benchmark not in ("openeqa", "sqa3d"):
        raise ValueError(f"unsupported QA benchmark: {benchmark!r}")
    global _BENCHMARK_FOR_BUILDER
    _BENCHMARK_FOR_BUILDER = benchmark

    clip_dir = data_root / clip_id
    pack_dir = clip_dir / pack_name
    bev_dir = pack_dir / "bev"
    bev_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = pack_dir / "scene_catalog.json"
    traj_out_path = pack_dir / "camera_trajectory.json"
    raw_rgb_template = str(clip_dir / "raw" / "{frame_id:06d}-rgb.png")
    bev_path = bev_dir / f"scene_bev_{benchmark}.png"

    catalog = from_conceptgraph_objects(
        pcd_saves_dir=clip_dir / "conceptgraph" / "pcd_saves",
        detections_dir=clip_dir / "conceptgraph" / "gsa_detections_ram_withbg_allclasses",
        view_to_objects=view_to_objects,
        scene_id=clip_id,
        bev_image_path=str(bev_path),
        scene_category=scene_category,
        valid_frame_ids=list(valid_frame_ids),
        raw_rgb_template=raw_rgb_template,
    )
    _render_qa_bev(
        scene_id=clip_id,
        data_root=data_root,
        proposals=catalog.proposals,
        output_path=bev_path,
        highlight_ids=None,
    )
    catalog_path.write_text(
        json.dumps(catalog.model_dump(), ensure_ascii=False, indent=2)
    )
    traj = _build_camera_trajectory_qa(clip_dir)
    traj_out_path.write_text(json.dumps({str(k): v for k, v in traj.items()}))
    return {
        "bev_image_path": str(bev_path),
        "scene_catalog_path": str(catalog_path),
        "camera_trajectory_path": str(traj_out_path),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=("openeqa", "sqa3d"), required=True)
    p.add_argument("--sample-ids", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--pack-name", default="pack_openeqa_v9_catalog_first")
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


__all__ = ["write_qa_scene_artifacts", "parse_args"]
