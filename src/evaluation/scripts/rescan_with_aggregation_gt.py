"""Re-aggregate a ScanRefer side-by-side run against ScanNet aggregation-based GT.

Given v1's `side_by_side.json` (which contains per-sample agent decisions
with `predicted_bbox_3d_9dof` = the chosen Mask3D candidate's bbox), this
script swaps the GT side of every IoU computation from Phase 8 GT-CG to
the aggregation-derived GT bbox (from `_vh_clean_2.ply` + segs.json +
aggregation.json + axis-align matrix), recomputes IoU per sample, and
writes a new `side_by_side.json` whose numbers are paper-comparable to
ZSVG3D / SeeGround / CSVG / Z3D.

No agent re-run is needed. The agent's pick (predicted bbox) is invariant
to GT derivation — only the IoU-target side changes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

from benchmarks.embodiedscan_eval import compute_oriented_iou_3d
from benchmarks.scanrefer_aggregation_gt import load_aggregation_gt_bboxes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side-by-side", required=True, type=Path,
                   help="v1 side_by_side.json from the runner")
    p.add_argument("--scannet-aux-root", default=Path("data/nr3d/scannet_aux"),
                   type=Path)
    p.add_argument("--mesh-root", default=Path("data/nr3d/scannet_aux_meshes"),
                   type=Path)
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to write the v2 side_by_side.json")
    p.add_argument("--backend", default="pack_v1")
    p.add_argument("--apply-axis-alignment", action=argparse.BooleanOptionalAction,
                   default=True)
    return p.parse_args()


def rescan(
    *,
    side_by_side_path: Path,
    output_dir: Path,
    scannet_aux_root: Path = Path("data/nr3d/scannet_aux"),
    mesh_root: Path = Path("data/nr3d/scannet_aux_meshes"),
    backend: str = "pack_v1",
    apply_axis_alignment: bool = True,
) -> dict[str, Any]:
    """Read v1 side_by_side, recompute IoUs against aggregation GT, return v2 payload.

    Side effects: writes `output_dir / side_by_side.json` (v2-style payload).

    Returns:
        The new payload dict. Same shape as v1 side_by_side.json.

    Raises:
        FileNotFoundError on missing side_by_side or aux files.
        ValueError on missing backend / per_sample.
    """
    payload_v1 = json.loads(side_by_side_path.read_text(encoding="utf-8"))
    if backend not in payload_v1:
        raise ValueError(f"side_by_side.json missing backend={backend!r}")
    metrics_v1 = payload_v1[backend]
    per_sample_v1 = metrics_v1.get("per_sample") or []
    if not isinstance(per_sample_v1, list) or not per_sample_v1:
        raise ValueError(f"side_by_side.json[{backend}].per_sample empty")

    # Group by scene to load each scene's aggregation GT once
    by_scene: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(per_sample_v1):
        scene = rec["sample_id"].split("::")[0].split("/")[-1]
        by_scene[scene].append(i)

    logger.info(
        "Re-aggregating {} samples across {} scenes",
        len(per_sample_v1), len(by_scene),
    )

    # Counters for stats
    n_total = len(per_sample_v1)
    n_missing_scene = 0
    n_missing_target = 0
    n_no_prediction = 0  # already-failed v1 samples

    new_per_sample: list[dict[str, Any]] = []
    for scene_idx, (scene, indices) in enumerate(sorted(by_scene.items()), start=1):
        try:
            agg_bboxes = load_aggregation_gt_bboxes(
                scene,
                scannet_aux_root=scannet_aux_root,
                mesh_root=mesh_root,
                apply_axis_alignment=apply_axis_alignment,
            )
        except FileNotFoundError as exc:
            logger.warning("scene {} missing aux/mesh: {}", scene, exc)
            for i in indices:
                rec = dict(per_sample_v1[i])
                rec["status"] = "failed"
                rec["error"] = f"v2: aggregation GT missing ({exc})"
                rec["iou"] = 0.0
                rec["gt_bbox_3d_9dof"] = None
                new_per_sample.append(rec)
                n_missing_scene += 1
            continue

        for i in indices:
            rec_v1 = per_sample_v1[i]
            rec = dict(rec_v1)
            sid = rec["sample_id"]
            target_id = int(sid.split("::")[1])
            new_gt = agg_bboxes.get(target_id)
            pred = rec.get("predicted_bbox_3d_9dof")

            if new_gt is None:
                rec["status"] = "failed"
                rec["error"] = "v2: aggregation GT missing for target_id"
                rec["iou"] = 0.0
                rec["gt_bbox_3d_9dof"] = None
                n_missing_target += 1
            elif pred is None:
                # Agent already failed in v1; carry over with new GT but iou=0
                rec["iou"] = 0.0
                rec["gt_bbox_3d_9dof"] = list(new_gt)
                n_no_prediction += 1
            else:
                iou = compute_oriented_iou_3d(pred, new_gt)
                rec["iou"] = float(iou)
                rec["gt_bbox_3d_9dof"] = list(new_gt)
            new_per_sample.append(rec)

        if scene_idx % 30 == 0:
            logger.info("  processed {}/{} scenes", scene_idx, len(by_scene))

    # Reorder new_per_sample to match v1 ordering by sample_id
    by_sid = {r["sample_id"]: r for r in new_per_sample}
    new_per_sample = [by_sid[r["sample_id"]] for r in per_sample_v1]

    completed = [r for r in new_per_sample if r.get("status") != "failed"]
    ious = [float(r["iou"]) if r.get("iou") is not None else 0.0
            for r in new_per_sample]
    n = len(new_per_sample)
    mean_iou = sum(ious) / n if n else 0.0
    acc25 = sum(1 for v in ious if v >= 0.25) / n if n else 0.0
    acc50 = sum(1 for v in ious if v >= 0.50) / n if n else 0.0

    payload_v2 = dict(payload_v1)
    payload_v2[backend] = {
        **metrics_v1,
        "n": n,
        "mean_iou": mean_iou,
        "Acc@0.25": acc25,
        "Acc@0.50": acc50,
        "per_sample": new_per_sample,
        "v2_rescan_stats": {
            "source_side_by_side": str(side_by_side_path),
            "n_missing_scene": n_missing_scene,
            "n_missing_target": n_missing_target,
            "n_no_prediction": n_no_prediction,
            "n_completed_with_new_gt": len(completed),
            "apply_axis_alignment": apply_axis_alignment,
            "scannet_aux_root": str(scannet_aux_root),
            "mesh_root": str(mesh_root),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "side_by_side.json"
    out_path.write_text(
        json.dumps(payload_v2, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "wrote {} (n={} mean_iou={:.4f} acc25={:.4f} acc50={:.4f})",
        out_path, n, mean_iou, acc25, acc50,
    )
    logger.info(
        "stats: missing_scene={} missing_target={} no_prediction={} completed_new_gt={}",
        n_missing_scene, n_missing_target, n_no_prediction, len(completed),
    )
    return payload_v2


def main() -> None:
    args = parse_args()
    rescan(
        side_by_side_path=args.side_by_side,
        output_dir=args.output_dir,
        scannet_aux_root=args.scannet_aux_root,
        mesh_root=args.mesh_root,
        backend=args.backend,
        apply_axis_alignment=args.apply_axis_alignment,
    )


if __name__ == "__main__":
    main()


__all__ = ["rescan"]
