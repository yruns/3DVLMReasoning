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


# ---------- IO + Visibility ----------

import argparse
import json

DEFAULT_SCANNET200_TAXONOMY = Path("conceptgraph/scannet200_classes.txt")
DEFAULT_MASK3D_ROOT = Path("data/scanrefer/Mask3d/scannet200")
DEFAULT_RAW_ROOT = Path("data/nr3d/scannet")
DEFAULT_OUTPUT_ROOT = Path("data/scanrefer/scannet")


def _load_camera(raw_dir: Path) -> tuple[np.ndarray, list[np.ndarray], list[Path | None]]:
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
    depth_paths: list[Path | None] = []
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
        output_path=output_visibility,
    )

    output_scene_info.parent.mkdir(parents=True, exist_ok=True)
    output_scene_info.write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "source_npz": str(mask3d_npz_path),
                "raw_dir": str(raw_dir),
                "num_objects": n_kept,
                "num_dropped": n_dropped,
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
