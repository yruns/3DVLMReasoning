#!/usr/bin/env python3
"""Prepare EmbodiedScan ScanNet scenes for the ConceptGraph pipeline.

Uses ``embodiedscan_infos_val.pkl`` as the authoritative frame list, which
has already filtered out bad-pose frames (tracking failures with -inf poses).

For each ScanNet scene, the script:
  1. Reads the valid frame list from infos_val.pkl
  2. Links the corresponding RGB/depth/pose files from posed_images/
  3. Writes intrinsics from the PKL (not from posed_images/ files)
  4. Strictly validates: frame counts must match, no inf/nan in poses

Output layout per scene::

    data/embodiedscan/scannet/<scene_id>/
      raw/
        000000-rgb.jpg, 000000-depth.png, 000000.txt, ...
        intrinsic_color.txt, intrinsic_depth.txt
        extrinsic_color.txt, extrinsic_depth.txt
        traj.txt, scene_info.json
      conceptgraph/
        traj.txt -> ../raw/traj.txt  (+ other symlinks)
        scene_info.json

Usage:
    python -m src.scripts.prepare_embodiedscan_scene \\
        --pkl data/embodiedscan/embodiedscan_infos_val.pkl \\
        --data_root data/embodiedscan/scannet \\
        [--scene_id scene0006_00]   # omit to process all scannet scenes
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np


def _link(src: Path, dst: Path) -> None:
    """Create a symlink dst -> src. Skip if dst already exists."""
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())


def prepare_scene(
    scene_id: str,
    entry: dict,
    data_root: Path,
) -> dict:
    """Prepare a single scene using infos_val.pkl entry as truth.

    Parameters
    ----------
    scene_id : str
        e.g. ``scene0006_00``
    entry : dict
        The data_list entry from infos_val.pkl for this scene.
    data_root : Path
        Root containing ``posed_images/`` and output scene dirs.

    Returns
    -------
    dict
        Scene metadata.

    Raises
    ------
    FileNotFoundError
        If source files are missing.
    ValueError
        If frame counts mismatch or bad poses detected.
    """
    posed_dir = data_root / "posed_images" / scene_id
    if not posed_dir.is_dir():
        raise FileNotFoundError(f"posed_images not found: {posed_dir}")

    scene_dir = data_root / scene_id
    raw_dir = scene_dir / "raw"
    cg_dir = scene_dir / "conceptgraph"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cg_dir.mkdir(parents=True, exist_ok=True)

    images = entry["images"]
    expected_count = len(images)

    # --- Link RGB, depth, pose for each valid frame ---
    for out_idx, img_entry in enumerate(images):
        # Extract original frame number from img_path
        # e.g. "scannet/posed_images/scene0006_00/00150.jpg" -> "00150"
        m = re.search(r"/(\d+)\.jpg$", img_entry["img_path"])
        if not m:
            raise ValueError(f"Cannot parse frame num from: {img_entry['img_path']}")
        frame_str = m.group(1)

        rgb_src = posed_dir / f"{frame_str}.jpg"
        depth_src = posed_dir / f"{frame_str}.png"
        pose_src = posed_dir / f"{frame_str}.txt"

        if not rgb_src.exists():
            raise FileNotFoundError(f"RGB not found: {rgb_src}")
        if not depth_src.exists():
            raise FileNotFoundError(f"Depth not found: {depth_src}")

        out_prefix = f"{out_idx:06d}"
        _link(rgb_src, raw_dir / f"{out_prefix}-rgb.jpg")
        _link(depth_src, raw_dir / f"{out_prefix}-depth.png")

        # Write pose from PKL cam2global (authoritative, not from .txt file)
        pose_path = raw_dir / f"{out_prefix}.txt"
        if not pose_path.exists():
            cam2global = np.array(img_entry["cam2global"], dtype=np.float64)
            if not np.all(np.isfinite(cam2global)):
                raise ValueError(
                    f"Bad pose in PKL for {scene_id} frame {frame_str}: "
                    f"contains inf/nan"
                )
            np.savetxt(str(pose_path), cam2global, fmt="%.6f")

    # --- Validate frame count ---
    actual_rgb = len(list(raw_dir.glob("*-rgb.jpg")))
    if actual_rgb != expected_count:
        raise ValueError(
            f"{scene_id}: frame count mismatch! "
            f"Expected {expected_count} from PKL, got {actual_rgb} in raw/"
        )

    # --- Intrinsics from PKL ---
    cam2img = np.array(entry["cam2img"], dtype=np.float64)
    intrinsic_path = raw_dir / "intrinsic_color.txt"
    if not intrinsic_path.exists():
        np.savetxt(str(intrinsic_path), cam2img, fmt="%.6f")

    if "depth_cam2img" in entry:
        depth_cam2img = np.array(entry["depth_cam2img"], dtype=np.float64)
        depth_intrinsic_path = raw_dir / "intrinsic_depth.txt"
        if not depth_intrinsic_path.exists():
            np.savetxt(str(depth_intrinsic_path), depth_cam2img, fmt="%.6f")

    # --- Extrinsics (identity) ---
    identity = np.eye(4)
    for name in ("extrinsic_color.txt", "extrinsic_depth.txt"):
        p = raw_dir / name
        if not p.exists():
            np.savetxt(str(p), identity, fmt="%.6f")

    # --- Trajectory file ---
    traj_path = raw_dir / "traj.txt"
    if not traj_path.exists():
        with traj_path.open("w") as f:
            for img_entry in images:
                pose = np.array(img_entry["cam2global"], dtype=np.float64).reshape(16)
                f.write(" ".join(f"{v:.8f}" for v in pose) + "\n")

    # --- Final pose validation ---
    for pose_file in raw_dir.glob("[0-9]*.txt"):
        pose = np.loadtxt(str(pose_file))
        if not np.all(np.isfinite(pose)):
            raise ValueError(f"Bad pose in {pose_file}: contains inf/nan")

    # --- Symlink key files into conceptgraph/ ---
    for name in ("traj.txt", "intrinsic_color.txt", "intrinsic_depth.txt",
                 "extrinsic_color.txt", "extrinsic_depth.txt"):
        src = raw_dir / name
        dst = cg_dir / name
        if src.exists():
            _link(src, dst)

    # --- Scene info ---
    info = {
        "scene_id": scene_id,
        "dataset": "embodiedscan",
        "source_pkl": "embodiedscan_infos_val.pkl",
        "raw_dir": str(raw_dir),
        "scene_dir": str(cg_dir),
        "num_frames": expected_count,
        "num_frames_actual": actual_rgb,
    }
    for target_dir in (raw_dir, cg_dir):
        info_path = target_dir / "scene_info.json"
        if not info_path.exists():
            info_path.write_text(
                json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pkl", type=Path, required=True,
        help="Path to embodiedscan_infos_val.pkl",
    )
    parser.add_argument(
        "--data_root", type=Path, required=True,
        help="Root directory (contains posed_images/ and scene output dirs).",
    )
    parser.add_argument(
        "--scene_id", default=None,
        help="Process a single scene. Omit to process all ScanNet scenes.",
    )
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        val_info = pickle.load(f)

    # Build scene_id -> entry mapping for ScanNet scenes
    scene_map: dict[str, dict] = {}
    for entry in val_info["data_list"]:
        sid = entry["sample_idx"]
        if sid.startswith("scannet/"):
            scene_id = sid.replace("scannet/", "")
            scene_map[scene_id] = entry

    if args.scene_id:
        targets = [args.scene_id]
    else:
        targets = sorted(scene_map.keys())

    print(f"[INFO] Processing {len(targets)} scenes")

    ok = 0
    fail = 0
    for scene_id in targets:
        if scene_id not in scene_map:
            print(f"[SKIP] {scene_id} not in PKL")
            continue
        try:
            info = prepare_scene(scene_id, scene_map[scene_id], args.data_root)
            print(f"[OK] {scene_id}: {info['num_frames']} frames")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {scene_id}: {e}", file=sys.stderr)
            fail += 1

    print(f"\n[DONE] {ok} ok, {fail} failed, {len(targets)} total")
    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
