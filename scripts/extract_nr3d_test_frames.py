#!/usr/bin/env python3
"""Extract NR3D ScanNet .sens frames into the OpenEQA raw layout."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scripts.nr3d_gt_conceptgraph import load_scene_ids, parse_axis_alignment  # noqa: E402


@dataclass(frozen=True)
class ExtractStats:
    scene_id: str
    frame_count: int
    dropped_bad_pose: int
    raw_total: int
    elapsed_seconds: float
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nr3d-root", type=Path, default=Path("data/nr3d"))
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=Path("data/nr3d/raw/test_scans.txt"),
    )
    parser.add_argument(
        "--scannet-root",
        type=Path,
        default=Path("/home/ysh/Datasets/ScanNet"),
    )
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--invalid-pose-policy",
        choices=["drop", "fail"],
        default="drop",
        help="How to handle stride-selected frames whose poses contain NaN/Inf.",
    )
    parser.add_argument(
        "--max-bad-pose-ratio",
        type=float,
        default=0.5,
        help="Raise if dropped_bad_pose / raw_total exceeds this value for one scene.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError(f"--stride must be positive, got {args.stride}")
    if args.workers <= 0:
        raise ValueError(f"--workers must be positive, got {args.workers}")
    if not 0.0 <= args.max_bad_pose_ratio <= 1.0:
        raise ValueError(
            f"--max-bad-pose-ratio must be in [0, 1], got {args.max_bad_pose_ratio}"
        )
    if args.force and args.resume:
        raise ValueError("--force and --resume are mutually exclusive")
    scene_ids = args.scenes if args.scenes else load_scene_ids(args.scene_list)
    start = time.time()
    stats: list[ExtractStats] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                extract_scene,
                scene_id,
                args.nr3d_root,
                args.scannet_root,
                args.stride,
                args.force,
                args.resume,
                args.invalid_pose_policy,
                args.max_bad_pose_ratio,
            )
            for scene_id in scene_ids
        ]
        for future in as_completed(futures):
            item = future.result()
            stats.append(item)
            print(
                json.dumps(
                    {
                        "scene_id": item.scene_id,
                        "frame_count": item.frame_count,
                        "dropped_bad_pose": item.dropped_bad_pose,
                        "raw_total": item.raw_total,
                        "elapsed_seconds": round(item.elapsed_seconds, 2),
                        "status": item.status,
                    }
                ),
                flush=True,
            )
    stats.sort(key=lambda item: item.scene_id)
    summary = {
        "num_scenes": len(stats),
        "num_frames": sum(item.frame_count for item in stats),
        "dropped_bad_pose": sum(item.dropped_bad_pose for item in stats),
        "raw_total": sum(item.raw_total for item in stats),
        "elapsed_seconds": round(time.time() - start, 2),
        "statuses": {
            status: sum(1 for item in stats if item.status == status)
            for status in sorted({item.status for item in stats})
        },
    }
    _write_frame_extraction_log(args.nr3d_root, stats)
    print(json.dumps(summary, indent=2), flush=True)


def extract_scene(
    scene_id: str,
    nr3d_root: Path,
    scannet_root: Path,
    stride: int,
    force: bool,
    resume: bool,
    invalid_pose_policy: str,
    max_bad_pose_ratio: float,
) -> ExtractStats:
    start = time.time()
    scene_root = nr3d_root / "scannet" / scene_id
    raw_dir = scene_root / "raw"
    raw_temp = scene_root / "raw_temp"
    sens_path = scannet_root / "scans" / scene_id / f"{scene_id}.sens"
    if not sens_path.exists():
        raise FileNotFoundError(f"{scene_id}: .sens not found: {sens_path}")
    axis_alignment = parse_axis_alignment(
        nr3d_root / "scannet_aux" / scene_id / f"{scene_id}.txt"
    )

    if raw_dir.exists():
        if force:
            shutil.rmtree(raw_dir)
        elif resume and _raw_complete(raw_dir):
            if raw_temp.exists():
                shutil.rmtree(raw_temp)
            frame_count, dropped_bad_pose, raw_total = _read_existing_raw_stats(raw_dir)
            return ExtractStats(
                scene_id=scene_id,
                frame_count=frame_count,
                dropped_bad_pose=dropped_bad_pose,
                raw_total=raw_total,
                elapsed_seconds=time.time() - start,
                status="already_complete",
            )
        elif resume:
            shutil.rmtree(raw_dir)
        else:
            raise FileExistsError(
                f"{scene_id}: raw output already exists; use --force or --resume: {raw_dir}"
            )
    if raw_temp.exists():
        shutil.rmtree(raw_temp)

    raw_temp.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "src.scripts.scannet_process.reader",
        "--filename",
        str(sens_path),
        "--output_path",
        str(raw_temp),
        "--export_color_images",
        "--export_depth_images",
        "--export_poses",
        "--export_intrinsics",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    try:
        frame_count, dropped_bad_pose, raw_total = _normalize_raw_temp(
            scene_id,
            raw_temp,
            raw_dir,
            stride,
            sens_path,
            axis_alignment,
            invalid_pose_policy,
            max_bad_pose_ratio,
        )
    except Exception:
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        raise
    shutil.rmtree(raw_temp)
    ratio = dropped_bad_pose / raw_total if raw_total else 0.0
    print(
        f"scene {scene_id}: kept={frame_count} dropped_bad_pose={dropped_bad_pose} "
        f"ratio={ratio:.6f}",
        flush=True,
    )
    return ExtractStats(
        scene_id=scene_id,
        frame_count=frame_count,
        dropped_bad_pose=dropped_bad_pose,
        raw_total=raw_total,
        elapsed_seconds=time.time() - start,
        status="extracted",
    )


def _normalize_raw_temp(
    scene_id: str,
    raw_temp: Path,
    raw_dir: Path,
    stride: int,
    sens_path: Path,
    axis_alignment: np.ndarray,
    invalid_pose_policy: str,
    max_bad_pose_ratio: float,
) -> tuple[int, int, int]:
    color_dir = raw_temp / "color"
    depth_dir = raw_temp / "depth"
    pose_dir = raw_temp / "pose"
    intrinsic_dir = raw_temp / "intrinsic"
    for path in [color_dir, depth_dir, pose_dir, intrinsic_dir]:
        if not path.exists():
            raise FileNotFoundError(f"{scene_id}: expected reader output missing: {path}")

    color_files = {int(path.stem): path for path in color_dir.glob("*.jpg")}
    depth_files = {int(path.stem): path for path in depth_dir.glob("*.png")}
    pose_files = {int(path.stem): path for path in pose_dir.glob("*.txt")}
    if not color_files:
        raise FileNotFoundError(f"{scene_id}: no color frames exported under {color_dir}")
    if set(color_files) != set(depth_files) or set(color_files) != set(pose_files):
        raise ValueError(
            f"{scene_id}: reader frame id mismatch color/depth/pose "
            f"{len(color_files)}/{len(depth_files)}/{len(pose_files)}"
        )
    stride_frame_ids = [
        frame_id for frame_id in sorted(color_files) if frame_id % stride == 0
    ]
    if not stride_frame_ids:
        raise ValueError(f"{scene_id}: no frames kept at stride={stride}")

    poses: dict[int, np.ndarray] = {}
    kept: list[int] = []
    dropped_bad_pose: list[int] = []
    for frame_id in stride_frame_ids:
        pose = np.loadtxt(pose_files[frame_id]).astype(np.float64)
        if pose.shape != (4, 4):
            raise ValueError(
                f"{scene_id}: pose frame {frame_id} must be 4x4, got {pose.shape}"
            )
        if not np.isfinite(pose).all():
            if invalid_pose_policy == "fail":
                raise ValueError(f"{scene_id}: pose frame {frame_id} contains NaN/Inf")
            dropped_bad_pose.append(frame_id)
            continue
        aligned_pose = axis_alignment @ pose
        if not np.isfinite(aligned_pose).all():
            raise ValueError(f"{scene_id}: aligned pose frame {frame_id} contains NaN/Inf")
        poses[frame_id] = aligned_pose
        kept.append(frame_id)
    raw_total = len(stride_frame_ids)
    if not kept:
        raise ValueError(
            f"{scene_id}: all {raw_total} stride-selected frames have invalid poses"
        )
    bad_pose_ratio = len(dropped_bad_pose) / raw_total
    if bad_pose_ratio > max_bad_pose_ratio:
        raise ValueError(
            f"{scene_id}: dropped_bad_pose ratio {bad_pose_ratio:.6f} exceeds "
            f"threshold {max_bad_pose_ratio:.6f} "
            f"({len(dropped_bad_pose)}/{raw_total})"
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "intrinsic_color.txt",
        "intrinsic_depth.txt",
        "extrinsic_color.txt",
        "extrinsic_depth.txt",
    ]:
        src = intrinsic_dir / name
        if not src.exists():
            raise FileNotFoundError(f"{scene_id}: intrinsic file missing: {src}")
        shutil.copy2(src, raw_dir / name)

    traj_lines: list[str] = []
    for frame_id in kept:
        stem = f"{frame_id:06d}"
        with Image.open(color_files[frame_id]) as image:
            image.convert("RGB").save(raw_dir / f"{stem}-rgb.png")
        shutil.copy2(depth_files[frame_id], raw_dir / f"{stem}-depth.png")
        pose = poses[frame_id]
        np.savetxt(raw_dir / f"{stem}.txt", pose, fmt="%.8f")
        traj_lines.append(" ".join(f"{value:.8f}" for value in pose.reshape(-1)))
    (raw_dir / "traj.txt").write_text("\n".join(traj_lines) + "\n", encoding="utf-8")
    scene_info = {
        "scene_id": scene_id,
        "source_sens": str(sens_path.resolve()),
        "frame_stride": stride,
        "axis_alignment_applied": True,
        "pose_transform": "axisAlignment @ sens_pose",
        "axis_alignment": axis_alignment.reshape(-1).tolist(),
        "kept_frame_ids": kept,
        "dropped_bad_pose_frame_ids": dropped_bad_pose,
        "dropped_bad_pose": len(dropped_bad_pose),
        "raw_total_stride_frames": raw_total,
        "num_rgb_frames": len(kept),
        "num_depth_frames": len(kept),
        "num_pose_files": len(kept),
    }
    (raw_dir / "scene_info.json").write_text(
        json.dumps(scene_info, indent=2),
        encoding="utf-8",
    )
    return len(kept), len(dropped_bad_pose), raw_total


def _read_existing_raw_stats(raw_dir: Path) -> tuple[int, int, int]:
    frame_count = len(sorted(raw_dir.glob("*-rgb.png")))
    info_path = raw_dir / "scene_info.json"
    if not info_path.exists():
        return frame_count, 0, frame_count
    info = json.loads(info_path.read_text(encoding="utf-8"))
    dropped_bad_pose = int(info.get("dropped_bad_pose", 0))
    raw_total = int(info.get("raw_total_stride_frames", frame_count + dropped_bad_pose))
    return frame_count, dropped_bad_pose, raw_total


def _write_frame_extraction_log(nr3d_root: Path, stats: list[ExtractStats]) -> None:
    log_path = nr3d_root / "scannet" / ".frame_extraction_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "scene": item.scene_id,
            "kept": item.frame_count,
            "dropped_bad_pose": item.dropped_bad_pose,
            "raw_total": item.raw_total,
        }
        for item in sorted(stats, key=lambda value: value.scene_id)
    ]
    log_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def _raw_complete(raw_dir: Path) -> bool:
    required = [
        raw_dir / "traj.txt",
        raw_dir / "scene_info.json",
        raw_dir / "intrinsic_color.txt",
        raw_dir / "intrinsic_depth.txt",
        raw_dir / "extrinsic_color.txt",
        raw_dir / "extrinsic_depth.txt",
    ]
    if not all(path.exists() for path in required):
        return False
    info = json.loads((raw_dir / "scene_info.json").read_text(encoding="utf-8"))
    if info.get("axis_alignment_applied") is not True:
        return False
    rgb_paths = sorted(raw_dir.glob("*-rgb.png"))
    depth_paths = sorted(raw_dir.glob("*-depth.png"))
    pose_paths = sorted(raw_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].txt"))
    return bool(rgb_paths) and len(rgb_paths) == len(depth_paths) == len(pose_paths)


if __name__ == "__main__":
    main()
