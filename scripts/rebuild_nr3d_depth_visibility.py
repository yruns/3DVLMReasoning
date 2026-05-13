#!/usr/bin/env python3
"""Rebuild NR3D object-frame visibility indices with depth occlusion."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from scripts.build_visibility_index import build_visibility_index, save_visibility_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/nr3d/scannet"))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--scene-list", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-distance", type=float, default=5.0)
    parser.add_argument("--min-visible-ratio", type=float, default=0.03)
    parser.add_argument("--min-visible-points", type=int, default=5)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def load_scene_ids(data_root: Path, scenes: list[str] | None, scene_list: Path | None) -> list[str]:
    if scenes:
        return sorted(dict.fromkeys(scenes))
    if scene_list:
        payload = json.loads(scene_list.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{scene_list} must contain a JSON list")
        return sorted(dict.fromkeys(str(item) for item in payload))
    return sorted(path.name for path in data_root.glob("scene*_??") if path.is_dir())


def _frame_id(path: Path) -> str:
    return path.name[:6]


def _read_image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"failed to read RGB image: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def _object_label(obj: dict[str, Any]) -> str:
    names = obj.get("class_name")
    if isinstance(names, list) and names:
        return str(names[0])
    if isinstance(names, str):
        return names
    return ""


def rebuild_scene_visibility(
    scene_root: Path,
    *,
    max_distance: float,
    min_visible_ratio: float,
    min_visible_points: int,
) -> dict[str, Any]:
    start = time.time()
    scene_id = scene_root.name
    raw_dir = scene_root / "raw"
    pcd_path = scene_root / "conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz"
    vis_path = scene_root / "conceptgraph/indices/visibility_index.pkl"
    if not pcd_path.exists():
        raise FileNotFoundError(f"{scene_id}: missing object pkl: {pcd_path}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"{scene_id}: missing raw dir: {raw_dir}")

    rgb_paths = sorted(raw_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9]-rgb.png"))
    depth_paths = sorted(raw_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9]-depth.png"))
    pose_paths = sorted(raw_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].txt"))
    if not rgb_paths or not depth_paths or not pose_paths:
        raise FileNotFoundError(f"{scene_id}: missing RGB/depth/pose files in {raw_dir}")
    rgb_ids = [_frame_id(path) for path in rgb_paths]
    depth_ids = [_frame_id(path) for path in depth_paths]
    pose_ids = [path.stem for path in pose_paths]
    if rgb_ids != depth_ids or rgb_ids != pose_ids:
        raise ValueError(f"{scene_id}: RGB/depth/pose frame ids do not match")

    with gzip.open(pcd_path, "rb") as f:
        objects = pickle.load(f)["objects"]
    poses = [np.loadtxt(path).astype(np.float64) for path in pose_paths]
    intrinsic = np.loadtxt(raw_dir / "intrinsic_color.txt").astype(np.float64)
    image_width, image_height = _read_image_size(rgb_paths[0])

    old_mappings = None
    old_use_depth = None
    if vis_path.exists():
        with vis_path.open("rb") as f:
            old_payload = pickle.load(f)
        old_mappings = sum(len(v) for v in old_payload.get("object_to_views", {}).values())
        old_meta = old_payload.get("metadata")
        if isinstance(old_meta, dict):
            old_use_depth = old_meta.get("use_depth")

    object_to_views, view_to_objects = build_visibility_index(
        objects=objects,
        poses=poses,
        depth_paths=depth_paths,
        intrinsics=intrinsic,
        max_distance=max_distance,
        use_depth=True,
        stride=1,
        img_w=image_width,
        img_h=image_height,
        min_visible_ratio=min_visible_ratio,
        min_visible_points=min_visible_points,
    )

    missing_foreground: list[dict[str, Any]] = []
    unobserved_background: list[dict[str, Any]] = []
    for obj_idx, obj in enumerate(objects):
        if object_to_views.get(obj_idx):
            continue
        entry = {
            "object_index": obj_idx,
            "label": _object_label(obj),
            "reason": "no depth-visible points in kept frames",
        }
        if int(obj.get("is_background") or 0):
            unobserved_background.append(entry)
        else:
            missing_foreground.append(entry)
    if missing_foreground:
        raise ValueError(
            f"{scene_id}: foreground objects have no depth-visible views: "
            f"{missing_foreground[:5]}"
        )

    new_mappings = sum(len(v) for v in object_to_views.values())
    save_visibility_index(
        object_to_views=object_to_views,
        view_to_objects=view_to_objects,
        output_path=vis_path,
        metadata={
            "scene_path": "conceptgraph",
            "pcd_file": "conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz",
            "stride": 1,
            "max_distance": max_distance,
            "use_depth": True,
            "visibility_kind": "depth_occlusion_point_visibility",
            "num_objects": len(objects),
            "num_views": len(poses),
            "num_object_mappings": new_mappings,
            "num_view_mappings": sum(len(v) for v in view_to_objects.values()),
            "num_projection_fallback_objects": 0,
            "projection_fallback_objects": [],
            "num_unobserved_background_objects": len(unobserved_background),
            "unobserved_background_objects": unobserved_background,
            "build_time": time.time() - start,
            "rebuilt_from_use_depth": old_use_depth,
            "rebuilt_from_num_object_mappings": old_mappings,
        },
    )
    return {
        "scene_id": scene_id,
        "num_objects": len(objects),
        "num_views": len(poses),
        "old_use_depth": old_use_depth,
        "old_mappings": old_mappings,
        "new_mappings": new_mappings,
        "unobserved_background": len(unobserved_background),
        "elapsed_sec": round(time.time() - start, 3),
    }


def _worker(payload: tuple[str, str, float, float, int]) -> dict[str, Any]:
    scene_root, scene_id, max_distance, min_visible_ratio, min_visible_points = payload
    return rebuild_scene_visibility(
        Path(scene_root) / scene_id,
        max_distance=max_distance,
        min_visible_ratio=min_visible_ratio,
        min_visible_points=min_visible_points,
    )


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    scene_ids = load_scene_ids(args.data_root, args.scenes, args.scene_list)
    if not scene_ids:
        raise ValueError(f"no scenes found under {args.data_root}")
    payloads = [
        (
            str(args.data_root),
            scene_id,
            args.max_distance,
            args.min_visible_ratio,
            args.min_visible_points,
        )
        for scene_id in scene_ids
    ]
    results: list[dict[str, Any]] = []
    if args.workers == 1:
        iterator = (_worker(payload) for payload in payloads)
        for result in tqdm(iterator, total=len(payloads), desc="Rebuilding scenes"):
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, payload): payload[1] for payload in payloads}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Rebuilding scenes",
            ):
                results.append(future.result())
    results.sort(key=lambda item: item["scene_id"])
    summary = {
        "data_root": str(args.data_root),
        "num_scenes": len(results),
        "total_old_mappings": sum(item["old_mappings"] or 0 for item in results),
        "total_new_mappings": sum(item["new_mappings"] for item in results),
        "scenes": results,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
