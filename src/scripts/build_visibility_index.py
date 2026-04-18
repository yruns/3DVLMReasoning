"""
Offline Visibility Index Builder

Builds a bidirectional object-view visibility index for a scene.
This is a preprocessing step that should be run once per scene.

Usage:
    python -m src.scripts.build_visibility_index \
        --scene_path /path/to/scene \
        --stride 5 \
        --use_depth  # Optional: use depth maps for occlusion detection

Output:
    scene_path/indices/visibility_index.pkl
"""

from __future__ import annotations

import argparse
import gzip
import os
import pickle
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm


def to_dataset_relative_path(path: Path, dataset_root: Path) -> str:
    """Return dataset-root-relative path string."""
    path = path.resolve()
    dataset_root = dataset_root.resolve()
    try:
        return path.relative_to(dataset_root).as_posix()
    except ValueError:
        return Path(os.path.relpath(str(path), str(dataset_root))).as_posix()


def load_poses(traj_file: Path) -> list[np.ndarray]:
    """Load camera poses from trajectory file.

    Supports two formats:
    1. One 4x4 matrix per line (16 values, space-separated)
    2. One 4x4 matrix per 5 lines (frame_id, then 4 rows)
    """
    poses = []
    with open(traj_file) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if not lines:
        return poses

    # Detect format by counting values in first line
    first_values = lines[0].split()

    if len(first_values) == 16:
        # Format 1: Each line is a flattened 4x4 matrix
        for line in lines:
            values = [float(x) for x in line.split()]
            if len(values) == 16:
                pose = np.array(values).reshape(4, 4)
                poses.append(pose)
    else:
        # Format 2: 5 lines per pose (frame_id + 4 rows)
        for i in range(0, len(lines), 5):
            if i + 4 >= len(lines):
                break
            pose = np.zeros((4, 4))
            for j in range(4):
                pose[j] = [float(x) for x in lines[i + 1 + j].split()]
            poses.append(pose)

    return poses


def load_objects(pcd_file: Path) -> list[dict[str, Any]]:
    """Load objects from pkl.gz file."""
    logger.info(f"Loading objects from {pcd_file}")

    with gzip.open(pcd_file, "rb") as f:
        data = pickle.load(f)

    objects = data.get("objects", [])
    logger.info(f"Loaded {len(objects)} objects")

    return objects


def get_object_centroid(obj: dict[str, Any]) -> np.ndarray:
    """Extract object centroid from raw object data."""
    # Try pcd_np first (numpy array of points)
    if "pcd_np" in obj:
        points = obj["pcd_np"]
        if isinstance(points, np.ndarray) and len(points) > 0:
            return points.mean(axis=0)

    # Try bbox_np (bounding box corners)
    if "bbox_np" in obj:
        bbox = obj["bbox_np"]
        if isinstance(bbox, np.ndarray) and len(bbox) > 0:
            return bbox.mean(axis=0)

    # Try Open3D pcd object
    if "pcd" in obj:
        pcd = obj["pcd"]
        if hasattr(pcd, "get_center"):
            center = pcd.get_center()
            return np.array(center)
        elif hasattr(pcd, "points"):
            points = np.asarray(pcd.points)
            if len(points) > 0:
                return points.mean(axis=0)

    # Try Open3D bbox object
    if "bbox" in obj:
        bbox = obj["bbox"]
        if hasattr(bbox, "get_center"):
            return np.array(bbox.get_center())

    return np.zeros(3)


def compute_geometric_visibility(
    centroid: np.ndarray,
    pose: np.ndarray,
    max_distance: float = 5.0,
) -> tuple[float, float]:
    """Compute geometric visibility score.

    Args:
        centroid: Object centroid [x, y, z]
        pose: Camera pose matrix 4x4
        max_distance: Maximum viewing distance

    Returns:
        (distance_score, angle_score)
    """
    # Camera position
    cam_pos = pose[:3, 3]

    # Distance
    distance = np.linalg.norm(centroid - cam_pos)
    if distance > max_distance:
        return 0.0, 0.0

    dist_score = max(0, 1 - distance / max_distance)

    # Viewing angle
    view_dir = centroid - cam_pos
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)

    # Camera forward (-Z axis)
    cam_forward = -pose[:3, 2]
    angle_score = max(0, np.dot(view_dir, cam_forward))

    return dist_score, angle_score


def check_depth_visibility(
    centroid: np.ndarray,
    pose: np.ndarray,
    depth_path: Path,
    intrinsics: np.ndarray,
    tolerance: float = 0.3,
) -> float:
    """Check if object is visible using depth map (not occluded).

    Args:
        centroid: Object centroid [x, y, z]
        pose: Camera pose matrix 4x4
        depth_path: Path to depth image
        intrinsics: Camera intrinsic matrix 3x3
        tolerance: Depth tolerance in meters

    Returns:
        Occlusion ratio (0 = fully visible, 1 = fully occluded)
    """
    if not depth_path.exists():
        return 0.0  # Assume visible if no depth

    try:
        # Load depth
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            return 0.0

        # Convert to meters (assuming mm)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0

        # Transform centroid to camera frame
        pose_inv = np.linalg.inv(pose)
        centroid_h = np.append(centroid, 1.0)
        centroid_cam = pose_inv @ centroid_h

        if centroid_cam[2] <= 0:
            return 1.0  # Behind camera

        # Project to image
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        u = int(fx * centroid_cam[0] / centroid_cam[2] + cx)
        v = int(fy * centroid_cam[1] / centroid_cam[2] + cy)

        # Check bounds
        h, w = depth.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return 1.0  # Outside image

        # Check depth
        measured_depth = depth[v, u]
        expected_depth = centroid_cam[2]

        if measured_depth < 0.1:  # Invalid depth
            return 0.0

        # Object is occluded if measured depth is significantly less than expected
        if measured_depth < expected_depth - tolerance:
            return 1.0

        return 0.0

    except Exception as e:
        logger.warning(f"Depth check failed: {e}")
        return 0.0


def _project_points(
    pts_3d: np.ndarray,
    pose: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D world points to 2D pixel coordinates.

    Args:
        pts_3d: (N, 3) world-frame points
        pose: 4x4 cam-to-world matrix
        K: 3x3 intrinsic matrix
        img_w, img_h: image dimensions

    Returns:
        u, v: (M,) pixel coords of valid points
        z_cam: (M,) depth in camera frame
        valid_mask: (N,) bool mask — True for points in front of camera
                    AND inside image bounds
    """
    w2c = np.linalg.inv(pose)
    pts_h = np.hstack([pts_3d, np.ones((len(pts_3d), 1))])  # (N, 4)
    pts_cam = (w2c @ pts_h.T).T  # (N, 4)

    z = pts_cam[:, 2]
    in_front = z > 0.05

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.full(len(pts_3d), -1.0)
    v = np.full(len(pts_3d), -1.0)
    u[in_front] = fx * pts_cam[in_front, 0] / z[in_front] + cx
    v[in_front] = fy * pts_cam[in_front, 1] / z[in_front] + cy

    in_bounds = in_front & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    return u, v, z, in_bounds


def build_visibility_index(
    objects: list[dict[str, Any]],
    poses: list[np.ndarray],
    depth_paths: list[Path] | None = None,
    intrinsics: np.ndarray | None = None,
    max_distance: float = 5.0,
    use_depth: bool = True,
    stride: int = 1,
    img_w: int = 1296,
    img_h: int = 968,
    depth_scale: float = 1000.0,
    depth_tolerance: float = 0.15,
    min_visible_ratio: float = 0.03,
    min_visible_points: int = 5,
) -> tuple[dict[int, list[tuple[int, float]]], dict[int, list[tuple[int, float]]]]:
    """Build bidirectional visibility index via 3D point-cloud projection.

    For every (object, view) pair the algorithm:
      1. Projects the object's 3D point cloud into the camera frame.
      2. Discards points behind the camera (z <= 0).
      3. Discards points outside image bounds.
      4. (Optional) Checks depth-map occlusion — a projected point is
         considered occluded when the measured depth is significantly
         less than the point's camera-frame depth.
      5. Computes a visibility score = visible_ratio * coverage_score,
         where visible_ratio = n_visible / n_total and coverage_score
         reflects how much of the image the projected bbox occupies
         (penalising edge-clipped projections).

    Args:
        objects: Object list from conceptgraph pkl (each has 'pcd_np').
        poses: Per-frame cam-to-world 4x4 matrices.
        depth_paths: Per-frame depth image paths (same order as poses).
        intrinsics: 3x3 or 4x4 camera intrinsic matrix.
        max_distance: Ignore objects whose centroid is farther than this.
        use_depth: If True *and* depth_paths is provided, do occlusion
                   filtering with the depth map.
        stride: Kept for API compat (unused — caller slices poses).
        img_w, img_h: RGB image dimensions (pixels).
        depth_scale: Divisor to convert uint16 depth to metres.
        depth_tolerance: Depth tolerance in metres for occlusion check.
        min_visible_ratio: Minimum fraction of points visible to count.
        min_visible_points: Minimum absolute visible point count.

    Returns:
        (object_to_views, view_to_objects) — each maps id to
        [(other_id, score), …] sorted by score descending.
    """
    if intrinsics is None:
        raise ValueError("intrinsics is required for projection-based visibility")

    K = intrinsics[:3, :3] if intrinsics.shape[0] >= 4 else intrinsics
    img_area = img_w * img_h

    do_depth = use_depth and depth_paths is not None and len(depth_paths) > 0
    logger.info(
        f"Building projection-based visibility index: "
        f"{len(objects)} objects × {len(poses)} views, "
        f"depth_occlusion={'ON' if do_depth else 'OFF'}"
    )

    # Pre-extract per-object 3D points and centroids
    obj_points: list[np.ndarray | None] = []
    obj_centroids: list[np.ndarray | None] = []
    for obj in objects:
        pts = obj.get("pcd_np")
        if pts is not None and len(pts) > 0:
            pts = np.asarray(pts, dtype=np.float64)
            if pts.ndim == 1:
                pts = pts.reshape(-1, 3)
            obj_points.append(pts[:, :3])
            obj_centroids.append(pts[:, :3].mean(axis=0))
        else:
            obj_points.append(None)
            obj_centroids.append(None)

    object_to_views: dict[int, list[tuple[int, float]]] = {}
    view_to_objects: dict[int, list[tuple[int, float]]] = {}

    # Pre-compute world-to-camera and camera positions
    w2c_list = []
    cam_positions = []
    for pose in poses:
        w2c_list.append(np.linalg.inv(pose))
        cam_positions.append(pose[:3, 3])

    for obj_idx in tqdm(range(len(objects)), desc="Building visibility index"):
        pts = obj_points[obj_idx]
        centroid = obj_centroids[obj_idx]
        if pts is None or centroid is None:
            continue

        n_total = len(pts)
        pts_h = np.hstack([pts, np.ones((n_total, 1))])  # (N, 4)
        scores: list[tuple[int, float]] = []

        for view_id in range(len(poses)):
            # Quick distance cull
            cam_pos = cam_positions[view_id]
            dist = np.linalg.norm(centroid - cam_pos)
            if dist > max_distance:
                continue

            # Project to camera frame
            w2c = w2c_list[view_id]
            pts_cam = (w2c @ pts_h.T).T  # (N, 4)
            z = pts_cam[:, 2]

            in_front = z > 0.05
            if in_front.sum() < min_visible_points:
                continue

            # Project to pixel coords
            z_valid = z[in_front]
            u = K[0, 0] * pts_cam[in_front, 0] / z_valid + K[0, 2]
            v = K[1, 1] * pts_cam[in_front, 1] / z_valid + K[1, 2]

            in_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
            n_in_bounds = int(in_bounds.sum())
            if n_in_bounds < min_visible_points:
                continue

            # Depth occlusion check
            n_visible = n_in_bounds
            if do_depth and view_id < len(depth_paths):
                depth_map = cv2.imread(str(depth_paths[view_id]), cv2.IMREAD_UNCHANGED)
                if depth_map is not None:
                    depth_m = depth_map.astype(np.float32) / depth_scale
                    dh, dw = depth_m.shape[:2]
                    # Scale projection coords from RGB resolution to depth resolution
                    scale_u = dw / img_w
                    scale_v = dh / img_h
                    u_d = np.clip((u[in_bounds] * scale_u).astype(int), 0, dw - 1)
                    v_d = np.clip((v[in_bounds] * scale_v).astype(int), 0, dh - 1)
                    z_proj = z_valid[in_bounds]
                    measured = depth_m[v_d, u_d]
                    # Point is visible if depth is valid and not occluded
                    valid_depth = measured > 0.1
                    not_occluded = measured >= (z_proj - depth_tolerance)
                    visible_mask = valid_depth & not_occluded
                    n_visible = int(visible_mask.sum())

            if n_visible < min_visible_points:
                continue

            visible_ratio = n_visible / n_total
            if visible_ratio < min_visible_ratio:
                continue

            # Coverage score: projected bbox area / image area (capped)
            u_vis = u[in_bounds]
            v_vis = v[in_bounds]
            bbox_area = (u_vis.max() - u_vis.min()) * (v_vis.max() - v_vis.min())
            coverage = min(1.0, bbox_area / (img_area * 0.3))

            # Edge-clip penalty
            margin = 10
            is_clipped = (
                u_vis.min() < margin
                or v_vis.min() < margin
                or u_vis.max() > img_w - margin
                or v_vis.max() > img_h - margin
            )
            clip_penalty = 0.2 if is_clipped else 0.0

            # Combined score
            score = 0.5 * visible_ratio + 0.3 * max(0, coverage - clip_penalty) + 0.2 * (1.0 - dist / max_distance)

            scores.append((view_id, float(score)))
            view_to_objects.setdefault(view_id, []).append((obj_idx, float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)
        object_to_views[obj_idx] = scores

    # Sort reverse index
    for view_id in view_to_objects:
        view_to_objects[view_id].sort(key=lambda x: x[1], reverse=True)

    total_mappings = sum(len(v) for v in object_to_views.values())
    logger.success(
        f"Built projection-based index: {len(object_to_views)} objects, "
        f"{len(view_to_objects)} views, {total_mappings} mappings"
    )

    return object_to_views, view_to_objects


def save_visibility_index(
    object_to_views: dict[int, list[tuple[int, float]]],
    view_to_objects: dict[int, list[tuple[int, float]]],
    output_path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save bidirectional visibility index to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "object_to_views": object_to_views,  # object_id -> [(view_id, score), ...]
        "view_to_objects": view_to_objects,  # view_id -> [(object_id, score), ...]
        "metadata": metadata or {},
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    logger.success(f"Saved bidirectional visibility index to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build offline visibility index")
    parser.add_argument("--scene_path", type=str, required=True, help="Path to scene")
    parser.add_argument(
        "--pcd_file", type=str, default=None, help="Path to pkl.gz file"
    )
    parser.add_argument("--stride", type=int, default=5, help="Frame stride")
    parser.add_argument(
        "--max_distance", type=float, default=5.0, help="Max viewing distance"
    )
    parser.add_argument(
        "--use_depth", action="store_true", help="Use depth for occlusion"
    )
    parser.add_argument("--output", type=str, default=None, help="Output path")

    args = parser.parse_args()

    scene_path = Path(args.scene_path)

    # Find PCD file
    if args.pcd_file:
        pcd_file = Path(args.pcd_file)
    else:
        pcd_dir = scene_path / "pcd_saves"
        pcd_files = list(pcd_dir.glob("*ram*_post.pkl.gz"))
        if not pcd_files:
            pcd_files = list(pcd_dir.glob("*_post.pkl.gz"))
        if not pcd_files:
            raise FileNotFoundError(f"No pkl.gz files found in {pcd_dir}")
        pcd_file = pcd_files[0]

    logger.info(f"Using PCD file: {pcd_file}")

    # Load objects
    objects = load_objects(pcd_file)

    # Load poses — prefer sibling raw/ directory
    raw_dir = scene_path.parent / "raw"
    traj_file = raw_dir / "traj.txt" if (raw_dir / "traj.txt").exists() else scene_path / "traj.txt"
    if not traj_file.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_file}")

    all_poses = load_poses(traj_file)
    poses = all_poses[:: args.stride]
    logger.info(f"Loaded {len(poses)} poses (stride={args.stride})")

    # Load intrinsics — try scene-level files first, then raw/
    intrinsics = None
    for candidate in [
        scene_path / "intrinsic_color.txt",
        raw_dir / "intrinsic_color.txt",
        scene_path / "intrinsic.txt",
    ]:
        if candidate.exists():
            intrinsics = np.loadtxt(candidate).astype(np.float32)
            logger.info(f"Loaded intrinsics from {candidate}")
            break
    if intrinsics is None:
        raise FileNotFoundError(
            f"No intrinsic file found for {scene_path}. "
            "Expected intrinsic_color.txt in scene or raw directory."
        )

    # Detect image dimensions from first RGB
    img_w, img_h = 1296, 968  # default
    rgb_candidates = sorted((raw_dir if raw_dir.exists() else scene_path).glob("*-rgb.*"))
    if rgb_candidates:
        _img = cv2.imread(str(rgb_candidates[0]))
        if _img is not None:
            img_h, img_w = _img.shape[:2]
            logger.info(f"Detected image size: {img_w}x{img_h}")

    # Depth paths — collect from raw/ directory
    depth_paths: list[Path] | None = None
    if args.use_depth:
        depth_dir = raw_dir if raw_dir.exists() else scene_path
        depth_files = sorted(depth_dir.glob("*-depth.png"))
        if not depth_files:
            depth_files = sorted(depth_dir.glob("depth*.png"))
        if depth_files:
            depth_paths = depth_files[:: args.stride]
            logger.info(f"Found {len(depth_paths)} depth images for occlusion check")
        else:
            logger.warning("No depth images found, disabling occlusion check")

    # Build bidirectional index
    start_time = time.time()

    object_to_views, view_to_objects = build_visibility_index(
        objects=objects,
        poses=poses,
        depth_paths=depth_paths,
        intrinsics=intrinsics,
        max_distance=args.max_distance,
        use_depth=args.use_depth,
        stride=args.stride,
        img_w=img_w,
        img_h=img_h,
    )

    elapsed = time.time() - start_time
    logger.info(f"Index built in {elapsed:.2f} seconds")

    # Save
    output_path = (
        Path(args.output)
        if args.output
        else scene_path / "indices" / "visibility_index.pkl"
    )

    dataset_root = scene_path.parent
    metadata = {
        "scene_path": to_dataset_relative_path(scene_path, dataset_root),
        "pcd_file": to_dataset_relative_path(pcd_file, dataset_root),
        "stride": args.stride,
        "max_distance": args.max_distance,
        "use_depth": args.use_depth,
        "num_objects": len(objects),
        "num_views": len(poses),
        "num_object_mappings": sum(len(v) for v in object_to_views.values()),
        "num_view_mappings": sum(len(v) for v in view_to_objects.values()),
        "build_time": elapsed,
    }

    save_visibility_index(object_to_views, view_to_objects, output_path, metadata)


if __name__ == "__main__":
    main()
