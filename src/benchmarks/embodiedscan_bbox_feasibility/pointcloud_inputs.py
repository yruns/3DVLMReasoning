from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import backproject_depth, transform_points
from .models import DetectorInputRecord, EmbodiedScanTarget, FailureTag
from .observations import centered_frame_window


RECON_CONDITIONS = {"single_frame_recon", "multi_frame_recon"}
SCANNET_CONDITIONS = {"scannet_pose_crop", "scannet_full"}
SUPPORTED_CONDITIONS = RECON_CONDITIONS | SCANNET_CONDITIONS


def find_scannet_mesh_path(scannet_root: str | Path, scene_id: str) -> Path:
    root = Path(scannet_root)
    candidates = [
        root / "scans" / scene_id / f"{scene_id}_vh_clean_2.ply",
        root / "scans" / scene_id / f"{scene_id}_vh_clean.ply",
        root / "scans_test" / scene_id / f"{scene_id}_vh_clean_2.ply",
        root / "scans_test" / scene_id / f"{scene_id}_vh_clean.ply",
        root / scene_id / f"{scene_id}_vh_clean_2.ply",
        root / scene_id / f"{scene_id}_vh_clean.ply",
        root / f"{scene_id}_vh_clean_2.ply",
        root / f"{scene_id}_vh_clean.ply",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing ScanNet mesh for {scene_id}; checked {', '.join(str(p) for p in candidates)}"
    )


def materialize_detector_input(
    *,
    target: EmbodiedScanTarget,
    condition: str,
    scene_data_root: str | Path,
    output_dir: str | Path,
    scannet_root: str | Path | None = None,
    frame_ids: list[int] | None = None,
    multi_frame_size: int = 5,
    crop_padding: float = 1.5,
    depth_scale: float = 1000.0,
    max_points: int | None = None,
) -> DetectorInputRecord:
    if condition not in SUPPORTED_CONDITIONS:
        raise ValueError(f"Unsupported detector input condition: {condition}")

    scene_root = Path(scene_data_root) / target.scene_id
    try:
        if condition in RECON_CONDITIONS:
            selected_frame_ids = frame_ids or _select_recon_frame_ids(
                scene_root=scene_root,
                target_id=target.target_id,
                window_size=1 if condition == "single_frame_recon" else multi_frame_size,
            )
            points = reconstruct_frames_to_points(
                scene_root=scene_root,
                frame_ids=selected_frame_ids,
                depth_scale=depth_scale,
            )
            if len(points) == 0:
                raise ValueError("Reconstructed point cloud contains no valid depth points")
        elif condition == "scannet_full":
            if scannet_root is None:
                raise FileNotFoundError("scannet_root is required for scannet_full")
            selected_frame_ids = []
            points = load_scannet_mesh_points(
                find_scannet_mesh_path(scannet_root, target.scene_id)
            )
        else:
            if scannet_root is None:
                raise FileNotFoundError("scannet_root is required for scannet_pose_crop")
            selected_frame_ids = frame_ids or _select_recon_frame_ids(
                scene_root=scene_root,
                target_id=target.target_id,
                window_size=multi_frame_size,
            )
            mesh_points = load_scannet_mesh_points(
                find_scannet_mesh_path(scannet_root, target.scene_id)
            )
            centers = load_camera_centers(scene_root, selected_frame_ids)
            points = crop_points_by_pose_bounds(
                mesh_points,
                centers=centers,
                padding=crop_padding,
            )

        points = _downsample_points(points, max_points=max_points)
        pointcloud_path = _pointcloud_output_path(
            output_dir=output_dir,
            condition=condition,
            scene_id=target.scene_id,
            target_id=target.target_id,
        )
        write_xyz_ply(pointcloud_path, points)
        return DetectorInputRecord(
            scan_id=target.scan_id,
            scene_id=target.scene_id,
            target_id=target.target_id,
            input_condition=condition,
            pointcloud_path=str(pointcloud_path),
            frame_ids=selected_frame_ids,
            metadata={
                "num_points": int(len(points)),
                "scene_root": str(scene_root),
            },
        )
    except (FileNotFoundError, ImportError, ValueError) as exc:
        return DetectorInputRecord(
            scan_id=target.scan_id,
            scene_id=target.scene_id,
            target_id=target.target_id,
            input_condition=condition,
            pointcloud_path=None,
            frame_ids=frame_ids or [],
            failure_tag=FailureTag.INPUT_BLOCKED,
            metadata={
                "reason": str(exc),
                "scene_root": str(scene_root),
            },
        )


def reconstruct_frames_to_points(
    *,
    scene_root: str | Path,
    frame_ids: list[int],
    depth_scale: float = 1000.0,
) -> np.ndarray:
    if not frame_ids:
        raise FileNotFoundError("No frame ids available for RGB-D reconstruction")
    root = Path(scene_root)
    intrinsic = load_intrinsic_matrix(root)

    clouds: list[np.ndarray] = []
    for frame_id in frame_ids:
        depth_path = resolve_depth_path(root, frame_id)
        pose = load_pose_matrix(resolve_pose_path(root, frame_id))
        depth = np.asarray(Image.open(depth_path))
        points_camera = backproject_depth(
            depth,
            intrinsic,
            depth_scale=depth_scale,
        )
        if len(points_camera):
            clouds.append(transform_points(points_camera, pose))
    if not clouds:
        return np.empty((0, 3), dtype=np.float32)
    return np.vstack(clouds).astype(np.float32)


def resolve_depth_path(scene_root: str | Path, frame_id: int) -> Path:
    root = Path(scene_root)
    stem = f"{int(frame_id):06d}"
    candidates = [
        root / "raw" / f"{stem}-depth.png",
        root / "raw" / f"{stem}_depth.png",
        root / "depth" / f"depth{stem}.png",
        root / "depth" / f"{stem}.png",
        root / "results" / f"depth{stem}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing depth image for frame {frame_id}; checked "
        f"{', '.join(str(p) for p in candidates)}"
    )


def resolve_pose_path(scene_root: str | Path, frame_id: int) -> Path:
    root = Path(scene_root)
    stem = f"{int(frame_id):06d}"
    candidates = [
        root / "raw" / f"{stem}.txt",
        root / "pose" / f"frame{stem}.txt",
        root / "pose" / f"{stem}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing pose for frame {frame_id}; checked {', '.join(str(p) for p in candidates)}"
    )


def load_intrinsic_matrix(scene_root: str | Path) -> np.ndarray:
    root = Path(scene_root)
    candidates = [
        root / "raw" / "intrinsic_depth.txt",
        root / "raw" / "intrinsic_color.txt",
        root / "intrinsic" / "intrinsic_depth.txt",
        root / "intrinsic" / "intrinsic_color.txt",
        root / "intrinsic_depth.txt",
        root / "intrinsic_color.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            mat = np.loadtxt(candidate, dtype=np.float64)
            if mat.ndim != 2 or mat.shape[0] < 3 or mat.shape[1] < 3:
                raise ValueError(f"Intrinsic matrix must be at least 3x3: {candidate}")
            return mat[:3, :3]
    raise FileNotFoundError(
        f"Missing camera intrinsics; checked {', '.join(str(p) for p in candidates)}"
    )


def load_pose_matrix(path: str | Path) -> np.ndarray:
    pose = np.loadtxt(path, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(pose).all():
        raise ValueError(f"Pose matrix contains non-finite values: {path}")
    return pose


def load_camera_centers(scene_root: str | Path, frame_ids: list[int]) -> np.ndarray:
    if not frame_ids:
        raise FileNotFoundError("No frame ids available for ScanNet pose crop")
    centers = []
    for frame_id in frame_ids:
        pose = load_pose_matrix(resolve_pose_path(scene_root, frame_id))
        centers.append(pose[:3, 3])
    return np.asarray(centers, dtype=np.float64)


def load_scannet_mesh_points(mesh_path: str | Path) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d is required to read ScanNet mesh point clouds") from exc

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    points = np.asarray(mesh.vertices, dtype=np.float64)
    if len(points) == 0:
        point_cloud = o3d.io.read_point_cloud(str(mesh_path))
        points = np.asarray(point_cloud.points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError(f"ScanNet mesh has no vertices: {mesh_path}")
    if not np.isfinite(points).all():
        raise ValueError(f"ScanNet mesh contains non-finite vertices: {mesh_path}")
    return points[:, :3].astype(np.float32)


def crop_points_by_pose_bounds(
    points: np.ndarray,
    *,
    centers: np.ndarray,
    padding: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    ctr = np.asarray(centers, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must have shape (N, >=3)")
    if ctr.ndim != 2 or ctr.shape[1] != 3 or len(ctr) == 0:
        raise ValueError("centers must have shape (N, 3) with N > 0")
    pad = float(padding)
    if not np.isfinite(pad) or pad < 0.0:
        raise ValueError("padding must be finite and non-negative")

    lo = ctr.min(axis=0) - pad
    hi = ctr.max(axis=0) + pad
    mask = np.all((pts[:, :3] >= lo) & (pts[:, :3] <= hi), axis=1)
    cropped = pts[mask, :3]
    if len(cropped) == 0:
        raise ValueError("ScanNet pose crop contains no points")
    return cropped.astype(np.float32)


def write_xyz_ply(path: str | Path, points: np.ndarray) -> None:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must have shape (N, >=3)")
    if not np.isfinite(pts[:, :3]).all():
        raise ValueError("points must contain only finite coordinates")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(pts)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for x, y, z in pts[:, :3]:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def _select_recon_frame_ids(
    *,
    scene_root: Path,
    target_id: int,
    window_size: int,
) -> list[int]:
    visible = _visibility_frame_ids(scene_root, target_id)
    available = _available_pose_frame_ids(scene_root)
    if visible:
        center = visible[0]
        return centered_frame_window(center, available or visible, window_size)
    if available:
        center = available[len(available) // 2]
        return centered_frame_window(center, available, window_size)
    raise FileNotFoundError(f"No pose frames found under {scene_root}")


def _visibility_frame_ids(scene_root: Path, target_id: int) -> list[int]:
    visibility_path = scene_root / "conceptgraph" / "indices" / "visibility_index.pkl"
    if not visibility_path.exists():
        visibility_path = scene_root / "indices" / "visibility_index.pkl"
    if not visibility_path.exists():
        return []
    with visibility_path.open("rb") as handle:
        visibility = pickle.load(handle)
    object_to_views = visibility.get("object_to_views", {})
    views = object_to_views.get(target_id, [])
    return [int(view_id) for view_id, _score in views]


def _available_pose_frame_ids(scene_root: Path) -> list[int]:
    ids: set[int] = set()
    for folder, pattern in (("raw", "*.txt"), ("pose", "*.txt")):
        for path in (scene_root / folder).glob(pattern):
            if path.name.startswith(("intrinsic_", "extrinsic_")):
                continue
            match = re.search(r"(\d+)", path.stem)
            if match:
                ids.add(int(match.group(1)))
    return sorted(ids)


def _downsample_points(points: np.ndarray, *, max_points: int | None) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if max_points is None or len(pts) <= max_points:
        return pts
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    indices = np.linspace(0, len(pts) - 1, num=max_points, dtype=np.int64)
    return pts[indices]


def _pointcloud_output_path(
    *,
    output_dir: str | Path,
    condition: str,
    scene_id: str,
    target_id: int,
) -> Path:
    return Path(output_dir) / condition / f"{scene_id}_target{target_id}.ply"
