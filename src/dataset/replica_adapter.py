"""Replica dataset adapter.

Implements the DatasetAdapter interface for the Replica dataset, which contains
high-quality reconstructed indoor scenes with RGB-D frames and camera poses.

Replica dataset structure:
    replica/
    ├── room0/
    │   ├── results/
    │   │   ├── frame000000.jpg
    │   │   ├── depth000000.png
    │   │   └── ...
    │   ├── traj.txt
    │   └── mesh.ply (optional)
    ├── room1/
    └── ...
"""

import glob
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from natsort import natsorted

from .base import (
    CameraIntrinsics,
    CoordinateSystem,
    DatasetAdapter,
    FrameData,
    SceneMetadata,
)
from .registry import register_adapter
from .replica_constants import REPLICA_CLASSES, REPLICA_SCENE_IDS

# Default camera intrinsics for Replica dataset (from iMAP paper)
DEFAULT_REPLICA_INTRINSICS = CameraIntrinsics(
    fx=600.0,
    fy=600.0,
    cx=599.5,
    cy=339.5,
    width=1200,
    height=680,
)

# Depth scale: Replica PNG depth is in millimeters, divide by 1000 for meters
REPLICA_DEPTH_SCALE = 6553.5  # From NICE-SLAM config


@register_adapter("replica", aliases=["replica-v1", "replica-imap"])
class ReplicaAdapter(DatasetAdapter):
    """Dataset adapter for the Replica dataset.

    Replica is a high-quality 3D reconstruction dataset with photorealistic
    indoor scenes. This adapter supports the iMAP/NICE-SLAM version of the
    dataset with pre-rendered RGB-D frames.

    Attributes:
        intrinsics: Camera intrinsic parameters (same for all frames)
        depth_scale: Scale factor for converting depth to meters
    """

    def __init__(
        self,
        data_root: str | Path,
        intrinsics: CameraIntrinsics | None = None,
        depth_scale: float = REPLICA_DEPTH_SCALE,
        **kwargs,
    ):
        """Initialize the Replica adapter.

        Args:
            data_root: Root directory containing Replica scenes
            intrinsics: Camera intrinsics (uses default if not specified)
            depth_scale: Scale factor for depth (PNG value / scale = meters)
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(data_root, **kwargs)
        self.intrinsics = intrinsics or DEFAULT_REPLICA_INTRINSICS
        self.depth_scale = depth_scale

        # Cache for scene metadata
        self._scene_cache: dict[str, SceneMetadata] = {}

    @property
    def dataset_name(self) -> str:
        """Return the canonical dataset name."""
        return "replica"

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """Return the native coordinate system (OpenGL convention)."""
        return CoordinateSystem.OPENGL

    def get_scene_ids(self) -> list[str]:
        """Return all available scene IDs.

        Returns:
            List of scene IDs found in the data root
        """
        scene_ids = []
        for scene_dir in self.data_root.iterdir():
            if scene_dir.is_dir():
                # Check if it has the expected structure
                results_dir = scene_dir / "results"
                traj_file = scene_dir / "traj.txt"
                if results_dir.exists() and traj_file.exists():
                    scene_ids.append(scene_dir.name)
        return sorted(scene_ids)

    def load_scene_metadata(self, scene_id: str) -> SceneMetadata:
        """Load metadata for a scene.

        Args:
            scene_id: Scene identifier

        Returns:
            SceneMetadata with scene information

        Raises:
            ValueError: If scene_id is not found
        """
        if scene_id in self._scene_cache:
            return self._scene_cache[scene_id]

        scene_path = self.data_root / scene_id
        if not scene_path.exists():
            available = ", ".join(self.get_scene_ids())
            raise ValueError(f"Scene '{scene_id}' not found. Available: {available}")

        # Count frames
        results_dir = scene_path / "results"
        color_files = sorted(glob.glob(str(results_dir / "frame*.jpg")))
        num_frames = len(color_files)

        # Check for mesh
        mesh_path = scene_path / "mesh.ply"
        has_mesh = mesh_path.exists()

        # Check for semantics
        semantic_path = scene_path / "semantic_instance.ply"
        has_semantics = semantic_path.exists()

        metadata = SceneMetadata(
            scene_id=scene_id,
            dataset_name=self.dataset_name,
            num_frames=num_frames,
            has_depth=True,
            has_poses=True,
            has_mesh=has_mesh,
            has_semantics=has_semantics,
            coordinate_system=self.coordinate_system,
            intrinsics=self.intrinsics,
            extra={
                "class_names": REPLICA_CLASSES,
                "default_scenes": REPLICA_SCENE_IDS,
            },
        )

        self._scene_cache[scene_id] = metadata
        return metadata

    def _load_poses(self, scene_id: str) -> list[np.ndarray]:
        """Load all camera poses from trajectory file.

        Args:
            scene_id: Scene identifier

        Returns:
            List of 4x4 camera-to-world transformation matrices
        """
        traj_path = self.data_root / scene_id / "traj.txt"
        poses = []

        with open(traj_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                values = list(map(float, line.split()))
                pose = np.array(values, dtype=np.float64).reshape(4, 4)
                poses.append(pose)

        return poses

    def _get_frame_paths(self, scene_id: str) -> tuple[list[Path], list[Path]]:
        """Get sorted lists of RGB and depth file paths.

        Args:
            scene_id: Scene identifier

        Returns:
            Tuple of (color_paths, depth_paths)
        """
        results_dir = self.data_root / scene_id / "results"

        color_paths = [
            Path(p) for p in natsorted(glob.glob(str(results_dir / "frame*.jpg")))
        ]
        depth_paths = [
            Path(p) for p in natsorted(glob.glob(str(results_dir / "depth*.png")))
        ]

        if len(color_paths) != len(depth_paths):
            raise ValueError(
                f"Mismatch: {len(color_paths)} color vs {len(depth_paths)} depth"
            )

        return color_paths, depth_paths

    def _load_rgb(self, path: Path) -> np.ndarray:
        """Load RGB image.

        Args:
            path: Path to RGB image

        Returns:
            RGB image as HxWx3 uint8 array
        """
        # cv2.imread loads as BGR, convert to RGB
        img = cv2.imread(str(path))
        if img is None:
            raise OSError(f"Failed to load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_depth(self, path: Path) -> np.ndarray:
        """Load depth image.

        Args:
            path: Path to depth image

        Returns:
            Depth image as HxW float32 array in meters
        """
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise OSError(f"Failed to load depth: {path}")

        # Convert to float32 meters
        depth = depth.astype(np.float32) / self.depth_scale
        return depth

    def iter_frames(
        self,
        scene_id: str,
        stride: int = 1,
        start: int = 0,
        end: int | None = None,
    ) -> Iterator[FrameData]:
        """Iterate over frames in a scene.

        Args:
            scene_id: Scene identifier
            stride: Sample every Nth frame
            start: Starting frame index
            end: Ending frame index (exclusive), None for all frames

        Yields:
            FrameData for each sampled frame
        """
        color_paths, depth_paths = self._get_frame_paths(scene_id)
        poses = self._load_poses(scene_id)

        if end is None:
            end = len(color_paths)

        for idx in range(start, min(end, len(color_paths)), stride):
            yield self._build_frame_data(
                frame_id=idx,
                color_path=color_paths[idx],
                depth_path=depth_paths[idx],
                pose=poses[idx] if idx < len(poses) else None,
            )

    def load_frame(self, scene_id: str, frame_id: int) -> FrameData:
        """Load a specific frame.

        Args:
            scene_id: Scene identifier
            frame_id: Frame index

        Returns:
            FrameData for the specified frame

        Raises:
            ValueError: If frame_id is out of range
        """
        color_paths, depth_paths = self._get_frame_paths(scene_id)
        poses = self._load_poses(scene_id)

        if frame_id < 0 or frame_id >= len(color_paths):
            raise ValueError(f"Frame {frame_id} out of range [0, {len(color_paths)})")

        return self._build_frame_data(
            frame_id=frame_id,
            color_path=color_paths[frame_id],
            depth_path=depth_paths[frame_id],
            pose=poses[frame_id] if frame_id < len(poses) else None,
        )

    def _build_frame_data(
        self,
        frame_id: int,
        color_path: Path,
        depth_path: Path,
        pose: np.ndarray | None,
    ) -> FrameData:
        """Build a FrameData object from file paths.

        Args:
            frame_id: Frame index
            color_path: Path to RGB image
            depth_path: Path to depth image
            pose: Optional 4x4 pose matrix

        Returns:
            FrameData object
        """
        rgb = self._load_rgb(color_path)
        depth = self._load_depth(depth_path)

        return FrameData(
            frame_id=frame_id,
            rgb=rgb,
            depth=depth,
            pose=pose,
            intrinsics=self.intrinsics,
        )

    def load_mesh(self, scene_id: str) -> Path | None:
        """Get path to scene mesh if available.

        Args:
            scene_id: Scene identifier

        Returns:
            Path to mesh.ply or None
        """
        mesh_path = self.data_root / scene_id / "mesh.ply"
        return mesh_path if mesh_path.exists() else None

    def load_semantic_info(self, scene_id: str) -> dict[int, str] | None:
        """Load semantic class mapping.

        Args:
            scene_id: Scene identifier

        Returns:
            Dict mapping class IDs to class names
        """
        # Replica has a fixed set of semantic classes
        return {i: name for i, name in enumerate(REPLICA_CLASSES)}
