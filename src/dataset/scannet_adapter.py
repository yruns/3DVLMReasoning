"""ScanNet dataset adapter.

Implements the DatasetAdapter interface for the ScanNet dataset, which contains
RGB-D video sequences from real-world indoor environments with 3D reconstructions.

ScanNet dataset structure:
    scannet/
    ├── scene0000_00/
    │   ├── color/
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   └── ...
    │   ├── depth/
    │   │   ├── 0.png
    │   │   ├── 1.png
    │   │   └── ...
    │   ├── pose/
    │   │   ├── 0.txt
    │   │   ├── 1.txt
    │   │   └── ...
    │   ├── intrinsic/
    │   │   ├── intrinsic_color.txt
    │   │   └── intrinsic_depth.txt
    │   └── label-filt/ (optional)
    ├── scene0000_01/
    └── ...
"""

import glob
import re
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

# ScanNet semantic label mapping (subset of NYU40 classes)
SCANNET_CLASSES = {
    0: "unannotated",
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    13: "blinds",
    14: "desk",
    15: "shelves",
    16: "curtain",
    17: "dresser",
    18: "pillow",
    19: "mirror",
    20: "floor_mat",
    21: "clothes",
    22: "ceiling",
    23: "books",
    24: "refrigerator",
    25: "television",
    26: "paper",
    27: "towel",
    28: "shower_curtain",
    29: "box",
    30: "whiteboard",
    31: "person",
    32: "nightstand",
    33: "toilet",
    34: "sink",
    35: "lamp",
    36: "bathtub",
    37: "bag",
    38: "otherstructure",
    39: "otherfurniture",
    40: "otherprop",
}

# Default depth scale for ScanNet: PNG depth is in millimeters
SCANNET_DEPTH_SCALE = 1000.0


@register_adapter("scannet", aliases=["scannet-v2"])
class ScanNetAdapter(DatasetAdapter):
    """Dataset adapter for the ScanNet dataset.

    ScanNet is a richly-annotated dataset of 3D reconstructed indoor scenes
    captured using an RGB-D sensor. This adapter supports the standard ScanNet
    v2 format with per-frame intrinsics and poses.

    Attributes:
        depth_scale: Scale factor for converting depth to meters
        use_depth_intrinsics: Whether to use depth camera intrinsics
    """

    def __init__(
        self,
        data_root: str | Path,
        depth_scale: float = SCANNET_DEPTH_SCALE,
        use_depth_intrinsics: bool = False,
        **kwargs,
    ):
        """Initialize the ScanNet adapter.

        Args:
            data_root: Root directory containing ScanNet scenes
            depth_scale: Scale factor for depth (PNG value / scale = meters)
            use_depth_intrinsics: Use depth camera intrinsics instead of color
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(data_root, **kwargs)
        self.depth_scale = depth_scale
        self.use_depth_intrinsics = use_depth_intrinsics

        # Cache for scene metadata and intrinsics
        self._scene_cache: dict[str, SceneMetadata] = {}
        self._intrinsics_cache: dict[str, CameraIntrinsics] = {}

    @property
    def dataset_name(self) -> str:
        """Return the canonical dataset name."""
        return "scannet"

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """Return the native coordinate system."""
        return CoordinateSystem.SCANNET

    def get_scene_ids(self) -> list[str]:
        """Return all available scene IDs.

        Returns:
            List of scene IDs matching pattern sceneXXXX_XX
        """
        scene_ids = []
        pattern = re.compile(r"scene\d{4}_\d{2}")

        for scene_dir in self.data_root.iterdir():
            if scene_dir.is_dir() and pattern.match(scene_dir.name):
                # Verify it has expected structure
                color_dir = scene_dir / "color"
                depth_dir = scene_dir / "depth"
                pose_dir = scene_dir / "pose"
                if color_dir.exists() and depth_dir.exists() and pose_dir.exists():
                    scene_ids.append(scene_dir.name)

        return sorted(scene_ids)

    def _load_intrinsics(self, scene_id: str) -> CameraIntrinsics:
        """Load camera intrinsics for a scene.

        Args:
            scene_id: Scene identifier

        Returns:
            CameraIntrinsics object
        """
        if scene_id in self._intrinsics_cache:
            return self._intrinsics_cache[scene_id]

        intrinsic_dir = self.data_root / scene_id / "intrinsic"
        intrinsic_file = (
            "intrinsic_depth.txt"
            if self.use_depth_intrinsics
            else "intrinsic_color.txt"
        )
        intrinsic_path = intrinsic_dir / intrinsic_file

        if not intrinsic_path.exists():
            raise FileNotFoundError(f"Intrinsic file not found: {intrinsic_path}")

        # Load 4x4 intrinsic matrix (ScanNet format)
        K = np.loadtxt(str(intrinsic_path))
        if K.shape == (4, 4):
            K = K[:3, :3]

        # Get image dimensions from first color image
        color_dir = self.data_root / scene_id / "color"
        first_image = sorted(glob.glob(str(color_dir / "*.jpg")))[0]
        img = cv2.imread(first_image)
        height, width = img.shape[:2]

        intrinsics = CameraIntrinsics(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            width=width,
            height=height,
        )

        self._intrinsics_cache[scene_id] = intrinsics
        return intrinsics

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
            available = ", ".join(self.get_scene_ids()[:10])
            raise ValueError(
                f"Scene '{scene_id}' not found. Available (first 10): {available}"
            )

        # Count frames
        color_dir = scene_path / "color"
        color_files = sorted(glob.glob(str(color_dir / "*.jpg")))
        num_frames = len(color_files)

        # Load intrinsics
        intrinsics = self._load_intrinsics(scene_id)

        # Check for mesh
        mesh_path = scene_path / f"{scene_id}_vh_clean_2.ply"
        has_mesh = mesh_path.exists()

        # Check for semantics
        label_dir = scene_path / "label-filt"
        has_semantics = label_dir.exists() and len(list(label_dir.glob("*.png"))) > 0

        metadata = SceneMetadata(
            scene_id=scene_id,
            dataset_name=self.dataset_name,
            num_frames=num_frames,
            has_depth=True,
            has_poses=True,
            has_mesh=has_mesh,
            has_semantics=has_semantics,
            coordinate_system=self.coordinate_system,
            intrinsics=intrinsics,
            extra={
                "class_names": SCANNET_CLASSES,
                "nyu40_mapping": True,
            },
        )

        self._scene_cache[scene_id] = metadata
        return metadata

    def _get_frame_paths(
        self, scene_id: str
    ) -> tuple[list[Path], list[Path], list[Path]]:
        """Get sorted lists of RGB, depth, and pose file paths.

        Args:
            scene_id: Scene identifier

        Returns:
            Tuple of (color_paths, depth_paths, pose_paths)
        """
        scene_path = self.data_root / scene_id

        color_paths = [
            Path(p) for p in natsorted(glob.glob(str(scene_path / "color" / "*.jpg")))
        ]
        depth_paths = [
            Path(p) for p in natsorted(glob.glob(str(scene_path / "depth" / "*.png")))
        ]
        pose_paths = [
            Path(p) for p in natsorted(glob.glob(str(scene_path / "pose" / "*.txt")))
        ]

        # ScanNet may have mismatched counts due to invalid poses
        min_count = min(len(color_paths), len(depth_paths), len(pose_paths))
        color_paths = color_paths[:min_count]
        depth_paths = depth_paths[:min_count]
        pose_paths = pose_paths[:min_count]

        return color_paths, depth_paths, pose_paths

    def _load_rgb(self, path: Path) -> np.ndarray:
        """Load RGB image.

        Args:
            path: Path to RGB image

        Returns:
            RGB image as HxWx3 uint8 array
        """
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

    def _load_pose(self, path: Path) -> np.ndarray | None:
        """Load camera pose.

        Args:
            path: Path to pose text file

        Returns:
            4x4 camera-to-world transformation matrix, or None if invalid
        """
        pose = np.loadtxt(str(path))

        # Check for invalid pose (contains inf or nan)
        if not np.isfinite(pose).all():
            return None

        return pose.astype(np.float64)

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
            FrameData for each sampled frame (skips frames with invalid poses)
        """
        color_paths, depth_paths, pose_paths = self._get_frame_paths(scene_id)
        intrinsics = self._load_intrinsics(scene_id)

        if end is None:
            end = len(color_paths)

        for idx in range(start, min(end, len(color_paths)), stride):
            pose = self._load_pose(pose_paths[idx])

            # Skip frames with invalid poses
            if pose is None:
                continue

            yield self._build_frame_data(
                frame_id=idx,
                color_path=color_paths[idx],
                depth_path=depth_paths[idx],
                pose=pose,
                intrinsics=intrinsics,
            )

    def load_frame(self, scene_id: str, frame_id: int) -> FrameData:
        """Load a specific frame.

        Args:
            scene_id: Scene identifier
            frame_id: Frame index

        Returns:
            FrameData for the specified frame

        Raises:
            ValueError: If frame_id is out of range or has invalid pose
        """
        color_paths, depth_paths, pose_paths = self._get_frame_paths(scene_id)
        intrinsics = self._load_intrinsics(scene_id)

        if frame_id < 0 or frame_id >= len(color_paths):
            raise ValueError(f"Frame {frame_id} out of range [0, {len(color_paths)})")

        pose = self._load_pose(pose_paths[frame_id])
        if pose is None:
            raise ValueError(f"Frame {frame_id} has invalid pose")

        return self._build_frame_data(
            frame_id=frame_id,
            color_path=color_paths[frame_id],
            depth_path=depth_paths[frame_id],
            pose=pose,
            intrinsics=intrinsics,
        )

    def _build_frame_data(
        self,
        frame_id: int,
        color_path: Path,
        depth_path: Path,
        pose: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> FrameData:
        """Build a FrameData object from file paths.

        Args:
            frame_id: Frame index
            color_path: Path to RGB image
            depth_path: Path to depth image
            pose: 4x4 pose matrix
            intrinsics: Camera intrinsics

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
            intrinsics=intrinsics,
        )

    def load_mesh(self, scene_id: str) -> Path | None:
        """Get path to scene mesh if available.

        Args:
            scene_id: Scene identifier

        Returns:
            Path to mesh or None
        """
        scene_path = self.data_root / scene_id

        # Try different mesh variants
        mesh_variants = [
            f"{scene_id}_vh_clean_2.ply",
            f"{scene_id}_vh_clean.ply",
            f"{scene_id}.ply",
        ]

        for variant in mesh_variants:
            mesh_path = scene_path / variant
            if mesh_path.exists():
                return mesh_path

        return None

    def load_semantic_info(self, scene_id: str) -> dict[int, str] | None:
        """Load semantic class mapping.

        Args:
            scene_id: Scene identifier

        Returns:
            Dict mapping class IDs to class names
        """
        return SCANNET_CLASSES.copy()

    def load_semantic_labels(self, scene_id: str, frame_id: int) -> np.ndarray | None:
        """Load semantic segmentation labels for a frame.

        Args:
            scene_id: Scene identifier
            frame_id: Frame index

        Returns:
            Semantic label mask as HxW uint8 array, or None if not available
        """
        label_path = self.data_root / scene_id / "label-filt" / f"{frame_id}.png"

        if not label_path.exists():
            return None

        labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        return labels
