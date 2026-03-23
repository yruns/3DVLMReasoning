"""Base abstractions for dataset adapters.

This module provides dataset-agnostic interfaces for loading 3D scene data
from various RGB-D datasets (Replica, ScanNet, HM3D, etc.).

The design separates:
- SceneMetadata: Dataset-agnostic scene information
- FrameData: Single frame with RGB, depth, pose, intrinsics
- DatasetAdapter: Abstract interface for loading scenes and frames
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class CoordinateSystem(str, Enum):
    """Standard coordinate system conventions used by different datasets."""

    REPLICA = "replica"  # OpenGL: +X right, +Y up, -Z forward
    SCANNET = "scannet"  # +X right, +Y forward, +Z up
    HABITAT = "habitat"  # +X right, +Y up, -Z forward (same as Replica)
    OPENCV = "opencv"  # +X right, +Y down, +Z forward
    OPENGL = "opengl"  # +X right, +Y up, -Z forward
    CUSTOM = "custom"


@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera intrinsic parameters.

    Attributes:
        fx: Focal length in x (pixels)
        fy: Focal length in y (pixels)
        cx: Principal point x coordinate (pixels)
        cy: Principal point y coordinate (pixels)
        width: Image width in pixels
        height: Image height in pixels
        distortion: Optional distortion coefficients [k1, k2, p1, p2, k3]
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: tuple[float, ...] | None = None

    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 intrinsic matrix K."""
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        return K

    @classmethod
    def from_matrix(
        cls,
        K: np.ndarray,
        width: int,
        height: int,
        distortion: tuple[float, ...] | None = None,
    ) -> "CameraIntrinsics":
        """Create from 3x3 intrinsic matrix K."""
        return cls(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            width=width,
            height=height,
            distortion=distortion,
        )

    def scale(self, scale_x: float, scale_y: float) -> "CameraIntrinsics":
        """Return scaled intrinsics for resized images."""
        new_width = int(self.width * scale_x)
        new_height = int(self.height * scale_y)
        return CameraIntrinsics(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            width=new_width,
            height=new_height,
            distortion=self.distortion,
        )


@dataclass
class SceneMetadata:
    """Dataset-agnostic scene metadata.

    Attributes:
        scene_id: Unique identifier for the scene
        dataset_name: Name of the source dataset (e.g., "replica", "scannet")
        num_frames: Total number of frames in the scene
        has_depth: Whether depth images are available
        has_poses: Whether camera poses are available
        has_mesh: Whether 3D mesh is available
        has_semantics: Whether semantic annotations are available
        coordinate_system: Coordinate system convention
        intrinsics: Camera intrinsic parameters (if uniform across frames)
        scene_bounds: Optional axis-aligned bounding box [(min_x, min_y, min_z), (max_x, max_y, max_z)]
        extra: Additional dataset-specific metadata
    """

    scene_id: str
    dataset_name: str
    num_frames: int
    has_depth: bool = True
    has_poses: bool = True
    has_mesh: bool = False
    has_semantics: bool = False
    coordinate_system: CoordinateSystem = CoordinateSystem.OPENGL
    intrinsics: CameraIntrinsics | None = None
    scene_bounds: tuple[np.ndarray, np.ndarray] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameData:
    """Single frame data with optional fields.

    Attributes:
        frame_id: Frame index within the scene
        rgb: RGB image as HxWx3 uint8 array
        depth: Optional depth image as HxW float32 array (in meters)
        pose: Optional 4x4 camera-to-world transformation matrix
        intrinsics: Optional per-frame camera intrinsics
        timestamp: Optional timestamp in seconds
        semantic_mask: Optional semantic segmentation mask as HxW int32 array
        instance_mask: Optional instance segmentation mask as HxW int32 array
        extra: Additional frame-specific data
    """

    frame_id: int
    rgb: np.ndarray
    depth: np.ndarray | None = None
    pose: np.ndarray | None = None
    intrinsics: CameraIntrinsics | None = None
    timestamp: float | None = None
    semantic_mask: np.ndarray | None = None
    instance_mask: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate frame data shapes and types."""
        if self.rgb is not None:
            if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
                raise ValueError(f"RGB must be HxWx3, got {self.rgb.shape}")

        if self.depth is not None:
            if self.depth.ndim != 2:
                raise ValueError(f"Depth must be HxW, got {self.depth.shape}")
            if self.rgb is not None:
                if self.depth.shape != self.rgb.shape[:2]:
                    raise ValueError(
                        f"Depth shape {self.depth.shape} != RGB shape {self.rgb.shape[:2]}"
                    )

        if self.pose is not None:
            if self.pose.shape != (4, 4):
                raise ValueError(f"Pose must be 4x4, got {self.pose.shape}")

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self.rgb.shape[1]

    def get_camera_position(self) -> np.ndarray | None:
        """Extract camera position from pose matrix."""
        if self.pose is None:
            return None
        return self.pose[:3, 3].copy()

    def get_camera_rotation(self) -> np.ndarray | None:
        """Extract 3x3 rotation matrix from pose."""
        if self.pose is None:
            return None
        return self.pose[:3, :3].copy()


class DatasetAdapter(ABC):
    """Abstract base class for dataset adapters.

    Provides a unified interface for loading 3D scene data from various
    RGB-D datasets. Subclasses implement dataset-specific loading logic.

    Example usage:
        adapter = ReplicaAdapter(data_root="/path/to/replica")
        scene_ids = adapter.get_scene_ids()

        for scene_id in scene_ids:
            metadata = adapter.load_scene_metadata(scene_id)
            for frame in adapter.iter_frames(scene_id, stride=5):
                process_frame(frame)
    """

    def __init__(self, data_root: str | Path, **kwargs):
        """Initialize the adapter.

        Args:
            data_root: Root directory containing the dataset
            **kwargs: Additional dataset-specific configuration
        """
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return the canonical dataset name."""
        pass

    @property
    @abstractmethod
    def coordinate_system(self) -> CoordinateSystem:
        """Return the native coordinate system of this dataset."""
        pass

    @abstractmethod
    def get_scene_ids(self) -> list[str]:
        """Return all available scene IDs.

        Returns:
            List of scene identifiers that can be loaded
        """
        pass

    @abstractmethod
    def load_scene_metadata(self, scene_id: str) -> SceneMetadata:
        """Load metadata for a scene.

        Args:
            scene_id: Scene identifier

        Returns:
            SceneMetadata with scene information

        Raises:
            ValueError: If scene_id is not found
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def get_coordinate_transform(
        self, target: CoordinateSystem = CoordinateSystem.OPENGL
    ) -> np.ndarray:
        """Return 4x4 transform from native to target coordinate system.

        Args:
            target: Target coordinate system

        Returns:
            4x4 transformation matrix
        """
        if self.coordinate_system == target:
            return np.eye(4, dtype=np.float64)

        # Common transforms between coordinate systems
        transforms = {
            (CoordinateSystem.OPENCV, CoordinateSystem.OPENGL): np.array(
                [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float64,
            ),
            (CoordinateSystem.SCANNET, CoordinateSystem.OPENGL): np.array(
                [
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float64,
            ),
        }

        key = (self.coordinate_system, target)
        if key in transforms:
            return transforms[key]

        # Try reverse
        reverse_key = (target, self.coordinate_system)
        if reverse_key in transforms:
            return np.linalg.inv(transforms[reverse_key])

        raise NotImplementedError(
            f"Transform from {self.coordinate_system} to {target} not implemented"
        )

    def load_mesh(self, scene_id: str) -> Any | None:
        """Load 3D mesh for a scene if available.

        Args:
            scene_id: Scene identifier

        Returns:
            Mesh data (format depends on implementation) or None
        """
        return None

    def load_semantic_info(self, scene_id: str) -> dict[int, str] | None:
        """Load semantic class mapping if available.

        Args:
            scene_id: Scene identifier

        Returns:
            Dict mapping class IDs to class names, or None
        """
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data_root={self.data_root})"
