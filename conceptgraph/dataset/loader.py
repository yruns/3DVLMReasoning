"""PyTorch dataset classes for posed RGB-D datasets in GradSLAM format.

Supports Replica, ScanNet, ICL, AI2Thor, Azure Kinect, Realsense,
Record3D, Multiscan, HM3D, and OpenEQA variants.

Adapted from https://github.com/cvg/nice-slam (datasets.py).
"""

from __future__ import annotations

import abc
import glob
import json
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from gradslam.datasets import datautils
from gradslam.geometry.geometryutils import relative_transformation
from natsort import natsorted

from conceptgraph.utils.general import measure_time, to_scalar

# -------------------------------------------------------------------
# Intrinsics helpers
# -------------------------------------------------------------------


def as_intrinsics_matrix(
    intrinsics: list[float] | tuple[float, ...],
) -> np.ndarray:
    """Build a 3x3 camera intrinsics matrix from (fx, fy, cx, cy)."""
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def from_intrinsics_matrix(
    K: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Extract (fx, fy, cx, cy) scalars from a 3x3 intrinsics matrix."""
    return (
        to_scalar(K[0, 0]),
        to_scalar(K[1, 1]),
        to_scalar(K[0, 2]),
        to_scalar(K[1, 2]),
    )


def _read_exr_depth(filename: str) -> np.ndarray:
    """Read a depth channel from an OpenEXR file."""
    import Imath
    import OpenEXR as exr

    exr_file = exr.InputFile(filename)
    header = exr_file.header()
    dw = header["dataWindow"]
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channel_data: dict[str, np.ndarray] = {}
    for c in header["channels"]:
        raw = exr_file.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        channel_data[c] = np.frombuffer(raw, dtype=np.float32).reshape(size)

    return channel_data.get("Y")  # type: ignore[return-value]


# -------------------------------------------------------------------
# Base dataset
# -------------------------------------------------------------------


class GradSLAMDataset(torch.utils.data.Dataset):
    """Abstract base for GradSLAM-compatible posed RGB-D datasets."""

    def __init__(
        self,
        config_dict: dict,
        stride: int | None = 1,
        start: int | None = 0,
        end: int | None = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float,
        load_embeddings: bool = False,
        embedding_dir: str = "feat_lseg_240_320",
        embedding_dim: int = 512,
        relative_pose: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.name = config_dict["dataset_name"]
        self.device = device
        self.dtype = dtype
        cam = config_dict["camera_params"]

        self.png_depth_scale = cam["png_depth_scale"]
        self.orig_height = cam["image_height"]
        self.orig_width = cam["image_width"]
        self.fx = cam["fx"]
        self.fy = cam["fy"]
        self.cx = cam["cx"]
        self.cy = cam["cy"]

        self.desired_height = desired_height
        self.desired_width = desired_width
        self.height_downsample_ratio = float(desired_height) / self.orig_height
        self.width_downsample_ratio = float(desired_width) / self.orig_width
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        self.load_embeddings = load_embeddings
        self.embedding_dir = embedding_dir
        self.embedding_dim = embedding_dim
        self.relative_pose = relative_pose

        self.start = start
        self.end = end
        if start < 0:
            raise ValueError(f"start must be positive, got {start}")
        if end != -1 and end <= start:
            raise ValueError(f"end ({end}) must be -1 or greater than start ({start})")

        self.distortion = np.array(cam["distortion"]) if "distortion" in cam else None
        self.crop_size = cam.get("crop_size")
        self.crop_edge = cam.get("crop_edge")

        self.color_paths, self.depth_paths, self.embedding_paths = self.get_filepaths()
        if len(self.color_paths) != len(self.depth_paths):
            raise ValueError("Number of color and depth images must match.")
        if self.load_embeddings and len(self.color_paths) != len(self.embedding_paths):
            raise ValueError("Number of color images and embedding files must match.")

        self.num_imgs = len(self.color_paths)
        self.poses = self.load_poses()

        if self.end == -1:
            self.end = self.num_imgs

        s = slice(self.start, self.end, stride)
        self.color_paths = self.color_paths[s]
        self.depth_paths = self.depth_paths[s]
        if self.load_embeddings:
            self.embedding_paths = self.embedding_paths[s]
        self.poses = self.poses[s]
        self.retained_inds = torch.arange(self.num_imgs)[s]
        self.num_imgs = len(self.color_paths)

        self.poses = torch.stack(self.poses)
        if self.relative_pose:
            self.transformed_poses = self._preprocess_poses(self.poses)
        else:
            self.transformed_poses = self.poses

    def __len__(self) -> int:
        return self.num_imgs

    @abc.abstractmethod
    def get_filepaths(
        self,
    ) -> tuple[list[str], list[str], list[str] | None]:
        """Return (color_paths, depth_paths, embedding_paths)."""
        ...

    @abc.abstractmethod
    def load_poses(self) -> list[torch.Tensor]:
        """Load per-frame camera poses as list of 4x4 tensors."""
        ...

    def _preprocess_color(self, color: np.ndarray) -> np.ndarray:
        color = cv2.resize(
            color,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.normalize_color:
            color = datautils.normalize_image(color)
        if self.channels_first:
            color = datautils.channels_first(color)
        return color

    def _preprocess_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = cv2.resize(
            depth.astype(float),
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            depth = datautils.channels_first(depth)
        return depth / self.png_depth_scale

    def _preprocess_poses(self, poses: torch.Tensor) -> torch.Tensor:
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )

    def get_cam_K(self) -> torch.Tensor:
        """Return the 3x3 camera intrinsics matrix."""
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        return torch.from_numpy(K)

    def read_embedding_from_file(self, embedding_path: str) -> torch.Tensor:
        """Read embedding from file.  Override in subclasses."""
        raise NotImplementedError

    def __getitem__(self, index: int):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]

        color = np.asarray(imageio.imread(color_path), dtype=float)
        color = self._preprocess_color(color)
        color = torch.from_numpy(color)

        if depth_path.endswith(".png"):
            depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)
        elif depth_path.endswith(".exr"):
            depth = _read_exr_depth(depth_path)
        elif depth_path.endswith(".npy"):
            depth = np.load(depth_path)
        else:
            raise NotImplementedError(f"Unsupported depth format: {depth_path}")

        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)

        if self.distortion is not None:
            color = cv2.undistort(color, K, self.distortion)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        K = datautils.scale_intrinsics(
            K,
            self.height_downsample_ratio,
            self.width_downsample_ratio,
        )
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]

        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                embedding.to(self.device),
            )

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
        )


# -------------------------------------------------------------------
# Concrete dataset implementations
# -------------------------------------------------------------------


class ICLDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict: dict,
        basedir: Path | str,
        sequence: Path | str,
        stride: int | None = 1,
        start: int | None = 0,
        end: int | None = -1,
        desired_height: int | None = 480,
        desired_width: int | None = 640,
        load_embeddings: bool | None = False,
        embedding_dir: Path | str | None = "embeddings",
        embedding_dim: int | None = 512,
        embedding_file_extension: str | None = "pt",
        **kwargs,
    ):
        self.input_folder = str(Path(basedir) / sequence)
        pose_files = glob.glob(f"{self.input_folder}/*.gt.sim")
        if not pose_files:
            raise ValueError("Need pose file ending in extension `*.gt.sim`")
        self.pose_path = pose_files[0]
        self.embedding_file_extension = embedding_file_extension
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(
                    f"{self.input_folder}/{self.embedding_dir}"
                    f"/*.{self.embedding_file_extension}"
                )
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        with open(self.pose_path) as f:
            lines = f.readlines()

        pose_arr = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            pose_arr.append([float(v) for v in parts])
        pose_arr = np.array(pose_arr)

        poses = []
        for i in range(0, pose_arr.shape[0], 3):
            mat = np.zeros((4, 4))
            mat[3, 3] = 3
            mat[0] = pose_arr[i]
            mat[1] = pose_arr[i + 1]
            mat[2] = pose_arr[i + 2]
            poses.append(torch.from_numpy(mat).float())
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)


class ReplicaDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict: dict,
        basedir: Path | str,
        sequence: Path | str,
        stride: int | None = None,
        start: int | None = 0,
        end: int | None = -1,
        desired_height: int | None = 480,
        desired_width: int | None = 640,
        load_embeddings: bool | None = False,
        embedding_dir: str | None = "embeddings",
        embedding_dim: int | None = 512,
        **kwargs,
    ):
        self.input_folder = str(Path(basedir) / sequence)
        self.pose_path = str(Path(self.input_folder) / "traj.txt")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        with open(self.pose_path) as f:
            lines = f.readlines()
        poses = []
        for i in range(self.num_imgs):
            c2w = np.array(list(map(float, lines[i].split()))).reshape(4, 4)
            poses.append(torch.from_numpy(c2w).float())
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)


class ScannetDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict: dict,
        basedir: Path | str,
        sequence: Path | str,
        stride: int | None = None,
        start: int | None = 0,
        end: int | None = -1,
        desired_height: int | None = 968,
        desired_width: int | None = 1296,
        load_embeddings: bool | None = False,
        embedding_dir: str | None = "embeddings",
        embedding_dim: int | None = 512,
        **kwargs,
    ):
        self.input_folder = str(Path(basedir) / sequence)
        self.pose_path = None

        intrinsic_path = Path(self.input_folder) / "intrinsic" / "intrinsic_color.txt"
        intrinsic = np.loadtxt(str(intrinsic_path))
        config_dict["camera_params"]["fx"] = intrinsic[0, 0]
        config_dict["camera_params"]["fy"] = intrinsic[1, 1]
        config_dict["camera_params"]["cx"] = intrinsic[0, 2]
        config_dict["camera_params"]["cy"] = intrinsic[1, 2]

        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        posefiles = natsorted(glob.glob(f"{self.input_folder}/pose/*.txt"))
        return [torch.from_numpy(np.loadtxt(pf)) for pf in posefiles]

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)


class Ai2thorDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict: dict,
        basedir: Path | str,
        sequence: Path | str,
        stride: int | None = None,
        start: int | None = 0,
        end: int | None = -1,
        desired_height: int | None = 968,
        desired_width: int | None = 1296,
        load_embeddings: bool | None = False,
        embedding_dir: str | None = "embeddings",
        embedding_dim: int | None = 512,
        **kwargs,
    ):
        self.input_folder = str(Path(basedir) / sequence)
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            ext = "png" if self.embedding_dir == "embed_semseg" else "pt"
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.{ext}")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        posefiles = natsorted(glob.glob(f"{self.input_folder}/pose/*.txt"))
        return [torch.from_numpy(np.loadtxt(pf)) for pf in posefiles]

    def read_embedding_from_file(self, embedding_file_path):
        if self.embedding_dir == "embed_semseg":
            embedding = imageio.imread(embedding_file_path)
            embedding = cv2.resize(
                embedding,
                (self.desired_width, self.desired_height),
                interpolation=cv2.INTER_NEAREST,
            )
            embedding = torch.from_numpy(embedding).long()
            embedding = F.one_hot(embedding, num_classes=self.embedding_dim)
            embedding = embedding.half().permute(2, 0, 1).unsqueeze(0)
        else:
            embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)


class AzureKinectDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict: dict,
        basedir: Path | str,
        sequence: Path | str,
        stride: int | None = None,
        start: int | None = 0,
        end: int | None = -1,
        desired_height: int | None = 480,
        desired_width: int | None = 640,
        load_embeddings: bool | None = False,
        embedding_dir: str | None = "embeddings",
        embedding_dim: int | None = 512,
        **kwargs,
    ):
        self.input_folder = str(Path(basedir) / sequence)
        self.pose_path = None

        dvo_path = Path(basedir) / sequence / "poses_global_dvo.txt"
        if dvo_path.is_file():
            self.pose_path = str(dvo_path)

        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        if self.pose_path is None:
            print("WARNING: No poses found. " "Returning identity transforms.")
            return [torch.eye(4).float() for _ in range(self.num_imgs)]

        if self.pose_path.endswith(".log"):
            with open(self.pose_path) as f:
                lines = f.readlines()
            if len(lines) % 5 != 0:
                raise ValueError(
                    "Incorrect .log format: line count must be " "a multiple of 5."
                )
            poses = []
            for i in range(len(lines) // 5):
                rows = [list(map(float, lines[5 * i + r].split())) for r in range(1, 5)]
                mat = np.array(rows).reshape(4, 4)
                poses.append(torch.from_numpy(mat))
            return poses

        with open(self.pose_path) as f:
            lines = f.readlines()
        poses = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            c2w = np.array(list(map(float, parts))).reshape(4, 4)
            poses.append(torch.from_numpy(c2w))
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        return torch.load(embedding_file_path)


class RealsenseDataset(GradSLAMDataset):
    """Dataset for depth images captured by a Realsense camera."""

    def __init__(
        self,
        config_dict: dict,
        basedir: Path | str,
        sequence: Path | str,
        stride: int | None = None,
        start: int | None = 0,
        end: int | None = -1,
        desired_height: int | None = 480,
        desired_width: int | None = 640,
        load_embeddings: bool | None = False,
        embedding_dir: str | None = "embeddings",
        embedding_dim: int | None = 512,
        **kwargs,
    ):
        self.input_folder = str(Path(basedir) / sequence)
        self.pose_path = str(Path(self.input_folder) / "poses")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        posefiles = natsorted(glob.glob(f"{self.pose_path}/*.npy"))
        P = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        ).float()
        poses = []
        for pf in posefiles:
            c2w = torch.from_numpy(np.load(pf)).float()
            poses.append(P @ c2w @ P.T)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)


class Record3DDataset(GradSLAMDataset):
    """Dataset for frames saved by ``save_record3d_stream.py``."""

    def __init__(
        self,
        config_dict: dict,
        basedir: Path | str,
        sequence: Path | str,
        stride: int | None = None,
        start: int | None = 0,
        end: int | None = -1,
        desired_height: int | None = 480,
        desired_width: int | None = 640,
        load_embeddings: bool | None = False,
        embedding_dir: str | None = "embeddings",
        embedding_dim: int | None = 512,
        **kwargs,
    ):
        self.input_folder = str(Path(basedir) / sequence)
        self.pose_path = str(Path(self.input_folder) / "poses")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        posefiles = natsorted(glob.glob(f"{self.pose_path}/*.npy"))
        P = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        ).float()
        poses = []
        for pf in posefiles:
            c2w = torch.from_numpy(np.load(pf)).float()
            poses.append(P @ c2w @ P.T)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)


class MultiscanDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict: dict,
        basedir: Path | str,
        sequence: Path | str,
        stride: int | None = None,
        start: int | None = 0,
        end: int | None = -1,
        desired_height: int | None = 480,
        desired_width: int | None = 640,
        load_embeddings: bool | None = False,
        embedding_dir: str | None = "embeddings",
        embedding_dim: int | None = 512,
        **kwargs,
    ):
        self.input_folder = str(Path(basedir) / sequence)
        self.pose_path = str(Path(self.input_folder) / f"{sequence}.jsonl")

        meta_path = Path(self.input_folder) / f"{sequence}.json"
        with open(meta_path) as f:
            scene_meta = json.load(f)

        cam_K = np.array(scene_meta["streams"][0]["intrinsics"]).reshape(3, 3).T
        config_dict["camera_params"]["fx"] = cam_K[0, 0]
        config_dict["camera_params"]["fy"] = cam_K[1, 1]
        config_dict["camera_params"]["cx"] = cam_K[0, 2]
        config_dict["camera_params"]["cy"] = cam_K[1, 2]
        res = scene_meta["streams"][0]["resolution"]
        config_dict["camera_params"]["image_height"] = res[0]
        config_dict["camera_params"]["image_width"] = res[1]

        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/outputs/color/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/outputs/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        with open(self.pose_path) as f:
            lines = f.readlines()

        n_sampled = len(glob.glob(f"{self.input_folder}/outputs/color/*.png"))
        step = round(len(lines) / float(n_sampled))

        poses = []
        for i in range(0, len(lines), step):
            info = json.loads(lines[i])
            xform = np.asarray(info["transform"]).reshape(4, 4, order="F")
            xform = xform @ np.diag([1, -1, -1, 1])
            xform = xform / xform[3, 3]
            poses.append(torch.from_numpy(xform).float())
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)


class Hm3dDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict: dict,
        basedir: Path | str,
        sequence: Path | str,
        stride: int | None = None,
        start: int | None = 0,
        end: int | None = -1,
        desired_height: int | None = 480,
        desired_width: int | None = 640,
        load_embeddings: bool | None = False,
        embedding_dir: str | None = "embeddings",
        embedding_dim: int | None = 512,
        **kwargs,
    ):
        self.input_folder = str(Path(basedir) / sequence)
        self.pose_path = None
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/*_depth.npy"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        posefiles = natsorted(glob.glob(f"{self.input_folder}/*.json"))
        P = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        ).float()
        poses = []
        for pf in posefiles:
            with open(pf) as f:
                pose_raw = json.load(f)
            pose = torch.from_numpy(np.asarray(pose_raw["pose"])).float()
            poses.append(P @ pose @ P.T)
        return poses


class Hm3dOpeneqaDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict: dict,
        basedir: Path | str,
        sequence: Path | str,
        stride: int | None = None,
        start: int | None = 0,
        end: int | None = -1,
        desired_height: int | None = 480,
        desired_width: int | None = 640,
        load_embeddings: bool | None = False,
        embedding_dir: str | None = "embeddings",
        embedding_dim: int | None = 512,
        **kwargs,
    ):
        self.input_folder = str(Path(basedir) / sequence)
        self.pose_path = None

        intrinsic_path = Path(self.input_folder) / "intrinsic_color.txt"
        intrinsic = np.loadtxt(str(intrinsic_path))
        config_dict["camera_params"]["fx"] = intrinsic[0, 0]
        config_dict["camera_params"]["fy"] = intrinsic[1, 1]
        config_dict["camera_params"]["cx"] = intrinsic[0, 2]
        config_dict["camera_params"]["cy"] = intrinsic[1, 2]

        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/*-rgb.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/*-depth.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        posefiles = natsorted(glob.glob(f"{self.input_folder}/[0-9]*.txt"))
        P = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        ).float()
        poses = []
        for pf in posefiles:
            pose = torch.from_numpy(np.loadtxt(pf)).float()
            poses.append(P @ pose @ P.T)
        return poses


class ScannetOpenEQADataset(GradSLAMDataset):
    """OpenEQA ScanNet clip dataset.

    Supports two directory layouts:
      - **Legacy**: frames live inside the scene dir itself.
      - **Split** (preferred): frames live in a sibling ``raw/``
        directory, and the scene dir holds only pipeline outputs.

    Resolution order for input files (rgb, depth, poses, intrinsics):
      1. ``<basedir>/<sequence>/../raw/``  (split layout)
      2. ``<basedir>/<sequence>/``         (legacy layout)
    """

    def __init__(
        self,
        config_dict: dict,
        basedir: Path | str,
        sequence: Path | str,
        stride: int | None = None,
        start: int | None = 0,
        end: int | None = -1,
        desired_height: int | None = 968,
        desired_width: int | None = 1296,
        load_embeddings: bool | None = False,
        embedding_dir: str | None = "embeddings",
        embedding_dim: int | None = 512,
        **kwargs,
    ):
        self.input_folder = str(Path(basedir) / sequence)
        self.pose_path = None

        # Resolve the directory that actually holds frames / intrinsics.
        raw_candidate = Path(self.input_folder).parent / "raw"
        if raw_candidate.is_dir() and (
            any(raw_candidate.glob("*-rgb.png"))
            or any(raw_candidate.glob("*-rgb.jpg"))
        ):
            self.raw_folder = str(raw_candidate)
        else:
            self.raw_folder = self.input_folder

        intrinsic_path = Path(self.raw_folder) / "intrinsic_color.txt"
        intrinsic = np.loadtxt(str(intrinsic_path))
        config_dict["camera_params"]["fx"] = intrinsic[0, 0]
        config_dict["camera_params"]["fy"] = intrinsic[1, 1]
        config_dict["camera_params"]["cx"] = intrinsic[0, 2]
        config_dict["camera_params"]["cy"] = intrinsic[1, 2]

        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.raw_folder}/*-rgb.png"))
        if not color_paths:
            color_paths = natsorted(glob.glob(f"{self.raw_folder}/*-rgb.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.raw_folder}/*-depth.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.raw_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        posefiles = natsorted(glob.glob(f"{self.raw_folder}/[0-9]*.txt"))
        return [torch.from_numpy(np.loadtxt(pf)).float() for pf in posefiles]


# -------------------------------------------------------------------
# Config loading utilities
# -------------------------------------------------------------------


def load_dataset_config(
    path: str | Path, default_path: str | Path | None = None
) -> dict:
    """Load a dataset YAML config with optional inheritance."""
    with open(path) as f:
        cfg_special = yaml.full_load(f)

    inherit_from = cfg_special.get("inherit_from")
    if inherit_from is not None:
        cfg = load_dataset_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path) as f:
            cfg = yaml.full_load(f)
    else:
        cfg = {}

    _update_recursive(cfg, cfg_special)
    return cfg


def _update_recursive(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in-place."""
    for k, v in override.items():
        if k not in base:
            base[k] = {}
        if isinstance(v, dict):
            _update_recursive(base[k], v)
        else:
            base[k] = v


# -------------------------------------------------------------------
# Batch conversion helper
# -------------------------------------------------------------------


def common_dataset_to_batch(
    dataset: GradSLAMDataset,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Convert an entire dataset into a single batched tuple."""
    colors, depths, poses = [], [], []
    intrinsics = None
    embeddings: list[torch.Tensor] | None = None

    for idx in range(len(dataset)):
        _color, _depth, intrinsics, _pose, _embedding = dataset[idx]
        colors.append(_color)
        depths.append(_depth)
        poses.append(_pose)
        if _embedding is not None:
            if embeddings is None:
                embeddings = [_embedding]
            else:
                embeddings.append(_embedding)

    stacked_embeddings = None
    if embeddings is not None:
        stacked_embeddings = torch.stack(embeddings, dim=1).float()

    return (
        torch.stack(colors).unsqueeze(0).float(),
        torch.stack(depths).unsqueeze(0).float(),
        intrinsics.unsqueeze(0).unsqueeze(0).float(),
        torch.stack(poses).unsqueeze(0).float(),
        stacked_embeddings,
    )


# -------------------------------------------------------------------
# Dataset factory
# -------------------------------------------------------------------

_DATASET_REGISTRY: dict[str, type[GradSLAMDataset]] = {
    "icl": ICLDataset,
    "replica": ReplicaDataset,
    "azure": AzureKinectDataset,
    "azurekinect": AzureKinectDataset,
    "scannet": ScannetDataset,
    "ai2thor": Ai2thorDataset,
    "record3d": Record3DDataset,
    "realsense": RealsenseDataset,
    "multiscan": MultiscanDataset,
    "hm3d": Hm3dDataset,
    "hm3d-openeqa": Hm3dOpeneqaDataset,
    "scannet-openeqa": ScannetOpenEQADataset,
}


@measure_time
def get_dataset(
    dataconfig: str | Path,
    basedir: str | Path,
    sequence: str,
    **kwargs,
) -> GradSLAMDataset:
    """Instantiate the appropriate dataset class from a YAML config."""
    config_dict = load_dataset_config(dataconfig)
    name = config_dict["dataset_name"].lower()
    cls = _DATASET_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown dataset name: {name}")
    return cls(config_dict, basedir, sequence, **kwargs)
