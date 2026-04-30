#!/usr/bin/env python3
"""Build GT-injected ConceptGraph objects for NR3D ScanNet scenes."""

from __future__ import annotations

import argparse
import colorsys
import csv
import gzip
import hashlib
import importlib
import json
import pickle
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import open_clip
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from plyfile import PlyData

_PKG_PREFIX = "src." if (__package__ or "").startswith("src.") else ""
_visibility_module = importlib.import_module(
    f"{_PKG_PREFIX}benchmarks.embodiedscan_bbox_feasibility.visibility_index"
)
_visibility_builder_module = importlib.import_module(
    f"{_PKG_PREFIX}scripts.build_visibility_index"
)
project_bbox_3d_to_2d = _visibility_module.project_bbox_3d_to_2d
build_visibility_index = _visibility_builder_module.build_visibility_index
save_visibility_index = _visibility_builder_module.save_visibility_index

BACKGROUND_LABELS = {"wall", "floor", "ceiling"}
CLIP_MODEL_NAME = "ViT-H-14"
CLIP_PRETRAINED = "laion2b_s32b_b79k"
DEFAULT_MAX_CLIP_VIEWS_PER_OBJECT = 8
DEFAULT_NR3D_ROOT = Path("data/nr3d")
DEFAULT_SCANNET_ROOT = Path("/home/ysh/Datasets/ScanNet")
DEFAULT_OPENEQA_REFERENCE = Path(
    "/home/ysh/Datasets/OpenEQA/scannet/002-scannet-scene0709_00/conceptgraph"
)
DEFAULT_REPORT_PATH = Path.home() / ".super-orchestrator/nr3d/artifacts/nr3d-gt-cg.md"

CONCEPTGRAPH_OBJECT_FIELD_ORDER = [
    "image_idx",
    "mask_idx",
    "color_path",
    "class_name",
    "class_id",
    "num_detections",
    "mask",
    "xyxy",
    "conf",
    "n_points",
    "pixel_area",
    "contain_number",
    "inst_color",
    "is_background",
    "clip_ft",
    "text_ft",
    "pcd_np",
    "bbox_np",
    "pcd_color_np",
]
CONCEPTGRAPH_OBJECT_FIELDS = set(CONCEPTGRAPH_OBJECT_FIELD_ORDER)


@dataclass(frozen=True)
class GtObject:
    """ScanNet GT object geometry before per-view ConceptGraph fields."""

    object_id: int
    label: str
    class_id: int
    bbox_np: np.ndarray
    pcd_np: np.ndarray
    pcd_color_np: np.ndarray
    is_background: int


@dataclass(frozen=True)
class RawScene:
    scene_id: str
    raw_dir: Path
    rgb_paths: list[Path]
    depth_paths: list[Path]
    pose_paths: list[Path]
    poses: list[np.ndarray]
    intrinsic_color: np.ndarray
    intrinsic_depth: np.ndarray
    image_width: int
    image_height: int
    depth_width: int
    depth_height: int


@dataclass
class SceneBuildStats:
    scene_id: str
    num_objects: int
    num_frames: int
    num_visible_mappings: int
    num_clip_features: int
    output_path: str
    elapsed_seconds: float


def load_scene_ids(path: Path) -> list[str]:
    """Load the NR3D JSON-array scene list."""
    if not path.exists():
        raise FileNotFoundError(f"NR3D scene list not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"NR3D scene list is not a file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"Scene list must be a JSON array of strings: {path}")
    duplicates = sorted({item for item in data if data.count(item) > 1})
    if duplicates:
        raise ValueError(f"Scene list contains duplicate scene ids: {duplicates}")
    if not data:
        raise ValueError(f"Scene list is empty: {path}")
    return data


def parse_axis_alignment(path: Path) -> np.ndarray:
    """Read the ScanNet axisAlignment 4x4 matrix from a scene .txt file."""
    if not path.exists():
        raise FileNotFoundError(f"ScanNet metadata file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("axisAlignment"):
            continue
        _, raw_values = stripped.split("=", maxsplit=1)
        values = [float(value) for value in raw_values.split()]
        if len(values) != 16:
            raise ValueError(f"axisAlignment in {path} must contain 16 values")
        matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
        _require_finite(matrix, f"axisAlignment in {path}")
        return matrix
    raise ValueError(f"axisAlignment not found in {path}")


def verify_aux_files(
    scene_ids: list[str],
    *,
    nr3d_root: Path = DEFAULT_NR3D_ROOT,
    scannet_root: Path = DEFAULT_SCANNET_ROOT,
) -> dict[str, Any]:
    """Parse-check all required ScanNet aux files for the scene list."""
    aux_root = nr3d_root / "scannet_aux"
    checked: list[str] = []
    for scene_id in scene_ids:
        paths = _aux_paths(aux_root, scene_id)
        for description, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{description} missing for {scene_id}: {path}")
            if not path.is_file():
                raise FileNotFoundError(
                    f"{description} is not a file for {scene_id}: {path}"
                )
        parse_axis_alignment(paths["metadata"])
        aggregation = _load_json(paths["aggregation"], "aggregation")
        segs = _load_json(paths["segments"], "segments")
        if not isinstance(aggregation.get("segGroups"), list):
            raise ValueError(f"{paths['aggregation']} missing segGroups list")
        if not isinstance(segs.get("segIndices"), list):
            raise ValueError(f"{paths['segments']} missing segIndices list")
        sens_path = _scannet_scene_dir(scannet_root, scene_id) / f"{scene_id}.sens"
        mesh_path = _mesh_path(scannet_root, scene_id)
        if not sens_path.exists():
            raise FileNotFoundError(f".sens missing for {scene_id}: {sens_path}")
        if not mesh_path.exists():
            raise FileNotFoundError(f"ScanNet mesh missing for {scene_id}: {mesh_path}")
        checked.append(scene_id)
    return {"num_scenes": len(checked), "scenes": checked}


def build_gt_objects_for_scene(
    scene_id: str,
    *,
    aux_root: Path,
    scannet_root: Path,
    class_to_idx: dict[str, int],
) -> list[GtObject]:
    """Derive axis-aligned GT ConceptGraph object geometry from ScanNet GT."""
    paths = _aux_paths(aux_root, scene_id)
    axis_align = parse_axis_alignment(paths["metadata"])
    aggregation = _load_json(paths["aggregation"], "aggregation")
    segments = _load_json(paths["segments"], "segments")
    seg_groups = aggregation.get("segGroups")
    seg_indices = segments.get("segIndices")
    if not isinstance(seg_groups, list):
        raise ValueError(f"{paths['aggregation']} missing segGroups list")
    if not isinstance(seg_indices, list):
        raise ValueError(f"{paths['segments']} missing segIndices list")

    vertices, colors = _load_mesh_vertices(_mesh_path(scannet_root, scene_id))
    if len(seg_indices) != len(vertices):
        raise ValueError(
            f"{scene_id}: segIndices length {len(seg_indices)} does not match "
            f"mesh vertex count {len(vertices)}"
        )

    seg_to_vertices = _build_seg_to_vertices(seg_indices)
    objects: list[GtObject] = []
    seen_object_ids: set[int] = set()
    for group_idx, group in enumerate(seg_groups):
        if not isinstance(group, dict):
            raise ValueError(f"{scene_id}: segGroups[{group_idx}] is not an object")
        object_id = _require_int(group.get("objectId"), f"{scene_id} objectId")
        if object_id in seen_object_ids:
            raise ValueError(f"{scene_id}: duplicate objectId {object_id}")
        seen_object_ids.add(object_id)
        label = _require_label(group.get("label"), f"{scene_id} objectId {object_id}")
        group_segments = group.get("segments")
        if not isinstance(group_segments, list) or not group_segments:
            raise ValueError(f"{scene_id} objectId {object_id}: empty segments list")
        missing = [int(seg) for seg in group_segments if int(seg) not in seg_to_vertices]
        if missing:
            raise ValueError(
                f"{scene_id} objectId {object_id}: segments absent from segIndices: "
                f"{missing[:10]}"
            )

        vertex_indices = np.unique(
            np.concatenate([seg_to_vertices[int(seg)] for seg in group_segments])
        )
        if len(vertex_indices) == 0:
            raise ValueError(f"{scene_id} objectId {object_id}: no vertices")

        pcd_np = _transform_points(vertices[vertex_indices], axis_align)
        pcd_color_np = colors[vertex_indices].astype(np.float64) / 255.0
        _require_finite(pcd_np, f"{scene_id} objectId {object_id} pcd_np")
        _require_finite(pcd_color_np, f"{scene_id} objectId {object_id} pcd_color_np")
        bbox_np = _oriented_bbox_points(pcd_np, scene_id=scene_id, object_id=object_id)
        if label not in class_to_idx:
            raise ValueError(f"{scene_id} label {label!r} absent from class_to_idx")
        objects.append(
            GtObject(
                object_id=object_id,
                label=label,
                class_id=int(class_to_idx[label]),
                bbox_np=bbox_np,
                pcd_np=pcd_np,
                pcd_color_np=pcd_color_np,
                is_background=int(label.strip().lower() in BACKGROUND_LABELS),
            )
        )
    return objects


def stable_class_colors(class_names: list[str]) -> dict[str, list[float]]:
    """Create deterministic ConceptGraph-style class-index color map."""
    colors: dict[str, list[float]] = {}
    for idx, name in enumerate(class_names):
        digest = hashlib.sha1(name.encode("utf-8")).digest()
        hue = int.from_bytes(digest[:2], "big") / 65535.0
        sat = 0.55 + digest[2] / 255.0 * 0.35
        val = 0.70 + digest[3] / 255.0 * 0.25
        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        colors[str(idx)] = [float(channel) for channel in rgb]
    return colors


def build_cfg(
    *,
    scene_id: str,
    image_width: int,
    image_height: int,
    save_objects_all_frames: bool = False,
) -> DictConfig:
    """Build a DictConfig with the OpenEQA ConceptGraph key surface."""
    return OmegaConf.create(
        {
            "dataset_root": ".",
            "dataset_config": "conceptgraph/dataset/dataconfigs/scannet/openeqa_clip.yaml",
            "scene_id": f"{scene_id}/conceptgraph",
            "start": 0,
            "end": -1,
            "stride": 1,
            "image_height": int(image_height),
            "image_width": int(image_width),
            "gsa_variant": "gt",
            "detection_folder_name": "gsa_detections_gt",
            "det_vis_folder_name": "gsa_vis_gt",
            "color_file_name": "gsa_classes_gt",
            "device": "cuda",
            "use_iou": True,
            "spatial_sim_type": "overlap",
            "phys_bias": 0.0,
            "match_method": "sim_sum",
            "semantic_threshold": 0.5,
            "physical_threshold": 0.5,
            "sim_threshold": 1.2,
            "use_contain_number": False,
            "contain_area_thresh": 0.95,
            "contain_mismatch_penalty": 0.5,
            "mask_area_threshold": 25,
            "mask_conf_threshold": 0.1,
            "max_bbox_area_ratio": 0.5,
            "skip_bg": False,
            "min_points_threshold": 16,
            "downsample_voxel_size": 0.025,
            "dbscan_remove_noise": True,
            "dbscan_eps": 0.1,
            "dbscan_min_points": 10,
            "obj_min_points": 0,
            "obj_min_detections": 1,
            "merge_overlap_thresh": 0.7,
            "merge_visual_sim_thresh": 0.8,
            "merge_text_sim_thresh": 0.8,
            "denoise_interval": 20,
            "filter_interval": -1,
            "merge_interval": 20,
            "save_pcd": True,
            "save_suffix": "axisaligned",
            "vis_render": False,
            "debug_render": False,
            "class_agnostic": True,
            "save_objects_all_frames": bool(save_objects_all_frames),
            "render_camera_path": "replica_room0.json",
        }
    )


def load_raw_scene(scene_id: str, *, nr3d_root: Path = DEFAULT_NR3D_ROOT) -> RawScene:
    """Load raw posed frames exported for one scene."""
    raw_dir = nr3d_root / "scannet" / scene_id / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"{scene_id}: raw directory not found: {raw_dir}")

    rgb_paths = sorted(raw_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9]-rgb.png"))
    depth_paths = sorted(raw_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9]-depth.png"))
    pose_paths = sorted(raw_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].txt"))
    if not rgb_paths or not depth_paths or not pose_paths:
        raise FileNotFoundError(f"{scene_id}: missing RGB/depth/pose files in {raw_dir}")
    if not (len(rgb_paths) == len(depth_paths) == len(pose_paths)):
        raise ValueError(
            f"{scene_id}: RGB/depth/pose count mismatch: "
            f"{len(rgb_paths)}/{len(depth_paths)}/{len(pose_paths)}"
        )
    _require_matching_frame_ids(scene_id, rgb_paths, depth_paths, pose_paths)

    intrinsic_color = _load_matrix(raw_dir / "intrinsic_color.txt", (4, 4))
    intrinsic_depth = _load_matrix(raw_dir / "intrinsic_depth.txt", (4, 4))
    poses = [_load_pose(scene_id, pose_path) for pose_path in pose_paths]
    image_height, image_width = _image_shape(rgb_paths[0], scene_id=scene_id)
    depth_height, depth_width = _image_shape(depth_paths[0], scene_id=scene_id)
    return RawScene(
        scene_id=scene_id,
        raw_dir=raw_dir,
        rgb_paths=rgb_paths,
        depth_paths=depth_paths,
        pose_paths=pose_paths,
        poses=poses,
        intrinsic_color=intrinsic_color,
        intrinsic_depth=intrinsic_depth,
        image_width=image_width,
        image_height=image_height,
        depth_width=depth_width,
        depth_height=depth_height,
    )


def build_object_dict(
    gt_object: GtObject,
    *,
    visible_views: list[tuple[int, float]],
    poses: list[np.ndarray],
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    raw_rgb_paths: list[Path],
    inst_color: list[float] | np.ndarray,
    clip_ft: np.ndarray,
    text_ft: np.ndarray,
    allow_empty_detections: bool = False,
) -> dict[str, Any]:
    """Create one OpenEQA-compatible ConceptGraph object dict."""
    views = sorted((int(view_id), float(score)) for view_id, score in visible_views)
    if not views and not allow_empty_detections:
        raise ValueError(f"GT object {gt_object.object_id} has no visible views")
    clip_ft = _validate_feature(clip_ft, f"object {gt_object.object_id} clip_ft")
    text_ft = _validate_feature(text_ft, f"object {gt_object.object_id} text_ft")
    intrinsic3 = intrinsic[:3, :3] if intrinsic.shape == (4, 4) else intrinsic
    if intrinsic3.shape != (3, 3):
        raise ValueError(f"intrinsic must be 3x3 or 4x4, got {intrinsic.shape}")

    image_width, image_height = image_size
    bbox_9dof = _bbox_np_to_axis_aligned_9dof(gt_object.bbox_np)
    image_idx: list[int] = []
    mask_idx: list[int] = []
    color_path: list[str] = []
    class_name: list[str] = []
    class_id: list[int] = []
    masks: list[np.ndarray] = []
    xyxy: list[np.ndarray] = []
    conf: list[np.float32] = []
    n_points: list[int] = []
    pixel_area: list[float] = []
    contain_number: list[int] = []

    for det_idx, (view_id, _score) in enumerate(views):
        if view_id < 0 or view_id >= len(poses):
            raise IndexError(
                f"object {gt_object.object_id}: view_id {view_id} out of range "
                f"for {len(poses)} poses"
            )
        pose = poses[view_id]
        _require_finite(pose, f"object {gt_object.object_id} pose view {view_id}")
        world_to_cam = np.linalg.inv(pose)
        rect = project_bbox_3d_to_2d(
            bbox_9dof,
            intrinsic3,
            world_to_cam,
            image_size=(image_width, image_height),
            depth_max=20.0,
        )
        if rect is None:
            raise ValueError(
                f"object {gt_object.object_id}: visible view {view_id} "
                "did not project to a 2D rectangle"
            )
        x1, y1, x2, y2 = rect
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"object {gt_object.object_id}: degenerate xyxy {rect} in view {view_id}"
            )
        mask = np.zeros((image_height, image_width), dtype=bool)
        mask[y1 : y2 + 1, x1 : x2 + 1] = True

        image_idx.append(view_id)
        mask_idx.append(det_idx)
        color_path.append(str(raw_rgb_paths[view_id].resolve()))
        class_name.append(gt_object.label)
        class_id.append(gt_object.class_id)
        masks.append(mask)
        xyxy.append(np.asarray([x1, y1, x2, y2], dtype=np.float32))
        conf.append(np.float32(1.0))
        n_points.append(int(len(gt_object.pcd_np)))
        pixel_area.append(float((x2 - x1) * (y2 - y1)))
        contain_number.append(0)

    return {
        "image_idx": image_idx,
        "mask_idx": mask_idx,
        "color_path": color_path,
        "class_name": class_name,
        "class_id": class_id,
        "num_detections": len(image_idx),
        "mask": masks,
        "xyxy": xyxy,
        "conf": conf,
        "n_points": n_points,
        "pixel_area": pixel_area,
        "contain_number": contain_number,
        "inst_color": np.asarray(inst_color, dtype=np.float64),
        "is_background": int(gt_object.is_background),
        "clip_ft": clip_ft,
        "text_ft": text_ft,
        "pcd_np": np.asarray(gt_object.pcd_np, dtype=np.float64),
        "bbox_np": np.asarray(gt_object.bbox_np, dtype=np.float64),
        "pcd_color_np": np.asarray(gt_object.pcd_color_np, dtype=np.float64),
    }


def build_payload(
    objects: list[dict[str, Any]],
    cfg: DictConfig,
    class_names: list[str],
    class_colors: dict[str, list[float]],
) -> dict[str, Any]:
    """Build the OpenEQA ConceptGraph top-level pickle payload."""
    for obj_idx, obj in enumerate(objects):
        fields = set(obj.keys())
        if fields != CONCEPTGRAPH_OBJECT_FIELDS:
            raise ValueError(
                f"object {obj_idx} field mismatch: missing "
                f"{sorted(CONCEPTGRAPH_OBJECT_FIELDS - fields)}, extra "
                f"{sorted(fields - CONCEPTGRAPH_OBJECT_FIELDS)}"
            )
    return {
        "objects": objects,
        "bg_objects": None,
        "cfg": cfg,
        "class_names": class_names,
        "class_colors": class_colors,
    }


def build_scene(
    scene_id: str,
    *,
    nr3d_root: Path = DEFAULT_NR3D_ROOT,
    scannet_root: Path = DEFAULT_SCANNET_ROOT,
    clip_runtime: dict[str, Any],
    use_depth: bool = False,
    max_distance: float = 5.0,
    min_visible_ratio: float = 0.03,
    min_visible_points: int = 5,
    max_clip_views_per_object: int = DEFAULT_MAX_CLIP_VIEWS_PER_OBJECT,
) -> SceneBuildStats:
    """Build final GT ConceptGraph outputs for one NR3D scene."""
    start = time.time()
    aux_root = nr3d_root / "scannet_aux"
    raw_scene = load_raw_scene(scene_id, nr3d_root=nr3d_root)
    class_names = _scene_class_names(scene_id, aux_root=aux_root)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    class_colors = stable_class_colors(class_names)
    gt_objects = build_gt_objects_for_scene(
        scene_id,
        aux_root=aux_root,
        scannet_root=scannet_root,
        class_to_idx=class_to_idx,
    )

    visibility_start = time.time()
    object_to_views, view_to_objects = build_visibility_index(
        objects=[{"pcd_np": obj.pcd_np} for obj in gt_objects],
        poses=raw_scene.poses,
        depth_paths=raw_scene.depth_paths,
        intrinsics=raw_scene.intrinsic_color,
        max_distance=max_distance,
        use_depth=use_depth,
        stride=1,
        img_w=raw_scene.image_width,
        img_h=raw_scene.image_height,
        min_visible_ratio=min_visible_ratio,
        min_visible_points=min_visible_points,
    )
    visibility_elapsed = time.time() - visibility_start

    fallback_objects, unobserved_objects = _add_projection_fallback_views(
        gt_objects=gt_objects,
        raw_scene=raw_scene,
        object_to_views=object_to_views,
        view_to_objects=view_to_objects,
    )

    conceptgraph_dir = nr3d_root / "scannet" / scene_id / "conceptgraph"
    indices_dir = conceptgraph_dir / "indices"
    pcd_dir = conceptgraph_dir / "pcd_saves"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    save_visibility_index(
        object_to_views,
        view_to_objects,
        indices_dir / "visibility_index.pkl",
        metadata={
            "scene_path": "conceptgraph",
            "pcd_file": "conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz",
            "stride": 1,
            "max_distance": max_distance,
            "use_depth": use_depth,
            "num_objects": len(gt_objects),
            "num_views": len(raw_scene.poses),
            "num_object_mappings": sum(len(v) for v in object_to_views.values()),
            "num_view_mappings": sum(len(v) for v in view_to_objects.values()),
            "num_projection_fallback_objects": len(fallback_objects),
            "projection_fallback_objects": fallback_objects,
            "num_unobserved_background_objects": len(unobserved_objects),
            "unobserved_background_objects": unobserved_objects,
            "build_time": visibility_elapsed,
        },
    )

    object_drafts = []
    for obj_idx, gt_object in enumerate(gt_objects):
        visible_views = object_to_views.get(obj_idx, [])
        if visible_views:
            object_drafts.append(
                _build_projection_draft(
                    gt_object,
                    visible_views=visible_views,
                    raw_scene=raw_scene,
                    inst_color=class_colors[str(gt_object.class_id)],
                )
            )
        else:
            if not any(
                item["object_index"] == obj_idx for item in unobserved_objects
            ):
                raise ValueError(
                    f"{scene_id} objectId {gt_object.object_id}: no visible views"
                )
            object_drafts.append(
                {
                    "gt_object": gt_object,
                    "visible_views": [],
                    "xyxy": [],
                    "rgb_paths": [],
                    "inst_color": class_colors[str(gt_object.class_id)],
                    "unobserved": True,
                }
            )

    feature_map = extract_clip_features_for_scene(
        object_drafts,
        clip_runtime=clip_runtime,
        max_views_per_object=max_clip_views_per_object,
    )
    final_objects = []
    for draft in object_drafts:
        gt_object = draft["gt_object"]
        final_objects.append(
            build_object_dict(
                gt_object,
                visible_views=draft["visible_views"],
                poses=raw_scene.poses,
                intrinsic=raw_scene.intrinsic_color,
                image_size=(raw_scene.image_width, raw_scene.image_height),
                raw_rgb_paths=raw_scene.rgb_paths,
                inst_color=draft["inst_color"],
                clip_ft=feature_map[gt_object.object_id]["clip_ft"],
                text_ft=feature_map[gt_object.object_id]["text_ft"],
                allow_empty_detections=bool(draft.get("unobserved", False)),
            )
        )

    cfg = build_cfg(
        scene_id=scene_id,
        image_width=raw_scene.image_width,
        image_height=raw_scene.image_height,
    )
    payload = build_payload(final_objects, cfg, class_names, class_colors)
    output_path = pcd_dir / "full_pcd_gt_axisaligned_post.pkl.gz"
    with gzip.open(output_path, "wb") as handle:
        pickle.dump(payload, handle)

    (conceptgraph_dir / "gsa_classes_gt.json").write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8",
    )
    (conceptgraph_dir / "gsa_classes_gt_colors.json").write_text(
        json.dumps(class_colors, indent=2),
        encoding="utf-8",
    )
    scene_info = {
        "scene_id": scene_id,
        "raw_dir": str(raw_scene.raw_dir.resolve()),
        "conceptgraph_dir": str(conceptgraph_dir.resolve()),
        "mesh_path": str(_mesh_path(scannet_root, scene_id).resolve()),
        "num_rgb_frames": len(raw_scene.rgb_paths),
        "num_depth_frames": len(raw_scene.depth_paths),
        "num_pose_files": len(raw_scene.pose_paths),
        "num_objects": len(final_objects),
        "num_visibility_mappings": sum(len(v) for v in object_to_views.values()),
        "num_projection_fallback_objects": len(fallback_objects),
        "projection_fallback_objects": fallback_objects,
        "num_unobserved_background_objects": len(unobserved_objects),
        "unobserved_background_objects": unobserved_objects,
        "max_clip_views_per_object": max_clip_views_per_object,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "nr3d-gt-cg-producer",
        "bbox_geometry": "ScanNet GT axisAlignment + Open3D OBB corners",
        "mask_note": "2D masks are rectangular projected-bbox approximations.",
    }
    (conceptgraph_dir / "scene_info.json").write_text(
        json.dumps(scene_info, indent=2),
        encoding="utf-8",
    )

    elapsed = time.time() - start
    logger.success(
        "{}: wrote {} objects, {} frames, {} visibility mappings in {:.1f}s",
        scene_id,
        len(final_objects),
        len(raw_scene.rgb_paths),
        sum(len(v) for v in object_to_views.values()),
        elapsed,
    )
    return SceneBuildStats(
        scene_id=scene_id,
        num_objects=len(final_objects),
        num_frames=len(raw_scene.rgb_paths),
        num_visible_mappings=sum(len(v) for v in object_to_views.values()),
        num_clip_features=len(final_objects),
        output_path=str(output_path),
        elapsed_seconds=elapsed,
    )


def create_clip_runtime(device: str) -> dict[str, Any]:
    """Load OpenCLIP runtime on the requested CUDA device."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NR3D GT CLIP extraction")
    if device == "cuda:1" or device.endswith(":1"):
        raise ValueError("GPU 1 is broken on this server; do not use cuda:1")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, CLIP_PRETRAINED
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    return {
        "device": device,
        "model": model,
        "preprocess": preprocess,
        "tokenizer": tokenizer,
    }


def extract_clip_features_for_scene(
    object_drafts: list[dict[str, Any]],
    *,
    clip_runtime: dict[str, Any],
    batch_size: int = 64,
    max_views_per_object: int = DEFAULT_MAX_CLIP_VIEWS_PER_OBJECT,
) -> dict[int, dict[str, np.ndarray]]:
    """Compute per-object image/text CLIP features for one scene."""
    if max_views_per_object <= 0:
        raise ValueError(
            f"max_views_per_object must be positive, got {max_views_per_object}"
        )
    model = clip_runtime["model"]
    preprocess = clip_runtime["preprocess"]
    tokenizer = clip_runtime["tokenizer"]
    device = clip_runtime["device"]
    image_cache: dict[Path, Image.Image] = {}

    labels = sorted({draft["gt_object"].label for draft in object_drafts})
    text_features = _encode_text_features(labels, model=model, tokenizer=tokenizer, device=device)

    result: dict[int, dict[str, np.ndarray]] = {}
    for draft in object_drafts:
        gt_object: GtObject = draft["gt_object"]
        crops = []
        selected_views = list(zip(draft["rgb_paths"], draft["xyxy"], strict=True))[
            :max_views_per_object
        ]
        for rgb_path, xyxy in selected_views:
            crops.append(_preprocess_crop(rgb_path, xyxy, preprocess, image_cache))
        if not crops:
            if draft.get("unobserved"):
                result[gt_object.object_id] = {
                    "clip_ft": text_features[gt_object.label],
                    "text_ft": text_features[gt_object.label],
                }
                continue
            raise ValueError(f"{gt_object.object_id}: no crops for CLIP extraction")
        crop_features = []
        for start in range(0, len(crops), batch_size):
            batch = torch.stack(crops[start : start + batch_size]).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                encoded = model.encode_image(batch)
                encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            crop_features.append(encoded.float().cpu().numpy())
        stacked = np.concatenate(crop_features, axis=0)
        image_feature = stacked.mean(axis=0).astype(np.float32)
        image_norm = float(np.linalg.norm(image_feature))
        if image_norm <= 0 or not np.isfinite(image_norm):
            raise ValueError(f"{gt_object.object_id}: invalid CLIP image feature norm")
        image_feature = (image_feature / image_norm).astype(np.float32)
        result[gt_object.object_id] = {
            "clip_ft": image_feature,
            "text_ft": text_features[gt_object.label],
        }
    return result


def verify_outputs(
    scene_ids: list[str],
    *,
    nr3d_root: Path = DEFAULT_NR3D_ROOT,
    openeqa_reference: Path = DEFAULT_OPENEQA_REFERENCE,
) -> dict[str, Any]:
    """Verify generated NR3D outputs and compare object schema to OpenEQA."""
    reference_pkl = _find_reference_pkl(openeqa_reference)
    with gzip.open(reference_pkl, "rb") as handle:
        reference = pickle.load(handle)
    reference_fields = set(reference["objects"][0].keys())
    generated_summaries = []
    total_objects = 0
    total_frames = 0
    total_mappings = 0
    for scene_id in scene_ids:
        scene_root = nr3d_root / "scannet" / scene_id
        pkl_path = scene_root / "conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz"
        visibility_path = scene_root / "conceptgraph/indices/visibility_index.pkl"
        raw_scene = load_raw_scene(scene_id, nr3d_root=nr3d_root)
        with gzip.open(pkl_path, "rb") as handle:
            payload = pickle.load(handle)
        objects = payload["objects"]
        for obj_idx, obj in enumerate(objects):
            fields = set(obj.keys())
            if fields != reference_fields:
                raise ValueError(
                    f"{scene_id} object {obj_idx}: field mismatch vs reference: "
                    f"missing={sorted(reference_fields - fields)} "
                    f"extra={sorted(fields - reference_fields)}"
                )
            clip_ft = np.asarray(obj["clip_ft"], dtype=np.float32)
            if clip_ft.shape != (1024,) or not np.isfinite(clip_ft).all():
                raise ValueError(f"{scene_id} object {obj_idx}: invalid clip_ft")
            if float(np.linalg.norm(clip_ft)) <= 0:
                raise ValueError(f"{scene_id} object {obj_idx}: zero clip_ft")
        with open(visibility_path, "rb") as handle:
            visibility = pickle.load(handle)
        mappings = sum(len(v) for v in visibility["object_to_views"].values())
        generated_summaries.append(
            {
                "scene_id": scene_id,
                "objects": len(objects),
                "frames": len(raw_scene.rgb_paths),
                "visibility_mappings": mappings,
            }
        )
        total_objects += len(objects)
        total_frames += len(raw_scene.rgb_paths)
        total_mappings += mappings

    round_trip = _verify_nr3d_round_trip(nr3d_root=nr3d_root)
    return {
        "reference_pkl": str(reference_pkl),
        "schema_field_diff": {
            "missing_in_generated": [],
            "extra_in_generated": [],
            "reference_fields": sorted(reference_fields),
        },
        "total_scenes": len(scene_ids),
        "total_objects": total_objects,
        "mean_objects_per_scene": total_objects / len(scene_ids),
        "total_frames": total_frames,
        "mean_frames_per_scene": total_frames / len(scene_ids),
        "total_visibility_mappings": total_mappings,
        "round_trip": round_trip,
        "scenes": generated_summaries,
    }


def write_report(
    *,
    report_path: Path,
    scene_ids: list[str],
    step_status: dict[str, str],
    stats: list[SceneBuildStats],
    verification: dict[str, Any],
    wall_times: dict[str, float],
    nr3d_root: Path,
    deviations: list[str],
    open_issues: list[str],
) -> None:
    """Write the worker artifact requested by the tmux brief."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_root = nr3d_root / "scannet"
    disk_used = _du_human(output_root)
    lines = [
        "# NR3D GT ConceptGraph Producer Report",
        "",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Scene count: {len(scene_ids)}",
        f"- Output root: `{output_root.resolve()}`",
        f"- Disk used: `{disk_used}`",
        "",
        "## Files Created",
        "",
        "- `scripts/extract_nr3d_test_frames.py`",
        "- `src/scripts/nr3d_gt_conceptgraph.py`",
        "- `src/scripts/build_nr3d_gt_objects.py`",
        "- `src/scripts/extract_nr3d_gt_clip_feats.py`",
        "- `src/scripts/tests/test_nr3d_gt_conceptgraph.py`",
        "",
        "## Per-Step Status",
        "",
    ]
    for step in range(1, 9):
        key = f"Step {step}"
        lines.append(f"- {key}: {step_status.get(key, 'DEFERRED - not run')}")
    lines.extend(["", "## Wall-Clock Time", ""])
    for key, value in wall_times.items():
        lines.append(f"- {key}: {value:.1f}s")
    lines.extend(["", "## Build Stats", ""])
    lines.append(f"- Total scenes built: {len(stats)}")
    lines.append(f"- Total objects: {sum(item.num_objects for item in stats)}")
    lines.append(f"- Total frames: {sum(item.num_frames for item in stats)}")
    lines.append(
        f"- Total visibility mappings: {sum(item.num_visible_mappings for item in stats)}"
    )
    lines.append(f"- Total CLIP features: {sum(item.num_clip_features for item in stats)}")
    lines.extend(["", "## Verification", ""])
    lines.append("```json")
    lines.append(json.dumps(verification, indent=2, ensure_ascii=False)[:20000])
    lines.append("```")
    lines.extend(["", "## Deviations", ""])
    lines.extend([f"- {item}" for item in deviations] or ["- None"])
    lines.extend(["", "## Open Issues", ""])
    lines.extend([f"- {item}" for item in open_issues] or ["- None"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["verify-aux", "build-scenes", "verify-outputs"],
        help="Pipeline command to run.",
    )
    parser.add_argument("--nr3d-root", type=Path, default=DEFAULT_NR3D_ROOT)
    parser.add_argument("--scannet-root", type=Path, default=DEFAULT_SCANNET_ROOT)
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=DEFAULT_NR3D_ROOT / "raw/test_scans.txt",
    )
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--use-depth",
        action="store_true",
        help="Enable depth occlusion in visibility. Default is off to match the local OpenEQA reference.",
    )
    parser.add_argument("--max-distance", type=float, default=5.0)
    parser.add_argument("--min-visible-ratio", type=float, default=0.03)
    parser.add_argument("--min-visible-points", type=int, default=5)
    parser.add_argument(
        "--max-clip-views-per-object",
        type=int,
        default=DEFAULT_MAX_CLIP_VIEWS_PER_OBJECT,
    )
    parser.add_argument("--openeqa-reference", type=Path, default=DEFAULT_OPENEQA_REFERENCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_ids = args.scenes if args.scenes else load_scene_ids(args.scene_list)
    if args.command == "verify-aux":
        summary = verify_aux_files(
            scene_ids,
            nr3d_root=args.nr3d_root,
            scannet_root=args.scannet_root,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "build-scenes":
        if args.gpu == 1:
            raise ValueError("GPU 1 is broken on this server; choose another GPU")
        clip_runtime = create_clip_runtime(f"cuda:{args.gpu}")
        stats = [
            build_scene(
                scene_id,
                nr3d_root=args.nr3d_root,
                scannet_root=args.scannet_root,
                clip_runtime=clip_runtime,
                use_depth=args.use_depth,
                max_distance=args.max_distance,
                min_visible_ratio=args.min_visible_ratio,
                min_visible_points=args.min_visible_points,
                max_clip_views_per_object=args.max_clip_views_per_object,
            )
            for scene_id in scene_ids
        ]
        print(json.dumps([item.__dict__ for item in stats], indent=2))
        return

    if args.command == "verify-outputs":
        verification = verify_outputs(
            scene_ids,
            nr3d_root=args.nr3d_root,
            openeqa_reference=args.openeqa_reference,
        )
        print(json.dumps(verification, indent=2, ensure_ascii=False))
        return


def _aux_paths(aux_root: Path, scene_id: str) -> dict[str, Path]:
    scene_aux = aux_root / scene_id
    return {
        "metadata": scene_aux / f"{scene_id}.txt",
        "aggregation": scene_aux / f"{scene_id}.aggregation.json",
        "segments": scene_aux / f"{scene_id}_vh_clean_2.0.010000.segs.json",
    }


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{description} JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{description} JSON must be an object: {path}")
    return data


def _scannet_scene_dir(scannet_root: Path, scene_id: str) -> Path:
    candidate = scannet_root / "scans" / scene_id
    if candidate.exists():
        return candidate
    test_candidate = scannet_root / "scans_test" / scene_id
    if test_candidate.exists():
        return test_candidate
    return candidate


def _mesh_path(scannet_root: Path, scene_id: str) -> Path:
    return _scannet_scene_dir(scannet_root, scene_id) / f"{scene_id}_vh_clean_2.ply"


def _load_mesh_vertices(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not mesh_path.exists():
        raise FileNotFoundError(f"ScanNet mesh not found: {mesh_path}")
    ply = PlyData.read(mesh_path)
    vertex = ply["vertex"]
    names = vertex.data.dtype.names or ()
    required = {"x", "y", "z", "red", "green", "blue"}
    if not required.issubset(names):
        raise ValueError(f"Mesh {mesh_path} missing vertex fields {sorted(required - set(names))}")
    points = np.vstack([vertex.data["x"], vertex.data["y"], vertex.data["z"]]).T
    colors = np.vstack([vertex.data["red"], vertex.data["green"], vertex.data["blue"]]).T
    points = points.astype(np.float64)
    colors = colors.astype(np.float64)
    _require_finite(points, f"mesh points {mesh_path}")
    _require_finite(colors, f"mesh colors {mesh_path}")
    return points, colors


def _build_seg_to_vertices(seg_indices: list[Any]) -> dict[int, np.ndarray]:
    seg_to_vertices: dict[int, list[int]] = {}
    for vertex_idx, seg_idx in enumerate(seg_indices):
        seg_to_vertices.setdefault(int(seg_idx), []).append(vertex_idx)
    return {
        seg_idx: np.asarray(indices, dtype=np.int64)
        for seg_idx, indices in seg_to_vertices.items()
    }


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.shape != (4, 4):
        raise ValueError(f"axis alignment matrix must be 4x4, got {matrix.shape}")
    points = np.asarray(points, dtype=np.float64)
    points_h = np.hstack([points, np.ones((len(points), 1), dtype=np.float64)])
    return (matrix @ points_h.T).T[:, :3].astype(np.float64)


def _oriented_bbox_points(points: np.ndarray, *, scene_id: str, object_id: int) -> np.ndarray:
    if len(points) < 4:
        raise ValueError(f"{scene_id} objectId {object_id}: OBB needs at least 4 points")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    try:
        obb = pcd.get_oriented_bounding_box()
    except RuntimeError as exc:
        raise RuntimeError(
            f"{scene_id} objectId {object_id}: Open3D OBB failed"
        ) from exc
    bbox_np = np.asarray(obb.get_box_points(), dtype=np.float64)
    if bbox_np.shape != (8, 3):
        raise ValueError(f"{scene_id} objectId {object_id}: OBB shape {bbox_np.shape}")
    _require_finite(bbox_np, f"{scene_id} objectId {object_id} bbox_np")
    return bbox_np


def _require_int(value: Any, context: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{context} must be an int, got {value!r}")
    return int(value)


def _require_label(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} label must be a non-empty string")
    return value.strip()


def _require_finite(array: np.ndarray, context: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{context} contains NaN/Inf")


def _scene_class_names(scene_id: str, *, aux_root: Path) -> list[str]:
    aggregation = _load_json(_aux_paths(aux_root, scene_id)["aggregation"], "aggregation")
    seg_groups = aggregation.get("segGroups")
    if not isinstance(seg_groups, list):
        raise ValueError(f"{scene_id}: segGroups missing")
    names = sorted({_require_label(group.get("label"), f"{scene_id} group") for group in seg_groups})
    if not names:
        raise ValueError(f"{scene_id}: no class names in aggregation")
    return names


def _load_matrix(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Matrix file not found: {path}")
    matrix = np.loadtxt(path).astype(np.float64)
    if matrix.shape != shape:
        raise ValueError(f"{path} must have shape {shape}, got {matrix.shape}")
    _require_finite(matrix, str(path))
    return matrix


def _load_pose(scene_id: str, pose_path: Path) -> np.ndarray:
    pose = _load_matrix(pose_path, (4, 4))
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{scene_id}: invalid homogeneous row in pose {pose_path}")
    return pose


def _image_shape(path: Path, *, scene_id: str) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"{scene_id}: failed to read image {path}")
    height, width = image.shape[:2]
    return int(height), int(width)


def _require_matching_frame_ids(
    scene_id: str,
    rgb_paths: list[Path],
    depth_paths: list[Path],
    pose_paths: list[Path],
) -> None:
    rgb_ids = [path.name[:6] for path in rgb_paths]
    depth_ids = [path.name[:6] for path in depth_paths]
    pose_ids = [path.stem for path in pose_paths]
    if rgb_ids != depth_ids or rgb_ids != pose_ids:
        raise ValueError(f"{scene_id}: RGB/depth/pose frame ids do not match")


def _bbox_np_to_axis_aligned_9dof(bbox_np: np.ndarray) -> list[float]:
    bbox_np = np.asarray(bbox_np, dtype=np.float64)
    if bbox_np.shape != (8, 3):
        raise ValueError(f"bbox_np must have shape (8, 3), got {bbox_np.shape}")
    mins = bbox_np.min(axis=0)
    maxs = bbox_np.max(axis=0)
    dims = maxs - mins
    if np.any(dims <= 0):
        raise ValueError(f"bbox_np has non-positive extent: {dims.tolist()}")
    center = (mins + maxs) / 2.0
    return [float(v) for v in [*center.tolist(), *dims.tolist(), 0.0, 0.0, 0.0]]


def _validate_feature(feature: np.ndarray, context: str) -> np.ndarray:
    arr = np.asarray(feature, dtype=np.float32).reshape(-1)
    if arr.shape != (1024,):
        raise ValueError(f"{context} must have shape (1024,), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{context} contains NaN/Inf")
    if float(np.linalg.norm(arr)) <= 0:
        raise ValueError(f"{context} is zero")
    return arr.astype(np.float32)


def _add_projection_fallback_views(
    *,
    gt_objects: list[GtObject],
    raw_scene: RawScene,
    object_to_views: dict[int, list[tuple[int, float]]],
    view_to_objects: dict[int, list[tuple[int, float]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Add one bbox-projection view for GT objects missed by point visibility."""
    fallback_objects: list[dict[str, Any]] = []
    unobserved_objects: list[dict[str, Any]] = []
    intrinsic = raw_scene.intrinsic_color[:3, :3]
    image_size = (raw_scene.image_width, raw_scene.image_height)
    image_area = raw_scene.image_width * raw_scene.image_height
    for obj_idx, gt_object in enumerate(gt_objects):
        if object_to_views.get(obj_idx):
            continue
        bbox_9dof = _bbox_np_to_axis_aligned_9dof(gt_object.bbox_np)
        center = np.asarray(gt_object.pcd_np, dtype=np.float64).mean(axis=0)
        candidates: list[tuple[float, int, list[int]]] = []
        for view_id, pose in enumerate(raw_scene.poses):
            world_to_cam = np.linalg.inv(pose)
            rect = project_bbox_3d_to_2d(
                bbox_9dof,
                intrinsic,
                world_to_cam,
                image_size=image_size,
                depth_max=20.0,
            )
            if rect is None:
                continue
            x1, y1, x2, y2 = rect
            if x2 <= x1 or y2 <= y1:
                continue
            area = float((x2 - x1) * (y2 - y1))
            dist = float(np.linalg.norm(center - pose[:3, 3]))
            score = 1e-6 + 0.01 * min(1.0, area / image_area) + 0.01 / (1.0 + dist)
            candidates.append((score, view_id, list(rect)))
        if not candidates:
            if gt_object.is_background:
                unobserved_objects.append(
                    {
                        "object_index": obj_idx,
                        "object_id": gt_object.object_id,
                        "label": gt_object.label,
                        "reason": "no projected bbox in kept frames",
                    }
                )
                continue
            raise ValueError(
                f"{raw_scene.scene_id} objectId {gt_object.object_id}: "
                "no visible views and no projected bbox fallback views"
            )
        candidates.sort(reverse=True)
        score, view_id, rect = candidates[0]
        object_to_views[obj_idx] = [(view_id, float(score))]
        view_to_objects.setdefault(view_id, []).append((obj_idx, float(score)))
        fallback_objects.append(
            {
                "object_index": obj_idx,
                "object_id": gt_object.object_id,
                "label": gt_object.label,
                "view_id": view_id,
                "score": float(score),
                "xyxy": rect,
            }
        )
    for view_id in view_to_objects:
        view_to_objects[view_id].sort(key=lambda item: item[1], reverse=True)
    return fallback_objects, unobserved_objects


def _build_projection_draft(
    gt_object: GtObject,
    *,
    visible_views: list[tuple[int, float]],
    raw_scene: RawScene,
    inst_color: list[float],
) -> dict[str, Any]:
    views = sorted((int(view_id), float(score)) for view_id, score in visible_views)
    if not views:
        raise ValueError(f"{raw_scene.scene_id} objectId {gt_object.object_id}: no visible views")
    bbox_9dof = _bbox_np_to_axis_aligned_9dof(gt_object.bbox_np)
    intrinsic = raw_scene.intrinsic_color[:3, :3]
    xyxy: list[np.ndarray] = []
    rgb_paths: list[Path] = []
    for view_id, _score in views:
        world_to_cam = np.linalg.inv(raw_scene.poses[view_id])
        rect = project_bbox_3d_to_2d(
            bbox_9dof,
            intrinsic,
            world_to_cam,
            (raw_scene.image_width, raw_scene.image_height),
            depth_max=20.0,
        )
        if rect is None:
            raise ValueError(
                f"{raw_scene.scene_id} objectId {gt_object.object_id}: "
                f"visible view {view_id} has no projected bbox"
            )
        x1, y1, x2, y2 = rect
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"{raw_scene.scene_id} objectId {gt_object.object_id}: "
                f"degenerate projected bbox {rect}"
            )
        xyxy.append(np.asarray(rect, dtype=np.float32))
        rgb_paths.append(raw_scene.rgb_paths[view_id])
    return {
        "gt_object": gt_object,
        "visible_views": views,
        "xyxy": xyxy,
        "rgb_paths": rgb_paths,
        "inst_color": inst_color,
    }


def _encode_text_features(
    labels: list[str],
    *,
    model: Any,
    tokenizer: Any,
    device: str,
) -> dict[str, np.ndarray]:
    tokens = tokenizer(labels).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        encoded = model.encode_text(tokens)
        encoded = encoded / encoded.norm(dim=-1, keepdim=True)
    arr = encoded.float().cpu().numpy().astype(np.float32)
    return {label: arr[idx] for idx, label in enumerate(labels)}


def _preprocess_crop(
    rgb_path: Path,
    xyxy: np.ndarray,
    preprocess: Any,
    image_cache: dict[Path, Image.Image],
) -> torch.Tensor:
    if rgb_path not in image_cache:
        image_cache[rgb_path] = Image.open(rgb_path).convert("RGB")
    image = image_cache[rgb_path]
    x1, y1, x2, y2 = [int(round(float(value))) for value in xyxy]
    x1, y1, x2, y2 = _expand_crop_to_min_size(x1, y1, x2, y2, image.width, image.height)
    crop = image.crop((x1, y1, x2 + 1, y2 + 1))
    return preprocess(crop)


def _expand_crop_to_min_size(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
    min_size: int = 16,
) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
        x1 = max(0, x2 - 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
        y1 = max(0, y2 - 1)
    crop_w = x2 - x1 + 1
    crop_h = y2 - y1 + 1
    if crop_w < min_size:
        deficit = min_size - crop_w
        x1 = max(0, x1 - deficit // 2)
        x2 = min(width - 1, x2 + deficit - deficit // 2)
    if crop_h < min_size:
        deficit = min_size - crop_h
        y1 = max(0, y1 - deficit // 2)
        y2 = min(height - 1, y2 + deficit - deficit // 2)
    return x1, y1, x2, y2


def _find_reference_pkl(reference_dir: Path) -> Path:
    pcd_dir = reference_dir / "pcd_saves"
    matches = sorted(pcd_dir.glob("*clip019*_post.pkl.gz"))
    if not matches:
        matches = sorted(pcd_dir.glob("*_post.pkl.gz"))
    if not matches:
        raise FileNotFoundError(f"No OpenEQA reference pkl found under {pcd_dir}")
    return matches[0]


def _verify_nr3d_round_trip(*, nr3d_root: Path) -> dict[str, Any]:
    scene_id = "scene0164_00"
    csv_path = nr3d_root / "raw/nr3d.csv"
    target_row: dict[str, str] | None = None
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["scan_id"] == scene_id:
                target_row = row
                break
    if target_row is None:
        raise ValueError(f"No NR3D row found for {scene_id} in {csv_path}")
    target_id = int(target_row["target_id"])
    instance_type = target_row["instance_type"]
    pkl_path = nr3d_root / f"scannet/{scene_id}/conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz"
    with gzip.open(pkl_path, "rb") as handle:
        payload = pickle.load(handle)
    aggregation = _load_json(
        nr3d_root / f"scannet_aux/{scene_id}/{scene_id}.aggregation.json",
        "aggregation",
    )
    object_id_to_index = {
        int(group["objectId"]): idx for idx, group in enumerate(aggregation["segGroups"])
    }
    if target_id not in object_id_to_index:
        raise ValueError(f"{scene_id}: target_id {target_id} not in aggregation")
    obj = payload["objects"][object_id_to_index[target_id]]
    label = obj["class_name"][0]
    bbox_np = np.asarray(obj["bbox_np"])
    if label != instance_type:
        raise ValueError(
            f"{scene_id}: target label mismatch for {target_id}: "
            f"{label!r} vs {instance_type!r}"
        )
    if bbox_np.shape != (8, 3) or not np.isfinite(bbox_np).all():
        raise ValueError(f"{scene_id}: invalid bbox for target_id {target_id}")
    return {
        "scene_id": scene_id,
        "target_id": target_id,
        "instance_type": instance_type,
        "object_index": object_id_to_index[target_id],
        "bbox_shape": list(bbox_np.shape),
        "bbox_min": bbox_np.min(axis=0).tolist(),
        "bbox_max": bbox_np.max(axis=0).tolist(),
    }


def _du_human(path: Path) -> str:
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    units = ["B", "K", "M", "G", "T"]
    value = float(total)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}T"


if __name__ == "__main__":
    main()
