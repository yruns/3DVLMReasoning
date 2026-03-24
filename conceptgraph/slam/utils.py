from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import cv2
import faiss
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from conceptgraph.dataset.datasets_common import (
    from_intrinsics_matrix,
)
from conceptgraph.slam.models import DetectionList, MapObjectList
from conceptgraph.utils.general import to_numpy, to_tensor
from conceptgraph.utils.ious import (
    compute_3d_iou,
    compute_3d_iou_accurate_batch,
    compute_iou_batch,
    mask_subtract_contained,
)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def to_dataset_relative_path(
    path_value: str | Path | None,
    dataset_root: str | Path,
) -> str:
    """Convert *path_value* to a dataset-root-relative POSIX string."""
    if path_value is None:
        return ""
    p = Path(path_value)
    if not p.is_absolute():
        return p.as_posix()
    root = Path(dataset_root).resolve()
    try:
        return p.resolve().relative_to(root).as_posix()
    except ValueError:
        return Path(str(p.resolve()).replace(str(root), ".")).as_posix()


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


def get_classes_colors(
    classes: list[str],
) -> dict[int, tuple[float, float, float]]:
    """Random RGB colour per class index (plus -1 for unknown)."""
    colors: dict[int, tuple[float, float, float]] = {}
    for idx in range(len(classes)):
        colors[idx] = (
            np.random.randint(0, 256) / 255.0,
            np.random.randint(0, 256) / 255.0,
            np.random.randint(0, 256) / 255.0,
        )
    colors[-1] = (0.0, 0.0, 0.0)
    return colors


def create_or_load_colors(
    cfg: dict,
    filename: str = "gsa_classes_tag2text",
) -> tuple[list[str], dict]:
    """Load class list and per-class colours, creating if needed."""
    scene = Path(cfg["dataset_root"]) / cfg["scene_id"]
    classes_fp = scene / f"{filename}.json"
    with open(classes_fp) as f:
        classes = json.load(f)

    colors_fp = scene / f"{filename}_colors.json"
    if colors_fp.exists():
        with open(colors_fp) as f:
            class_colors = json.load(f)
        print("Loaded class colors from", colors_fp)
    else:
        class_colors = {str(k): v for k, v in get_classes_colors(classes).items()}
        with open(colors_fp, "w") as f:
            json.dump(class_colors, f)
        print("Saved class colors to", colors_fp)
    return classes, class_colors


# ---------------------------------------------------------------------------
# Point cloud processing
# ---------------------------------------------------------------------------


def create_object_pcd(
    depth_array: np.ndarray,
    mask: np.ndarray,
    cam_K: np.ndarray | torch.Tensor,
    image: np.ndarray,
    obj_color: np.ndarray | None = None,
) -> o3d.geometry.PointCloud:
    """Back-project masked depth pixels into a coloured point cloud."""
    fx, fy, cx, cy = from_intrinsics_matrix(cam_K)
    mask = np.logical_and(mask, depth_array > 0)

    if mask.sum() == 0:
        return o3d.geometry.PointCloud()

    h, w = depth_array.shape
    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))

    d = depth_array[mask]
    u = u[mask]
    v = v[mask]

    x = (u - cx) * d / fx
    y = (v - cy) * d / fy

    points = np.stack((x, y, d), axis=-1).reshape(-1, 3)
    points += np.random.normal(0, 4e-3, points.shape)

    colors = (
        image[mask] / 255.0 if obj_color is None else np.full(points.shape, obj_color)
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def pcd_denoise_dbscan(
    pcd: o3d.geometry.PointCloud,
    eps: float = 0.02,
    min_points: int = 10,
) -> o3d.geometry.PointCloud:
    """Keep only the largest DBSCAN cluster."""
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)

    counter = Counter(labels)
    counter.pop(-1, None)

    if not counter:
        return pcd

    largest_label = counter.most_common(1)[0][0]
    keep = labels == largest_label

    if keep.sum() < 5:
        return pcd

    clean = o3d.geometry.PointCloud()
    clean.points = o3d.utility.Vector3dVector(pts[keep])
    clean.colors = o3d.utility.Vector3dVector(cols[keep])
    return clean


def process_pcd(
    pcd: o3d.geometry.PointCloud,
    cfg,
    run_dbscan: bool = True,
) -> o3d.geometry.PointCloud:
    """Downsample and optionally denoise a point cloud."""
    pcd = pcd.voxel_down_sample(voxel_size=cfg.downsample_voxel_size)
    if cfg.dbscan_remove_noise and run_dbscan:
        pcd = pcd_denoise_dbscan(
            pcd,
            eps=cfg.dbscan_eps,
            min_points=cfg.dbscan_min_points,
        )
    return pcd


def get_bounding_box(
    cfg, pcd: o3d.geometry.PointCloud
) -> o3d.geometry.OrientedBoundingBox | o3d.geometry.AxisAlignedBoundingBox:
    """Choose oriented or axis-aligned bounding box based on config."""
    use_oriented = (
        "accurate" in cfg.spatial_sim_type or "overlap" in cfg.spatial_sim_type
    ) and len(pcd.points) >= 4

    if use_oriented:
        try:
            return pcd.get_oriented_bounding_box(robust=True)
        except RuntimeError as e:
            print(f"Met {e}, using axis-aligned bounding box instead")
    return pcd.get_axis_aligned_bounding_box()


# ---------------------------------------------------------------------------
# Object merging
# ---------------------------------------------------------------------------


def merge_obj2_into_obj1(
    cfg,
    obj1: dict,
    obj2: dict,
    run_dbscan: bool = True,
) -> dict:
    """Merge *obj2* into *obj1* in-place and return *obj1*."""
    n1 = obj1["num_detections"]
    n2 = obj2["num_detections"]

    for k in obj1:
        if k in ("pcd", "bbox", "clip_ft", "text_ft"):
            continue
        if k == "caption":
            for k2, v2 in obj2["caption"].items():
                obj1["caption"][k2 + n1] = v2
        elif k == "inst_color":
            pass  # keep original colour
        elif isinstance(obj1[k], (list, int)):
            obj1[k] += obj2[k]
        else:
            raise NotImplementedError(f"Cannot merge key '{k}' of type {type(obj1[k])}")

    obj1["pcd"] += obj2["pcd"]
    obj1["pcd"] = process_pcd(obj1["pcd"], cfg, run_dbscan=run_dbscan)
    obj1["bbox"] = get_bounding_box(cfg, obj1["pcd"])
    obj1["bbox"].color = [0, 1, 0]

    obj1["clip_ft"] = F.normalize(
        (obj1["clip_ft"] * n1 + obj2["clip_ft"] * n2) / (n1 + n2),
        dim=0,
    )

    obj1["text_ft"] = to_tensor(obj1["text_ft"], cfg.device)
    obj2["text_ft"] = to_tensor(obj2["text_ft"], cfg.device)
    obj1["text_ft"] = F.normalize(
        (obj1["text_ft"] * n1 + obj2["text_ft"] * n2) / (n1 + n2),
        dim=0,
    )

    return obj1


# ---------------------------------------------------------------------------
# Overlap computation
# ---------------------------------------------------------------------------


def compute_overlap_matrix(cfg, objects: MapObjectList) -> np.ndarray:
    """Pairwise point-level overlap ratio (N x N)."""
    n = len(objects)
    overlap = np.zeros((n, n))

    pts = [np.asarray(o["pcd"].points, dtype=np.float32) for o in objects]
    indices = [faiss.IndexFlatL2(a.shape[1]) for a in pts]
    for idx, a in zip(indices, pts):
        idx.add(a)

    thresh_sq = cfg.downsample_voxel_size**2
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if compute_3d_iou(objects[i]["bbox"], objects[j]["bbox"]) == 0:
                continue
            D, _ = indices[j].search(pts[i], 1)
            overlap[i, j] = (D < thresh_sq).sum() / len(pts[i])

    return overlap


def compute_overlap_matrix_2set(
    cfg,
    objects_map: MapObjectList,
    objects_new: DetectionList,
) -> np.ndarray:
    """Point-level overlap ratio between map objects and new detections.

    Returns:
        (m, n) matrix where m = len(objects_map), n = len(objects_new).
    """
    m, n = len(objects_map), len(objects_new)
    overlap = np.zeros((m, n))

    pts_map = [np.asarray(o["pcd"].points, dtype=np.float32) for o in objects_map]
    indices = [faiss.IndexFlatL2(a.shape[1]) for a in pts_map]
    for idx, a in zip(indices, pts_map):
        idx.add(a)

    pts_new = [np.asarray(o["pcd"].points, dtype=np.float32) for o in objects_new]

    bbox_map = objects_map.get_stacked_values_torch("bbox")
    bbox_new = objects_new.get_stacked_values_torch("bbox")

    try:
        iou = compute_3d_iou_accurate_batch(bbox_map, bbox_new)
    except ValueError:
        print("Coplanar vertex error; falling back to axis-aligned IoU")
        bbox_map_aa = torch.from_numpy(
            np.stack(
                [
                    np.asarray(p.get_axis_aligned_bounding_box().get_box_points())
                    for p in objects_map.get_values("pcd")
                ]
            )
        )
        bbox_new_aa = torch.from_numpy(
            np.stack(
                [
                    np.asarray(p.get_axis_aligned_bounding_box().get_box_points())
                    for p in objects_new.get_values("pcd")
                ]
            )
        )
        iou = compute_iou_batch(bbox_map_aa, bbox_new_aa)

    thresh_sq = cfg.downsample_voxel_size**2
    for i in range(m):
        for j in range(n):
            if iou[i, j] < 1e-6:
                continue
            D, _ = indices[i].search(pts_new[j], 1)
            overlap[i, j] = (D < thresh_sq).sum() / len(pts_new[j])

    return overlap


# ---------------------------------------------------------------------------
# High-level map operations
# ---------------------------------------------------------------------------


def merge_overlap_objects(
    cfg,
    objects: MapObjectList,
    overlap_matrix: np.ndarray,
) -> MapObjectList:
    """Merge object pairs that exceed overlap and similarity thresholds."""
    x, y = overlap_matrix.nonzero()
    ratios = overlap_matrix[x, y]
    order = np.argsort(ratios)[::-1]
    x, y, ratios = x[order], y[order], ratios[order]

    kept = np.ones(len(objects), dtype=bool)
    for i, j, ratio in zip(x, y, ratios):
        if ratio <= cfg.merge_overlap_thresh:
            break
        vis_sim = F.cosine_similarity(
            to_tensor(objects[i]["clip_ft"]),
            to_tensor(objects[j]["clip_ft"]),
            dim=0,
        )
        txt_sim = F.cosine_similarity(
            to_tensor(objects[i]["text_ft"]),
            to_tensor(objects[j]["text_ft"]),
            dim=0,
        )
        if (
            vis_sim > cfg.merge_visual_sim_thresh
            and txt_sim > cfg.merge_text_sim_thresh
            and kept[j]
        ):
            objects[j] = merge_obj2_into_obj1(
                cfg, objects[j], objects[i], run_dbscan=True
            )
            kept[i] = False

    return MapObjectList([o for o, k in zip(objects, kept) if k])


def denoise_objects(cfg, objects: MapObjectList) -> MapObjectList:
    """Re-process point clouds to remove noise."""
    for i in range(len(objects)):
        original = objects[i]["pcd"]
        objects[i]["pcd"] = process_pcd(objects[i]["pcd"], cfg, run_dbscan=True)
        if len(objects[i]["pcd"].points) < 4:
            objects[i]["pcd"] = original
            continue
        objects[i]["bbox"] = get_bounding_box(cfg, objects[i]["pcd"])
        objects[i]["bbox"].color = [0, 1, 0]
    return objects


def filter_objects(cfg, objects: MapObjectList) -> MapObjectList:
    """Remove objects with too few points or detections."""
    print("Before filtering:", len(objects))
    kept = MapObjectList(
        [
            o
            for o in objects
            if len(o["pcd"].points) >= cfg.obj_min_points
            and o["num_detections"] >= cfg.obj_min_detections
        ]
    )
    print("After filtering:", len(kept))
    return kept


def merge_objects(cfg, objects: MapObjectList) -> MapObjectList:
    """Merge overlapping objects if threshold is positive."""
    if cfg.merge_overlap_thresh > 0:
        overlap = compute_overlap_matrix(cfg, objects)
        print("Before merging:", len(objects))
        objects = merge_overlap_objects(cfg, objects, overlap)
        print("After merging:", len(objects))
    return objects


# ---------------------------------------------------------------------------
# Detection filtering / resizing / conversion
# ---------------------------------------------------------------------------


def filter_gobs(
    cfg: DictConfig,
    gobs: dict,
    image: np.ndarray,
    bg_classes: list[str] | None = None,
) -> dict:
    """Filter detections by area, confidence and class."""
    if bg_classes is None:
        bg_classes = ["wall", "floor", "ceiling"]

    if len(gobs["xyxy"]) == 0:
        return gobs

    keep: list[int] = []
    for idx in range(len(gobs["xyxy"])):
        cls_id = gobs["class_id"][idx]
        cls_name = gobs["classes"][cls_id]

        if gobs["mask"][idx].sum() < max(cfg.mask_area_threshold, 10):
            continue
        if cfg.skip_bg and cls_name in bg_classes:
            continue
        if cls_name not in bg_classes:
            x1, y1, x2, y2 = gobs["xyxy"][idx]
            bbox_area = (x2 - x1) * (y2 - y1)
            img_area = image.shape[0] * image.shape[1]
            if bbox_area > cfg.max_bbox_area_ratio * img_area:
                continue
        if gobs["confidence"] is not None:
            if gobs["confidence"][idx] < cfg.mask_conf_threshold:
                continue
        keep.append(idx)

    for k in gobs:
        if isinstance(gobs[k], str) or k == "classes":
            continue
        if isinstance(gobs[k], list):
            gobs[k] = [gobs[k][i] for i in keep]
        elif isinstance(gobs[k], np.ndarray):
            gobs[k] = gobs[k][keep]
        else:
            raise NotImplementedError(f"Unhandled type {type(gobs[k])}")
    return gobs


def resize_gobs(gobs: dict, image: np.ndarray) -> dict:
    """Resize masks and xyxy to match *image* dimensions."""
    new_masks: list[np.ndarray] = []
    for idx in range(len(gobs["xyxy"])):
        mask = gobs["mask"][idx]
        if mask.shape == image.shape[:2]:
            continue
        x1, y1, x2, y2 = gobs["xyxy"][idx]
        h_img, w_img = image.shape[:2]
        h_mask, w_mask = mask.shape
        x1 = round(x1 * w_img / w_mask)
        y1 = round(y1 * h_img / h_mask)
        x2 = round(x2 * w_img / w_mask)
        y2 = round(y2 * h_img / h_mask)
        gobs["xyxy"][idx] = [x1, y1, x2, y2]

        resized = cv2.resize(
            mask.astype(np.uint8),
            (w_img, h_img),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        new_masks.append(resized)

    if new_masks:
        gobs["mask"] = np.asarray(new_masks)
    return gobs


def gobs_to_detection_list(
    cfg,
    image: np.ndarray,
    depth_array: np.ndarray,
    cam_K: np.ndarray | torch.Tensor,
    idx: int,
    gobs: dict,
    trans_pose: np.ndarray | None = None,
    class_names: list[str] | None = None,
    bg_classes: list[str] | None = None,
    color_path: str | Path | None = None,
) -> tuple[DetectionList, DetectionList]:
    """Build foreground and background DetectionLists from raw detections."""
    if bg_classes is None:
        bg_classes = ["wall", "floor", "ceiling"]

    fg = DetectionList()
    bg = DetectionList()
    rel_color = to_dataset_relative_path(color_path, cfg.dataset_root)

    gobs = resize_gobs(gobs, image)
    gobs = filter_gobs(cfg, gobs, image, bg_classes)

    if len(gobs["xyxy"]) == 0:
        return fg, bg

    gobs["mask"] = mask_subtract_contained(gobs["xyxy"], gobs["mask"])

    for mask_idx in range(len(gobs["xyxy"])):
        local_cls = gobs["class_id"][mask_idx]
        mask = gobs["mask"][mask_idx]
        cls_name = gobs["classes"][local_cls]
        global_cls = -1 if class_names is None else class_names.index(cls_name)

        cam_pcd = create_object_pcd(depth_array, mask, cam_K, image)
        if len(cam_pcd.points) < max(cfg.min_points_threshold, 5):
            continue

        global_pcd = (
            cam_pcd.transform(trans_pose) if trans_pose is not None else cam_pcd
        )
        global_pcd = process_pcd(global_pcd, cfg)

        bbox = get_bounding_box(cfg, global_pcd)
        bbox.color = [0, 1, 0]
        if bbox.volume() < 1e-6:
            continue

        det = {
            "image_idx": [idx],
            "mask_idx": [mask_idx],
            "color_path": [rel_color],
            "class_name": [cls_name],
            "class_id": [global_cls],
            "num_detections": 1,
            "mask": [mask],
            "xyxy": [gobs["xyxy"][mask_idx]],
            "conf": [gobs["confidence"][mask_idx]],
            "n_points": [len(global_pcd.points)],
            "pixel_area": [mask.sum()],
            "contain_number": [None],
            "inst_color": np.random.rand(3),
            "is_background": cls_name in bg_classes,
            "pcd": global_pcd,
            "bbox": bbox,
            "clip_ft": to_tensor(gobs["image_feats"][mask_idx]),
            "text_ft": to_tensor(gobs["text_feats"][mask_idx]),
        }

        if cls_name in bg_classes:
            bg.append(det)
        else:
            fg.append(det)

    return fg, bg


def transform_detection_list(
    detection_list: DetectionList,
    transform: torch.Tensor | np.ndarray,
    deepcopy: bool = False,
) -> DetectionList:
    """Apply a 4x4 rigid transform to every detection's geometry."""
    t = to_numpy(transform)
    if deepcopy:
        detection_list = copy.deepcopy(detection_list)
    for d in detection_list:
        d["pcd"] = d["pcd"].transform(t)
        d["bbox"] = d["bbox"].rotate(t[:3, :3], center=(0, 0, 0))
        d["bbox"] = d["bbox"].translate(t[:3, 3])
    return detection_list
