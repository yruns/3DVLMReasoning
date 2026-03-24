from __future__ import annotations

import numpy as np
import open3d as o3d
import torch

# ---------------------------------------------------------------------------
# Single-pair helpers (Open3D bounding boxes)
# ---------------------------------------------------------------------------


def compute_3d_iou(
    bbox1: o3d.geometry.OrientedBoundingBox | o3d.geometry.AxisAlignedBoundingBox,
    bbox2: o3d.geometry.OrientedBoundingBox | o3d.geometry.AxisAlignedBoundingBox,
    padding: float = 0,
    use_iou: bool = True,
) -> float:
    """Axis-aligned 3-D IoU (or max overlap) between two bounding boxes."""
    bbox1_min = np.asarray(bbox1.get_min_bound()) - padding
    bbox1_max = np.asarray(bbox1.get_max_bound()) + padding
    bbox2_min = np.asarray(bbox2.get_min_bound()) - padding
    bbox2_max = np.asarray(bbox2.get_max_bound()) + padding

    overlap_min = np.maximum(bbox1_min, bbox2_min)
    overlap_max = np.minimum(bbox1_max, bbox2_max)
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)

    overlap_vol = np.prod(overlap_size)
    vol1 = np.prod(bbox1_max - bbox1_min)
    vol2 = np.prod(bbox2_max - bbox2_min)

    if use_iou:
        return overlap_vol / (vol1 + vol2 - overlap_vol)
    return max(overlap_vol / vol1, overlap_vol / vol2)


def compute_3d_giou(
    bbox1: o3d.geometry.OrientedBoundingBox | o3d.geometry.AxisAlignedBoundingBox,
    bbox2: o3d.geometry.OrientedBoundingBox | o3d.geometry.AxisAlignedBoundingBox,
) -> float:
    """Axis-aligned 3-D Generalised IoU between two bounding boxes."""
    bbox1_min = np.asarray(bbox1.get_min_bound())
    bbox1_max = np.asarray(bbox1.get_max_bound())
    bbox2_min = np.asarray(bbox2.get_min_bound())
    bbox2_max = np.asarray(bbox2.get_max_bound())

    inter_min = np.maximum(bbox1_min, bbox2_min)
    inter_max = np.minimum(bbox1_max, bbox2_max)
    inter_vol = np.prod(np.maximum(inter_max - inter_min, 0.0))

    vol1 = np.prod(bbox1_max - bbox1_min)
    vol2 = np.prod(bbox2_max - bbox2_min)
    union_vol = vol1 + vol2 - inter_vol

    iou = inter_vol / union_vol

    enc_min = np.minimum(bbox1_min, bbox2_min)
    enc_max = np.maximum(bbox1_max, bbox2_max)
    enc_vol = np.prod(np.maximum(enc_max - enc_min, 0.0))

    return iou - (enc_vol - union_vol) / enc_vol


# ---------------------------------------------------------------------------
# Batch helpers (torch tensors) -- axis-aligned approximation
# ---------------------------------------------------------------------------


def compute_iou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """Batch axis-aligned 3-D IoU.

    Args:
        bbox1: (M, V, D) box corners, e.g. (M, 8, 3).
        bbox2: (N, V, D) box corners.

    Returns:
        (M, N) IoU matrix.
    """
    b1_min = bbox1.min(dim=1).values.unsqueeze(1)  # (M, 1, D)
    b1_max = bbox1.max(dim=1).values.unsqueeze(1)
    b2_min = bbox2.min(dim=1).values.unsqueeze(0)  # (1, N, D)
    b2_max = bbox2.max(dim=1).values.unsqueeze(0)

    inter_min = torch.max(b1_min, b2_min)
    inter_max = torch.min(b1_max, b2_max)
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)

    vol1 = torch.prod(b1_max - b1_min, dim=2)
    vol2 = torch.prod(b2_max - b2_min, dim=2)

    return inter_vol / (vol1 + vol2 - inter_vol + 1e-10)


def compute_giou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """Batch axis-aligned 3-D GIoU.

    Args:
        bbox1: (M, V, D) box corners.
        bbox2: (N, V, D) box corners.

    Returns:
        (M, N) GIoU matrix.
    """
    b1_min = bbox1.min(dim=1).values.unsqueeze(1)
    b1_max = bbox1.max(dim=1).values.unsqueeze(1)
    b2_min = bbox2.min(dim=1).values.unsqueeze(0)
    b2_max = bbox2.max(dim=1).values.unsqueeze(0)

    inter_min = torch.max(b1_min, b2_min)
    inter_max = torch.min(b1_max, b2_max)
    enc_min = torch.min(b1_min, b2_min)
    enc_max = torch.max(b1_max, b2_max)

    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)
    enc_vol = torch.prod(enc_max - enc_min, dim=2)

    vol1 = torch.prod(b1_max - b1_min, dim=2)
    vol2 = torch.prod(b2_max - b2_min, dim=2)
    union_vol = vol1 + vol2 - inter_vol

    iou = inter_vol / (union_vol + 1e-10)
    return iou - (enc_vol - union_vol) / (enc_vol + 1e-10)


# ---------------------------------------------------------------------------
# Accurate (oriented) batch helpers -- require pytorch3d
# ---------------------------------------------------------------------------

# Open3D corner order:  ---, +--, -+-, --+, +++, -++, +-+, ++-
# pytorch3d expects:    ---, -+-, -++, --+, +--, ++-, +++, +-+
_O3D_TO_P3D = [0, 2, 5, 3, 1, 7, 4, 6]


def expand_3d_box(bbox: torch.Tensor, eps: float = 0.02) -> torch.Tensor:
    """Expand each side of an oriented box to at least *eps* length.

    Args:
        bbox: (N, 8, D) box corners in Open3D convention.

    Returns:
        (N, 8, D) expanded corners.
    """
    center = bbox.mean(dim=1)  # (N, D)

    va = bbox[:, 1, :] - bbox[:, 0, :]
    vb = bbox[:, 2, :] - bbox[:, 0, :]
    vc = bbox[:, 3, :] - bbox[:, 0, :]

    a = torch.linalg.vector_norm(va, ord=2, dim=1, keepdim=True)
    b = torch.linalg.vector_norm(vb, ord=2, dim=1, keepdim=True)
    c = torch.linalg.vector_norm(vc, ord=2, dim=1, keepdim=True)

    va = torch.where(a < eps, va / a * eps, va)
    vb = torch.where(b < eps, vb / b * eps, vb)
    vc = torch.where(c < eps, vc / c * eps, vc)

    new_bbox = torch.stack(
        [
            center - va / 2 - vb / 2 - vc / 2,
            center + va / 2 - vb / 2 - vc / 2,
            center - va / 2 + vb / 2 - vc / 2,
            center - va / 2 - vb / 2 + vc / 2,
            center + va / 2 + vb / 2 + vc / 2,
            center - va / 2 + vb / 2 + vc / 2,
            center + va / 2 - vb / 2 + vc / 2,
            center + va / 2 + vb / 2 - vc / 2,
        ],
        dim=1,
    )
    return new_bbox.to(dtype=bbox.dtype, device=bbox.device)


def compute_3d_box_volume_batch(bbox: torch.Tensor) -> torch.Tensor:
    """Volume of axis-aligned rectangular boxes (Open3D corner order).

    Args:
        bbox: (N, 8, D).

    Returns:
        (N,) volumes.
    """
    a = torch.linalg.vector_norm(bbox[:, 0] - bbox[:, 1], ord=2, dim=1)
    b = torch.linalg.vector_norm(bbox[:, 0] - bbox[:, 2], ord=2, dim=1)
    c = torch.linalg.vector_norm(bbox[:, 0] - bbox[:, 3], ord=2, dim=1)
    return a * b * c


def compute_3d_iou_accurate_batch(
    bbox1: torch.Tensor, bbox2: torch.Tensor
) -> torch.Tensor:
    """Oriented 3-D IoU via pytorch3d.

    Args:
        bbox1: (M, 8, 3).
        bbox2: (N, 8, 3).

    Returns:
        (M, N) IoU matrix.
    """
    import pytorch3d.ops as ops

    bbox1 = expand_3d_box(bbox1, 0.02)
    bbox2 = expand_3d_box(bbox2, 0.02)

    _, iou = ops.box3d_overlap(
        bbox1[:, _O3D_TO_P3D].float(),
        bbox2[:, _O3D_TO_P3D].float(),
    )
    return iou


def compute_3d_giou_accurate(obj1: dict, obj2: dict) -> float:
    """Oriented 3-D GIoU for a single pair of map objects."""
    import pytorch3d.ops as ops

    bbox1 = obj1["bbox"]
    bbox2 = obj2["bbox"]
    pcd1 = obj1["pcd"]
    pcd2 = obj2["pcd"]

    pts1 = np.asarray(bbox1.get_box_points())[_O3D_TO_P3D]
    pts2 = np.asarray(bbox2.get_box_points())[_O3D_TO_P3D]

    try:
        vols, ious = ops.box3d_overlap(
            torch.from_numpy(pts1).unsqueeze(0).float(),
            torch.from_numpy(pts2).unsqueeze(0).float(),
        )
        union_volume = vols[0, 0].item()
        iou = ious[0, 0].item()
    except ValueError:
        union_volume = 0.0
        iou = 0.0

    pcd_union = pcd1 + pcd2
    enc_box = pcd_union.get_oriented_bounding_box()
    enc_vol = enc_box.volume()

    return iou - (enc_vol - union_volume) / enc_vol


def compute_enclosing_vol(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """Pairwise enclosing volume via convex hull (accurate, slow).

    Args:
        bbox1: (M, 8, D).
        bbox2: (N, 8, D).

    Returns:
        (M, N) enclosing volumes.
    """
    M, N = bbox1.shape[0], bbox2.shape[0]
    enc_vol = torch.zeros((M, N), dtype=bbox1.dtype, device=bbox1.device)

    for i in range(M):
        for j in range(N):
            pts = torch.cat([bbox1[i], bbox2[j]], dim=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
            mesh, _ = pcd.compute_convex_hull(joggle_inputs=True)
            try:
                enc_vol[i, j] = mesh.get_volume()
            except RuntimeError:
                aabb = pcd.get_axis_aligned_bounding_box()
                enc_vol[i, j] = aabb.volume()

    return enc_vol


def compute_enclosing_vol_fast(
    bbox1: torch.Tensor, bbox2: torch.Tensor
) -> torch.Tensor:
    """Pairwise enclosing volume via AABB (fast, approximate).

    Args:
        bbox1: (M, 8, 3).
        bbox2: (N, 8, 3).

    Returns:
        (M, N) enclosing volumes.
    """
    M, N = bbox1.shape[0], bbox2.shape[0]

    b1 = bbox1.unsqueeze(1).expand(-1, N, -1, -1)
    b2 = bbox2.unsqueeze(0).expand(M, -1, -1, -1)

    min_coords = torch.minimum(b1, b2).amin(dim=2)
    max_coords = torch.maximum(b1, b2).amax(dim=2)
    dims = torch.clamp(max_coords - min_coords, min=0)

    return dims[:, :, 0] * dims[:, :, 1] * dims[:, :, 2]


def compute_3d_giou_accurate_batch(
    bbox1: torch.Tensor, bbox2: torch.Tensor
) -> torch.Tensor:
    """Oriented 3-D GIoU via pytorch3d.

    Args:
        bbox1: (M, 8, D).
        bbox2: (N, 8, D).

    Returns:
        (M, N) GIoU matrix.
    """
    import pytorch3d.ops as ops

    bbox1 = expand_3d_box(bbox1, 0.02)
    bbox2 = expand_3d_box(bbox2, 0.02)

    vol1 = compute_3d_box_volume_batch(bbox1)
    vol2 = compute_3d_box_volume_batch(bbox2)

    inter_vol, iou = ops.box3d_overlap(
        bbox1[:, _O3D_TO_P3D].float(),
        bbox2[:, _O3D_TO_P3D].float(),
    )
    union_vol = vol1.unsqueeze(1) + vol2.unsqueeze(0) - inter_vol
    enc_vol = compute_enclosing_vol(bbox1, bbox2)

    return iou - (enc_vol - union_vol) / enc_vol


def compute_3d_contain_ratio_accurate_batch(
    bbox1: torch.Tensor, bbox2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """How much of bbox1[i] is contained in bbox2[j].

    Args:
        bbox1: (M, 8, D).
        bbox2: (N, 8, D).

    Returns:
        Tuple of (M, N) containment ratios and (M, N) IoU.
    """
    import pytorch3d.ops as ops

    bbox1 = expand_3d_box(bbox1)
    bbox2 = expand_3d_box(bbox2)

    vol1 = compute_3d_box_volume_batch(bbox1)

    inter_vol, iou = ops.box3d_overlap(
        bbox1[:, _O3D_TO_P3D].float(),
        bbox2[:, _O3D_TO_P3D].float(),
    )
    contain_ratio = (inter_vol / vol1.unsqueeze(1)).clamp(0, 1)
    return contain_ratio, iou


# ---------------------------------------------------------------------------
# 2-D helpers
# ---------------------------------------------------------------------------


def compute_2d_box_contained_batch(
    bbox: torch.Tensor, thresh: float = 0.95
) -> torch.Tensor:
    """Count how many other boxes contain each box (2-D).

    Args:
        bbox: (N, 4) in (x1, y1, x2, y2) format.
        thresh: intersection-over-own-area threshold.

    Returns:
        (N,) counts (excluding self).
    """
    areas = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

    lt = torch.max(bbox[:, None, :2], bbox[:, :2])
    rb = torch.min(bbox[:, None, 2:], bbox[:, 2:])
    inter = (rb - lt).clamp(min=0)
    inter_areas = inter[:, :, 0] * inter[:, :, 1]

    mask = inter_areas > (areas * thresh).unsqueeze(1)
    return mask.sum(dim=1) - 1


def mask_subtract_contained(
    xyxy: np.ndarray,
    mask: np.ndarray,
    th1: float = 0.8,
    th2: float = 0.7,
) -> np.ndarray:
    """Subtract contained masks from their containers.

    For each mask, subtract the mask of boxes that are contained by it.

    Args:
        xyxy: (N, 4) boxes in (x1, y1, x2, y2) format.
        mask: (N, H, W) binary masks.
        th1: threshold for intersection-over-box2.
        th2: threshold for intersection-over-box1.

    Returns:
        (N, H, W) adjusted binary masks.
    """
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

    lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])
    rb = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])
    inter = (rb - lt).clip(min=0)
    inter_areas = inter[:, :, 0] * inter[:, :, 1]

    inter_over_box1 = inter_areas / areas[:, None]
    inter_over_box2 = inter_over_box1.T

    contained = (inter_over_box1 < th2) & (inter_over_box2 > th1)
    rows, cols = contained.nonzero()

    mask_sub = mask.copy()
    for r, c in zip(rows, cols):
        mask_sub[r] = mask_sub[r] & (~mask_sub[c])

    return mask_sub
