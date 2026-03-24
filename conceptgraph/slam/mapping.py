from __future__ import annotations

import torch
import torch.nn.functional as F

from conceptgraph.slam.models import DetectionList, MapObjectList
from conceptgraph.slam.utils import (
    compute_overlap_matrix_2set,
    merge_obj2_into_obj1,
)
from conceptgraph.utils.ious import (
    compute_3d_giou_accurate_batch,
    compute_3d_iou_accurate_batch,
    compute_giou_batch,
    compute_iou_batch,
)


def compute_spatial_similarities(
    cfg,
    detection_list: DetectionList,
    objects: MapObjectList,
) -> torch.Tensor:
    """Pairwise spatial similarity between detections and map objects.

    Args:
        cfg: config with ``spatial_sim_type`` attribute.
        detection_list: M new detections.
        objects: N existing map objects.

    Returns:
        (M, N) similarity tensor.
    """
    det_bboxes = detection_list.get_stacked_values_torch("bbox")
    obj_bboxes = objects.get_stacked_values_torch("bbox")

    sim_type = cfg.spatial_sim_type
    if sim_type == "iou":
        return compute_iou_batch(det_bboxes, obj_bboxes)
    if sim_type == "giou":
        return compute_giou_batch(det_bboxes, obj_bboxes)
    if sim_type == "iou_accurate":
        return compute_3d_iou_accurate_batch(det_bboxes, obj_bboxes)
    if sim_type == "giou_accurate":
        return compute_3d_giou_accurate_batch(det_bboxes, obj_bboxes)
    if sim_type == "overlap":
        overlap = compute_overlap_matrix_2set(cfg, objects, detection_list)
        return torch.from_numpy(overlap).T
    raise ValueError(f"Invalid spatial_sim_type: {sim_type}")


def compute_visual_similarities(
    cfg,
    detection_list: DetectionList,
    objects: MapObjectList,
) -> torch.Tensor:
    """Pairwise CLIP cosine similarity between detections and objects.

    Args:
        cfg: unused, kept for interface consistency.
        detection_list: M new detections.
        objects: N existing map objects.

    Returns:
        (M, N) similarity tensor.
    """
    det_fts = detection_list.get_stacked_values_torch("clip_ft").unsqueeze(
        -1
    )  # (M, D, 1)
    obj_fts = objects.get_stacked_values_torch("clip_ft").T.unsqueeze(0)  # (1, D, N)
    return F.cosine_similarity(det_fts, obj_fts, dim=1)


def aggregate_similarities(
    cfg,
    spatial_sim: torch.Tensor,
    visual_sim: torch.Tensor,
) -> torch.Tensor:
    """Combine spatial and visual similarities.

    Args:
        cfg: config with ``match_method`` and ``phys_bias``.
        spatial_sim: (M, N).
        visual_sim: (M, N).

    Returns:
        (M, N) aggregated similarity tensor.
    """
    if cfg.match_method == "sim_sum":
        return (1 + cfg.phys_bias) * spatial_sim + (1 - cfg.phys_bias) * visual_sim
    raise ValueError(f"Unknown match_method: {cfg.match_method}")


def merge_detections_to_objects(
    cfg,
    detection_list: DetectionList,
    objects: MapObjectList,
    agg_sim: torch.Tensor,
) -> MapObjectList:
    """Assign each detection to an object or create a new one."""
    for i in range(agg_sim.shape[0]):
        if agg_sim[i].max() == float("-inf"):
            objects.append(detection_list[i])
        else:
            j = agg_sim[i].argmax()
            objects[j] = merge_obj2_into_obj1(
                cfg, objects[j], detection_list[i], run_dbscan=False
            )
    return objects
