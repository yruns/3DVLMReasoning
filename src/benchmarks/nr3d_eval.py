"""NR3D visual grounding evaluation.

The metric is detection-style 9-DOF 3D IoU against EmbodiedScan-derived
ground-truth boxes, reported as Acc@0.25, Acc@0.50, and mean IoU.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from .embodiedscan_eval import compute_oriented_iou_3d
from .nr3d_loader import Nr3dVGSample


def evaluate_vg_predictions(
    predictions: list[dict[str, Any]],
    samples: list[Nr3dVGSample],
) -> dict[str, Any]:
    """Evaluate NR3D VG predictions against ground-truth bboxes.

    Args:
        predictions: List of ``{"sample_id": str, "bbox_3d": [floats] | None}``.
        samples: Ground-truth NR3D samples.

    Returns:
        ``acc_025``, ``acc_050``, ``mean_iou``, ``per_category``, and
        ``num_samples``.

    Raises:
        ValueError: If samples/predictions are empty or prediction ids are
            not present in the samples.
    """
    if not samples:
        raise ValueError("samples is empty")
    if not predictions:
        raise ValueError("predictions is empty")

    sample_map = {sample.sample_id: sample for sample in samples}
    pred_map: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        sample_id = prediction.get("sample_id")
        if not isinstance(sample_id, str):
            raise ValueError(f"Prediction missing string sample_id: {prediction!r}")
        if sample_id not in sample_map:
            raise ValueError(f"Prediction sample_id {sample_id!r} not in dataset")
        if sample_id in pred_map:
            raise ValueError(f"Duplicate prediction sample_id {sample_id!r}")
        pred_map[sample_id] = prediction

    ious: list[float] = []
    cat_ious: dict[str, list[float]] = {}

    for sample in samples:
        prediction = pred_map.get(sample.sample_id)
        pred_bbox = prediction.get("bbox_3d") if prediction is not None else None
        iou = _compute_prediction_iou(sample, pred_bbox)
        ious.append(iou)
        cat_ious.setdefault(sample.target, []).append(iou)

    ious_arr = np.array(ious, dtype=np.float64)
    per_category: dict[str, dict[str, Any]] = {}
    for category, values in sorted(cat_ious.items()):
        arr = np.array(values, dtype=np.float64)
        per_category[category] = {
            "acc_025": float((arr >= 0.25).mean()),
            "acc_050": float((arr >= 0.50).mean()),
            "mean_iou": float(arr.mean()),
            "count": len(values),
        }

    return {
        "acc_025": float((ious_arr >= 0.25).mean()),
        "acc_050": float((ious_arr >= 0.50).mean()),
        "mean_iou": float(ious_arr.mean()),
        "per_category": per_category,
        "num_samples": len(samples),
    }


def _compute_prediction_iou(sample: Nr3dVGSample, pred_bbox: Any) -> float:
    if pred_bbox is None or sample.gt_bbox_3d is None:
        return 0.0
    pred = _coerce_bbox_9dof(pred_bbox, sample.sample_id, "pred")
    gt = _coerce_bbox_9dof(sample.gt_bbox_3d, sample.sample_id, "gt")
    if pred is None or gt is None:
        return 0.0
    return compute_oriented_iou_3d(pred, gt)


def _coerce_bbox_9dof(value: Any, sample_id: str, label: str) -> list[float] | None:
    try:
        bbox = [float(v) for v in value[:9]]
    except (TypeError, ValueError, IndexError) as exc:
        logger.warning("Invalid {} bbox for {}: {}", label, sample_id, exc)
        return None
    if len(bbox) < 6:
        logger.warning("{} bbox for {} has fewer than 6 values: {}", label, sample_id, bbox)
        return None
    if not np.isfinite(np.array(bbox, dtype=np.float64)).all():
        logger.warning("{} bbox for {} contains non-finite values: {}", label, sample_id, bbox)
        return None
    while len(bbox) < 9:
        bbox.append(0.0)
    return bbox
