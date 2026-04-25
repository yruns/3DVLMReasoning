from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


VDETR_REPO_URL = "https://github.com/V-DETR/V-DETR.git"
VDETR_SCANNET_CKPT_URL = (
    "https://huggingface.co/byshen/vdetr/resolve/main/scannet_540ep.pth"
)

SCANNET_CLASS_NAMES = [
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "showercurtrain",
    "toilet",
    "sink",
    "bathtub",
    "garbagebin",
]


def build_vdetr_command_template(
    *,
    repo_dir: str | Path = "external/V-DETR",
    checkpoint_path: str | Path = "external/V-DETR/checkpoints/scannet_540ep.pth",
    export_script: str | Path = "scripts/vdetr_export_predictions.py",
    python_executable: str | Path = "python",
    num_points: int = 40000,
    conf_thresh: float = 0.05,
    top_k: int = 256,
) -> str:
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    return (
        f"{python_executable} {export_script} "
        f"--repo-dir {repo_dir} "
        f"--checkpoint {checkpoint_path} "
        "--pointcloud {pointcloud_path} "
        "--output {output_path} "
        f"--num-points {int(num_points)} "
        f"--conf-thresh {float(conf_thresh)} "
        f"--top-k {int(top_k)}"
    )


def corners_to_aabb_9dof(corners: Any) -> list[float]:
    arr = np.asarray(corners, dtype=np.float64)
    if arr.shape != (8, 3):
        raise ValueError("corners must have shape (8, 3)")
    if not np.isfinite(arr).all():
        raise ValueError("corners must contain only finite values")
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    center = (lo + hi) / 2.0
    extent = hi - lo
    return [
        float(center[0]),
        float(center[1]),
        float(center[2]),
        float(extent[0]),
        float(extent[1]),
        float(extent[2]),
        0.0,
        0.0,
        0.0,
    ]


def camera_corners_to_depth_corners(corners: Any) -> np.ndarray:
    arr = np.asarray(corners, dtype=np.float64)
    if arr.shape != (8, 3):
        raise ValueError("corners must have shape (8, 3)")
    if not np.isfinite(arr).all():
        raise ValueError("corners must contain only finite values")
    depth = arr.copy()
    depth[..., [0, 1, 2]] = depth[..., [0, 2, 1]]
    depth[..., 2] *= -1
    return depth


def write_vdetr_proposal_json(
    *,
    output_path: str | Path,
    predictions: list[dict[str, Any]],
    top_k: int,
) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    sorted_predictions = sorted(
        predictions,
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )[:top_k]
    proposals = []
    for prediction in sorted_predictions:
        label = prediction.get("label")
        class_id = prediction.get("class_id")
        if label is None and class_id is not None:
            label = class_name_from_id(int(class_id))
        proposals.append(
            {
                "bbox_3d": corners_to_aabb_9dof(prediction["corners"]),
                "score": float(prediction.get("score", 0.0)),
                "label": label,
                "source": "vdetr",
                "metadata": {
                    "class_id": class_id,
                    "box_format": "aabb_from_vdetr_corners",
                },
            }
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"proposals": proposals}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def class_name_from_id(class_id: int) -> str:
    if class_id < 0 or class_id >= len(SCANNET_CLASS_NAMES):
        return f"class_{class_id}"
    return SCANNET_CLASS_NAMES[class_id]
