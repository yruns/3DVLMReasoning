#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _load_vdetr_helpers() -> tuple[Any, Any]:
    helper_path = SRC_ROOT / "benchmarks" / "embodiedscan_bbox_feasibility" / "vdetr.py"
    spec = importlib.util.spec_from_file_location("_embodiedscan_vdetr_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load V-DETR helper module: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        module.camera_corners_to_depth_corners,
        module.class_name_from_id,
        module.write_vdetr_proposal_json,
    )


(
    camera_corners_to_depth_corners,
    class_name_from_id,
    write_vdetr_proposal_json,
) = _load_vdetr_helpers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export V-DETR proposals for one prepared detector-input PLY."
    )
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pointcloud", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-points", type=int, default=40000)
    parser.add_argument("--conf-thresh", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument(
        "--no-per-class-proposal",
        action="store_true",
        help="Keep only the max class per V-DETR query instead of class-expanded boxes.",
    )
    return parser.parse_args()


def load_ascii_xyz_ply(path: str | Path) -> np.ndarray:
    ply_path = Path(path)
    with ply_path.open("r", encoding="utf-8") as handle:
        vertex_count: int | None = None
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
            if stripped == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY is missing an element vertex header: {ply_path}")
        points: list[list[float]] = []
        for _ in range(vertex_count):
            parts = handle.readline().strip().split()
            if len(parts) < 3:
                raise ValueError(f"PLY vertex row must contain xyz: {ply_path}")
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"PLY must contain xyz points: {ply_path}")
    if not np.isfinite(arr).all():
        raise ValueError(f"PLY contains non-finite xyz values: {ply_path}")
    return arr


def sample_points(points: np.ndarray, num_points: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must have shape (N, >=3)")
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if len(pts) <= num_points:
        return pts[:, :3]
    indices = np.linspace(0, len(pts) - 1, num=num_points, dtype=np.int64)
    return pts[indices, :3]


def prepare_model_points(points: np.ndarray, vdetr_args: argparse.Namespace) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    channels = [pts]
    if getattr(vdetr_args, "use_color", False):
        channels.append(np.zeros((len(pts), 3), dtype=np.float32))
    if getattr(vdetr_args, "use_normals", False):
        channels.append(np.zeros((len(pts), 3), dtype=np.float32))
    return np.concatenate(channels, axis=1)


def ensure_vdetr_arg_defaults(vdetr_args: argparse.Namespace) -> None:
    if not hasattr(vdetr_args, "random_fps"):
        vdetr_args.random_fps = False


def run_vdetr(args: argparse.Namespace) -> None:
    if not args.repo_dir.exists():
        raise FileNotFoundError(f"V-DETR repo not found: {args.repo_dir}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"V-DETR checkpoint not found: {args.checkpoint}")
    if not args.pointcloud.exists():
        raise FileNotFoundError(f"Point cloud not found: {args.pointcloud}")

    if str(args.repo_dir) not in sys.path:
        sys.path.insert(0, str(args.repo_dir))

    try:
        import torch
        from datasets.scannet import ScannetDatasetConfig
        from main import auto_reload, make_args_parser
        from models import build_model
        from utils.ap_calculator import get_ap_config_dict, parse_predictions
    except ImportError as exc:
        raise ImportError(
            "V-DETR dependencies are not importable. Install the external V-DETR "
            "environment with MinkowskiEngine, mmcv-full, and pointnet2 before "
            "running this script."
        ) from exc

    points = sample_points(load_ascii_xyz_ply(args.pointcloud), args.num_points)
    if len(points) == 0:
        raise ValueError(f"Point cloud contains no points: {args.pointcloud}")

    vdetr_args = make_args_parser().parse_args(
        [
            "--dataset_name",
            "scannet",
            "--test_only",
            "--test_ckpt",
            str(args.checkpoint),
            "--num_points",
            str(args.num_points),
            "--conf_thresh",
            str(args.conf_thresh),
        ]
    )
    vdetr_args.wandb_activate = False
    vdetr_args.no_per_class_proposal = bool(args.no_per_class_proposal)
    auto_reload(vdetr_args)
    ensure_vdetr_arg_defaults(vdetr_args)

    dataset_config = ScannetDatasetConfig()
    model = build_model(vdetr_args, dataset_config)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval().cuda()

    model_points = prepare_model_points(points, vdetr_args)
    point_cloud = torch.from_numpy(model_points[None, :, :]).float().cuda()
    dims_min = point_cloud[:, :, :3].min(dim=1).values
    dims_max = point_cloud[:, :, :3].max(dim=1).values
    with torch.no_grad():
        outputs = model(
            {
                "point_clouds": point_cloud,
                "point_cloud_dims_min": dims_min,
                "point_cloud_dims_max": dims_max,
            }
        )["outputs"]
        if vdetr_args.cls_loss.split("_")[0] == "focalloss":
            outputs["sem_cls_prob"] = outputs["sem_cls_prob"].sigmoid()
        if vdetr_args.axis_align_test:
            outputs["box_corners"] = outputs["box_corners_axis_align"]
        predicted_box_csa = torch.cat(
            (
                outputs["center_unnormalized"],
                outputs["size_unnormalized"],
                outputs["angle_continuous"].unsqueeze(-1),
            ),
            dim=-1,
        )
        pred_maps = parse_predictions(
            outputs["box_corners"],
            outputs["sem_cls_prob"],
            outputs["objectness_prob"],
            outputs["angle_prob"],
            point_cloud,
            get_ap_config_dict(
                remove_empty_box=False,
                use_3d_nms=not vdetr_args.no_3d_nms,
                nms_iou=vdetr_args.nms_iou,
                use_old_type_nms=vdetr_args.use_old_type_nms,
                cls_nms=not vdetr_args.no_cls_nms,
                per_class_proposal=not vdetr_args.no_per_class_proposal,
                use_cls_confidence_only=vdetr_args.use_cls_confidence_only,
                conf_thresh=vdetr_args.conf_thresh,
                no_nms=vdetr_args.test_no_nms,
                dataset_config=dataset_config,
                empty_pt_thre=vdetr_args.empty_pt_thre,
                rotated_nms=vdetr_args.rotated_nms,
                angle_nms=vdetr_args.angle_nms,
                angle_conf=vdetr_args.angle_conf,
            ),
            predicted_box_csa,
        )

    predictions = [_prediction_to_dict(item) for item in pred_maps[0]]
    write_vdetr_proposal_json(
        output_path=args.output,
        predictions=predictions,
        top_k=args.top_k,
    )


def _prediction_to_dict(item: tuple[Any, Any, Any]) -> dict[str, Any]:
    class_id, corners, score = item
    class_int = int(class_id)
    return {
        "class_id": class_int,
        "label": class_name_from_id(class_int),
        "score": float(score),
        "corners": camera_corners_to_depth_corners(corners).astype(np.float32),
    }


def main() -> None:
    run_vdetr(parse_args())


if __name__ == "__main__":
    main()
