"""Extract ConceptGraph fused objects as detector proposal records."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


POST_PCD_GLOB = "full_pcd_ram_withbg_allclasses_overlap_maskconf*_post.pkl.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract class-agnostic ConceptGraph proposals to JSONL."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--include-bg", type=parse_bool, default=True)
    parser.add_argument("--min-points", type=int, default=50)
    return parser.parse_args()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected boolean value for --include-bg, got {value!r}"
    )


def compute_bbox_np_9dof(
    bbox_np: Any,
    *,
    scene_id: str,
    object_index: int,
) -> tuple[list[float], list[list[float]]]:
    corners = np.asarray(bbox_np, dtype=np.float64)
    if corners.shape != (8, 3):
        raise ValueError(
            f"{scene_id} object {object_index} bbox_np must have shape (8, 3)"
        )
    if not np.isfinite(corners).all():
        raise ValueError(f"{scene_id} object {object_index} bbox_np is not finite")

    corner_min = corners.min(axis=0)
    corner_max = corners.max(axis=0)
    center = (corner_min + corner_max) / 2.0
    extent = corner_max - corner_min
    if np.any(extent <= 0.0):
        raise ValueError(f"{scene_id} object {object_index} bbox_np is degenerate")

    bbox = np.concatenate([center, extent, [0.0, 0.0, 0.0]]).astype(np.float64)
    if bbox.shape != (9,) or not np.isfinite(bbox).all():
        raise ValueError(f"{scene_id} object {object_index} bbox_3d is not finite")
    return bbox.tolist(), corners.tolist()


def extract_scene_record(
    *,
    data_root: Path,
    scene_id: str,
    include_bg: bool = True,
    min_points: int = 50,
) -> dict[str, Any]:
    if min_points <= 0:
        raise ValueError("min_points must be positive")
    post_pcd_path = find_post_pcd_path(data_root=data_root, scene_id=scene_id)
    payload = load_post_pcd(post_pcd_path)

    objects = require_list(payload.get("objects"), scene_id=scene_id, key="objects")
    bg_objects = payload.get("bg_objects")
    if bg_objects is None:
        bg_list: list[Any] = []
    else:
        bg_list = require_list(bg_objects, scene_id=scene_id, key="bg_objects")

    candidates: list[tuple[Any, bool]] = [(obj, False) for obj in objects]
    if include_bg:
        candidates.extend((obj, True) for obj in bg_list)

    if not candidates:
        raise ValueError(f"{scene_id} has zero ConceptGraph objects")

    proposals: list[dict[str, Any]] = []
    filtered_min_points = 0
    for object_index, (obj, is_background) in enumerate(candidates):
        point_count = object_point_count(obj, scene_id=scene_id, object_index=object_index)
        if point_count < min_points:
            filtered_min_points += 1
            continue
        proposals.append(
            object_to_proposal(
                obj,
                scene_id=scene_id,
                object_index=object_index,
                is_background=is_background,
            )
        )

    if not proposals:
        raise ValueError(
            f"{scene_id} has zero ConceptGraph proposals after min_points={min_points}"
        )

    return {
        "scene_id": scene_id,
        "scan_id": f"scannet/{scene_id}",
        "target_id": None,
        "method": "3d-conceptgraph",
        "input_condition": "post_pcd",
        "proposals": proposals,
        "failure_tag": None,
        "metadata": {
            "post_pcd_path": str(post_pcd_path),
            "n_objects_total": len(candidates),
            "n_objects_filtered_min_points": filtered_min_points,
            "n_proposals": len(proposals),
            "include_bg": bool(include_bg),
            "min_points": int(min_points),
        },
    }


def find_post_pcd_path(*, data_root: Path, scene_id: str) -> Path:
    pcd_dir = data_root / "scannet" / scene_id / "conceptgraph" / "pcd_saves"
    if not pcd_dir.exists():
        raise FileNotFoundError(f"{scene_id} ConceptGraph pcd_saves not found: {pcd_dir}")
    matches = sorted(pcd_dir.glob(POST_PCD_GLOB), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(
            f"{scene_id} ConceptGraph post pickle not found under {pcd_dir}"
        )
    return matches[-1]


def load_post_pcd(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"ConceptGraph post pickle must contain a dict: {path}")
    return payload


def require_list(value: Any, *, scene_id: str, key: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{scene_id} ConceptGraph {key} must be a list")
    return value


def object_to_proposal(
    obj: Any,
    *,
    scene_id: str,
    object_index: int,
    is_background: bool,
) -> dict[str, Any]:
    bbox_np = require_object_value(
        obj,
        "bbox_np",
        scene_id=scene_id,
        object_index=object_index,
    )
    bbox_3d, raw_corners = compute_bbox_np_9dof(
        bbox_np,
        scene_id=scene_id,
        object_index=object_index,
    )
    score = object_score(obj, scene_id=scene_id, object_index=object_index)
    point_count = object_point_count(obj, scene_id=scene_id, object_index=object_index)
    num_detections = object_num_detections(
        obj,
        scene_id=scene_id,
        object_index=object_index,
    )
    return {
        "bbox_3d": bbox_3d,
        "score": score,
        "label": "object",
        "source": "conceptgraph",
        "metadata": {
            "class_id": -1,
            "detector": "ConceptGraph",
            "raw_corners": raw_corners,
            "n_points": point_count,
            "num_detections": num_detections,
            "is_background": int(is_background),
            "box_format": "aabb_from_conceptgraph_bbox_np",
        },
    }


def object_point_count(obj: Any, *, scene_id: str, object_index: int) -> int:
    points = require_object_value(
        obj,
        "pcd_np",
        scene_id=scene_id,
        object_index=object_index,
    )
    arr = np.asarray(points)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"{scene_id} object {object_index} pcd_np must have shape (N, 3)"
        )
    return int(arr.shape[0])


def object_num_detections(obj: Any, *, scene_id: str, object_index: int) -> int:
    value = require_object_value(
        obj,
        "num_detections",
        scene_id=scene_id,
        object_index=object_index,
    )
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{scene_id} object {object_index} num_detections is invalid"
        ) from exc
    if out < 0:
        raise ValueError(f"{scene_id} object {object_index} num_detections is negative")
    return out


def object_score(obj: Any, *, scene_id: str, object_index: int) -> float:
    conf = optional_object_value(obj, "conf")
    if conf is not None:
        conf_values = finite_float_list(
            conf,
            scene_id=scene_id,
            object_index=object_index,
            field_name="conf",
        )
        if conf_values:
            return max(conf_values)

    num_detections = object_num_detections(
        obj,
        scene_id=scene_id,
        object_index=object_index,
    )
    score = min(1.0, float(num_detections) / 30.0)
    if not np.isfinite(score):
        raise ValueError(f"{scene_id} object {object_index} score is not finite")
    return score


def finite_float_list(
    value: Any,
    *,
    scene_id: str,
    object_index: int,
    field_name: str,
) -> list[float]:
    if isinstance(value, np.ndarray):
        seq: Iterable[Any] = value.reshape(-1).tolist()
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        seq = value
    else:
        raise ValueError(f"{scene_id} object {object_index} {field_name} is invalid")
    out = [float(item) for item in seq]
    if not all(np.isfinite(item) for item in out):
        raise ValueError(f"{scene_id} object {object_index} {field_name} is not finite")
    return out


def require_object_value(
    obj: Any,
    key: str,
    *,
    scene_id: str,
    object_index: int,
) -> Any:
    value = optional_object_value(obj, key)
    if value is None:
        raise ValueError(f"{scene_id} object {object_index} missing {key}")
    return value


def optional_object_value(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def load_scene_ids(scene_list_path: Path) -> list[str]:
    raw = json.loads(scene_list_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"scene list must be a JSON list: {scene_list_path}")
    scene_ids: list[str] = []
    for index, row in enumerate(raw):
        if isinstance(row, str):
            scene_id = row.strip()
        elif isinstance(row, Mapping):
            scene_id = str(row.get("scene_id") or "").strip()
        else:
            raise ValueError(f"scene list row {index} must be a string or object")
        if not scene_id:
            raise ValueError(f"scene list row {index} has empty scene_id")
        scene_ids.append(scene_id)
    if not scene_ids:
        raise ValueError(f"scene list is empty: {scene_list_path}")
    return scene_ids


def write_detector_records(
    *,
    data_root: Path,
    scene_list_path: Path,
    output_jsonl: Path,
    include_bg: bool = True,
    min_points: int = 50,
) -> list[dict[str, Any]]:
    scene_ids = load_scene_ids(scene_list_path)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for scene_id in scene_ids:
            record = extract_scene_record(
                data_root=data_root,
                scene_id=scene_id,
                include_bg=include_bg,
                min_points=min_points,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
    return records


def main() -> None:
    args = parse_args()
    write_detector_records(
        data_root=args.data_root,
        scene_list_path=args.scene_list,
        output_jsonl=args.output_jsonl,
        include_bg=args.include_bg,
        min_points=args.min_points,
    )


if __name__ == "__main__":
    main()
