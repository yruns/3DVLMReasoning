from __future__ import annotations

import gzip
import pickle
from collections import Counter
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import aabb_from_points, is_non_degenerate_bbox
from .models import BBox3DProposal, FailureTag, ProposalRecord

_BG = {"wall", "floor", "ceiling"}
_UNKNOWN_NAMES = {"item", "object", "none"}


def generate_conceptgraph_proposals(
    *,
    scene_path: str | Path,
    scan_id: str,
    scene_id: str,
) -> ProposalRecord:
    scene = Path(scene_path)
    pkl_path = _find_pcd_file(scene)
    if pkl_path is None:
        return ProposalRecord(
            scene_id=scene_id,
            scan_id=scan_id,
            method="2d-cg",
            input_condition="conceptgraph_scene",
            proposals=[],
            failure_tag=FailureTag.NO_PROPOSAL,
            metadata={"reason": "no_pcd_file", "scene_path": str(scene)},
        )

    payload = _load_pcd_payload(pkl_path)
    objects = _payload_objects(payload, pkl_path)

    proposals: list[BBox3DProposal] = []
    for obj_idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue

        category = _category(obj)
        if category.strip().lower() in _BG:
            continue

        pcd_np = obj.get("pcd_np")
        if pcd_np is None:
            continue

        try:
            points = np.asarray(pcd_np, dtype=np.float32)
            if len(points) == 0:
                continue
            bbox = aabb_from_points(points)
        except (TypeError, ValueError):
            continue

        if not is_non_degenerate_bbox(bbox):
            continue

        proposals.append(
            BBox3DProposal(
                bbox_3d=bbox,
                score=_score(obj),
                source="conceptgraph",
                metadata={
                    "obj_idx": obj_idx,
                    "category": category,
                    "num_points": int(len(points)),
                    "pkl_path": str(pkl_path),
                },
            )
        )

    return ProposalRecord(
        scene_id=scene_id,
        scan_id=scan_id,
        target_id=None,
        method="2d-cg",
        input_condition="conceptgraph_scene",
        proposals=proposals,
        failure_tag=None if proposals else FailureTag.NO_PROPOSAL,
    )


def _find_pcd_file(scene_path: Path) -> Path | None:
    pcd_dir = scene_path / "pcd_saves"
    for pattern in ("*ram*_post.pkl.gz", "*_post.pkl.gz", "*.pkl.gz"):
        matches = sorted(pcd_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _load_pcd_payload(pkl_path: Path) -> Any:
    try:
        with gzip.open(pkl_path, "rb") as f:
            return pickle.load(f)
    except (EOFError, OSError, pickle.PickleError, ValueError) as exc:
        raise ValueError(f"Failed to load ConceptGraph PCD file {pkl_path}") from exc


def _payload_objects(payload: Any, pkl_path: Path) -> list[Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid ConceptGraph PCD payload in {pkl_path}: expected dict")
    if "objects" not in payload:
        raise ValueError(
            f"Invalid ConceptGraph PCD payload in {pkl_path}: missing objects list"
        )
    objects = payload["objects"]
    if not isinstance(objects, list):
        raise ValueError(
            f"Invalid ConceptGraph PCD payload in {pkl_path}: expected objects list"
        )
    return objects


def _category(obj: dict[str, Any]) -> str:
    names = [str(n).strip() for n in _as_list(obj.get("class_name")) if n]
    valid = [n for n in names if n and n.lower() not in _UNKNOWN_NAMES]
    if not valid:
        return "unknown"
    return Counter(valid).most_common(1)[0][0]


def _score(obj: dict[str, Any]) -> float | None:
    conf = _as_list(obj.get("conf"))
    if not conf:
        return None
    try:
        arr = np.asarray(conf, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return None
    score = float(finite.mean())
    return score if isfinite(score) else None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    try:
        return list(value)
    except TypeError:
        return [value]
