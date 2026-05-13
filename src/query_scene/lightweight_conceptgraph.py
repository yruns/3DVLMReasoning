"""Lightweight ConceptGraph object caches for benchmark preparation.

The original ConceptGraph ``full_pcd_*_post.pkl.gz`` files contain per-detection
segmentation masks. They compress well on disk but can expand to many GB during
``pickle.load``. Query-driven keyframe selection and VG pack preparation only
need object metadata, centroids, boxes, and optional CLIP features, so this
module writes a stripped sidecar cache that is safe to load at high concurrency.
"""

from __future__ import annotations

import gzip
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

LIGHTWEIGHT_FORMAT_VERSION = "conceptgraph_lightweight_objects_v1"

DROP_FIELDS = frozenset({"mask", "pcd_np", "pcd_color_np"})
KEEP_FIELDS = frozenset(
    {
        "bbox_np",
        "class_name",
        "class_id",
        "conf",
        "xyxy",
        "n_points",
        "pixel_area",
        "num_detections",
        "contain_number",
        "is_background",
        "inst_color",
        "clip_ft",
        "text_ft",
        "image_idx",
        "mask_idx",
        "color_path",
    }
)


def lightweight_pcd_path(pcd_path: Path) -> Path:
    """Return the stripped sidecar path for a ConceptGraph object pickle."""

    path = Path(pcd_path)
    name = path.name
    if name.endswith(".light.pkl.gz"):
        return path
    if name.endswith(".pkl.gz"):
        return path.with_name(f"{name[:-7]}.light.pkl.gz")
    return path.with_suffix(f"{path.suffix}.light.pkl.gz")


def load_conceptgraph_payload(
    pcd_path: Path,
    *,
    prefer_lightweight: bool = True,
    ensure_lightweight: bool = False,
) -> dict[str, Any]:
    """Load a ConceptGraph payload, preferring the stripped sidecar cache."""

    path = Path(pcd_path)
    light_path = lightweight_pcd_path(path)
    if prefer_lightweight and ensure_lightweight and not light_path.exists():
        ensure_lightweight_conceptgraph_cache(path)
    if prefer_lightweight and light_path.exists():
        return _load_pickle_gz(light_path)
    return _load_pickle_gz(path)


def load_conceptgraph_objects(
    pcd_path: Path,
    *,
    prefer_lightweight: bool = True,
    ensure_lightweight: bool = False,
) -> list[dict[str, Any]]:
    payload = load_conceptgraph_payload(
        pcd_path,
        prefer_lightweight=prefer_lightweight,
        ensure_lightweight=ensure_lightweight,
    )
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"{pcd_path} must contain an objects list")
    return objects


def write_lightweight_conceptgraph_cache(
    pcd_path: Path,
    *,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a stripped sidecar cache and return its path."""

    return _with_cache_build_lock(
        pcd_path,
        lambda: _write_lightweight_conceptgraph_cache_unlocked(
            pcd_path,
            output_path=output_path,
            overwrite=overwrite,
        ),
    )


def ensure_lightweight_conceptgraph_cache(pcd_path: Path) -> Path:
    """Create the stripped sidecar cache if missing, serialized across workers."""

    return write_lightweight_conceptgraph_cache(pcd_path, overwrite=False)


def _write_lightweight_conceptgraph_cache_unlocked(
    pcd_path: Path,
    *,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    source_path = Path(pcd_path)
    target_path = output_path or lightweight_pcd_path(source_path)
    if target_path.exists() and not overwrite:
        return target_path

    payload = _load_pickle_gz(source_path)
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"{source_path} must contain an objects list")

    light_payload = {
        "format_version": LIGHTWEIGHT_FORMAT_VERSION,
        "source": str(source_path),
        "dropped_fields": sorted(DROP_FIELDS),
        "objects": [strip_object(obj) for obj in objects],
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp")
    with gzip.open(tmp_path, "wb") as handle:
        pickle.dump(light_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(target_path)
    return target_path


def _with_cache_build_lock(pcd_path: Path, func: Callable[[], Path]) -> Path:
    lock_path = _cache_build_lock_path(Path(pcd_path))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a", encoding="utf-8") as lock_handle:
        try:
            import fcntl

            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            try:
                return func()
            finally:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)
        except ImportError:
            return func()


def _cache_build_lock_path(pcd_path: Path) -> Path:
    path = Path(pcd_path)
    if len(path.parents) >= 4:
        return path.parents[3] / ".conceptgraph_lightweight_cache.lock"
    return path.parent / ".conceptgraph_lightweight_cache.lock"


def strip_object(obj: dict[str, Any]) -> dict[str, Any]:
    """Return the fields needed by selector/prep without masks or raw point clouds."""

    if not isinstance(obj, dict):
        raise TypeError(f"ConceptGraph object must be a dict, got {type(obj).__name__}")

    out = {key: value for key, value in obj.items() if key in KEEP_FIELDS}
    centroid = _object_centroid(obj)
    if centroid is not None:
        out["centroid"] = centroid.astype(np.float32, copy=False)
    return out


def _object_centroid(obj: dict[str, Any]) -> np.ndarray | None:
    raw_centroid = obj.get("centroid")
    if raw_centroid is not None:
        arr = np.asarray(raw_centroid, dtype=np.float32).reshape(-1)
        if arr.size >= 3:
            return arr[:3]

    pcd_np = obj.get("pcd_np")
    if pcd_np is not None:
        arr = np.asarray(pcd_np, dtype=np.float32)
        if arr.size and arr.ndim >= 2 and arr.shape[-1] >= 3:
            return arr.reshape(-1, arr.shape[-1])[:, :3].mean(axis=0)

    bbox_np = obj.get("bbox_np")
    if bbox_np is not None:
        arr = np.asarray(bbox_np, dtype=np.float32)
        if arr.size and arr.ndim >= 2 and arr.shape[-1] >= 3:
            return arr.reshape(-1, arr.shape[-1])[:, :3].mean(axis=0)

    return None


def _load_pickle_gz(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing ConceptGraph pkl: {path}")
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a dict payload")
    return payload
