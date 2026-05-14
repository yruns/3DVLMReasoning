"""Run ScanRefer VG pack-v1 backend and report metrics."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import math
import pickle
import statistics
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agents.core.agent_config import Stage2DeepAgentConfig, Stage2TaskType
from agents.core.task_types import Stage2TaskSpec
from benchmarks.embodiedscan_eval import compute_oriented_iou_3d

BackendName = Literal["pack_v1"]
Stage2DeepResearchAgent: Any | None = None
build_pack_v1_bundle: Any | None = None

# Per-scene KeyframeSelector cache for the Stage 2 → Stage 1 callback loop.
# Each scene's selector loads pcd + enriched_objects + visibility (~1-3s) and
# is reused across all workers/samples in the same scene.
_MAX_SELECTOR_CACHE_SIZE = 4
_SELECTOR_CACHE: OrderedDict[str, Any] = OrderedDict()
_SELECTOR_CACHE_LOCK = threading.Lock()
_SELECTOR_BUILD_LOCKS: dict[str, threading.Lock] = {}
_MAX_CONCEPTGRAPH_OBJECT_CACHE_SIZE = 4
_CONCEPTGRAPH_OBJECT_CACHE: OrderedDict[
    tuple[str, str], list[dict[str, Any]]
] = OrderedDict()
_CONCEPTGRAPH_OBJECT_CACHE_LOCK = threading.Lock()


def _remember_keyframe_selector(scene_id: str, selector: Any | None) -> None:
    _SELECTOR_CACHE[scene_id] = selector
    _SELECTOR_CACHE.move_to_end(scene_id)
    while len(_SELECTOR_CACHE) > _MAX_SELECTOR_CACHE_SIZE:
        _SELECTOR_CACHE.popitem(last=False)


def _remember_conceptgraph_objects(
    cache_key: tuple[str, str],
    objects: list[dict[str, Any]],
) -> None:
    _CONCEPTGRAPH_OBJECT_CACHE[cache_key] = objects
    _CONCEPTGRAPH_OBJECT_CACHE.move_to_end(cache_key)
    while len(_CONCEPTGRAPH_OBJECT_CACHE) > _MAX_CONCEPTGRAPH_OBJECT_CACHE_SIZE:
        _CONCEPTGRAPH_OBJECT_CACHE.popitem(last=False)


def _get_or_build_keyframe_selector(
    scene_id: str,
    phase8_data_root: Path,
    llm_model: str = "gemini-2.5-pro",
) -> Any | None:
    """Return a cached KeyframeSelector for `scene_id`, or build one lazily.

    Returns None when the scene has no enriched_objects.json (callback should
    be disabled for that scene; gt_target back-compat). Thread-safe via a
    per-scene build lock so concurrent workers don't double-instantiate.
    """
    with _SELECTOR_CACHE_LOCK:
        if scene_id in _SELECTOR_CACHE:
            _SELECTOR_CACHE.move_to_end(scene_id)
            return _SELECTOR_CACHE[scene_id]
        build_lock = _SELECTOR_BUILD_LOCKS.setdefault(scene_id, threading.Lock())

    with build_lock:
        # Re-check inside the per-scene lock in case another worker built it
        with _SELECTOR_CACHE_LOCK:
            if scene_id in _SELECTOR_CACHE:
                _SELECTOR_CACHE.move_to_end(scene_id)
                return _SELECTOR_CACHE[scene_id]

        cg_root = phase8_data_root / scene_id / "conceptgraph"
        enriched = cg_root / "enriched_objects.json"
        if not enriched.exists():
            with _SELECTOR_CACHE_LOCK:
                _remember_keyframe_selector(scene_id, None)
            return None

        from query_scene.keyframe_selector import KeyframeSelector

        # ScanRefer Phase-8 visibility indices are saved at stride=1. Using a
        # different stride makes callback view IDs fail path resolution.
        selector = KeyframeSelector.from_scene_path(
            str(cg_root),
            stride=1,
            llm_model=llm_model,
        )
        with _SELECTOR_CACHE_LOCK:
            _remember_keyframe_selector(scene_id, selector)
        return selector


@dataclass(frozen=True)
class ParsedScanRefSampleId:
    scan_id: str
    scene_id: str
    target_id: int
    ann_id: str


def run_one_sample(
    sample_id: str,
    backend: BackendName,
    *,
    data_root: Path,
    pack_name: str = "pack_scanrefer_v1",
    config: Stage2DeepAgentConfig | None = None,
    sample_retries: int = 0,
) -> dict[str, Any]:
    if sample_retries < 0:
        raise ValueError("sample_retries must be non-negative")
    last_error: Exception | None = None
    for attempt in range(sample_retries + 1):
        try:
            return _run_one_sample_once(
                sample_id,
                backend,
                data_root=data_root,
                pack_name=pack_name,
                config=config,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= sample_retries or not is_retryable_sample_error(exc):
                raise
            time.sleep(2.0 * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError("run_one_sample retry loop exited unexpectedly")


def _run_one_sample_once(
    sample_id: str,
    backend: BackendName,
    *,
    data_root: Path,
    pack_name: str = "pack_scanrefer_v1",
    config: Stage2DeepAgentConfig | None = None,
) -> dict[str, Any]:
    if backend != "pack_v1":
        raise ValueError(f"backend={backend!r} is not supported; use 'pack_v1'")
    sample = load_sample_artifact(data_root, sample_id, pack_name=pack_name)
    gt_bbox = coerce_bbox_9dof(
        sample.get("gt_bbox_3d_9dof"),
        field_name=f"{sample_id}.gt_bbox_3d_9dof",
    )
    cfg = config_for_backend(backend, config)
    raw_result = run_pack_v1_sample(sample, data_root, cfg, pack_name=pack_name)
    prediction = extract_pack_v1_prediction(raw_result)
    tool_trace = extract_result_tool_trace(raw_result)

    status = prediction.get("status")
    if status is None:
        raise ValueError(
            f"{sample_id}: {backend} prediction missing status; payload={prediction!r}"
        )
    selected_id = prediction.get("selected_object_id")
    pred_bbox_raw = prediction.get("bbox_3d")
    if _is_failed_marker(selected_id, status):
        return {
            "sample_id": sample_id,
            "backend": backend,
            "status": "failed",
            "iou": 0.0,
            "predicted_bbox_3d_9dof": None,
            "gt_bbox_3d_9dof": gt_bbox,
            "selected_object_id": selected_id,
            "confidence": prediction.get("confidence"),
            "query": sample.get("query"),
            "tool_trace": tool_trace,
        }
    if pred_bbox_raw is None:
        raise ValueError(
            f"{sample_id}: {backend} payload missing bbox_3d but is not the "
            f"failed-sample marker; payload={prediction!r}"
        )
    pred_bbox = coerce_bbox_9dof(
        pred_bbox_raw,
        field_name=f"{sample_id}.{backend}.bbox_3d",
    )
    iou = compute_oriented_iou_3d(pred_bbox, gt_bbox)
    return {
        "sample_id": sample_id,
        "backend": backend,
        "status": status,
        "iou": iou,
        "predicted_bbox_3d_9dof": pred_bbox,
        "gt_bbox_3d_9dof": gt_bbox,
        "selected_object_id": selected_id,
        "confidence": prediction.get("confidence"),
        "query": sample.get("query"),
        "tool_trace": tool_trace,
    }


def compare_backends(
    *,
    sample_ids: Sequence[str],
    output_dir: Path,
    data_root: Path,
    pack_name: str = "pack_scanrefer_v1",
    config: Stage2DeepAgentConfig | None = None,
    sample_retries: int = 0,
    workers: int = 1,
    return_results: bool = True,
    write_side_by_side: bool = True,
    max_new_samples: int | None = None,
) -> dict[str, Any] | None:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if max_new_samples is not None and max_new_samples < 0:
        raise ValueError("max_new_samples must be non-negative")
    validate_unique_sample_ids(sample_ids)
    if sample_ids:
        preflight_pack_sample_exists(data_root, sample_ids[0], pack_name=pack_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for backend in ("pack_v1",):

        def run_sample(
            sample_id: str, backend: BackendName = backend
        ) -> dict[str, Any]:
            try:
                result = run_one_sample(
                    sample_id,
                    backend,
                    data_root=data_root,
                    pack_name=pack_name,
                    config=config,
                    sample_retries=sample_retries,
                )
            except Exception as exc:
                # Per-sample uncaught failure: persist a failed sentinel so the
                # whole run does not collapse on one rotten sample (e.g. an
                # upstream image-processing 500 that survives all retries).
                # The sample still counts toward Acc/n with iou=0.
                result = {
                    "sample_id": sample_id,
                    "backend": backend,
                    "status": "failed",
                    "iou": 0.0,
                    "predicted_bbox_3d_9dof": None,
                    "gt_bbox_3d_9dof": None,
                    "selected_object_id": None,
                    "confidence": None,
                    "query": None,
                    "error": f"{type(exc).__name__}: {str(exc)[:480]}",
                }
            write_sample_result_checkpoint(
                output_dir,
                backend,
                result,
                pack_name=pack_name,
            )
            return result

        missing_sample_ids = [
            sample_id
            for sample_id in sample_ids
            if not sample_result_path(
                output_dir,
                backend,
                sample_id,
                pack_name=pack_name,
            ).exists()
        ]
        if max_new_samples is not None:
            missing_sample_ids = missing_sample_ids[:max_new_samples]
        if workers == 1:
            for sample_id in missing_sample_ids:
                run_sample(sample_id)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(run_sample, sample_id)
                    for sample_id in missing_sample_ids
                ]
                for future in concurrent.futures.as_completed(futures):
                    future.result()

        if not write_side_by_side:
            continue
        if return_results:
            results[backend] = build_backend_payload_from_checkpoints(
                sample_ids=sample_ids,
                output_dir=output_dir,
                backend=backend,
                pack_name=pack_name,
                include_per_sample=True,
            )
        else:
            stream_side_by_side_from_checkpoints(
                sample_ids=sample_ids,
                output_dir=output_dir,
                backend=backend,
                pack_name=pack_name,
            )
    if return_results and write_side_by_side:
        (output_dir / "side_by_side.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return results
    return None


def run_pack_v1_sample(
    sample: dict[str, Any],
    data_root: Path,
    config: Stage2DeepAgentConfig,
    *,
    pack_name: str = "pack_scanrefer_v1",
    phase8_data_root: Path = Path("data/nr3d/scannet"),
    enable_stage1_callback: bool = True,
) -> Any:
    bundle = build_pack_v1_bundle_from_sample(
        sample,
        data_root,
        pack_name=pack_name,
        phase8_data_root=phase8_data_root,
    )
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        user_query=str(sample["query"]),
    )
    agent_cls = Stage2DeepResearchAgent
    if agent_cls is None:
        from agents.stage2_deep_agent import Stage2DeepResearchAgent as agent_cls

    # v9 catalog-first: Stage-1 more-views / hypothesis flows are exposed
    # directly as selector tools. Only the crop callback (object-centric
    # red-bbox crops) is retained for the `request_crops` tool.
    crop_callback = None
    if enable_stage1_callback:
        scene_id = str(sample["scene_id"])
        selector = _get_or_build_keyframe_selector(scene_id, phase8_data_root)
        if selector is not None:
            from agents.stage1_callbacks import create_crop_callback

            crop_callback = create_crop_callback(
                selector,
                scene_id=scene_id,
                crop_scale=2.0,
            )

    agent = agent_cls(
        config=config,
        crop_callback=crop_callback,
    )
    return agent.run(task=task, bundle=bundle)


def _bbox_minmax_from_9dof(bbox: Sequence[Any]) -> tuple[list[float], list[float]]:
    cx, cy, cz, dx, dy, dz = [float(x) for x in bbox[:6]]
    mins = [cx - dx / 2.0, cy - dy / 2.0, cz - dz / 2.0]
    maxs = [cx + dx / 2.0, cy + dy / 2.0, cz + dz / 2.0]
    return mins, maxs


def _bbox_minmax_from_points(points: Any) -> tuple[list[float], list[float]]:
    rows = list(points)
    if not rows:
        raise ValueError("bbox_np must contain at least one point")
    mins = [min(float(row[i]) for row in rows) for i in range(3)]
    maxs = [max(float(row[i]) for row in rows) for i in range(3)]
    return mins, maxs


def _axis_aligned_iou_3d(
    a_min: Sequence[float],
    a_max: Sequence[float],
    b_min: Sequence[float],
    b_max: Sequence[float],
) -> float:
    inter = 1.0
    vol_a = 1.0
    vol_b = 1.0
    for idx in range(3):
        inter_len = max(0.0, min(a_max[idx], b_max[idx]) - max(a_min[idx], b_min[idx]))
        inter *= inter_len
        vol_a *= max(0.0, a_max[idx] - a_min[idx])
        vol_b *= max(0.0, b_max[idx] - b_min[idx])
    union = vol_a + vol_b - inter
    return inter / union if union > 0 else 0.0


def _load_conceptgraph_objects_for_frame_views(
    scene_id: str,
    phase8_data_root: Path,
) -> list[dict[str, Any]]:
    cache_key = (str(phase8_data_root), scene_id)
    with _CONCEPTGRAPH_OBJECT_CACHE_LOCK:
        cached = _CONCEPTGRAPH_OBJECT_CACHE.get(cache_key)
        if cached is not None:
            _CONCEPTGRAPH_OBJECT_CACHE.move_to_end(cache_key)
            return cached

    pcd_path = (
        phase8_data_root
        / scene_id
        / "conceptgraph"
        / "pcd_saves"
        / "full_pcd_gt_axisaligned_post.pkl.gz"
    )
    if not pcd_path.exists():
        objects: list[dict[str, Any]] = []
    else:
        with gzip.open(pcd_path, "rb") as fh:
            raw = pickle.load(fh)
        objects = []
        for obj in raw.get("objects", []):
            if not isinstance(obj, dict) or obj.get("bbox_np") is None:
                continue
            try:
                bbox_min, bbox_max = _bbox_minmax_from_points(obj["bbox_np"])
            except ValueError:
                continue
            frame_views: dict[int, dict[str, Any]] = {}
            image_ids = list(obj.get("image_idx", []) or [])
            xys = list(obj.get("xyxy", []) or [])
            confs = list(obj.get("conf", []) or [])
            for idx, frame_raw in enumerate(image_ids):
                if idx >= len(xys):
                    continue
                frame_id = int(frame_raw)
                xyxy = [int(round(float(v))) for v in list(xys[idx])[:4]]
                if len(xyxy) != 4:
                    continue
                frame_views[frame_id] = {
                    "frame_id": frame_id,
                    "bbox_2d": xyxy,
                    "raw_rgb_path": str(
                        phase8_data_root
                        / scene_id
                        / "raw"
                        / f"{frame_id * 10:06d}-rgb.png"
                    ),
                    "visibility_weight": (
                        float(confs[idx]) if idx < len(confs) else 1.0
                    ),
                }
            if frame_views:
                objects.append(
                    {
                        "bbox_min": bbox_min,
                        "bbox_max": bbox_max,
                        "frame_views": frame_views,
                    }
                )

    with _CONCEPTGRAPH_OBJECT_CACHE_LOCK:
        _remember_conceptgraph_objects(cache_key, objects)
    return objects


def _attach_conceptgraph_frame_views_to_pool(
    pool: dict[str, Any],
    *,
    scene_id: str,
    phase8_data_root: Path,
    min_iou: float = 0.25,
) -> None:
    """Attach ConceptGraph 2D frame boxes to matched VG proposals.

    This uses only prepared scene objects, not ScanRefer GT annotations.
    """
    objects = _load_conceptgraph_objects_for_frame_views(scene_id, phase8_data_root)
    if not objects:
        return
    for proposal in pool.get("proposals", []) or []:
        if not isinstance(proposal, dict) or proposal.get("frame_views"):
            continue
        bbox = proposal.get("bbox_3d_9dof")
        if not isinstance(bbox, list) or len(bbox) < 6:
            continue
        prop_min, prop_max = _bbox_minmax_from_9dof(bbox)
        best_obj: dict[str, Any] | None = None
        best_iou = 0.0
        for obj in objects:
            iou = _axis_aligned_iou_3d(
                prop_min,
                prop_max,
                obj["bbox_min"],
                obj["bbox_max"],
            )
            if iou > best_iou:
                best_iou = iou
                best_obj = obj
        if best_obj is None or best_iou < min_iou:
            continue
        proposal_id = int(proposal["id"])
        proposal["frame_views"] = {
            str(frame_id): {
                **view,
                "proposal_id": proposal_id,
            }
            for frame_id, view in best_obj["frame_views"].items()
        }


def build_pack_v1_bundle_from_sample(
    sample: dict[str, Any],
    data_root: Path,
    *,
    pack_name: str = "pack_scanrefer_v1",
    phase8_data_root: Path = Path("data/nr3d/scannet"),
):
    source = sample.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError(
            f"Sample {sample.get('sample_id')} must include non-empty source"
        )
    scene_dir = resolve_scene_artifacts_dir(sample, data_root, pack_name=pack_name)
    scene_id = str(sample["scene_id"])
    visibility_json = scene_dir / "visibility.json"
    if not visibility_json.exists():
        raise FileNotFoundError(f"Missing visibility index: {visibility_json}")
    frame_visibility = {
        int(k): [int(x) for x in v]
        for k, v in json.loads(visibility_json.read_text(encoding="utf-8")).items()
    }
    keyframes = [
        (
            int(kf["keyframe_idx"]),
            clean_keyframe_image_path(
                raw_frames_root=phase8_data_root,
                scene_id=scene_id,
                image_path=str(kf["image_path"]),
                frame_id=int(kf["frame_id"]),
            ),
            int(kf["frame_id"]),
        )
        for kf in sample.get("keyframes", [])
    ]
    if not keyframes:
        raise ValueError(f"Sample {sample.get('sample_id')} has no keyframes")
    bundle_builder = build_pack_v1_bundle
    if bundle_builder is None:
        from agents.examples.embodiedscan_vg_pack_v1_pilot import (
            build_pack_v1_bundle as bundle_builder,
        )

    bundle = bundle_builder(
        proposals_jsonl=scene_dir / "proposals.jsonl",
        source=source,
        annotated_image_dir=scene_dir / "annotated",
        frame_visibility=frame_visibility,
        keyframes=keyframes,
        scene_id=scene_id,
        query=sample.get("query"),
    )
    pool = (bundle.extra_metadata or {}).get("vg_proposal_pool")
    if isinstance(pool, dict):
        _attach_conceptgraph_frame_views_to_pool(
            pool,
            scene_id=str(sample["scene_id"]),
            phase8_data_root=phase8_data_root,
        )
    return bundle


def clean_keyframe_image_path(
    *,
    raw_frames_root: Path,
    scene_id: str,
    image_path: str,
    frame_id: int,
) -> str:
    path = Path(image_path)
    if path.parent.name != "annotated":
        return str(path)
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import (
        _resolve_raw_rgb_path,
    )

    return str(_resolve_raw_rgb_path(Path(raw_frames_root) / scene_id, int(frame_id)))


def resolve_scene_artifacts_dir(
    sample: dict[str, Any],
    data_root: Path,
    *,
    pack_name: str = "pack_scanrefer_v1",
) -> Path:
    raw = sample.get("scene_artifacts_dir")
    if raw:
        scene_dir = Path(raw)
        if pack_name not in scene_dir.parts:
            raise ValueError(
                f"sample {sample.get('sample_id')} has scene_artifacts_dir={scene_dir} "
                f"but caller requested pack_name={pack_name}"
            )
        if scene_dir.is_absolute() or scene_dir.exists():
            return scene_dir
        return data_root / scene_dir
    return data_root / str(sample["scene_id"]) / pack_name


def load_sample_artifact(
    data_root: Path,
    sample_id: str,
    *,
    pack_name: str = "pack_scanrefer_v1",
) -> dict[str, Any]:
    sample_path = expected_pack_sample_path(data_root, sample_id, pack_name=pack_name)
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing prepared sample artifact: {sample_path}")
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    if payload.get("sample_id") != sample_id:
        raise ValueError(
            f"Sample artifact {sample_path} has sample_id={payload.get('sample_id')!r}, "
            f"expected {sample_id!r}"
        )
    return payload


def expected_pack_sample_path(
    data_root: Path,
    sample_id: str,
    *,
    pack_name: str = "pack_scanrefer_v1",
) -> Path:
    parsed = parse_scanrefer_sample_id(sample_id)
    return (
        data_root
        / parsed.scene_id
        / pack_name
        / "samples"
        / f"{safe_sample_id(sample_id)}.json"
    )


def preflight_pack_sample_exists(
    data_root: Path,
    sample_id: str,
    *,
    pack_name: str = "pack_scanrefer_v1",
) -> None:
    expected_path = expected_pack_sample_path(data_root, sample_id, pack_name=pack_name)
    if expected_path.exists():
        return
    raise FileNotFoundError(
        f"pack_name={pack_name!r}: expected first sample at {expected_path}, "
        "but the file does not exist. Did you forget to run "
        "prepare_pack_v1_inputs_scanrefer?"
    )


def sample_result_path(
    output_dir: Path,
    backend: BackendName,
    sample_id: str,
    *,
    pack_name: str = "pack_scanrefer_v1",
) -> Path:
    if backend != "pack_v1":
        raise ValueError(f"backend={backend!r} is not supported; use 'pack_v1'")
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:12]
    safe = safe_sample_id(sample_id)
    if not safe:
        safe = "sample"
    return output_dir / "per_sample" / pack_name / f"{safe}_{digest}.json"


def load_sample_result_checkpoint(
    output_dir: Path,
    backend: BackendName,
    sample_id: str,
    *,
    pack_name: str = "pack_scanrefer_v1",
) -> dict[str, Any] | None:
    path = sample_result_path(
        output_dir,
        backend,
        sample_id,
        pack_name=pack_name,
    )
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must be a JSON object: {path}")
    if payload.get("sample_id") != sample_id:
        raise ValueError(
            f"Checkpoint {path} has sample_id={payload.get('sample_id')!r}, "
            f"expected {sample_id!r}"
        )
    if payload.get("backend") != backend:
        raise ValueError(
            f"Checkpoint {path} has backend={payload.get('backend')!r}, "
            f"expected {backend!r}"
        )
    return payload


def write_sample_result_checkpoint(
    output_dir: Path,
    backend: BackendName,
    result: dict[str, Any],
    *,
    pack_name: str = "pack_scanrefer_v1",
) -> Path:
    sample_id = result.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"Cannot checkpoint result without sample_id: {result!r}")
    if result.get("backend") != backend:
        raise ValueError(
            f"Cannot checkpoint {sample_id}: backend={result.get('backend')!r}, "
            f"expected {backend!r}"
        )
    path = sample_result_path(
        output_dir,
        backend,
        sample_id,
        pack_name=pack_name,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def load_required_sample_result_checkpoint(
    output_dir: Path,
    backend: BackendName,
    sample_id: str,
    *,
    pack_name: str = "pack_scanrefer_v1",
) -> dict[str, Any]:
    payload = load_sample_result_checkpoint(
        output_dir,
        backend,
        sample_id,
        pack_name=pack_name,
    )
    if payload is None:
        path = sample_result_path(
            output_dir,
            backend,
            sample_id,
            pack_name=pack_name,
        )
        raise FileNotFoundError(f"Missing per-sample checkpoint: {path}")
    return payload


def build_backend_payload_from_checkpoints(
    *,
    sample_ids: Sequence[str],
    output_dir: Path,
    backend: BackendName,
    pack_name: str = "pack_scanrefer_v1",
    include_per_sample: bool = True,
) -> dict[str, Any]:
    per_sample: list[dict[str, Any]] = []
    ious: list[float] = []
    for sample_id in sample_ids:
        record = load_required_sample_result_checkpoint(
            output_dir,
            backend,
            sample_id,
            pack_name=pack_name,
        )
        if record.get("iou") is not None:
            ious.append(float(record["iou"]))
        if include_per_sample:
            per_sample.append(record)
    payload: dict[str, Any] = {
        "n": len(sample_ids),
        "mean_iou": statistics.mean(ious) if ious else 0.0,
        "Acc@0.25": sum(1 for v in ious if v >= 0.25) / max(len(ious), 1),
        "Acc@0.50": sum(1 for v in ious if v >= 0.50) / max(len(ious), 1),
    }
    if include_per_sample:
        payload["per_sample"] = per_sample
    return payload


def stream_side_by_side_from_checkpoints(
    *,
    sample_ids: Sequence[str],
    output_dir: Path,
    backend: BackendName,
    pack_name: str = "pack_scanrefer_v1",
) -> Path:
    payload = build_backend_payload_from_checkpoints(
        sample_ids=sample_ids,
        output_dir=output_dir,
        backend=backend,
        pack_name=pack_name,
        include_per_sample=False,
    )
    output_path = output_dir / "side_by_side.json"
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        fh.write("{")
        fh.write(json.dumps(backend, ensure_ascii=False))
        fh.write(":{")
        for key in ("n", "mean_iou", "Acc@0.25", "Acc@0.50"):
            fh.write(json.dumps(key, ensure_ascii=False))
            fh.write(":")
            fh.write(json.dumps(payload[key], ensure_ascii=False))
            fh.write(",")
        fh.write('"per_sample":[')
        for index, sample_id in enumerate(sample_ids):
            if index:
                fh.write(",")
            record = load_required_sample_result_checkpoint(
                output_dir,
                backend,
                sample_id,
                pack_name=pack_name,
            )
            json.dump(record, fh, ensure_ascii=False)
        fh.write("]}}")
    tmp_path.replace(output_path)
    return output_path


def parse_scanrefer_sample_id(sample_id: str) -> ParsedScanRefSampleId:
    if not isinstance(sample_id, str):
        raise TypeError(
            "Expected sample_id in '<scene>::<target_id>::<ann_id>' "
            f"string format, got {type(sample_id).__name__}: {sample_id!r}"
        )
    parts = sample_id.split("::")
    if len(parts) != 3:
        raise ValueError(
            "Expected sample_id in '<scene>::<target_id>::<ann_id>' "
            f"format, got {sample_id!r}"
        )
    scan_id, target_text, ann_id = parts
    if not scan_id or not ann_id:
        raise ValueError(f"Invalid ScanRefer sample_id={sample_id!r}")
    try:
        target_id = int(target_text)
    except ValueError as exc:
        raise ValueError(f"Invalid target_id in sample_id={sample_id!r}") from exc
    return ParsedScanRefSampleId(
        scan_id=scan_id,
        scene_id=scan_id.split("/")[-1],
        target_id=target_id,
        ann_id=ann_id,
    )


def safe_sample_id(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("::", "__")


def validate_unique_sample_ids(sample_ids: Sequence[str]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for sample_id in sample_ids:
        if sample_id in seen:
            duplicates.append(sample_id)
        seen.add(sample_id)
    if duplicates:
        raise ValueError(
            "sample_ids must be unique for per-sample checkpointing; "
            f"duplicates={duplicates[:10]}"
        )


def config_for_backend(
    backend: BackendName,
    config: Stage2DeepAgentConfig | None,
) -> Stage2DeepAgentConfig:
    if backend != "pack_v1":
        raise ValueError(f"backend={backend!r} is not supported; use 'pack_v1'")
    if config is None:
        return Stage2DeepAgentConfig(vg_backend=backend)
    return config.model_copy(update={"vg_backend": backend})


def extract_pack_v1_prediction(result: Any) -> dict[str, Any]:
    payload = extract_result_payload(result)
    bbox_3d = payload.get("bbox_3d")
    selected_id = payload.get("selected_object_id", payload.get("proposal_id"))
    status = payload.get("status")
    if _is_failed_marker(selected_id, status) or selected_id == -1:
        return {
            "status": "failed",
            "selected_object_id": None,
            "bbox_3d": None,
            "confidence": payload.get("confidence", extract_result_confidence(result)),
        }
    if bbox_3d is None and selected_id is not None:
        bbox_3d = resolve_pack_v1_bbox_from_result(result, selected_id)
    if status is None and bbox_3d is not None:
        status = "completed"
    return {
        "status": status,
        "selected_object_id": selected_id,
        "bbox_3d": bbox_3d,
        "confidence": payload.get("confidence", extract_result_confidence(result)),
    }


def extract_result_payload(result: Any) -> dict[str, Any]:
    result_obj = getattr(result, "result", None)
    if result_obj is not None:
        payload = getattr(result_obj, "payload", None)
        if isinstance(payload, dict):
            return payload
    raise ValueError("pack_v1 result must expose result.payload as a dict")


def resolve_pack_v1_bbox_from_result(
    result: Any, proposal_id: Any
) -> list[float] | None:
    """Resolve a structured-response proposal_id to a pack-v1 bbox."""
    try:
        pid = int(proposal_id)
    except (TypeError, ValueError):
        return None
    final_bundle = getattr(result, "final_bundle", None)
    extra = (
        getattr(final_bundle, "extra_metadata", None)
        if final_bundle is not None
        else None
    )
    if not isinstance(extra, dict):
        return None
    pool = extra.get("vg_proposal_pool")
    if not isinstance(pool, dict):
        return None
    proposals = pool.get("proposals")
    if not isinstance(proposals, list):
        return None
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        try:
            candidate_id = int(proposal.get("id", -999999))
        except (TypeError, ValueError):
            continue
        if candidate_id != pid:
            continue
        bbox = proposal.get("bbox_3d_9dof")
        if isinstance(bbox, list) and len(bbox) == 9:
            return [float(x) for x in bbox]
    return None


def extract_result_confidence(result: Any) -> float | None:
    if result is None:
        return None
    result_obj = getattr(result, "result", result)
    value = getattr(result_obj, "confidence", None)
    return float(value) if value is not None else None


def extract_result_tool_trace(result: Any) -> list[dict[str, Any]]:
    """Return JSON-serializable tool observations from a Stage2AgentResult."""
    trace = getattr(result, "tool_trace", None)
    if not trace:
        return []
    out: list[dict[str, Any]] = []
    for item in trace:
        if hasattr(item, "model_dump"):
            out.append(item.model_dump())
        elif isinstance(item, dict):
            out.append(dict(item))
        else:
            out.append({"response_text": str(item)})
    return out


def coerce_bbox_9dof(raw: Any, *, field_name: str) -> list[float]:
    if isinstance(raw, str):
        raw = json.loads(raw.strip())
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"{field_name} must be a list/tuple, got {type(raw).__name__}")
    values = [float(v) for v in raw]
    if len(values) != 9:
        raise ValueError(
            f"{field_name} must contain exactly 9 floats, got {len(values)}"
        )
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{field_name} contains non-finite values: {values}")
    return values


def is_retryable_sample_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "prediction missing status",
            "429",
            "500",
            "503",
            "504",
            "timeout",
            "timed out",
            "connection reset",
            "internal error",
            "service hit an internal error",
            "-4399",
            "-4201",
            "-4307",
            "-4321",
            "invalid_prompt",
        )
    )


def _is_failed_marker(selected_object_id: Any, status: Any) -> bool:
    return selected_object_id is None and str(status).lower() == "failed"


def load_sample_ids(path: Path) -> list[str]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(
            f"sample_ids JSON must be a list, got {type(items).__name__}: {items!r}"
        )
    if all(isinstance(item, str) and item for item in items):
        return list(items)
    if all(isinstance(item, dict) for item in items):
        sample_ids: list[str] = []
        for index, item in enumerate(items):
            sample_id = item.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(
                    f"sample_ids[{index}] must include non-empty string "
                    f"'sample_id'; item={item!r}"
                )
            sample_ids.append(sample_id)
        return sample_ids
    raise ValueError(
        "sample_ids JSON must be either a list of strings or a list of dicts "
        f"with sample_id; got {items!r}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-ids", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help=(
            "ScanRefer scannet root, e.g. data/scanrefer/scannet; samples are read from "
            "<data_root>/<scene>/<pack_name>/samples/<safe_sample_id>.json."
        ),
    )
    parser.add_argument("--pack-name", default="pack_scanrefer_v1")
    parser.add_argument("--sample-retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        default=False,
        help=(
            "Only fill per-sample checkpoints; skip side_by_side.json assembly. "
            "Use with --max-new-samples for bounded-memory batch resumes."
        ),
    )
    parser.add_argument(
        "--max-new-samples",
        type=int,
        default=None,
        help=(
            "Process at most this many missing per-sample checkpoints in this "
            "process. Existing checkpoints are skipped without loading."
        ),
    )
    parser.add_argument(
        "--use-clip-visible-aug",
        action="store_true",
        default=False,
        help=(
            "Enable v3.5 CVRA visible-proposal CLIP augmentation in "
            "find_proposals_by_category."
        ),
    )
    parser.add_argument(
        "--use-tool-answer-disagreement-gate",
        action="store_true",
        default=False,
        help=(
            "Enable TADG: soft-block submit_final when the submitted "
            "proposal_id disagrees with the most recent matched-relation "
            "compare_proposals_spatial rank-1. Default off."
        ),
    )
    parser.add_argument(
        "--use-no-match-candidate-guard",
        action="store_true",
        default=False,
        help=(
            "Enable v3.7 no-match guard: soft-block submit_final(-1) when "
            "the agent's own tool trace still contains unresolved candidates."
        ),
    )
    parser.add_argument(
        "--use-evidence-frame-guard",
        action="store_true",
        default=False,
        help=(
            "Enable v3.7 evidence-frame guard: soft-block submit_final when "
            "the final rationale cites a marked frame that does not contain "
            "the submitted proposal id."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_ids = load_sample_ids(args.sample_ids)
    config = Stage2DeepAgentConfig(
        use_clip_visible_aug=args.use_clip_visible_aug,
        use_tool_answer_disagreement_gate=args.use_tool_answer_disagreement_gate,
        use_no_match_candidate_guard=args.use_no_match_candidate_guard,
        use_evidence_frame_guard=args.use_evidence_frame_guard,
    )
    compare_backends(
        sample_ids=sample_ids,
        output_dir=args.output_dir,
        data_root=args.data_root,
        pack_name=args.pack_name,
        config=config,
        sample_retries=args.sample_retries,
        workers=args.workers,
        return_results=False,
        write_side_by_side=not args.checkpoint_only,
        max_new_samples=args.max_new_samples,
    )


if __name__ == "__main__":
    main()
