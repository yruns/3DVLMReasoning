"""Run ScanRefer VG pack-v1 backend and report metrics."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import threading
import time
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
_SELECTOR_CACHE: dict[str, Any] = {}
_SELECTOR_CACHE_LOCK = threading.Lock()
_SELECTOR_BUILD_LOCKS: dict[str, threading.Lock] = {}


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
            return _SELECTOR_CACHE[scene_id]
        build_lock = _SELECTOR_BUILD_LOCKS.setdefault(scene_id, threading.Lock())

    with build_lock:
        # Re-check inside the per-scene lock in case another worker built it
        with _SELECTOR_CACHE_LOCK:
            if scene_id in _SELECTOR_CACHE:
                return _SELECTOR_CACHE[scene_id]

        cg_root = phase8_data_root / scene_id / "conceptgraph"
        enriched = cg_root / "enriched_objects.json"
        if not enriched.exists():
            with _SELECTOR_CACHE_LOCK:
                _SELECTOR_CACHE[scene_id] = None
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
            _SELECTOR_CACHE[scene_id] = selector
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
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    validate_unique_sample_ids(sample_ids)
    if sample_ids:
        preflight_pack_sample_exists(data_root, sample_ids[0], pack_name=pack_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for backend in ("pack_v1",):

        def run_sample(
            sample_id: str, backend: BackendName = backend
        ) -> dict[str, Any]:
            cached = load_sample_result_checkpoint(
                output_dir,
                backend,
                sample_id,
                pack_name=pack_name,
            )
            if cached is not None:
                return cached
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

        if workers == 1:
            per_sample = [run_sample(sample_id) for sample_id in sample_ids]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                per_sample = list(executor.map(run_sample, sample_ids))
        ious = [float(r["iou"]) for r in per_sample if r.get("iou") is not None]
        results[backend] = {
            "n": len(per_sample),
            "mean_iou": statistics.mean(ious) if ious else 0.0,
            "Acc@0.25": sum(1 for v in ious if v >= 0.25) / max(len(ious), 1),
            "Acc@0.50": sum(1 for v in ious if v >= 0.50) / max(len(ious), 1),
            "per_sample": per_sample,
        }
    (output_dir / "side_by_side.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results


def run_pack_v1_sample(
    sample: dict[str, Any],
    data_root: Path,
    config: Stage2DeepAgentConfig,
    *,
    pack_name: str = "pack_scanrefer_v1",
    phase8_data_root: Path = Path("data/nr3d/scannet"),
    enable_stage1_callback: bool = True,
) -> Any:
    bundle = build_pack_v1_bundle_from_sample(sample, data_root, pack_name=pack_name)
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        user_query=str(sample["query"]),
    )
    agent_cls = Stage2DeepResearchAgent
    if agent_cls is None:
        from agents.stage2_deep_agent import Stage2DeepResearchAgent as agent_cls

    # Wire all three Stage 1 ↔ Stage 2 callbacks so the agent has the full
    # set of OpenEQA-style evidence acquisition tools available:
    #   - request_more_views (targeted/explore/temporal_fan modes) —
    #     more_views_callback
    #   - request_crops (object-centric red-bbox crops) — crop_callback
    #   - switch_or_expand_hypothesis (re-run Stage 1 with refined query) —
    #     hypothesis_callback
    # Combined, the agent can iteratively pull fresh visual evidence from any
    # of four paths: cross-frame view (view_keyframe_marked, no callback),
    # OpenEQA targeted/explore views (request_more_views), object crops
    # (request_crops), or full Stage 1 re-query (switch_or_expand_hypothesis).
    # This iterative retrieval loop is the project's key innovation vs
    # one-shot Camp-A baselines (Z3D / ZSVG3D / SeeGround / CSVG).
    more_views_callback = None
    crop_callback = None
    hypothesis_callback = None
    if enable_stage1_callback:
        scene_id = str(sample["scene_id"])
        selector = _get_or_build_keyframe_selector(scene_id, phase8_data_root)
        if selector is not None:
            from agents.stage1_callbacks import (
                create_crop_callback,
                create_hypothesis_callback,
                create_more_views_callback,
            )

            more_views_callback = create_more_views_callback(
                selector,
                scene_id=scene_id,
                max_additional_views=3,
            )
            crop_callback = create_crop_callback(
                selector,
                scene_id=scene_id,
                crop_scale=2.0,
            )
            hypothesis_callback = create_hypothesis_callback(
                selector,
                scene_id=scene_id,
                max_new_keyframes=3,
            )

    agent = agent_cls(
        config=config,
        more_views_callback=more_views_callback,
        crop_callback=crop_callback,
        hypothesis_callback=hypothesis_callback,
    )
    return agent.run(task=task, bundle=bundle)


def build_pack_v1_bundle_from_sample(
    sample: dict[str, Any],
    data_root: Path,
    *,
    pack_name: str = "pack_scanrefer_v1",
):
    source = sample.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError(
            f"Sample {sample.get('sample_id')} must include non-empty source"
        )
    scene_dir = resolve_scene_artifacts_dir(sample, data_root, pack_name=pack_name)
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
            str(kf["image_path"]),
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

    return bundle_builder(
        proposals_jsonl=scene_dir / "proposals.jsonl",
        source=source,
        annotated_image_dir=scene_dir / "annotated",
        frame_visibility=frame_visibility,
        keyframes=keyframes,
        scene_id=str(sample["scene_id"]),
    )


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
            "timeout",
            "timed out",
            "connection reset",
            "internal error",
            "service hit an internal error",
            "-4399",
            "-4201",
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
        "--use-clip-visible-aug",
        action="store_true",
        default=False,
        help=(
            "Enable v3.5 CVRA visible-proposal CLIP augmentation in "
            "find_proposals_by_category."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_ids = load_sample_ids(args.sample_ids)
    config = Stage2DeepAgentConfig(
        use_clip_visible_aug=args.use_clip_visible_aug,
    )
    compare_backends(
        sample_ids=sample_ids,
        output_dir=args.output_dir,
        data_root=args.data_root,
        pack_name=args.pack_name,
        config=config,
        sample_retries=args.sample_retries,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
