"""Run EmbodiedScan VG pack-v1 backend and report metrics."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import functools
import json
import math
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from agents.core.agent_config import Stage2DeepAgentConfig, Stage2TaskType
from agents.core.task_types import Stage2TaskSpec
from agents.examples.embodiedscan_vg_pack_v1_pilot import build_pack_v1_bundle
from agents.stage2_deep_agent import Stage2DeepResearchAgent
from benchmarks.embodiedscan_eval import compute_oriented_iou_3d

BackendName = Literal["pack_v1"]


def run_one_sample(
    sample_id: str,
    backend: BackendName,
    *,
    data_root: Path,
    config: Stage2DeepAgentConfig | None = None,
    sample_retries: int = 0,
) -> dict:
    """Run one sample through a backend and score predicted bbox against GT."""
    if sample_retries < 0:
        raise ValueError("sample_retries must be non-negative")
    last_error: Exception | None = None
    for attempt in range(sample_retries + 1):
        try:
            return _run_one_sample_once(
                sample_id,
                backend,
                data_root=data_root,
                config=config,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= sample_retries or not is_retryable_sample_error(exc):
                raise
            wait_s = 2.0 * (attempt + 1)
            logger.warning(
                "{} {} failed with retryable error on attempt {}/{}: {}. retry in {:.1f}s",
                sample_id,
                backend,
                attempt + 1,
                sample_retries + 1,
                exc,
                wait_s,
            )
            time.sleep(wait_s)
    if last_error is not None:
        raise last_error
    raise RuntimeError("run_one_sample retry loop exited unexpectedly")


def _run_one_sample_once(
    sample_id: str,
    backend: BackendName,
    *,
    data_root: Path,
    config: Stage2DeepAgentConfig | None = None,
) -> dict:
    """Run one sample once through a backend and score predicted bbox against GT."""
    sample = load_sample_artifact(data_root, sample_id)
    gt_bbox = coerce_bbox_9dof(
        sample.get("gt_bbox_3d_9dof"),
        field_name=f"{sample_id}.gt_bbox_3d_9dof",
    )
    cfg = config_for_backend(backend, config)

    if backend != "pack_v1":
        raise ValueError(
            f"backend={backend!r} no longer supported after Plan C; "
            "run the pack_v1 backend."
        )
    raw_result = run_pack_v1_sample(sample, data_root, cfg)
    prediction = extract_pack_v1_prediction(raw_result)

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
    }


def compare_backends(
    *,
    sample_ids: Sequence[str],
    output_dir: Path,
    data_root: Path,
    config: Stage2DeepAgentConfig | None = None,
    sample_retries: int = 0,
    workers: int = 1,
) -> dict:
    if workers <= 0:
        raise ValueError("workers must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for backend in ("pack_v1",):
        if workers == 1:
            per_sample = [
                run_one_sample(
                    s,
                    backend,
                    data_root=data_root,
                    config=config,
                    sample_retries=sample_retries,
                )
                for s in sample_ids
            ]
        else:
            run_sample = functools.partial(
                run_one_sample,
                backend=backend,
                data_root=data_root,
                config=config,
                sample_retries=sample_retries,
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                per_sample = list(executor.map(run_sample, sample_ids))
        ious = [r["iou"] for r in per_sample if r.get("iou") is not None]
        acc25 = sum(1 for v in ious if v >= 0.25) / max(len(ious), 1)
        acc50 = sum(1 for v in ious if v >= 0.50) / max(len(ious), 1)
        results[backend] = {
            "n": len(per_sample),
            "mean_iou": statistics.mean(ious) if ious else 0.0,
            "Acc@0.25": acc25,
            "Acc@0.50": acc50,
            "per_sample": per_sample,
        }
    (output_dir / "side_by_side.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("pack_v1: Acc@0.25={:.3f}", results["pack_v1"]["Acc@0.25"])
    return results


def config_for_backend(
    backend: BackendName,
    config: Stage2DeepAgentConfig | None,
) -> Stage2DeepAgentConfig:
    if backend != "pack_v1":
        raise ValueError(
            f"backend={backend!r} no longer supported after Plan C; "
            "run the pack_v1 backend."
        )
    if config is None:
        return Stage2DeepAgentConfig(vg_backend=backend)
    return config.model_copy(update={"vg_backend": backend})


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
            "read tcp",
            "internal error",
            "service hit an internal error",
            "-4399",
            "-4201",
        )
    )


def load_sample_artifact(data_root: Path, sample_id: str) -> dict[str, Any]:
    scene_id, target_id = parse_scene_target_id(sample_id)
    sample_path = data_root / scene_id / "pack_v1" / "samples" / f"{target_id}.json"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing prepared sample artifact: {sample_path}")
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    if payload.get("sample_id") != sample_id:
        raise ValueError(
            f"Sample artifact {sample_path} has sample_id={payload.get('sample_id')!r}, "
            f"expected {sample_id!r}"
        )
    return payload


def parse_scene_target_id(sample_id: str) -> tuple[str, int]:
    if not isinstance(sample_id, str):
        raise TypeError(
            f"Expected sample_id in '<scene_id>::<target_id>' string format, "
            f"got {type(sample_id).__name__}: {sample_id!r}"
        )
    if "::" not in sample_id:
        raise ValueError(
            f"Expected sample_id in '<scene_id>::<target_id>' format, got {sample_id!r}"
        )
    scene_id, target_text = sample_id.split("::", 1)
    if not scene_id:
        raise ValueError(f"Empty scene_id in sample_id={sample_id!r}")
    try:
        target_id = int(target_text)
    except ValueError as exc:
        raise ValueError(f"Invalid target_id in sample_id={sample_id!r}") from exc
    return scene_id, target_id


def run_pack_v1_sample(
    sample: dict[str, Any],
    data_root: Path,
    config: Stage2DeepAgentConfig,
) -> Any:
    bundle = build_pack_v1_bundle_from_sample(sample, data_root)
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        user_query=str(sample["query"]),
    )
    agent = Stage2DeepResearchAgent(config=config)
    return agent.run(task=task, bundle=bundle)


def build_pack_v1_bundle_from_sample(
    sample: dict[str, Any],
    data_root: Path,
):
    scene_dir = resolve_scene_artifacts_dir(sample, data_root)
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

    return build_pack_v1_bundle(
        proposals_jsonl=scene_dir / "proposals.jsonl",
        source=str(sample.get("source", "gt")),
        annotated_image_dir=scene_dir / "annotated",
        frame_visibility=frame_visibility,
        keyframes=keyframes,
        scene_id=str(sample["scene_id"]),
    )


def resolve_scene_artifacts_dir(
    sample: dict[str, Any],
    data_root: Path,
) -> Path:
    raw = sample.get("scene_artifacts_dir")
    if raw:
        scene_dir = Path(raw)
        if scene_dir.is_absolute() or scene_dir.exists():
            return scene_dir
        return data_root / scene_dir
    return data_root / str(sample["scene_id"]) / "pack_v1"


def extract_pack_v1_prediction(result: Any) -> dict[str, Any]:
    payload = extract_result_payload(result)
    bbox_3d = payload.get("bbox_3d")
    selected_id = payload.get("selected_object_id")
    status = payload.get("status")
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
    raise ValueError(
        "pack_v1 result must expose result.payload as a dict; "
        f"actual shape={describe_result_shape(result)}"
    )


def describe_result_shape(result: Any) -> str:
    if result is None:
        return "None"
    if isinstance(result, dict):
        return f"dict(keys={sorted(result.keys())!r})"
    result_obj = getattr(result, "result", None)
    result_shape = "missing"
    if result_obj is not None:
        if isinstance(result_obj, dict):
            result_shape = f"dict(keys={sorted(result_obj.keys())!r})"
        else:
            result_shape = (
                f"{type(result_obj).__name__}"
                f"(has_payload={hasattr(result_obj, 'payload')})"
            )
    return f"{type(result).__name__}(result={result_shape})"


def extract_result_confidence(result: Any) -> float | None:
    if result is None:
        return None
    if isinstance(result, dict):
        raw_result = result.get("result", result)
        if isinstance(raw_result, dict):
            value = raw_result.get("confidence")
            return float(value) if value is not None else None
    result_obj = getattr(result, "result", result)
    value = getattr(result_obj, "confidence", None)
    return float(value) if value is not None else None


def coerce_bbox_9dof(raw: Any, *, field_name: str) -> list[float]:
    if isinstance(raw, str):
        text = raw.strip()
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            try:
                raw = ast.literal_eval(text)
            except (SyntaxError, ValueError) as exc:
                raise TypeError(
                    f"{field_name} must be a list/tuple or serialized list, "
                    f"got str: {raw!r}"
                ) from exc
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
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sample-ids", required=True, type=Path, help="JSON file with [sample_id, ...]"
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="EmbodiedScan data root; samples are read from "
        "<data_root>/<scene_id>/pack_v1/samples/<target_id>.json.",
    )
    p.add_argument("--sample-retries", type=int, default=2)
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent sample workers for pack_v1 inference.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sample_ids = load_sample_ids(args.sample_ids)
    compare_backends(
        sample_ids=sample_ids,
        output_dir=args.output_dir,
        data_root=args.data_root,
        sample_retries=args.sample_retries,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
