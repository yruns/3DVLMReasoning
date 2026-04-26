"""Run EmbodiedScan VG legacy + pack-v1 backends side-by-side and report."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from agents.core.agent_config import Stage2DeepAgentConfig, Stage2TaskType
from agents.core.task_types import Stage2TaskSpec
from agents.examples.embodiedscan_vg_pack_v1_pilot import build_pack_v1_bundle
from agents.stage2_deep_agent import Stage2DeepResearchAgent
from benchmarks.embodiedscan_eval import compute_oriented_iou_3d

BackendName = Literal["legacy", "pack_v1"]


def run_one_sample(
    sample_id: str,
    backend: BackendName,
    *,
    pack_v1_inputs_dir: Path,
    embodiedscan_data_root: Path,
    config: Stage2DeepAgentConfig | None = None,
) -> dict:
    """Run one sample through a backend and score predicted bbox against GT."""
    sample = load_sample_artifact(pack_v1_inputs_dir, sample_id)
    gt_bbox = coerce_bbox_9dof(
        sample.get("gt_bbox_3d_9dof"),
        field_name=f"{sample_id}.gt_bbox_3d_9dof",
    )
    cfg = config_for_backend(backend, config)

    if backend == "pack_v1":
        raw_result = run_pack_v1_sample(sample, pack_v1_inputs_dir, cfg)
        prediction = extract_pack_v1_prediction(raw_result)
    elif backend == "legacy":
        raw_result = run_legacy_pilot_sample(
            sample_id,
            embodiedscan_data_root=embodiedscan_data_root,
            config=cfg,
        )
        prediction = extract_legacy_prediction(raw_result)
    else:
        raise ValueError(f"Unknown backend={backend!r}")

    status = prediction.get("status") or "completed"
    proposal_id = prediction.get("proposal_id")
    selected_id = prediction.get("selected_object_id")
    if proposal_id is None and selected_id is not None:
        proposal_id = selected_id

    if _is_failed_marker(proposal_id, status, prediction.get("bbox_3d")):
        return {
            "sample_id": sample_id,
            "backend": backend,
            "status": "failed",
            "iou": 0.0,
            "predicted_bbox_3d": None,
            "gt_bbox_3d": gt_bbox,
            "selected_object_id": selected_id,
            "confidence": prediction.get("confidence"),
            "query": sample.get("query"),
        }

    pred_bbox = coerce_optional_bbox_9dof(
        prediction.get("bbox_3d"),
        field_name=f"{sample_id}.{backend}.bbox_3d",
    )
    iou = compute_oriented_iou_3d(pred_bbox, gt_bbox) if pred_bbox else 0.0
    if pred_bbox is None and status == "completed":
        status = "failed"

    return {
        "sample_id": sample_id,
        "backend": backend,
        "status": status,
        "iou": iou,
        "predicted_bbox_3d": pred_bbox,
        "gt_bbox_3d": gt_bbox,
        "selected_object_id": selected_id,
        "confidence": prediction.get("confidence"),
        "query": sample.get("query"),
    }


def compare_backends(
    *,
    sample_ids: Sequence[str],
    output_dir: Path,
    pack_v1_inputs_dir: Path,
    embodiedscan_data_root: Path,
    config: Stage2DeepAgentConfig | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for backend in ("legacy", "pack_v1"):
        per_sample = [
            run_one_sample(
                s,
                backend,
                pack_v1_inputs_dir=pack_v1_inputs_dir,
                embodiedscan_data_root=embodiedscan_data_root,
                config=config,
            )
            for s in sample_ids
        ]
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
    logger.info(
        "legacy: Acc@0.25={:.3f}  pack_v1: Acc@0.25={:.3f}",
        results["legacy"]["Acc@0.25"],
        results["pack_v1"]["Acc@0.25"],
    )
    return results


def config_for_backend(
    backend: BackendName,
    config: Stage2DeepAgentConfig | None,
) -> Stage2DeepAgentConfig:
    if config is None:
        return Stage2DeepAgentConfig(vg_backend=backend)
    return config.model_copy(update={"vg_backend": backend})


def load_sample_artifact(pack_v1_inputs_dir: Path, sample_id: str) -> dict[str, Any]:
    scene_id, target_id = parse_scene_target_id(sample_id)
    sample_path = pack_v1_inputs_dir / "samples" / f"{scene_id}__{target_id}.json"
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
    pack_v1_inputs_dir: Path,
    config: Stage2DeepAgentConfig,
) -> Any:
    bundle = build_pack_v1_bundle_from_sample(sample, pack_v1_inputs_dir)
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        user_query=str(sample["query"]),
    )
    agent = Stage2DeepResearchAgent(config=config)
    return agent.run(task=task, bundle=bundle)


def build_pack_v1_bundle_from_sample(
    sample: dict[str, Any],
    pack_v1_inputs_dir: Path,
):
    scene_dir = resolve_scene_artifacts_dir(sample, pack_v1_inputs_dir)
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
        source=str(sample.get("source", "vdetr")),
        annotated_image_dir=scene_dir / "annotated",
        frame_visibility=frame_visibility,
        keyframes=keyframes,
        scene_id=str(sample["scene_id"]),
    )


def resolve_scene_artifacts_dir(
    sample: dict[str, Any],
    pack_v1_inputs_dir: Path,
) -> Path:
    raw = sample.get("scene_artifacts_dir")
    if raw:
        scene_dir = Path(raw)
        if scene_dir.is_absolute() or scene_dir.exists():
            return scene_dir
        return pack_v1_inputs_dir / scene_dir
    return pack_v1_inputs_dir / "scenes" / str(sample["scene_id"])


def run_legacy_pilot_sample(
    sample_id: str,
    *,
    embodiedscan_data_root: Path,
    config: Stage2DeepAgentConfig,
) -> dict[str, Any]:
    """Run the existing legacy EmbodiedScan pilot for a scene::target sample."""
    from agents.adapters.embodiedscan_adapter import EmbodiedScanVGAdapter
    from agents.examples.embodiedscan_vg_pilot import (
        run_one_sample as run_legacy_one_sample,
    )

    scene_id, target_id = parse_scene_target_id(sample_id)
    adapter = EmbodiedScanVGAdapter(
        data_root=embodiedscan_data_root,
        scene_data_root=embodiedscan_data_root,
    )
    samples = adapter.load_samples(split="val", source_filter=None)
    sample = next(
        (
            s
            for s in samples
            if getattr(s, "scene_id", None) == scene_id
            and int(getattr(s, "target_id", -1)) == target_id
        ),
        None,
    )
    if sample is None:
        raise ValueError(
            f"Sample {sample_id!r} not found under {embodiedscan_data_root}"
        )
    return run_legacy_one_sample(sample, adapter, config)


def extract_pack_v1_prediction(result: Any) -> dict[str, Any]:
    payload = extract_result_payload(result)
    return {
        "status": payload.get("status", "completed"),
        "proposal_id": payload.get("proposal_id"),
        "selected_object_id": payload.get("selected_object_id"),
        "bbox_3d": payload.get("bbox_3d"),
        "confidence": payload.get("confidence", extract_result_confidence(result)),
    }


def extract_legacy_prediction(result: dict[str, Any]) -> dict[str, Any]:
    agent_result = result.get("result") if isinstance(result, dict) else None
    raw_state = extract_raw_state(agent_result)
    prediction = result.get("prediction", {}) if isinstance(result, dict) else {}
    bbox_3d = raw_state.get("vg_selected_bbox_3d")
    if bbox_3d is None:
        bbox_3d = prediction.get("bbox_3d")
    selected_id = raw_state.get("vg_selected_object_id")
    if selected_id is None:
        selected_id = prediction.get("selected_object_id")
    return {
        "status": "failed" if bbox_3d is None else "completed",
        "selected_object_id": selected_id,
        "bbox_3d": bbox_3d,
        "confidence": prediction.get(
            "confidence", extract_result_confidence(agent_result)
        ),
    }


def extract_result_payload(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        if isinstance(result.get("payload"), dict):
            return result["payload"]
        structured = result.get("structured_response")
        if isinstance(structured, dict) and isinstance(structured.get("payload"), dict):
            return structured["payload"]
        if isinstance(result.get("result"), dict):
            nested_result = result["result"]
            if isinstance(nested_result.get("payload"), dict):
                return nested_result["payload"]

    result_obj = getattr(result, "result", None)
    if result_obj is not None:
        payload = getattr(result_obj, "payload", None)
        if isinstance(payload, dict):
            return payload
        if isinstance(result_obj, dict) and isinstance(result_obj.get("payload"), dict):
            return result_obj["payload"]

    final_bundle = getattr(result, "final_bundle", None)
    extra_metadata = getattr(final_bundle, "extra_metadata", None)
    if isinstance(extra_metadata, dict):
        submission = extra_metadata.get("stage2_submission")
        if isinstance(submission, dict):
            return submission

    return {}


def extract_raw_state(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        raw_state = result.get("raw_state")
        return raw_state if isinstance(raw_state, dict) else {}
    raw_state = getattr(result, "raw_state", None)
    return raw_state if isinstance(raw_state, dict) else {}


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


def coerce_optional_bbox_9dof(raw: Any, *, field_name: str) -> list[float] | None:
    if raw is None:
        return None
    return coerce_bbox_9dof(raw, field_name=field_name)


def coerce_bbox_9dof(raw: Any, *, field_name: str) -> list[float]:
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            raise ValueError(f"{field_name} is empty")
        raw = json.loads(raw)

    if (
        isinstance(raw, (list, tuple))
        and len(raw) == 1
        and isinstance(raw[0], (list, tuple))
    ):
        raw = raw[0]

    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{field_name} must be a list, got {type(raw).__name__}")
    if len(raw) < 6:
        raise ValueError(f"{field_name} must contain at least 6 values, got {len(raw)}")

    values = [float(v) for v in raw[:9]]
    while len(values) < 9:
        values.append(0.0)
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{field_name} contains non-finite values")
    return values


def _is_failed_marker(proposal_id: Any, status: Any, bbox_3d: Any) -> bool:
    if proposal_id is not None:
        try:
            if int(proposal_id) == -1:
                return True
        except (TypeError, ValueError):
            pass
    return str(status).lower() == "failed" and bbox_3d is None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sample-ids", required=True, type=Path, help="JSON file with [sample_id, ...]"
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--pack-v1-inputs-dir", required=True, type=Path)
    p.add_argument("--embodiedscan-data-root", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sample_ids = json.loads(args.sample_ids.read_text())
    compare_backends(
        sample_ids=sample_ids,
        output_dir=args.output_dir,
        pack_v1_inputs_dir=args.pack_v1_inputs_dir,
        embodiedscan_data_root=args.embodiedscan_data_root,
    )


if __name__ == "__main__":
    main()
