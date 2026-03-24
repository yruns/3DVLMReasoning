#!/usr/bin/env python
"""Single-scene OpenEQA pilot for Stage 1, Stage 2, and end-to-end validation.

This script keeps the local OpenEQA data immutable. It creates a derived runtime
overlay scene under ``tmp/`` that exposes:

- ConceptGraph assets from ``conceptgraph/``
- Stage-1-compatible ``results/frameXXXXXX.jpg`` links built from ``raw/*-rgb.png``
- Stage-1-compatible ``results/depthXXXXXX.png`` links built from ``raw/*-depth.png``

Modes:
- ``stage1``: run only keyframe retrieval
- ``stage2``: run Stage 2 on a fixed evidence bundle, without Stage 1 callbacks
- ``e2e``: run Stage 1 -> Stage 2 with Stage 1 callbacks enabled
- ``all``: run the three checks in order and save artifacts
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from agents import (  # noqa: E402
    Stage1BackendCallbacks,
    Stage2DeepAgentConfig,
    Stage2DeepResearchAgent,
    Stage2TaskSpec,
    Stage2TaskType,
    build_stage2_evidence_bundle,
)
from query_scene.keyframe_selector import KeyframeResult, KeyframeSelector  # noqa: E402

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "OpenEQA" / "scannet"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "tmp" / "openeqa_runtime_cache"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tmp" / "openeqa_pilot_runs"
DEFAULT_CLIP_ID = "124-scannet-scene0131_02"
DEFAULT_STAGE1_QUERY = "the keyboard near the desktop computer"
DEFAULT_STAGE2_QUERY = (
    "Identify the object referred to by the query and briefly describe the visible "
    "setup around it."
)
DEFAULT_MODEL = "gemini-2.5-pro"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-scene OpenEQA pilot checks for Stage 1 / Stage 2 / E2E."
    )
    parser.add_argument(
        "--mode",
        choices=["stage1", "stage2", "e2e", "all"],
        default="all",
        help="Which validation slice to run.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing OpenEQA scene folders.",
    )
    parser.add_argument(
        "--clip-id",
        default=DEFAULT_CLIP_ID,
        help="Scene folder name under data-root.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="Derived runtime overlay root. Original data is not modified.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory to save stage artifacts.",
    )
    parser.add_argument(
        "--force-rebuild-overlay",
        action="store_true",
        help="Rebuild the derived runtime overlay even if it already exists.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=0,
        help="Frame stride for Stage 1. Default 0 means infer from visibility index metadata.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of initial keyframes to retrieve in Stage 1.",
    )
    parser.add_argument(
        "--stage1-query",
        default=DEFAULT_STAGE1_QUERY,
        help="Stage 1 retrieval query.",
    )
    parser.add_argument(
        "--stage2-query",
        default=DEFAULT_STAGE2_QUERY,
        help="Stage 2 user query.",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_MODEL,
        help="Parser / agent model name.",
    )
    parser.add_argument(
        "--max-reasoning-turns",
        type=int,
        default=3,
        help="Maximum Stage 2 reasoning turns.",
    )
    parser.add_argument(
        "--max-additional-views",
        type=int,
        default=2,
        help="Maximum extra views the Stage 1 callback can append in E2E mode.",
    )
    return parser.parse_args()


def ensure_symlink(target: Path, source: Path) -> None:
    if target.is_symlink():
        try:
            if target.resolve() == source.resolve():
                return
        except FileNotFoundError:
            pass
        target.unlink()
    elif target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def infer_stride(conceptgraph_dir: Path) -> int:
    visibility_index = conceptgraph_dir / "indices" / "visibility_index.pkl"
    if visibility_index.exists():
        with open(visibility_index, "rb") as f:
            data = pickle.load(f)
        stride = int((data.get("metadata") or {}).get("stride") or 0)
        if stride > 0:
            return stride
    return 5


def rebuild_results_overlay(results_dir: Path, raw_dir: Path) -> None:
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for rgb_path in sorted(raw_dir.glob("*-rgb.png")):
        frame_id = rgb_path.name.split("-")[0]
        link_path = results_dir / f"frame{frame_id}.jpg"
        link_path.symlink_to(rgb_path.resolve())

    for depth_path in sorted(raw_dir.glob("*-depth.png")):
        frame_id = depth_path.name.split("-")[0]
        link_path = results_dir / f"depth{frame_id}.png"
        link_path.symlink_to(depth_path.resolve())


def ensure_runtime_scene(
    scene_root: Path,
    cache_root: Path,
    force_rebuild_overlay: bool = False,
) -> Path:
    conceptgraph_dir = scene_root / "conceptgraph"
    raw_dir = scene_root / "raw"
    if not conceptgraph_dir.is_dir():
        raise FileNotFoundError(f"Missing conceptgraph dir: {conceptgraph_dir}")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing raw dir: {raw_dir}")

    overlay_dir = cache_root / scene_root.name
    if force_rebuild_overlay and overlay_dir.exists():
        shutil.rmtree(overlay_dir)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for child in conceptgraph_dir.iterdir():
        if child.name == "results":
            continue
        ensure_symlink(overlay_dir / child.name, child)

    results_dir = overlay_dir / "results"
    expected_rgb = len(list(raw_dir.glob("*-rgb.png")))
    expected_depth = len(list(raw_dir.glob("*-depth.png")))
    actual_rgb = len(list(results_dir.glob("frame*.jpg"))) if results_dir.exists() else 0
    actual_depth = (
        len(list(results_dir.glob("depth*.png"))) if results_dir.exists() else 0
    )
    if force_rebuild_overlay or actual_rgb != expected_rgb or actual_depth != expected_depth:
        rebuild_results_overlay(results_dir, raw_dir)

    return overlay_dir


def serialize_stage1_result(
    scene_root: Path,
    runtime_scene: Path,
    stride: int,
    selector: KeyframeSelector,
    result: KeyframeResult,
) -> dict[str, Any]:
    return {
        "scene_id": scene_root.name,
        "runtime_scene": str(runtime_scene),
        "stride": stride,
        "object_count": len(selector.objects),
        "pose_count": len(selector.camera_poses),
        "image_count": len(selector.image_paths),
        "query": result.query,
        "status": result.metadata.get("status"),
        "target_term": result.target_term,
        "anchor_term": result.anchor_term,
        "keyframe_indices": result.keyframe_indices,
        "keyframe_paths": [str(path) for path in result.keyframe_paths],
        "frame_mappings": result.metadata.get("frame_mappings", []),
        "hypothesis_kind": result.metadata.get("selected_hypothesis_kind"),
        "hypothesis_rank": result.metadata.get("selected_hypothesis_rank"),
        "target_object_count": len(result.target_objects),
        "anchor_object_count": len(result.anchor_objects),
    }


def serialize_stage2_result(name: str, result, initial_keyframes: int) -> dict[str, Any]:
    return {
        "run_name": name,
        "status": result.result.status.value,
        "confidence": result.result.confidence,
        "summary": result.result.summary,
        "payload": result.result.payload,
        "uncertainties": result.result.uncertainties,
        "cited_frame_indices": result.result.cited_frame_indices,
        "evidence_items": [item.model_dump() for item in result.result.evidence_items],
        "tool_trace": [item.model_dump() for item in result.tool_trace],
        "initial_keyframes": initial_keyframes,
        "final_keyframes": len(result.final_bundle.keyframes),
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def run_stage1(
    runtime_scene: Path,
    scene_root: Path,
    stride: int,
    query: str,
    k: int,
    llm_model: str,
) -> tuple[KeyframeSelector, KeyframeResult, dict[str, Any]]:
    logger.info("[Stage 1] scene={} stride={} query={!r}", scene_root.name, stride, query)
    selector = KeyframeSelector.from_scene_path(
        str(runtime_scene),
        stride=stride,
        llm_model=llm_model,
        use_pool=None,
    )
    result = selector.select_keyframes_v2(query, k=k)
    summary = serialize_stage1_result(scene_root, runtime_scene, stride, selector, result)
    if not result.keyframe_paths:
        raise RuntimeError(
            f"Stage 1 produced no keyframes. status={result.metadata.get('status')}"
        )
    logger.info(
        "[Stage 1] status={} keyframes={} target={} anchor={}",
        summary["status"],
        len(result.keyframe_paths),
        result.target_term,
        result.anchor_term,
    )
    return selector, result, summary


def build_bundle(
    selector: KeyframeSelector,
    stage1_result: KeyframeResult,
    scene_id: str,
) -> Any:
    return build_stage2_evidence_bundle(
        stage1_result,
        scene_id=scene_id,
        scene_summary=f"OpenEQA scene {scene_id} with {len(selector.objects)} detected objects.",
    )


def run_stage2(
    bundle,
    task_query: str,
    max_reasoning_turns: int,
    enable_callbacks: bool,
    selector: KeyframeSelector | None,
    scene_id: str,
    max_additional_views: int,
) -> Any:
    callbacks = None
    if enable_callbacks:
        if selector is None:
            raise ValueError("selector is required when callbacks are enabled")
        callbacks = Stage1BackendCallbacks(
            keyframe_selector=selector,
            scene_id=scene_id,
            max_additional_views=max_additional_views,
        )

    agent = Stage2DeepResearchAgent(
        config=Stage2DeepAgentConfig(
            include_thoughts=False,
            max_images=6,
            max_tokens=4000,
        ),
        more_views_callback=callbacks.more_views if callbacks else None,
        crop_callback=callbacks.crops if callbacks else None,
        hypothesis_callback=callbacks.hypothesis if callbacks else None,
    )
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.QA,
        user_query=task_query,
        max_reasoning_turns=max_reasoning_turns,
    )
    return agent.run(task, bundle.model_copy(deep=True))


def main() -> None:
    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:7} | {message}")

    scene_root = args.data_root / args.clip_id
    if not scene_root.is_dir():
        raise FileNotFoundError(f"Scene not found: {scene_root}")

    runtime_scene = ensure_runtime_scene(
        scene_root=scene_root,
        cache_root=args.cache_root,
        force_rebuild_overlay=args.force_rebuild_overlay,
    )
    stride = args.stride or infer_stride(scene_root / "conceptgraph")
    output_dir = args.output_root / args.clip_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("OpenEQA single-scene pilot")
    logger.info("scene_root={}", scene_root)
    logger.info("runtime_scene={}", runtime_scene)
    logger.info("output_dir={}", output_dir)

    selector, stage1_result, stage1_summary = run_stage1(
        runtime_scene=runtime_scene,
        scene_root=scene_root,
        stride=stride,
        query=args.stage1_query,
        k=args.k,
        llm_model=args.llm_model,
    )
    save_json(output_dir / "stage1.json", stage1_summary)

    bundle = build_bundle(selector, stage1_result, scene_root.name)

    if args.mode in {"stage2", "all"}:
        logger.info("[Stage 2] running without Stage 1 callbacks")
        stage2_result = run_stage2(
            bundle=bundle,
            task_query=args.stage2_query,
            max_reasoning_turns=args.max_reasoning_turns,
            enable_callbacks=False,
            selector=None,
            scene_id=scene_root.name,
            max_additional_views=args.max_additional_views,
        )
        stage2_summary = serialize_stage2_result(
            "stage2",
            stage2_result,
            initial_keyframes=len(bundle.keyframes),
        )
        save_json(output_dir / "stage2.json", stage2_summary)
        logger.info(
            "[Stage 2] status={} confidence={:.2f} tool_calls={}",
            stage2_summary["status"],
            stage2_summary["confidence"],
            len(stage2_summary["tool_trace"]),
        )

    if args.mode in {"e2e", "all"}:
        logger.info("[E2E] running with Stage 1 callbacks enabled")
        e2e_result = run_stage2(
            bundle=bundle,
            task_query=args.stage2_query,
            max_reasoning_turns=args.max_reasoning_turns,
            enable_callbacks=True,
            selector=selector,
            scene_id=scene_root.name,
            max_additional_views=args.max_additional_views,
        )
        e2e_summary = serialize_stage2_result(
            "e2e",
            e2e_result,
            initial_keyframes=len(bundle.keyframes),
        )
        save_json(output_dir / "e2e.json", e2e_summary)
        logger.info(
            "[E2E] status={} confidence={:.2f} tool_calls={} final_keyframes={}",
            e2e_summary["status"],
            e2e_summary["confidence"],
            len(e2e_summary["tool_trace"]),
            e2e_summary["final_keyframes"],
        )

    if args.mode == "stage1":
        logger.info("[Stage 1] artifact={}", output_dir / "stage1.json")
    elif args.mode == "stage2":
        logger.info("[Stage 2] artifacts={}, {}", output_dir / "stage1.json", output_dir / "stage2.json")
    elif args.mode == "e2e":
        logger.info("[E2E] artifacts={}, {}", output_dir / "stage1.json", output_dir / "e2e.json")
    else:
        logger.info(
            "[All] artifacts={}, {}, {}",
            output_dir / "stage1.json",
            output_dir / "stage2.json",
            output_dir / "e2e.json",
        )


if __name__ == "__main__":
    main()
