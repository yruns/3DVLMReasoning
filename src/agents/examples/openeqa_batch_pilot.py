#!/usr/bin/env python
"""Batch OpenEQA pilot runner built on top of the single-scene pilot flow.

This script:
1. Proposes per-scene queries from local ConceptGraph object metadata
2. Validates those queries with a real Stage 1 run
3. Saves a query set JSONL
4. Runs Stage 2 and E2E on a small batch of validated scenes
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.examples.openeqa_single_scene_pilot import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL,
    build_bundle,
    ensure_runtime_scene,
    infer_stride,
    run_stage1,
    run_stage2,
    save_json,
    serialize_stage2_result,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tmp" / "openeqa_batch_pilot_runs"
GENERIC_CATEGORIES = {
    "item",
    "object",
    "other item",
    "equipment",
    "appliance",
    "green",
    "seat",
    "writing",
    "toy",
    "bin",
    "container",
    "package",
    "box",
    "brown",
    "cloth",
    "tie",
    "balustrade",
    "pillar",
    "rope",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small-batch OpenEQA pilot using the Stage 1/2 single-scene pipeline."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing OpenEQA scene folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for batch artifacts and query set.",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=5,
        help="Number of validated scenes to include in the pilot batch.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Initial Stage 1 keyframe count per sample.",
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
    parser.add_argument(
        "--query-set-path",
        type=Path,
        default=None,
        help="Optional existing query set JSONL to reuse. If omitted, generate a validated one.",
    )
    return parser.parse_args()


def extract_category(obj: dict[str, Any]) -> str | None:
    names = [
        name
        for name in obj.get("class_name", [])
        if name and str(name).lower() not in ("item", "object", "none", "")
    ]
    if not names:
        return None
    return Counter(names).most_common(1)[0][0]


def propose_scene_query(scene_root: Path) -> dict[str, Any] | None:
    pcd_dir = scene_root / "conceptgraph" / "pcd_saves"
    pcd_files = list(pcd_dir.glob("*post.pkl.gz"))
    if not pcd_files:
        return None

    with gzip.open(pcd_files[0], "rb") as f:
        data = pickle.load(f)

    raw_objects: list[tuple[str, np.ndarray]] = []
    for obj in data.get("objects", []):
        category = extract_category(obj)
        if not category:
            continue
        pcd_np = obj.get("pcd_np")
        if pcd_np is None or len(pcd_np) == 0:
            continue
        centroid = np.asarray(pcd_np, dtype=np.float32).mean(axis=0)
        raw_objects.append((category, centroid))

    counts = Counter(category for category, _ in raw_objects)
    filtered = [
        (category, centroid)
        for category, centroid in raw_objects
        if counts[category] >= 2 and category.lower() not in GENERIC_CATEGORIES
    ]

    best_pair: tuple[float, str, str, float] | None = None
    for idx_a, (target, center_a) in enumerate(filtered):
        for idx_b, (anchor, center_b) in enumerate(filtered):
            if idx_a == idx_b or target == anchor:
                continue
            dist = float(np.linalg.norm(center_a - center_b))
            if dist > 1.8:
                continue
            score = dist + 0.05 * (counts[target] + counts[anchor])
            if best_pair is None or score < best_pair[0]:
                best_pair = (score, target, anchor, dist)

    if best_pair is not None:
        _, target, anchor, dist = best_pair
        return {
            "clip_id": scene_root.name,
            "stage1_query": f"the {target} near the {anchor}",
            "stage2_query": (
                f"List the major objects around the {target} and describe the overall "
                f"setup near the {anchor}. If the current images do not show enough "
                "context, request more views before answering."
            ),
            "query_strategy": "nearest_pair",
            "target_category": target,
            "anchor_category": anchor,
            "pair_distance": round(dist, 4),
        }

    if not filtered:
        return None

    target = counts.most_common(1)[0][0]
    return {
        "clip_id": scene_root.name,
        "stage1_query": f"the {target}",
        "stage2_query": (
            f"Describe the visible {target} and the nearby setup. If the current "
            "images are insufficient, request more views before answering."
        ),
        "query_strategy": "single_category_fallback",
        "target_category": target,
        "anchor_category": None,
        "pair_distance": None,
    }


def load_query_set(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        entries.append(json.loads(line))
    return entries


def save_query_set(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_batch_sample(
    sample: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    clip_id = sample["clip_id"]
    scene_root = args.data_root / clip_id
    runtime_scene = ensure_runtime_scene(scene_root, args.output_root / "runtime_cache")
    stride = sample.get("stride") or infer_stride(scene_root / "conceptgraph")

    selector, stage1_result, stage1_summary = run_stage1(
        runtime_scene=runtime_scene,
        scene_root=scene_root,
        stride=stride,
        query=sample["stage1_query"],
        k=args.k,
        llm_model=args.llm_model,
    )

    scene_output_dir = args.output_root / "runs" / clip_id
    save_json(scene_output_dir / "stage1.json", stage1_summary)

    bundle = build_bundle(selector, stage1_result, clip_id)

    stage2_result = run_stage2(
        bundle=bundle,
        task_query=sample["stage2_query"],
        max_reasoning_turns=args.max_reasoning_turns,
        enable_callbacks=False,
        selector=None,
        scene_id=clip_id,
        max_additional_views=args.max_additional_views,
    )
    stage2_summary = serialize_stage2_result(
        "stage2",
        stage2_result,
        initial_keyframes=len(bundle.keyframes),
    )
    save_json(scene_output_dir / "stage2.json", stage2_summary)

    e2e_result = run_stage2(
        bundle=bundle,
        task_query=sample["stage2_query"],
        max_reasoning_turns=args.max_reasoning_turns,
        enable_callbacks=True,
        selector=selector,
        scene_id=clip_id,
        max_additional_views=args.max_additional_views,
    )
    e2e_summary = serialize_stage2_result(
        "e2e",
        e2e_result,
        initial_keyframes=len(bundle.keyframes),
    )
    save_json(scene_output_dir / "e2e.json", e2e_summary)

    return {
        "clip_id": clip_id,
        "stage1_query": sample["stage1_query"],
        "stage2_query": sample["stage2_query"],
        "query_strategy": sample["query_strategy"],
        "target_category": sample.get("target_category"),
        "anchor_category": sample.get("anchor_category"),
        "pair_distance": sample.get("pair_distance"),
        "stage1_status": stage1_summary["status"],
        "stage1_keyframes": len(stage1_result.keyframe_paths),
        "stage2_status": stage2_summary["status"],
        "stage2_confidence": stage2_summary["confidence"],
        "stage2_tool_calls": len(stage2_summary["tool_trace"]),
        "e2e_status": e2e_summary["status"],
        "e2e_confidence": e2e_summary["confidence"],
        "e2e_tool_calls": len(e2e_summary["tool_trace"]),
        "e2e_final_keyframes": e2e_summary["final_keyframes"],
        "artifact_dir": str(scene_output_dir),
    }


def main() -> None:
    args = parse_args()

    logger.remove()
    logger.add(
        sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:7} | {message}"
    )

    query_set_path = args.query_set_path or (args.output_root / "query_set.jsonl")

    if query_set_path.exists():
        query_set = load_query_set(query_set_path)
        logger.info(
            "Loaded existing query set: {} entries from {}",
            len(query_set),
            query_set_path,
        )
    else:
        logger.info(
            "Generating validated query set for up to {} scenes", args.num_scenes
        )
        query_set = []
        for scene_root in sorted(
            path for path in args.data_root.iterdir() if path.is_dir()
        ):
            if len(query_set) >= args.num_scenes:
                break
            proposal = propose_scene_query(scene_root)
            if proposal is None:
                continue
            try:
                runtime_scene = ensure_runtime_scene(
                    scene_root,
                    args.output_root / "runtime_cache",
                )
                stride = infer_stride(scene_root / "conceptgraph")
                _selector, stage1_result, _summary = run_stage1(
                    runtime_scene=runtime_scene,
                    scene_root=scene_root,
                    stride=stride,
                    query=proposal["stage1_query"],
                    k=args.k,
                    llm_model=args.llm_model,
                )
                if not stage1_result.keyframe_paths:
                    continue
                proposal["stride"] = stride
                query_set.append(proposal)
                logger.info(
                    "[QuerySet] accepted scene={} query={!r}",
                    scene_root.name,
                    proposal["stage1_query"],
                )
            except Exception as exc:
                logger.warning(
                    "[QuerySet] rejected scene={} query={!r} error={}",
                    scene_root.name,
                    proposal["stage1_query"],
                    exc,
                )
        save_query_set(query_set_path, query_set)
        logger.info(
            "Saved query set with {} entries to {}", len(query_set), query_set_path
        )

    if len(query_set) == 0:
        raise RuntimeError("No valid query-set entries available.")

    selected = query_set[: args.num_scenes]
    batch_rows: list[dict[str, Any]] = []

    for sample in selected:
        clip_id = sample["clip_id"]
        logger.info("[Batch] running clip={}", clip_id)
        try:
            row = run_batch_sample(sample, args)
            batch_rows.append(row)
            logger.info(
                "[Batch] done clip={} stage1={} stage2={} e2e={} tools(e2e)={}",
                clip_id,
                row["stage1_status"],
                row["stage2_status"],
                row["e2e_status"],
                row["e2e_tool_calls"],
            )
        except Exception as exc:
            logger.exception("[Batch] failed clip={}", clip_id)
            batch_rows.append(
                {
                    "clip_id": clip_id,
                    "stage1_query": sample["stage1_query"],
                    "stage2_query": sample["stage2_query"],
                    "error": str(exc),
                }
            )

    summary = {
        "num_requested": args.num_scenes,
        "num_selected": len(selected),
        "query_set_path": str(query_set_path),
        "results": batch_rows,
    }
    save_json(args.output_root / "batch_summary.json", summary)
    logger.info("Saved batch summary to {}", args.output_root / "batch_summary.json")


if __name__ == "__main__":
    main()
