#!/usr/bin/env python
"""Official OpenEQA question adapter for local prepared ScanNet scenes.

This adapter reuses the single-scene pilot pipeline but sources questions from
the official ``open-eqa-v0.json`` file. It supports:

- one-question runs via ``--question-id``
- scene-filtered runs via ``--clip-id``
- small batches via ``--max-samples``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.openeqa_official_eval import (  # noqa: E402
    DEFAULT_OFFICIAL_REPO_ROOT,
    evaluate_predictions_with_official_llm_match,
)
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

DEFAULT_JSON_PATH = PROJECT_ROOT / "data" / "open-eqa-v0.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tmp" / "openeqa_official_pilot_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local Stage 1/2 pipeline on official OpenEQA ScanNet questions."
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help="Path to official open-eqa-v0.json.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root containing local prepared OpenEQA ScanNet scenes.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for per-question artifacts and batch summary.",
    )
    parser.add_argument(
        "--question-id",
        default=None,
        help="Run a specific official question id.",
    )
    parser.add_argument(
        "--clip-id",
        default=None,
        help="Restrict to a specific local clip id.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Restrict to one official OpenEQA category.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="Maximum number of official questions to run when not using --question-id.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Initial Stage 1 keyframe count.",
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
        help="Maximum extra views that E2E callbacks can add per tool call.",
    )
    parser.add_argument(
        "--require-stage1-success",
        action="store_true",
        help="When batching, skip official questions that fail to retrieve keyframes in Stage 1.",
    )
    parser.add_argument(
        "--unique-scenes",
        action="store_true",
        help="When batching, keep at most one official question per local clip id.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run official OpenEQA LLM-match evaluation on stage2/e2e predictions.",
    )
    parser.add_argument(
        "--eval-model",
        default="gemini-2.5-pro",
        help="Evaluation judge model used via the current Azure-compatible backend.",
    )
    parser.add_argument(
        "--official-repo-root",
        type=Path,
        default=DEFAULT_OFFICIAL_REPO_ROOT,
        help="Path to the cloned official OpenEQA repo.",
    )
    return parser.parse_args()


def load_official_scannet_samples(
    json_path: Path,
    data_root: Path,
) -> list[dict[str, Any]]:
    raw = json.loads(json_path.read_text())
    local_clips = {path.name for path in data_root.iterdir() if path.is_dir()}

    samples: list[dict[str, Any]] = []
    for item in raw:
        episode_history = str(item.get("episode_history") or "")
        if not episode_history.startswith("scannet-v0/"):
            continue
        clip_id = episode_history.split("/", 1)[1]
        if clip_id not in local_clips:
            continue
        sample = dict(item)
        sample["clip_id"] = clip_id
        sample["category"] = item.get("category", "unknown")
        samples.append(sample)
    return samples


def matches_filters(sample: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.question_id and sample["question_id"] != args.question_id:
        return False
    if args.clip_id and sample["clip_id"] != args.clip_id:
        return False
    if args.category and sample["category"] != args.category:
        return False
    return True


def build_candidate_pool(
    all_samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    filtered = [sample for sample in all_samples if matches_filters(sample, args)]
    if args.question_id:
        return filtered[:1]

    if args.unique_scenes:
        unique: list[dict[str, Any]] = []
        seen_clips: set[str] = set()
        for sample in filtered:
            clip_id = sample["clip_id"]
            if clip_id in seen_clips:
                continue
            seen_clips.add(clip_id)
            unique.append(sample)
        filtered = unique

    return filtered


def build_stage1_query_candidates(question: str) -> list[str]:
    base = question.strip()
    if not base:
        return []

    trimmed = base.rstrip(" ?")
    lowered = trimmed.lower()
    candidates = [base, trimmed]

    heuristics = [
        lowered,
        re.sub(r"^what\s+is\s+", "", lowered),
        re.sub(r"^where\s+is\s+", "", lowered),
        re.sub(r"^where\s+are\s+", "", lowered),
        re.sub(r"^which\s+", "", lowered),
        re.sub(r"^what\s+", "", lowered),
        re.sub(r"\bis the below\b", "below", lowered),
        re.sub(r"\bis the above\b", "above", lowered),
    ]

    for item in heuristics:
        item = item.strip(" ?")
        item = re.sub(r"\s+", " ", item).strip()
        if not item:
            continue
        candidates.append(item)
        if not item.startswith("the "):
            candidates.append(f"the {item}")

    seen = set()
    unique: list[str] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def run_stage1_with_fallback(
    sample: dict[str, Any],
    scene_root: Path,
    runtime_scene: Path,
    stride: int,
    args: argparse.Namespace,
):
    queries = build_stage1_query_candidates(sample["question"])
    last_error: Exception | None = None

    for query in queries:
        try:
            selector, stage1_result, stage1_summary = run_stage1(
                runtime_scene=runtime_scene,
                scene_root=scene_root,
                stride=stride,
                query=query,
                k=args.k,
                llm_model=args.llm_model,
            )
            stage1_summary["official_question"] = sample["question"]
            stage1_summary["stage1_query_used"] = query
            stage1_summary["stage1_query_candidates"] = queries
            return selector, stage1_result, stage1_summary
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Stage 1 failed for official question after {len(queries)} query variants: {last_error}"
    )


def extract_prediction_text(stage_summary: dict[str, Any]) -> str | None:
    payload = stage_summary.get("payload")
    if isinstance(payload, dict):
        answer = payload.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
    summary = stage_summary.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return None


def run_one_sample(sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    clip_id = sample["clip_id"]
    scene_root = args.data_root / clip_id
    runtime_scene = ensure_runtime_scene(scene_root, args.output_root / "runtime_cache")
    stride = infer_stride(scene_root / "conceptgraph")

    selector, stage1_result, stage1_summary = run_stage1_with_fallback(
        sample=sample,
        scene_root=scene_root,
        runtime_scene=runtime_scene,
        stride=stride,
        args=args,
    )

    sample_dir = args.output_root / "runs" / clip_id / sample["question_id"]
    sample_meta = {
        "question_id": sample["question_id"],
        "question": sample["question"],
        "answer": sample["answer"],
        "category": sample["category"],
        "episode_history": sample["episode_history"],
        "clip_id": clip_id,
        "stage1_query_used": stage1_summary["stage1_query_used"],
        "stage1_query_candidates": stage1_summary["stage1_query_candidates"],
    }
    save_json(sample_dir / "sample.json", sample_meta)
    save_json(sample_dir / "stage1.json", stage1_summary)

    bundle = build_bundle(selector, stage1_result, clip_id)

    stage2_result = run_stage2(
        bundle=bundle,
        task_query=sample["question"],
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
    save_json(sample_dir / "stage2.json", stage2_summary)
    stage2_answer = extract_prediction_text(stage2_summary)

    e2e_result = run_stage2(
        bundle=bundle,
        task_query=sample["question"],
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
    save_json(sample_dir / "e2e.json", e2e_summary)
    e2e_answer = extract_prediction_text(e2e_summary)

    return {
        **sample_meta,
        "stage1_status": stage1_summary["status"],
        "stage1_keyframes": len(stage1_summary["keyframe_paths"]),
        "stage2_status": stage2_summary["status"],
        "stage2_answer": stage2_answer,
        "stage2_confidence": stage2_summary["confidence"],
        "stage2_tool_calls": len(stage2_summary["tool_trace"]),
        "e2e_status": e2e_summary["status"],
        "e2e_answer": e2e_answer,
        "e2e_confidence": e2e_summary["confidence"],
        "e2e_tool_calls": len(e2e_summary["tool_trace"]),
        "e2e_final_keyframes": e2e_summary["final_keyframes"],
        "artifact_dir": str(sample_dir),
    }


def strip_local_fields(sample: dict[str, Any]) -> dict[str, Any]:
    item = dict(sample)
    item.pop("clip_id", None)
    return item


def build_prediction_file(results: list[dict[str, Any]], field_name: str) -> list[dict[str, Any]]:
    return [
        {
            "question_id": row["question_id"],
            "answer": row.get(field_name),
        }
        for row in results
        if "question_id" in row
    ]


def main() -> None:
    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:7} | {message}")

    samples = load_official_scannet_samples(args.json_path, args.data_root)
    logger.info(
        "Loaded {} official scannet questions from {}",
        len(samples),
        args.json_path,
    )

    candidate_pool = build_candidate_pool(samples, args)
    if not candidate_pool:
        raise RuntimeError("No official OpenEQA samples matched the requested filters.")

    target_results = 1 if args.question_id else args.max_samples
    results: list[dict[str, Any]] = []
    for sample in candidate_pool:
        if len(results) >= target_results:
            break
        logger.info(
            "[Official] running question_id={} clip={} category={}",
            sample["question_id"],
            sample["clip_id"],
            sample["category"],
        )
        try:
            row = run_one_sample(sample, args)
            results.append(row)
            logger.info(
                "[Official] done question_id={} stage1={} stage2={} e2e={} tools(e2e)={}",
                sample["question_id"],
                row["stage1_status"],
                row["stage2_status"],
                row["e2e_status"],
                row["e2e_tool_calls"],
            )
        except Exception as exc:
            logger.exception("[Official] failed question_id={}", sample["question_id"])
            if args.require_stage1_success and not args.question_id:
                continue
            results.append(
                {
                    **sample,
                    "error": str(exc),
                }
            )

    summary = {
        "json_path": str(args.json_path),
        "num_loaded_scannet": len(samples),
        "num_candidate_pool": len(candidate_pool),
        "requested_results": target_results,
        "num_results": len(results),
        "unique_scenes": args.unique_scenes,
        "results": results,
    }

    if args.evaluate and results:
        question_id_to_sample = {sample["question_id"]: sample for sample in samples}
        dataset_subset = [
            strip_local_fields(question_id_to_sample[row["question_id"]])
            for row in results
            if row.get("question_id") in question_id_to_sample
        ]
        dataset_path = args.output_root / "official_selected_questions.json"
        save_json(dataset_path, dataset_subset)

        stage2_predictions = build_prediction_file(results, "stage2_answer")
        stage2_predictions_path = args.output_root / "official_predictions_stage2.json"
        save_json(stage2_predictions_path, stage2_predictions)
        stage2_eval = evaluate_predictions_with_official_llm_match(
            dataset_items=dataset_subset,
            predictions=stage2_predictions,
            output_path=args.output_root / "official_predictions_stage2-metrics.json",
            official_repo_root=args.official_repo_root,
            eval_model=args.eval_model,
        )

        e2e_predictions = build_prediction_file(results, "e2e_answer")
        e2e_predictions_path = args.output_root / "official_predictions_e2e.json"
        save_json(e2e_predictions_path, e2e_predictions)
        e2e_eval = evaluate_predictions_with_official_llm_match(
            dataset_items=dataset_subset,
            predictions=e2e_predictions,
            output_path=args.output_root / "official_predictions_e2e-metrics.json",
            official_repo_root=args.official_repo_root,
            eval_model=args.eval_model,
        )

        summary["evaluation"] = {
            "official_repo_root": str(args.official_repo_root),
            "eval_model": args.eval_model,
            "dataset_path": str(dataset_path),
            "stage2_predictions_path": str(stage2_predictions_path),
            "stage2": stage2_eval,
            "e2e_predictions_path": str(e2e_predictions_path),
            "e2e": e2e_eval,
        }
    elif args.evaluate:
        summary["evaluation"] = {
            "official_repo_root": str(args.official_repo_root),
            "eval_model": args.eval_model,
            "warning": "No results were available to evaluate.",
        }

    save_json(args.output_root / "official_batch_summary.json", summary)
    logger.info(
        "Saved official summary to {}",
        args.output_root / "official_batch_summary.json",
    )


if __name__ == "__main__":
    main()
