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

import threading  # noqa: E402

from agents.examples.openeqa_single_scene_pilot import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL,
    build_bundle,
    ensure_runtime_scene,
    infer_stride,
    run_stage2,
    save_json,
    serialize_stage1_result,
    serialize_stage2_result,
)
from query_scene.keyframe_selector import KeyframeSelector  # noqa: E402
from benchmarks.openeqa_official_eval import (  # noqa: E402
    DEFAULT_OFFICIAL_REPO_ROOT,
    evaluate_predictions_with_official_llm_match,
)
from utils.llm_client import get_langchain_chat_model  # noqa: E402

# Per-scene lock to prevent parallel workers from racing on runtime cache setup
_scene_locks: dict[str, threading.Lock] = {}
_scene_locks_guard = threading.Lock()


def _get_scene_lock(clip_id: str) -> threading.Lock:
    with _scene_locks_guard:
        if clip_id not in _scene_locks:
            _scene_locks[clip_id] = threading.Lock()
        return _scene_locks[clip_id]


DEFAULT_JSON_PATH = PROJECT_ROOT / "data" / "open-eqa-v0.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tmp" / "openeqa_official_pilot_runs"

_QA_TO_RETRIEVAL_PROMPT = """\
You are a query rewriter for a 3D scene retrieval system. Given a question about a 3D scene, \
extract concise object-centric retrieval queries that would help locate the relevant objects.

Rules:
- Output 1-3 retrieval queries, one per line
- Each query should name the TARGET OBJECT and optionally its spatial context (anchor)
- Remove question words (what, where, which, how, is, are, does)
- Keep spatial relations (near, behind, under, left of, on top of, etc.)
- If the question asks about an attribute (color, size, material), query for the object itself
- Be concise: "red object below window" not "What is the red object that is below the windows"

Examples:
Q: What red object is below the windows?
red object below the windows
fire extinguisher near window

Q: What color are the blinds?
blinds
blinds near window

Q: What is behind the vacuum cleaner?
vacuum cleaner
objects behind vacuum cleaner

Q: What food is on the counter?
food on counter
counter top

Q: What is to the left of the office table?
objects left of office table
office table

Now rewrite:
Q: {question}
"""


def llm_rewrite_qa_to_retrieval(question: str) -> list[str]:
    """Use a fast LLM call to convert a QA question into retrieval queries."""
    prompt = _QA_TO_RETRIEVAL_PROMPT.format(question=question)
    client = get_langchain_chat_model("gemini-2.5-pro", temperature=0.0, max_tokens=128)
    response = client.invoke(prompt)
    text = getattr(response, "content", str(response)).strip()
    queries = [line.strip() for line in text.splitlines() if line.strip()]
    return queries


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
        "--num-scenes",
        type=int,
        default=None,
        help="Select N distinct scenes first, then pick questions from each.",
    )
    parser.add_argument(
        "--questions-per-scene",
        type=int,
        default=None,
        help="Number of questions to run per scene (requires --num-scenes).",
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
        default=10,
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel workers for running samples. "
            "Recommended: gemini_pool_size * 1.5 (e.g., 8 for 5-key pool). "
            "Set to 1 for sequential execution."
        ),
    )
    parser.add_argument(
        "--llm-rewrite",
        action="store_true",
        help=(
            "Use an LLM to rewrite QA-style questions into retrieval-oriented "
            "queries before Stage 1. Improves direct_grounded rate."
        ),
    )
    parser.add_argument(
        "--confidence-guard",
        type=float,
        default=0.6,
        help=(
            "Stage2 confidence threshold for skipping E2E. Only skip E2E when "
            "Stage2 completed with confidence >= this value. Below this threshold, "
            "non-deterministic downgrades. Set to 0 to disable."
        ),
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

    # --num-scenes N --questions-per-scene M: pick N scenes, M questions each
    if args.num_scenes is not None and args.questions_per_scene is not None:
        from collections import defaultdict

        by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sample in filtered:
            by_scene[sample["clip_id"]].append(sample)
        # Pick scenes that have enough questions, sorted by clip_id for reproducibility
        eligible = sorted(
            (
                (cid, qs)
                for cid, qs in by_scene.items()
                if len(qs) >= args.questions_per_scene
            ),
            key=lambda t: t[0],
        )
        selected: list[dict[str, Any]] = []
        for _clip_id, questions in eligible[: args.num_scenes]:
            selected.extend(questions[: args.questions_per_scene])
        return selected

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


_GROUNDING_RANK = {"direct_grounded": 0, "proxy_grounded": 1, "context_only": 2}


def run_stage1_ranked(
    sample: dict[str, Any],
    scene_root: Path,
    runtime_scene: Path,
    stride: int,
    args: argparse.Namespace,
):
    """Run Stage 1 with all query candidates, return the best by grounding quality.

    Ranking: direct_grounded > proxy_grounded > context_only.
    LLM rewrite is optional enhancement — rate-limit errors skip it, not abort.
    Stage 1 execution errors propagate (no silent fallback).
    """
    queries = build_stage1_query_candidates(sample["question"])

    # LLM rewrite: optional enhancement, inserts retrieval-oriented queries
    if getattr(args, "llm_rewrite", False):
        try:
            rewritten = llm_rewrite_qa_to_retrieval(sample["question"])
        except Exception as exc:
            logger.warning("[LLMRewrite] skipped ({}): {}", type(exc).__name__, exc)
            rewritten = None
        if rewritten:
            logger.info(
                "[LLMRewrite] {} -> {}",
                sample["question"][:60],
                rewritten,
            )
            seen = set()
            merged: list[str] = []
            for q in rewritten + queries:
                q_key = q.strip().lower()
                if q_key not in seen:
                    seen.add(q_key)
                    merged.append(q)
            queries = merged

    # Build selector once (expensive: loads scene data, CLIP features, BEV)
    selector = KeyframeSelector.from_scene_path(
        str(runtime_scene),
        stride=stride,
        llm_model=args.llm_model,
        use_pool=None,
    )

    best: tuple[int, str, KeyframeResult, dict[str, Any]] | None = None

    for query in queries:
        result = selector.select_keyframes_v2(query, k=args.k)
        summary = serialize_stage1_result(
            scene_root, runtime_scene, stride, selector, result
        )
        status = summary["status"]

        if not result.keyframe_paths:
            logger.warning(
                "[Stage1Ranked] query={!r} produced no keyframes (status={})",
                query,
                status,
            )
            continue

        rank = _GROUNDING_RANK.get(status, 99)
        logger.info(
            "[Stage1Ranked] query={!r} status={} rank={} keyframes={}",
            query,
            status,
            rank,
            len(result.keyframe_paths),
        )

        # direct_grounded is optimal — return immediately
        if status == "direct_grounded":
            summary["official_question"] = sample["question"]
            summary["stage1_query_used"] = query
            summary["stage1_query_candidates"] = queries
            return selector, result, summary

        if best is None or rank < best[0]:
            best = (rank, query, result, summary)

    if best is None:
        raise RuntimeError(
            f"Stage 1 produced no keyframes for any of {len(queries)} query variants"
        )

    _, used_query, best_result, best_summary = best
    best_summary["official_question"] = sample["question"]
    best_summary["stage1_query_used"] = used_query
    best_summary["stage1_query_candidates"] = queries
    return selector, best_result, best_summary


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

    # Per-scene lock: ensure_runtime_scene creates symlinks and is not thread-safe
    with _get_scene_lock(clip_id):
        runtime_scene = ensure_runtime_scene(
            scene_root, args.output_root / "runtime_cache"
        )
        stride = infer_stride(scene_root / "conceptgraph")

    selector, stage1_result, stage1_summary = run_stage1_ranked(
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

    # Stage 2 (no tools): baseline VLM reasoning on keyframes
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

    # E2E (with tools): run when Stage2 needs more evidence OR when Stage2
    # completed but with low confidence (likely a guess, not a confident answer).
    # Only skip E2E when Stage2 completed with high confidence.
    guard_threshold = getattr(args, "confidence_guard", 0.6)
    s2_confident = (
        stage2_summary["status"] == "completed"
        and stage2_summary["confidence"] >= guard_threshold
    )

    if not s2_confident:
        reason = (
            f"status={stage2_summary['status']}"
            if stage2_summary["status"] != "completed"
            else f"completed but low conf={stage2_summary['confidence']:.2f} < {guard_threshold}"
        )
        logger.info("[E2E] Stage2 {}, running E2E with tools", reason)
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
    else:
        logger.info(
            "[E2E] Stage2 completed with high confidence ({:.2f} >= {:.2f}), skipping E2E",
            stage2_summary["confidence"],
            guard_threshold,
        )
        e2e_summary = stage2_summary
        e2e_answer = stage2_answer

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
        "e2e_final_keyframes": e2e_summary.get("final_keyframes", len(bundle.keyframes)),
        "artifact_dir": str(sample_dir),
    }


def strip_local_fields(sample: dict[str, Any]) -> dict[str, Any]:
    item = dict(sample)
    item.pop("clip_id", None)
    return item


def build_prediction_file(
    results: list[dict[str, Any]], field_name: str
) -> list[dict[str, Any]]:
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
    logger.add(
        sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:7} | {message}"
    )

    samples = load_official_scannet_samples(args.json_path, args.data_root)
    logger.info(
        "Loaded {} official scannet questions from {}",
        len(samples),
        args.json_path,
    )

    candidate_pool = build_candidate_pool(samples, args)
    if not candidate_pool:
        raise RuntimeError("No official OpenEQA samples matched the requested filters.")

    if args.question_id:
        target_results = 1
    elif args.num_scenes is not None and args.questions_per_scene is not None:
        target_results = len(candidate_pool)  # already bounded by build_candidate_pool
    else:
        target_results = args.max_samples
    work_items = candidate_pool[:target_results]
    results: list[dict[str, Any]] = []

    def _run_sample(sample: dict[str, Any]) -> dict[str, Any]:
        logger.info(
            "[Official] running question_id={} clip={} category={}",
            sample["question_id"],
            sample["clip_id"],
            sample["category"],
        )
        row = run_one_sample(sample, args)
        logger.info(
            "[Official] done question_id={} stage1={} stage2={} e2e={} tools(e2e)={}",
            sample["question_id"],
            row["stage1_status"],
            row["stage2_status"],
            row["e2e_status"],
            row["e2e_tool_calls"],
        )
        return row

    num_workers = max(1, getattr(args, "workers", 1))
    if num_workers > 1:
        import concurrent.futures

        logger.info(
            "[Parallel] Running {} samples with {} workers",
            len(work_items),
            num_workers,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_sample = {
                executor.submit(_run_sample, sample): sample for sample in work_items
            }
            for future in concurrent.futures.as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    row = future.result()
                except Exception as exc:
                    logger.error(
                        "[Official] question_id={} failed: {}",
                        sample["question_id"],
                        exc,
                    )
                    continue
                results.append(row)
                logger.info(
                    "[Parallel] progress {}/{}",
                    len(results),
                    len(work_items),
                )
    else:
        for sample in work_items:
            row = _run_sample(sample)
            results.append(row)

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
