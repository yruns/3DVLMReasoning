"""Space3D-Bench evaluation script for Stage 2 Agent.

This script runs Stage 2 Agent on Space3D-Bench questions using
the benchmark-provided object detections as scene context.

Usage:
    .venv/bin/python -m agents.examples.space3d_bench_eval \
        --scene room_0 --num_questions 5

    # Or run on all scenes with limited questions per scene:
    .venv/bin/python -m agents.examples.space3d_bench_eval --all --per_scene 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.models import Stage2DeepAgentConfig
from agents.space3d_bench_adapter import (
    Space3DBenchAdapter,
    evaluate_answer,
)
from agents.stage2_deep_agent import Stage2DeepResearchAgent


@dataclass
class EvalResult:
    """Single evaluation result."""

    scene_id: str
    question_id: str
    question: str
    prediction: str
    ground_truth: dict[str, Any]
    evaluation: dict[str, Any]
    latency_ms: float
    status: str
    error: str | None = None


@dataclass
class EvalSummary:
    """Evaluation summary across questions."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    subjective: int = 0
    results: list[EvalResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        evaluated = self.total - self.subjective - self.errors
        if evaluated == 0:
            return 0.0
        return self.passed / evaluated

    def add(self, result: EvalResult) -> None:
        self.results.append(result)
        self.total += 1

        if result.status == "error":
            self.errors += 1
        elif result.evaluation.get("type") == "subjective":
            self.subjective += 1
        elif result.evaluation.get("match", False):
            self.passed += 1
        else:
            self.failed += 1


def create_agent_config() -> Stage2DeepAgentConfig:
    """Create Stage 2 Agent config.

    Uses default config which has API key baked in.
    Can override via environment variables if needed.
    """
    config = Stage2DeepAgentConfig()

    # Allow override from environment
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
    if api_key:
        config = Stage2DeepAgentConfig(
            api_key=api_key,
            session_id="space3d_bench_eval",
        )
    else:
        config = Stage2DeepAgentConfig(
            temperature=0.2,  # Lower for more deterministic answers
            session_id="space3d_bench_eval",
        )

    return config


def run_single_question(
    agent: Stage2DeepResearchAgent,
    adapter: Space3DBenchAdapter,
    scene_id: str,
    question_id: str,
) -> EvalResult:
    """Run agent on a single Space3D-Bench question."""

    sample = adapter.loader.get_sample(scene_id, question_id)
    gt = adapter.get_ground_truth(scene_id, question_id)

    logger.info(f"Q{question_id}: {sample.question}")

    start = time.time()

    try:
        # Prepare inputs from adapter
        task, bundle = adapter.prepare_inputs(scene_id, question_id)

        # Run agent
        result = agent.run(task, bundle)

        latency_ms = (time.time() - start) * 1000

        # Extract prediction from structured response
        prediction = ""
        if result.result:
            payload = result.result.payload
            if isinstance(payload, dict):
                # Combine relevant fields
                parts = []
                if "answer" in payload:
                    parts.append(str(payload["answer"]))
                if "target_description" in payload:
                    parts.append(str(payload["target_description"]))
                if "result" in payload:
                    parts.append(str(payload["result"]))
                prediction = " ".join(parts) if parts else json.dumps(payload)
            else:
                prediction = str(payload)

            # Also use summary if available
            if not prediction and result.result.summary:
                prediction = result.result.summary

        # Evaluate
        eval_result = evaluate_answer(prediction, gt)

        logger.info(f"  Answer: {prediction[:100]}...")
        logger.info(f"  GT: {gt.get('answer', 'N/A')}")
        logger.info(f"  Match: {eval_result.get('match', 'N/A')}")

        return EvalResult(
            scene_id=scene_id,
            question_id=question_id,
            question=sample.question,
            prediction=prediction,
            ground_truth=gt,
            evaluation=eval_result,
            latency_ms=latency_ms,
            status="success",
        )

    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        logger.error(f"  Error: {e}")

        return EvalResult(
            scene_id=scene_id,
            question_id=question_id,
            question=sample.question,
            prediction="",
            ground_truth=gt,
            evaluation={},
            latency_ms=latency_ms,
            status="error",
            error=str(e),
        )


def run_evaluation(
    data_root: Path,
    scenes: list[str] | None = None,
    num_questions: int | None = None,
    question_types: list[str] | None = None,
) -> EvalSummary:
    """Run Space3D-Bench evaluation.

    Args:
        data_root: Path to Space3D-Bench data
        scenes: List of scene IDs to evaluate (None = all)
        num_questions: Max questions per scene (None = all)
        question_types: Filter by question type (e.g., ["yes_no", "position"])
    """
    # Initialize adapter
    adapter = Space3DBenchAdapter.from_data_root(data_root)
    available_scenes = adapter.loader.list_scenes()

    if not available_scenes:
        logger.error(f"No scenes found in {data_root}")
        return EvalSummary()

    logger.info(f"Available scenes: {available_scenes}")

    # Filter scenes
    if scenes:
        target_scenes = [s for s in scenes if s in available_scenes]
        if not target_scenes:
            logger.error(f"None of {scenes} found in benchmark")
            return EvalSummary()
    else:
        target_scenes = available_scenes

    # Initialize agent
    config = create_agent_config()
    agent = Stage2DeepResearchAgent(config=config)

    summary = EvalSummary()

    for scene_id in target_scenes:
        logger.info(f"\n=== Evaluating scene: {scene_id} ===")

        scene = adapter.loader.load_scene(scene_id)
        question_ids = list(scene.questions.keys())

        # Limit questions
        if num_questions:
            question_ids = question_ids[:num_questions]

        for qid in question_ids:
            result = run_single_question(agent, adapter, scene_id, qid)
            summary.add(result)

            # Brief pause to avoid rate limiting
            time.sleep(0.5)

    return summary


def print_summary(summary: EvalSummary) -> None:
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("SPACE3D-BENCH EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions:     {summary.total}")
    print(f"Passed:              {summary.passed}")
    print(f"Failed:              {summary.failed}")
    print(f"Errors:              {summary.errors}")
    print(f"Subjective (VLM):    {summary.subjective}")
    print(f"Accuracy:            {summary.accuracy:.2%}")
    print("=" * 60)

    # Per-scene breakdown
    scene_stats: dict[str, dict[str, int]] = {}
    for r in summary.results:
        if r.scene_id not in scene_stats:
            scene_stats[r.scene_id] = {"total": 0, "passed": 0}
        scene_stats[r.scene_id]["total"] += 1
        if r.evaluation.get("match", False):
            scene_stats[r.scene_id]["passed"] += 1

    print("\nPer-scene breakdown:")
    for scene_id, stats in sorted(scene_stats.items()):
        acc = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {scene_id}: {stats['passed']}/{stats['total']} ({acc:.1%})")

    # Sample failed questions
    failed = [
        r
        for r in summary.results
        if r.status == "success" and not r.evaluation.get("match", True)
    ]
    if failed:
        print("\nSample failed questions (up to 5):")
        for r in failed[:5]:
            print(f"  [{r.scene_id}/Q{r.question_id}] {r.question[:50]}...")
            print(f"    Pred: {r.prediction[:60]}...")
            print(f"    GT:   {r.ground_truth.get('answer', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="space3D-Bench Evaluation")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data/benchmarks/Space3D-Bench"),
        help="Path to Space3D-Bench data",
    )
    parser.add_argument(
        "--scene",
        type=str,
        help="Single scene to evaluate (e.g., room_0)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all scenes",
    )
    parser.add_argument(
        "--num_questions",
        "-n",
        type=int,
        default=5,
        help="Max questions per scene (default: 5)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output JSON file for detailed results",
    )

    args = parser.parse_args()

    # Determine scenes to evaluate
    scenes = None
    if args.scene:
        scenes = [args.scene]
    elif not args.all:
        # Default: just room_0 for quick testing
        scenes = ["room_0"]

    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Scenes: {scenes or 'all'}")
    logger.info(f"Questions per scene: {args.num_questions}")

    # Run evaluation
    summary = run_evaluation(
        data_root=args.data_root,
        scenes=scenes,
        num_questions=args.num_questions,
    )

    # Print summary
    print_summary(summary)

    # Save detailed results
    if args.output:
        results_data = {
            "summary": {
                "total": summary.total,
                "passed": summary.passed,
                "failed": summary.failed,
                "errors": summary.errors,
                "subjective": summary.subjective,
                "accuracy": summary.accuracy,
            },
            "results": [
                {
                    "scene_id": r.scene_id,
                    "question_id": r.question_id,
                    "question": r.question,
                    "prediction": r.prediction,
                    "ground_truth": r.ground_truth,
                    "evaluation": r.evaluation,
                    "latency_ms": r.latency_ms,
                    "status": r.status,
                    "error": r.error,
                }
                for r in summary.results
            ],
        }

        with open(args.output, "w") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
