#!/usr/bin/env python
"""End-to-end Stage 2 test with HTML trace visualization.

This script demonstrates the full pipeline with visual debugging:
1. Stage 1 text frame selector wiring
2. Stage 2 agentic reasoning
3. HTML trace report generation

Run: REPLICA_ROOT=/path/to/Replica .venv/bin/python -m agents.examples.e2e_with_trace

Output: Opens an HTML report showing the execution trace with embedded images.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:7} | {message}")

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agents import (
    Stage1BackendCallbacks,
    Stage2DeepAgentConfig,
    Stage2DeepResearchAgent,
    Stage2EvidenceBundle,
    Stage2TaskSpec,
    Stage2TaskType,
    save_trace_report,
)
from query_scene import KeyframeSelector as TextFrameSelector


def run_with_trace(
    scene_path: Path,
    stage1_query: str,
    task_query: str,
    k: int = 2,
    max_turns: int = 4,
    output_dir: Path | None = None,
) -> Path:
    """Run Stage 1 -> Stage 2 pipeline and generate HTML trace report.

    Returns:
        Path to the generated HTML report
    """

    output_dir = output_dir or Path("trace_reports")
    output_dir.mkdir(exist_ok=True)

    logger.info("=" * 70)
    logger.info("Stage 2 E2E with Trace Visualization")
    logger.info("=" * 70)

    del stage1_query, k

    # Stage 1: build the selector used by select_by_text.
    logger.info("[Stage 1] Loading text frame selector...")
    selector = TextFrameSelector.from_scene_path(
        str(scene_path),
        llm_model="gpt-5.2-2025-12-11",
        use_pool=False,
    )
    logger.info(f"  Loaded {len(selector.objects)} objects")

    # Build evidence bundle without first-person seed frames.
    bundle = Stage2EvidenceBundle(
        scene_id=scene_path.name,
        scene_summary=f"Replica scene {scene_path.name} with {len(selector.objects)} detected objects.",
    )

    # Stage 2: Create agent with selector tools and crop callback.
    logger.info("[Stage 2] Creating agent...")
    callbacks = Stage1BackendCallbacks(
        text_frame_selector=selector,
        scene_id=scene_path.name,
    )

    agent = Stage2DeepResearchAgent(
        config=Stage2DeepAgentConfig(include_thoughts=False),
        crop_callback=callbacks.crops,
        text_frame_selector=selector,
    )

    # Create task
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.QA,
        user_query=task_query,
        max_reasoning_turns=max_turns,
    )

    # Run agent
    logger.info("[Stage 2] Running agent...")
    result = agent.run(task, bundle)

    logger.info(f"  Status: {result.result.status.value}")
    logger.info(f"  Confidence: {result.result.confidence:.2f}")
    logger.info(f"  Summary: {result.result.summary[:100]}...")
    logger.info(f"  Tool calls: {len(result.tool_trace)}")

    # Generate trace report
    timestamp = int(__import__("time").time())
    safe_query = "".join(c if c.isalnum() else "_" for c in task_query[:30])
    report_path = output_dir / f"trace_{safe_query}_{timestamp}.html"

    logger.info(f"[Trace] Generating HTML report: {report_path}")
    save_trace_report(
        result=result,
        task=task,
        initial_bundle=bundle,
        output_path=report_path,
    )

    return report_path


def main():
    # Find Replica scene
    replica_root = os.environ.get("REPLICA_ROOT")
    if not replica_root:
        candidates = [
            Path.home() / "Datasets" / "Replica",
            Path.home() / "Replica",
            Path("/Users/bytedance/Replica"),
        ]
        for c in candidates:
            if c.exists():
                replica_root = str(c)
                break

    if not replica_root:
        logger.error("REPLICA_ROOT not set")
        sys.exit(1)

    scene_path = Path(replica_root) / "room0"
    if not scene_path.exists():
        logger.error(f"Scene not found: {scene_path}")
        sys.exit(1)

    # Test cases that might trigger tool usage
    test_cases = [
        # Simple case - probably won't use tools
        ("pillow on the sofa", "What color is the pillow and is it on the sofa?"),
        # Harder case - might need more views
        (
            "objects on the table",
            "List all objects visible on or near the table. For each, describe its position relative to other objects.",
        ),
    ]

    reports = []
    for stage1_query, task_query in test_cases[:1]:  # Run first case
        try:
            report = run_with_trace(
                scene_path=scene_path,
                stage1_query=stage1_query,
                task_query=task_query,
                k=2,  # Start with fewer keyframes
                max_turns=4,
            )
            reports.append(report)
            logger.success(f"Report generated: {report}")
        except Exception as e:
            logger.error(f"Failed: {e}")
            import traceback

            traceback.print_exc()

    # Open first report in browser
    if reports:
        logger.info(f"\nOpening report in browser: {reports[0]}")
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(reports[0])], check=False)
            elif sys.platform == "linux":
                subprocess.run(["xdg-open", str(reports[0])], check=False)
        except Exception:
            logger.info(f"Please open manually: file://{reports[0].absolute()}")


if __name__ == "__main__":
    main()
