#!/usr/bin/env python
"""EmbodiedScan 3D Visual Grounding evaluation pilot.

Runs the full Stage1 → Stage2 pipeline on EmbodiedScan VG samples:
1. Load VG samples via EmbodiedScanVGAdapter
2. For each sample, run Stage 1 keyframe selection
3. Build evidence bundle with VG candidate objects
4. Run Stage 2 VLM agent for object grounding
5. Evaluate with oriented 3D IoU (Acc@0.25, Acc@0.50)

Usage::

    # Single-scene quick test (5 samples)
    python -m agents.examples.embodiedscan_vg_pilot --max-samples 5

    # Full val evaluation on ScanNet subset
    python -m agents.examples.embodiedscan_vg_pilot --source-filter scannet

    # Use mini VG set
    python -m agents.examples.embodiedscan_vg_pilot --mini
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import threading  # noqa: E402

from agents.adapters.embodiedscan_adapter import (  # noqa: E402
    EmbodiedScanVGAdapter,
)
from agents.core.agent_config import Stage2DeepAgentConfig  # noqa: E402
from agents.core.task_types import Stage2EvidenceBundle  # noqa: E402
from agents.stage1_adapters import (  # noqa: E402
    build_object_context,
    build_stage2_evidence_bundle,
)
from agents.stage2_deep_agent import Stage2DeepResearchAgent  # noqa: E402
from benchmarks.embodiedscan_loader import EmbodiedScanVGSample  # noqa: E402
from query_scene.keyframe_selector import KeyframeSelector  # noqa: E402

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "embodiedscan"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tmp" / "embodiedscan_vg_runs"

# Per-scene lock to prevent parallel workers from racing on scene data
_scene_locks: dict[str, threading.Lock] = {}
_scene_locks_guard = threading.Lock()


def _get_scene_lock(scene_id: str) -> threading.Lock:
    with _scene_locks_guard:
        if scene_id not in _scene_locks:
            _scene_locks[scene_id] = threading.Lock()
        return _scene_locks[scene_id]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EmbodiedScan VG evaluation pilot"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Path to data/embodiedscan/ directory",
    )
    parser.add_argument(
        "--scene-data-root",
        type=Path,
        default=None,
        help="Path to scene ConceptGraph data (defaults to data-root)",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--source-filter",
        default="scannet",
        choices=["scannet", "3rscan", "matterport3d"],
        help="Filter to specific source dataset",
    )
    parser.add_argument("--no-source-filter", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--k", type=int, default=5, help="Top-k keyframes")
    parser.add_argument(
        "--llm-model",
        default="gemini-2.5-pro",
        help="LLM model for Stage 1 query parsing",
    )
    parser.add_argument(
        "--stage2-model",
        default=None,
        help="Override Stage 2 VLM model (default: config default)",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load samples and show stats without running Stage 2",
    )
    parser.add_argument(
        "--diverse",
        action="store_true",
        help="Sample diversely across scenes (N per scene)",
    )
    parser.add_argument(
        "--per-scene",
        type=int,
        default=3,
        help="Samples per scene in diverse mode",
    )
    return parser.parse_args()


def run_one_sample(
    sample: EmbodiedScanVGSample,
    adapter: EmbodiedScanVGAdapter,
    config: Stage2DeepAgentConfig,
    k: int = 5,
    llm_model: str = "gemini-2.5-pro",
) -> dict[str, Any]:
    """Run full pipeline on a single VG sample.

    Returns dict with sample, prediction, and optional error.
    """
    scene_path = adapter.get_scene_path(sample)
    cg_path = scene_path / "conceptgraph"

    if not cg_path.exists():
        logger.warning("ConceptGraph not found: {}", cg_path)
        return {
            "sample": sample,
            "prediction": {
                "sample_id": sample.sample_id,
                "bbox_3d": None,
            },
            "error": f"Missing conceptgraph: {cg_path}",
        }

    try:
        # Acquire per-scene lock to prevent parallel workers
        # from racing on the same scene's cached data
        with _get_scene_lock(sample.scene_id):
            selector = KeyframeSelector.from_scene_path(
                str(cg_path), llm_model=llm_model, stride=1
            )
        stage1_result = selector.select_keyframes_v2(
            sample.query, k=k, use_visual_context=False
        )

        # Build evidence bundle
        object_context = build_object_context(selector.objects)
        bundle = build_stage2_evidence_bundle(
            stage1_result,
            scene_id=sample.scene_id,
            object_context=object_context,
        )

        # Inject VG candidates from scene graph objects
        # Apply axis alignment to transform CG coords → EmbodiedScan coords
        align_mat = adapter.get_axis_align_matrix(sample.scan_id)
        vg_candidates = adapter.build_vg_candidates(
            selector.objects, axis_align_matrix=align_mat
        )
        bundle = bundle.model_copy(
            update={
                "extra_metadata": {
                    **bundle.extra_metadata,
                    "vg_candidates": vg_candidates,
                }
            }
        )

        # Stage 2: VLM grounding
        task = adapter.build_task_spec(sample)
        agent = Stage2DeepResearchAgent(config=config)
        result = agent.run(task, bundle)

        prediction = adapter.extract_prediction(sample, result)
        return {
            "sample": sample,
            "prediction": prediction,
            "result": result,
        }

    except Exception as exc:
        err_str = str(exc)
        # Retry on 429 rate limit with backoff
        if "429" in err_str:
            for retry in range(1, 4):
                wait = 30 * retry
                logger.warning(
                    "429 rate limit on {} — retry {}/3 after {}s",
                    sample.sample_id, retry, wait,
                )
                time.sleep(wait)
                try:
                    agent = Stage2DeepResearchAgent(config=config)
                    result = agent.run(task, bundle)
                    prediction = adapter.extract_prediction(sample, result)
                    return {
                        "sample": sample,
                        "prediction": prediction,
                        "result": result,
                    }
                except Exception as retry_exc:
                    if "429" not in str(retry_exc):
                        err_str = str(retry_exc)
                        break

        logger.error("Failed on {}: {}", sample.sample_id, err_str)
        return {
            "sample": sample,
            "prediction": {
                "sample_id": sample.sample_id,
                "bbox_3d": None,
            },
            "error": err_str,
        }


def main() -> None:
    args = parse_args()
    source_filter = None if args.no_source_filter else args.source_filter

    # 1. Load benchmark
    adapter = EmbodiedScanVGAdapter(
        data_root=args.data_root,
        scene_data_root=args.scene_data_root,
    )
    # Load all samples (no max) if diverse, then subsample later
    load_max = None if args.diverse else args.max_samples
    samples = adapter.load_samples(
        split=args.split,
        source_filter=source_filter,
        max_samples=load_max,
        mini=args.mini,
    )

    # Diverse sampling: N samples per scene, only from scenes with CG data
    if args.diverse:
        import os
        from collections import defaultdict

        scene_data_root = args.scene_data_root or args.data_root
        by_scene: dict[str, list] = defaultdict(list)
        for s in samples:
            scene_name = s.scan_id.split("/")[-1] if hasattr(s, "scan_id") else s.scene_id
            cg_path = scene_data_root / scene_name / "conceptgraph"
            if cg_path.is_dir():
                by_scene[scene_name].append(s)

        diverse_samples = []
        for scene_name in sorted(by_scene.keys()):
            diverse_samples.extend(by_scene[scene_name][: args.per_scene])

        if args.max_samples:
            diverse_samples = diverse_samples[: args.max_samples]
        samples = diverse_samples
        logger.info(
            "Diverse sampling: {} samples from {} scenes",
            len(samples), len(by_scene),
        )

    logger.info("Loaded {} VG samples", len(samples))

    if args.dry_run:
        cats = adapter.dataset.get_target_categories()
        top10 = list(cats.items())[:10]
        logger.info("Top 10 target categories: {}", top10)
        logger.info("Scenes: {}", len(adapter.dataset.get_scenes()))
        return

    # 2. Prepare output
    run_id = f"vg_{args.split}_{int(time.time())}"
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output: {}", output_dir)

    config_kwargs = {"confidence_threshold": 0.15}
    if args.stage2_model:
        config_kwargs["model_name"] = args.stage2_model
    config = Stage2DeepAgentConfig(**config_kwargs)

    # 3. Run pipeline
    all_results: list[dict[str, Any]] = []
    vg_samples = [s for s in samples if isinstance(s, EmbodiedScanVGSample)]

    if args.workers <= 1:
        for i, sample in enumerate(vg_samples):
            logger.info(
                "[{}/{}] {} — {}",
                i + 1,
                len(vg_samples),
                sample.sample_id,
                sample.query[:60],
            )
            result = run_one_sample(sample, adapter, config, args.k, args.llm_model)
            all_results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    run_one_sample, s, adapter, config, args.k, args.llm_model
                ): s
                for s in vg_samples
            }
            for future in as_completed(futures):
                all_results.append(future.result())

    # 4. Evaluate
    predictions = [r["prediction"] for r in all_results]
    metrics = adapter.evaluate(predictions, vg_samples)

    logger.info("=== Results ===")
    logger.info("Acc@0.25: {:.3f}", metrics["acc_025"])
    logger.info("Acc@0.50: {:.3f}", metrics["acc_050"])
    logger.info("Mean IoU: {:.3f}", metrics["mean_iou"])
    logger.info("Samples:  {}", metrics["num_samples"])

    # 5. Save results
    output = {
        "metrics": metrics,
        "predictions": predictions,
        "args": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        },
    }
    output_path = output_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved results to {}", output_path)


if __name__ == "__main__":
    main()
