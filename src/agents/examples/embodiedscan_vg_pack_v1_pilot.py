#!/usr/bin/env python
"""EmbodiedScan VG pilot using the pack-v1 backend.

This is the only supported EmbodiedScan VG pilot after Plan C. It builds a
proposal-pool bundle and terminates through the chassis submit_final tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import agents.packs  # noqa: F401, E402  (triggers VG_PACK registration)
from agents.core.agent_config import Stage2DeepAgentConfig, Stage2TaskType  # noqa: E402
from agents.core.task_types import (  # noqa: E402
    Stage2EvidenceBundle,
    Stage2TaskSpec,
)
from agents.packs.vg_embodiedscan.proposal_pool import (
    build_vg_proposal_pool,  # noqa: E402
)
from agents.stage2_deep_agent import Stage2DeepResearchAgent  # noqa: E402


def build_pack_v1_bundle(
    *,
    proposals_jsonl: Path,
    source: str,
    annotated_image_dir: Path,
    frame_visibility: dict[int, list[int]],
    scene_id: str,
    axis_align_matrix: np.ndarray | None = None,
    query: str | None = None,
    scene_catalog: dict | None = None,
    bev_image_path: str | None = None,
    camera_trajectory: dict | None = None,
) -> Stage2EvidenceBundle:
    pool = build_vg_proposal_pool(
        proposals_jsonl=proposals_jsonl,
        source=source,
        annotated_image_dir=annotated_image_dir,
        frame_visibility=frame_visibility,
        axis_align_matrix=axis_align_matrix,
    )
    extra: dict = {"vg_proposal_pool": pool}
    if scene_catalog is not None:
        extra["scene_catalog"] = scene_catalog
    if bev_image_path is not None:
        extra["bev_image_path"] = bev_image_path
    if camera_trajectory is not None:
        extra["camera_trajectory"] = camera_trajectory
    return Stage2EvidenceBundle(
        scene_id=scene_id,
        # Mirror the convention used by the OpenEQA / SQA3D / etc.
        # benchmark adapters at `agents/benchmark_adapters.py:305`. TADG
        # (and any future submit-time tool) reads `bundle.stage1_query`
        # to identify the user's spatial relations.
        stage1_query=str(query or ""),
        extra_metadata=extra,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scene-id", required=True)
    p.add_argument("--proposals-jsonl", type=Path, required=True)
    p.add_argument("--source", choices=["gt", "vdetr", "conceptgraph"], default="gt")
    p.add_argument("--annotated-image-dir", type=Path, required=True)
    p.add_argument("--visibility-json", type=Path, required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    visibility = {
        int(k): [int(x) for x in v]
        for k, v in json.loads(args.visibility_json.read_text()).items()
    }

    bundle = build_pack_v1_bundle(
        proposals_jsonl=args.proposals_jsonl,
        source=args.source,
        annotated_image_dir=args.annotated_image_dir,
        frame_visibility=visibility,
        scene_id=args.scene_id,
    )

    # This pilot does not build a per-scene text frame selector, so Stage-1
    # text retrieval cannot be served — disable explicitly to keep the agent's
    # tool surface consistent with what the runtime can fulfill.
    cfg = Stage2DeepAgentConfig(
        vg_backend="pack_v1",
        enable_stage1_text_retrieval=False,
    )
    agent = Stage2DeepResearchAgent(config=cfg)
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        user_query=args.query,
    )
    result = agent.run(task=task, bundle=bundle)
    args.output.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("pack-v1 pilot done; result -> {}", args.output)


if __name__ == "__main__":
    main()
