#!/usr/bin/env python
"""EmbodiedScan VG pilot using pack-v1 backend.

Differences from legacy embodiedscan_vg_pilot.py:
- Stage2DeepAgentConfig(vg_backend="pack_v1")
- bundle.extra_metadata.vg_proposal_pool populated by build_vg_proposal_pool
- agent terminates via chassis submit_final, not select_object side effect
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import agents.packs  # noqa: F401, E402  (triggers VG_PACK registration)
from agents.core.agent_config import Stage2DeepAgentConfig, Stage2TaskType  # noqa: E402
from agents.core.task_types import (  # noqa: E402
    KeyframeEvidence,
    Stage2EvidenceBundle,
    Stage2TaskSpec,
)
from agents.packs.vg_embodiedscan.proposal_pool import build_vg_proposal_pool  # noqa: E402
from agents.stage2_deep_agent import Stage2DeepResearchAgent  # noqa: E402


def build_pack_v1_bundle(
    *,
    proposals_jsonl: Path,
    source: str,
    annotated_image_dir: Path,
    frame_visibility: dict[int, list[int]],
    keyframes: Sequence[tuple[int, str, int]],
    scene_id: str,
    axis_align_matrix: np.ndarray | None = None,
) -> Stage2EvidenceBundle:
    pool = build_vg_proposal_pool(
        proposals_jsonl=proposals_jsonl,
        source=source,
        annotated_image_dir=annotated_image_dir,
        frame_visibility=frame_visibility,
        axis_align_matrix=axis_align_matrix,
    )
    return Stage2EvidenceBundle(
        scene_id=scene_id,
        keyframes=[
            KeyframeEvidence(keyframe_idx=idx, image_path=path, frame_id=fid)
            for idx, path, fid in keyframes
        ],
        extra_metadata={"vg_proposal_pool": pool},
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scene-id", required=True)
    p.add_argument("--proposals-jsonl", type=Path, required=True)
    p.add_argument("--source", choices=["vdetr", "conceptgraph"], required=True)
    p.add_argument("--annotated-image-dir", type=Path, required=True)
    p.add_argument("--visibility-json", type=Path, required=True)
    p.add_argument("--keyframes-json", type=Path, required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    visibility = {int(k): [int(x) for x in v]
                  for k, v in json.loads(args.visibility_json.read_text()).items()}
    keyframes = json.loads(args.keyframes_json.read_text())  # [[idx, path, fid], ...]

    bundle = build_pack_v1_bundle(
        proposals_jsonl=args.proposals_jsonl,
        source=args.source,
        annotated_image_dir=args.annotated_image_dir,
        frame_visibility=visibility,
        keyframes=keyframes,
        scene_id=args.scene_id,
    )

    cfg = Stage2DeepAgentConfig(vg_backend="pack_v1")
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
