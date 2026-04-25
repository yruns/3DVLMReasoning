"""Typed runtime ctx for the EmbodiedScan VG pack."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from agents.core.task_types import Stage2EvidenceBundle


@dataclass(frozen=True)
class Proposal:
    id: int
    bbox_3d_9dof: list[float]
    category: str
    score: float


@dataclass
class VgEmbodiedScanCtx:
    proposal_pool_source: Literal["vdetr", "conceptgraph"]
    proposals: list[Proposal]
    frame_index: dict[int, list[int]]      # frame_id -> [proposal_id]
    proposal_index: dict[int, list[int]]   # proposal_id -> [frame_id]
    annotated_image_dir: Path
    axis_align_matrix: np.ndarray | None = None


def build_ctx_from_bundle(bundle: Stage2EvidenceBundle) -> VgEmbodiedScanCtx:
    extra = bundle.extra_metadata or {}
    pool = extra.get("vg_proposal_pool")
    if pool is None:
        raise ValueError("bundle.extra_metadata.vg_proposal_pool is missing")

    source = pool.get("source")
    if source not in ("vdetr", "conceptgraph"):
        raise ValueError(
            f"proposal_pool_source must be 'vdetr' or 'conceptgraph', got {source!r}"
        )

    if "proposals" not in pool:
        raise ValueError("vg_proposal_pool.proposals key is required")
    raw_proposals = pool["proposals"]
    if not isinstance(raw_proposals, list):
        raise ValueError(
            "vg_proposal_pool.proposals must be a list; got "
            f"{type(raw_proposals).__name__}"
        )

    proposals: list[Proposal] = []
    for i, p in enumerate(raw_proposals):
        for required_key in ("id", "bbox_3d_9dof", "category", "score"):
            if required_key not in p:
                raise ValueError(
                    f"vg_proposal_pool.proposals[{i}].{required_key} is required"
                )
        bbox = p["bbox_3d_9dof"]
        if not isinstance(bbox, list) or len(bbox) != 9:
            raise ValueError(
                f"vg_proposal_pool.proposals[{i}].bbox_3d_9dof must be a 9-element list"
            )
        proposals.append(
            Proposal(
                id=int(p["id"]),
                bbox_3d_9dof=[float(x) for x in bbox],
                category=str(p["category"]),
                score=float(p["score"]),
            )
        )

    if "frame_index" not in pool:
        raise ValueError("vg_proposal_pool.frame_index key is required")
    if "proposal_index" not in pool:
        raise ValueError("vg_proposal_pool.proposal_index key is required")
    frame_index = {
        int(k): [int(x) for x in v] for k, v in pool["frame_index"].items()
    }
    proposal_index = {
        int(k): [int(x) for x in v] for k, v in pool["proposal_index"].items()
    }

    annotated_dir = Path(pool.get("annotated_image_dir") or "")
    if not annotated_dir.exists() or not annotated_dir.is_dir():
        raise ValueError(
            f"annotated_image_dir must exist and be a directory: {annotated_dir}"
        )

    axis_align_matrix: np.ndarray | None = None
    matrix = pool.get("axis_align_matrix")
    if matrix is not None:
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.shape != (4, 4):
            raise ValueError(
                f"axis_align_matrix must be 4x4, got shape {arr.shape}"
            )
        axis_align_matrix = arr

    return VgEmbodiedScanCtx(
        proposal_pool_source=source,
        proposals=proposals,
        frame_index=frame_index,
        proposal_index=proposal_index,
        annotated_image_dir=annotated_dir,
        axis_align_matrix=axis_align_matrix,
    )


__all__ = ["Proposal", "VgEmbodiedScanCtx", "build_ctx_from_bundle"]
