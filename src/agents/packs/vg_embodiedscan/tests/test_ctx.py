"""VgEmbodiedScanCtx + proposal_pool adapter."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agents.core.task_types import Stage2EvidenceBundle
from agents.packs.vg_embodiedscan.ctx import (
    Proposal,
    VgEmbodiedScanCtx,
    build_ctx_from_bundle,
)


def test_proposal_round_trip() -> None:
    p = Proposal(
        id=3,
        bbox_3d_9dof=[0, 0, 0, 1, 1, 1, 0, 0, 0],
        category="chair",
        score=0.9,
    )
    assert p.id == 3 and p.category == "chair"


def test_build_ctx_from_bundle_minimal(tmp_path: Path) -> None:
    annotated = tmp_path / "ann"
    annotated.mkdir()
    bundle = Stage2EvidenceBundle(
        extra_metadata={
            "vg_proposal_pool": {
                "source": "vdetr",
                "proposals": [
                    {"id": 1, "bbox_3d_9dof": [0]*9, "category": "chair", "score": 0.5},
                    {"id": 2, "bbox_3d_9dof": [1]*9, "category": "desk", "score": 0.8},
                ],
                "frame_index": {10: [1, 2], 11: [2]},
                "proposal_index": {1: [10], 2: [10, 11]},
                "annotated_image_dir": str(annotated),
                "axis_align_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
            }
        }
    )
    ctx = build_ctx_from_bundle(bundle)
    assert isinstance(ctx, VgEmbodiedScanCtx)
    assert ctx.proposal_pool_source == "vdetr"
    assert {p.id for p in ctx.proposals} == {1, 2}
    assert ctx.frame_index[10] == [1, 2]
    assert ctx.proposal_index[2] == [10, 11]
    assert ctx.annotated_image_dir == annotated
    assert ctx.axis_align_matrix.shape == (4, 4)


def test_build_ctx_rejects_unknown_source(tmp_path: Path) -> None:
    bundle = Stage2EvidenceBundle(
        extra_metadata={
            "vg_proposal_pool": {
                "source": "foo",
                "proposals": [],
                "frame_index": {},
                "proposal_index": {},
                "annotated_image_dir": str(tmp_path),
            }
        }
    )
    with pytest.raises(ValueError, match="proposal_pool_source"):
        build_ctx_from_bundle(bundle)


def test_build_ctx_rejects_unreadable_annotated_dir(tmp_path: Path) -> None:
    bundle = Stage2EvidenceBundle(
        extra_metadata={
            "vg_proposal_pool": {
                "source": "vdetr",
                "proposals": [],
                "frame_index": {},
                "proposal_index": {},
                "annotated_image_dir": str(tmp_path / "nope"),
            }
        }
    )
    with pytest.raises(ValueError, match="annotated_image_dir"):
        build_ctx_from_bundle(bundle)
