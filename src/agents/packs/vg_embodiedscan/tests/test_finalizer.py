"""VG FinalizerSpec validates payload + adapts to Stage2StructuredResponse fields."""
from __future__ import annotations

from pathlib import Path

import pytest

from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.packs.vg_embodiedscan.ctx import (
    Proposal,
    VgEmbodiedScanCtx,
)
from agents.packs.vg_embodiedscan.finalizer import (
    VG_FINALIZER,
    VgPayload,
)


def _runtime_with_ctx(tmp_path: Path) -> Stage2RuntimeState:
    rs = Stage2RuntimeState(bundle=Stage2EvidenceBundle())
    rs.task_ctx = VgEmbodiedScanCtx(
        proposal_pool_source="vdetr",
        proposals=[
            Proposal(id=0, bbox_3d_9dof=[0]*9, category="chair", score=0.9),
            Proposal(id=1, bbox_3d_9dof=[1]*9, category="desk", score=0.8),
        ],
        frame_index={10: [0, 1]},
        proposal_index={0: [10], 1: [10]},
        annotated_image_dir=tmp_path,
    )
    return rs


def test_validator_resolves_known_proposal(tmp_path: Path) -> None:
    rs = _runtime_with_ctx(tmp_path)
    payload = VgPayload(proposal_id=1, confidence=0.9)
    result = VG_FINALIZER.validator(payload, rs)
    assert result.proposal_id == 1


def test_validator_raises_unknown_proposal_id(tmp_path: Path) -> None:
    rs = _runtime_with_ctx(tmp_path)
    payload = VgPayload(proposal_id=99, confidence=0.5)
    with pytest.raises(ValueError, match="proposal_id 99 not in pool"):
        VG_FINALIZER.validator(payload, rs)


def test_validator_accepts_minus_one_as_failed_marker(tmp_path: Path) -> None:
    rs = _runtime_with_ctx(tmp_path)
    payload = VgPayload(proposal_id=-1, confidence=0.0)
    result = VG_FINALIZER.validator(payload, rs)
    assert result.proposal_id == -1


def test_adapter_emits_bbox_3d_for_known_proposal(tmp_path: Path) -> None:
    rs = _runtime_with_ctx(tmp_path)
    payload = VgPayload(proposal_id=1, confidence=0.9)
    validated = VG_FINALIZER.validator(payload, rs)
    out = VG_FINALIZER.adapter(validated, rs)
    assert out["selected_object_id"] == 1
    assert out["bbox_3d"] == [1.0]*9
    assert out["status"] == "completed"


def test_adapter_emits_failed_status_for_minus_one(tmp_path: Path) -> None:
    rs = _runtime_with_ctx(tmp_path)
    payload = VgPayload(proposal_id=-1, confidence=0.0)
    validated = VG_FINALIZER.validator(payload, rs)
    out = VG_FINALIZER.adapter(validated, rs)
    assert out["status"] == "failed"
    assert out["selected_object_id"] is None
