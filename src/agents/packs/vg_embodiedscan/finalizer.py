"""VG FinalizerSpec: validate proposal_id, resolve to bbox_3d."""
from __future__ import annotations

from pydantic import BaseModel, Field

from agents.skills.finalizer import FinalizerSpec
from agents.packs.vg_embodiedscan.ctx import VgEmbodiedScanCtx


class VgPayload(BaseModel):
    """submit_final payload schema for VG."""

    proposal_id: int = Field(
        description="Selected proposal id from the pool. Use -1 to mark this sample as failed (GT not in pool)."
    )
    confidence: float = Field(ge=0.0, le=1.0)


def vg_validator(payload: VgPayload, runtime) -> VgPayload:
    if payload.proposal_id == -1:
        return payload
    ctx: VgEmbodiedScanCtx = runtime.task_ctx
    ids = {p.id for p in ctx.proposals}
    if payload.proposal_id not in ids:
        raise ValueError(
            f"proposal_id {payload.proposal_id} not in pool (have {sorted(ids)})"
        )
    return payload


def vg_adapter(payload: VgPayload, runtime) -> dict:
    """Convert validated payload to fields suitable for Stage2StructuredResponse."""
    if payload.proposal_id == -1:
        return {
            "status": "failed",
            "selected_object_id": None,
            "bbox_3d": None,
            "confidence": payload.confidence,
            "rationale_marker": "GT not in proposal pool",
        }
    ctx: VgEmbodiedScanCtx = runtime.task_ctx
    proposal = next(p for p in ctx.proposals if p.id == payload.proposal_id)
    return {
        "status": "completed",
        "selected_object_id": proposal.id,
        "bbox_3d": list(proposal.bbox_3d_9dof),
        "category": proposal.category,
        "confidence": payload.confidence,
    }


VG_FINALIZER = FinalizerSpec(
    payload_model=VgPayload,
    validator=vg_validator,
    adapter=vg_adapter,
)


__all__ = ["VgPayload", "VG_FINALIZER", "vg_validator", "vg_adapter"]
