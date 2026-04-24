from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import BBox3DProposal, FailureTag, ProposalRecord


def load_detector_proposals_json(
    *,
    path: str | Path,
    scene_id: str,
    scan_id: str,
    method: str,
    input_condition: str,
    target_id: int | None = None,
) -> ProposalRecord:
    pred_path = Path(path)
    data = json.loads(pred_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "proposals" not in data:
        raise ValueError(f"Detector proposal JSON must contain a proposals list: {pred_path}")
    raw_proposals = data["proposals"]
    if not isinstance(raw_proposals, list):
        raise ValueError(f"Detector proposal JSON proposals must be a list: {pred_path}")

    proposals: list[BBox3DProposal] = []
    for idx, raw in enumerate(raw_proposals):
        if not isinstance(raw, dict):
            raise ValueError(f"Detector proposal at index {idx} must be an object: {pred_path}")
        if "bbox_3d" not in raw:
            raise ValueError(f"Detector proposal at index {idx} missing bbox_3d: {pred_path}")

        metadata: dict[str, Any] = dict(raw.get("metadata", {}))
        if "label" in raw:
            metadata["label"] = raw["label"]
        metadata["path"] = str(pred_path)
        proposals.append(
            BBox3DProposal(
                bbox_3d=raw["bbox_3d"],
                score=raw.get("score"),
                source="detector",
                metadata=metadata,
            )
        )

    return ProposalRecord(
        scene_id=scene_id,
        scan_id=scan_id,
        target_id=target_id,
        method=method,
        input_condition=input_condition,
        proposals=proposals,
        failure_tag=None if proposals else FailureTag.NO_PROPOSAL,
        metadata={"path": str(pred_path)},
    )


def model_blocked_record(
    *,
    scene_id: str,
    scan_id: str,
    method: str,
    input_condition: str,
    reason: str,
    target_id: int | None = None,
) -> ProposalRecord:
    return ProposalRecord(
        scene_id=scene_id,
        scan_id=scan_id,
        target_id=target_id,
        method=method,
        input_condition=input_condition,
        proposals=[],
        failure_tag=FailureTag.MODEL_BLOCKED,
        metadata={"reason": reason},
    )
