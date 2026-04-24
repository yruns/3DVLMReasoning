from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from benchmarks.embodiedscan_eval import compute_oriented_iou_3d

from .geometry import is_non_degenerate_bbox
from .models import AggregateMetrics, EmbodiedScanTarget, ProposalRecord, TargetScore


@dataclass(frozen=True)
class EvaluationResult:
    scores: list[TargetScore]
    metrics: AggregateMetrics


def evaluate_records(
    targets: list[EmbodiedScanTarget],
    records: list[ProposalRecord],
) -> EvaluationResult:
    if not records:
        raise ValueError("records must not be empty")

    method = records[0].method
    input_condition = records[0].input_condition
    scores: list[TargetScore] = []
    proposal_counts: list[int] = []
    valid_box_count = 0
    total_box_count = 0
    failure_counts: Counter[str] = Counter()

    for target in targets:
        record = _find_record(target, records)
        if record is None:
            scores.append(_score_missing(target, method, input_condition))
            failure_counts["no_proposal"] += 1
            continue

        if record.failure_tag is not None:
            failure_counts[record.failure_tag.value] += 1
        proposal_counts.append(len(record.proposals))

        best_iou = 0.0
        best_idx: int | None = None
        for idx, proposal in enumerate(record.proposals):
            total_box_count += 1
            if is_non_degenerate_bbox(proposal.bbox_3d):
                valid_box_count += 1
            iou = compute_oriented_iou_3d(proposal.bbox_3d, target.gt_bbox_3d)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        scores.append(
            TargetScore(
                scan_id=target.scan_id,
                scene_id=target.scene_id,
                target_id=target.target_id,
                method=record.method,
                input_condition=record.input_condition,
                best_iou=float(best_iou),
                best_proposal_index=best_idx,
                failure_tag=record.failure_tag,
            )
        )

    ious = np.asarray([score.best_iou for score in scores], dtype=np.float64)
    metrics = AggregateMetrics(
        method=method,
        input_condition=input_condition,
        num_targets=len(scores),
        mean_best_iou=float(ious.mean()) if len(ious) else 0.0,
        median_best_iou=float(np.median(ious)) if len(ious) else 0.0,
        acc_025=float((ious >= 0.25).mean()) if len(ious) else 0.0,
        acc_050=float((ious >= 0.50).mean()) if len(ious) else 0.0,
        mean_proposals_per_record=(
            float(np.mean(proposal_counts)) if proposal_counts else 0.0
        ),
        non_degenerate_box_ratio=(
            float(valid_box_count / total_box_count) if total_box_count else 0.0
        ),
        failure_counts=dict(failure_counts),
    )
    return EvaluationResult(scores=scores, metrics=metrics)


def _find_record(
    target: EmbodiedScanTarget,
    records: list[ProposalRecord],
) -> ProposalRecord | None:
    for record in records:
        if record.scan_id == target.scan_id and record.target_id == target.target_id:
            return record
    for record in records:
        if record.scan_id == target.scan_id and record.target_id is None:
            return record
    return None


def _score_missing(
    target: EmbodiedScanTarget,
    method: str,
    input_condition: str,
) -> TargetScore:
    return TargetScore(
        scan_id=target.scan_id,
        scene_id=target.scene_id,
        target_id=target.target_id,
        method=method,
        input_condition=input_condition,
        best_iou=0.0,
        best_proposal_index=None,
    )
