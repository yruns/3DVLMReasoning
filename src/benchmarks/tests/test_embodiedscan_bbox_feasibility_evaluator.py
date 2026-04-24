import pytest

from benchmarks.embodiedscan_bbox_feasibility.evaluator import (
    evaluate_records,
)
from benchmarks.embodiedscan_bbox_feasibility.models import (
    BBox3DProposal,
    EmbodiedScanTarget,
    FailureTag,
    ProposalRecord,
)


def _target(target_id: int, scan_id: str = "scannet/scene0001_00") -> EmbodiedScanTarget:
    return EmbodiedScanTarget(
        sample_ids=[f"sample-{target_id}"],
        scan_id=scan_id,
        scene_id=scan_id.split("/")[-1],
        target_id=target_id,
        target_category="chair",
        gt_bbox_3d=[0, 0, 0, 1, 1, 1, 0, 0, 0],
    )


def test_evaluate_records_uses_best_iou_per_target() -> None:
    target = _target(1)
    record = ProposalRecord(
        scene_id="scene0001_00",
        scan_id="scannet/scene0001_00",
        target_id=None,
        method="unit",
        input_condition="scene",
        proposals=[
            BBox3DProposal(bbox_3d=[5, 5, 5, 1, 1, 1], score=0.1, source="unit"),
            BBox3DProposal(bbox_3d=[0, 0, 0, 1, 1, 1], score=0.9, source="unit"),
        ],
    )

    result = evaluate_records([target], [record])

    assert result.metrics.mean_best_iou == pytest.approx(1.0)
    assert result.metrics.acc_025 == pytest.approx(1.0)
    assert result.metrics.acc_050 == pytest.approx(1.0)
    assert result.metrics.non_degenerate_box_ratio == pytest.approx(1.0)
    assert result.scores[0].best_proposal_index == 1


def test_evaluate_records_prefers_target_specific_record() -> None:
    target = _target(1)
    scene_record = ProposalRecord(
        scene_id="scene0001_00",
        scan_id="scannet/scene0001_00",
        target_id=None,
        method="unit",
        input_condition="scene",
        proposals=[BBox3DProposal(bbox_3d=[5, 5, 5, 1, 1, 1], source="unit")],
    )
    target_record = ProposalRecord(
        scene_id="scene0001_00",
        scan_id="scannet/scene0001_00",
        target_id=1,
        method="unit",
        input_condition="target",
        proposals=[BBox3DProposal(bbox_3d=[0, 0, 0, 1, 1, 1], source="unit")],
    )

    result = evaluate_records([target], [scene_record, target_record])

    assert result.scores[0].input_condition == "target"
    assert result.scores[0].best_iou == pytest.approx(1.0)


def test_evaluate_records_counts_missing_and_record_failures() -> None:
    targets = [_target(1), _target(2, scan_id="scannet/scene0002_00")]
    record = ProposalRecord(
        scene_id="scene0001_00",
        scan_id="scannet/scene0001_00",
        target_id=None,
        method="unit",
        input_condition="scene",
        proposals=[],
        failure_tag=FailureTag.NO_PROPOSAL,
    )

    result = evaluate_records(targets, [record])

    assert [score.best_iou for score in result.scores] == [0.0, 0.0]
    assert result.metrics.failure_counts == {"no_proposal": 2}
    assert result.metrics.mean_proposals_per_record == pytest.approx(0.0)


def test_evaluate_records_requires_non_empty_records() -> None:
    with pytest.raises(ValueError, match="records must not be empty"):
        evaluate_records([_target(1)], [])
