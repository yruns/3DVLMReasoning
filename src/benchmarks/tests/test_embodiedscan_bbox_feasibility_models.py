import pytest

from benchmarks.embodiedscan_bbox_feasibility.models import (
    AggregateMetrics,
    BBox3DProposal,
    EmbodiedScanTarget,
    FailureTag,
    ObservationRecord,
    ProposalRecord,
    TargetScore,
)


def test_bbox_proposal_normalizes_to_nine_floats() -> None:
    proposal = BBox3DProposal(
        bbox_3d=[1, 2, 3, 4, 5, 6],
        score=0.5,
        source="unit",
    )
    assert proposal.bbox_3d == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0]


def test_bbox_proposal_rejects_short_box() -> None:
    with pytest.raises(ValueError, match="at least 6"):
        BBox3DProposal(bbox_3d=[1, 2, 3, 4, 5], score=1.0, source="unit")


@pytest.mark.parametrize(
    "bbox_3d",
    [
        [1, 2, 3, 4, 5, float("nan")],
        [1, 2, 3, 4, 5, float("inf")],
    ],
)
def test_bbox_proposal_rejects_non_finite_box_values(bbox_3d: list[float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        BBox3DProposal(bbox_3d=bbox_3d, score=1.0, source="unit")


def test_bbox_proposal_rejects_long_box() -> None:
    with pytest.raises(ValueError, match="at most 9"):
        BBox3DProposal(bbox_3d=list(range(10)), score=1.0, source="unit")


def test_bbox_proposal_accepts_nine_value_box() -> None:
    proposal = BBox3DProposal(
        bbox_3d=[1, 2, 3, 4, 5, 6, 0.1, 0.2, 0.3],
        score=1.0,
        source="unit",
    )
    assert proposal.bbox_3d == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.2, 0.3]


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_bbox_proposal_rejects_non_finite_score(score: float) -> None:
    with pytest.raises(ValueError):
        BBox3DProposal(bbox_3d=[1, 2, 3, 4, 5, 6], score=score, source="unit")


def test_bbox_proposal_accepts_uncalibrated_finite_score() -> None:
    proposal = BBox3DProposal(bbox_3d=[1, 2, 3, 4, 5, 6], score=2.5, source="unit")
    assert proposal.score == 2.5


def test_bbox_proposal_metadata_accepts_json_safe_nested_values() -> None:
    proposal = BBox3DProposal(
        bbox_3d=[1, 2, 3, 4, 5, 6],
        source="unit",
        metadata={"nested": {"ok": [1, "x", True, None]}},
    )
    assert proposal.model_dump_json()


def test_observation_record_metadata_rejects_object_value() -> None:
    with pytest.raises(ValueError):
        ObservationRecord(policy="unit", metadata={"bad": object()})


def test_proposal_record_metadata_rejects_non_finite_float() -> None:
    with pytest.raises(ValueError):
        ProposalRecord(
            scene_id="scene0001_00",
            scan_id="scannet/scene0001_00",
            method="unit",
            input_condition="unit",
            metadata={"bad": float("nan")},
        )


def test_bbox_proposal_metadata_rejects_non_string_key() -> None:
    with pytest.raises(ValueError):
        BBox3DProposal(
            bbox_3d=[1, 2, 3, 4, 5, 6],
            source="unit",
            metadata={1: "bad-key"},
        )


@pytest.mark.parametrize("best_iou", [-0.1, 1.1])
def test_target_score_rejects_best_iou_outside_unit_interval(best_iou: float) -> None:
    with pytest.raises(ValueError):
        TargetScore(
            scan_id="scannet/scene0001_00",
            scene_id="scene0001_00",
            target_id=7,
            method="unit",
            input_condition="unit",
            best_iou=best_iou,
        )


def test_target_score_rejects_negative_best_proposal_index() -> None:
    with pytest.raises(ValueError):
        TargetScore(
            scan_id="scannet/scene0001_00",
            scene_id="scene0001_00",
            target_id=7,
            method="unit",
            input_condition="unit",
            best_iou=0.5,
            best_proposal_index=-1,
        )


def test_aggregate_metrics_rejects_accuracy_above_one() -> None:
    with pytest.raises(ValueError):
        AggregateMetrics(
            method="unit",
            input_condition="unit",
            num_targets=1,
            mean_best_iou=0.0,
            median_best_iou=0.0,
            acc_025=1.1,
            acc_050=0.0,
            mean_proposals_per_record=0.0,
            non_degenerate_box_ratio=0.0,
        )


def test_aggregate_metrics_rejects_negative_num_targets() -> None:
    with pytest.raises(ValueError):
        AggregateMetrics(
            method="unit",
            input_condition="unit",
            num_targets=-1,
            mean_best_iou=0.0,
            median_best_iou=0.0,
            acc_025=0.0,
            acc_050=0.0,
            mean_proposals_per_record=0.0,
            non_degenerate_box_ratio=0.0,
        )


def test_aggregate_metrics_rejects_negative_failure_count() -> None:
    with pytest.raises(ValueError):
        AggregateMetrics(
            method="unit",
            input_condition="unit",
            num_targets=1,
            mean_best_iou=0.0,
            median_best_iou=0.0,
            acc_025=0.0,
            acc_050=0.0,
            mean_proposals_per_record=0.0,
            non_degenerate_box_ratio=0.0,
            failure_counts={"no_proposal": -1},
        )


def test_proposal_record_keeps_target_conditioned_observation() -> None:
    target = EmbodiedScanTarget(
        sample_ids=["sample-a"],
        scan_id="scannet/scene0001_00",
        scene_id="scene0001_00",
        target_id=7,
        target_category="chair",
        gt_bbox_3d=[0, 0, 0, 1, 1, 1, 0, 0, 0],
    )
    obs = ObservationRecord(
        policy="target_best_visible_centered_window",
        frame_ids=[10, 12, 14],
    )
    record = ProposalRecord(
        scene_id=target.scene_id,
        scan_id=target.scan_id,
        target_id=target.target_id,
        method="3d-mv-recon-detector",
        input_condition="multi_frame_recon_3",
        observation=obs,
        proposals=[],
        failure_tag=FailureTag.NO_PROPOSAL,
    )
    assert record.target_id == 7
    assert record.observation.frame_ids == [10, 12, 14]
    assert record.failure_tag == FailureTag.NO_PROPOSAL
