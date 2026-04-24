import pytest

from benchmarks.embodiedscan_bbox_feasibility.models import (
    BBox3DProposal,
    EmbodiedScanTarget,
    FailureTag,
    ObservationRecord,
    ProposalRecord,
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
