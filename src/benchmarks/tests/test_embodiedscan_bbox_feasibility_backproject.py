import numpy as np

from benchmarks.embodiedscan_bbox_feasibility.backproject import (
    proposal_from_depth_mask,
)


def test_proposal_from_depth_mask_backprojects_and_transforms() -> None:
    depth = np.array([[1.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    mask = np.array([[True, False], [False, True]])
    intrinsic = np.eye(3, dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [10, 20, 30]

    proposal = proposal_from_depth_mask(
        depth=depth,
        mask=mask,
        intrinsic=intrinsic,
        camera_to_world=pose,
        source="2d-backproject",
        min_points=1,
    )

    assert proposal is not None
    assert proposal.source == "2d-backproject"
    assert proposal.bbox_3d[:3] == [11.0, 21.0, 31.5]
    assert proposal.bbox_3d[3:6] == [2.0, 2.0, 1.0]
    assert proposal.metadata["num_points"] == 2


def test_proposal_from_depth_mask_returns_none_when_points_below_threshold() -> None:
    proposal = proposal_from_depth_mask(
        depth=np.array([[1.0, 1.0], [1.0, 2.0]], dtype=np.float32),
        mask=np.array([[True, False], [False, False]]),
        intrinsic=np.eye(3, dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float32),
        source="2d-backproject",
        min_points=2,
    )

    assert proposal is None


def test_proposal_from_depth_mask_keeps_score_and_metadata() -> None:
    proposal = proposal_from_depth_mask(
        depth=np.array([[1.0, 1.0], [1.0, 2.0]], dtype=np.float32),
        mask=np.ones((2, 2), dtype=bool),
        intrinsic=np.eye(3, dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float32),
        source="2d-backproject",
        score=0.7,
        min_points=1,
        metadata={"frame_id": 5},
    )

    assert proposal is not None
    assert proposal.score == 0.7
    assert proposal.metadata == {"frame_id": 5, "num_points": 4}
