import gzip
import pickle
from pathlib import Path

import numpy as np

from benchmarks.embodiedscan_bbox_feasibility.conceptgraph import (
    generate_conceptgraph_proposals,
)


def test_generate_conceptgraph_proposals_reads_pkl_without_keyframe_selector(tmp_path: Path) -> None:
    scene = tmp_path / "scene0001_00" / "conceptgraph"
    pcd_dir = scene / "pcd_saves"
    pcd_dir.mkdir(parents=True)
    pkl_path = pcd_dir / "full_pcd_mock_post.pkl.gz"
    payload = {
        "objects": [
            {
                "pcd_np": np.array([[0, 0, 0], [2, 4, 6]], dtype=np.float32),
                "class_name": ["chair"],
                "conf": [0.9],
            },
            {
                "pcd_np": np.array([[1, 1, 1]], dtype=np.float32),
                "class_name": ["floor"],
            },
        ]
    }
    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(payload, f)

    record = generate_conceptgraph_proposals(
        scene_path=scene,
        scan_id="scannet/scene0001_00",
        scene_id="scene0001_00",
    )

    assert record.method == "2d-cg"
    assert record.target_id is None
    assert len(record.proposals) == 1
    assert record.proposals[0].bbox_3d[:6] == [1.0, 2.0, 3.0, 2.0, 4.0, 6.0]
    assert record.proposals[0].metadata["category"] == "chair"


def test_generate_conceptgraph_proposals_skips_non_finite_points(tmp_path: Path) -> None:
    scene = tmp_path / "scene0001_00" / "conceptgraph"
    pcd_dir = scene / "pcd_saves"
    pcd_dir.mkdir(parents=True)
    pkl_path = pcd_dir / "full_pcd_mock_post.pkl.gz"
    payload = {
        "objects": [
            {
                "pcd_np": np.array([[0, 0, 0], [np.inf, 4, 6]], dtype=np.float32),
                "class_name": ["chair"],
            },
            {
                "pcd_np": np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float32),
                "class_name": ["table"],
            },
        ]
    }
    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(payload, f)

    record = generate_conceptgraph_proposals(
        scene_path=scene,
        scan_id="scannet/scene0001_00",
        scene_id="scene0001_00",
    )

    assert len(record.proposals) == 1
    assert record.proposals[0].metadata["category"] == "table"


def test_generate_conceptgraph_proposals_drops_non_finite_score(tmp_path: Path) -> None:
    scene = tmp_path / "scene0001_00" / "conceptgraph"
    pcd_dir = scene / "pcd_saves"
    pcd_dir.mkdir(parents=True)
    pkl_path = pcd_dir / "full_pcd_mock_post.pkl.gz"
    payload = {
        "objects": [
            {
                "pcd_np": np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float32),
                "class_name": ["table"],
                "conf": [np.nan],
            },
        ]
    }
    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(payload, f)

    record = generate_conceptgraph_proposals(
        scene_path=scene,
        scan_id="scannet/scene0001_00",
        scene_id="scene0001_00",
    )

    assert len(record.proposals) == 1
    assert record.proposals[0].score is None
