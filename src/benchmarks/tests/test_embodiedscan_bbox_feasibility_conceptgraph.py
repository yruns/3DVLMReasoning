import gzip
import pickle
from pathlib import Path

import numpy as np
import pytest

from benchmarks.embodiedscan_bbox_feasibility.conceptgraph import (
    generate_conceptgraph_proposals,
)
from benchmarks.embodiedscan_bbox_feasibility.models import FailureTag


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
        scene,
        "scannet/scene0001_00",
        "scene0001_00",
    )

    assert record.method == "2d-cg"
    assert record.target_id is None
    assert len(record.proposals) == 1
    assert record.proposals[0].bbox_3d[:6] == [1.0, 2.0, 3.0, 2.0, 4.0, 6.0]
    assert record.proposals[0].metadata["category"] == "chair"


def test_generate_conceptgraph_proposals_applies_axis_align_matrix(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "scene0001_00" / "conceptgraph"
    pcd_dir = scene / "pcd_saves"
    pcd_dir.mkdir(parents=True)
    pkl_path = pcd_dir / "full_pcd_mock_post.pkl.gz"
    payload = {
        "objects": [
            {
                "pcd_np": np.array([[0, 0, 0], [2, 4, 6]], dtype=np.float32),
                "class_name": ["chair"],
            },
        ]
    }
    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(payload, f)

    record = generate_conceptgraph_proposals(
        scene,
        "scannet/scene0001_00",
        "scene0001_00",
        axis_align_matrix=[
            [1, 0, 0, 10],
            [0, 1, 0, 20],
            [0, 0, 1, 30],
            [0, 0, 0, 1],
        ],
    )

    assert record.proposals[0].bbox_3d[:6] == [
        11.0,
        22.0,
        33.0,
        2.0,
        4.0,
        6.0,
    ]
    assert record.metadata["axis_align_applied"] is True


def test_generate_conceptgraph_proposals_uses_bbox_np_when_available(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "scene0001_00" / "conceptgraph"
    pcd_dir = scene / "pcd_saves"
    pcd_dir.mkdir(parents=True)
    pkl_path = pcd_dir / "full_pcd_mock_post.pkl.gz"
    payload = {
        "objects": [
            {
                "pcd_np": np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
                "bbox_np": np.array([[0, 0, 0], [2, 4, 6]], dtype=np.float32),
                "class_name": ["chair"],
            },
        ]
    }
    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(payload, f)

    record = generate_conceptgraph_proposals(
        scene,
        "scannet/scene0001_00",
        "scene0001_00",
    )

    variants = {proposal.metadata["geometry_variant"]: proposal for proposal in record.proposals}
    assert variants["pcd_aabb"].bbox_3d[:6] == [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
    assert variants["bbox_np_aabb"].bbox_3d[:6] == [1.0, 2.0, 3.0, 2.0, 4.0, 6.0]

def test_generate_conceptgraph_proposals_reports_no_pcd_file(tmp_path: Path) -> None:
    scene = tmp_path / "scene0001_00" / "conceptgraph"
    pcd_dir = scene / "pcd_saves"
    pcd_dir.mkdir(parents=True)
    (pcd_dir / "notes.txt").write_text("not a pkl", encoding="utf-8")

    record = generate_conceptgraph_proposals(
        scene_path=scene,
        scan_id="scannet/scene0001_00",
        scene_id="scene0001_00",
    )

    assert record.method == "2d-cg"
    assert record.input_condition == "conceptgraph_scene"
    assert record.proposals == []
    assert record.failure_tag == FailureTag.NO_PROPOSAL
    assert record.metadata["reason"] == "no_pcd_file"
    assert record.metadata["scene_path"] == str(scene)


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


def test_generate_conceptgraph_proposals_skips_invalid_objects(tmp_path: Path) -> None:
    scene = tmp_path / "scene0001_00" / "conceptgraph"
    pcd_dir = scene / "pcd_saves"
    pcd_dir.mkdir(parents=True)
    pkl_path = pcd_dir / "full_pcd_mock_post.pkl.gz"
    payload = {
        "objects": [
            "not an object dict",
            {
                "pcd_np": None,
                "class_name": ["chair"],
            },
            {
                "pcd_np": np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
                "class_name": ["lamp"],
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
    assert record.proposals[0].metadata["category"] == "lamp"


def test_generate_conceptgraph_proposals_strips_background_labels(tmp_path: Path) -> None:
    scene = tmp_path / "scene0001_00" / "conceptgraph"
    pcd_dir = scene / "pcd_saves"
    pcd_dir.mkdir(parents=True)
    pkl_path = pcd_dir / "full_pcd_mock_post.pkl.gz"
    payload = {
        "objects": [
            {
                "pcd_np": np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float32),
                "class_name": [" floor "],
            },
            {
                "pcd_np": np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
                "class_name": ["item", " chair "],
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
    assert record.proposals[0].metadata["category"] == "chair"


def test_generate_conceptgraph_proposals_reads_plain_pkl_gz_fallback(tmp_path: Path) -> None:
    scene = tmp_path / "scene0001_00" / "conceptgraph"
    pcd_dir = scene / "pcd_saves"
    pcd_dir.mkdir(parents=True)
    pkl_path = pcd_dir / "object_map.pkl.gz"
    payload = {
        "objects": [
            {
                "pcd_np": np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32),
                "class_name": ["chair"],
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
    assert record.proposals[0].metadata["category"] == "chair"
    pkl_path_metadata = record.proposals[0].metadata["pkl_path"]
    assert pkl_path_metadata.endswith("object_map.pkl.gz")


def test_generate_conceptgraph_proposals_prefers_ram_post_pkl(tmp_path: Path) -> None:
    scene = tmp_path / "scene0001_00" / "conceptgraph"
    pcd_dir = scene / "pcd_saves"
    pcd_dir.mkdir(parents=True)

    files = [
        ("generic_objects.pkl.gz", "generic"),
        ("full_pcd_mock_post.pkl.gz", "post"),
        ("full_pcd_ram_mock_post.pkl.gz", "ram_post"),
    ]
    for filename, category in files:
        payload = {
            "objects": [
                {
                    "pcd_np": np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32),
                    "class_name": [category],
                },
            ]
        }
        with gzip.open(pcd_dir / filename, "wb") as f:
            pickle.dump(payload, f)

    record = generate_conceptgraph_proposals(
        scene_path=scene,
        scan_id="scannet/scene0001_00",
        scene_id="scene0001_00",
    )

    assert len(record.proposals) == 1
    assert record.proposals[0].metadata["category"] == "ram_post"
    pkl_path_metadata = record.proposals[0].metadata["pkl_path"]
    assert pkl_path_metadata.endswith("full_pcd_ram_mock_post.pkl.gz")


def test_generate_conceptgraph_proposals_reports_unreadable_pkl_path(tmp_path: Path) -> None:
    scene = tmp_path / "scene0001_00" / "conceptgraph"
    pcd_dir = scene / "pcd_saves"
    pcd_dir.mkdir(parents=True)
    pkl_path = pcd_dir / "full_pcd_mock_post.pkl.gz"
    pkl_path.write_bytes(b"not a gzip pickle")

    with pytest.raises(ValueError) as exc_info:
        generate_conceptgraph_proposals(
            scene_path=scene,
            scan_id="scannet/scene0001_00",
            scene_id="scene0001_00",
        )

    message = str(exc_info.value)
    assert "Failed to load ConceptGraph PCD file" in message
    assert str(pkl_path) in message


def test_generate_conceptgraph_proposals_wraps_incompatible_pickle_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "scene0001_00" / "conceptgraph"
    pcd_dir = scene / "pcd_saves"
    pcd_dir.mkdir(parents=True)
    pkl_path = pcd_dir / "full_pcd_mock_post.pkl.gz"
    with gzip.open(pkl_path, "wb") as f:
        pickle.dump({"objects": []}, f)

    def raise_module_error(_file: object) -> object:
        raise ModuleNotFoundError("missing serialized dependency")

    monkeypatch.setattr(pickle, "load", raise_module_error)

    with pytest.raises(ValueError) as exc_info:
        generate_conceptgraph_proposals(
            scene_path=scene,
            scan_id="scannet/scene0001_00",
            scene_id="scene0001_00",
        )

    message = str(exc_info.value)
    assert "Failed to load ConceptGraph PCD file" in message
    assert str(pkl_path) in message
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "a", "dict"],
        {},
        {"objects": {"not": "a list"}},
    ],
)
def test_generate_conceptgraph_proposals_reports_malformed_payload_path(
    tmp_path: Path, payload: object
) -> None:
    scene = tmp_path / "scene0001_00" / "conceptgraph"
    pcd_dir = scene / "pcd_saves"
    pcd_dir.mkdir(parents=True)
    pkl_path = pcd_dir / "full_pcd_mock_post.pkl.gz"
    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(payload, f)

    with pytest.raises(ValueError) as exc_info:
        generate_conceptgraph_proposals(
            scene_path=scene,
            scan_id="scannet/scene0001_00",
            scene_id="scene0001_00",
        )

    message = str(exc_info.value)
    assert "Invalid ConceptGraph PCD payload" in message
    assert str(pkl_path) in message
