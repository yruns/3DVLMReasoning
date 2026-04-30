"""Tests for ConceptGraph proposal extraction."""

from __future__ import annotations

import gzip
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def test_object_to_proposal_uses_bbox_np_not_observed_pcd_surface() -> None:
    extractor = _load_script_module("extract_conceptgraph_proposals")

    bbox_np = _box_surface_points(center=(1.0, 2.0, 3.0), extent=(2.0, 4.0, 6.0))
    visible_surface = _box_surface_points(
        center=(1.0, 2.0, 3.0),
        extent=(0.2, 0.4, 0.6),
    )

    proposal = extractor.object_to_proposal(
        {
            "pcd_np": visible_surface,
            "bbox_np": bbox_np,
            "conf": [0.9],
            "num_detections": 5,
        },
        scene_id="scene0001_00",
        object_index=0,
        is_background=False,
    )

    bbox_3d = proposal["bbox_3d"]
    assert len(bbox_3d) == 9
    assert np.all(np.isfinite(bbox_3d))
    assert np.allclose(bbox_3d[:3], [1.0, 2.0, 3.0], atol=1e-5)
    assert np.allclose(bbox_3d[3:6], [2.0, 4.0, 6.0], atol=1e-5)
    assert np.allclose(bbox_3d[6:9], [0.0, 0.0, 0.0], atol=1e-9)
    raw_corners = proposal["metadata"]["raw_corners"]
    assert np.asarray(raw_corners).shape == (8, 3)
    assert np.allclose(raw_corners, bbox_np)
    assert proposal["metadata"]["box_format"] == "aabb_from_conceptgraph_bbox_np"


def test_empty_conceptgraph_pickle_raises(tmp_path: Path) -> None:
    extractor = _load_script_module("extract_conceptgraph_proposals")

    _write_conceptgraph_pickle(
        tmp_path,
        "scene0001_00",
        {"objects": [], "bg_objects": None},
    )

    with pytest.raises(ValueError, match="scene0001_00.*zero ConceptGraph objects"):
        extractor.extract_scene_record(
            data_root=tmp_path,
            scene_id="scene0001_00",
            include_bg=True,
            min_points=50,
        )


def test_min_points_filter_counts_filtered_objects(tmp_path: Path) -> None:
    extractor = _load_script_module("extract_conceptgraph_proposals")

    _write_conceptgraph_pickle(
        tmp_path,
        "scene0001_00",
        {
            "objects": [
                _conceptgraph_object(10, conf=[0.9], num_detections=5),
                _conceptgraph_object(64, conf=[0.8], num_detections=8),
            ],
            "bg_objects": None,
        },
    )

    record = extractor.extract_scene_record(
        data_root=tmp_path,
        scene_id="scene0001_00",
        include_bg=True,
        min_points=50,
    )

    assert len(record["proposals"]) == 1
    assert record["metadata"]["n_objects_total"] == 2
    assert record["metadata"]["n_objects_filtered_min_points"] == 1
    assert record["proposals"][0]["metadata"]["n_points"] == 64


def test_emitted_record_satisfies_detector_pack_schema(tmp_path: Path) -> None:
    from benchmarks.embodiedscan_bbox_feasibility.models import ProposalRecord

    extractor = _load_script_module("extract_conceptgraph_proposals")
    pack_inputs = _load_script_module("prepare_detector_pack_inputs")

    _write_conceptgraph_pickle(
        tmp_path,
        "scene0001_00",
        {
            "objects": [_conceptgraph_object(64, conf=[0.35, 0.75], num_detections=4)],
            "bg_objects": [],
        },
    )

    raw = extractor.extract_scene_record(
        data_root=tmp_path,
        scene_id="scene0001_00",
        include_bg=True,
        min_points=50,
    )
    record = ProposalRecord.model_validate(json.loads(json.dumps(raw)))
    proposals = pack_inputs.build_proposals_from_detector_record(
        record,
        max_proposals_per_scene=256,
    )

    assert record.scene_id == "scene0001_00"
    assert record.scan_id == "scannet/scene0001_00"
    assert record.method == "3d-conceptgraph"
    assert proposals[0]["metadata"]["detector"] == "ConceptGraph"
    assert np.asarray(proposals[0]["metadata"]["raw_corners"]).shape == (8, 3)
    assert proposals[0]["metadata"]["class_id"] == -1
    assert proposals[0]["score"] == pytest.approx(0.75)


def _write_conceptgraph_pickle(root: Path, scene_id: str, payload: dict) -> Path:
    pcd_dir = root / "scannet" / scene_id / "conceptgraph" / "pcd_saves"
    pcd_dir.mkdir(parents=True)
    path = (
        pcd_dir
        / "full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz"
    )
    with gzip.open(path, "wb") as handle:
        pickle.dump(payload, handle)
    return path


def _load_script_module(module_name: str) -> ModuleType:
    script_path = Path("src") / "evaluation" / "scripts" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _conceptgraph_object(
    n_points: int,
    *,
    conf: list[float] | None,
    num_detections: int,
) -> dict:
    points = _box_surface_points(center=(0.0, 0.0, 0.0), extent=(1.0, 2.0, 3.0))
    repeats = int(np.ceil(n_points / points.shape[0]))
    points = np.tile(points, (repeats, 1))[:n_points]
    return {
        "pcd_np": points.astype(np.float64),
        "bbox_np": _box_surface_points(
            center=(0.0, 0.0, 0.0),
            extent=(1.0, 2.0, 3.0),
        ),
        "conf": conf,
        "num_detections": num_detections,
        "n_points": n_points,
    }


def _box_surface_points(
    *,
    center: tuple[float, float, float],
    extent: tuple[float, float, float],
) -> np.ndarray:
    cx, cy, cz = center
    dx, dy, dz = [value / 2.0 for value in extent]
    xs = [cx - dx, cx + dx]
    ys = [cy - dy, cy + dy]
    zs = [cz - dz, cz + dz]
    return np.asarray([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)
