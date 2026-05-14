import gzip
import pickle
from pathlib import Path

from agents.catalog.adapters import (
    from_conceptgraph_objects,
    from_gt_embodiedscan,
)


def _write_conceptgraph_assets(
    base: Path,
    *,
    objects: list[dict],
) -> tuple[Path, Path]:
    pcd_dir = base / "pcd_saves"
    det_dir = base / "gsa_detections_ram_withbg_allclasses"
    pcd_dir.mkdir(parents=True)
    det_dir.mkdir(parents=True)
    payload = {"objects": objects}
    with gzip.open(pcd_dir / "full_pcd_v9.pkl.gz", "wb") as fh:
        pickle.dump(payload, fh)
    return pcd_dir, det_dir


def test_from_conceptgraph_objects_builds_catalog(tmp_path: Path):
    objects = [
        {
            "id": 0,
            "category": "chair",
            "bbox_3d_9dof": [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0],
        },
        {
            "id": 1,
            "category": "table",
            "bbox_3d_9dof": [1.0, 1.0, 0.0, 1.0, 1.0, 0.8, 0.0, 0.0, 0.0],
        },
    ]
    pcd_dir, det_dir = _write_conceptgraph_assets(tmp_path, objects=objects)
    view_to_objects = {
        10: [(0, 0.9), (1, 0.7)],
        20: [(1, 0.8)],
    }
    catalog = from_conceptgraph_objects(
        pcd_saves_dir=pcd_dir,
        detections_dir=det_dir,
        view_to_objects=view_to_objects,
        scene_id="openeqa/scene0709_00",
        bev_image_path="bev/scene_bev_qa.png",
        scene_category="bedroom",
        valid_frame_ids=[10, 20],
        raw_rgb_template=str(tmp_path / "raw/{frame_id:06d}-rgb.png"),
    )
    assert catalog.scene_id == "openeqa/scene0709_00"
    assert catalog.total_frames == 2
    assert {p.proposal_id for p in catalog.proposals} == {0, 1}
    chair = next(p for p in catalog.proposals if p.proposal_id == 0)
    assert chair.category == "chair"
    assert chair.position_3d == (0.0, 0.0, 0.0)
    assert chair.source == "conceptgraph"
    assert set(chair.frame_views.keys()) == {10}
    assert chair.frame_views[10].visibility_weight == 0.9
    table = next(p for p in catalog.proposals if p.proposal_id == 1)
    assert set(table.frame_views.keys()) == {10, 20}


def test_from_conceptgraph_objects_skips_objects_with_no_visible_frame(tmp_path: Path):
    objects = [
        {"id": 0, "category": "chair", "bbox_3d_9dof": [0] * 9},
        {"id": 5, "category": "monitor", "bbox_3d_9dof": [1] * 9},
    ]
    pcd_dir, det_dir = _write_conceptgraph_assets(tmp_path, objects=objects)
    view_to_objects = {10: [(0, 0.5)]}
    catalog = from_conceptgraph_objects(
        pcd_saves_dir=pcd_dir,
        detections_dir=det_dir,
        view_to_objects=view_to_objects,
        scene_id="s",
        bev_image_path="b.png",
        scene_category=None,
        valid_frame_ids=[10],
        raw_rgb_template=str(tmp_path / "raw/{frame_id:06d}-rgb.png"),
    )
    proposal_ids = {p.proposal_id for p in catalog.proposals}
    # Monitor (id=5) has zero visible frames -> excluded from catalog
    assert proposal_ids == {0}


def test_from_gt_embodiedscan_uses_gt_source(tmp_path: Path):
    es_annotations = {
        "instances": [
            {
                "bbox_id": 7,
                "category": "sofa",
                "bbox_3d_9dof": [0.0, 0.0, 0.4, 1.5, 1.0, 0.5, 0.0, 0.0, 0.0],
            }
        ],
        "view_to_objects": {12: [(7, 1.0)]},
    }
    catalog = from_gt_embodiedscan(
        es_annotations=es_annotations,
        scene_id="es/scene0000_00",
        bev_image_path="bev/scene_bev_gt.png",
        valid_frame_ids=[12],
        raw_rgb_template=str(tmp_path / "raw/{frame_id:06d}-rgb.png"),
    )
    assert catalog.proposals[0].source == "gt"
    assert catalog.proposals[0].proposal_id == 7
