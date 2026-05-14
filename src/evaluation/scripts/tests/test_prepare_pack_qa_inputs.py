"""v9 catalog-first QA pack prep tests (OpenEQA + SQA3D)."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import pytest

from evaluation.scripts.prepare_pack_qa_inputs import write_qa_scene_artifacts


def _make_clip_layout(tmp_path: Path) -> Path:
    clip = tmp_path / "002-scannet-scene0709_00"
    cg = clip / "conceptgraph"
    raw = clip / "raw"
    cg.mkdir(parents=True)
    raw.mkdir(parents=True)
    (cg / "traj.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    (raw / "intrinsic_color.txt").write_text(
        "577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n"
    )
    (cg / "scene_info.json").write_text(json.dumps({"scan_id": "scene0709_00"}))
    pcd = cg / "pcd_saves"
    pcd.mkdir()
    with gzip.open(pcd / "full_pcd_v9.pkl.gz", "wb") as fh:
        pickle.dump(
            {
                "objects": [
                    {
                        "id": 0,
                        "category": "chair",
                        "bbox_3d_9dof": [0.0] * 9,
                    },
                    {
                        "id": 1,
                        "category": "table",
                        "bbox_3d_9dof": [1.0] * 9,
                    },
                ]
            },
            fh,
        )
    det = cg / "gsa_detections_ram_withbg_allclasses"
    det.mkdir()
    return clip


def test_write_qa_scene_artifacts_openeqa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _make_clip_layout(tmp_path)

    def _fake_render(*, scene_id, data_root, proposals, output_path, highlight_ids):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"\x89PNG\r\n\x1a\n")
        return Path(output_path)

    monkeypatch.setattr(
        "evaluation.scripts.prepare_pack_qa_inputs._render_qa_bev",
        _fake_render,
    )
    paths = write_qa_scene_artifacts(
        benchmark="openeqa",
        clip_id="002-scannet-scene0709_00",
        data_root=tmp_path,
        pack_name="pack_openeqa_v9_catalog_first",
        view_to_objects={5: [(0, 0.9)], 7: [(1, 0.8)]},
        valid_frame_ids=[5, 7],
        scene_category="bedroom",
    )
    catalog = json.loads(Path(paths["scene_catalog_path"]).read_text())
    assert catalog["scene_id"] == "002-scannet-scene0709_00"
    assert catalog["scene_category"] == "bedroom"
    assert {p["proposal_id"] for p in catalog["proposals"]} == {0, 1}
    assert Path(paths["bev_image_path"]).exists()
    assert Path(paths["camera_trajectory_path"]).exists()


def test_write_qa_scene_artifacts_sqa3d(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """SQA3D layout: data_root/<scene_id> (no clip prefix), same conceptgraph subdir."""
    scene_dir = tmp_path / "scene0050_00"
    cg = scene_dir / "conceptgraph"
    raw = scene_dir / "raw"
    cg.mkdir(parents=True)
    raw.mkdir(parents=True)
    (cg / "traj.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    (raw / "intrinsic_color.txt").write_text(
        "577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n"
    )
    pcd = cg / "pcd_saves"
    pcd.mkdir()
    with gzip.open(pcd / "full_pcd_v9.pkl.gz", "wb") as fh:
        pickle.dump(
            {"objects": [{"id": 3, "category": "sofa", "bbox_3d_9dof": [0] * 9}]},
            fh,
        )
    det = cg / "gsa_detections_ram_withbg_allclasses"
    det.mkdir()

    def _fake_render(**kw):
        Path(kw["output_path"]).write_bytes(b"\x89PNG\r\n\x1a\n")
        return Path(kw["output_path"])

    monkeypatch.setattr(
        "evaluation.scripts.prepare_pack_qa_inputs._render_qa_bev",
        _fake_render,
    )
    paths = write_qa_scene_artifacts(
        benchmark="sqa3d",
        clip_id="scene0050_00",
        data_root=tmp_path,
        pack_name="pack_sqa3d_v9_catalog_first",
        view_to_objects={12: [(3, 1.0)]},
        valid_frame_ids=[12],
        scene_category=None,
    )
    catalog = json.loads(Path(paths["scene_catalog_path"]).read_text())
    assert catalog["scene_id"] == "scene0050_00"
    assert catalog["proposals"][0]["proposal_id"] == 3


def test_write_qa_scene_artifacts_rejects_unknown_benchmark(tmp_path: Path):
    with pytest.raises(ValueError, match="unsupported QA benchmark"):
        write_qa_scene_artifacts(
            benchmark="bogus",
            clip_id="x",
            data_root=tmp_path,
            pack_name="p",
            view_to_objects={},
            valid_frame_ids=[1],
            scene_category=None,
        )
