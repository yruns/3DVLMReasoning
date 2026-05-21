"""v9 catalog-first NR3D pack prep tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.scripts.prepare_pack_v1_inputs_nr3d import write_v9_scene_artifacts


def _write_dummy_proposals(scene_dir: Path) -> Path:
    p_path = scene_dir / "pack_nr3d_v1" / "proposals.jsonl"
    p_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "gt",
        "scene_id": scene_dir.name,
        "axis_align_matrix": None,
        "proposals": [
            {
                "id": 0,
                "bbox_3d": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                "score": 0.9,
                "label": "chair",
                "frame_views": {
                    "10": {
                        "frame_id": 10,
                        "bbox_2d": [0, 0, 50, 50],
                        "raw_rgb_path": str(scene_dir / "raw" / "000010-rgb.png"),
                    }
                },
            }
        ],
    }
    p_path.write_text(json.dumps(payload))
    return p_path


def _write_traj(scene_dir: Path) -> None:
    cg = scene_dir / "conceptgraph"
    cg.mkdir(parents=True, exist_ok=True)
    traj = "\n".join(
        " ".join(map(str, row))
        for row in [
            [1.0, 0, 0, 0],
            [0, 1.0, 0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 0, 1.0],
        ]
    )
    (cg / "traj.txt").write_text(traj + "\n")
    (cg / "intrinsic_color.txt").write_text(
        "577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n"
    )


def _write_enrichment(scene_dir: Path) -> None:
    cg = scene_dir / "conceptgraph"
    cg.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": "enriched_objects_v1",
        "objects": [
            {
                "obj_id": 0,
                "status": "success",
                "original_label": "chair",
                "enrichment": {
                    "category": "office chair",
                    "description": "A black wheeled office chair with a padded seat.",
                    "location": "Near the desk and wall.",
                    "nearby_objects": ["desk", "wall"],
                    "color": "black",
                    "usability": "Provides seating for working at the desk.",
                },
            }
        ],
    }
    (cg / "enriched_objects.json").write_text(json.dumps(payload), encoding="utf-8")


def test_write_v9_scene_artifacts_emits_catalog_and_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    scene_dir = tmp_path / "scene0000_00"
    _write_dummy_proposals(scene_dir)
    _write_traj(scene_dir)
    _write_enrichment(scene_dir)

    captured: dict = {}

    def _fake_bev(*, scene_id, data_root, proposals, output_path, highlight_ids):
        captured["scene_id"] = scene_id
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        return output_path

    monkeypatch.setattr(
        "evaluation.scripts.prepare_pack_v1_inputs_nr3d._render_v9_bev",
        _fake_bev,
    )

    paths = write_v9_scene_artifacts(
        scene_id="scene0000_00",
        data_root=tmp_path,
        pack_name="pack_nr3d_v9_catalog_first",
        proposals_jsonl=scene_dir / "pack_nr3d_v1" / "proposals.jsonl",
        scene_category="kitchen",
        valid_frame_ids=[10, 20, 30],
    )

    bev_path = paths["bev_image_path"]
    catalog_path = paths["scene_catalog_path"]
    assert Path(bev_path).exists()
    assert Path(catalog_path).exists()
    catalog = json.loads(Path(catalog_path).read_text())
    assert catalog["scene_id"] == "scene0000_00"
    assert catalog["scene_category"] == "kitchen"
    assert catalog["bev_image_path"].endswith(".png")
    assert "valid_frame_ids" in catalog
    proposal = catalog["proposals"][0]
    assert proposal["category"] == "chair"
    assert proposal["enriched_category"] == "office chair"
    assert "black wheeled office chair" in proposal["compact_note"]
    assert proposal["enrichment"]["location"] == "Near the desk and wall."
    traj = json.loads(Path(paths["camera_trajectory_path"]).read_text())
    assert "0" in traj or 0 in traj
    assert captured["scene_id"] == "scene0000_00"
