"""v9 catalog-first ScanRefer pack prep tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import (
    write_v9_scene_artifacts_scanrefer,
)


def test_scanrefer_v9_emits_catalog_and_traj(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    scene_dir = tmp_path / "scene0050_01"
    pack_dir = scene_dir / "pack_scanrefer_v9_catalog_first"
    pack_dir.mkdir(parents=True)
    cg = scene_dir / "conceptgraph"
    cg.mkdir(parents=True)
    (cg / "traj.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    (cg / "intrinsic_color.txt").write_text(
        "577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n"
    )
    proposals = pack_dir / "proposals.jsonl"
    proposals.write_text(
        json.dumps(
            {
                "proposals": [
                    {
                        "id": 2,
                        "bbox_3d": [0, 0, 0, 1, 1, 1, 0, 0, 0],
                        "score": 0.5,
                        "label": "couch",
                        "frame_views": [
                            {
                                "frame_id": 5,
                                "bbox_2d": [0, 0, 10, 10],
                                "raw_rgb_path": "x.png",
                            }
                        ],
                    }
                ],
                "source": "mask3d",
            }
        )
    )

    def _fake(*, scene_id, data_root, proposals, output_path, highlight_ids):
        Path(output_path).write_bytes(b"\x89PNG\r\n\x1a\n")
        return Path(output_path)

    monkeypatch.setattr(
        "evaluation.scripts.prepare_pack_v1_inputs_scanrefer._render_v9_bev_scanrefer",
        _fake,
    )

    paths = write_v9_scene_artifacts_scanrefer(
        scene_id="scene0050_01",
        data_root=tmp_path,
        pack_name="pack_scanrefer_v9_catalog_first",
        proposals_jsonl=proposals,
        scene_category=None,
        valid_frame_ids=[5],
    )
    catalog = json.loads(Path(paths["scene_catalog_path"]).read_text())
    assert catalog["scene_id"] == "scene0050_01"
    assert catalog["bev_image_path"].endswith(".png")
    assert Path(paths["bev_image_path"]).exists()
    assert Path(paths["camera_trajectory_path"]).exists()
