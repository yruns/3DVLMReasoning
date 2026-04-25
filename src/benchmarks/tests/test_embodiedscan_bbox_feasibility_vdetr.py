from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks.embodiedscan_bbox_feasibility.vdetr import (
    VDETR_SCANNET_CKPT_URL,
    build_vdetr_command_template,
    camera_corners_to_depth_corners,
    corners_to_aabb_9dof,
    write_vdetr_proposal_json,
)


def test_build_vdetr_command_template_uses_repo_ckpt_and_placeholders(tmp_path: Path) -> None:
    template = build_vdetr_command_template(
        repo_dir=tmp_path / "V-DETR",
        checkpoint_path=tmp_path / "scannet_540ep.pth",
        python_executable="/opt/vdetr/bin/python",
        num_points=40000,
        conf_thresh=0.05,
        top_k=128,
    )

    assert template.startswith("/opt/vdetr/bin/python ")
    assert "scripts/vdetr_export_predictions.py" in template
    assert f"--repo-dir {tmp_path / 'V-DETR'}" in template
    assert f"--checkpoint {tmp_path / 'scannet_540ep.pth'}" in template
    assert "--pointcloud {pointcloud_path}" in template
    assert "--output {output_path}" in template
    assert "--num-points 40000" in template
    assert "--conf-thresh 0.05" in template
    assert "--top-k 128" in template


def test_vdetr_checkpoint_url_points_to_official_huggingface_asset() -> None:
    assert VDETR_SCANNET_CKPT_URL.endswith("/resolve/main/scannet_540ep.pth")


def test_corners_to_aabb_9dof_converts_eight_corners() -> None:
    corners = np.asarray(
        [
            [-1, -2, -3],
            [3, -2, -3],
            [3, 4, -3],
            [-1, 4, -3],
            [-1, -2, 5],
            [3, -2, 5],
            [3, 4, 5],
            [-1, 4, 5],
        ],
        dtype=np.float32,
    )

    assert corners_to_aabb_9dof(corners) == [1.0, 1.0, 1.0, 4.0, 6.0, 8.0, 0.0, 0.0, 0.0]


def test_camera_corners_to_depth_corners_inverts_vdetr_axis_flip() -> None:
    depth_corners = np.asarray(
        [
            [-1, -2, -3],
            [3, -2, -3],
            [3, 4, -3],
            [-1, 4, -3],
            [-1, -2, 5],
            [3, -2, 5],
            [3, 4, 5],
            [-1, 4, 5],
        ],
        dtype=np.float32,
    )
    camera_corners = depth_corners.copy()
    camera_corners[:, [0, 1, 2]] = camera_corners[:, [0, 2, 1]]
    camera_corners[:, 1] *= -1

    restored = camera_corners_to_depth_corners(camera_corners)

    np.testing.assert_allclose(restored, depth_corners)


def test_write_vdetr_proposal_json_sorts_and_limits_predictions(tmp_path: Path) -> None:
    output = tmp_path / "pred.json"
    predictions = [
        {"label": "chair", "score": 0.2, "corners": np.zeros((8, 3))},
        {"label": "table", "score": 0.9, "corners": np.ones((8, 3))},
    ]

    write_vdetr_proposal_json(
        output_path=output,
        predictions=predictions,
        top_k=1,
    )

    data = json.loads(output.read_text())
    assert len(data["proposals"]) == 1
    assert data["proposals"][0]["label"] == "table"
    assert data["proposals"][0]["score"] == 0.9
    assert data["proposals"][0]["source"] == "vdetr"
