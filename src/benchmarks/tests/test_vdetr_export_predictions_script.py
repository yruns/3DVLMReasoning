from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np


def _load_script_module():
    script_path = Path("scripts/vdetr_export_predictions.py")
    spec = importlib.util.spec_from_file_location("vdetr_export_predictions", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_import_does_not_import_benchmarks_package() -> None:
    sys.modules.pop("benchmarks", None)

    _load_script_module()

    assert "benchmarks" not in sys.modules


def test_load_ascii_xyz_ply_reads_generated_detector_input(tmp_path: Path) -> None:
    module = _load_script_module()
    ply = tmp_path / "cloud.ply"
    ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "1 2 3",
                "4 5 6",
                "",
            ]
        ),
        encoding="utf-8",
    )

    points = module.load_ascii_xyz_ply(ply)

    np.testing.assert_allclose(points, np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32))


def test_sample_points_is_deterministic_without_replacement() -> None:
    module = _load_script_module()
    points = np.arange(30, dtype=np.float32).reshape(10, 3)

    sampled = module.sample_points(points, 4)

    np.testing.assert_allclose(sampled, points[[0, 3, 6, 9]])


def test_prepare_model_points_adds_mean_color_channels_for_xyz_color_checkpoint() -> None:
    module = _load_script_module()
    points = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    class Args:
        use_color = True
        xyz_color = True
        use_normals = False

    prepared = module.prepare_model_points(points, Args())

    assert prepared.shape == (2, 6)
    np.testing.assert_allclose(prepared[:, :3], points)
    np.testing.assert_allclose(prepared[:, 3:], np.zeros((2, 3), dtype=np.float32))


def test_ensure_vdetr_arg_defaults_adds_missing_random_fps_flag() -> None:
    module = _load_script_module()
    args = Namespace()

    module.ensure_vdetr_arg_defaults(args)

    assert args.random_fps is False


def test_prediction_to_dict_converts_camera_corners_to_depth_frame() -> None:
    module = _load_script_module()
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

    row = module._prediction_to_dict((2, camera_corners, 0.7))

    np.testing.assert_allclose(row["corners"], depth_corners)
