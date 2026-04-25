from __future__ import annotations

import numpy as np
from PIL import Image

from benchmarks.embodiedscan_bbox_feasibility.models import (
    EmbodiedScanTarget,
    FailureTag,
)
from benchmarks.embodiedscan_bbox_feasibility.pointcloud_inputs import (
    find_scannet_mesh_path,
    materialize_detector_input,
)


def _target() -> EmbodiedScanTarget:
    return EmbodiedScanTarget(
        sample_ids=["sample-a"],
        scan_id="scannet/scene0001_00",
        scene_id="scene0001_00",
        target_id=7,
        target_category="chair",
        gt_bbox_3d=[0, 0, 0, 1, 1, 1],
    )


def _target_with_axis_align() -> EmbodiedScanTarget:
    target = _target()
    target.axis_align_matrix = [
        [1, 0, 0, 10],
        [0, 1, 0, 20],
        [0, 0, 1, 30],
        [0, 0, 0, 1],
    ]
    return target


def _target_with_visible_frame(frame_id: int) -> EmbodiedScanTarget:
    target = _target()
    target.visible_frame_ids = [frame_id]
    return target


def test_find_scannet_mesh_path_accepts_standard_scans_layout(tmp_path) -> None:
    mesh = tmp_path / "scans" / "scene0001_00" / "scene0001_00_vh_clean_2.ply"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("ply\n", encoding="utf-8")

    assert find_scannet_mesh_path(tmp_path, "scene0001_00") == mesh


def test_find_scannet_mesh_path_accepts_downloaded_clean_mesh_layout(tmp_path) -> None:
    mesh = tmp_path / "scene0001_00" / "scene0001_00_vh_clean.ply"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("ply\n", encoding="utf-8")

    assert find_scannet_mesh_path(tmp_path, "scene0001_00") == mesh


def test_materialize_single_frame_recon_writes_pointcloud(tmp_path) -> None:
    scene_root = tmp_path / "scenes" / "scene0001_00"
    raw = scene_root / "raw"
    raw.mkdir(parents=True)
    Image.fromarray(
        np.asarray([[1000, 0], [2000, 1000]], dtype=np.uint16)
    ).save(raw / "000001-depth.png")
    np.savetxt(raw / "intrinsic_depth.txt", np.eye(3), fmt="%.8f")
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    np.savetxt(raw / "000001.txt", pose, fmt="%.8f")

    record = materialize_detector_input(
        target=_target(),
        condition="single_frame_recon",
        scene_data_root=tmp_path / "scenes",
        output_dir=tmp_path / "outputs",
        frame_ids=[1],
    )

    assert record.failure_tag is None
    assert record.frame_ids == [1]
    assert record.metadata["num_points"] == 3
    assert record.pointcloud_path is not None
    ply_path = (
        tmp_path / "outputs" / "single_frame_recon" / "scene0001_00_target7.ply"
    )
    ply_text = ply_path.read_text(encoding="utf-8")
    assert "element vertex 3" in ply_text


def test_materialize_single_frame_recon_applies_axis_align(tmp_path) -> None:
    scene_root = tmp_path / "scenes" / "scene0001_00"
    raw = scene_root / "raw"
    raw.mkdir(parents=True)
    Image.fromarray(np.asarray([[1000]], dtype=np.uint16)).save(
        raw / "000001-depth.png"
    )
    np.savetxt(raw / "intrinsic_depth.txt", np.eye(3), fmt="%.8f")
    np.savetxt(raw / "000001.txt", np.eye(4), fmt="%.8f")

    record = materialize_detector_input(
        target=_target_with_axis_align(),
        condition="single_frame_recon",
        scene_data_root=tmp_path / "scenes",
        output_dir=tmp_path / "outputs",
        frame_ids=[1],
    )

    assert record.failure_tag is None
    assert record.metadata["axis_align_applied"] is True
    ply_path = (
        tmp_path / "outputs" / "single_frame_recon" / "scene0001_00_target7.ply"
    )
    lines = ply_path.read_text(encoding="utf-8").splitlines()
    assert lines[-1] == "10.000000 20.000000 31.000000"


def test_materialize_single_frame_recon_prefers_target_visible_frame_ids(
    tmp_path,
) -> None:
    scene_root = tmp_path / "scenes" / "scene0001_00"
    raw = scene_root / "raw"
    raw.mkdir(parents=True)
    for frame_id in [1, 3]:
        Image.fromarray(np.asarray([[1000]], dtype=np.uint16)).save(
            raw / f"{frame_id:06d}-depth.png"
        )
        np.savetxt(raw / f"{frame_id:06d}.txt", np.eye(4), fmt="%.8f")
    np.savetxt(raw / "intrinsic_depth.txt", np.eye(3), fmt="%.8f")

    record = materialize_detector_input(
        target=_target_with_visible_frame(3),
        condition="single_frame_recon",
        scene_data_root=tmp_path / "scenes",
        output_dir=tmp_path / "outputs",
    )

    assert record.failure_tag is None
    assert record.frame_ids == [3]


def test_materialize_single_frame_recon_marks_missing_depth_as_input_blocked(
    tmp_path,
) -> None:
    scene_root = tmp_path / "scenes" / "scene0001_00"
    raw = scene_root / "raw"
    raw.mkdir(parents=True)
    np.savetxt(raw / "intrinsic_depth.txt", np.eye(3), fmt="%.8f")
    np.savetxt(raw / "000001.txt", np.eye(4), fmt="%.8f")

    record = materialize_detector_input(
        target=_target(),
        condition="single_frame_recon",
        scene_data_root=tmp_path / "scenes",
        output_dir=tmp_path / "outputs",
        frame_ids=[1],
    )

    assert record.failure_tag == FailureTag.INPUT_BLOCKED
    assert record.pointcloud_path is None
    assert "depth" in record.metadata["reason"].lower()


def test_materialize_scannet_pose_crop_writes_cropped_mesh_points(tmp_path) -> None:
    scene_root = tmp_path / "scenes" / "scene0001_00"
    raw = scene_root / "raw"
    raw.mkdir(parents=True)
    np.savetxt(raw / "000001.txt", np.eye(4), fmt="%.8f")

    mesh = tmp_path / "scannet" / "scans" / "scene0001_00" / "scene0001_00_vh_clean_2.ply"
    mesh.parent.mkdir(parents=True)
    mesh.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
                "1 1 1",
                "10 10 10",
                "",
            ]
        ),
        encoding="utf-8",
    )

    record = materialize_detector_input(
        target=_target(),
        condition="scannet_pose_crop",
        scene_data_root=tmp_path / "scenes",
        output_dir=tmp_path / "outputs",
        scannet_root=tmp_path / "scannet",
        frame_ids=[1],
        crop_padding=2.0,
    )

    assert record.failure_tag is None
    assert record.metadata["num_points"] == 2
    assert record.pointcloud_path is not None
