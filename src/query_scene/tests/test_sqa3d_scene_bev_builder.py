from pathlib import Path

import pytest

from query_scene.scene_bev_builder import Sqa3dScanNetBEVBuilder


def test_resolve_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scene = tmp_path / "scene0050_00"
    (scene / "conceptgraph").mkdir(parents=True)
    (scene / "raw").mkdir(parents=True)
    (scene / "conceptgraph" / "traj.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    (scene / "raw" / "intrinsic_color.txt").write_text(
        "577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n"
    )
    fake_mesh = tmp_path / "scannetv2" / "scene0050_00" / "scene0050_00_vh_clean.ply"
    fake_mesh.parent.mkdir(parents=True)
    fake_mesh.write_text("ply\n")
    monkeypatch.setenv("SCANNET_DATA_ROOT", str(tmp_path / "scannetv2"))
    builder = Sqa3dScanNetBEVBuilder()
    mesh, traj, intr = builder.resolve_paths("scene0050_00", tmp_path)
    assert mesh == fake_mesh
    assert traj == scene / "conceptgraph" / "traj.txt"
    assert intr == scene / "raw" / "intrinsic_color.txt"


def test_benchmark_tag_is_sqa3d():
    assert Sqa3dScanNetBEVBuilder.benchmark == "sqa3d"
