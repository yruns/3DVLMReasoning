from pathlib import Path

import pytest

from query_scene.scene_bev_builder import Nr3dScanNetBEVBuilder


def _make_nr3d_layout(tmp_path: Path, scene_id: str) -> Path:
    """Create a Phase-8 NR3D scene layout: <scene>/conceptgraph/{traj.txt} + <scene>/raw/intrinsic_color.txt."""
    scene_dir = tmp_path / scene_id
    (scene_dir / "conceptgraph").mkdir(parents=True)
    (scene_dir / "raw").mkdir(parents=True)
    (scene_dir / "conceptgraph" / "traj.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    intr = "577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n"
    (scene_dir / "raw" / "intrinsic_color.txt").write_text(intr)
    return scene_dir


def test_resolve_paths_finds_mesh_traj_intrinsic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scene_dir = _make_nr3d_layout(tmp_path, "scene0000_00")
    fake_mesh = tmp_path / "scannetv2" / "scene0000_00" / "scene0000_00_vh_clean.ply"
    fake_mesh.parent.mkdir(parents=True)
    fake_mesh.write_text("ply\n")
    monkeypatch.setenv("SCANNET_DATA_ROOT", str(tmp_path / "scannetv2"))
    builder = Nr3dScanNetBEVBuilder()
    mesh, traj, intr = builder.resolve_paths("scene0000_00", tmp_path)
    assert mesh == fake_mesh
    assert traj == scene_dir / "conceptgraph" / "traj.txt"
    assert intr == scene_dir / "raw" / "intrinsic_color.txt"


def test_resolve_paths_missing_mesh_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _make_nr3d_layout(tmp_path, "scene0000_00")
    monkeypatch.setenv("SCANNET_DATA_ROOT", str(tmp_path / "scannetv2_empty"))
    builder = Nr3dScanNetBEVBuilder()
    with pytest.raises(FileNotFoundError, match="scannet mesh"):
        builder.resolve_paths("scene0000_00", tmp_path)


def test_benchmark_tag_is_nr3d():
    assert Nr3dScanNetBEVBuilder.benchmark == "nr3d"
