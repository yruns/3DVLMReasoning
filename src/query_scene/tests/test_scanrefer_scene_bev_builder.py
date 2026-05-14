from pathlib import Path

import pytest

from query_scene.scene_bev_builder import ScanReferScanNetBEVBuilder


def _make_scanrefer_layout(tmp_path: Path, scene_id: str) -> Path:
    scene_dir = tmp_path / scene_id
    cg = scene_dir / "conceptgraph"
    cg.mkdir(parents=True)
    (cg / "traj.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    (cg / "intrinsic_color.txt").write_text("577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n")
    return scene_dir


def test_resolve_paths_uses_conceptgraph_intrinsic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scene_dir = _make_scanrefer_layout(tmp_path, "scene0050_01")
    fake_mesh = tmp_path / "scannetv2" / "scene0050_01" / "scene0050_01_vh_clean_2.ply"
    fake_mesh.parent.mkdir(parents=True)
    fake_mesh.write_text("ply\n")
    monkeypatch.setenv("SCANNET_DATA_ROOT", str(tmp_path / "scannetv2"))
    builder = ScanReferScanNetBEVBuilder()
    mesh, traj, intr = builder.resolve_paths("scene0050_01", tmp_path)
    assert mesh == fake_mesh
    assert traj == scene_dir / "conceptgraph" / "traj.txt"
    assert intr == scene_dir / "conceptgraph" / "intrinsic_color.txt"


def test_benchmark_tag_is_scanrefer():
    assert ScanReferScanNetBEVBuilder.benchmark == "scanrefer"
