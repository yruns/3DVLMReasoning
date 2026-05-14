from pathlib import Path

import pytest

from query_scene.scene_bev_builder import OpenEqaScanNetBEVBuilder


def _make_openeqa_layout(tmp_path: Path, clip_id: str, scene_id: str) -> Path:
    clip_dir = tmp_path / clip_id
    cg = clip_dir / "conceptgraph"
    raw = clip_dir / "raw"
    cg.mkdir(parents=True)
    raw.mkdir(parents=True)
    (cg / "traj.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    (raw / "intrinsic_color.txt").write_text("577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n")
    info = {"scan_id": scene_id, "scene_id": scene_id}
    (cg / "scene_info.json").write_text(__import__("json").dumps(info))
    return clip_dir


def test_resolve_paths_uses_scene_info(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    clip_dir = _make_openeqa_layout(tmp_path, "002-scannet-scene0709_00", "scene0709_00")
    fake_mesh = tmp_path / "scannetv2" / "scene0709_00" / "scene0709_00_vh_clean.ply"
    fake_mesh.parent.mkdir(parents=True)
    fake_mesh.write_text("ply\n")
    monkeypatch.setenv("SCANNET_DATA_ROOT", str(tmp_path / "scannetv2"))
    builder = OpenEqaScanNetBEVBuilder()
    mesh, traj, intr = builder.resolve_paths("002-scannet-scene0709_00", tmp_path)
    assert mesh == fake_mesh
    assert traj == clip_dir / "conceptgraph" / "traj.txt"
    assert intr == clip_dir / "raw" / "intrinsic_color.txt"


def test_resolve_paths_falls_back_to_clip_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    clip = tmp_path / "041-scannet-scene0011_00"
    (clip / "conceptgraph").mkdir(parents=True)
    (clip / "raw").mkdir(parents=True)
    (clip / "conceptgraph" / "traj.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    (clip / "raw" / "intrinsic_color.txt").write_text("577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n")
    fake_mesh = tmp_path / "scannetv2" / "scene0011_00" / "scene0011_00_vh_clean.ply"
    fake_mesh.parent.mkdir(parents=True)
    fake_mesh.write_text("ply\n")
    monkeypatch.setenv("SCANNET_DATA_ROOT", str(tmp_path / "scannetv2"))
    builder = OpenEqaScanNetBEVBuilder()
    mesh, *_ = builder.resolve_paths("041-scannet-scene0011_00", tmp_path)
    assert mesh == fake_mesh
