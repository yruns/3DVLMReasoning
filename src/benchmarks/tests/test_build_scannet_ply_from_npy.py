from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import numpy as np


def _load_module():
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "build_scannet_ply_from_npy.py"
    spec = importlib.util.spec_from_file_location("build_scannet_ply_from_npy", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_scene_ids_from_db_preserves_first_seen_order(tmp_path: Path) -> None:
    module = _load_module()
    db_path = tmp_path / "runs.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE samples (run_id TEXT NOT NULL, sample_id TEXT NOT NULL, scene_id TEXT, target_id INTEGER)"
        )
        conn.executemany(
            "INSERT INTO samples (run_id, sample_id, scene_id, target_id) VALUES (?, ?, ?, ?)",
            [
                ("v3", "scene_b::2", "scene_b", 2),
                ("v3", "scene_a::1", "scene_a", 1),
                ("v3", "scene_b::3", "scene_b", 3),
                ("other", "scene_c::1", "scene_c", 1),
            ],
        )

    assert module.load_scene_ids_from_db(db_path, "v3") == ["scene_b", "scene_a"]


def test_build_vertex_array_uses_scannet_vertex_schema() -> None:
    module = _load_module()
    coord = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    color = np.asarray([[7, 8, 9], [10, 11, 12]], dtype=np.uint8)
    normal = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

    vertex = module.build_vertex_array(coord, color, normal)

    assert vertex.dtype.names == (
        "x",
        "y",
        "z",
        "red",
        "green",
        "blue",
        "nx",
        "ny",
        "nz",
    )
    assert vertex.dtype["x"] == np.dtype("float32")
    assert vertex.dtype["red"] == np.dtype("uint8")
    assert vertex.dtype["nx"] == np.dtype("float32")
    assert vertex["x"].tolist() == [1.0, 4.0]
    assert vertex["green"].tolist() == [8, 11]
    np.testing.assert_allclose(vertex["nz"], np.asarray([0.3, 0.6], dtype=np.float32))


def test_materialize_scene_skips_existing_ply_with_same_vertex_count(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    source_scene = tmp_path / "source" / "scene0001_00"
    source_scene.mkdir(parents=True)
    np.save(source_scene / "coord.npy", np.zeros((2, 3), dtype=np.float32))
    np.save(source_scene / "color.npy", np.zeros((2, 3), dtype=np.uint8))
    np.save(source_scene / "normal.npy", np.zeros((2, 3), dtype=np.float32))
    output_ply = tmp_path / "out" / "scans" / "scene0001_00" / "scene0001_00_vh_clean_2.ply"
    output_ply.parent.mkdir(parents=True)
    output_ply.write_bytes(b"already here")

    monkeypatch.setattr(module, "read_ply_vertex_count", lambda path: 2)

    def fail_write(*_args, **_kwargs):
        raise AssertionError("write_ply should not be called for same-count output")

    monkeypatch.setattr(module, "write_ply", fail_write)

    result = module.materialize_scene(
        "scene0001_00",
        source_root=tmp_path / "source",
        output_root=tmp_path / "out",
    )

    assert result.status == "skipped"
    assert result.vertex_count == 2
    assert result.output_path == output_ply
