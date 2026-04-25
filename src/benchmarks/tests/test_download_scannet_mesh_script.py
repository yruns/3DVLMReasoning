from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    script_path = Path("scripts/download_scannet_mesh.py")
    spec = importlib.util.spec_from_file_location("download_scannet_mesh", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_scene_file_strips_scan_prefix_and_comments(tmp_path: Path) -> None:
    module = _load_script_module()
    scene_file = tmp_path / "scenes.txt"
    scene_file.write_text(
        "\n".join(
            [
                "# comment",
                "scannet/scene0415_00",
                "scene0076_00",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert module.read_scene_file(scene_file) == ["scene0415_00", "scene0076_00"]
