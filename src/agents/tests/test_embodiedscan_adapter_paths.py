"""Path resolution tests for EmbodiedScan adapters."""

from pathlib import Path

from agents.adapters.embodiedscan_adapter import EmbodiedScanVGAdapter
from benchmarks.embodiedscan_loader import EmbodiedScanVGSample


def test_get_scene_path_preserves_scan_source_prefix(tmp_path: Path) -> None:
    adapter = EmbodiedScanVGAdapter(
        data_root=tmp_path / "embodiedscan",
        scene_data_root=tmp_path / "embodiedscan",
    )
    sample = EmbodiedScanVGSample(
        sample_id="s1",
        scene_id="scene0040_00",
        scan_id="scannet/scene0040_00",
        query="find the chair",
    )

    assert adapter.get_scene_path(sample) == (
        tmp_path / "embodiedscan" / "scannet" / "scene0040_00"
    )
