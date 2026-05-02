"""Smoke tests for ScanRefer side-by-side runner."""

from __future__ import annotations

import pytest


def test_parse_sample_id_canonical():
    from evaluation.scripts.run_scanrefer_vg_side_by_side import parse_scanrefer_sample_id
    parsed = parse_scanrefer_sample_id("scannet/scene0088_00::5::3")
    assert parsed.scan_id == "scannet/scene0088_00"
    assert parsed.scene_id == "scene0088_00"
    assert parsed.target_id == 5
    assert parsed.ann_id == "3"


def test_parse_sample_id_rejects_malformed():
    from evaluation.scripts.run_scanrefer_vg_side_by_side import parse_scanrefer_sample_id
    with pytest.raises(ValueError, match="format"):
        parse_scanrefer_sample_id("scannet/scene_x::5")


def test_safe_sample_id():
    from evaluation.scripts.run_scanrefer_vg_side_by_side import safe_sample_id
    assert safe_sample_id("scannet/scene_a::5::3") == "scannet__scene_a__5__3"


def test_compare_backends_persists_failed_sentinel_on_sample_exception(tmp_path, monkeypatch):
    """Mirror of NR3D test: a per-sample exception → failed sentinel, not crash."""
    from evaluation.scripts import run_scanrefer_vg_side_by_side as mod

    def fake_run_one_sample(sample_id, backend, **kwargs):
        if sample_id.endswith("::3"):
            raise RuntimeError("simulated upstream 500")
        return {"sample_id": sample_id, "backend": backend, "status": "completed",
                "iou": 1.0, "predicted_bbox_3d_9dof": [0]*9,
                "gt_bbox_3d_9dof": [0]*9, "selected_object_id": 0,
                "confidence": 0.9, "query": "test"}

    monkeypatch.setattr(mod, "run_one_sample", fake_run_one_sample)
    monkeypatch.setattr(mod, "validate_unique_sample_ids", lambda _: None)
    monkeypatch.setattr(mod, "preflight_pack_sample_exists", lambda *a, **kw: None)

    out = tmp_path / "out"
    sample_ids = ["scannet/scene_a::1::0", "scannet/scene_a::2::3"]
    results = mod.compare_backends(
        sample_ids=sample_ids, output_dir=out, data_root=tmp_path,
        pack_name="pack_scanrefer_v1", workers=1,
    )
    per_sample = results["pack_v1"]["per_sample"]
    assert len(per_sample) == 2
    failed = [r for r in per_sample if r["sample_id"].endswith("::3")][0]
    assert failed["status"] == "failed"
    assert failed["iou"] == 0.0
    assert "simulated upstream 500" in failed["error"]
