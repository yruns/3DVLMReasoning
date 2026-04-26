"""Side-by-side runs both backends and produces a comparison table."""
from __future__ import annotations

import pytest


@pytest.mark.integration
def test_side_by_side_emits_comparison_dict(monkeypatch, tmp_path) -> None:
    from evaluation.scripts.run_embodiedscan_vg_side_by_side import (
        compare_backends,
    )

    def fake_run_one(sample_id, backend):
        return {"sample_id": sample_id, "iou": 0.5 if backend == "pack_v1" else 0.4}

    monkeypatch.setattr(
        "evaluation.scripts.run_embodiedscan_vg_side_by_side.run_one_sample",
        fake_run_one,
    )
    out = compare_backends(sample_ids=["s1", "s2"], output_dir=tmp_path)
    assert "legacy" in out and "pack_v1" in out
    assert out["pack_v1"]["mean_iou"] >= out["legacy"]["mean_iou"]
