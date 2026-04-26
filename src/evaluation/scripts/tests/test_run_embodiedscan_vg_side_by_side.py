"""Side-by-side runs both backends and produces a comparison table."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _write_pack_v1_inputs(tmp_path, *, sample_id: str = "scene0001_00::72"):
    scene_id, target_id = sample_id.split("::")
    pack_dir = tmp_path / "pack_v1"
    scene_dir = pack_dir / "scenes" / scene_id
    annotated = scene_dir / "annotated"
    samples = pack_dir / "samples"
    annotated.mkdir(parents=True)
    samples.mkdir(parents=True)

    (annotated / "frame_10.png").write_bytes(b"\x89PNG")
    (scene_dir / "proposals.jsonl").write_text(
        json.dumps(
            {
                "proposals": [
                    {
                        "bbox_3d": [0, 0, 0, 1, 1, 1, 0, 0, 0],
                        "score": 0.9,
                        "label": "chair",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "visibility.json").write_text(
        json.dumps({"10": [0]}),
        encoding="utf-8",
    )
    sample_path = samples / f"{scene_id}__{target_id}.json"
    sample_path.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "target_id": int(target_id),
                "category": "chair",
                "query": "the chair by the table",
                "gt_bbox_3d_9dof": [0, 0, 0, 1, 1, 1, 0, 0, 0],
                "scene_artifacts_dir": str(scene_dir),
                "source": "vdetr",
                "keyframes": [
                    {
                        "keyframe_idx": 0,
                        "image_path": str(annotated / "frame_10.png"),
                        "frame_id": 10,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return pack_dir


@pytest.mark.integration
def test_side_by_side_emits_comparison_dict(monkeypatch, tmp_path) -> None:
    from evaluation.scripts.run_embodiedscan_vg_side_by_side import (
        compare_backends,
    )

    def fake_run_one(sample_id, backend, **_kwargs):
        return {"sample_id": sample_id, "iou": 0.5 if backend == "pack_v1" else 0.4}

    monkeypatch.setattr(
        "evaluation.scripts.run_embodiedscan_vg_side_by_side.run_one_sample",
        fake_run_one,
    )
    out = compare_backends(
        sample_ids=["s1", "s2"],
        output_dir=tmp_path,
        pack_v1_inputs_dir=tmp_path / "pack_v1",
        embodiedscan_data_root=tmp_path / "embodiedscan",
    )
    assert "legacy" in out and "pack_v1" in out
    assert out["pack_v1"]["mean_iou"] >= out["legacy"]["mean_iou"]


@pytest.mark.integration
def test_pack_v1_run_one_sample_scores_agent_bbox(monkeypatch, tmp_path) -> None:
    from evaluation.scripts import run_embodiedscan_vg_side_by_side as runner

    pack_dir = _write_pack_v1_inputs(tmp_path)

    class FakeAgent:
        def __init__(self, config):
            assert config.vg_backend == "pack_v1"

        def run(self, task, bundle):
            assert task.user_query == "the chair by the table"
            assert bundle.scene_id == "scene0001_00"
            return SimpleNamespace(
                result=SimpleNamespace(
                    payload={
                        "status": "completed",
                        "proposal_id": 0,
                        "bbox_3d": [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    },
                    confidence=0.8,
                ),
                raw_state={},
            )

    monkeypatch.setattr(runner, "Stage2DeepResearchAgent", FakeAgent)

    out = runner.run_one_sample(
        "scene0001_00::72",
        "pack_v1",
        pack_v1_inputs_dir=pack_dir,
        embodiedscan_data_root=tmp_path / "embodiedscan",
    )

    assert out["status"] == "completed"
    assert out["predicted_bbox_3d"] == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert out["iou"] == pytest.approx(1.0)


@pytest.mark.integration
def test_pack_v1_failed_marker_scores_zero(monkeypatch, tmp_path) -> None:
    from evaluation.scripts import run_embodiedscan_vg_side_by_side as runner

    pack_dir = _write_pack_v1_inputs(tmp_path)

    class FakeAgent:
        def __init__(self, config):
            assert config.vg_backend == "pack_v1"

        def run(self, task, bundle):
            return SimpleNamespace(
                result=SimpleNamespace(
                    payload={
                        "status": "failed",
                        "proposal_id": -1,
                        "bbox_3d": None,
                    },
                    confidence=0.0,
                ),
                raw_state={},
            )

    monkeypatch.setattr(runner, "Stage2DeepResearchAgent", FakeAgent)

    out = runner.run_one_sample(
        "scene0001_00::72",
        "pack_v1",
        pack_v1_inputs_dir=pack_dir,
        embodiedscan_data_root=tmp_path / "embodiedscan",
    )

    assert out["status"] == "failed"
    assert out["predicted_bbox_3d"] is None
    assert out["iou"] == 0.0


@pytest.mark.integration
def test_legacy_run_one_sample_scores_pilot_prediction(monkeypatch, tmp_path) -> None:
    from evaluation.scripts import run_embodiedscan_vg_side_by_side as runner

    pack_dir = _write_pack_v1_inputs(tmp_path)

    def fake_legacy_pilot(sample_id, *, embodiedscan_data_root, config):
        assert sample_id == "scene0001_00::72"
        assert embodiedscan_data_root == tmp_path / "embodiedscan"
        assert config.vg_backend == "legacy"
        return {
            "prediction": {"bbox_3d": [0, 0, 0, 1, 1, 1, 0, 0, 0]},
            "result": SimpleNamespace(
                raw_state={"vg_selected_bbox_3d": [0, 0, 0, 1, 1, 1, 0, 0, 0]}
            ),
        }

    monkeypatch.setattr(runner, "run_legacy_pilot_sample", fake_legacy_pilot)

    out = runner.run_one_sample(
        "scene0001_00::72",
        "legacy",
        pack_v1_inputs_dir=pack_dir,
        embodiedscan_data_root=tmp_path / "embodiedscan",
    )

    assert out["status"] == "completed"
    assert out["predicted_bbox_3d"] == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert out["iou"] == pytest.approx(1.0)
