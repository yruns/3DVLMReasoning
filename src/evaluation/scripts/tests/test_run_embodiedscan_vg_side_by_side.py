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
                        "selected_object_id": 0,
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
    assert out["predicted_bbox_3d_9dof"] == [
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert out["gt_bbox_3d_9dof"] == [
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert out["iou"] == pytest.approx(1.0)


@pytest.mark.integration
def test_pack_v1_failed_marker_via_selected_object_id_none_and_status_failed(
    monkeypatch, tmp_path
) -> None:
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
                        "selected_object_id": None,
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
    assert out["predicted_bbox_3d_9dof"] is None
    assert out["iou"] == 0.0


@pytest.mark.integration
def test_pack_v1_completed_payload_without_bbox_raises(monkeypatch, tmp_path) -> None:
    from evaluation.scripts import run_embodiedscan_vg_side_by_side as runner

    pack_dir = _write_pack_v1_inputs(tmp_path)

    class FakeAgent:
        def __init__(self, config):
            assert config.vg_backend == "pack_v1"

        def run(self, task, bundle):
            return SimpleNamespace(
                result=SimpleNamespace(
                    payload={
                        "status": "completed",
                        "selected_object_id": 0,
                        "bbox_3d": None,
                    },
                    confidence=0.8,
                ),
                raw_state={},
            )

    monkeypatch.setattr(runner, "Stage2DeepResearchAgent", FakeAgent)

    with pytest.raises(ValueError, match="pack_v1 payload missing bbox_3d"):
        runner.run_one_sample(
            "scene0001_00::72",
            "pack_v1",
            pack_v1_inputs_dir=pack_dir,
            embodiedscan_data_root=tmp_path / "embodiedscan",
        )


@pytest.mark.integration
def test_pack_v1_payload_missing_status_raises(monkeypatch, tmp_path) -> None:
    from evaluation.scripts import run_embodiedscan_vg_side_by_side as runner

    pack_dir = _write_pack_v1_inputs(tmp_path)

    class FakeAgent:
        def __init__(self, config):
            assert config.vg_backend == "pack_v1"

        def run(self, task, bundle):
            return SimpleNamespace(
                result=SimpleNamespace(
                    payload={
                        "selected_object_id": 0,
                        "bbox_3d": [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    },
                    confidence=0.8,
                ),
                raw_state={},
            )

    monkeypatch.setattr(runner, "Stage2DeepResearchAgent", FakeAgent)

    with pytest.raises(ValueError, match="pack_v1 payload missing 'status' field"):
        runner.run_one_sample(
            "scene0001_00::72",
            "pack_v1",
            pack_v1_inputs_dir=pack_dir,
            embodiedscan_data_root=tmp_path / "embodiedscan",
        )


def test_coerce_bbox_9dof_requires_exactly_nine_floats() -> None:
    from evaluation.scripts.run_embodiedscan_vg_side_by_side import coerce_bbox_9dof

    assert coerce_bbox_9dof(
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        field_name="bbox",
    ) == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="exactly 9 floats"):
        coerce_bbox_9dof([0, 0, 0, 1, 1, 1], field_name="bbox")
    with pytest.raises(ValueError, match="non-finite"):
        coerce_bbox_9dof([0, 0, 0, 1, 1, 1, 0, 0, float("nan")], field_name="bbox")


def test_load_sample_ids_accepts_string_list_or_dict_list(tmp_path) -> None:
    from evaluation.scripts.run_embodiedscan_vg_side_by_side import load_sample_ids

    string_path = tmp_path / "strings.json"
    string_path.write_text(json.dumps(["scene0001_00::72"]), encoding="utf-8")
    assert load_sample_ids(string_path) == ["scene0001_00::72"]

    dict_path = tmp_path / "dicts.json"
    dict_path.write_text(
        json.dumps([{"sample_id": "scene0002_00::4", "category": "chair"}]),
        encoding="utf-8",
    )
    assert load_sample_ids(dict_path) == ["scene0002_00::4"]

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps([{"scene_id": "scene0003_00"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="sample_id"):
        load_sample_ids(bad_path)


@pytest.mark.integration
def test_main_accepts_extractor_dict_sample_ids_json(monkeypatch, tmp_path) -> None:
    from evaluation.scripts import run_embodiedscan_vg_side_by_side as runner

    sample_ids_path = tmp_path / "batch30_sample_ids.json"
    sample_ids_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "scene0001_00::72",
                    "scene_id": "scene0001_00",
                    "target_id": 72,
                    "category": "chair",
                }
            ]
        ),
        encoding="utf-8",
    )
    seen: list[tuple[str, str]] = []

    def fake_run_one(sample_id, backend, **_kwargs):
        seen.append((sample_id, backend))
        return {"sample_id": sample_id, "backend": backend, "iou": 0.5}

    monkeypatch.setattr(runner, "run_one_sample", fake_run_one)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_embodiedscan_vg_side_by_side.py",
            "--sample-ids",
            str(sample_ids_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--pack-v1-inputs-dir",
            str(tmp_path / "pack_v1"),
            "--embodiedscan-data-root",
            str(tmp_path / "embodiedscan"),
        ],
    )

    runner.main()

    assert seen == [
        ("scene0001_00::72", "legacy"),
        ("scene0001_00::72", "pack_v1"),
    ]


def test_extract_result_payload_requires_result_payload_shape() -> None:
    from evaluation.scripts.run_embodiedscan_vg_side_by_side import (
        extract_result_payload,
    )

    payload = {"status": "completed", "selected_object_id": 0, "bbox_3d": []}
    assert (
        extract_result_payload(SimpleNamespace(result=SimpleNamespace(payload=payload)))
        is payload
    )
    with pytest.raises(ValueError, match="result.payload"):
        extract_result_payload({"payload": payload})


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
    assert out["predicted_bbox_3d_9dof"] == [
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert out["iou"] == pytest.approx(1.0)
