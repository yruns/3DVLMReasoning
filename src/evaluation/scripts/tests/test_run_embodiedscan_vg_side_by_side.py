"""Pack-v1 EmbodiedScan VG runner tests (per-scene layout, GT pool)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _write_pack_v1_inputs(tmp_path, *, sample_id: str = "scene0001_00::72"):
    """Write a per-scene pack-v1 layout under ``tmp_path`` (== data_root)."""
    scene_id, target_str = sample_id.split("::")
    target_id = int(target_str)
    data_root = tmp_path
    scene_dir = data_root / scene_id / "pack_v1"
    annotated = scene_dir / "annotated"
    samples = scene_dir / "samples"
    annotated.mkdir(parents=True)
    samples.mkdir(parents=True)

    (annotated / "frame_10.png").write_bytes(b"\x89PNG")
    (scene_dir / "proposals.jsonl").write_text(
        json.dumps(
            {
                "source": "gt",
                "scene_id": scene_id,
                "proposals": [
                    {
                        "id": target_id,
                        "bbox_3d": [0, 0, 0, 1, 1, 1, 0, 0, 0],
                        "score": 1.0,
                        "label": "chair",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "visibility.json").write_text(
        json.dumps({"10": [target_id]}),
        encoding="utf-8",
    )
    sample_path = samples / f"{target_id}.json"
    sample_path.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "target_id": target_id,
                "category": "chair",
                "query": "the chair by the table",
                "gt_bbox_3d_9dof": [0, 0, 0, 1, 1, 1, 0, 0, 0],
                "scene_artifacts_dir": str(scene_dir),
                "source": "gt",
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
    return data_root


@pytest.mark.integration
def test_runner_emits_pack_v1_metrics(monkeypatch, tmp_path) -> None:
    from evaluation.scripts.run_embodiedscan_vg_side_by_side import (
        compare_backends,
    )

    def fake_run_one(sample_id, backend, **_kwargs):
        return {"sample_id": sample_id, "iou": 0.5, "backend": backend}

    monkeypatch.setattr(
        "evaluation.scripts.run_embodiedscan_vg_side_by_side.run_one_sample",
        fake_run_one,
    )
    out = compare_backends(
        sample_ids=["s1", "s2"],
        output_dir=tmp_path,
        data_root=tmp_path / "data_root",
    )
    assert sorted(out) == ["pack_v1"]
    assert out["pack_v1"]["mean_iou"] == pytest.approx(0.5)


@pytest.mark.integration
def test_runner_workers_preserve_sample_order(monkeypatch, tmp_path) -> None:
    from evaluation.scripts.run_embodiedscan_vg_side_by_side import (
        compare_backends,
    )

    def fake_run_one(sample_id, backend, **_kwargs):
        return {
            "sample_id": sample_id,
            "backend": backend,
            "status": "completed",
            "iou": 1.0,
            "predicted_bbox_3d_9dof": [0, 0, 0, 1, 1, 1, 0, 0, 0],
            "gt_bbox_3d_9dof": [0, 0, 0, 1, 1, 1, 0, 0, 0],
            "selected_object_id": int(sample_id.rsplit("::", 1)[1]),
            "confidence": 0.9,
            "query": sample_id,
        }

    monkeypatch.setattr(
        "evaluation.scripts.run_embodiedscan_vg_side_by_side.run_one_sample",
        fake_run_one,
    )
    sample_ids = [f"scene0001_00::{idx}" for idx in range(8)]
    out = compare_backends(
        sample_ids=sample_ids,
        output_dir=tmp_path,
        data_root=tmp_path / "data_root",
        workers=4,
    )

    observed = [r["sample_id"] for r in out["pack_v1"]["per_sample"]]
    assert observed == sample_ids


@pytest.mark.integration
def test_runner_workers_record_sample_errors_and_continue(monkeypatch, tmp_path) -> None:
    from evaluation.scripts.run_embodiedscan_vg_side_by_side import (
        compare_backends,
    )

    def fake_run_one(sample_id, backend, **_kwargs):
        if sample_id.endswith("::1"):
            raise RuntimeError("invalid_prompt")
        return {
            "sample_id": sample_id,
            "backend": backend,
            "status": "completed",
            "iou": 1.0,
        }

    monkeypatch.setattr(
        "evaluation.scripts.run_embodiedscan_vg_side_by_side.run_one_sample",
        fake_run_one,
    )
    sample_ids = [f"scene0001_00::{idx}" for idx in range(3)]

    out = compare_backends(
        sample_ids=sample_ids,
        output_dir=tmp_path,
        data_root=tmp_path / "data_root",
        workers=2,
    )

    per_sample = out["pack_v1"]["per_sample"]
    assert [r["sample_id"] for r in per_sample] == sample_ids
    assert [r["status"] for r in per_sample] == [
        "completed",
        "error",
        "completed",
    ]
    assert per_sample[1]["error_type"] == "RuntimeError"
    assert per_sample[1]["iou"] == 0.0
    assert out["pack_v1"]["Acc@0.50"] == pytest.approx(2 / 3)


@pytest.mark.integration
def test_runner_resumes_from_per_sample_checkpoint(monkeypatch, tmp_path) -> None:
    from evaluation.scripts.run_embodiedscan_vg_side_by_side import (
        compare_backends,
        sample_result_path,
    )

    cached = {
        "sample_id": "scene0001_00::0",
        "backend": "pack_v1",
        "status": "completed",
        "iou": 0.25,
    }
    cached_path = sample_result_path(tmp_path, "pack_v1", cached["sample_id"])
    cached_path.parent.mkdir(parents=True)
    cached_path.write_text(json.dumps(cached), encoding="utf-8")
    calls: list[str] = []

    def fake_run_one(sample_id, backend, **_kwargs):
        calls.append(sample_id)
        return {
            "sample_id": sample_id,
            "backend": backend,
            "status": "completed",
            "iou": 1.0,
        }

    monkeypatch.setattr(
        "evaluation.scripts.run_embodiedscan_vg_side_by_side.run_one_sample",
        fake_run_one,
    )
    sample_ids = ["scene0001_00::0", "scene0001_00::1"]

    out = compare_backends(
        sample_ids=sample_ids,
        output_dir=tmp_path,
        data_root=tmp_path / "data_root",
        workers=2,
    )

    assert calls == ["scene0001_00::1"]
    per_sample = out["pack_v1"]["per_sample"]
    assert [r["sample_id"] for r in per_sample] == sample_ids
    assert [r["iou"] for r in per_sample] == [0.25, 1.0]
    written_path = sample_result_path(tmp_path, "pack_v1", "scene0001_00::1")
    assert json.loads(written_path.read_text(encoding="utf-8"))["iou"] == 1.0
    assert out["pack_v1"]["Acc@0.25"] == pytest.approx(1.0)


@pytest.mark.integration
def test_pack_v1_run_one_sample_scores_agent_bbox(monkeypatch, tmp_path) -> None:
    from evaluation.scripts import run_embodiedscan_vg_side_by_side as runner

    data_root = _write_pack_v1_inputs(tmp_path)

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
                        "selected_object_id": 72,
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
        data_root=data_root,
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


def test_removed_backend_is_rejected(tmp_path) -> None:
    from evaluation.scripts import run_embodiedscan_vg_side_by_side as runner

    data_root = _write_pack_v1_inputs(tmp_path)

    with pytest.raises(ValueError, match="no longer supported"):
        runner.run_one_sample(
            "scene0001_00::72",
            "legacy",
            data_root=data_root,
        )


def test_is_retryable_sample_error_accepts_connection_reset() -> None:
    from evaluation.scripts.run_embodiedscan_vg_side_by_side import (
        is_retryable_sample_error,
    )

    assert is_retryable_sample_error(
        RuntimeError("Error code: 400 - connection reset by peer, code -4201")
    )


@pytest.mark.integration
def test_pack_v1_failed_marker_via_selected_object_id_none_and_status_failed(
    monkeypatch, tmp_path
) -> None:
    from evaluation.scripts import run_embodiedscan_vg_side_by_side as runner

    data_root = _write_pack_v1_inputs(tmp_path)

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
        data_root=data_root,
    )

    assert out["status"] == "failed"
    assert out["predicted_bbox_3d_9dof"] is None
    assert out["iou"] == 0.0


@pytest.mark.integration
def test_pack_v1_completed_payload_without_bbox_raises(monkeypatch, tmp_path) -> None:
    from evaluation.scripts import run_embodiedscan_vg_side_by_side as runner

    data_root = _write_pack_v1_inputs(tmp_path)

    class FakeAgent:
        def __init__(self, config):
            assert config.vg_backend == "pack_v1"

        def run(self, task, bundle):
            return SimpleNamespace(
                result=SimpleNamespace(
                    payload={
                        "status": "completed",
                        "selected_object_id": 72,
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
            data_root=data_root,
        )


@pytest.mark.integration
def test_pack_v1_payload_missing_status_with_bbox_infers_completed(
    monkeypatch, tmp_path
) -> None:
    from evaluation.scripts import run_embodiedscan_vg_side_by_side as runner

    data_root = _write_pack_v1_inputs(tmp_path)

    class FakeAgent:
        def __init__(self, config):
            assert config.vg_backend == "pack_v1"

        def run(self, task, bundle):
            return SimpleNamespace(
                result=SimpleNamespace(
                    payload={
                        "selected_object_id": 72,
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
        data_root=data_root,
    )

    assert out["status"] == "completed"
    assert out["selected_object_id"] == 72
    assert out["iou"] == pytest.approx(1.0)


def test_coerce_bbox_9dof_requires_exactly_nine_floats() -> None:
    from evaluation.scripts.run_embodiedscan_vg_side_by_side import coerce_bbox_9dof

    assert coerce_bbox_9dof(
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        field_name="bbox",
    ) == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert coerce_bbox_9dof(
        "[0, 0, 0, 1, 1, 1, 0, 0, 0]",
        field_name="bbox",
    ) == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="exactly 9 floats"):
        coerce_bbox_9dof([0, 0, 0, 1, 1, 1], field_name="bbox")
    with pytest.raises(ValueError, match="non-finite"):
        coerce_bbox_9dof([0, 0, 0, 1, 1, 1, 0, 0, float("nan")], field_name="bbox")
    with pytest.raises(TypeError, match="serialized list"):
        coerce_bbox_9dof("not a bbox", field_name="bbox")


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

    sample_ids_path = tmp_path / "frozen_sample_ids.json"
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
            "--data-root",
            str(tmp_path / "data_root"),
        ],
    )

    runner.main()

    assert seen == [
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
