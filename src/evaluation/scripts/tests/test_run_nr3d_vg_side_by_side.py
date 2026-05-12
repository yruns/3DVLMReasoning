"""NR3D pack-v1 side-by-side runner tests."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest


def _write_nr3d_pack_inputs(
    tmp_path,
    *,
    sample_id: str = "scannet/scene0001_00::72::A1",
    pack_name: str = "pack_nr3d_v1",
    source: str = "gt",
):
    from evaluation.scripts.run_nr3d_vg_side_by_side import safe_sample_id

    scene_segment, target_text, _assignment = sample_id.split("::")
    scene_id = scene_segment.split("/")[-1]
    target_id = int(target_text)
    scene_dir = tmp_path / scene_id / pack_name
    annotated = scene_dir / "annotated"
    samples = scene_dir / "samples"
    annotated.mkdir(parents=True, exist_ok=True)
    samples.mkdir(parents=True, exist_ok=True)
    (annotated / "frame_10.png").write_bytes(b"\x89PNG")
    (scene_dir / "proposals.jsonl").write_text(
        json.dumps(
            {
                "source": source,
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
    sample_path = samples / f"{safe_sample_id(sample_id)}.json"
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
                "source": source,
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
    return tmp_path


def test_parse_nr3d_sample_id_accepts_three_segments() -> None:
    from evaluation.scripts.run_nr3d_vg_side_by_side import parse_nr3d_sample_id

    parsed = parse_nr3d_sample_id("scannet/scene0001_00::72::A1")

    assert parsed.scene_id == "scene0001_00"
    assert parsed.scan_id == "scannet/scene0001_00"
    assert parsed.target_id == 72
    assert parsed.assignment_id == "A1"


def test_expected_pack_sample_path_uses_safe_three_segment_filename(tmp_path) -> None:
    from evaluation.scripts.run_nr3d_vg_side_by_side import expected_pack_sample_path

    assert expected_pack_sample_path(
        tmp_path,
        "scannet/scene0001_00::72::A1",
    ) == (
        tmp_path
        / "scene0001_00"
        / "pack_nr3d_v1"
        / "samples"
        / "scannet__scene0001_00__72__A1.json"
    )


def test_sample_result_path_namespaces_by_nr3d_pack(tmp_path) -> None:
    from evaluation.scripts.run_nr3d_vg_side_by_side import sample_result_path

    path = sample_result_path(
        tmp_path,
        "pack_v1",
        "scannet/scene0001_00::72::A1",
    )

    assert path.parent == tmp_path / "per_sample" / "pack_nr3d_v1"
    assert path.name.startswith("scannet__scene0001_00__72__A1_")


def test_run_one_sample_scores_agent_bbox(monkeypatch, tmp_path) -> None:
    from evaluation.scripts import run_nr3d_vg_side_by_side as runner

    data_root = _write_nr3d_pack_inputs(tmp_path)

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
                )
            )

    monkeypatch.setattr(runner, "Stage2DeepResearchAgent", FakeAgent)
    monkeypatch.setattr(
        runner,
        "build_pack_v1_bundle",
        lambda **kwargs: SimpleNamespace(scene_id=kwargs["scene_id"]),
    )

    out = runner.run_one_sample(
        "scannet/scene0001_00::72::A1",
        "pack_v1",
        data_root=data_root,
    )

    assert out["status"] == "completed"
    assert out["iou"] == pytest.approx(1.0)
    assert out["selected_object_id"] == 72


def test_run_one_sample_preserves_tool_trace(monkeypatch, tmp_path) -> None:
    from evaluation.scripts import run_nr3d_vg_side_by_side as runner

    data_root = _write_nr3d_pack_inputs(tmp_path)

    class FakeToolTrace:
        def model_dump(self):
            return {
                "tool_name": "view_keyframe_marked",
                "tool_input": {"frame_id": 10},
                "response_text": "saw chair",
            }

    class FakeAgent:
        def __init__(self, config):
            pass

        def run(self, task, bundle):
            return SimpleNamespace(
                result=SimpleNamespace(
                    payload={
                        "status": "completed",
                        "selected_object_id": 72,
                        "bbox_3d": [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    },
                    confidence=0.8,
                ),
                tool_trace=[
                    FakeToolTrace(),
                    {
                        "tool_name": "submit_final",
                        "tool_input": {"selected_object_id": 72},
                        "response_text": "accepted",
                    },
                ],
            )

    monkeypatch.setattr(runner, "Stage2DeepResearchAgent", FakeAgent)
    monkeypatch.setattr(
        runner,
        "build_pack_v1_bundle",
        lambda **kwargs: SimpleNamespace(scene_id=kwargs["scene_id"]),
    )

    out = runner.run_one_sample(
        "scannet/scene0001_00::72::A1",
        "pack_v1",
        data_root=data_root,
    )

    assert out["tool_trace"] == [
        {
            "tool_name": "view_keyframe_marked",
            "tool_input": {"frame_id": 10},
            "response_text": "saw chair",
        },
        {
            "tool_name": "submit_final",
            "tool_input": {"selected_object_id": 72},
            "response_text": "accepted",
        },
    ]


def test_extract_result_tool_trace_accepts_raw_dict() -> None:
    from evaluation.scripts import run_nr3d_vg_side_by_side as runner

    trace = runner.extract_result_tool_trace(
        {
            "tool_trace": [
                {
                    "tool_name": "view_keyframe_marked",
                    "tool_input": {"frame_id": 10},
                    "response_text": "ok",
                },
                {
                    "tool_name": "submit_final",
                    "tool_input": {"selected_object_id": SimpleNamespace(value=72)},
                    "response_text": "accepted",
                },
            ]
        }
    )

    assert trace == [
        {
            "tool_name": "view_keyframe_marked",
            "tool_input": {"frame_id": 10},
            "response_text": "ok",
        },
        {
            "tool_name": "submit_final",
            "tool_input": {"selected_object_id": "namespace(value=72)"},
            "response_text": "accepted",
        },
    ]


def test_compare_backends_writes_side_by_side_and_checkpoints(
    monkeypatch, tmp_path
) -> None:
    from evaluation.scripts import run_nr3d_vg_side_by_side as runner

    sample_ids = [
        "scannet/scene0001_00::72::A1",
        "scannet/scene0001_00::73::A2",
    ]
    data_root = _write_nr3d_pack_inputs(tmp_path / "data_root", sample_id=sample_ids[0])
    _write_nr3d_pack_inputs(tmp_path / "data_root", sample_id=sample_ids[1])

    def fake_run_one(sample_id, backend, **_kwargs):
        return {
            "sample_id": sample_id,
            "backend": backend,
            "status": "completed",
            "iou": 0.5,
        }

    monkeypatch.setattr(runner, "run_one_sample", fake_run_one)

    out = runner.compare_backends(
        sample_ids=sample_ids,
        output_dir=tmp_path / "out",
        data_root=data_root,
        workers=1,
    )

    assert out["pack_v1"]["n"] == 2
    assert out["pack_v1"]["Acc@0.25"] == pytest.approx(1.0)
    assert (tmp_path / "out" / "side_by_side.json").exists()
    assert (
        len(list((tmp_path / "out" / "per_sample" / "pack_nr3d_v1").glob("*.json")))
        == 2
    )


def test_compare_backends_checkpoint_only_respects_max_new_samples(
    monkeypatch, tmp_path
) -> None:
    from evaluation.scripts import run_nr3d_vg_side_by_side as runner

    sample_ids = [
        "scannet/scene0001_00::72::A1",
        "scannet/scene0001_00::73::A2",
        "scannet/scene0001_00::74::A3",
    ]
    calls = []

    def fake_run_one(sample_id, backend, **_kwargs):
        calls.append(sample_id)
        return {
            "sample_id": sample_id,
            "backend": backend,
            "status": "completed",
            "iou": 1.0,
        }

    monkeypatch.setattr(runner, "run_one_sample", fake_run_one)
    monkeypatch.setattr(runner, "preflight_pack_sample_exists", lambda *a, **kw: None)

    result = runner.compare_backends(
        sample_ids=sample_ids,
        output_dir=tmp_path / "out",
        data_root=tmp_path,
        workers=1,
        return_results=False,
        write_side_by_side=False,
        max_new_samples=1,
    )

    assert result is None
    assert calls == [sample_ids[0]]
    assert not (tmp_path / "out" / "side_by_side.json").exists()


def test_build_backend_payload_from_checkpoints_preserves_requested_order(
    tmp_path,
) -> None:
    from evaluation.scripts import run_nr3d_vg_side_by_side as runner

    output_dir = tmp_path / "out"
    first = "scannet/scene0001_00::72::A1"
    second = "scannet/scene0001_00::73::A2"
    for sample_id, iou in [(second, 0.1), (first, 0.6)]:
        runner.write_sample_result_checkpoint(
            output_dir,
            "pack_v1",
            {
                "sample_id": sample_id,
                "backend": "pack_v1",
                "status": "completed",
                "iou": iou,
            },
        )

    payload = runner.build_backend_payload_from_checkpoints(
        [first, second],
        output_dir,
        "pack_v1",
    )
    summary_only = runner.build_backend_payload_from_checkpoints(
        [first, second],
        output_dir,
        "pack_v1",
        include_per_sample=False,
    )

    assert [row["sample_id"] for row in payload["per_sample"]] == [first, second]
    assert payload["n"] == 2
    assert payload["mean_iou"] == pytest.approx(0.35)
    assert payload["Acc@0.25"] == pytest.approx(0.5)
    assert payload["Acc@0.50"] == pytest.approx(0.5)
    assert "per_sample" not in summary_only


def test_build_backend_payload_from_checkpoints_requires_all_requested(
    tmp_path,
) -> None:
    from evaluation.scripts import run_nr3d_vg_side_by_side as runner

    sample_id = "scannet/scene0001_00::72::A1"

    with pytest.raises(FileNotFoundError, match="Missing per-sample checkpoint"):
        runner.build_backend_payload_from_checkpoints(
            [sample_id],
            tmp_path / "out",
            "pack_v1",
        )


def test_compare_backends_streams_side_by_side_from_existing_checkpoints(
    monkeypatch,
    tmp_path,
) -> None:
    from evaluation.scripts import run_nr3d_vg_side_by_side as runner

    sample_ids = [
        "scannet/scene0001_00::72::A1",
        "scannet/scene0001_00::73::A2",
    ]
    output_dir = tmp_path / "out"
    for sample_id, iou in [(sample_ids[1], 0.1), (sample_ids[0], 0.6)]:
        runner.write_sample_result_checkpoint(
            output_dir,
            "pack_v1",
            {
                "sample_id": sample_id,
                "backend": "pack_v1",
                "status": "completed",
                "iou": iou,
            },
        )

    def fail_run_one(*_args, **_kwargs):
        raise AssertionError("run_one_sample should not be called")

    monkeypatch.setattr(runner, "run_one_sample", fail_run_one)
    monkeypatch.setattr(runner, "preflight_pack_sample_exists", lambda *a, **kw: None)

    result = runner.compare_backends(
        sample_ids=sample_ids,
        output_dir=output_dir,
        data_root=tmp_path,
        workers=1,
        return_results=False,
        write_side_by_side=True,
        max_new_samples=0,
    )

    assert result is None
    payload = json.loads((output_dir / "side_by_side.json").read_text(encoding="utf-8"))
    per_sample = payload["pack_v1"]["per_sample"]
    assert [row["sample_id"] for row in per_sample] == sample_ids
    assert [row["iou"] for row in per_sample] == [0.6, 0.1]


def test_compare_backends_persists_failed_sentinel_on_sample_exception(
    monkeypatch, tmp_path
) -> None:
    from evaluation.scripts import run_nr3d_vg_side_by_side as runner

    sample_id = "scannet/scene0001_00::72::A1"
    data_root = _write_nr3d_pack_inputs(tmp_path / "data_root", sample_id=sample_id)

    def fake_run_one(sample_id, backend, **_kwargs):
        raise RuntimeError("endpoint unreachable")

    monkeypatch.setattr(runner, "run_one_sample", fake_run_one)

    out = runner.compare_backends(
        sample_ids=[sample_id],
        output_dir=tmp_path / "out",
        data_root=data_root,
    )

    # One rotten sample no longer kills the whole run; instead the per-sample
    # record is a 'failed' sentinel with iou=0 and the error string preserved.
    assert out["pack_v1"]["n"] == 1
    assert out["pack_v1"]["Acc@0.25"] == 0.0
    assert out["pack_v1"]["mean_iou"] == 0.0
    only = out["pack_v1"]["per_sample"][0]
    assert only["status"] == "failed"
    assert only["iou"] == 0.0
    assert only["selected_object_id"] is None
    assert "endpoint unreachable" in only["error"]
    assert "RuntimeError" in only["error"]
    # The sentinel is also written to the per-sample checkpoint dir so resumes
    # do not re-attempt the rotten sample.
    ckpts = list((tmp_path / "out" / "per_sample" / "pack_nr3d_v1").glob("*.json"))
    assert len(ckpts) == 1


def test_main_wires_all_guard_flags(tmp_path, monkeypatch) -> None:
    from evaluation.scripts import run_nr3d_vg_side_by_side as runner

    sample_ids = tmp_path / "samples.json"
    sample_ids.write_text("[]", encoding="utf-8")
    captured = {}

    def fake_compare_backends(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(runner, "compare_backends", fake_compare_backends)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_nr3d_vg_side_by_side",
            "--sample-ids",
            str(sample_ids),
            "--output-dir",
            str(tmp_path / "out"),
            "--data-root",
            str(tmp_path),
            "--checkpoint-only",
            "--max-new-samples",
            "3",
            "--use-tool-answer-disagreement-gate",
            "--use-no-match-candidate-guard",
            "--use-evidence-frame-guard",
        ],
    )

    runner.main()

    config = captured["config"]
    assert config.use_tool_answer_disagreement_gate is True
    assert config.use_no_match_candidate_guard is True
    assert config.use_evidence_frame_guard is True
    assert captured["return_results"] is False
    assert captured["write_side_by_side"] is False
    assert captured["max_new_samples"] == 3


def test_load_sample_ids_accepts_strings_or_dicts(tmp_path) -> None:
    from evaluation.scripts.run_nr3d_vg_side_by_side import load_sample_ids

    strings = tmp_path / "strings.json"
    strings.write_text(json.dumps(["scannet/scene0001_00::72::A1"]), encoding="utf-8")
    assert load_sample_ids(strings) == ["scannet/scene0001_00::72::A1"]

    dicts = tmp_path / "dicts.json"
    dicts.write_text(
        json.dumps([{"sample_id": "scannet/scene0001_00::73::A2"}]),
        encoding="utf-8",
    )
    assert load_sample_ids(dicts) == ["scannet/scene0001_00::73::A2"]
