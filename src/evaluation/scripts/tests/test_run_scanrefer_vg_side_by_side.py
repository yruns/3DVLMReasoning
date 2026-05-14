"""Smoke tests for ScanRefer side-by-side runner."""

from __future__ import annotations

import gzip
import json
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_parse_sample_id_canonical():
    from evaluation.scripts.run_scanrefer_vg_side_by_side import (
        parse_scanrefer_sample_id,
    )

    parsed = parse_scanrefer_sample_id("scannet/scene0088_00::5::3")
    assert parsed.scan_id == "scannet/scene0088_00"
    assert parsed.scene_id == "scene0088_00"
    assert parsed.target_id == 5
    assert parsed.ann_id == "3"


def test_parse_sample_id_rejects_malformed():
    from evaluation.scripts.run_scanrefer_vg_side_by_side import (
        parse_scanrefer_sample_id,
    )

    with pytest.raises(ValueError, match="format"):
        parse_scanrefer_sample_id("scannet/scene_x::5")


def test_safe_sample_id():
    from evaluation.scripts.run_scanrefer_vg_side_by_side import safe_sample_id

    assert safe_sample_id("scannet/scene_a::5::3") == "scannet__scene_a__5__3"


def test_clean_keyframe_image_path_rewrites_annotated_scanrefer(tmp_path):
    from evaluation.scripts.run_scanrefer_vg_side_by_side import (
        clean_keyframe_image_path,
    )

    scene_id = "scene0001_00"
    raw = tmp_path / scene_id / "raw"
    raw.mkdir(parents=True)
    (raw / "000040-rgb.png").write_bytes(b"\x89PNG")
    (raw / "scene_info.json").write_text(
        json.dumps({"kept_frame_ids": [0, 10, 20, 30, 40]}),
        encoding="utf-8",
    )

    assert clean_keyframe_image_path(
        raw_frames_root=tmp_path,
        scene_id=scene_id,
        image_path=str(tmp_path / scene_id / "pack" / "annotated" / "frame_4.png"),
        frame_id=4,
    ) == str(raw / "000040-rgb.png")


def test_compare_backends_persists_failed_sentinel_on_sample_exception(
    tmp_path, monkeypatch
):
    """Mirror of NR3D test: a per-sample exception → failed sentinel, not crash."""
    from evaluation.scripts import run_scanrefer_vg_side_by_side as mod

    def fake_run_one_sample(sample_id, backend, **kwargs):
        if sample_id.endswith("::3"):
            raise RuntimeError("simulated upstream 500")
        return {
            "sample_id": sample_id,
            "backend": backend,
            "status": "completed",
            "iou": 1.0,
            "predicted_bbox_3d_9dof": [0] * 9,
            "gt_bbox_3d_9dof": [0] * 9,
            "selected_object_id": 0,
            "confidence": 0.9,
            "query": "test",
        }

    monkeypatch.setattr(mod, "run_one_sample", fake_run_one_sample)
    monkeypatch.setattr(mod, "validate_unique_sample_ids", lambda _: None)
    monkeypatch.setattr(mod, "preflight_pack_sample_exists", lambda *a, **kw: None)

    out = tmp_path / "out"
    sample_ids = ["scannet/scene_a::1::0", "scannet/scene_a::2::3"]
    results = mod.compare_backends(
        sample_ids=sample_ids,
        output_dir=out,
        data_root=tmp_path,
        pack_name="pack_scanrefer_v1",
        workers=1,
    )
    per_sample = results["pack_v1"]["per_sample"]
    assert len(per_sample) == 2
    failed = [r for r in per_sample if r["sample_id"].endswith("::3")][0]
    assert failed["status"] == "failed"
    assert failed["iou"] == 0.0
    assert "simulated upstream 500" in failed["error"]


def test_compare_backends_low_memory_mode_resumes_from_checkpoints(
    tmp_path, monkeypatch
):
    """CLI-scale runs should not keep all cached per-sample traces in RAM."""
    from evaluation.scripts import run_scanrefer_vg_side_by_side as mod

    sample_ids = ["scannet/scene_a::1::0", "scannet/scene_a::2::0"]
    out = tmp_path / "out"
    cached = {
        "sample_id": sample_ids[0],
        "backend": "pack_v1",
        "status": "completed",
        "iou": 0.25,
        "predicted_bbox_3d_9dof": [0] * 9,
        "gt_bbox_3d_9dof": [0] * 9,
        "selected_object_id": 1,
        "confidence": 0.8,
        "query": "cached",
        "tool_trace": [{"response_text": "large cached trace"}],
    }
    mod.write_sample_result_checkpoint(
        out,
        "pack_v1",
        cached,
        pack_name="pack_scanrefer_v1",
    )
    calls = []

    def fake_run_one_sample(sample_id, backend, **kwargs):
        calls.append(sample_id)
        return {
            "sample_id": sample_id,
            "backend": backend,
            "status": "completed",
            "iou": 1.0,
            "predicted_bbox_3d_9dof": [1] * 9,
            "gt_bbox_3d_9dof": [1] * 9,
            "selected_object_id": 2,
            "confidence": 0.9,
            "query": "fresh",
            "tool_trace": [{"response_text": "fresh trace"}],
        }

    monkeypatch.setattr(mod, "run_one_sample", fake_run_one_sample)
    monkeypatch.setattr(mod, "validate_unique_sample_ids", lambda _: None)
    monkeypatch.setattr(mod, "preflight_pack_sample_exists", lambda *a, **kw: None)

    result = mod.compare_backends(
        sample_ids=sample_ids,
        output_dir=out,
        data_root=tmp_path,
        pack_name="pack_scanrefer_v1",
        workers=1,
        return_results=False,
    )

    assert result is None
    assert calls == [sample_ids[1]]
    payload = __import__("json").loads((out / "side_by_side.json").read_text())
    per_sample = payload["pack_v1"]["per_sample"]
    assert [r["sample_id"] for r in per_sample] == sample_ids
    assert [r["iou"] for r in per_sample] == [0.25, 1.0]


def test_main_uses_low_memory_compare_mode(tmp_path, monkeypatch) -> None:
    from evaluation.scripts import run_scanrefer_vg_side_by_side as mod

    sample_ids = tmp_path / "samples.json"
    sample_ids.write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "out"
    captured = {}

    def fake_compare_backends(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(mod, "compare_backends", fake_compare_backends)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scanrefer_vg_side_by_side",
            "--sample-ids",
            str(sample_ids),
            "--output-dir",
            str(output_dir),
            "--data-root",
            str(tmp_path),
        ],
    )

    mod.main()

    assert captured["return_results"] is False


def test_compare_backends_checkpoint_only_respects_max_new_samples(
    tmp_path, monkeypatch
) -> None:
    from evaluation.scripts import run_scanrefer_vg_side_by_side as mod

    sample_ids = [
        "scannet/scene_a::1::0",
        "scannet/scene_a::2::0",
        "scannet/scene_a::3::0",
    ]
    calls = []

    def fake_run_one_sample(sample_id, backend, **kwargs):
        calls.append(sample_id)
        return {
            "sample_id": sample_id,
            "backend": backend,
            "status": "completed",
            "iou": 1.0,
            "predicted_bbox_3d_9dof": [1] * 9,
            "gt_bbox_3d_9dof": [1] * 9,
            "selected_object_id": 2,
            "confidence": 0.9,
            "query": "fresh",
        }

    monkeypatch.setattr(mod, "run_one_sample", fake_run_one_sample)
    monkeypatch.setattr(mod, "validate_unique_sample_ids", lambda _: None)
    monkeypatch.setattr(mod, "preflight_pack_sample_exists", lambda *a, **kw: None)

    result = mod.compare_backends(
        sample_ids=sample_ids,
        output_dir=tmp_path / "out",
        data_root=tmp_path,
        pack_name="pack_scanrefer_v1",
        workers=1,
        return_results=False,
        write_side_by_side=False,
        max_new_samples=1,
    )

    assert result is None
    assert calls == [sample_ids[0]]
    assert not (tmp_path / "out" / "side_by_side.json").exists()


def test_main_wires_checkpoint_only_batch_flags(tmp_path, monkeypatch) -> None:
    from evaluation.scripts import run_scanrefer_vg_side_by_side as mod

    sample_ids = tmp_path / "samples.json"
    sample_ids.write_text("[]", encoding="utf-8")
    captured = {}

    def fake_compare_backends(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(mod, "compare_backends", fake_compare_backends)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scanrefer_vg_side_by_side",
            "--sample-ids",
            str(sample_ids),
            "--output-dir",
            str(tmp_path / "out"),
            "--data-root",
            str(tmp_path),
            "--checkpoint-only",
            "--max-new-samples",
            "7",
        ],
    )

    mod.main()

    assert captured["return_results"] is False
    assert captured["write_side_by_side"] is False
    assert captured["max_new_samples"] == 7


def test_callback_keyframe_selector_uses_scanrefer_visibility_stride(
    tmp_path,
    monkeypatch,
):
    import query_scene.keyframe_selector as keyframe_selector_mod
    from evaluation.scripts import run_scanrefer_vg_side_by_side as mod

    mod._SELECTOR_CACHE.clear()
    mod._SELECTOR_BUILD_LOCKS.clear()
    cg_root = tmp_path / "scene_a" / "conceptgraph"
    cg_root.mkdir(parents=True)
    (cg_root / "enriched_objects.json").write_text("{}", encoding="utf-8")
    captured = {}
    fake_selector = object()

    class _FakeKeyframeSelector:
        @staticmethod
        def from_scene_path(path, *, stride, llm_model):
            captured["path"] = path
            captured["stride"] = stride
            captured["llm_model"] = llm_model
            return fake_selector

    monkeypatch.setattr(
        keyframe_selector_mod,
        "KeyframeSelector",
        _FakeKeyframeSelector,
    )

    selector = mod._get_or_build_keyframe_selector(
        "scene_a",
        tmp_path,
        llm_model="test-model",
    )

    assert selector is fake_selector
    assert captured["path"] == str(cg_root)
    assert captured["stride"] == 1
    assert captured["llm_model"] == "test-model"


def test_callback_keyframe_selector_cache_evicts_least_recent_scene(
    tmp_path,
    monkeypatch,
):
    import query_scene.keyframe_selector as keyframe_selector_mod
    from evaluation.scripts import run_scanrefer_vg_side_by_side as mod

    mod._SELECTOR_CACHE.clear()
    mod._SELECTOR_BUILD_LOCKS.clear()
    monkeypatch.setattr(mod, "_MAX_SELECTOR_CACHE_SIZE", 2)
    for scene_id in ("scene_a", "scene_b", "scene_c"):
        cg_root = tmp_path / scene_id / "conceptgraph"
        cg_root.mkdir(parents=True)
        (cg_root / "enriched_objects.json").write_text("{}", encoding="utf-8")

    class _FakeKeyframeSelector:
        @staticmethod
        def from_scene_path(path, *, stride, llm_model):
            return {"path": path, "stride": stride, "llm_model": llm_model}

    monkeypatch.setattr(
        keyframe_selector_mod,
        "KeyframeSelector",
        _FakeKeyframeSelector,
    )

    mod._get_or_build_keyframe_selector("scene_a", tmp_path)
    mod._get_or_build_keyframe_selector("scene_b", tmp_path)
    mod._get_or_build_keyframe_selector("scene_a", tmp_path)
    mod._get_or_build_keyframe_selector("scene_c", tmp_path)

    assert list(mod._SELECTOR_CACHE) == ["scene_a", "scene_c"]
    assert "scene_b" not in mod._SELECTOR_CACHE


def test_attach_conceptgraph_frame_views_to_pool_matches_by_3d_bbox(tmp_path):
    from evaluation.scripts import run_scanrefer_vg_side_by_side as mod

    mod._CONCEPTGRAPH_OBJECT_CACHE.clear()
    scene_dir = tmp_path / "scene0000_00"
    pcd_dir = scene_dir / "conceptgraph" / "pcd_saves"
    pcd_dir.mkdir(parents=True)
    (scene_dir / "raw").mkdir()
    raw = {
        "objects": [
            {
                "bbox_np": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                "image_idx": [57],
                "xyxy": [np.array([10.2, 20.6, 30.1, 40.4], dtype=np.float32)],
                "conf": [0.9],
            }
        ]
    }
    with gzip.open(pcd_dir / "full_pcd_gt_axisaligned_post.pkl.gz", "wb") as fh:
        pickle.dump(raw, fh)

    pool = {
        "proposals": [
            {
                "id": 10,
                "bbox_3d_9dof": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0, 0, 0],
                "category": "shelf",
                "score": 1.0,
            }
        ]
    }

    mod._attach_conceptgraph_frame_views_to_pool(
        pool,
        scene_id="scene0000_00",
        phase8_data_root=tmp_path,
    )

    frame_view = pool["proposals"][0]["frame_views"]["57"]
    assert frame_view["proposal_id"] == 10
    assert frame_view["frame_id"] == 57
    assert frame_view["bbox_2d"] == [10, 21, 30, 40]
    assert frame_view["raw_rgb_path"].endswith("scene0000_00/raw/000570-rgb.png")


def test_conceptgraph_object_cache_evicts_least_recent_scene(tmp_path, monkeypatch):
    from evaluation.scripts import run_scanrefer_vg_side_by_side as mod

    mod._CONCEPTGRAPH_OBJECT_CACHE.clear()
    monkeypatch.setattr(mod, "_MAX_CONCEPTGRAPH_OBJECT_CACHE_SIZE", 2)

    mod._load_conceptgraph_objects_for_frame_views("scene_a", tmp_path)
    mod._load_conceptgraph_objects_for_frame_views("scene_b", tmp_path)
    mod._load_conceptgraph_objects_for_frame_views("scene_a", tmp_path)
    mod._load_conceptgraph_objects_for_frame_views("scene_c", tmp_path)

    assert list(mod._CONCEPTGRAPH_OBJECT_CACHE) == [
        (str(tmp_path), "scene_a"),
        (str(tmp_path), "scene_c"),
    ]
    assert (str(tmp_path), "scene_b") not in mod._CONCEPTGRAPH_OBJECT_CACHE


def test_scanrefer_pack_wires_crop_callback(monkeypatch):
    """v9: only the crop callback survives the Stage-1 → tools migration."""
    import agents.stage1_callbacks as callbacks
    from evaluation.scripts import run_scanrefer_vg_side_by_side as mod

    selector = object()
    captured: dict = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["agent_kwargs"] = kwargs

        def run(self, *, task, bundle):
            captured["task"] = task
            captured["bundle"] = bundle
            return "ok"

    monkeypatch.setattr(mod, "Stage2DeepResearchAgent", FakeAgent)
    monkeypatch.setattr(
        mod, "_get_or_build_keyframe_selector", lambda *a, **kw: selector
    )
    monkeypatch.setattr(
        mod,
        "build_pack_v1_bundle_from_sample",
        lambda *a, **kw: SimpleNamespace(),
    )
    monkeypatch.setattr(callbacks, "create_crop_callback", lambda *a, **kw: "crop")

    result = mod.run_pack_v1_sample(
        {"scene_id": "scene0558_00", "query": "the chair"},
        data_root=SimpleNamespace(),
        config=SimpleNamespace(),
    )

    assert result == "ok"
    assert captured["agent_kwargs"]["crop_callback"] == "crop"
    assert "more_views_callback" not in captured["agent_kwargs"]
    assert "hypothesis_callback" not in captured["agent_kwargs"]


def test_extract_pack_v1_prediction_resolves_structured_proposal_payload():
    from evaluation.scripts.run_scanrefer_vg_side_by_side import (
        extract_pack_v1_prediction,
    )

    result = SimpleNamespace(
        result=SimpleNamespace(payload={"proposal_id": 7, "confidence": 0.8}),
        final_bundle=SimpleNamespace(
            extra_metadata={
                "vg_proposal_pool": {
                    "proposals": [
                        {"id": 7, "bbox_3d_9dof": [1, 2, 3, 4, 5, 6, 0, 0, 0]},
                    ],
                },
            },
        ),
    )

    prediction = extract_pack_v1_prediction(result)

    assert prediction["status"] == "completed"
    assert prediction["selected_object_id"] == 7
    assert prediction["bbox_3d"] == [1, 2, 3, 4, 5, 6, 0, 0, 0]
    assert prediction["confidence"] == 0.8


def test_main_wires_use_clip_visible_aug_flag(tmp_path, monkeypatch) -> None:
    from evaluation.scripts import run_scanrefer_vg_side_by_side as mod

    sample_ids = tmp_path / "samples.json"
    sample_ids.write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "out"
    captured = {}

    def fake_compare_backends(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(mod, "compare_backends", fake_compare_backends)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scanrefer_vg_side_by_side",
            "--sample-ids",
            str(sample_ids),
            "--output-dir",
            str(output_dir),
            "--data-root",
            str(tmp_path),
            "--use-clip-visible-aug",
        ],
    )

    mod.main()

    assert captured["config"].use_clip_visible_aug is True


def test_main_wires_no_match_candidate_guard_flag(tmp_path, monkeypatch) -> None:
    from evaluation.scripts import run_scanrefer_vg_side_by_side as mod

    sample_ids = tmp_path / "samples.json"
    sample_ids.write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "out"
    captured = {}

    def fake_compare_backends(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(mod, "compare_backends", fake_compare_backends)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scanrefer_vg_side_by_side",
            "--sample-ids",
            str(sample_ids),
            "--output-dir",
            str(output_dir),
            "--data-root",
            str(tmp_path),
            "--use-no-match-candidate-guard",
        ],
    )

    mod.main()

    assert captured["config"].use_no_match_candidate_guard is True


def test_main_wires_evidence_frame_guard_flag(tmp_path, monkeypatch) -> None:
    from evaluation.scripts import run_scanrefer_vg_side_by_side as mod

    sample_ids = tmp_path / "samples.json"
    sample_ids.write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "out"
    captured = {}

    def fake_compare_backends(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(mod, "compare_backends", fake_compare_backends)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scanrefer_vg_side_by_side",
            "--sample-ids",
            str(sample_ids),
            "--output-dir",
            str(output_dir),
            "--data-root",
            str(tmp_path),
            "--use-evidence-frame-guard",
        ],
    )

    mod.main()

    assert captured["config"].use_evidence_frame_guard is True


def test_is_retryable_sample_error_accepts_504_internal_code_4307() -> None:
    from evaluation.scripts.run_scanrefer_vg_side_by_side import (
        is_retryable_sample_error,
    )

    assert is_retryable_sample_error(
        RuntimeError(
            "InternalServerError: Error code: 504 - " "{'error': {'code': '-4307'}}"
        )
    )


def test_is_retryable_sample_error_accepts_transient_invalid_prompt_code() -> None:
    from evaluation.scripts.run_scanrefer_vg_side_by_side import (
        is_retryable_sample_error,
    )

    assert is_retryable_sample_error(
        RuntimeError(
            "BadRequestError: Error code: 400 - "
            "{'error': {'message': 'code: invalid_prompt; message: Invalid prompt', "
            "'code': '-4321'}}"
        )
    )
