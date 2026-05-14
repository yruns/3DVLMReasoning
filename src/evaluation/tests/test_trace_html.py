from __future__ import annotations

import base64
import json
from pathlib import Path

from evaluation.trace_html import (
    load_vg_trace_session,
    render_vg_trace_html,
)

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG_1X1)


def _sample_id(idx: int) -> str:
    return f"scannet/scene000{idx}_00::{idx}::{1000 + idx}"


def _safe_sample_name(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("::", "__")


def _write_sample_artifacts(root: Path, sample_id: str, target_id: int) -> Path:
    scene_id = sample_id.split("::", 1)[0].split("/")[-1]
    raw = root / scene_id / "raw" / "000010-rgb.png"
    marked = root / scene_id / "pack_test" / "annotated" / "frame_10.png"
    _write_png(raw)
    _write_png(marked)
    _write_json(
        root
        / scene_id
        / "pack_test"
        / "samples"
        / f"{_safe_sample_name(sample_id)}.json",
        {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "target_id": target_id,
            "category": "chair",
            "query": f"query {target_id}",
            "keyframes": [
                {
                    "keyframe_idx": 0,
                    "frame_id": 10,
                    "image_path": str(raw),
                }
            ],
        },
    )
    return marked


def test_load_vg_trace_session_selects_correct_and_failed_samples(tmp_path: Path):
    session_dir = tmp_path / "eval_session"
    data_root = tmp_path / "data" / "nr3d" / "scannet"
    rows = []
    metrics = []
    for idx, is_correct in enumerate([True, True, False, False, True]):
        sample_id = _sample_id(idx)
        target_id = idx
        selected_id = target_id if is_correct else target_id + 100
        marked = _write_sample_artifacts(data_root, sample_id, target_id)
        rows.append(
            {
                "sample_id": sample_id,
                "backend": "pack_v1",
                "status": "completed",
                "iou": 1.0 if is_correct else 0.0,
                "selected_object_id": selected_id,
                "confidence": 0.9,
                "query": f"query {idx}",
                "tool_trace": [
                    {
                        "tool_name": "view_keyframe_marked",
                        "tool_input": {"frame_id": 10},
                        "response_text": (
                            f"frame_id=10 marked image at {marked}; "
                            "visible_proposals=[0]"
                        ),
                    }
                ],
            }
        )
        metrics.append(
            {
                "sample_id": sample_id,
                "target_id": target_id,
                "selected_object_id": selected_id,
                "is_correct": is_correct,
                "is_easy": True,
                "is_view_dep": False,
            }
        )

    _write_json(session_dir / "side_by_side.json", {"pack_v1": {"per_sample": rows}})
    _write_json(session_dir / "leaderboard_metrics.json", {"per_sample": metrics})
    (session_dir / "per_sample" / "pack_test").mkdir(parents=True)

    trace = load_vg_trace_session(
        session_dir=session_dir,
        backend="pack_v1",
        data_root=data_root,
        num_correct=2,
        num_failed=2,
    )

    assert [sample.is_correct for sample in trace.samples] == [
        True,
        True,
        False,
        False,
    ]
    assert trace.pack_name == "pack_test"
    assert trace.samples[0].target_id == 0
    assert trace.samples[0].initial_keyframes[0].frame_id == 10
    assert trace.samples[0].tool_calls[0].images[0].role == "marked"


def test_render_vg_trace_html_includes_query_gt_tools_and_images(tmp_path: Path):
    session_dir = tmp_path / "eval_session"
    data_root = tmp_path / "data" / "nr3d" / "scannet"
    sample_id = _sample_id(0)
    marked = _write_sample_artifacts(data_root, sample_id, target_id=0)
    row = {
        "sample_id": sample_id,
        "backend": "pack_v1",
        "status": "completed",
        "iou": 1.0,
        "selected_object_id": 0,
        "confidence": 0.95,
        "query": "pick the grey chair next to the green one.",
        "tool_trace": [
            {
                "tool_name": "view_keyframe_marked",
                "tool_input": {"frame_id": 10},
                "response_text": f"frame_id=10 marked image at {marked}",
            },
            {
                "tool_name": "submit_final",
                "tool_input": {"proposal_id": 0, "confidence": 0.95},
                "response_text": "accepted",
            },
        ],
    }
    metric = {
        "sample_id": sample_id,
        "target_id": 0,
        "selected_object_id": 0,
        "is_correct": True,
    }
    _write_json(session_dir / "side_by_side.json", {"pack_v1": {"per_sample": [row]}})
    _write_json(session_dir / "leaderboard_metrics.json", {"per_sample": [metric]})
    (session_dir / "per_sample" / "pack_test").mkdir(parents=True)

    trace = load_vg_trace_session(
        session_dir=session_dir,
        backend="pack_v1",
        data_root=data_root,
        sample_ids=[sample_id],
    )
    html = render_vg_trace_html(trace)

    assert "VG Agent Trace 可视化" in html
    assert "真实任务 Query" in html
    assert "pick the grey chair next to the green one." in html
    assert "GT: #0" in html
    assert "预测: #0" in html
    assert "view_keyframe_marked" in html
    assert "submit_final" in html
    assert "真实输入" in html
    assert "真实输出" in html
    assert "data:image/png;base64," in html
