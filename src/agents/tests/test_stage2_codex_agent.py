from __future__ import annotations

import json

from agents.core.agent_config import Stage2DeepAgentConfig, Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle, Stage2TaskSpec
from agents.runtime.codex_sdk_agent import CodexSdkStage2Runtime


def _bundle(tmp_path) -> Stage2EvidenceBundle:
    annotated = tmp_path / "annotated"
    annotated.mkdir()
    return Stage2EvidenceBundle(
        scene_id="scene0001_00",
        stage1_query="the square table by the door",
        extra_metadata={
            "gt_bbox_3d_9dof": [9999.0, 9998.0, 9997.0, 1, 1, 1, 0, 0, 0],
            "scene_catalog": {
                "scene_id": "scene0001_00",
                "scene_category": "room",
                "total_frames": 12,
                "frame_id_range": [0, 11],
                "proposals": [
                    {
                        "proposal_id": 6,
                        "category": "table",
                        "position_3d": [1.0, 2.0, 0.4],
                        "bbox_3d_9dof": [1, 2, 0.4, 1.2, 1.1, 0.8, 0, 0, 0],
                        "enriched_category": "square table",
                        "compact_note": "Square table near the door.",
                    }
                ],
            },
            "vg_proposal_pool": {
                "source": "gt",
                "annotated_image_dir": str(annotated),
                "frame_index": {"10": [6]},
                "proposal_index": {"6": [10]},
                "proposals": [
                    {
                        "id": 6,
                        "bbox_3d_9dof": [1, 2, 0.4, 1.2, 1.1, 0.8, 0, 0, 0],
                        "category": "table",
                        "score": 1.0,
                        "enriched_category": "square table",
                        "compact_note": "Square table near the door.",
                        "frame_views": {
                            "10": {
                                "frame_id": 10,
                                "bbox_2d": [100, 120, 300, 320],
                                "raw_rgb_path": "raw/000100-rgb.png",
                            }
                        },
                    }
                ],
            },
        },
    )


def test_codex_prompt_excludes_gt_fields(tmp_path) -> None:
    runtime = CodexSdkStage2Runtime(
        config=Stage2DeepAgentConfig(enable_stage1_text_retrieval=False)
    )
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        user_query="the square table by the door",
    )

    prompt = runtime.build_decision_prompt(task, _bundle(tmp_path))

    assert "gt_bbox_3d_9dof" not in prompt
    assert "9999" not in prompt
    assert "#6" in prompt
    assert "Square table near the door" in prompt


def test_codex_runtime_wraps_json_decision(monkeypatch, tmp_path) -> None:
    runtime = CodexSdkStage2Runtime(
        config=Stage2DeepAgentConfig(enable_stage1_text_retrieval=False)
    )
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        user_query="the square table by the door",
    )

    def fake_run_turn(*_args, **_kwargs):
        return (
            json.dumps(
                {
                    "proposal_id": 6,
                    "confidence": 0.88,
                    "summary": "Proposal 6 is the square table closest to the door.",
                    "uncertainties": [],
                    "cited_frame_indices": [10],
                }
            ),
            {"id": "turn_mock", "status": "completed"},
        )

    monkeypatch.setattr(runtime, "_run_codex_turn", fake_run_turn)

    result = runtime.run(task, _bundle(tmp_path))

    assert result.result.status.value == "completed"
    assert result.result.payload["proposal_id"] == 6
    assert result.result.payload["selected_object_id"] == 6
    assert result.result.confidence == 0.88
    assert result.tool_trace[0].tool_name == "codex_sdk_turn"
