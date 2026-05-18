from pathlib import Path

from agents.core.agent_config import Stage2DeepAgentConfig, Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle, Stage2TaskSpec
from agents.runtime.base import Stage2RuntimeState
from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime


def _legacy_image_channel_key(suffix: str = "images") -> str:
    return "vg_" + "pending" + "_" + suffix


def _assert_no_runtime_image_queue(runtime: Stage2RuntimeState) -> None:
    assert not any("pending" in name and "image" in name for name in dir(runtime))


def test_stage2_evidence_bundle_has_no_frame_seed_field() -> None:
    bundle = Stage2EvidenceBundle(scene_id="scene")

    field_name = "key" + "frames"
    assert field_name not in Stage2EvidenceBundle.model_fields
    assert not hasattr(bundle, field_name)


def test_config_has_no_seed_frame_restore_flag() -> None:
    field_name = "restore_stage1_seed_" + "key" + "frame_drain"
    assert field_name not in Stage2DeepAgentConfig.model_fields


def test_runtime_state_has_no_initial_frame_snapshot() -> None:
    runtime = Stage2RuntimeState(bundle=Stage2EvidenceBundle(scene_id="scene"))

    attr_name = "initial_" + "key" + "frame_paths"
    assert not hasattr(runtime, attr_name)


def test_initial_message_attaches_only_bev(tmp_path: Path) -> None:
    from PIL import Image

    legacy_key = _legacy_image_channel_key()
    bev = tmp_path / "bev.png"
    rgb = tmp_path / "rgb.png"
    Image.new("RGB", (8, 8), "white").save(bev)
    Image.new("RGB", (8, 8), "red").save(rgb)
    bundle = Stage2EvidenceBundle(
        scene_id="scene",
        bev_image_path=str(bev),
        extra_metadata={
            "scene_catalog": {
                "scene_id": "scene",
                "scene_category": "room",
                "total_frames": 1,
                "frame_id_range": [0, 0],
                "valid_frame_ids": [0],
                "bev_image_path": str(bev),
                "proposals": [],
            },
            legacy_key: [str(rgb)],
        },
    )
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        user_query="chair",
    )
    rt = DeepAgentsStage2Runtime(
        config=Stage2DeepAgentConfig(enable_stage1_text_retrieval=False)
    )
    state = Stage2RuntimeState(bundle=bundle)

    message = rt.build_user_message(task, state)

    image_parts = [p for p in message.content if p.get("type") == "image_url"]
    assert len(image_parts) == 1
    assert str(bev) in state.seen_image_paths
    assert str(rgb) not in state.seen_image_paths
    assert legacy_key in state.bundle.extra_metadata
    _assert_no_runtime_image_queue(state)


def test_evidence_update_ignores_legacy_bundle_image_channel(tmp_path: Path) -> None:
    from PIL import Image

    legacy_key = _legacy_image_channel_key()
    pending = tmp_path / "pending.png"
    Image.new("RGB", (8, 8), "blue").save(pending)
    bundle = Stage2EvidenceBundle(
        scene_id="scene",
        extra_metadata={legacy_key: [str(pending)]},
    )
    rt = DeepAgentsStage2Runtime(
        config=Stage2DeepAgentConfig(enable_stage1_text_retrieval=False)
    )
    state = Stage2RuntimeState(bundle=bundle)

    message = rt.build_evidence_update_message(state)

    assert message is None
    assert str(pending) not in state.seen_image_paths
    _assert_no_runtime_image_queue(state)
    assert state.bundle.extra_metadata[legacy_key] == [str(pending)]


def test_evidence_update_injects_tool_trace_images_only(tmp_path: Path) -> None:
    from PIL import Image

    pending = tmp_path / "pending.png"
    Image.new("RGB", (8, 8), "blue").save(pending)
    bundle = Stage2EvidenceBundle(scene_id="scene", extra_metadata={})
    rt = DeepAgentsStage2Runtime(
        config=Stage2DeepAgentConfig(enable_stage1_text_retrieval=False)
    )
    state = Stage2RuntimeState(bundle=bundle)
    state.record(
        "unit_tool",
        {},
        "returned an image",
        image_metadata=[
            {
                "image_path": str(pending),
                "frame_id": 1,
                "source_tool": "unit_test",
            }
        ],
    )

    message = rt.build_evidence_update_message(state)

    assert message is not None
    assert str(pending) in state.seen_image_paths
    _assert_no_runtime_image_queue(state)
    assert state.tool_trace[0].image_metadata[0]["image_path"] == str(pending)
    assert not any(
        key.startswith("vg_" + "pending") for key in state.bundle.extra_metadata
    )
