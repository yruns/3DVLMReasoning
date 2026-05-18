from agents.core.task_types import Stage2EvidenceBundle
from agents.stage1_callbacks import create_crop_callback


def test_stage1_crop_callback_fails_loud_without_concrete_crop_backend() -> None:
    callback = create_crop_callback(scene_id="scene")
    result = callback(
        Stage2EvidenceBundle(scene_id="scene"),
        {"object_terms": ["chair"], "frame_indices": [0]},
    )

    assert result.response_text.startswith("ERROR:")
    assert "No crops generated" in result.response_text
    assert result.updated_bundle is None
    assert result.image_metadata == []
