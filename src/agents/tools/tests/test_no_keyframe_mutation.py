from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.runtime.scene_runtime import queue_pending_image_if_new
from agents.tools.request_crops import (
    BBox2D,
    CropBackend,
    CropBackendConfig,
    CropRequest,
)


def test_queue_pending_image_records_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"not-used")
    runtime = SimpleNamespace(
        bundle=Stage2EvidenceBundle(scene_id="scene"),
        seen_image_paths=set(),
        mark_evidence_updated=lambda: None,
    )

    queued = queue_pending_image_if_new(
        runtime,
        str(image_path),
        metadata={
            "frame_id": 10,
            "source_tool": "select_by_text",
            "selected_because": "unit-test",
        },
    )

    assert queued is True
    extra = runtime.bundle.extra_metadata
    assert extra["vg_pending_images"] == [str(image_path)]
    assert extra["vg_pending_image_metadata"] == [
        {
            "image_path": str(image_path),
            "frame_id": 10,
            "source_tool": "select_by_text",
            "selected_because": "unit-test",
        }
    ]


def test_crop_backend_queues_crops_without_keyframe_field(tmp_path: Path) -> None:
    image_path = tmp_path / "frame_7.jpg"
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    img[20:60, 20:60] = [255, 0, 0]
    cv2.imwrite(str(image_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    bundle = Stage2EvidenceBundle(
        scene_id="scene",
        extra_metadata={
            "vg_pending_image_metadata": [
                {
                    "image_path": str(image_path),
                    "frame_id": 7,
                    "source_tool": "select_by_text",
                }
            ]
        },
    )
    backend = CropBackend(CropBackendConfig(output_dir=str(tmp_path / "crops")))

    results, updated = backend.process_requests(
        [CropRequest(frame_idx=7, bbox=BBox2D(20, 20, 60, 60), note="target")],
        bundle,
    )

    assert len(results) == 1
    assert results[0].success is True
    assert not hasattr(updated, "key" + "frames")
    pending = updated.extra_metadata["vg_pending_images"]
    assert len(pending) == 1
    assert Path(pending[0]).exists()
    crop_meta = updated.extra_metadata["vg_pending_image_metadata"][-1]
    assert crop_meta["source_tool"] == "request_crops"
    assert crop_meta["frame_id"] == 7


def test_runtime_state_has_no_keyframes_after_crop(tmp_path: Path) -> None:
    image_path = tmp_path / "frame_3.jpg"
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[16:48, 16:48] = [0, 255, 0]
    cv2.imwrite(str(image_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    state = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(
            scene_id="scene",
            extra_metadata={
                "vg_pending_image_metadata": [
                    {"image_path": str(image_path), "frame_id": 3}
                ]
            },
        )
    )
    backend = CropBackend(CropBackendConfig(output_dir=str(tmp_path / "crops")))

    _, state.bundle = backend.process_requests(
        [CropRequest(frame_idx=3, bbox=BBox2D(16, 16, 48, 48))],
        state.bundle,
    )

    assert not hasattr(state.bundle, "key" + "frames")
    assert len(state.bundle.extra_metadata["vg_pending_images"]) == 1
