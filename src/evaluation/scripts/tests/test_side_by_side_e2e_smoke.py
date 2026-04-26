"""End-to-end side-by-side smoke with offline inputs and mocked agents."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


@pytest.mark.integration
def test_side_by_side_e2e_smoke_with_mocked_agents(tmp_path, monkeypatch) -> None:
    from evaluation.scripts import prepare_pack_v1_inputs as prep
    from evaluation.scripts import run_embodiedscan_vg_side_by_side as runner

    scene_id = "scene_demo"
    target_id = 42
    sample_id = f"{scene_id}::{target_id}"
    gt_bbox = [0.0, 0.0, 5.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    sample_ids_path = tmp_path / "batch_sample_ids.json"
    sample_ids_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": sample_id,
                    "scene_id": scene_id,
                    "target_id": target_id,
                    "category": "chair",
                }
            ]
        ),
        encoding="utf-8",
    )

    vdetr_dir = tmp_path / "vdetr"
    predictions_path = vdetr_dir / scene_id / "predictions.json"
    predictions_path.parent.mkdir(parents=True)
    predictions_path.write_text(
        json.dumps(
            {
                "proposals": [
                    {
                        "bbox_3d": gt_bbox,
                        "score": 0.95,
                        "label": "chair",
                    },
                    {
                        "bbox_3d": [2.0, 0.0, 5.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        "score": 0.60,
                        "label": "table",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    embodiedscan_data_root = tmp_path / "embodiedscan"
    rgb_path = embodiedscan_data_root / scene_id / "posed_images" / "000010.jpg"
    rgb_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), color="white").save(rgb_path)

    scene_info = {
        "sample_idx": f"scannet/{scene_id}",
        "cam2img": [[50.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]],
        "axis_align_matrix": np.eye(4).tolist(),
        "images": [
            {
                "frame_id": 10,
                "img_path": str(rgb_path.relative_to(embodiedscan_data_root)),
                "cam2global": np.eye(4).tolist(),
            }
        ],
    }
    sample = SimpleNamespace(
        sample_id=sample_id,
        scene_id=scene_id,
        scan_id=f"scannet/{scene_id}",
        target_id=target_id,
        target="chair",
        query="the chair in the center of the room",
        gt_bbox_3d=gt_bbox,
    )
    adapter = SimpleNamespace(
        dataset=SimpleNamespace(get_scene_info=lambda scan_id: scene_info),
        get_axis_align_matrix=lambda scan_id: np.eye(4),
    )

    monkeypatch.setattr(
        prep,
        "load_sample_lookup",
        lambda data_root: (adapter, {(scene_id, target_id): sample}),
    )
    monkeypatch.setattr(
        prep,
        "select_keyframes_for_sample",
        lambda *args, **kwargs: [
            {"keyframe_idx": 0, "image_path": str(rgb_path), "frame_id": 10}
        ],
    )

    pack_v1_inputs_dir = tmp_path / "outputs" / "pack_v1"
    written = prep.prepare_pack_v1_inputs(
        sample_ids_path=sample_ids_path,
        vdetr_proposals_dir=vdetr_dir,
        embodiedscan_data_root=embodiedscan_data_root,
        output_dir=pack_v1_inputs_dir,
        source="vdetr",
    )

    scene_dir = pack_v1_inputs_dir / "scenes" / scene_id
    proposals_jsonl = scene_dir / "proposals.jsonl"
    visibility_json = scene_dir / "visibility.json"
    annotated_frames = sorted((scene_dir / "annotated").glob("frame_*.png"))
    sample_json = pack_v1_inputs_dir / "samples" / f"{scene_id}__{target_id}.json"

    assert written == [sample_json]
    assert proposals_jsonl.exists()
    assert visibility_json.exists()
    assert json.loads(visibility_json.read_text(encoding="utf-8"))
    assert annotated_frames
    assert sample_json.exists()

    import agents.packs.vg_embodiedscan
    from agents.core.agent_config import Stage2TaskType
    from agents.skills import PACKS
    from agents.skills.validate import validate_packs

    if Stage2TaskType.VISUAL_GROUNDING not in PACKS:
        importlib.reload(agents.packs.vg_embodiedscan)

    sample_payload = json.loads(sample_json.read_text(encoding="utf-8"))
    bundle = runner.build_pack_v1_bundle_from_sample(
        sample_payload,
        pack_v1_inputs_dir,
    )
    validate_packs(Stage2TaskType.VISUAL_GROUNDING, bundle)

    class FakeStage2Agent:
        def __init__(self, config):
            self.config = config

        def run(self, task, bundle):
            assert self.config.vg_backend == "pack_v1"
            assert task.user_query == "the chair in the center of the room"
            proposal = bundle.extra_metadata["vg_proposal_pool"]["proposals"][0]
            return SimpleNamespace(
                result=SimpleNamespace(
                    payload={
                        "status": "completed",
                        "proposal_id": int(proposal["id"]),
                        "bbox_3d": list(proposal["bbox_3d_9dof"]),
                        "confidence": 0.99,
                    },
                    confidence=0.99,
                ),
                raw_state={},
            )

    monkeypatch.setattr(runner, "Stage2DeepResearchAgent", FakeStage2Agent)

    from agents.adapters.embodiedscan_adapter import EmbodiedScanVGAdapter
    from agents.examples import embodiedscan_vg_pilot

    monkeypatch.setattr(
        EmbodiedScanVGAdapter,
        "load_samples",
        lambda self, split="val", source_filter=None, **kwargs: [sample],
    )

    def fake_legacy_pilot(sample_arg, adapter_arg, config, *args, **kwargs):
        assert sample_arg.scene_id == scene_id
        assert sample_arg.target_id == target_id
        assert config.vg_backend == "legacy"
        return {
            "sample": sample_arg,
            "prediction": {"sample_id": sample_id, "bbox_3d": None},
            "result": SimpleNamespace(
                raw_state={
                    "vg_selected_object_id": target_id,
                    "vg_selected_bbox_3d": gt_bbox,
                }
            ),
        }

    monkeypatch.setattr(embodiedscan_vg_pilot, "run_one_sample", fake_legacy_pilot)

    out_dir = tmp_path / "out"
    results = runner.compare_backends(
        sample_ids=[sample_id],
        output_dir=out_dir,
        pack_v1_inputs_dir=pack_v1_inputs_dir,
        embodiedscan_data_root=embodiedscan_data_root,
    )

    assert (out_dir / "side_by_side.json").exists()
    for backend in ("legacy", "pack_v1"):
        assert results[backend]["n"] == 1
        assert results[backend]["mean_iou"] == pytest.approx(1.0)
        assert results[backend]["Acc@0.25"] == 1.0
        assert results[backend]["Acc@0.50"] == 1.0
