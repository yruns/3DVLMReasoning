"""Smoke: pack-v1 pilot can build a bundle + run validate_packs."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.integration
def test_pilot_build_bundle_passes_validate_packs(tmp_path: Path) -> None:
    """Verify the pilot's bundle builder produces a bundle that
    passes validate_packs without exception. Mocks proposal file."""
    import agents.packs.vg_embodiedscan
    from agents.skills import PACKS
    from agents.core.agent_config import Stage2TaskType

    # Defensive: ensure VG_PACK is registered (autouse fixtures in
    # other test modules may have cleared PACKS in the same session).
    if Stage2TaskType.VISUAL_GROUNDING not in PACKS:
        importlib.reload(agents.packs.vg_embodiedscan)

    from agents.examples.embodiedscan_vg_pack_v1_pilot import (
        build_pack_v1_bundle,
    )

    # mock proposal file
    proposals_path = tmp_path / "props.json"
    proposals_path.write_text(json.dumps({"proposals": [
        {"bbox_3d": [0]*9, "score": 0.9, "label": "chair"}
    ]}), encoding="utf-8")
    annotated = tmp_path / "ann"
    annotated.mkdir()
    (annotated / "frame_10.png").write_bytes(b"\x89PNG")

    bundle = build_pack_v1_bundle(
        proposals_jsonl=proposals_path,
        source="vdetr",
        annotated_image_dir=annotated,
        frame_visibility={10: [0]},
        keyframes=[(0, "/tmp/a.png", 10)],
        scene_id="scene0415_00",
    )
    assert "vg_proposal_pool" in bundle.extra_metadata

    from agents.skills.validate import validate_packs
    validate_packs(Stage2TaskType.VISUAL_GROUNDING, bundle)  # no raise
