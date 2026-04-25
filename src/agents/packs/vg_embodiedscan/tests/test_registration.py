"""VG pack registration smoke test."""
from __future__ import annotations

import importlib

import agents.packs  # noqa: F401 — keep cached, will reload below
import agents.packs.vg_embodiedscan
from agents.core.agent_config import Stage2TaskType
from agents.skills import PACKS


def test_vg_pack_registers_on_import() -> None:
    # Verify the auto-register-on-import contract end-to-end. Python caches
    # `agents.packs.vg_embodiedscan` after the first import, so a plain
    # `import` after `PACKS.clear()` is a no-op. Use `importlib.reload`
    # to force the package's __init__.py (which calls register()) to
    # re-run, mirroring what happens on a fresh interpreter start.
    PACKS.clear()
    importlib.reload(agents.packs.vg_embodiedscan)
    assert Stage2TaskType.VISUAL_GROUNDING in PACKS
    pack = PACKS[Stage2TaskType.VISUAL_GROUNDING]
    skill_names = sorted(s.name for s in pack.skills)
    assert skill_names == [
        "evidence-scouting",
        "vg-grounding-playbook",
        "vg-spatial-disambiguation",
    ]
    assert pack.required_primary_skill == "vg-grounding-playbook"
    assert pack.required_extra_metadata == ["vg_proposal_pool"]
