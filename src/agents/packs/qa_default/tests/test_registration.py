"""QA pack registration smoke test."""
from __future__ import annotations

import importlib

import agents.packs  # noqa: F401 - keep cached, will reload below
import agents.packs.qa_default
from agents.core.agent_config import Stage2TaskType
from agents.skills import PACKS


def test_qa_pack_registers_on_import() -> None:
    PACKS.clear()
    importlib.reload(agents.packs.qa_default)

    assert Stage2TaskType.QA in PACKS
    pack = PACKS[Stage2TaskType.QA]
    skill_names = sorted(s.name for s in pack.skills)
    assert skill_names == ["evidence-scouting", "qa-answering-playbook"]
    assert pack.required_primary_skill == "qa-answering-playbook"
    assert pack.required_extra_metadata == []
    assert pack.ctx_factory("bundle") == "bundle"
