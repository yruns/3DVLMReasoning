"""Registry: SkillSpec/TaskPack/FinalizerSpec wiring."""
from __future__ import annotations

from pathlib import Path

import pytest

from agents.core.agent_config import Stage2TaskType
from agents.skills import (
    FinalizerSpec,
    PACKS,
    SkillSpec,
    TaskPack,
    register_pack,
    skills_for,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    PACKS.clear()
    yield
    PACKS.clear()


def _stub_finalizer() -> FinalizerSpec:
    return FinalizerSpec(
        payload_model=dict,
        validator=lambda payload, runtime: None,
        adapter=lambda payload, runtime: {},
    )


def test_register_pack_stores_pack_by_task_type(tmp_path: Path) -> None:
    body = tmp_path / "skill.md"
    body.write_text("# stub", encoding="utf-8")
    skill = SkillSpec(
        name="vg-grounding-playbook",
        description="stub",
        body_path=body,
        task_types={Stage2TaskType.VISUAL_GROUNDING},
    )
    pack = TaskPack(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        tool_builder=lambda runtime: [],
        skills=[skill],
        finalizer=_stub_finalizer(),
        required_primary_skill="vg-grounding-playbook",
        required_extra_metadata=["vg_proposal_pool"],
        ctx_factory=lambda bundle: {},
    )
    register_pack(pack)
    assert PACKS[Stage2TaskType.VISUAL_GROUNDING] is pack


def test_register_pack_rejects_duplicate(tmp_path: Path) -> None:
    body = tmp_path / "skill.md"
    body.write_text("# stub", encoding="utf-8")
    pack = TaskPack(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        tool_builder=lambda runtime: [],
        skills=[],
        finalizer=_stub_finalizer(),
        required_primary_skill="x",
        required_extra_metadata=[],
        ctx_factory=lambda bundle: {},
    )
    register_pack(pack)
    with pytest.raises(RuntimeError, match="duplicate pack"):
        register_pack(pack)


def test_skills_for_filters_by_task_type(tmp_path: Path) -> None:
    body = tmp_path / "skill.md"
    body.write_text("# stub", encoding="utf-8")
    vg_skill = SkillSpec(
        name="vg-grounding-playbook",
        description="vg",
        body_path=body,
        task_types={Stage2TaskType.VISUAL_GROUNDING},
    )
    qa_skill = SkillSpec(
        name="qa-answering-playbook",
        description="qa",
        body_path=body,
        task_types={Stage2TaskType.QA},
    )
    register_pack(
        TaskPack(
            task_type=Stage2TaskType.VISUAL_GROUNDING,
            tool_builder=lambda r: [],
            skills=[vg_skill, qa_skill],
            finalizer=_stub_finalizer(),
            required_primary_skill="vg-grounding-playbook",
            required_extra_metadata=[],
            ctx_factory=lambda b: {},
        )
    )
    names = [s.name for s in skills_for(Stage2TaskType.VISUAL_GROUNDING)]
    assert names == ["vg-grounding-playbook"]
