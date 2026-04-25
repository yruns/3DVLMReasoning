"""Chassis tools: list_skills, load_skill, submit_final."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.skills import (
    FinalizerSpec,
    PACKS,
    SkillSpec,
    TaskPack,
    register_pack,
)
from agents.skills.chassis_tools import build_chassis_tools


@pytest.fixture(autouse=True)
def _reset_registry():
    PACKS.clear()
    yield
    PACKS.clear()


def _runtime(task_type: Stage2TaskType) -> Stage2RuntimeState:
    rs = Stage2RuntimeState(bundle=Stage2EvidenceBundle())
    rs.task_type = task_type
    return rs


def _register_vg_pack(tmp_path: Path) -> None:
    body = tmp_path / "vg_grounding_playbook.md"
    body.write_text("# VG Grounding Playbook\n...details...", encoding="utf-8")
    register_pack(
        TaskPack(
            task_type=Stage2TaskType.VISUAL_GROUNDING,
            tool_builder=lambda r: [],
            skills=[
                SkillSpec(
                    name="vg-grounding-playbook",
                    description="VG main loop.",
                    body_path=body,
                    task_types={Stage2TaskType.VISUAL_GROUNDING},
                ),
            ],
            finalizer=FinalizerSpec(
                payload_model=dict,
                validator=lambda payload, runtime: payload,
                adapter=lambda payload, runtime: {"answer": payload},
            ),
            required_primary_skill="vg-grounding-playbook",
            required_extra_metadata=[],
            ctx_factory=lambda b: object(),
        )
    )


def test_list_skills_returns_catalog(tmp_path: Path) -> None:
    _register_vg_pack(tmp_path)
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    list_skills, _, _ = build_chassis_tools(rs)
    payload = json.loads(list_skills.invoke({}))
    assert payload == [{"name": "vg-grounding-playbook", "description": "VG main loop."}]


def test_load_skill_returns_body_and_records(tmp_path: Path) -> None:
    _register_vg_pack(tmp_path)
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    _, load_skill, _ = build_chassis_tools(rs)
    body = load_skill.invoke({"skill_name": "vg-grounding-playbook"})
    assert "VG Grounding Playbook" in body
    assert "vg-grounding-playbook" in rs.skills_loaded
    assert any(t.tool_name == "load_skill" for t in rs.tool_trace)


def test_load_skill_unknown_returns_error_first(tmp_path: Path) -> None:
    _register_vg_pack(tmp_path)
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    _, load_skill, _ = build_chassis_tools(rs)
    response = load_skill.invoke({"skill_name": "no-such-skill"})
    assert response.startswith("ERROR:")


def test_load_skill_unknown_twice_raises(tmp_path: Path) -> None:
    _register_vg_pack(tmp_path)
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    _, load_skill, _ = build_chassis_tools(rs)
    load_skill.invoke({"skill_name": "no-such-skill"})
    with pytest.raises(RuntimeError, match="repeated unknown skill"):
        load_skill.invoke({"skill_name": "no-such-skill"})


def test_submit_final_calls_validator_and_adapter(tmp_path: Path) -> None:
    _register_vg_pack(tmp_path)
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    _, _, submit_final = build_chassis_tools(rs)
    response = submit_final.invoke(
        {"payload": {"value": 42}, "rationale": "ok", "evidence_refs": []}
    )
    # Validator returns payload; adapter wraps as {"answer": payload}
    # The chassis stores resolved payload + signals termination
    assert "submitted" in response.lower()
    assert rs.skills_loaded.intersection({"vg-grounding-playbook"}) == set()  # no auto-load
