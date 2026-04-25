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


def test_submit_final_sets_final_submission_on_success(tmp_path: Path) -> None:
    """Pack-v1 chassis terminal signal: successful submit_final must set
    runtime.final_submission so the run loop can break."""
    _register_vg_pack(tmp_path)
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    assert rs.final_submission is None  # default
    _, _, submit_final = build_chassis_tools(rs)
    submit_final.invoke(
        {"payload": {"value": 7}, "rationale": "ok", "evidence_refs": []}
    )
    # Adapter wraps as {"answer": payload}; final_submission mirrors it.
    assert rs.final_submission == {"answer": {"value": 7}}


def test_submit_final_does_not_set_final_submission_on_no_pack(tmp_path: Path) -> None:
    """When no pack is registered, submit_final returns ERROR and must NOT
    set the terminal signal — the run loop should keep going."""
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    _, _, submit_final = build_chassis_tools(rs)
    response = submit_final.invoke(
        {"payload": {}, "rationale": "x", "evidence_refs": []}
    )
    assert response.startswith("ERROR")
    assert rs.final_submission is None


def _ensure_vg_pack_registered() -> None:
    """Pull in the real VG_PACK; its `register()` runs on package import.
    Idempotent in case the autouse fixture cleared PACKS before this test."""
    import importlib
    import agents.packs.vg_embodiedscan
    if Stage2TaskType.VISUAL_GROUNDING not in PACKS:
        importlib.reload(agents.packs.vg_embodiedscan)


def test_submit_final_coerces_dict_to_payload_model_for_pydantic(tmp_path: Path) -> None:
    """When the pack's payload_model is a Pydantic BaseModel, the chassis
    must coerce raw dict payloads into the typed model before invoking
    the validator (which expects model attributes, not dict keys)."""
    _ensure_vg_pack_registered()
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    # task_ctx is required by VG_FINALIZER.validator for non-(-1) ids;
    # for the -1 path it's not dereferenced, but set a stub for safety.
    rs.task_ctx = type("Ctx", (), {"proposals": []})()
    _, _, submit_final = build_chassis_tools(rs)
    response = submit_final.invoke(
        {
            "payload": {"proposal_id": -1, "confidence": 0.0},  # raw dict
            "rationale": "ood",
            "evidence_refs": [],
        }
    )
    assert "submitted" in response.lower()
    # Adapter routes -1 to the failed-status payload.
    assert rs.final_submission is not None
    assert rs.final_submission["status"] == "failed"
    assert rs.final_submission["selected_object_id"] is None


def test_submit_final_returns_error_on_pydantic_payload_schema_mismatch(tmp_path: Path) -> None:
    """If the dict can't be coerced into the typed payload_model
    (e.g. wrong type, out-of-range), surface a clean ERROR string instead
    of an AttributeError."""
    _ensure_vg_pack_registered()
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    _, _, submit_final = build_chassis_tools(rs)
    response = submit_final.invoke(
        {"payload": {"proposal_id": -1, "confidence": 9.9}, "rationale": "x"}
    )
    assert response.startswith("ERROR:")
    assert "schema mismatch" in response


def test_submit_final_propagates_unrelated_exception(tmp_path: Path) -> None:
    """FAIL-LOUD: only narrow validation errors should be caught and surfaced
    as ERROR strings. Anything else (a bug in the validator, etc.) must
    propagate so build_agent crashes loudly and tests catch it."""
    body = tmp_path / "skill.md"
    body.write_text("# stub", encoding="utf-8")
    register_pack(
        TaskPack(
            task_type=Stage2TaskType.VISUAL_GROUNDING,
            tool_builder=lambda r: [],
            skills=[
                SkillSpec(
                    name="vg-grounding-playbook",
                    description="x",
                    body_path=body,
                    task_types={Stage2TaskType.VISUAL_GROUNDING},
                ),
            ],
            finalizer=FinalizerSpec(
                payload_model=dict,
                validator=lambda payload, runtime: 1 / 0,  # ZeroDivisionError
                adapter=lambda payload, runtime: {},
            ),
            required_primary_skill="vg-grounding-playbook",
            required_extra_metadata=[],
            ctx_factory=lambda b: object(),
        )
    )
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    _, _, submit_final = build_chassis_tools(rs)
    with pytest.raises(ZeroDivisionError):
        submit_final.invoke(
            {"payload": {"value": 1}, "rationale": "ok", "evidence_refs": []}
        )
