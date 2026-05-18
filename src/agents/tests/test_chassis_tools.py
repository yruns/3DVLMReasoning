"""Chassis tools: list_skills, load_skill, submit_final."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

import agents.packs.qa_default
import agents.packs.vg_embodiedscan
from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.skills import (
    PACKS,
    FinalizerSpec,
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


def test_submit_final_does_not_overwrite_accepted_submission(
    tmp_path: Path,
) -> None:
    """DeepAgents may execute more tools before the outer loop sees the
    terminal signal. Once a final is accepted, later same-turn submissions must
    not replace it."""
    _register_vg_pack(tmp_path)
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    _, _, submit_final = build_chassis_tools(rs)

    first = submit_final.invoke(
        {"payload": {"value": 7}, "rationale": "first", "evidence_refs": []}
    )
    second = submit_final.invoke(
        {"payload": {"value": 99}, "rationale": "second", "evidence_refs": []}
    )

    assert "submitted" in first.lower()
    assert "already_submitted" in second.lower()
    assert rs.final_submission == {"answer": {"value": 7}}
    assert rs.tool_trace[-1].response_text.startswith("ALREADY_SUBMITTED")


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


def test_submit_final_blocks_when_rationale_rules_out_payload_id(
    tmp_path: Path,
) -> None:
    _register_vg_pack(tmp_path)
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 31, "confidence": 0.94},
            "rationale": (
                "The newly provided first-person evidence shows door #31 directly "
                "adjacent to the vending-machine area, so it is the door to exclude. "
                "That rules out #31. Marked frame 114 verifies #21 is indeed a door. "
                "Therefore the best match is #21."
            ),
            "evidence_refs": [],
        }
    )

    assert response.startswith("RATIONALE_PAYLOAD_GUARD:")
    assert "rules out submitted proposal #31" in response
    assert rs.final_submission is None
    assert rs.tool_trace[-1].tool_input["rationale_payload_guard_blocked"] is True
    assert rs.tool_trace[-1].tool_input["rationale_payload_guard_final_ids"] == [21]


def test_submit_final_blocks_when_rationale_names_different_final_id(
    tmp_path: Path,
) -> None:
    _register_vg_pack(tmp_path)
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 31, "confidence": 0.91},
            "rationale": (
                "Marked frame 114 verifies #21 is the door in the outside corner. "
                "Therefore the final answer is #21."
            ),
            "evidence_refs": [],
        }
    )

    assert response.startswith("RATIONALE_PAYLOAD_GUARD:")
    assert "names final proposal #21" in response
    assert rs.final_submission is None


def test_submit_final_extracts_best_match_for_clause_final_id(
    tmp_path: Path,
) -> None:
    _register_vg_pack(tmp_path)
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 31, "confidence": 0.91},
            "rationale": (
                "Marked frame 114 verifies #21 is indeed a door. "
                "Therefore the best match for the door in the corner is #21."
            ),
            "evidence_refs": [],
        }
    )

    assert response.startswith("RATIONALE_PAYLOAD_GUARD:")
    assert rs.tool_trace[-1].tool_input["rationale_payload_guard_final_ids"] == [21]


def test_submit_final_allows_other_ids_when_payload_is_final_choice(
    tmp_path: Path,
) -> None:
    _register_vg_pack(tmp_path)
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 31, "confidence": 0.88},
            "rationale": (
                "Compared #21 and #31. #21 is farther from the corner, so "
                "proposal #31 is the best match."
            ),
            "evidence_refs": [],
        }
    )

    assert "submitted" in response.lower()
    assert rs.final_submission == {
        "answer": {"proposal_id": 31, "confidence": 0.88}
    }


def test_submit_final_allows_quoted_query_negation_for_payload_id(
    tmp_path: Path,
) -> None:
    _register_vg_pack(tmp_path)
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 6, "confidence": 0.88},
            "rationale": (
                "Choose proposal #6. Door #16 is the gray door with boxes in "
                "front, so it is explicitly the one the query says not to choose. "
                "Door #6 is the other door. Therefore the described door "
                "'next to the big grey box, NOT the one with the cardboard box "
                "in front of it' is proposal #6."
            ),
            "evidence_refs": [],
        }
    )

    assert "submitted" in response.lower()
    assert rs.final_submission == {"answer": {"proposal_id": 6, "confidence": 0.88}}


def test_submit_final_allows_distractor_negation_in_same_rationale(
    tmp_path: Path,
) -> None:
    _register_vg_pack(tmp_path)
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 28, "confidence": 0.84},
            "rationale": (
                "The correct target is proposal #28, not #27. In frames 47-49, "
                "only #28 is visible, and those frames identify #28's location "
                "but not the target state. In frames 63 and 64, #28 is upright "
                "while #27 is mostly just a head."
            ),
            "evidence_refs": [],
        }
    )

    assert "submitted" in response.lower()
    assert rs.final_submission == {
        "answer": {"proposal_id": 28, "confidence": 0.84}
    }


def test_submit_final_allows_rejected_distractors_that_reference_payload(
    tmp_path: Path,
) -> None:
    _register_vg_pack(tmp_path)
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 20, "confidence": 0.86},
            "rationale": (
                "Thus #20 is the only mouse that is consistently the leftmost. "
                "I rejected #24 because it is right of #20 in frame 10, and "
                "rejected #27 because it is right of #20 in frame 49."
            ),
            "evidence_refs": [],
        }
    )

    assert "submitted" in response.lower()
    assert rs.final_submission == {
        "answer": {"proposal_id": 20, "confidence": 0.86}
    }


def _ensure_vg_pack_registered() -> None:
    """Pull in the real VG_PACK; its `register()` runs on package import.
    Idempotent in case the autouse fixture cleared PACKS before this test."""
    if Stage2TaskType.VISUAL_GROUNDING not in PACKS:
        importlib.reload(agents.packs.vg_embodiedscan)


def _ensure_qa_pack_registered() -> None:
    if Stage2TaskType.QA not in PACKS:
        importlib.reload(agents.packs.qa_default)


def test_submit_final_unwraps_structured_response_payload_for_qa(tmp_path: Path) -> None:
    """QA agents may pass a full Stage2StructuredResponse-shaped object as
    the submit_final payload. The chassis should validate the nested
    task payload instead of rejecting the call."""
    _ensure_qa_pack_registered()
    rs = _runtime(Stage2TaskType.QA)
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {
                "task_type": "qa",
                "status": "completed",
                "summary": "A red fire extinguisher is below the windows.",
                "confidence": 0.91,
                "payload": {
                    "answer": "A fire extinguisher.",
                    "supporting_claims": ["A red extinguisher is visible."],
                },
            },
            "rationale": "direct visual evidence",
            "evidence_refs": [],
        }
    )

    assert "submitted" in response.lower()
    assert rs.final_submission == {
        "status": "completed",
        "answer": "A fire extinguisher.",
        "supporting_claims": ["A red extinguisher is visible."],
    }


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
