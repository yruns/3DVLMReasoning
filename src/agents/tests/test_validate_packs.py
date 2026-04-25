"""validate_packs runs at build_agent and FAILS LOUD on contract violations."""
from __future__ import annotations

from pathlib import Path

import pytest

from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.skills import (
    FinalizerSpec,
    PACKS,
    SkillSpec,
    TaskPack,
    register_pack,
)
from agents.skills.validate import validate_packs


@pytest.fixture(autouse=True)
def _reset_registry():
    PACKS.clear()
    yield
    PACKS.clear()


def _make_pack(tmp_path: Path, missing_body: bool = False) -> TaskPack:
    body = tmp_path / "playbook.md"
    if not missing_body:
        body.write_text("# stub", encoding="utf-8")
    return TaskPack(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        tool_builder=lambda r: [],
        skills=[
            SkillSpec(
                name="vg-grounding-playbook",
                description="VG.",
                body_path=body,
                task_types={Stage2TaskType.VISUAL_GROUNDING},
            )
        ],
        finalizer=FinalizerSpec(
            payload_model=dict,
            validator=lambda p, r: p,
            adapter=lambda p, r: {},
        ),
        required_primary_skill="vg-grounding-playbook",
        required_extra_metadata=["vg_proposal_pool"],
        ctx_factory=lambda b: object(),
    )


def test_validate_packs_passes_on_well_formed_pack(tmp_path: Path) -> None:
    register_pack(_make_pack(tmp_path))
    bundle = Stage2EvidenceBundle(
        extra_metadata={"vg_proposal_pool": {"proposals": [{}]}}
    )
    validate_packs(Stage2TaskType.VISUAL_GROUNDING, bundle)


def test_validate_packs_raises_when_no_pack(tmp_path: Path) -> None:
    bundle = Stage2EvidenceBundle()
    with pytest.raises(RuntimeError, match="no pack"):
        validate_packs(Stage2TaskType.NAV_PLAN, bundle, require_pack=True)


def test_validate_packs_raises_when_skill_body_missing(tmp_path: Path) -> None:
    register_pack(_make_pack(tmp_path, missing_body=True))
    bundle = Stage2EvidenceBundle(
        extra_metadata={"vg_proposal_pool": {"proposals": [{}]}}
    )
    with pytest.raises(RuntimeError, match="skill body not readable"):
        validate_packs(Stage2TaskType.VISUAL_GROUNDING, bundle)


def test_validate_packs_raises_on_missing_extra_metadata(tmp_path: Path) -> None:
    register_pack(_make_pack(tmp_path))
    bundle = Stage2EvidenceBundle(extra_metadata={})
    with pytest.raises(RuntimeError, match="missing required extra_metadata"):
        validate_packs(Stage2TaskType.VISUAL_GROUNDING, bundle)


def test_validate_packs_no_op_when_pack_absent_and_not_required(tmp_path: Path) -> None:
    bundle = Stage2EvidenceBundle()
    # QA without registered pack: ok (legacy QA path)
    validate_packs(Stage2TaskType.QA, bundle, require_pack=False)


class _StubTool:
    """Minimal duck-type for a langchain BaseTool: has a `.name` attribute."""

    def __init__(self, name: str) -> None:
        self.name = name


def test_validate_packs_raises_on_duplicate_tool_within_pack(tmp_path: Path) -> None:
    body = tmp_path / "playbook.md"
    body.write_text("# stub", encoding="utf-8")
    register_pack(
        TaskPack(
            task_type=Stage2TaskType.VISUAL_GROUNDING,
            tool_builder=lambda r: [_StubTool("dup_tool"), _StubTool("dup_tool")],
            skills=[
                SkillSpec(
                    name="vg-grounding-playbook",
                    description="VG.",
                    body_path=body,
                    task_types={Stage2TaskType.VISUAL_GROUNDING},
                ),
            ],
            finalizer=FinalizerSpec(
                payload_model=dict,
                validator=lambda p, r: p,
                adapter=lambda p, r: {},
            ),
            required_primary_skill="vg-grounding-playbook",
            required_extra_metadata=[],
            ctx_factory=lambda b: object(),
        )
    )
    bundle = Stage2EvidenceBundle()
    with pytest.raises(RuntimeError, match="duplicate tool name"):
        validate_packs(Stage2TaskType.VISUAL_GROUNDING, bundle)


def test_validate_packs_raises_on_tool_name_collision_across_packs(tmp_path: Path) -> None:
    body_a = tmp_path / "pa.md"
    body_a.write_text("# a", encoding="utf-8")
    body_b = tmp_path / "pb.md"
    body_b.write_text("# b", encoding="utf-8")
    register_pack(
        TaskPack(
            task_type=Stage2TaskType.QA,
            tool_builder=lambda r: [_StubTool("shared_tool")],
            skills=[
                SkillSpec(
                    name="qa-skill",
                    description="QA.",
                    body_path=body_a,
                    task_types={Stage2TaskType.QA},
                ),
            ],
            finalizer=FinalizerSpec(
                payload_model=dict,
                validator=lambda p, r: p,
                adapter=lambda p, r: {},
            ),
            required_primary_skill="qa-skill",
            required_extra_metadata=[],
            ctx_factory=lambda b: object(),
        )
    )
    register_pack(
        TaskPack(
            task_type=Stage2TaskType.VISUAL_GROUNDING,
            tool_builder=lambda r: [_StubTool("shared_tool")],
            skills=[
                SkillSpec(
                    name="vg-skill",
                    description="VG.",
                    body_path=body_b,
                    task_types={Stage2TaskType.VISUAL_GROUNDING},
                ),
            ],
            finalizer=FinalizerSpec(
                payload_model=dict,
                validator=lambda p, r: p,
                adapter=lambda p, r: {},
            ),
            required_primary_skill="vg-skill",
            required_extra_metadata=[],
            ctx_factory=lambda b: object(),
        )
    )
    bundle = Stage2EvidenceBundle()
    with pytest.raises(RuntimeError, match="tool name collision"):
        validate_packs(Stage2TaskType.VISUAL_GROUNDING, bundle)
