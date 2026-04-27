"""QA pack registration: assemble TaskPack and register it."""
from __future__ import annotations

from pathlib import Path

from agents.core.agent_config import Stage2TaskType
from agents.packs.qa_default.finalizer import QA_FINALIZER
from agents.packs.qa_default.tools import build_qa_tools
from agents.skills import SkillSpec, TaskPack, register_pack

_PACK_DIR = Path(__file__).resolve().parent
_SKILLS_DIR = _PACK_DIR / "skills"

QA_PACK = TaskPack(
    task_type=Stage2TaskType.QA,
    tool_builder=build_qa_tools,
    skills=[
        SkillSpec(
            name="qa-answering-playbook",
            description="Default QA loop: inspect evidence, request missing views/crops, then submit answer.",
            body_path=_SKILLS_DIR / "qa_answering_playbook.md",
            task_types={Stage2TaskType.QA},
        ),
        SkillSpec(
            name="evidence-scouting",
            description="Decide when to request more keyframes or crops, and how to phrase the request.",
            body_path=_SKILLS_DIR / "evidence_scouting.md",
            task_types={Stage2TaskType.QA},
        ),
    ],
    finalizer=QA_FINALIZER,
    required_primary_skill="qa-answering-playbook",
    required_extra_metadata=[],
    ctx_factory=lambda bundle: bundle,
)


def register() -> None:
    register_pack(QA_PACK)


__all__ = ["QA_PACK", "register"]
