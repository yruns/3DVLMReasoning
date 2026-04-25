"""VG pack registration: assemble TaskPack and register it."""
from __future__ import annotations

from pathlib import Path

from agents.core.agent_config import Stage2TaskType
from agents.skills import SkillSpec, TaskPack, register_pack
from agents.packs.vg_embodiedscan.ctx import build_ctx_from_bundle
from agents.packs.vg_embodiedscan.finalizer import VG_FINALIZER
from agents.packs.vg_embodiedscan.tools import build_vg_tools

_PACK_DIR = Path(__file__).resolve().parent
_SKILLS_DIR = _PACK_DIR / "skills"

VG_PACK = TaskPack(
    task_type=Stage2TaskType.VISUAL_GROUNDING,
    tool_builder=build_vg_tools,
    skills=[
        SkillSpec(
            name="vg-grounding-playbook",
            description="EmbodiedScan VG main loop: read marked keyframes, pick proposal, submit.",
            body_path=_SKILLS_DIR / "vg_grounding_playbook.md",
            task_types={Stage2TaskType.VISUAL_GROUNDING},
        ),
        SkillSpec(
            name="vg-spatial-disambiguation",
            description="Use when the query contains spatial relations like 'next to' or 'closest to'.",
            body_path=_SKILLS_DIR / "vg_spatial_disambiguation.md",
            task_types={Stage2TaskType.VISUAL_GROUNDING},
        ),
        SkillSpec(
            name="evidence-scouting",
            description="Decide when to request more keyframes or crops, and how to phrase the request.",
            body_path=_SKILLS_DIR / "evidence_scouting.md",
            task_types={Stage2TaskType.VISUAL_GROUNDING, Stage2TaskType.QA},
        ),
    ],
    finalizer=VG_FINALIZER,
    required_primary_skill="vg-grounding-playbook",
    required_extra_metadata=["vg_proposal_pool"],
    ctx_factory=build_ctx_from_bundle,
)


def register() -> None:
    register_pack(VG_PACK)


__all__ = ["VG_PACK", "register"]
