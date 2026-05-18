"""VG pack registration: assemble TaskPack and register it."""

from __future__ import annotations

from pathlib import Path

from agents.core.agent_config import Stage2TaskType
from agents.packs.vg_embodiedscan.ctx import build_ctx_from_bundle
from agents.packs.vg_embodiedscan.finalizer import VG_FINALIZER
from agents.packs.vg_embodiedscan.tools import build_vg_tools
from agents.skills import SkillSpec, TaskPack, register_pack

_PACK_DIR = Path(__file__).resolve().parent
_SKILLS_DIR = _PACK_DIR / "skills"
_SHARED_SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills" / "shared_skills"

VG_PACK = TaskPack(
    task_type=Stage2TaskType.VISUAL_GROUNDING,
    tool_builder=build_vg_tools,
    skills=[
        SkillSpec(
            name="scene-exploration-playbook",
            description=(
                "Prerequisite v9 gate. Cheapest-first selector ladder + BEV/catalog "
                "cheat sheet. Required before any selector / view tool runs."
            ),
            body_path=_SHARED_SKILLS_DIR / "scene_exploration_playbook.md",
            task_types={Stage2TaskType.VISUAL_GROUNDING, Stage2TaskType.QA},
        ),
        SkillSpec(
            name="vg-grounding-playbook",
            description="EmbodiedScan VG main loop: read selected frames, pick proposal, submit.",
            body_path=_SKILLS_DIR / "vg_grounding_playbook.md",
            task_types={Stage2TaskType.VISUAL_GROUNDING},
        ),
        SkillSpec(
            name="vg-spatial-disambiguation",
            description=(
                "Use when the query contains spatial relations like "
                "'next to', 'left of', 'above', or 'closest to'."
            ),
            body_path=_SKILLS_DIR / "vg_spatial_disambiguation.md",
            task_types={Stage2TaskType.VISUAL_GROUNDING},
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
