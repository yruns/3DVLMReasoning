"""Re-export the registry surface."""
from agents.skills.finalizer import FinalizerSpec
from agents.skills.registry import (
    PACKS,
    SkillSpec,
    TaskPack,
    register_pack,
    skills_for,
)

__all__ = [
    "FinalizerSpec",
    "PACKS",
    "SkillSpec",
    "TaskPack",
    "register_pack",
    "skills_for",
]
