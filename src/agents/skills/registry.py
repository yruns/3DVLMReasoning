"""SkillSpec / TaskPack registry shared across Stage-2 packs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import BaseTool

from agents.core.agent_config import Stage2TaskType
from agents.skills.finalizer import FinalizerSpec


@dataclass(frozen=True)
class SkillSpec:
    """A loadable skill: catalog entry + lazy markdown body."""

    name: str
    description: str
    body_path: Path
    task_types: frozenset[Stage2TaskType] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # Allow set inputs by normalizing to frozenset
        object.__setattr__(self, "task_types", frozenset(self.task_types))


@dataclass(frozen=True)
class TaskPack:
    """Per-task plug-in: tools, skills, finalizer, ctx factory."""

    task_type: Stage2TaskType
    tool_builder: Callable[[Any], list[BaseTool]]
    skills: list[SkillSpec]
    finalizer: FinalizerSpec
    required_primary_skill: str
    required_extra_metadata: list[str]
    ctx_factory: Callable[[Any], Any]


PACKS: dict[Stage2TaskType, TaskPack] = {}


def register_pack(pack: TaskPack) -> None:
    if pack.task_type in PACKS:
        raise RuntimeError(f"duplicate pack: {pack.task_type}")
    PACKS[pack.task_type] = pack


def skills_for(task_type: Stage2TaskType) -> list[SkillSpec]:
    pack = PACKS.get(task_type)
    if pack is None:
        return []
    return [s for s in pack.skills if task_type in s.task_types]


__all__ = [
    "SkillSpec",
    "TaskPack",
    "FinalizerSpec",
    "PACKS",
    "register_pack",
    "skills_for",
]
