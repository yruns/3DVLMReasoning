"""Pack-level static validation, run before the first LLM call."""
from __future__ import annotations

from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.skills.registry import PACKS


def validate_packs(
    task_type: Stage2TaskType,
    bundle: Stage2EvidenceBundle,
    *,
    require_pack: bool = False,
) -> None:
    """Validate that the active task's pack is well-formed and the bundle
    carries everything the pack needs. Raises RuntimeError on the first
    contract violation. No-ops when no pack is registered AND
    require_pack is False (the legacy QA path)."""
    pack = PACKS.get(task_type)
    if pack is None:
        if require_pack:
            raise RuntimeError(f"no pack registered for task_type={task_type}")
        return

    # 1. all skill body files exist + readable
    for skill in pack.skills:
        if not skill.body_path.exists() or not skill.body_path.is_file():
            raise RuntimeError(
                f"skill body not readable: {skill.body_path} (skill={skill.name})"
            )

    # 2. unique tool + skill names within pack
    skill_names = [s.name for s in pack.skills]
    if len(set(skill_names)) != len(skill_names):
        raise RuntimeError(f"duplicate skill name in pack {task_type}: {skill_names}")

    # 3. required_primary_skill is in pack
    if pack.required_primary_skill not in skill_names:
        raise RuntimeError(
            f"required_primary_skill {pack.required_primary_skill!r} not in "
            f"pack {task_type} skills {skill_names}"
        )

    # 4. bundle.extra_metadata carries every required key
    extra = bundle.extra_metadata or {}
    missing = [k for k in pack.required_extra_metadata if k not in extra]
    if missing:
        raise RuntimeError(
            f"bundle missing required extra_metadata for {task_type}: {missing}"
        )

    # 5. ctx_factory returns non-None
    ctx = pack.ctx_factory(bundle)
    if ctx is None:
        raise RuntimeError(
            f"ctx_factory for {task_type} returned None"
        )


__all__ = ["validate_packs"]
