"""Pack-level static validation, run before the first LLM call."""
from __future__ import annotations

from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
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

    # 1. all skill body files actually readable (open + decode UTF-8, not just exists())
    for skill in pack.skills:
        try:
            skill.body_path.read_text(encoding="utf-8")
        except (FileNotFoundError, IsADirectoryError, PermissionError, UnicodeDecodeError, OSError) as exc:
            raise RuntimeError(
                f"skill body not readable: {skill.body_path} (skill={skill.name}): {exc}"
            ) from exc

    # 2. unique skill names within pack
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

    # 6. tool names: pack-internal uniqueness + cross-pack uniqueness
    stub_runtime = Stage2RuntimeState(bundle=Stage2EvidenceBundle())
    stub_runtime.task_type = task_type
    try:
        my_tools = pack.tool_builder(stub_runtime)
    except Exception as exc:
        raise RuntimeError(
            f"tool_builder for {task_type} cannot enumerate tools without a "
            f"populated task_ctx; static name validation impossible. Underlying: {exc}"
        ) from exc
    my_tool_names = [t.name for t in my_tools]
    if len(set(my_tool_names)) != len(my_tool_names):
        raise RuntimeError(
            f"duplicate tool name within pack {task_type}: {my_tool_names}"
        )
    other_pack_tool_names: set[str] = set()
    for other_type, other_pack in PACKS.items():
        if other_type == task_type:
            continue
        try:
            other_tools = other_pack.tool_builder(
                Stage2RuntimeState(bundle=Stage2EvidenceBundle())
            )
        except Exception as exc:
            raise RuntimeError(
                f"tool_builder for {other_type} cannot enumerate tools without a "
                f"populated task_ctx; static name validation impossible. Underlying: {exc}"
            ) from exc
        other_pack_tool_names.update(t.name for t in other_tools)
    collision = set(my_tool_names) & other_pack_tool_names
    if collision:
        raise RuntimeError(
            f"tool name collision between {task_type} and other packs: {sorted(collision)}"
        )


__all__ = ["validate_packs"]
