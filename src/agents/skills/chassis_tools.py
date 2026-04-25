"""Chassis tools: list_skills, load_skill, submit_final.

These are always-on tools registered when the active task has a TaskPack
or when Stage2DeepAgentConfig.enable_chassis_tools=True.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool

from agents.skills.registry import PACKS, skills_for


def build_chassis_tools(runtime: Any) -> tuple[BaseTool, BaseTool, BaseTool]:
    """Construct the three chassis tools bound to one runtime state."""
    unknown_skill_loads: dict[str, int] = {}

    @tool
    def list_skills() -> str:
        """List skills available for the current task. Returns JSON array of {name, description}."""
        catalog = [
            {"name": s.name, "description": s.description}
            for s in skills_for(runtime.task_type)
        ]
        text = json.dumps(catalog, ensure_ascii=False)
        runtime.record("list_skills", {}, text)
        return text

    @tool
    def load_skill(skill_name: str) -> str:
        """Fetch the full instructions for a skill. Returns markdown body. Records load."""
        catalog = {s.name: s for s in skills_for(runtime.task_type)}
        if skill_name not in catalog:
            unknown_skill_loads[skill_name] = unknown_skill_loads.get(skill_name, 0) + 1
            available = sorted(catalog.keys())
            err = (
                f"ERROR: skill {skill_name!r} not registered for task_type "
                f"{runtime.task_type}; available: {available}"
            )
            runtime.record("load_skill", {"skill_name": skill_name}, err)
            if unknown_skill_loads[skill_name] >= 2:
                raise RuntimeError(
                    f"repeated unknown skill load: {skill_name!r}; available: {available}"
                )
            return err

        spec = catalog[skill_name]
        body = spec.body_path.read_text(encoding="utf-8")
        runtime.skills_loaded.add(skill_name)
        runtime.record("load_skill", {"skill_name": skill_name}, body)
        return body

    @tool
    def submit_final(
        payload: dict,
        rationale: str,
        evidence_refs: list[dict] | None = None,
    ) -> str:
        """Submit the final task answer. Payload must match this task's FinalizerSpec.schema.
        The chassis validates payload + preconditions; on success, terminates the run."""
        pack = PACKS.get(runtime.task_type)
        if pack is None:
            err = f"ERROR: no pack registered for {runtime.task_type}; cannot submit_final"
            runtime.record("submit_final", {"payload": payload}, err)
            return err
        try:
            validated = pack.finalizer.validator(payload, runtime)
            adapted = pack.finalizer.adapter(validated, runtime)
        except Exception as exc:
            err = f"ERROR: submit_final validation failed: {exc}"
            runtime.record("submit_final", {"payload": payload}, err)
            return err

        # Stash the resolved payload onto the runtime so build_agent's
        # downstream normalization can pick it up.
        runtime.bundle = runtime.bundle.model_copy(
            update={"extra_metadata": {**(runtime.bundle.extra_metadata or {}),
                                       "stage2_submission": adapted}}
        )
        msg = (
            f"submitted; rationale={rationale!r}; "
            f"evidence_refs={len(evidence_refs or [])}"
        )
        runtime.record(
            "submit_final",
            {"payload": payload, "rationale": rationale, "evidence_refs": evidence_refs or []},
            msg,
        )
        return msg

    return list_skills, load_skill, submit_final


__all__ = ["build_chassis_tools"]
