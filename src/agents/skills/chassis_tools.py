"""Chassis tools: list_skills, load_skill, submit_final.

These are always-on tools registered when the active task has a TaskPack
or when Stage2DeepAgentConfig.enable_chassis_tools=True.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from agents.skills.registry import PACKS, skills_for
from agents.skills.tadg import evaluate_tadg, tadg_record_fields


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
        tool_override_reason: str | None = None,
    ) -> str:
        """Submit the final task answer. Payload must match this task's
        FinalizerSpec.schema. The chassis validates payload + preconditions
        and stashes the adapted result on the bundle for the run loop to
        consume. (Run termination is wired by the active pack's runtime
        integration; without that, this tool is a no-op.)

        When TADG (`runtime.use_tool_answer_disagreement_gate`) is enabled
        and the submitted `proposal_id` disagrees with the most recent
        matched-relation `compare_proposals_spatial` rank-1, this tool
        returns a soft-block message instead of finalizing. Pass a
        non-empty `tool_override_reason` (≥ `tadg_override_min_chars`
        characters after strip) to record an explicit divergence and
        proceed.
        """
        # TADG (Tool-Answer Disagreement Gate) — runs BEFORE chassis
        # validator so a blocked submission never produces a final
        # adapted result. Spec: tmp/tadg_spec.md, src/agents/skills/tadg.py.
        #
        # Unwrap nested {payload: {payload: ...}} BEFORE the gate so
        # TADG sees the same proposal_id the chassis validator will
        # consume. Without this unwrap, an agent that nests its payload
        # (a real shape — see test_submit_final_unwraps_structured_response_payload_for_qa)
        # would have proposal_id read from the outer dict (None) and
        # bypass the gate entirely.
        gate_payload: Any = payload
        if isinstance(payload, dict):
            inner = payload.get("payload")
            if isinstance(inner, dict) and "proposal_id" in inner:
                gate_payload = inner
        decision = evaluate_tadg(
            runtime, gate_payload, tool_override_reason=tool_override_reason,
        )
        if decision.blocked:
            runtime.record(
                "submit_final",
                {
                    "payload": payload,
                    "rationale": rationale,
                    "evidence_refs": evidence_refs or [],
                    "tool_override_reason": tool_override_reason,
                    **tadg_record_fields(decision),
                },
                f"TADG_BLOCK: {decision.message}",
            )
            return f"TADG_BLOCK: {decision.message}"

        pack = PACKS.get(runtime.task_type)
        if pack is None:
            err = f"ERROR: no pack registered for {runtime.task_type}; cannot submit_final"
            runtime.record("submit_final", {"payload": payload}, err)
            return err

        # Coerce raw dict payload into the pack's typed payload_model
        # (e.g. Pydantic BaseModel) before handing it to the validator.
        # Stub finalizers that use payload_model=dict (or other types
        # without `model_validate`) pass through unchanged.
        payload_model = pack.finalizer.payload_model
        if hasattr(payload_model, "model_validate"):
            candidates = [payload]
            nested_payload = payload.get("payload") if isinstance(payload, dict) else None
            if isinstance(nested_payload, dict):
                candidates.append(nested_payload)
            last_exc: ValueError | TypeError | None = None
            for candidate in candidates:
                try:
                    typed_payload = payload_model.model_validate(candidate)
                    break
                except (ValueError, TypeError) as exc:
                    last_exc = exc
            else:
                err = f"ERROR: submit_final payload schema mismatch: {last_exc}"
                runtime.record("submit_final", {"payload": payload}, err)
                return err
        else:
            typed_payload = payload

        try:
            validated = pack.finalizer.validator(typed_payload, runtime)
            adapted = pack.finalizer.adapter(validated, runtime)
        except (ValueError, TypeError, KeyError, ValidationError) as exc:
            err = f"ERROR: submit_final validation failed: {exc}"
            runtime.record("submit_final", {"payload": payload}, err)
            return err

        # Stash the resolved payload onto the runtime so build_agent's
        # downstream normalization can pick it up.
        runtime.bundle = runtime.bundle.model_copy(
            update={"extra_metadata": {**(runtime.bundle.extra_metadata or {}),
                                       "stage2_submission": adapted}}
        )
        # Terminal signal: the run loop polls this and exits as soon
        # as it's set, so submit_final actually ends the agent run.
        runtime.final_submission = adapted
        msg = (
            f"submitted; rationale={rationale!r}; "
            f"evidence_refs={len(evidence_refs or [])}"
        )
        # Always emit the canonical TADG record fields (None values when
        # the gate was silent). This keeps BLOCK and SUCCESS records on
        # the same shape so downstream telemetry can collate without
        # schema gymnastics. See tadg.tadg_record_fields docstring.
        record_payload = {
            "payload": payload,
            "rationale": rationale,
            "evidence_refs": evidence_refs or [],
            **tadg_record_fields(decision),
        }
        if tool_override_reason:
            record_payload["tool_override_reason"] = tool_override_reason
        runtime.record("submit_final", record_payload, msg)
        return msg

    return list_skills, load_skill, submit_final


__all__ = ["build_chassis_tools"]
