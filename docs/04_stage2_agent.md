# 04 — Stage 2: DeepAgents VLM Runtime

Deep dive into `src/agents/runtime/` and `src/agents/tools/`. Line numbers against HEAD = `a8e651e`.

## 4.1 Public contract

**Entry**: `Stage2DeepResearchAgent(config, crop_callback, text_frame_selector).run(task, bundle)` (compatibility shim at `src/agents/stage2_deep_agent.py`).
**Actual runtime**: `DeepAgentsStage2Runtime` at `src/agents/runtime/deepagents_agent.py:33-692`.
**Input**: `Stage2TaskSpec` (task type, user query, plan mode, max reasoning turns, optional output schema) + `Stage2EvidenceBundle` (BEV, scene context, hypothesis prior, and metadata; see `02_architecture.md §2.3`). It does not carry first-person seed frames.
**Output**: `Stage2AgentResult(task, result: Stage2StructuredResponse, tool_trace, final_bundle)`.

## 4.2 Runtime lifecycle

```
Stage2DeepResearchAgent.run(task, bundle)
  │
  ├─ DeepAgentsStage2Runtime.build_agent(task, bundle)
  │    │
  │    ├─ build_runtime_tools(runtime) ────► list[BaseTool]
  │    ├─ build_subagents(task) ─────────── only in FULL plan mode
  │    ├─ build_system_prompt(task, object_context)
  │    └─ create_deep_agent(llm, tools, subagents, system_prompt) → LangChain graph
  │
  ├─ build_user_message(task, runtime)
  │
  └─ while turns_used < task.max_reasoning_turns:  ← loop at deepagents_agent.py:602
        raw_state = graph.invoke({"messages": messages})
        structured = raw_state.get("structured_response")

        if structured.status in (COMPLETED, FAILED):         break
        if runtime.consume_evidence_update():                inject evidence_update_message; continue
        if structured.status in (INSUFFICIENT_, NEEDS_MORE): inject _build_evidence_nudge; continue
        break                                                # no new evidence, no more work

      final_response = normalize_final_response(task, raw_state)
      final_response = apply_uncertainty_stopping(final_response, can_acquire_more_evidence)
      return Stage2AgentResult(...)
```

Three exit conditions, in priority order:
1. **Structured completion** — agent emits `COMPLETED` or `FAILED`.
2. **Evidence injection** — a selector / crop / mark tool returns image metadata in its tool observation; `build_evidence_update_message` injects only those tool-acquired images, then the loop continues.
3. **Nudge** — agent reported `INSUFFICIENT_EVIDENCE` or `NEEDS_MORE_EVIDENCE` and turns remain; a follow-up message urges tool use (`_build_evidence_nudge`, line 355; injected at line 644).

Post-loop, `apply_uncertainty_stopping` (`runtime/base.py:415-490`) may downgrade a `COMPLETED` response with sub-threshold confidence to `INSUFFICIENT_EVIDENCE` when the loop cannot acquire more evidence.

## 4.3 Tool Inventory

All tools are registered in `build_runtime_tools` (`deepagents_agent.py:84-249`). VG tools are gated on `runtime.task_type == Stage2TaskType.VISUAL_GROUNDING and runtime.vg_scene_objects is not None` (lines 204-208). Every call is logged via `runtime.record()` (`runtime/base.py:51-64`), which appends to `runtime.tool_trace` — the data source for the "tool under-use" diagnostic in `00_research_manifest.md` Claim 4.

| Tool group | Tools | Side-effect |
|---|---|---|
| Scene/catalog reads | `view_bev`, `list_scene_proposals`, `list_frame_proposals`, `inspect_proposal`, `retrieve_object_context` | read-only except `view_bev` image display |
| Active frame selection | `select_by_text`, `select_by_proposal`, `select_by_frame_neighbor`, `select_by_region`, `select_by_coverage` | records selected first-person RGB paths in tool-trace image metadata |
| Active image refinement | `mark_frame_with_bbox`, `request_crops` | records annotated frames or crops in tool-trace image metadata |
| Finalization / VG reasoning | `submit_final`, `compare_proposals_spatial` | records final payload or read-only ranking |

Design principle: **every tool that returns new visual evidence marks `evidence_updated`, every read-side tool does not**. The loop depends on this distinction to decide whether to inject an evidence-update message.

## 4.4 Evidence bundle + state

| Object | Location | Purpose |
|---|---|---|
| `Stage2EvidenceBundle` | `src/agents/core/task_types.py` | inter-stage contract (see `02_architecture.md §2.3`) |
| `Stage2RuntimeState` | `src/agents/runtime/base.py:33-70` | mutable runtime: `bundle`, `tool_trace`, `task_type`, VG-specific fields (`vg_scene_objects`, `vg_selected_bbox_3d`, `vg_selection_rationale`), an `_evidence_updated` flag |
| `Stage2StructuredResponse` | `src/agents/core/task_types.py` | final output schema: `task_type`, `status`, `summary`, `confidence ∈ [0,1]`, `uncertainties: list[str]`, `cited_frame_indices`, `evidence_items`, `plan`, `payload` |

`runtime.consume_evidence_update()` (`runtime/base.py:67-72`) is a one-shot read: `True` if set since the last consume, then resets. This guarantees each loop iteration only injects new evidence once.

## 4.5 Uncertainty-aware stopping

`BaseStage2Runtime.apply_uncertainty_stopping` at `runtime/base.py:415-502` — the post-loop normalisation step. Three cases:

| Case | Trigger | Action |
|---|---|---|
| 1 | `status == COMPLETED` AND `confidence < threshold` AND `can_acquire_more_evidence == False` | Downgrade to `INSUFFICIENT_EVIDENCE`; append uncertainty string with the specific confidence and threshold |
| 2 | `status == INSUFFICIENT_EVIDENCE` | Passes through (agent correctly self-reported) |
| 3 | `status == NEEDS_MORE_EVIDENCE` AND `can_acquire_more_evidence == False` | Upgrade to `INSUFFICIENT_EVIDENCE` (final) |

Defaults (`src/agents/core/agent_config.py:58-69`):
- `confidence_threshold = 0.4`
- `enable_uncertainty_stopping = True`

`can_acquire_more_evidence` is `True` when selector / crop / mark tools remain available and turns remain. Evidence injection is now tool-driven, not prep-driven.

## 4.6 E2E nudge loop

`DeepAgentsStage2Runtime._build_evidence_nudge` at `deepagents_agent.py:355-430`.

Triggered in `run()` at `:635-655` when:
- `structured_response.status ∈ {INSUFFICIENT_EVIDENCE, NEEDS_MORE_EVIDENCE}`
- `turns_used < task.max_reasoning_turns`

The nudge message (built from the response's listed uncertainties + available callbacks) is appended and the loop continues. Introduced in commit `80ebf21` (2026-03-27) after a 100-case analysis showed 29 cases of premature give-up despite remaining budget.

**Interaction with the outer E2E guard**: the *inner* nudge (above) fires within a single `run()` invocation. The *outer* E2E guard (in `openeqa_official_question_pilot.py:519-555`) triggers a whole second `run()` of Stage 2, with callbacks enabled, if the first run ends with `status != completed` OR `confidence < --confidence-guard` (default 0.6). Both levels exist because the first run often has callbacks disabled for speed, and the guard gives it a second chance with tools.

## 4.7 System-prompt evolution (v12 → v14)

`BaseStage2Runtime.build_system_prompt` at `runtime/base.py:312-413` is the single composition site. Its structure, annotated with version introduction:

```
Research role: ........................... base (pre-v9)
CRITICAL — Evidence-seeking protocol: ..... v11 callback era
Tool strategy (use in this order): ........ v11
MANDATORY tool-usage rules: ............... v12 (2abb404)
  - request_more_views when answer contradicts premise
  - request_crops for color/attribute (v13 44b9600 strengthened)
  - YES/NO state questions must tool-verify (v13)
  - BETWEEN: verify both anchors (v12)
  - Do NOT report confidence > 0.7 with zero tools (v13)
SELF-CHECK before final answer: ........... v13
  - Answer must not contradict premise
  - List 2-3 candidate objects before selecting
  - Warning: correct answer may be smaller / less prominent (v13)
Uncertainty-aware stopping: ............... v11 runtime; text in prompt
[VG section, if VG task] .................. commit 12d71ae (Phase 4)
Scene object inventory: ................... v14 (b4197a1) — injected via
                                              _format_scene_inventory(object_context)
                                              at build_system_prompt:395
Framework constraints: .................... base
Unified output contract: .................. base
Task-specific instruction + schema ........ base
```

HEAD spot-check: the verbatim line *"Stage 1 is a high-recall evidence retriever, not ground truth. Stage 2 must verify, repair, or reject Stage-1 hypotheses using pixels."* appears at `runtime/base.py:358-359`. This is the thesis sentence of the B1 angle (`11_academic_angles_catalog.md`) and the anchor for Claim 1 in `00_research_manifest.md`.

## 4.8 LLM client

`ToolChoiceCompatibleAzureChatOpenAI` at `src/agents/runtime/langchain_agent.py:10-38`:
- Subclass of LangChain `AzureChatOpenAI`.
- Overrides `bind_tools` to accept `tool_choice` consistently across Azure / OpenAI endpoints.
- Stable `session_id` in `extra_body` enables Azure-side prompt caching — the project deliberately uses a single-key client (no pool rotation) so cached prompt blocks stay warm across turns.

Config source: `Stage2DeepAgentConfig` at `src/agents/core/agent_config.py:40-69`. Defaults:

| Field | Default |
|---|---|
| `base_url` | `https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl` |
| `model_name` | `gpt-5.4-2026-03-05` (since v12 `2abb404`) |
| `api_version` | `2024-03-01-preview` |
| `max_tokens` | 10000 |
| `temperature` | 0.1 |
| `timeout` | 120 |
| `max_retries` | 2 |
| `max_images` | 6 (per message payload) |
| `image_max_size` | 900 |
| `enable_subagents` | True |
| `confidence_threshold` | 0.4 |
| `enable_uncertainty_stopping` | True |

Pilot override (`openeqa_single_scene_pilot.py:337-346`): `max_tokens=4000`, `max_images=6` — overrides the config default for evaluation runs.

## 4.9 Plan modes

`Stage2PlanMode` in `src/agents/core/agent_config.py:23-29`:
- `OFF` — todo list is optional; used for speed runs.
- `BRIEF` — short 2-4 item todo list (default at `run_stage2` call site).
- `FULL` — explicit todo list throughout; enables DeepAgents subagent decomposition. Subagents are configured in `build_subagents` at `deepagents_agent.py:251-281`.

The plan mode feeds into `build_system_prompt` via the `plan_instructions` block (`runtime/base.py:318-330`). In the 1050Q production runs the mode is `BRIEF`; `FULL` is used for experimental / ablation runs.

## 4.10 Tool trace → evidence items → predictions

Every tool call appends to `runtime.tool_trace: list[Stage2ToolResult]`. The final `Stage2AgentResult.tool_trace` is the durable record of which selector or image-refinement tools produced visual evidence.

For a QA task, `payload["answer"]` is the final natural-language string consumed by `extract_prediction_text` in the pilot. For VG, the runtime exports `vg_selected_object_id`, `vg_selected_bbox_3d`, `vg_selection_rationale` to `raw_state` (lines 681-684) — not through `payload`.

## 4.11 What is NOT in Stage 2

| Missing | Reason / alternative |
|---|---|
| Learned stopping policy | confidence gate is handcrafted; see `00_research_manifest.md` Claim 3 REQUIRES |
| Cross-turn tool memory beyond `tool_trace` | the agent re-reads the trace via `inspect_stage1_metadata` + own message history |
| Automatic image compression | `image_max_size=900` and `max_images=6` are static budgets |
| Per-task specialised prompts beyond QA / VG | `nav_plan` and `manipulation` task types are declared (`Stage2TaskType` enum) but no tool / prompt specialisation exists on HEAD |

Cross-ref: `03_stage1_retrieval.md` for what flows in; `05_evaluation.md` for what happens to the output.
