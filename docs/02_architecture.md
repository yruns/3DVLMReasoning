# 02 — Architecture

Component tree + data flow + design motivation. All file paths are relative to the repo root. All line numbers are against HEAD = `a8e651e`.

## 2.1 Two-stage pipeline (highest-level)

```
         question (natural language)
                   │
                   ▼
  ┌────────────────────────────────────────┐
  │ Stage 1: task-conditioned retrieval    │
  │ src/query_scene/keyframe_selector.py   │
  │                                        │
  │   query → HypothesisOutputV1           │
  │   hypothesis ranked execution          │
  │     DIRECT > PROXY > CONTEXT           │
  │   joint-coverage view selection        │
  │   pad to min(k, 3) keyframes           │
  └───────────────────┬────────────────────┘
                      │
                      │ KeyframeResult
                      ▼
         ┌────────────────────────────┐
         │ Stage 1 → Stage 2 bridge   │
         │ src/agents/stage1_adapters │
         │                            │
         │   build_stage2_evidence_   │
         │   bundle(): prior + ctx    │
         └─────────────┬──────────────┘
                       │ Stage2EvidenceBundle
                       ▼
  ┌────────────────────────────────────────┐
  │ Stage 2: evidence-seeking VLM agent    │
  │ src/agents/runtime/deepagents_agent.py │
  │                                        │
  │   DeepAgents graph (LangChain v1)      │
  │   5 core tools + 2 VG tools            │
  │   iterative evidence refinement loop   │
  │   uncertainty-aware stopping           │
  └───────────────────┬────────────────────┘
                      │
                      │ Stage2StructuredResponse
                      ▼
          (answer, confidence, cited frames,
           uncertainties, tool trace)
                      │
                      ▼
  ┌────────────────────────────────────────┐
  │ E2E guard / nudge (OpenEQA pilot only) │
  │ openeqa_official_question_pilot.py     │
  │                                        │
  │   if status != completed               │
  │   OR confidence < confidence_guard:    │
  │     rerun Stage 2 with callbacks on    │
  └───────────────────┬────────────────────┘
                      ▼
          final prediction → eval
```

## 2.2 Module inventory (orientation table)

Only modules that carry architectural weight are listed. Peer-level `tests/`, `examples/` (except the canonical pilot), and `__pycache__` omitted. Line counts are approximate HEAD values for context only.

| Area | Path | Role |
|---|---|---|
| **Benchmarks** | `src/benchmarks/base.py` | `BenchmarkAdapter` base class + `BenchmarkSample` (commit `493931a`, Phase 1-2) |
|  | `src/benchmarks/openeqa_loader.py` | Vanilla OpenEQA JSON loader. **Caveat** — expects `data/frames/<episode>` that does NOT exist locally; see `docs/07_data_layout.md` |
|  | `src/benchmarks/openeqa_official_eval.py` | Wrapper around upstream `openeqa.evaluation.llm_match`, swaps LLM backend; MNAS scaling here |
|  | `src/benchmarks/embodiedscan_loader.py` | EmbodiedScan VG loader (commit `493931a`) |
|  | `src/benchmarks/embodiedscan_eval.py` | Oriented 3D IoU evaluation (commit `e6a8412`) |
|  | `src/benchmarks/sqa3d_loader.py`, `scanrefer_loader.py` | Loaders only; no Stage-2 glue exercised at HEAD |
| **Stage 1 — retrieval** | `src/query_scene/keyframe_selector.py` | `KeyframeSelector` class + `select_keyframes_v2` public API |
|  | `src/query_scene/index_builder.py` | Hierarchical CLIP index (region / object / point) + FAISS/numpy back-ends |
|  | `src/query_scene/query_parser.py`, `src/query_scene/parsing/` | LLM-based query → `HypothesisOutputV1` parsing |
|  | `src/query_scene/core/hypotheses.py` | `HypothesisKind`, `ParseMode`, `QueryHypothesis`, `HypothesisOutputV1` dataclasses |
|  | `src/query_scene/query_executor.py` | Executes a parsed hypothesis against the scene indices |
|  | `src/query_scene/spatial_relations.py` | Geometric checks (above/below/between/near/…); used by PROXY hypotheses |
|  | `src/query_scene/bev_builder.py` | BEV rendering for multimodal query parsing |
|  | `schema/hypothesis_output_v1.json` | Pydantic JSON schema for the parser output |
| **Stage 1 → Stage 2 bridge** | `src/agents/stage1_adapters.py` | `build_stage2_evidence_bundle()` — the canonical bridge |
|  | `src/agents/adapters.py` | Backward-compat shim (same exports) |
|  | `src/agents/adapters/openeqa_adapter.py` | `BenchmarkAdapter` wrapper for OpenEQA (Phase 7, commit `60d6f51`) |
|  | `src/agents/adapters_pkg/` | Benchmark-adapter scaffolding |
|  | `src/agents/benchmark_adapters.py` | Multi-benchmark adapter with frame / mock modes |
|  | `src/agents/stage1_callbacks.py` | External Stage-1 callbacks for `request_more_views` / `request_crops` / `switch_or_expand_hypothesis` tools |
| **Stage 2 — agent runtime** | `src/agents/stage2_deep_agent.py` | Compatibility shim preserving the old public class `Stage2DeepResearchAgent` |
|  | `src/agents/runtime/__init__.py` | Runtime package exports |
|  | `src/agents/runtime/base.py` | `BaseStage2Runtime` + `Stage2RuntimeState` + `build_system_prompt` + uncertainty stopping |
|  | `src/agents/runtime/langchain_agent.py` | `ToolChoiceCompatibleAzureChatOpenAI` adapter |
|  | `src/agents/runtime/deepagents_agent.py` | `DeepAgentsStage2Runtime` — concrete runtime, the `run()` loop, 7 tool decorators |
|  | `src/agents/models.py` | Re-exports schemas from `core/` for back-compat |
|  | `src/agents/core/` | `agent_config.py` (config + enums), `task_types.py` (bundle + response schemas) |
| **Stage 2 — tools** | `src/agents/tools/request_crops.py` | Red-bbox PIL crop assembly |
|  | `src/agents/tools/hypothesis_repair.py` | Hypothesis-switch payloads |
|  | `src/agents/tools/select_object.py` | VG tool: scene-graph object → 9-DOF bbox |
|  | `src/agents/tools/spatial_compare.py` | VG tool: rank objects by 3D distance |
| **Tracing** | `src/agents/trace.py` | Per-run tool + evidence trace recorder |
|  | `src/agents/trace_server.py` | HTML rendering of traces |
| **Evaluation harness** | `src/evaluation/batch_eval.py` | Mock / generic parallel harness (not the production path for OpenEQA) |
|  | `src/evaluation/scripts/run_openeqa_stage2_full.py` | Legacy entry for the mock / frame-based path |
|  | `src/agents/examples/openeqa_official_question_pilot.py` | **Production OpenEQA entry point** |
|  | `src/agents/examples/openeqa_single_scene_pilot.py` | Single-scene iteration helper (`--mode stage1/stage2/e2e/all`) |
|  | `src/agents/examples/embodiedscan_vg_pilot.py` | EmbodiedScan VG entry (commit `60d6f51` Phase 6) |
|  | `src/evaluation/ablations/run_*.py` | Ablation-specific entry points (views-only, crops-only, hypothesis-repair-only, one-shot, uncertainty) |

## 2.3 Evidence bundle — the inter-stage contract

`Stage2EvidenceBundle` (defined in `src/agents/core/task_types.py`, re-exported via `src/agents/models.py`) is the single hand-off from Stage 1 to Stage 2. Its shape drives every design decision downstream.

| Field | Source | Consumer | Purpose |
|---|---|---|---|
| `keyframes: list[KeyframeEvidence]` | `KeyframeResult.keyframe_paths` + `frame_mappings` | Stage 2 initial image payload | Minimum evidence set |
| `bev_image_path: str \| None` | `bev/` output | Stage 2 optional multimodal input | Scene overview |
| `scene_summary: str` | `"OpenEQA scene X with N detected objects."` default | Stage 2 system prompt | One-line context |
| `object_context: dict[str, str]` | `enriched_objects.json` via `build_object_context` (adapters) | Stage 2 system prompt (v14) + `retrieve_object_context` tool | Per-object enrichment |
| `hypothesis: Stage1HypothesisSummary` | `KeyframeResult.metadata.selected_hypothesis_*` | Stage 2 `inspect_stage1_metadata` tool | Typed retrieval mode signal |
| `extra_metadata: dict` | Full Stage 1 dump | Stage 2 debugging + VG candidates | Escape hatch |

The bundle is deep-copied before each Stage 2 run so callbacks can mutate it without affecting the source.

## 2.4 Design motivations (why these components exist)

| Component | Why it exists |
|---|---|
| Ranked hypothesis tree (Stage 1) | Because CLIP alone mis-fires on compositional 3D queries; the DIRECT/PROXY/CONTEXT fallback was empirically necessary to get > 0 keyframes on queries whose literal anchor is absent. Added pre-v9, refined through v11's open-ended mode (`02ea2f3`) |
| Joint coverage + padding | Because a single keyframe often omits either the target or the anchor; v11 padding (`keyframe_selector.py:1684`) addressed 18/36 v10 low-score cases caused by single-frame fragility |
| `request_more_views` / `request_crops` / `switch_or_expand_hypothesis` | To make evidence acquisition a *policy*, not a pre-commit at Stage 1 time. v10 had zero callback calls; v11 enabled callbacks and recorded 61 invocations (`10_experiment_log/v11_callbacks_20260330.md`) |
| Uncertainty-aware stopping | Because 42 % of v9 runs ended with `insufficient_evidence` but the model was being asked to answer anyway; downgrading to `insufficient_evidence` when confidence < threshold prevents hallucinated answers |
| E2E nudge + guard | To handle two failure modes: (a) agent gives up prematurely (nudge), (b) agent reports completed with wrong confidence (guard triggers E2E rerun) |
| Scene inventory injection (v14) | Because 51 % of v13 failures had the answer in `enriched_objects.json` but the agent never called `retrieve_object_context`. Pre-injecting dominates tool-gating at current VLM capability (Claim 2 in `00_research_manifest.md`) |
| VG tool gating (commit `9988be9`) | To preserve a single runtime across task types without polluting QA with VG-only tools |

## 2.5 What is NOT part of the architecture at HEAD

| Thing | Status | Reason |
|---|---|---|
| Active-exploration (video-based) OpenEQA | Out of scope | Local data is EM-EQA only |
| OpenEQA HM3D split | Out of scope | 89 ScanNet clips locally; HM3D prep not done |
| Fine-tuned VLM | Out | Backbones are off-the-shelf Azure endpoints |
| Reinforcement-learning loop | Out | The confidence gate is a handcrafted policy, not learned |
| Video transformer / frame-interpolation | Out | Key-frame selection is geometric + semantic |
| SQA3D / ScanRefer inference | Loaders only | No Stage-2 adapter wired in |

## 2.6 Cross-references

- Data entering Stage 1: `docs/07_data_layout.md`.
- Stage 1 deep dive: `docs/03_stage1_retrieval.md`.
- Stage 2 deep dive: `docs/04_stage2_agent.md`.
- Evaluation flow after Stage 2: `docs/05_evaluation.md`.
- Per-version evolution of each component: `docs/06_evolution_v9_to_v14.md`.
