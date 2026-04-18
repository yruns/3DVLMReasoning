# 08 — Benchmarks Catalog

Status of each benchmark this project loads, evaluates, or claims to support. "Readiness" is scored at HEAD = `a8e651e`; lift readiness to green only when the end-to-end pipeline plus a named evaluator has been exercised.

## 8.1 Readiness matrix

| Benchmark | Split / scope | Loader | Stage-1 adapter | Stage-2 adapter | Evaluator | Scripted entry | HEAD readiness |
|---|---|---|---|---|---|---|---|
| **OpenEQA ScanNet EM-EQA** | Episodic memory, 1050 questions | `src/benchmarks/openeqa_loader.py` (vanilla; flawed for local paths — see `07_data_layout.md`) | `src/query_scene/keyframe_selector.py` (full) | `src/agents/adapters/openeqa_adapter.py` (wrapper) + production path via `src/agents/stage1_adapters.py` | `src/benchmarks/openeqa_official_eval.py` — monkey-patched upstream | `src/agents/examples/openeqa_official_question_pilot.py` | **green** — 73.1 MNAS at v14, 1050Q |
| **EmbodiedScan VG** | Visual grounding on prepared EmbodiedScan scenes | `src/benchmarks/embodiedscan_loader.py` (Phase 1-2, commit `493931a`) | shared `KeyframeSelector` (v9988be9 added `pcd_np + bbox_np` loading for precise 9-DOF bbox) | `src/agents/adapters_pkg/` + VG-specific tools `src/agents/tools/select_object.py`, `src/agents/tools/spatial_compare.py` (commit `9988be9`) | `src/benchmarks/embodiedscan_eval.py` — oriented 3D IoU (Phase 3, commit `e6a8412`) | `src/agents/examples/embodiedscan_vg_pilot.py` (Phase 6, commit `60d6f51`) | **yellow** — 3-sample smoke only (`9988be9`); mini / full splits not yet run |
| **SQA3D** | Situated QA on ScanNet | `src/benchmarks/sqa3d_loader.py` + tests | — (no integration wired) | Referenced by `src/agents/benchmark_adapters.py` in the mock / frame-based harness only | none on HEAD | none | **red** — loader only; no Stage-2 run has been attempted |
| **ScanRefer** | Referring expressions on ScanNet | `src/benchmarks/scanrefer_loader.py` + tests | — (no integration wired) | Referenced by `src/agents/benchmark_adapters.py` in the mock / frame-based harness only | none on HEAD | none | **red** — loader only; no Stage-2 run has been attempted |

"Yellow" for EmbodiedScan VG reflects that the infrastructure is wired end-to-end (loader → Stage 1 → Stage 2 → eval) and has produced smoke numbers, but the sample size is three.

## 8.2 OpenEQA ScanNet (production benchmark)

Scope actually exercised:
- Split: ScanNet-only (89 clips covered locally; HM3D not prepared — see `07_data_layout.md`).
- Subset: full 1050-question ScanNet split for v13 and v14; 100Q (20 scenes × 5 questions) for v9 – v12 iteration. The 100Q sub-sampling is implemented as `--num-scenes 20 --questions-per-scene 5` in the pilot CLI.
- Metric: MNAS via `evaluate_predictions_with_official_llm_match` → `final_score`. Per-category MNAS is computed ad-hoc from the raw `{qid: int}` metrics file (see `05_evaluation.md §5.7`).
- Evidence corpus: `docs/10_experiment_log/` — read for anything fold-sensitive or category-sensitive.

Not exercised:
- Active exploration (AE-EQA) — not wired.
- HM3D split — no prepared scenes locally.
- OpenEQA's own frame format (`data/frames/<episode>/`) — does not exist locally; we use the ConceptGraph prepared layout instead.

## 8.3 EmbodiedScan VG (active secondary benchmark)

Scope:
- Loader handles EmbodiedScan annotations; data preparation handoff is separate (commit `1eb5b82` "docs: EmbodiedScan data preparation handoff for Linux agent"; raw EmbodiedScan sens download in `scripts/download_embodiedscan_sens.sh`).
- Stage-1 reuses the same `KeyframeSelector` and scene graph (`9988be9` extended `_load_objects_from_pcd` to keep `pcd_np + bbox_np` for precise object extent — previously discarded as unneeded for QA).
- Stage-2 gates VG-only tools on `runtime.task_type == Stage2TaskType.VISUAL_GROUNDING` (`src/agents/runtime/deepagents_agent.py:204-247`):
  - `select_object(object_id, rationale)` — scene-graph lookup; must be called exactly once to finalise the VG answer.
  - `spatial_compare(target_category, relation, anchor_category)` — rank target-category objects by 3D distance; `closest_to` / `farthest_from`.
- Evaluator: `embodiedscan_eval.py` computes oriented 3D IoU (9-DOF bbox). Acc@0.25 / Acc@0.50 / mean IoU are the reported metrics.

Smoke result (3 samples, commit `9988be9`): Acc@0.25 66.7 %, Acc@0.50 33.3 %, mean IoU 0.298. Baseline (pre-VG-tools) on the same 3 samples: Acc@0.25 6.2 %, Acc@0.50 0 %, mean IoU 0.027. This is the only VG number the project currently owns; **any paper claim requires at least a mini-split (~500 samples) and a full-split run**.

## 8.4 SQA3D (loader only)

Loader at `src/benchmarks/sqa3d_loader.py`; unit tests at `src/benchmarks/tests/test_sqa3d.py`. `src/agents/benchmark_adapters.py` declares `"sqa3d"` as a `BenchmarkType` and the mock `ScanNetFrameProvider` can serve SQA3D-style frame lists, but:
- No `src/agents/adapters/sqa3d_adapter.py` subclassing `BenchmarkAdapter`.
- No scripted entry point under `src/agents/examples/` or `src/evaluation/scripts/`.
- No evaluator wired to SQA3D's answer format.

Opening an SQA3D branch requires, at minimum: (i) a `BenchmarkAdapter` that returns `Stage2TaskSpec(task_type=QA, …)` from SQA3D sample fields, (ii) an evaluator for SQA3D's Top-1 / Top-5 answer scoring, (iii) a pilot script. None of this exists on HEAD.

## 8.5 ScanRefer (loader only)

Same status as SQA3D. Loader at `src/benchmarks/scanrefer_loader.py`. No `BenchmarkAdapter`, no pilot, no evaluator on HEAD. A new branch would mirror the EmbodiedScan VG scaffolding (Phases 1-7 per commits `493931a` → `60d6f51`) closely since the task is the same family (referring expressions / VG); the main work is in the evaluator (ScanRefer uses Acc@0.25 / Acc@0.50 on oriented bbox IoU — compatible with `embodiedscan_eval.py`'s scoring code).

## 8.6 Shared infrastructure

The following components are used by every active benchmark (OpenEQA, EmbodiedScan VG) and would be reused by any future benchmark:

| Component | Used by | Purpose |
|---|---|---|
| `KeyframeSelector` | both | Stage 1 retrieval; task-agnostic given a `query` |
| `Stage1BackendCallbacks` | both | Tool-side access to the live selector for `request_more_views` / `request_crops` / `switch_or_expand_hypothesis` |
| `DeepAgentsStage2Runtime` | both | ReAct loop, uncertainty stopping, nudge |
| `Stage2EvidenceBundle` | both | Inter-stage contract |
| `BenchmarkAdapter` | OpenEQA (compat wrapper), EmbodiedScan VG | Pluggable pattern for new benchmarks (commit `493931a`) |
| `evaluate_predictions_with_official_llm_match` | OpenEQA only | MNAS scoring; **does not generalise** to other benchmarks (tightly coupled to OpenEQA's LLM-match prompt and scale) |

The upshot: adding a new QA benchmark is low-friction provided the benchmark has its own evaluator; adding a new VG-style benchmark is low-friction if oriented-IoU scoring suffices.

## 8.7 Cross-references

- `02_architecture.md §2.2` — module inventory with paths.
- `04_stage2_agent.md §4.3` — tool gating mechanics (why VG tools don't leak into QA runs).
- `11_academic_angles_catalog.md` Angle B2 — unified multi-task-policy paper framing that depends on EmbodiedScan VG readiness graduating from yellow to green.
