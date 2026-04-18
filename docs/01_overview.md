# 01 — Project Overview

Density, scope, and what is **new** relative to OpenEQA (CVPR 2024, arXiv:2312.15857). Prose kept short; numbers carry the story.

## One-sentence positioning

3DVLMReasoning is a two-stage framework for 3D scene understanding in which Stage 1 treats a pre-built ConceptGraph-style scene representation as a *high-recall but untrustworthy prior*, and Stage 2 is an evidence-seeking VLM agent that verifies, repairs, or rejects Stage 1's hypotheses by actively acquiring pixel evidence through a calibrated tool set.

## Scope

| Dimension | In scope | Out of scope |
|---|---|---|
| Benchmarks | OpenEQA ScanNet EM-EQA (primary), EmbodiedScan VG (active), SQA3D + ScanRefer (loaders only) | OpenEQA HM3D split, SR3D / NR3D stand-alone, OpenEQA active-exploration |
| Data | Prepared ConceptGraph scenes at `data/OpenEQA/scannet/<clip>/conceptgraph/`, 89 clips locally | Raw ScanNet video decoding — we consume the prepared outputs only |
| Task families | QA (primary), visual grounding (active), nav-plan / manipulation (task types declared but unexercised) | Long-horizon planning, dialog |
| VLM backend | `gpt-5.4-2026-03-05` (default since v12), Gemini pool (Stage 1 parser + judge) | Open-weight VLMs (Qwen-VL, LLaVA-3D) — not integrated |

Canonical entry point for OpenEQA evaluation: `src/agents/examples/openeqa_official_question_pilot.py`. Everything else is either a component, a test, or an alternative harness.

## What is new vs. OpenEQA CVPR 2024

OpenEQA provides the benchmark, the frame-level episodic-memory format, and four baseline families (GPT-4 + {LLaMA, LLaVA-1.5, Sparse Voxel Maps, ConceptGraphs} + blind variants). Every baseline treats the scene representation as input to a one-shot VLM or language model.

| Axis | OpenEQA CVPR 2024 baseline | This project (HEAD = `a8e651e`) |
|---|---|---|
| Retrieval | Uniform frame sampling (50f); scene-KB lookup for ConceptGraphs baseline | Query-driven `KeyframeSelector.select_keyframes_v2` with typed hypothesis ranking (`DIRECT_GROUNDED` > `PROXY_GROUNDED` > `CONTEXT_ONLY`), joint-coverage view selection, BETWEEN-midpoint retrieval, min-3 keyframe padding. Code: `src/query_scene/keyframe_selector.py:1570-1731` |
| Stage 2 | Single VLM forward pass over frames + optional KB text | ReAct-style DeepAgents loop with 5 + 2 tools, uncertainty-aware stopping, evidence nudge on insufficient-evidence. Code: `src/agents/runtime/deepagents_agent.py:570-692` |
| Relationship to scene graph | Ground-truth or KB; no verification | High-recall prior, demoted; agent system prompt explicitly says *"Stage 1 is a high-recall evidence retriever, not ground truth"* at `src/agents/runtime/base.py:358-359` |
| Tool use | None (baselines) | `inspect_stage1_metadata`, `retrieve_object_context`, `request_more_views`, `request_crops`, `switch_or_expand_hypothesis`, + `select_object` / `spatial_compare` for VG. Gating on task type at `src/agents/runtime/deepagents_agent.py:204-247` |
| Calibration policy | None | Scalar-confidence gate at three control points: runtime downgrade (`runtime/base.py:415`), nudge injection (`deepagents_agent.py:635`), E2E rerun (`pilot:519-555`) |
| Evaluation | LLM-match with GPT-4 → integer 1–5 → MNAS 0–100 | Same upstream scorer, judge swapped to Gemini 2.5 Pro for key-pool / concurrency reasons; pool-rotated retry + progressive resume. See `docs/05_evaluation.md` |
| Reported SOTA on OpenEQA ScanNet (their judge) | GPT-4 + ConceptGraphs 37.8; GPT-4V 500Q 51.3 | **MNAS 73.1 (1050Q, v14, commit `fbd642e`)**; judge is Gemini 2.5 Pro — not directly comparable, flagged at `docs/10_experiment_log/leaderboard.md` |

## Key numbers to remember

| Fact | Number | Fold | Anchor |
|---|---|---|---|
| HEAD MNAS | **73.1** | 1050Q full OpenEQA ScanNet EM-EQA | `10_experiment_log/v14_inventory_20260404.md`, commit `fbd642e` |
| v9 baseline | 46.5 | 100Q (20×5) | `10_experiment_log/v9_baseline_20260329.md` |
| Total lift v9 → v14 | **+26.6 MNAS** | trajectory (fold change v12→v13) | `10_experiment_log/README.md` |
| Local prepared scenes | 89 clips, ≈37 GB | under `data/OpenEQA/scannet/` | `CLAUDE.md` §Current Local Checkout |
| Stage 2 backbone | `gpt-5.4-2026-03-05` | since v12, commit `2abb404` | `src/agents/core/agent_config.py:44` |
| Judge | Gemini 2.5 Pro | since module introduction | `src/benchmarks/openeqa_official_eval.py:122` |
| Confidence threshold (runtime) | 0.4 | default | `src/agents/core/agent_config.py:58` |
| Confidence guard (E2E rerun) | 0.6 | default | `src/agents/examples/openeqa_official_question_pilot.py` |

## What this overview is NOT

- Not a tutorial. For running a job, see `10_experiment_log/README.md §Evaluation Commands`.
- Not a list of files. For module tree, see `02_architecture.md`.
- Not the paper. For the claim structure, see `00_research_manifest.md`.

Cross-refs: two-stage component tree + data flow in `02_architecture.md`; per-version deltas in `06_evolution_v9_to_v14.md`; foot-guns (stale CLAUDE.md, GPU 1 ban, conda vs `.venv`, deleted docs) in `09_gotchas.md`.
