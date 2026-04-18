# `docs/` — Agent-Readable Knowledge Base

Target reader: an AI agent tasked with mining academic value from the 3DVLMReasoning OpenEQA pipeline. Written for density, not tutorial flow.

**Root authority**: HEAD = `a8e651e` (branch `feat/embodiedscan-grounding`). Facts dated 2026-04-17. If a claim here conflicts with code, trust the code — open an issue.

**Numbering convention**:
- `00_` / `11_` — ⭐ load-bearing, read first.
- `01_…05_` — architecture deep-dive.
- `06_…09_` — evolution, data, catalogs, gotchas.
- `10_experiment_log/` — raw per-version results (verbatim copy of an external folder; treat as immutable evidence).

## Task → File

| If you need … | Go to |
|---|---|
| The 4–5 falsifiable academic claims, with evidence trails | `00_research_manifest.md` |
| The 6 candidate paper angles + per-angle literature positioning | `11_academic_angles_catalog.md` |
| "What is this project about, and what is NEW vs. OpenEQA CVPR 2024?" | `01_overview.md` |
| "Explain the two-stage architecture" | `02_architecture.md` |
| Stage 1 internals (KeyframeSelector, hypothesis types, joint-coverage view selection) | `03_stage1_retrieval.md` |
| Stage 2 internals (DeepAgents runtime, 7 tools, nudge loop, system-prompt evolution) | `04_stage2_agent.md` |
| MNAS definition, Gemini 2.5 Pro judge rationale, scoring protocol | `05_evaluation.md` |
| Per-version engineering deltas v9 → v14 (cross-linked to `10_experiment_log/`) | `06_evolution_v9_to_v14.md` |
| Local `data/` layout vs. vanilla `OpenEQADataset` expectations | `07_data_layout.md` |
| OpenEQA, EmbodiedScan VG, SQA3D, ScanRefer — scope and readiness matrix | `08_benchmarks_catalog.md` |
| Known foot-guns (stale CLAUDE.md, broken GPU 1, conda vs `.venv`, deleted docs) | `09_gotchas.md` |
| Raw per-version results (v9 46.5 MNAS 100Q → v14 73.1 MNAS 1050Q) | `10_experiment_log/README.md` |
| Full leaderboard with paper references (OpenEQA / CoV / 3D-Mem / GraphPad / R-EQA) | `10_experiment_log/leaderboard.md` |
| Specific version summary (e.g., why v14 moved enrichment into the prompt) | `10_experiment_log/vN_*.md` |
| Interactive dashboard / pipeline diagram / case studies | `10_experiment_log/*.html` |

## Conventions

- **Claim schema** (used in `00_` and `11_`): `Claim / Status / Evidence / Frontier / Prior / Novelty / Risk / Critique / REQUIRES`. Defined in `00_research_manifest.md`.
- **Anchors**: every quantitative number carries `(fold, commit|doc)`. Example: `MNAS 73.1 (1050Q, commit fbd642e)`.
- **Fold labels**: `30-scene` (120Q, v9 original), `100Q` (20×5 log, v9–v12), `1050Q` (full ScanNet EM-EQA, v13–v14). Numbers across folds are NOT directly comparable.
- **Forward refs**: `TODO(forward-ref: filename)` marks a dangling cross-link to a doc not yet written.

## Scope boundaries

- No per-`.py` file docs — module level only.
- No tutorials — assume the reader has full repo + pipeline-level handoff already in context.
- No speculation without `REQUIRES:` annotation.
- HTML viz in `10_experiment_log/` is kept verbatim; do not transcribe.

## Sibling reference

Project root `CLAUDE.md` contains operator-level instructions (env setup, tmux mandate, no-fallback rule, GPU 1 ban). `docs/` does not duplicate those.
