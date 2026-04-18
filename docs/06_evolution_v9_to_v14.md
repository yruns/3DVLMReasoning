# 06 — Evolution v9 → v14

Per-version deltas, cross-linked to the immutable `10_experiment_log/vN_*.md` files. The numbers in this doc are summaries only; primary evidence lives under `10_experiment_log/`. Line numbers against HEAD = `a8e651e`.

## 6.1 Fold change at v13

Before reading per-version rows, note the fold change: **v9 / v10 / v11 / v12 are 100Q (20 scenes × 5 questions); v13 / v14 are 1050Q full OpenEQA ScanNet EM-EQA.** Numbers within each fold are comparable; across the fold boundary they are not. The 100Q fold was the iteration harness; the 1050Q fold is the public-number harness.

## 6.2 One-row-per-version summary

| Ver | Date | Fold | MNAS | Δ (within fold) | Headline | Primary commit | Cross-ref |
|---|---|:-:|:-:|:-:|---|---|---|
| v9 | 2026-03-29 | 100Q | 46.5 | baseline | Parallel eval harness + `--confidence-guard` + `--llm-rewrite` | `b6a8aa6` | [v9_baseline_20260329.md](10_experiment_log/v9_baseline_20260329.md) |
| v10 | 2026-03-30 | 100Q | 55.4 | +8.9 | LLM object enrichment injected into Stage 1 | `f122b07` | [v10_enrichment_20260330.md](10_experiment_log/v10_enrichment_20260330.md) |
| v11 | 2026-03-30 | 100Q | 62.6 | +7.2 | Enable Stage-2 callbacks; pad-to-min-3; open-ended queries | `02ea2f3` | [v11_callbacks_20260330.md](10_experiment_log/v11_callbacks_20260330.md) |
| v12 | 2026-03-30 | 100Q | 65.0 | +2.4 | Mandatory tool-use prompt + BETWEEN midpoint + **GPT-5.2 → GPT-5.4** | `2abb404` | [v12_toolprompt_20260330.md](10_experiment_log/v12_toolprompt_20260330.md) |
| v13 | 2026-04-02 | 1050Q | 71.4 | (fold change) | Mandatory crops for color/attr; confidence cap 0.7 with zero tools; self-check + bias warnings | `44b9600` | [v13_calibration_20260330.md](10_experiment_log/v13_calibration_20260330.md) |
| v14 | 2026-04-04 | 1050Q | **73.1** | +1.8 | Inject enrichment inventory directly into Stage 2 system prompt | `b4197a1` | [v14_inventory_20260404.md](10_experiment_log/v14_inventory_20260404.md) |

Aggregate trajectory on comparable tail (v9 → v14 cross-fold): **+26.6 MNAS**, Score=1 44 % → 17 %, Score ≥ 4 42 % → 72 % — see [v14_inventory_20260404.md](10_experiment_log/v14_inventory_20260404.md) §Cumulative Progress.

## 6.3 Per-version delta detail (code + motivation + what broke)

Each subsection names the motivation (the *why*), the code locus on HEAD (the *what*), and the observed per-fold number. Failure-case analyses live in `10_experiment_log/` and are not duplicated here.

### v9 — baseline parallel harness (`b6a8aa6`)

| Field | Value |
|---|---|
| Motivation | Make OpenEQA eval runnable at scale; earlier single-threaded pilots timed out before finishing even 20 scenes. |
| Code on HEAD | `src/agents/examples/openeqa_official_question_pilot.py` — flags `--workers`, `--confidence-guard 0.6`, `--llm-rewrite`, `--num-scenes / --questions-per-scene`; per-scene lock at `_get_scene_lock(clip_id)`; E2E guard at `:519-555`. |
| Separable sub-effects | `--llm-rewrite` raised `direct_grounded` rate 26 % → 33 % on the 30-scene pilot; `--confidence-guard 0.6` cited as +10.5 MNAS on 5-scene pilot (commit message). |
| 100Q fold | 46.5 MNAS; 44 % Score=1. |
| Live on HEAD? | Yes — all four flags remain in the pilot; E2E guard logic still at lines 519-555. |

### v10 — LLM object enrichment (`f122b07`)

| Field | Value |
|---|---|
| Motivation | 36/5-scene low-score cases in v9 traced to impoverished Stage 1 object metadata (detection labels are noisy; ConceptGraphs' category is "most-common string"). Provide a richer description (`category / description / location / color / nearby_objects / usability`) for 5097 objects × 89 scenes. |
| Code on HEAD | `src/scripts/enrich_objects.py` (Gemini-2.5-Pro-based generator). Integration: `KeyframeSelector._load_enrichment()` at `src/query_scene/keyframe_selector.py:352-358`, **raises `FileNotFoundError` if `enriched_objects.json` missing** — strict no-fallback rule (ties into commit `17cc18c`). |
| 100Q fold | 55.4 MNAS (+8.9 within-fold). |
| Live on HEAD? | Yes — enrichment file mandatory; per-object enrichment still drives both `retrieve_object_context` (tool-gated) and v14's prompt injection. |

### v11 — evidence-seeking fixes (`02ea2f3`)

| Field | Value |
|---|---|
| Motivation | v10 recorded zero callback invocations — Stage 2 had the tools but never used them because `enable_callbacks=False`; single-keyframe selection was fragile (18/36 low-score cases); UNKNOW target queries (`"What is behind X?"`) returned empty. |
| Code on HEAD | (Fix A) pilot `run_stage2(..., enable_callbacks=True)` at `src/agents/examples/openeqa_official_question_pilot.py:503`. (Fix B) `_pad_keyframes_to_minimum()` invoked at `src/query_scene/keyframe_selector.py:1684`. (Fix C) `open_ended` mode flag in `QueryExecutor`. (Fix D) crop callback skips inverted/degenerate bboxes (`86bfb0f`). |
| 100Q fold | 62.6 MNAS (+7.2 within-fold). **Tool calls 0 → 61** on the same fold — the single largest behavioural step change in the v9 → v14 trajectory. |
| Live on HEAD? | Yes — all four fixes remain. |

### v12 — mandatory tool rules + BETWEEN + backbone bump (`2abb404`)

| Field | Value |
|---|---|
| Motivation | v11 had callbacks enabled but 13/29 remaining failures showed *under-use* — agent chose not to call tools when it should have. Also `open_ended` was over-triggering on comparative queries (`side_of`, `part_of`); BETWEEN spatial queries missed the geometric midpoint; GPT-5.4 had become available. |
| Code on HEAD | "MANDATORY tool-usage rules" in the system prompt at `src/agents/runtime/base.py:375-386`. BETWEEN-midpoint retrieval at `src/query_scene/keyframe_selector.py:1645-1665`. Comparative-relation exclusion inside `QueryExecutor` for `open_ended`. Model default → `gpt-5.4-2026-03-05` at `src/agents/core/agent_config.py:44`. |
| Side bump | commit `85545e3` within the same iteration window added scoring pool-key rotation + 12-worker concurrent scoring (see `docs/05_evaluation.md §5.4`). `25a67fb` made `--workers 6` the pilot default. |
| 100Q fold | 65.0 MNAS (+2.4 within-fold). Tool calls 61 → 90. |
| Live on HEAD? | Yes — all three prompt/code changes verified; model default unchanged since. CLAUDE.md still lists `gpt-5.2` as default (stale — see `docs/09_gotchas.md`). |

### v13 — prompt calibration + full-scale eval (`44b9600`)

| Field | Value |
|---|---|
| Motivation | v12's 24 remaining 100Q failures showed a shifted root cause: REASONING_ERROR 71 %, including 7 outright color mis-identifications. Agent was overconfident on zero-tool spatial/attribute questions. Also first run at full 1050Q scale — the 100Q iteration loop had exhausted its usefulness. |
| Code on HEAD | All text changes in `src/agents/runtime/base.py:378-392`: (a) "For ALL color/attribute questions, you MUST call `request_crops` BEFORE answering." (b) YES/NO state questions must tool-verify. (c) "Do NOT report confidence > 0.7 if you used zero tools and the question involves spatial relations, attributes, or identification." (d) Self-check directive ("does the answer contradict the question premise?"). (e) Prominent-object bias warning ("correct answer may be smaller or less prominent"). (f) Multi-component color instruction ("list ALL distinct colors"). `c8abd4d` added the per-sample 500/429/timeout retry (10/20/40/80/120 s). |
| 1050Q fold | 71.4 MNAS baseline for the full benchmark; 1050/1200 questions completed; 29 failed. Tool calls 1.36/Q, tool-usage rate 88.1 %. |
| Live on HEAD? | Yes — all six prompt bullets live verbatim; retry logic at pilot `:632-666`. |

### v14 — enrichment inventory into system prompt (`b4197a1`)

| Field | Value |
|---|---|
| Motivation | v13 full-scale failure analysis (commit `1887e03`, 96 cases) found that 51 % of low-score failures had the GT object already in `enriched_objects.json` — but the agent rarely called `retrieve_object_context`. Remove the tool gate and put the inventory into the prompt. |
| Code on HEAD | `_format_scene_inventory(object_context)` at `src/agents/runtime/base.py:296-310`; injected into `build_system_prompt` at `:395`. No retrieval removed — the `retrieve_object_context` tool still exists for partial / filtered lookups; only the default visibility changed. |
| 1050Q fold | **73.1 MNAS (+1.8 within-fold)**; Score=5 573 → 600, Score=1 195 → 178. Per-category delta Attribute +3.4, Spatial +2.6, Functional +2.6, Object State +1.9, World Knowledge +1.9, Object Recognition +1.8, Localization −1.6. Tool calls 1.36/Q → 1.33/Q (−0.03). |
| Live on HEAD? | Yes — inventory-injection call at line 395 is the load-bearing v14 change; the anchor for Claim 2 in `00_research_manifest.md`. |

## 6.4 Supporting commits that are NOT numbered versions but matter

| Commit | Date | Purpose | Still live? |
|---|---|---|---|
| `80ebf21` | 2026-03-27 | E2E nudge loop on `insufficient_evidence` | Yes — `_build_evidence_nudge` at `src/agents/runtime/deepagents_agent.py:355`, called at `:644` |
| `7fa0cc7` / `9072818` | 2026-03-28 | Removed original `ConfidenceGuard` class; E2E rerun logic simplified | `ConfidenceGuard` class is **removed**; CLI flag `--confidence-guard` survives |
| `4d92316` | 2026-03-28 | E2E also triggers on low-confidence (not just `insufficient_evidence`) | Yes — pilot `:519-555` |
| `17cc18c` | 2026-03-29 | Eliminated 9 silent fallbacks | Yes — the `enriched_objects.json` raise is one of them |
| `85545e3` | 2026-04-02 | Scoring pool-key rotation + concurrent eval | Yes — `src/benchmarks/openeqa_official_eval.py:67-211` |
| `c8abd4d` | 2026-04-03 | Per-sample 500/429/timeout retry with exponential backoff | Yes — pilot `:632-666` |
| `25a67fb` | 2026-04-02 | Default `--workers 6` | Yes |
| `598bbfe` + `9be20f1` | pre-v9 | Multi-label category index + BETWEEN anchor preservation | Yes — both in Stage 1 |

## 6.5 What to take into the paper

- v9 / v10 are foundations (parallel harness + data quality); not headline claims.
- v11 is **the** behavioural-change step (0 → 61 tool calls); anchors Claim 3 ablation.
- v12 mixes prompt + backbone, which is what makes per-commit attribution hard (flagged in `00_research_manifest.md` Claim 1 honest-critique).
- v13 is where we first see the full 1050Q numbers; anchor for matched-backbone controls.
- v14 is **the** Claim 2 anchor; clean ablation against v13 with only the prompt-injection change.

Per-capability attribution (distinct from per-commit attribution) requires a factorial ablation on HEAD — see REQUIRES on Claims 1, 2, 3 in `00_research_manifest.md`.
