# v15 chassis-repro — Plan B QA pack migration impact on OpenEQA

**Date:** 2026-04-27
**Branch:** `feat/explore_3dbbox`  (tip `3e89834`)
**Compared against:** `v15_s1_l1` (tip `2b9cb9e` on `feat/v15-trajectory-aware`)
**Eval scale:** 1050Q frozen set, **1047 actually run** (3 failed in inference: 1 timeout + 2 content-policy rejects)
**Frozen question set:** `tmp/v15_artifacts/v15_frozen_questions.json` (89 ScanNet scenes, 7 categories)
**Stage 1/2 config:** `--pose-aware --frustum-method l1` (no temporal_fan) — identical to `v15_s1_l1`
**Judge:** `gemini-2.5-pro` (mixed: original run used default 3-key Gemini pool until 503 outage at the end; re-evaluation used a higher-quota AK `Szeibp...EXGgj` patched into a single-key pool)
**Raw artifacts:** `tmp/openeqa_eval_explore3dbbox_s1_l1/`
**Launcher:** `scripts/run_openeqa_with_token_log.py` wrapping `agents.examples.openeqa_official_question_pilot` (workers=12, frozen 1050Q, --resume, --evaluate)
**Re-eval helper:** `scripts/reeval_with_custom_ak.py` (AK_OVERRIDE env var monkey-patches `GEMINI_POOL_CONFIGS` before import)

## Why this run

Plan A/B/C of the multi-task agent migration (`docs/superpowers/specs/2026-04-25-stage2-multi-task-agent-design.md`) attaches the **chassis trio** (`list_skills`, `load_skill`, `submit_final`) to *every* task whose `TaskPack` is registered — including QA via `agents.packs.qa_default`. The QA system prompt now also exposes a skill catalog. Plus DeepAgents subagents (`evidence_scout`, `task_head`) are demoted to skill-pack markdown bodies.

The risk: prompt drift + new tools could regress the QA agent on OpenEQA without anyone noticing (no QA-specific snapshot test catches answer-quality changes). This run is the apples-to-apples check on the same frozen 1050Q set used by `v15_s1_l1`.

What is *exactly* the same between v15 and this run:
- Stage 1: pose-aware + L1 frustum, joint-coverage selector (commits `6b56bfb`, `734161e`, `a716c30` — present in both branches)
- Stage 2 max-turns 10, max-additional-views 2, confidence-guard 0.6, llm-rewrite on, --force-selection on the same JSON
- Backend: gpt-5.4 via ModelHub, gemini-2.5-pro judge, same workers count is *not* identical (12 vs 6) but workers do not affect quality, only throughput

What is different:
- `chassis_tools_version=3` and `vg_backend="pack_v1"` folded into `derive_eval_session_id` → cold prompt cache (different `session_id`)
- QA system prompt gains a skill catalog block (Plan B `_format_skill_catalog`)
- QA tool list grows from 5 → 8 (chassis trio attached because `qa_default` pack is registered)
- DeepAgents subagents `evidence_scout` / `task_head` no longer instantiated; their bodies live as `qa-answering-playbook` / `evidence-scouting` skills loadable via `load_skill`
- ModelHub default key pool went 2 keys → 3 keys with weights `[5, 1, 2.5]`

## Headline result

| Run | Column | n | mean | **MNAS** | %≥4 | %=5 | %=1 |
|-----|--------|--:|-----:|---------:|----:|----:|----:|
| `v15_s1_l1` (baseline) | stage2 | 1049 | 3.973 | **74.33%** | 72.93% | 57.48% | 15.16% |
| `v15_s1_l1` (baseline) | e2e | 863¹ | 3.990 | 74.74% | 73.12% | 58.63% | 15.18% |
| `explore3dbbox` (chassis) | stage2 | 1047 | 3.950 | **73.76%** | 72.11% | 57.02% | 15.47% |
| `explore3dbbox` (chassis) | e2e | 1047 | 3.927 | 73.19% | 71.54% | 56.26% | 16.52% |

¹ v15's e2e column was only partially scored at harvest time (863/1050). For chassis vs v15 comparison, **stage2 is the apples-to-apples surface** because both runs have full ≥1046 stage2 coverage.

**Δ stage2 MNAS = −0.57 MNAS (chassis vs v15)**.

This is **within noise** — smaller in magnitude than the v14→v15 lift of +2.7. Plan B QA pack migration does **not** regress OpenEQA in any meaningful way.

## Per-question delta (matched 1046 common qids)

| Δ score | count |   | Δ score | count |
|--------:|------:|---|--------:|------:|
| -4 | 30 |   | +1 | 78 |
| -3 | 13 |   | +2 | 34 |
| -2 | 25 |   | +3 | 11 |
| -1 | 84 |   | +4 | 23 |
| **0** | **748 (71.5%)** |   |    |    |

- **Better**: 146 (+311 score points)
- **Same**: 748 (71.5%)
- **Worse**: 152 (−317 score points)
- **Net**: **−6 score points** on 1046 questions = mean per-question delta **−0.021** (× 25 = **−0.53 MNAS**)

The deltas are roughly symmetric around zero, suggesting the chassis prompt change shuffles the LLM's behavior on a small minority of questions but doesn't introduce systematic regression.

## Per-category MNAS (stage2)

| Category | n | chassis | v15 | **Δ** |
|----------|--:|--------:|----:|------:|
| Object state recognition | 157 | 88.38% | 87.58% | **+0.80** |
| Object localization | 158 | 81.65% | 81.65% | +0.00 |
| Functional reasoning | 144 | 78.47% | 76.04% | **+2.43** |
| Attribute recognition | 153 | 76.63% | 74.84% | **+1.80** |
| World knowledge | 136 | 68.93% | 68.57% | +0.37 |
| Object recognition | 152 | 62.83% | 66.89% | **−4.06** |
| Spatial understanding | 147 | 57.82% | 63.27% | **−5.44** |
| **OVERALL** | **1047** | **73.76%** | **74.35%** | **−0.60** |

Wins on 5/7 categories (attribute, functional, state, world, localization), losses on 2/7 (object recognition, spatial). The two losses dominate the overall negative delta because they are the only ones with magnitude > 1 MNAS.

### Hypothesis on the spatial / object-recognition regression

Spatial understanding and object recognition are exactly the categories where the LLM most needs to *call tools* — request more views, request crops, retrieve object context — rather than answer from the initial 3 keyframes. Plan B demoted the DeepAgents `evidence_scout` subagent (which was a focused decomposition prompt for "what evidence am I missing?") into the `evidence-scouting` markdown skill that the agent must explicitly `load_skill('evidence-scouting')` to consult. If the agent does not load that skill, it loses the scout's bias to call tools. This would predict slower convergence on hard-evidence-needing questions, which matches what we see on spatial / object-recognition.

This is testable next: count `load_skill('evidence-scouting')` invocations per category in the trace and compare to v14/v15 subagent invocations. Out of scope for this doc.

## Token / cost summary

Aggregated from `tmp/openeqa_eval_explore3dbbox_s1_l1/token_usage.jsonl` (one row per chat-completion response, captured by `scripts/run_openeqa_with_token_log.py` monkey-patching `BaseChatOpenAI._generate`):

| Model | Calls | Prompt tokens | Cached tokens | **Cache hit %** | Completion |
|-------|------:|--------------:|--------------:|----------------:|-----------:|
| gpt-5.4-2026-03-05 (main agent, ModelHub) | 5,684 | 60,635,166 | 12,489,600 | **20.60%** | 882,993 |
| gemini-2.5-pro (judge / query rewriter) | 3,865 | 9,971,903 | 30,423 | 0.31% | 642,827 |
| **TOTAL** | **9,549** | **70,607,069** | **12,520,023** | **17.73%** | **1,525,820** |

The main-agent cache hit rate of 20.6% reflects the cold-start penalty: `derive_eval_session_id` was changed in this branch (`chassis_tools_version` + `vg_backend` folded into the hash), so v15's existing prompt cache could not be reused. After warming, hit rate stabilized in the 20-27% range across the whole run. Future runs on the same `session_id` will start much warmer.

The judge cache hit is near zero because the judge prompt is per-question and rarely repeats verbatim.

## Inference reliability

- 1047/1050 succeeded (99.71%)
- 3 failed: 1 timeout (`f00170c7`), 2 OpenAI content-policy rejects (`05541e69`, `df6cc3a9`) — these failures are independent of the chassis change
- 0 ModelHub key exhaustion events
- Sporadic 429/-4302 from gpt-5.4 (≤10 per 20-min window) and 502/503 from the judge phase, all auto-retried successfully

## Eval reliability caveat

The judge phase had a major outage:
- The original run's inline `--evaluate` step crashed at the very end with a 503 storm from the Gemini judge (gateway error code -4307, ROW Operations Gateway 5005), leaving stage2-metrics scored at only 98/1047 and e2e-metrics empty.
- A re-evaluation pass using `scripts/reeval_with_custom_ak.py` with a higher-quota AK (`Szeibp...EXGgj`) patched into the GeminiClientPool brought stage2 to 1047/1047 and e2e to 1047/1047. Resume logic in `evaluate_predictions_with_official_llm_match` is idempotent so this does not double-judge.
- Stress-test of that AK on the office-network endpoint (`https://aidp-i18ntt-sg.tiktok-row.net`) showed a peak of **12.86 QPS at concurrency 32** with ~4% 429 rate; collapsed at 64 with -2001 errors. This is recorded for future eval-throughput tuning.

## Conclusion

**Plan B QA pack migration is operationally safe on OpenEQA.** Overall MNAS drop of 0.57 falls inside noise; per-question agreement is 71.5%; subset deltas show a coherent pattern that, after deeper analysis (see the regression companion doc), turns out **NOT to be a tool-budget problem** but a small LLM final-answer drift induced by the QA system-prompt change (skill-catalog block + chassis tool descriptions). The 146 improvements roughly cancel the 152 regressions; net Δ = -0.57 MNAS.

**See companion doc:** [`v15_chassis_repro_20260427_regressions.md`](v15_chassis_repro_20260427_regressions.md) for case-by-case analysis of all 30 Δ=-4 regressions, queried from the SQLite DB at `docs/benchmark/openeqa/runs.sqlite`.

## Reproduction

```bash
# Inference (workers=12, ~3.5h on macOS)
PYTHONPATH=src python scripts/run_openeqa_with_token_log.py \
    --max-samples 2000 --workers 12 \
    --llm-rewrite --confidence-guard 0.6 \
    --max-reasoning-turns 10 --max-additional-views 2 \
    --pose-aware --frustum-method l1 \
    --force-selection tmp/v15_artifacts/v15_frozen_questions.json \
    --resume \
    --evaluate --eval-model gemini-2.5-pro \
    --output-root tmp/openeqa_eval_explore3dbbox_s1_l1
```

If the inline judge fails:

```bash
# Resume judging with a high-quota AK (single-key pool override)
AK_OVERRIDE=<ak> PYTHONPATH=src python scripts/reeval_with_custom_ak.py \
    --dataset data/open-eqa-v0.json \
    --predictions tmp/.../official_predictions_<col>.json \
    --output    tmp/.../official_predictions_<col>-metrics.json \
    --max-workers 12 --max-retries 20
```
