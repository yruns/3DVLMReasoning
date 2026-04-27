# v15 — Trajectory-Aware Stage 1 + Optional Temporal Fan

**Date:** 2026-04-20
**Branch:** `feat/v15-trajectory-aware`
**Eval:** 1050Q full OpenEQA ScanNet (frozen set @ `tmp/v15_artifacts/v15_frozen_questions.json`, 89 scenes, 7 categories)
**Data:** `tmp/openeqa_eval_v15_{s1_l1,s1_l2,s1s2_l1}/`
**Launcher:** `scripts/run_v15_eval_matrix.sh`
**Frozen question set:** locked across runs via `--force-selection` so all v14/v15 columns judge the exact same 1050 IDs.

## What Changed

Three-knob ablation on top of v14:

1. **`--pose-aware` Stage 1**: trajectory-derived metadata in `KeyframeEvidence.note` (commit `734161e`); joint coverage uses dwell + frustum redundancy (commit `6b56bfb`).
2. **`--frustum-method {l1,l2}`**: choice of frustum overlap helper (`a716c30`).
3. **`--enable-temporal-fan`**: Stage 2 `request_more_views` learns a `temporal_fan` mode that pulls neighbours filtered by frustum overlap (commit `3963671`); ScanNet-realistic params (`8cd48d5`): overlap threshold 0.7, window 16, adaptive fallback.

Other plumbing: `_resolve_raw_dir` for canonical and overlay scene layouts (`a94bb8b`); raw-dir symlink in runtime_cache overlay (`483e69b`); ModelHub adapter with 2-key rotation + retry (`a6f11f6`); 403 retryable + `--resume` for incremental batch1 recovery (`6830394`).

## Run Matrix

Frozen 1050Q × 3 configs, all on `feat/v15-trajectory-aware` (commit `2b9cb9e`), Gemini 2.5 Pro judge, identical CLI baseline:

```
--max-samples 2000 --workers 6 --llm-rewrite --confidence-guard 0.6
--max-reasoning-turns 10 --max-additional-views 2
--force-selection $FROZEN --resume --evaluate --eval-model gemini-2.5-pro
```

| Tag | Stage 1 | Stage 2 trigger | Stage2 #scored | Stage2 mean | **Stage2 MNAS** | %≥4 | %=5 |
|-----|--------|-----------------|---------------:|------------:|----------------:|----:|----:|
| `v15_s1_l1`   | pose-aware + L1 frustum | (none) | 1049 | 3.973 | **74.33** | 72.93% | 57.48% |
| `v15_s1_l2`   | pose-aware + L2 frustum | (none) | 1050 | 3.894 | 72.36 | 71.71% | 55.62% |
| `v15_s1s2_l1` | pose-aware + L1 frustum | + temporal_fan | 1050 | 3.898 | 72.45 | 71.43% | 56.86% |

For matched-judge / matched-frozen-set baselines on the same machine:

| Tag | Stage2 MNAS | E2E MNAS | E2E #scored |
|-----|------------:|---------:|------------:|
| `v14_full`     | 71.60 | 73.14 | 1050 |
| `v14_rerun`    | 72.29 | 73.17 | 1050 |
| `v15_s1_l1`    | **74.33** | 74.74¹ | 863¹ |

¹ E2E judge column for `v15_s1_l1` was only partially scored at the time of harvest (863/1050). Stage2 column is the apples-to-apples surface across all v14/v15 runs, so the headline number is **Stage2 MNAS = 74.33** (vs `v14_full` 71.60 stage2 = **+2.73**, vs `v14_rerun` 72.29 = **+2.04**). The partial e2e column at 74.74 is consistent in direction but should not be used for cross-version comparison until re-judged.

## Key Observations

1. **`v15_s1_l1` wins** — pose-aware Stage 1 with L1 frustum, no temporal fan, beats both v14 columns and the L2 / temporal-fan variants.
2. **L2 frustum hurts** vs L1 (`s1_l2` 72.36 < `s1_l1` 74.33). L1 is the recommended setting.
3. **Stage 2 `temporal_fan` is neutral-to-mildly-negative** here on L1 (`s1s2_l1` 72.45 < `s1_l1` 74.33). The fan tool was designed for cases where Stage 1 misses a critical viewpoint; on the v15 frozen set Stage 1 already covers enough that the additional fan calls add noise more than evidence. It may still help on harder scenes — needs a per-category breakdown to confirm.
4. **No regression on Score=5**: `s1_l1` 57.48% ≥ `v14_full` 56.10% on the same surface (stage2).

## Caveats

- **Per-category breakdown not yet computed**. v14's table breaks down 7 OpenEQA categories; v15 should do the same to know whether the +2.7 MNAS comes from spatial / attribute / object-state.
- **Judge mismatch with paper SOTA**: still Gemini 2.5 Pro vs. paper's GPT-4. See `leaderboard.md §Important Notes`.
- **E2E column under-scored** for `v15_s1_l2` (327/1050) and `v15_s1s2_l1` (205/1050) — Gemini judging hadn't completed when harvest happened. Stage2 column is complete and is what the headline is computed against.
- **Different prompt-cache key vs v14**: `derive_eval_session_id` was extended on the next branch (`feat/explore_3dbbox`) to fold in `chassis_tools_version` and `vg_backend`. v15 runs themselves do not have these flags — they are pure v15 with the older session-id derivation.

## Leaderboard Position (ScanNet EM-EQA)

| # | Method | ScanNet MNAS | Notes |
|---|--------|:------------:|-------|
| - | Human | 87.7 | OpenEQA paper |
| **1** | **Ours (v15, `s1_l1` stage2)** | **74.33** | Frozen 1050Q, Gemini 2.5 Pro judge |
| 2 | Ours (v14, e2e) | 73.14 | Frozen 1050Q, Gemini 2.5 Pro judge |
| 3 | GPT-4V (500Q subset) | 51.3 | OpenEQA paper, GPT-4 judge |

Judge caveat applies — see `leaderboard.md`.

## Cumulative Progress (v9 → v15)

| Metric | v9 | v14 | v15 | v9→v15 |
|--------|---:|----:|----:|-------:|
| MNAS (best column) | 46.5 | 73.14 | **74.33** | **+27.8** |
| Eval scale | 100Q | 1050Q | 1050Q | — |

## Next Steps

- Per-category breakdown for `v15_s1_l1` (largest deltas vs v14 expected in Spatial / Localization given pose-aware change).
- Investigate whether `temporal_fan` helps on the harder Spatial / World-Knowledge subset, even if neutral overall.
- Re-evaluate `v15_s1_l2` and `v15_s1s2_l1` e2e columns to full 1050 to close the partial-judge gap.
- See `leaderboard.md §To-Do` for matched-judge GPT-4 re-evaluation, still open.
