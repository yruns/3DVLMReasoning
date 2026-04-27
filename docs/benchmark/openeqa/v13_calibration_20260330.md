# v13 — Mandatory Crops + Overconfidence Calibration

**Date:** 2026-03-30
**Branch:** `feat/v11-evidence-seeking-fixes`
**Eval:** 1050Q full OpenEQA ScanNet (first full-scale run)
**Data:** `tmp/openeqa_eval_v13_full/`

## What Changed

1. **Mandatory crops for color/attribute questions**: "For ALL color/attribute questions, you MUST call request_crops BEFORE answering"
2. **Overconfidence calibration**: "For YES/NO state questions, MUST use tools regardless of confidence"
3. **Self-check directive**: "Before final answer, verify — does your answer contradict the question premise?"
4. **Prominent-object bias correction**: "The correct answer may be a smaller or less prominent object"
5. **Multi-component color**: "List ALL distinct colors visible on the object"
6. **LLM failure retry**: up to 5 retries for 500/429/timeout errors per sample

## Results (1050Q Full)

| Metric | Value |
|--------|:-----:|
| **MNAS** | **71.4** |
| Raw E2E mean | 3.854 |
| Completed | 1050/1200 |
| Failed | 29 |
| Score>=4 | 70.6% |
| Score<=2 | 21.1% |
| Tool calls | 1423 (1.36/Q) |
| Tool usage rate | 88.1% |

### Per-Category MNAS

| Category | MNAS | vs Human |
|----------|:----:|:--------:|
| Object State | 82.8 | -15.9 |
| Localization | 79.6 | +2.3 |
| Functional | 74.3 | -7.5 |
| Attribute | 74.0 | -13.9 |
| World Knowledge | 65.3 | -21.9 |
| Object Rec | 64.2 | -23.7 |
| Spatial | 57.8 | -28.9 |

## Failure Analysis (96 cases sampled from 222)

| Root Cause | % |
|------------|:-:|
| VLM Perception | 35% |
| Retrieval | 24% |
| Visibility | 12% |
| Spatial | 8% |
| State | 7% |
| World Knowledge | 5% |
| GT Issues | 5% |
| Missing Object | 4% |

51% of failures have GT object already in enrichment — data available but not used.

Full analysis: `docs/eval_analysis_v13_full_20260403.md`

## Files Changed

- `runtime/base.py` — mandatory crop/self-check/bias prompt rules
- `openeqa_official_question_pilot.py` — 5x retry for LLM errors
