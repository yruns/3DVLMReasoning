# v12 — Tool Prompt Rules + GPT-5.4 + Between-Midpoint

**Date:** 2026-03-30
**Branch:** `feat/v11-evidence-seeking-fixes`
**Eval:** 100Q (20 scenes x 5 questions)
**Data:** `tmp/openeqa_eval_v12/`

## What Changed

1. **Mandatory tool-usage rules** in Stage 2 system prompt: must call tools when answer contradicts question premise, when target not visible, for low-confidence color questions
2. **open_ended regression fix**: comparative queries (side_of, part_of) excluded from auto-detection
3. **"Between" spatial midpoint retrieval**: geometric centroid between anchor objects guides view selection
4. **Model upgrade**: GPT-5.2 → GPT-5.4-2026-03-05
5. **Scoring retry**: `GeminiClientPool.invoke_with_full_retry()` with key rotation
6. **Concurrent scoring**: 12-worker parallel scoring + resume support
7. **Default eval workers**: 1 → 6

## Results

| Metric | v11.1 | v12 | Delta |
|--------|:-----:|:---:|:-----:|
| **MNAS** | 62.6 | **65.0** | **+2.4** |
| Score=1 | 29% | 22% | -7% |
| Score>=4 | 61% | 62% | +1% |
| Tool calls | 61 | 90 | +29 |
| Failed | 3 | 2 | -1 |

## Failure Analysis

24 low-score cases. Root cause shift: REASONING_ERROR now dominant (71%), not RETRIEVAL_FAILURE. 51% of failures have GT object already in enrichment — pipeline has data but VLM doesn't use it.

Full analysis: `docs/eval_analysis_v12_20260330.md`

## Files Changed

- `runtime/base.py` — tool-usage rules in system prompt
- `keyframe_selector.py` — open_ended regression fix, between-midpoint
- `agent_config.py` — GPT-5.4
- `llm_client.py` — invoke_with_full_retry()
- `openeqa_official_eval.py` — concurrent scoring
- `openeqa_official_question_pilot.py` — default workers=6
