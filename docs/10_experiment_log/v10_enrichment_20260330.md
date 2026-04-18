# v10 — LLM Object Enrichment

**Date:** 2026-03-30
**Branch:** `feat/stage2-evidence-seeking-v2`
**Eval:** 100Q (20 scenes x 5 questions)
**Data:** `tmp/openeqa_eval_enriched_v10/`

## What Changed

1. **Object enrichment pipeline** (`src/scripts/enrich_objects.py`): Gemini 2.5 Pro generates per-object metadata (category, description, location, color, nearby_objects, usability) for all 5097 objects across 89 scenes
2. **Pipeline integration** (`keyframe_selector.py:_load_enrichment()`): Populates SceneObject.category/summary/co_objects/affordances from enriched_objects.json, overriding noisy detection labels

## Results

| Metric | v9 | v10 | Delta |
|--------|:---:|:---:|:-----:|
| **MNAS** | 46.5 | **55.4** | **+8.9** |
| Score=1 | 44% | 34% | -10% |
| Score>=4 | 42% | 50% | +8% |
| Failed | 0 | 2 | +2 |

## Failure Analysis

36 low-score cases analyzed. Top issues:
- Tool callbacks not configured (20 cases)
- Single-keyframe fragility (18 cases)
- Query parser proxy term errors (12 cases)

Full analysis: `docs/eval_analysis_v10_enriched_20260330.md`

## Files Changed

- `src/scripts/enrich_objects.py` (+614 lines) — new
- `bashes/openeqa_scannet/8_enrich_objects.sh` (+45 lines) — new
- `src/query_scene/keyframe_selector.py` (+41 lines) — _load_enrichment()
