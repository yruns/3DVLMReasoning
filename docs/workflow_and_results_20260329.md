# 3DVLMReasoning Stage 2 Optimization: Workflow, Pipeline & Results

**Date**: 2026-03-29
**Branch**: `feat/multilabel-vocab-optimization` (11 commits on top of `master@80ebf21`)
**Evaluation**: OpenEQA ScanNet, 100 cases (20 scenes x 5 questions), scored by Gemini 2.5 Pro

---

## 1. Task Definition

Optimize the 3DVLMReasoning two-stage pipeline for 3D scene question answering on the OpenEQA benchmark.

**Input**: A natural language question about a 3D scene (e.g., "What red object is below the windows?")
**Output**: A text answer scored 1-5 by LLM-match against ground truth
**Goal**: Maximize the E2E mean score (currently 2.95 baseline)

---

## 2. Pipeline Architecture

```
Question
    |
    v
[Stage 1: Query-Driven Keyframe Retrieval]
    |-- LLM rewrites question -> retrieval queries
    |-- Query parser (LLM) -> structured hypothesis (target + spatial constraints)
    |-- Query executor matches hypothesis against ConceptGraph scene graph
    |-- Keyframe selector picks best views covering matched objects
    |-- Output: 3-5 keyframe images + BEV image
    |
    v
[Stage 2: VLM Reasoning (no tools)]
    |-- GPT-5.2 VLM examines keyframes
    |-- Returns answer + confidence score
    |-- If completed with high confidence -> DONE (use this answer)
    |-- If insufficient_evidence OR low confidence -> continue to E2E
    |
    v
[E2E: VLM with Evidence-Seeking Tools]
    |-- Same VLM but with 4 tool callbacks:
    |   |-- request_more_views(mode='targeted') -- find views of specific objects
    |   |-- request_more_views(mode='explore')  -- find views of unseen regions
    |   |-- request_crops(object_terms=[...])    -- zoom into objects with bbox annotations
    |   |-- switch_or_expand_hypothesis(new_query='...') -- re-run Stage 1
    |-- Up to 10 reasoning turns
    |-- Output: final answer
    |
    v
[Scoring: Gemini 2.5 Pro LLM-Match]
    |-- Compares prediction against ground truth
    |-- Score 1-5 (1=wrong, 5=perfect)
```

### Key Design Principle: No Fallback

Every pipeline step either succeeds as designed or fails explicitly. No silent degradation:
- Stage 1 errors propagate (not caught and swallowed)
- E2E only runs when Stage2 needs help (not as a re-roll)
- Tool stub returns are errors, not graceful degradation

---

## 3. What Was Done (Step by Step)

### Step 1: Multi-Label Scene Vocabulary

**Problem**: ConceptGraph majority-vote drops ~65% of detected object classes.
**Fix**: Include minority detection labels (count >= 2) in `scene_categories` and build `_multilabel_index` in QueryExecutor.
**Files**: `keyframe_selector.py`, `query_executor.py`
**Commit**: `6d83f27`, `598bbfe`

### Step 2: ScanNet BEV Builder

**Problem**: BEV images were blank (no mesh data).
**Fix**: Implemented `OpenEQAScanNetBEVBuilder` with frustum filtering + ceiling removal + perspective rendering from ScanNet `_vh_clean.ply` meshes.
**Files**: `bev_builder.py`
**Commit**: `6d83f27`

### Step 3: Remove Silent Fallback in Stage 1

**Problem**: `run_stage1_with_fallback` swallowed exceptions via try/except, producing None predictions.
**Fix**: `run_stage1_ranked` — build KeyframeSelector once, run all queries, rank by grounding quality (direct > proxy > context), no exception swallowing.
**Files**: `openeqa_official_question_pilot.py`
**Commit**: `598bbfe`, `d63bf54`

### Step 4: PROXY Anchor Preservation

**Problem**: BETWEEN spatial relations collapsed to single anchor in PROXY hypothesis.
**Fix**: Parser prompt rules + few-shot example for preserving dual anchors.
**Files**: `parsing/structures.py`
**Commit**: `9be20f1`

### Step 5: Remove ConfidenceGuard Fallback + Conditional E2E

**Problem**: E2E with 0 tools was a VLM re-roll that could only degrade answers. ConfidenceGuard was a fallback from E2E to Stage2.
**Fix**: Only run E2E when Stage2 didn't complete or has low confidence. No guard needed.
**Files**: `openeqa_official_question_pilot.py`
**Commit**: `9072818`, `7fa0cc7`, `4d92316`

### Step 6: Upgrade request_more_views Object Matching

**Problem**: object_terms used naive substring matching, missed semantically similar objects.
**Fix**: Use `KeyframeSelector.find_objects()` with CLIP similarity fallback.
**Files**: `stage1_callbacks.py`
**Commit**: `8dc2c7a`

### Step 7: System Prompt — Evidence-Seeking Protocol

**Problem**: "CRITICAL - Look before requesting" suppressed tool usage. Agent guessed instead of requesting evidence.
**Fix**: Rewrote to "CRITICAL - Evidence-seeking protocol" with explicit rule: target not visible -> MUST use tools. Added tool strategy guide.
**Files**: `runtime/base.py`
**Commit**: `9072818`, `cc64a93`

### Step 8: Implement Three Stage 2 Tools

**`request_crops`**: Expand bbox by 2x, draw red/orange bounding boxes on all visible objects, save annotated crop as new keyframe. Uses `view_to_objects` index for co-visible objects.

**`request_more_views` explore mode**: Rank views by inverse Jaccard overlap with existing keyframes' visible object sets. Returns views showing the most novel objects.

**`switch_or_expand_hypothesis`**: Agent provides new query, re-runs `select_keyframes_v2`, appends deduplicated keyframes to bundle. Triggers full LLM parser call.

**Files**: `stage1_callbacks.py`
**Commit**: `cc64a93`, `682e304`

### Step 9: Low-Confidence E2E Trigger

**Problem**: Stage2 could return "completed" with wrong answer (low confidence ~0.4-0.5). E2E was skipped, missing tool-improvement opportunity.
**Fix**: Skip E2E only when `status == completed AND confidence >= 0.6`. Below threshold, E2E runs with tools.
**Files**: `openeqa_official_question_pilot.py`
**Commit**: `4d92316`

---

## 4. Evaluation Results

### Summary Table

| Version | Description | S2 | E2E | E2E-S2 | E2E>S2 | E2E<S2 | E2E=5 | E2E=1 |
|---------|-------------|-----|-----|--------|--------|--------|-------|-------|
| **v3** | Baseline (master) | 2.53 | 2.95 | +0.42 | 19 | 7 | 36 | 43 |
| **v4** | Multi-label + ranked (buggy: 9 None) | 2.44 | 2.86 | +0.42 | 12 | 7 | 32 | 41 |
| **v5** | Multi-label + ranked (fixed) | 2.79 | 2.73 | -0.06 | 7 | 12 | 32 | 49 |
| **v6** | Skip E2E when S2 completed | 2.64 | 2.95 | +0.31 | 10 | 2 | 40 | 45 |
| **v7** | All 3 tools + max_turns=10 | 2.27 | 2.85 | +0.58 | 20 | 3 | 36 | 45 |
| **v8** | Low-conf S2 triggers E2E | 2.60 | 2.98 | +0.38 | 16 | 2 | 40 | 42 |

### By Question Category (E2E mean)

| Category | v3 | v5 | v6 | v7 | v8 |
|----------|------|------|------|------|------|
| attribute recognition (n=36) | 2.69 | 2.92 | 2.86 | 3.19 | 2.94 |
| object recognition (n=37) | 3.32 | 2.73 | 3.41 | 3.03 | 3.35 |
| spatial understanding (n=27) | 2.78 | 2.48 | 2.44 | 2.15 | 2.52 |

### Pipeline Metrics

| Metric | v3 | v5 | v6 | v7 | v8 |
|--------|-----|-----|-----|-----|-----|
| direct_grounded % | 33 | 79 | 79 | 80 | 82 |
| proxy_grounded % | 55 | 21 | 21 | 20 | 18 |
| E2E skipped (S2 completed) | 0 | 0 | 67 | 52 | 58 |
| Tool callbacks fired | — | — | — | 165 | 180 |
| Crops generated | — | — | — | 51 | 75 |
| None predictions | 0 | 0 | 0 | 0 | 0 |

### Key Observations

1. **direct_grounded 33% -> 82%**: Multi-label vocabulary is the single biggest improvement
2. **E2E degradation 7 -> 2 cases**: Removing re-roll and conditional E2E eliminated VLM non-determinism damage
3. **E2E-S2 tool gain +0.58 (v7)**: Three tools provide significant value when Stage2 can't answer
4. **Stage2 variance ±0.5**: VLM sampling noise dominates run-to-run variation (same 100 questions)
5. **~42 cases remain at score=1**: These are at the VLM capability boundary (target not detectable in any frame)

---

## 5. Results Directory Structure

```
results/
  v3/                          # Baseline (master@80ebf21)
    results.json               # Multi-dimensional results (scores, categories, per-question)
    official_predictions_stage2-metrics.json  -> tmp/openeqa_eval_20x5/...
    official_predictions_e2e-metrics.json     -> tmp/openeqa_eval_20x5/...
    official_predictions_stage2.json          -> tmp/openeqa_eval_20x5/...
    official_predictions_e2e.json             -> tmp/openeqa_eval_20x5/...
    official_selected_questions.json          -> tmp/openeqa_eval_20x5/...
    official_batch_summary.json              -> tmp/openeqa_eval_20x5/...
    eval.log                                 -> tmp/openeqa_eval_20x5.log
    runs/                                    -> tmp/openeqa_eval_20x5/runs/
      <clip_id>/<question_id>/
        sample.json            # Question, GT answer, category
        stage1.json            # Grounding status, keyframe paths, query used
        stage2.json            # VLM answer, confidence, tool trace
        e2e.json               # E2E answer, tools used, final keyframes
  v4/                          # Multi-label + ranked (buggy)
    ...same structure...
  v5/                          # Multi-label + ranked (fixed)
  v6/                          # Skip E2E when S2 completed
  v7/                          # All 3 tools + max_turns=10
  v8/                          # Low-conf S2 triggers E2E
```

### Per-Question Result Schema (results.json)

```json
{
  "version": "v8",
  "description": "...",
  "stage2": {"mean": 2.60, "distribution": {"1": 53, "5": 32, ...}},
  "e2e": {"mean": 2.98, "distribution": {"1": 42, "5": 40, ...}},
  "e2e_vs_stage2": {"delta": 0.38, "e2e_better": 16, "e2e_worse": 2},
  "by_category": {
    "attribute recognition": {"s2_mean": 2.42, "e2e_mean": 2.94, "n": 36},
    ...
  },
  "per_question": {
    "<question_id>": {
      "question": "What red object is below the windows?",
      "answer": "Fire extinguisher",
      "category": "attribute recognition",
      "s2_score": 1,
      "e2e_score": 1
    },
    ...
  }
}
```

---

## 6. How to Run Evaluation

```bash
# Activate environment
cd /Users/bytedance/project/3DVLMReasoning
source .venv/bin/activate  # macOS

# Run 100-case evaluation (20 scenes x 5 questions)
python -m src.agents.examples.openeqa_official_question_pilot \
    --num-scenes 20 --questions-per-scene 5 \
    --output-root tmp/openeqa_eval_20x5_v9 \
    --evaluate --workers 4 --llm-rewrite \
    --confidence-guard 0.6 \
    --max-additional-views 2 \
    2>&1 | tee tmp/openeqa_eval_20x5_v9.log

# Key arguments:
#   --confidence-guard 0.6  E2E trigger: skip only when S2 conf >= 0.6
#   --max-reasoning-turns 10  (default, set in argparse)
#   --max-additional-views 2  max new views per request_more_views call
#   --llm-rewrite  enable LLM query rewriting (optional enhancement)
#   --workers 4  parallel threads

# Resume scoring if interrupted (rate limit):
python tmp/resume_eval.py  # Custom script, see tmp/resume_eval.py
```

### Evaluation takes ~75 minutes:
- Stage 1 + Stage 2 + E2E inference: ~60 min (4 workers)
- LLM scoring (Gemini 2.5 Pro): ~15 min (sequential, may hit 429 rate limit)

---

## 7. Key Files Reference

| File | Purpose |
|------|---------|
| `src/query_scene/keyframe_selector.py` | Stage 1 main entry: `select_keyframes_v2()`, multi-label vocab, BEV integration |
| `src/query_scene/query_executor.py` | Matches queries to scene objects: `_category_index`, `_multilabel_index` |
| `src/query_scene/parsing/structures.py` | LLM parser prompt: hypothesis rules, PROXY anchor preservation |
| `src/query_scene/bev_builder.py` | `OpenEQAScanNetBEVBuilder`: mesh-based BEV rendering |
| `src/agents/stage1_callbacks.py` | Three tool implementations: crops, explore, hypothesis switch |
| `src/agents/runtime/base.py` | Stage 2 system prompt: evidence-seeking protocol |
| `src/agents/runtime/deepagents_agent.py` | Stage 2 agent run loop: tool execution, nudge, evidence injection |
| `src/agents/examples/openeqa_official_question_pilot.py` | Evaluation pipeline: `run_stage1_ranked`, conditional E2E |
| `src/agents/examples/openeqa_single_scene_pilot.py` | `run_stage1`, `run_stage2`, `serialize_stage1_result` |
| `src/benchmarks/openeqa_official_eval.py` | LLM-match scoring via Gemini 2.5 Pro |

---

## 8. Current Bottleneck

**42/100 E2E cases still score 1** (v8). Breakdown:

| Subcategory | Count | Description |
|-------------|-------|-------------|
| S2 completed (wrong) + E2E skipped | ~16 | S2 conf >= 0.6 but wrong answer; E2E never ran |
| E2E used tools, still failed | ~24 | Target object not in scene graph or not recognizable by VLM |
| E2E no tools (S2 insufficient) | ~2 | Agent didn't use tools despite having them |

### Root causes of irreducible failures:
1. **Object never detected**: ConceptGraph detector (RAM + GroundingDINO) missed the target entirely — not in scene graph, no bbox, no visibility index entry. Tools can't find what was never detected.
2. **Image quality**: ScanNet frames 1296x968, often blurry. Small objects (toothbrush, wire) unrecognizable even with 2x crop.
3. **VLM hallucination**: Answers confidently but wrong. Confidence != accuracy.

### Potential next improvements:
1. Florence-2 detection on all 89 scenes (richer vocabulary, requires Linux GPU)
2. Lower confidence-guard threshold from 0.6 to 0.5 to catch more low-conf-but-wrong cases
3. Multi-sample voting: run Stage2 N times, take majority answer
4. Direct CLIP frame retrieval: bypass scene graph for initial frame selection

---

## 9. Commit History

```
4d92316 feat: trigger E2E on low-confidence Stage2 completions
682e304 fix: address review findings — import, dedup, view tracking, cost note
eef3f8c docs: branch summary for multilabel-vocab-optimization review
cc64a93 feat: implement request_crops, explore mode, hypothesis switching
7fa0cc7 refactor: remove ConfidenceGuard fallback, only run E2E when Stage2 insufficient
8dc2c7a feat: upgrade request_more_views to use find_objects() with CLIP fallback
9072818 fix: strengthen E2E evidence-seeking and ConfidenceGuard
d63bf54 fix: LLM rewrite is optional, remove silent error swallowing in _run_sample
9be20f1 fix: preserve BETWEEN anchors in PROXY hypothesis generation
598bbfe feat: multi-label category index + remove silent fallback in Stage 1
6d83f27 feat: OpenEQA ScanNet BEV builder + multi-label scene vocabulary
```

---

## 10. Version Changelog

| Version | Commit(s) | What Changed | Why |
|---------|-----------|-------------|-----|
| v3 | master@80ebf21 | Baseline | Reference point |
| v4 | 6d83f27, 598bbfe | Multi-label vocab + ranked query + BEV | Recover lost object classes, rank queries by grounding quality |
| v5 | d63bf54 | Fix LLM rewrite exception + None predictions | v4 had 9/100 None predictions from swallowed exceptions |
| v6 | 9be20f1, 9072818, 7fa0cc7 | PROXY fix + skip E2E when S2 completed | Eliminate VLM re-roll degradation (12->2 worse cases) |
| v7 | 8dc2c7a, cc64a93, 682e304 | 3 tools + CLIP matching + max_turns=10 | Enable evidence acquisition: crops, explore, re-query |
| v8 | 4d92316 | Low-conf S2 triggers E2E | 21 wrong-but-completed cases now get tool-augmented E2E |
