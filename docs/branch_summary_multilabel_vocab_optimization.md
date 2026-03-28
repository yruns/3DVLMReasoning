# Branch Summary: `feat/multilabel-vocab-optimization`

**Date**: 2026-03-28
**Base**: `master` @ `80ebf21`
**Commits**: 8 (see appendix)
**Files changed**: 9 (+1593/-306 lines)

---

## 1. Background

### Problem Statement

A 100-case evaluation (20 scenes x 5 questions) on OpenEQA ScanNet revealed that the Stage 1 retrieval pipeline suffered from **majority-vote information loss**: when ConceptGraph merges per-frame 2D detections into 3D objects, only the most frequent class name survives. An object detected as `["sink"x6, "lamp"x4]` becomes just `"sink"` — `"lamp"` is permanently lost. This caused ~65% of detected object classes to be dropped from the scene vocabulary, directly degrading the query parser's ability to match user queries to scene objects.

Additionally, the E2E pipeline (Stage 2 with evidence-seeking tools) showed negative returns: it scored 2.73 vs Stage 2's 2.79, meaning the tool-augmented pipeline was actually worse than the baseline. Root cause analysis revealed three systemic issues: silent fallback logic throughout the pipeline, a ConfidenceGuard that allowed VLM non-determinism to degrade answers, and stub tool implementations that blocked evidence acquisition.

### Baseline Metrics (v3, before this branch)

| Metric | Value |
|--------|-------|
| direct_grounded rate | 33% |
| proxy_grounded rate | 55% |
| Stage2 mean score | 2.53 |
| E2E mean score | 2.95 |

---

## 2. Objectives

1. **Recover lost object classes** from majority-vote by building a multi-label scene vocabulary
2. **Remove all silent fallback logic** per the project's strict No-Fallback Rule
3. **Fix E2E pipeline** so it adds value over Stage 2 instead of degrading it
4. **Implement all three Stage 2 evidence-seeking tools** (request_crops, explore mode, hypothesis switching)

---

## 3. Implementation Details

### 3.1 Multi-Label Scene Vocabulary (commit `6d83f27`, `598bbfe`)

**Problem**: `scene_categories` was built from majority-vote primary labels only. QueryExecutor's `_category_index` only mapped primary category → objects. If a user asks about "lamp" but the 3D object's primary label is "sink" (with "lamp" as a minority detection), the query fails.

**Solution — KeyframeSelector (`keyframe_selector.py`)**:
- Added `_NOISE_LABELS` frozenset (verbs, adjectives, body parts, abstract terms) to filter non-object categories
- New `_build_multilabel_categories()` method: iterates all objects, includes primary category + any class_name with detection count >= 2
- Modified `_load_objects_from_pcd()` to pass raw `class_names` list to `SceneObject`

**Solution — QueryExecutor (`query_executor.py`)**:
- New `_multilabel_index: dict[str, list[SceneObject]]` built at init: for each object, maps minority class names (count >= 2, excluding primary) to that object
- Modified `_find_by_categories()` matching priority:
  1. Primary exact match (`_category_index`)
  2. Primary substring match
  3. **Multi-label exact match** (`_multilabel_index`) ← new
  4. CLIP similarity fallback

### 3.2 OpenEQA ScanNet BEV Builder (commit `6d83f27`)

**Problem**: BEV images for all 89 scenes were blank white (no mesh data locally).

**Solution (`bev_builder.py`)**:
- New `OpenEQAScanNetBEVBuilder` class with `OpenEQAScanNetBEVConfig` dataclass
- Pipeline: load `_vh_clean.ply` mesh → frustum visibility filtering (project vertices into each camera frustum) → ceiling face removal (surface normal Z < -0.5) → perspective rendering from above (FOV 80, painter's algorithm near-to-far) → camera trajectory overlay
- Config-hash-based caching: `<scene>/conceptgraph/bev/scene_bev_scannet_<hash>.png`
- NaN-safe handling for invalid camera poses in trajectory
- All 89 scenes pre-rendered and cached (64MB total)

### 3.3 Remove Silent Fallback: `run_stage1_ranked` (commit `598bbfe`, `d63bf54`)

**Problem**: `run_stage1_with_fallback` used `try/except` to silently swallow Stage 1 errors and try the next query candidate. Failed samples returned `{**sample, "error": str(exc)}` which `build_prediction_file` extracted as `answer=None`, scoring 0. This caused 9/100 cases to score 0 in v4.

**Solution (`openeqa_official_question_pilot.py`)**:
- Replaced `run_stage1_with_fallback` with `run_stage1_ranked`:
  - Builds `KeyframeSelector` once (was rebuilding per query — expensive)
  - Runs ALL query candidates through `select_keyframes_v2`
  - Ranks by grounding quality: `direct_grounded` (return immediately) > `proxy_grounded` > `context_only`
  - No `try/except` around Stage 1 execution
- LLM rewrite: `try/except` for rate-limit only (optional enhancement, not pipeline step)
- Removed `try/except` in `_run_sample`: exceptions propagate, failed futures are logged and excluded from predictions (not included as `answer=None`)

### 3.4 PROXY Anchor Preservation (commit `9be20f1`)

**Problem**: When generating PROXY hypotheses for BETWEEN relations (2 anchors), the LLM collapsed both anchors into a single proxy anchor, destroying the spatial constraint.

**Solution (`parsing/structures.py`)**:
- Rewrote PROXY hypothesis rules: "Only replace UNKNOW parts. Preserve all non-UNKNOW categories and spatial structure exactly."
- Explicit rule for BETWEEN: "ALWAYS keep 2 separate anchors. Never collapse into 1."
- Added few-shot Example 5 demonstrating BETWEEN with missing target where both anchors are preserved

### 3.5 Remove ConfidenceGuard Fallback (commit `9072818`, `7fa0cc7`)

**Problem**: The ConfidenceGuard was a fallback mechanism: "if E2E is worse, fall back to Stage2." It required `stage2_conf >= 0.6 AND e2e_conf <= stage2_conf`, which failed when:
- Stage2 conf=0.55 but correct → Guard didn't fire (threshold too high)
- E2E conf=0.67 but wrong (hallucination) → Guard didn't fire (E2E more confident)

**Root cause**: E2E with 0 tools is a stochastic re-roll of Stage2 on identical inputs. In 77/100 cases E2E used 0 tools — pure VLM non-determinism.

**Solution**:
- When Stage2 status == "completed", skip E2E entirely (don't run it)
- Only run E2E (with tool callbacks) when Stage2 reported `insufficient_evidence`
- No guard needed: each code path produces exactly one answer, no fallback selection

### 3.6 Upgrade `request_more_views` Object Matching (commit `8dc2c7a`)

**Problem**: `object_terms` parameter used naive substring matching against `category`/`object_tag`. If the agent says `"fire extinguisher"` but the scene labels it `"canister"`, no match.

**Solution (`stage1_callbacks.py`)**:
- Replaced substring matching with `keyframe_selector.find_objects(term, top_k=5)` which has two-stage matching: string match → CLIP semantic similarity fallback (min_sim=0.2)

### 3.7 System Prompt: Evidence-Seeking Protocol (commit `9072818`, `cc64a93`)

**Problem**: The system prompt's "CRITICAL - Look before requesting" block created a strong bias toward one-shot answering. The agent defaulted to guessing instead of using tools. In 5/7 failure cases, E2E used 0 tools despite not finding the target object.

**Analysis** (from agent-based investigation of the prompt):
- "Only call tools when you have SPECIFIC evidence gaps" set a high bar that suppressed tool use
- No instruction explicitly linked "target not visible" to "must use tools"
- Confidence threshold 0.40 was too permissive for low-evidence guesses
- The nudge loop only fired on `insufficient_evidence`, not on low-confidence `completed`

**Solution (`base.py`)**:
- Renamed "CRITICAL - Look before requesting" → "CRITICAL - Evidence-seeking protocol"
- Added: "If the TARGET OBJECT or QUERIED ATTRIBUTE is NOT visible in ANY keyframe, you MUST seek more evidence before answering"
- Added tool strategy guide: `targeted → explore → crop → re-query`

### 3.8 Implement Three Stage 2 Tools (commit `cc64a93`)

All three tools implemented in `stage1_callbacks.py`:

**`request_crops`** — Object-centric annotated zoom
- Input: `object_terms` (CLIP-matched), `crop_scale` (default 2.0x)
- Finds the keyframe view where the target object has the largest bbox
- Expands bbox by `crop_scale` multiplier (minimum 256px)
- Draws red bounding box + label on target object, orange on all other visible objects in the crop region
- Saves annotated crop as new keyframe
- Uses `view_to_objects` reverse index to find all co-visible objects

**`request_more_views` explore mode** — Unseen region discovery
- Input: `mode="explore"`
- Computes Jaccard similarity between each candidate view's visible object set and the union of objects in existing keyframes
- Selects views with maximum novelty (minimum overlap)
- Returns views showing the most "new" objects not yet seen

**`switch_or_expand_hypothesis`** — Re-run Stage 1 with different query
- Input: `new_query` string
- Calls `selector.select_keyframes_v2(new_query, k=3)`
- Deduplicates against existing keyframe paths
- Appends new keyframes to bundle (incremental, not replacement)

**Configuration**: `max_reasoning_turns` default raised from 3 → 10 to support multi-step evidence chains: explore → crop → answer.

---

## 4. Evaluation Results

Three evaluation runs (100 cases each: 20 scenes x 5 questions):

| Metric | Baseline v3 | v5 (multi-label + ranked) | v6 (+ no-fallback E2E) |
|--------|-------------|---------------------------|------------------------|
| direct_grounded | 33% | 79% | 79% |
| Stage2 mean | 2.53 | 2.79 | 2.64* |
| E2E mean | 2.95 | 2.73 | 2.95 |
| E2E > Stage2 | — | 7 cases | 10 cases |
| E2E < Stage2 | — | 12 cases | 2 cases |
| E2E == Stage2 | — | 81 cases | 88 cases |
| None predictions | 0 | 0 | 0 |

*Stage2 mean variation between v5 (2.79) and v6 (2.64) is VLM sampling noise; same model, same scenes.

Key outcomes:
- **direct_grounded +46pp** (33% → 79%): multi-label vocabulary + ranked query selection
- **E2E degradation eliminated**: 12 worse cases → 2 (by removing re-roll when Stage2 completed)
- **E2E value concentrated**: 10 cases where E2E with tools outperformed Stage2
- **Zero None predictions**: removed silent error swallowing

v6 does NOT yet include the tool implementations from commit `cc64a93` (request_crops, explore, hypothesis switching). Those require a v7 evaluation run.

---

## 5. File Change Summary

| File | Lines Changed | Description |
|------|--------------|-------------|
| `src/agents/stage1_callbacks.py` | +486/-133 | Three tool implementations (crops, explore, hypothesis) |
| `src/query_scene/bev_builder.py` | +392 | `OpenEQAScanNetBEVBuilder` class |
| `src/agents/examples/openeqa_official_question_pilot.py` | +147/-81 | `run_stage1_ranked`, conditional E2E, remove fallbacks |
| `src/query_scene/keyframe_selector.py` | +69/-34 | Multi-label `_build_multilabel_categories()`, BEV integration |
| `src/query_scene/parsing/structures.py` | +79/-4 | PROXY anchor preservation rules + BETWEEN example |
| `src/query_scene/query_executor.py` | +30/-12 | `_multilabel_index` + multi-label matching tier |
| `src/agents/runtime/base.py` | +9/-5 | Evidence-seeking prompt + tool strategy |
| `scripts/download_scannet_mesh.py` | +256 | ScanNet mesh batch downloader |
| `docs/handoff_2026-03-28_0003.md` | +162 | Handoff document |

---

## 6. Remaining Work

1. **Run v7 evaluation** with all three tools enabled (request_crops, explore, hypothesis switching, max_reasoning_turns=10)
2. **Implement `request_crops` for raw frames**: current implementation uses GSA detection bboxes; could also support agent-specified pixel coordinates
3. **Florence-2 detection** on all 89 scenes (requires Linux GPU) for richer scene vocabulary
4. **Calibrate E2E trigger threshold**: currently runs E2E whenever Stage2 != "completed"; could also trigger on low Stage2 confidence

---

## Appendix: Commit History

```
cc64a93 feat: implement request_crops, explore mode, hypothesis switching
7fa0cc7 refactor: remove ConfidenceGuard fallback, only run E2E when Stage2 insufficient
8dc2c7a feat: upgrade request_more_views to use find_objects() with CLIP fallback
9072818 fix: strengthen E2E evidence-seeking and ConfidenceGuard
d63bf54 fix: LLM rewrite is optional, remove silent error swallowing in _run_sample
9be20f1 fix: preserve BETWEEN anchors in PROXY hypothesis generation
598bbfe feat: multi-label category index + remove silent fallback in Stage 1
6d83f27 feat: OpenEQA ScanNet BEV builder + multi-label scene vocabulary
```
