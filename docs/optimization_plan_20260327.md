# Stage 1 Optimization Plan

**Date**: 2026-03-27
**Based on**: 100-case OpenEQA evaluation + 5-scene proxy audit
**Goal**: Fix the majority-vote information loss pipeline, improve proxy quality, and strengthen E2E evidence seeking

---

## Phase 1: Multi-Label Scene Vocabulary (P0, estimated +15-20% E2E pass rate)

### Problem
`scene_categories` only contains majority-vote primary labels (~35 classes). ~65% of RAM-detected classes are silently dropped, forcing unnecessary proxy grounding.

### 1.1 Expand scene_categories to Include All Detection Labels

**File**: `src/query_scene/keyframe_selector.py`

**Current** (lines 357-360):
```python
self.scene_categories = list(
    {obj.object_tag if obj.object_tag else obj.category for obj in self.objects}
)
```

**Target**:
```python
all_cats = set()
for obj in self.objects:
    # Primary category
    primary = obj.object_tag if obj.object_tag else obj.category
    all_cats.add(primary)
    # All detection class names with count >= min_detections
    if hasattr(obj, 'class_name') and isinstance(obj.class_name, list):
        from collections import Counter
        counts = Counter(obj.class_name)
        for cls, cnt in counts.items():
            if cnt >= 2 and cls.lower() not in _NOISE_LABELS:
                all_cats.add(cls)
self.scene_categories = sorted(all_cats)
```

Where `_NOISE_LABELS` filters out verbs/adjectives/abstract terms:
```python
_NOISE_LABELS = frozenset({
    "sit", "lay", "hang", "open", "fill", "push", "lead to", "take", "walk",
    "attach", "hide", "slide", "curl", "sleep", "stand", "make", "wrap",
    "black", "white", "gray", "red", "green", "brown", "dark", "flat",
    "peak", "shine", "comfort", "mess", "stuff", "dormitory", "camouflage",
    "couple", "selfie", "man", "woman", "person", "head", "nose", "hand",
    "foot", "face",
})
```

**Validation**: After this change, run the same 100 cases. The number of `direct_grounded` should increase from 33 to ~55+.

### 1.2 Build Multi-Label Category Index in QueryExecutor

**File**: `src/query_scene/query_executor.py`

**Current** (lines 112-118): `_category_index` maps only primary category → object IDs.

**Target**: Build a secondary index from all detection class names:

```python
# Primary index (exact match on primary category)
self._category_index: dict[str, list[int]] = defaultdict(list)
for i, obj in enumerate(objects):
    self._category_index[obj.category.lower()].append(i)

# Multi-label index (all detection classes with count >= 2)
self._multilabel_index: dict[str, list[int]] = defaultdict(list)
for i, obj in enumerate(objects):
    if hasattr(obj, 'class_name') and isinstance(obj.class_name, list):
        from collections import Counter
        for cls, cnt in Counter(obj.class_name).items():
            if cnt >= 2:
                self._multilabel_index[cls.lower()].append(i)
```

**Matching priority** in `_find_by_categories()`:
1. Exact match on primary category (existing)
2. Substring match on primary category (existing)
3. **NEW**: Exact match on multi-label index
4. CLIP semantic similarity fallback (existing)

### 1.3 Generate Accurate loaded_categories.json

After scene loading, write the actual operational vocabulary to a separate file:

```python
# In keyframe_selector.py after building scene_categories
loaded_cats_path = self.scene_path / "loaded_categories.json"
loaded_cats_path.write_text(json.dumps(sorted(self.scene_categories), indent=2))
```

This avoids confusion between `gsa_classes_ram_withbg_allclasses.json` (detector vocab) and actual queryable categories.

---

## Phase 2: Fallback Strategy Improvement (P1, estimated +5-8% on proxy cases)

### 2.1 Prefer direct_grounded Over proxy_grounded in Fallback

**File**: `src/agents/examples/openeqa_official_question_pilot.py`

**Current** (lines 387-400): Returns first non-exception result regardless of quality.

**Target**:
```python
def run_stage1_with_fallback(sample, scene_root, runtime_scene, stride, args):
    queries = build_stage1_query_candidates(sample["question"])
    # ... LLM rewrite logic ...

    best_proxy_result = None  # Track best proxy result as fallback

    for query in queries:
        try:
            selector, stage1_result, stage1_summary = run_stage1(...)
            status = stage1_summary.get("status", "")

            # Direct grounded — return immediately (best possible)
            if status == "direct_grounded":
                stage1_summary["official_question"] = sample["question"]
                stage1_summary["stage1_query_used"] = query
                stage1_summary["stage1_query_candidates"] = queries
                return selector, stage1_result, stage1_summary

            # Proxy/context — save as fallback, keep trying
            if best_proxy_result is None or (
                status == "proxy_grounded"
                and best_proxy_result[2].get("status") == "context_only"
            ):
                stage1_summary["official_question"] = sample["question"]
                stage1_summary["stage1_query_used"] = query
                stage1_summary["stage1_query_candidates"] = queries
                best_proxy_result = (selector, stage1_result, stage1_summary)

        except Exception as exc:
            last_error = exc

    if best_proxy_result is not None:
        return best_proxy_result

    raise RuntimeError(f"Stage 1 failed after {len(queries)} candidates: {last_error}")
```

### 2.2 Preserve Matched Anchors in Proxy Hypotheses

**File**: `src/query_scene/parsing/parser.py` (prompt engineering)

Add to the LLM system prompt for hypothesis generation:

```
PROXY ANCHOR RULE: When generating a PROXY hypothesis, if the anchor object's
category already exists in SCENE CATEGORIES, keep the original anchor category.
Only proxy the TARGET categories. Never replace a directly-matchable anchor
with a proxy term.

Example:
  Query: "blue object between table and fridge"
  SCENE CATEGORIES include: table, fridge
  WRONG proxy: target="bag", anchor="computer desk"
  CORRECT proxy: target="bin", anchors=["table", "fridge"]  (anchors preserved!)
```

---

## Phase 3: E2E Evidence Seeking Enhancement (P1, estimated +3-5%)

### 3.1 Lower Nudge Threshold

**Current**: Nudge only fires on `insufficient_evidence` / `needs_more_evidence`.

**Target**: Also nudge when `completed` with confidence < 0.4 AND tool calls = 0:
```python
# In deepagents_agent.py run() loop
if (
    response.status == Stage2Status.COMPLETED
    and response.confidence < 0.4
    and len(runtime.tool_trace) == 0
    and turns_used < task.max_reasoning_turns
):
    # Low confidence answer with no tool usage — nudge to verify
    nudge = self._build_verification_nudge(response, runtime)
    messages.append(nudge)
    continue
```

### 3.2 Attribute-Question Detection for Targeted Crops

When the question is about an attribute (color, size, material), automatically request crops of the target object in the initial E2E prompt:

```python
# Detect attribute questions
ATTRIBUTE_PATTERNS = ["what color", "what is the color", "what material", "how big", "how many"]
if any(p in task.user_query.lower() for p in ATTRIBUTE_PATTERNS):
    # Append instruction to seek crops
    prompt += (
        "\n\nThis is an ATTRIBUTE question. If the target object is visible but "
        "small or unclear, use request_crops to get a close-up view before "
        "answering. Attribute questions require high visual detail."
    )
```

### 3.3 Keyframe Diversity Filter

**File**: `src/query_scene/keyframe_selector.py`

Add a minimum camera-pose distance constraint when selecting keyframes:

```python
def _enforce_keyframe_diversity(
    self,
    candidate_views: list[int],
    selected_views: list[int],
    min_rotation_deg: float = 15.0,
    min_translation_m: float = 0.3,
) -> list[int]:
    """Filter out candidates too similar in pose to already selected views."""
    filtered = []
    for v in candidate_views:
        if v in selected_views:
            continue
        pose_v = self.poses[v]
        too_close = False
        for s in selected_views:
            pose_s = self.poses[s]
            # Check translation distance
            trans_dist = np.linalg.norm(pose_v[:3, 3] - pose_s[:3, 3])
            if trans_dist < min_translation_m:
                too_close = True
                break
        if not too_close:
            filtered.append(v)
    return filtered if filtered else candidate_views  # Fallback to original if all too close
```

---

## Phase 4: Detection Backend Upgrade (P2, long-term)

### 4.1 Florence-2 Hybrid Detection

The Florence-2 pipeline (already implemented in `conceptgraph/detection/generate_florence2.py`) uses ScanNet200 vocabulary (200 classes) with OD + vocabulary grounding. Previous experiments showed:
- +36% more objects detected
- 0.3% junk rate (vs 3.1% for RAM)
- 77 unique categories per scene (vs ~40 for RAM)

**Action items**:
1. Run Florence-2 detection on all 89 OpenEQA scenes
2. Run SLAM pipeline to generate `*florence2*_post.pkl.gz`
3. Update `keyframe_selector.py` auto-detect to prefer Florence-2 PCD files
4. Re-evaluate 100 cases — expect direct_grounded rate to jump from 33% to 50%+

### 4.2 Address Over-Merging in SLAM Pipeline

Objects like Scene 105's Object 17 (605 detections, 26 class labels) indicate the 3D merge threshold is too aggressive.

**File**: `conceptgraph/configs/slam_pipeline/base.yaml`

Tune:
```yaml
merge_overlap_thresh: 0.5     # Was 0.7 — reduce to split over-merged objects
merge_visual_sim_thresh: 0.85  # Was 0.8 — increase to prevent merging dissimilar items
obj_min_detections: 5          # Was 3 — increase to filter spurious objects
```

---

## Phase 5: RAM Noise Filtering (P2)

### 5.1 Filter Non-Object Detection Labels

RAM produces many verb/adjective/abstract labels that waste vocabulary capacity. Add a post-processing filter in detection scripts:

```python
NOISE_LABELS = {
    "sit", "lay", "hang", "open", "fill", "push", "lead to", "take", "walk",
    "attach", "hide", "slide", "curl", "sleep", "stand", "make", "wrap",
    "couple", "selfie", "peak", "shine", "comfort", "mess", "stuff",
    "dormitory", "camouflage",
}

def filter_noise_labels(classes: list[str]) -> list[str]:
    return [c for c in classes if c.lower() not in NOISE_LABELS]
```

Apply in `keyframe_selector.py` during object loading and in `scene_categories` construction.

---

## Evaluation Plan

### Metrics to Track

| Metric | Current (v3) | Target (after Phase 1+2) | Target (after Phase 1-4) |
|--------|-------------|-------------------------|-------------------------|
| direct_grounded rate | 33% | 55%+ | 65%+ |
| proxy_grounded rate | 45% | 25% | 15% |
| E2E mean score | 2.95/5 | 3.5/5 | 4.0/5 |
| E2E pass rate | 56% | 70%+ | 80%+ |
| E2E fail rate (1) | 43% | 25% | 15% |

### Evaluation Protocol

1. **Same 100 cases** (20 scenes x 5 questions) for A/B comparison
2. **Fixed eval model** (gemini-2.5-pro) for scoring consistency
3. **Report per-case deltas** to distinguish systematic improvement from non-determinism
4. **Separate S2 and E2E** to isolate Stage 1 vs Stage 2 contributions

### Milestones

| Phase | Effort | Expected Improvement | Dependencies |
|-------|--------|---------------------|--------------|
| Phase 1 (multi-label) | 2-3 days | +15-20% pass rate | None |
| Phase 2 (fallback) | 1 day | +5-8% on proxy cases | None |
| Phase 3 (E2E enhance) | 1-2 days | +3-5% | Phase 1 |
| Phase 4 (Florence-2) | 3-5 days | +10-15% | GPU access |
| Phase 5 (noise filter) | 0.5 day | +2-3% | None |
