# OpenEQA 100-Case Evaluation Analysis Report

**Date**: 2026-03-27
**Eval config**: 20 scenes x 5 questions = 100 cases, 4 parallel workers, LLM-rewrite + confidence guard + nudge-based multi-turn

## 1. Overall Results

| Metric | Stage 2 | E2E (with nudge) |
|--------|---------|-------------------|
| Mean score | 2.53/5 | 2.95/5 |
| Pass rate (>=3) | 43% | 56% |
| Perfect (5) | 30 | 36 |
| Fail (1) | 54 | 43 |
| E2E lift | — | +0.42 |

### Tool Usage

| Metric | Stage 2 | E2E |
|--------|---------|-----|
| Total tool calls | 33 | 46 |
| Cases with tools | 29 | 24 |
| Cases where tools improved score | — | 8/24 (avg +3.75) |
| Cases where tools hurt score | — | 1/24 |
| Confidence guard applied | — | 34/100 |

### Stage 1 Status Breakdown

| Status | Count | E2E Mean | Description |
|--------|-------|----------|-------------|
| direct_grounded | 33 | 3.58 | Target found directly in scene graph |
| proxy_grounded | 45 | 2.53 | Target missing, proxy used |
| context_only | 22 | 2.82 | No grounding, context fallback |

### By Category

| Category | n | S2 | E2E | Lift |
|----------|---|----|----|------|
| Object recognition | 37 | 2.78 | 3.32 | +0.54 |
| Attribute recognition | 36 | 2.31 | 2.69 | +0.39 |
| Spatial understanding | 27 | 2.48 | 2.78 | +0.30 |

---

## 2. Root Cause Analysis: Majority-Vote Information Loss

### The Problem

The system has a **critical information loss** between the 2D detection layer and the 3D scene graph:

```
RAM per-frame detection (115 classes)
    ↓ ConceptGraph 3D merge (aggregate detections per object)
    ↓ Majority-vote (keep only the #1 class per object)
    ↓ scene_categories (unique primary classes only)
    ↓ LLM parser vocabulary (hard enum constraint)
```

Each step discards information. The final `scene_categories` list typically retains only **30-40%** of the original detection vocabulary.

### Evidence Across 5 Audited Scenes

| Scene | RAM vocab | scene_categories | Loss | Key objects dropped |
|-------|-----------|-----------------|------|---------------------|
| 103 (bathroom) | 105 | 28 | 73% | lamp, hanger, phone, camera |
| 105 (apartment) | 115 | 40 | 65% | vacuum, fire extinguisher |
| 109 (dormitory) | 117 | ~35 | 70% | calculator, photo frame |
| 102 (dormitory) | 100 | ~40 | 60% | charger, recycling bin |
| 100 (library) | 59 | ~30 | 49% | bottle, trash bin |

### Concrete Example: "vacuum" in Scene 105

- Object 55 raw detections: bottle(6), appliance(5), baby_carriage(3), **vacuum(1)**, ...
- Majority-vote → primary category = "bottle"
- "vacuum" permanently lost from `scene_categories`
- LLM forced to proxy "vacuum cleaner" → "appliance" → retrieves fridge frames

### Concrete Example: "lamp" in Scene 103

- Object 55 raw detections: sink(6), **lamp(4)**, drain(3), ...
- Majority-vote → primary category = "sink"
- "lamp" permanently lost from `scene_categories`
- LLM forced to proxy "lamp" → "bottle" → retrieves irrelevant frames
- GT answer: "in the mirror above the sink" — the lamp reflection was detected but overruled

---

## 3. Confirmed System Bugs

### Bug 1: scene_categories Built from Primary Labels Only (CRITICAL)

**File**: `src/query_scene/keyframe_selector.py` lines 357-360

```python
self.scene_categories = list(
    {obj.object_tag if obj.object_tag else obj.category for obj in self.objects}
)
```

Only the majority-vote primary category is included. All minority detection classes are discarded.

**Impact**: ~65% of RAM-detected classes disappear from the parser vocabulary. This forces proxy grounding for objects that WERE actually detected by the 2D detector.

### Bug 2: QueryExecutor Category Index is Single-Label (HIGH)

**File**: `src/query_scene/query_executor.py` lines 112-118

The `_category_index` maps only the primary category to object IDs. Even if Bug 1 were fixed (so "vacuum" appears in `scene_categories`), searching for "vacuum" would still fail because Object 55's primary category is "bottle", not "vacuum".

**Impact**: Fixing Bug 1 alone is insufficient. The executor's matching logic must also consult multi-label detection histories.

### Bug 3: gsa_classes.json is Misleading (HIGH)

**File**: `data/OpenEQA/scannet/*/conceptgraph/gsa_classes_ram_withbg_allclasses.json`

This file lists ALL classes ever detected across any frame (the detector vocabulary union). It does NOT reflect the actual 3D scene graph categories after majority-vote. Anyone reading this file would incorrectly assume "lamp is in the scene graph."

**Impact**: Incorrect debugging, analysis, and expectations. The file name implies "all classes" but the operational vocabulary is a strict subset.

### Bug 4: run_stage1_with_fallback Accepts First Non-Empty Result (MEDIUM)

**File**: `src/agents/examples/openeqa_official_question_pilot.py` lines 387-400

```python
for query in queries:
    try:
        selector, stage1_result, stage1_summary = run_stage1(...)
        return selector, stage1_result, stage1_summary  # Returns immediately!
    except Exception as exc:
        last_error = exc
```

The first query candidate that produces ANY result (including low-quality `proxy_grounded`) causes immediate return. Later candidates that might achieve `direct_grounded` are never tried.

**Example**: For "Which device is the person using to record the video?", candidate "camera" → proxy_grounded via "picture frame". Next candidate "phone" was never tried (though in this case "phone" was also missing from scene_categories).

### Bug 5: BETWEEN Spatial Relation Destroyed in Proxy (MEDIUM)

When the LLM generates a proxy hypothesis for a query with BETWEEN relation (2 anchors), the proxy collapses both anchors into a single proxy anchor.

**Example**: "Which blue object is between table and fridge?" → proxy target "bag" with anchor "computer desk" (single). The dual-anchor spatial constraint was completely lost.

### Bug 6: Mega-Objects from Over-Merging (LOW)

Object 17 in Scene 105 has 605 detections across 26 different class labels. This means physically distinct counter objects (bottles, condiment, coffee machine, etc.) were merged into one 3D object. This makes spatial retrieval unreliable since the "object" spans the entire counter.

---

## 4. Proxy Grounding Pattern Analysis

### Proxy Target Selection Tendencies

| Proxy target | Frequency | Quality | Notes |
|--------------|-----------|---------|-------|
| "book" | 3 times | POOR | Used as catch-all for desk/shelf objects |
| "DVD" | 3 times | POOR | LLM default when uncertain |
| "other item" | 3 times | POOR | Too diffuse, no spatial specificity |
| "backpack" | 2 times | MEDIUM | Scene-specific, coincidental spatial overlap |
| "appliance" | 2 times | MEDIUM | Semantically reasonable but spatially imprecise |
| "bottle" | 2 times | POOR | Majority-vote artifact (many things → "bottle") |

### Proxy Success vs Failure Patterns

**When proxy succeeds** (14/45 = 31% score 5):
- Target object is large and spatially co-located with proxy object (bookshelf near backpack)
- Proxy directs keyframes to the correct general area by coincidence
- VLM is strong enough to identify the actual answer from visual evidence

**When proxy fails** (23/45 = 51% score 1):
- Target object is in a different spatial region than proxy
- Target object is small/floor-level/occluded
- Proxy keyword has wrong semantic direction (book for calculator, DVD for electronic item)
- Spatial constraints (BETWEEN, LEFT_OF) are lost in proxy translation

---

## 5. E2E Nudge Mechanism Evaluation

### Nudge Design

When the agent returns `insufficient_evidence` or `needs_more_evidence`, a follow-up message nudges it to use evidence-seeking tools (request_more_views, request_crops) before giving up.

### Results

- 24/100 cases used E2E tools
- 8/24 tool-using cases improved score (avg +3.75 points)
- 1/24 tool-using cases regressed
- Biggest wins: blinds color (1→5), room number (1→5), blue bin contents (1→5)

### Limitation

The nudge cannot overcome Stage 1 retrieval failures when the `more_views_callback` has no better frames to offer. If the scene graph lacks the target object entirely, requesting more views from the same flawed retrieval system won't help.

---

## 6. Additional Findings

### RAM Detector Noise Categories

The RAM detector's vocabulary contains many non-object labels that waste capacity:

| Type | Examples |
|------|----------|
| Verbs/Actions | sit, lay, hang, open, fill, push, lead to, take, walk |
| Adjectives/Colors | black, white, gray, red, green, brown, dark, flat |
| Abstract | peak, shine, comfort, mess, stuff, dormitory |

These account for 15-25% of detected "classes" per scene and provide no spatial grounding value.

### Floor-Level Object Disadvantage

Objects below z=0.25m (trash bins, shoes, cables) appear in far fewer frames than table/wall-height objects. The visibility-driven keyframe selection naturally biases toward prominent objects, making small ground-level items systematically harder to retrieve.

### Keyframe Diversity

Some cases received near-duplicate keyframes (e.g., frames 2 and 14 from the same camera trajectory segment). A camera-pose diversity filter would improve information coverage.
