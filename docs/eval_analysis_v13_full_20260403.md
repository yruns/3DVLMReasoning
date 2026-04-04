# v13 Full OpenEQA Failure Analysis (1050Q)

**Date**: 2026-04-03
**Eval**: `tmp/openeqa_eval_v13_full/` (1050/1200 questions, MNAS 71.4)
**Analyzed**: 96 of 222 low-score cases (score<=2) across 33 scenes
**Method**: 8 parallel agents (3 completed fully, 5 hit API limits)

---

## 1. Scale of Failures

| Metric | Value |
|--------|:-----:|
| Total questions | 1050 |
| Score<=2 | 222 (21.1%) |
| Score=1 | 195 (18.6%) |
| Score=2 | 27 (2.6%) |
| Score>=4 | 741 (70.6%) |

### By Category (failures as % of category total)

| Category | Total | Score<=2 | Fail Rate |
|----------|:-----:|:--------:|:---------:|
| Spatial Understanding | 148 | 51 | **34.5%** |
| Object Recognition | 153 | 38 | 24.8% |
| World Knowledge | 136 | 34 | 25.0% |
| Object Localization | 159 | 28 | 17.6% |
| Attribute Recognition | 154 | 26 | 16.9% |
| Object State Recognition | 157 | 26 | 16.6% |
| Functional Reasoning | 143 | 19 | **13.3%** |

Spatial Understanding has 2x+ the failure rate of Functional Reasoning.

---

## 2. Root Cause Distribution (96 cases sampled)

| Root Cause | Count | % | Addressable? |
|------------|:-----:|:-:|:------------:|
| **VLM Perception Error** | 34 | 35% | Partially (crops, resolution) |
| **Retrieval Failure** | 23 | 24% | Yes (better keyframe selection) |
| **Visibility/Resolution Limit** | 12 | 12% | Partially (higher-res crops, OCR) |
| **Spatial Reasoning** | 8 | 8% | Yes (BEV, multi-view) |
| **State Recognition** | 7 | 7% | Hard (inherent from still images) |
| **World Knowledge** | 5 | 5% | Hard (needs external knowledge) |
| **GT/Label Issues** | 5 | 5% | Not fixable |
| **Missing from Scene Graph** | 4 | 4% | Yes (better detection) |

### Key Insight: GT Object Already in Enrichment

| In Enrichment? | Count | % |
|:-:|:---:|:---:|
| YES | 49 | 51% |
| NO | 34 | 35% |
| PARTIAL | 12 | 12% |

**51% of failures have the answer object already in the enrichment data** — the pipeline has the information but fails to use it correctly. This is primarily a VLM reasoning problem, not a data problem.

---

## 3. Top Failure Patterns

### Pattern 1: VLM Misidentification (35% of failures)

The VLM sees the correct region but identifies the wrong object:
- Toothbrush → "water bottle" (small object confusion)
- Red cooler → "boots" (shape confusion in dark scene)
- Soccer banner → "curtains" (overlapping objects)
- Golden trophy → "beverage can" (metallic object confusion)
- Paper bag on shelf → "empty shelf" (overconfident, 0 tools)

**Root cause**: VLM defaults to the most visually prominent or prototypical object rather than carefully examining the image.

### Pattern 2: Retrieval Misses (24% of failures)

Stage 1 selects keyframes that don't show the answer:
- Couch/sofa not in scene graph → proxy grounding misses it entirely (scene 012)
- Red chairs exist in 80% of trajectory but keyframes from first 10% show only black chairs (scene 108)
- Patio objects visible through windows but retrieval targets indoor floor (scene 047)
- "Between X and Y" queries retrieve X and Y but not the space between

**Root cause**: Keyframe selection biased by CLIP similarity to proxy terms, not spatial coverage.

### Pattern 3: State/Affordance Judgment (7% of failures)

Still-image state recognition is inherently unreliable:
- Fan "on" vs "off" — impossible from a still image
- Shower "wet" vs "dry" — requires texture analysis beyond current VLM
- Drawers "full" — drawers are closed, contents not visible
- Curtains "drawn" vs "pushed aside" — VLM conflates "present" with "drawn"

**Root cause**: Many state questions are fundamentally unanswerable from still images.

### Pattern 4: Small Object / Text Reading (12%)

Objects below detection/resolution threshold:
- Room numbers, brand logos, sign text — too small to read
- Scissors, stapler, tape, glasses case — too small for scene graph
- Toothbrush, calculator, power strip — below enrichment detection

**Root cause**: ScanNet images at 640x480 resolution, combined with object detection thresholds, miss small objects.

### Pattern 5: World Knowledge Gap (5%)

Questions requiring external knowledge:
- "What company made the laptop?" — needs brand logo reading
- Window type "cantilever vs casement" — architectural domain knowledge
- "How to find a book in library?" — needs to know about catalog computers

**Root cause**: VLM lacks domain-specific knowledge or fails to connect visual evidence to world knowledge.

---

## 4. Improvement Opportunities

### Tier 1: High Impact, Addressable (targets ~30% of failures)

| Fix | Target | Expected Impact |
|-----|--------|:---------------:|
| **Force crops before answering when conf > 0.8 but 0 tools used** | Overconfident perception errors | ~15 cases |
| **Inject enrichment descriptions into Stage 2 prompt** | Cases where GT is in enrichment but VLM misidentifies | ~25 cases |
| **Wider keyframe diversity** — enforce trajectory coverage | Retrieval clustering failures | ~10 cases |

### Tier 2: Medium Impact (targets ~15% of failures)

| Fix | Target | Expected Impact |
|-----|--------|:---------------:|
| **BEV-guided spatial reasoning** for "left/right/between" queries | Spatial reasoning failures | ~8 cases |
| **Higher-resolution crop pipeline** (2x zoom) for small objects | Text reading, small object ID | ~12 cases |
| **Multi-view consistency check** — compare answers across views | Contradictory answers | ~5 cases |

### Tier 3: Hard / Diminishing Returns

| Fix | Target | Notes |
|-----|--------|-------|
| State recognition from video (not still) | State failures (7%) | Requires video input, architectural change |
| External knowledge grounding | World knowledge (5%) | Needs knowledge base integration |
| Better upstream detection (SAM v2) | Missing objects (4%) | Expensive reprocessing of all scenes |
| GT correction | GT issues (5%) | Not actionable from our side |

---

## 5. Ceiling Analysis

| Category | Current MNAS | Fixable Gap | Estimated Ceiling |
|----------|:-----------:|:----------:|:-----------------:|
| Object State | 82.8 | ~5 (state from stills is hard) | ~88 |
| Localization | 79.5 | ~5 (retrieval fixes) | ~85 |
| Attribute | 74.0 | ~8 (color perception + crops) | ~82 |
| Functional | 74.3 | ~3 (mostly good) | ~77 |
| World Knowledge | 65.3 | ~5 (knowledge gap is hard) | ~70 |
| Object Rec | 64.2 | ~10 (enrichment injection + crops) | ~74 |
| Spatial | 57.8 | ~10 (BEV + multi-view) | ~68 |
| **Overall** | **71.4** | **~7** | **~78** |

With Tier 1 + Tier 2 fixes, estimated ceiling is **MNAS ~78** (vs current 71.4, vs Human 87.7).
