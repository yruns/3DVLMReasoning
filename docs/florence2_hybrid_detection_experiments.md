# Florence-2 Hybrid Detection: Experiment Log

**Date**: 2026-03-25
**Goal**: Replace RAM+GroundingDINO with Florence-2 for scene graph construction, achieving comparable or better recall with zero junk labels.

## Background

The original ConceptGraph pipeline uses RAM (tagging) + GroundingDINO (grounding) + SAM (segmentation). RAM generates image-level tags, GDINO grounds each tag to bounding boxes, and SAM refines boxes into masks.

RAM+GDINO produces junk labels like `green`, `lead to`, `other item` (3.1% junk rate). The goal is to replace it with Florence-2, a unified vision-language model.

## Experiment 1: Florence-2 `<OD>` (Object Detection only)

**Approach**: Use Florence-2's `<OD>` task for open-vocabulary detection, then SAM for masks.

**Results** (5 scenes, stride=5):

| Metric | RAM+GDINO (v1) | Florence-2 OD |
|--------|----------------|---------------|
| Objects | 229 | 120 |
| Categories | 87 (3 junk) | 52 (0 junk) |
| Junk rate | 3.1% | 0% |

**Assessment**: Zero junk, but recall is catastrophically low — only 52% of v1's object count. Scene 014 detected only 7 objects (vs 51 v1). Florence-2's `<OD>` head is too conservative; it only finds the most salient objects.

## Experiment 2: `<DENSE_REGION_CAPTION>` mode

**Approach**: Use `<DENSE_REGION_CAPTION>` instead of `<OD>`, hoping richer captioning yields more detections.

**Results**: Returns identical bounding boxes as `<OD>` — just adds richer text descriptions. No recall improvement. The detection head is the same; only the output text format differs.

**Also tested**: `<REGION_PROPOSAL>` — same boxes, no labels. All three tasks share the same underlying detection mechanism.

## Experiment 3: Caption-then-ground approach

**Approach**: Use `<MORE_DETAILED_CAPTION>` to generate a description, then `<CAPTION_TO_PHRASE_GROUNDING>` to ground phrases from the caption.

**Results on scene 014 frame 0**:
- OD: 3 detections
- Caption grounding: 6 detections (improvement, but still low)
- Caption quality was decent but noun extraction was incomplete

**Assessment**: Marginal improvement. The caption doesn't mention every object in the scene.

## Experiment 4: Vocabulary grounding (BREAKTHROUGH)

**Approach**: Provide a broad indoor vocabulary list directly to `<CAPTION_TO_PHRASE_GROUNDING>`. Florence-2 grounds each term against the image.

**Key discovery**: A single long vocabulary string causes Florence-2 to concatenate tail terms into one detection. Solution: **batch the vocabulary into groups of ~10 terms**.

**Results on scene 014 frame 0**:
- OD: 3 detections
- Vocab grounding (single batch): 16 detections (5.3x improvement!)
- Observations: Duplicate detections exist (sofa/couch on same box), needs NMS

## Experiment 5: Hybrid pipeline (OD + batched vocab grounding + NMS) — FINAL

**Approach**: Two-pass detection:
1. `<OD>` for clean open-vocabulary detection (high precision)
2. `<CAPTION_TO_PHRASE_GROUNDING>` with 9 batched indoor vocabulary lists
3. IoU-based NMS (threshold=0.7) to merge duplicates, preferring smaller/more specific boxes

**Vocabulary batches** (9 groups covering ~90 indoor object categories):
- Seating/surfaces, Storage, Electronics, Lighting/decor, Fabrics
- Kitchen/bath, Small objects, Structural, Misc

**Full pipeline results** (5 scenes, stride=5, SAM masks):

| Metric | RAM+GDINO (v1) | Florence-2 OD-only | Florence-2 Hybrid |
|--------|----------------|---------------------|-------------------|
| Total objects | 229 | 120 | **314** |
| Unique categories | 87 | 52 | **72** |
| Junk labels | 3 (green, lead to, other item) | 0 | **1** (light) |
| Junk object count | 7 (3.1%) | 0 (0%) | **1 (0.3%)** |

### Per-scene breakdown

| Scene | v1 objects | OD-only | Hybrid | v1 cats | Hybrid cats |
|-------|-----------|---------|--------|---------|-------------|
| 002 (kitchen) | 49 | 29 | **73** | 23 | 30 |
| 003 (office) | 63 | 41 | **90** | 28 | 53 |
| 012 (living) | 37 | 29 | **70** | 18 | 34 |
| 013 (bedroom) | 29 | 14 | **44** | 17 | 28 |
| 014 (living/open) | 51 | 7 | 37 | 19 | 22 |

### Scene 014 quality analysis

Scene 014 has fewer hybrid objects than v1 (37 vs 51), but v1's count is inflated by repetitive/noisy detections:
- v1: `footrest ×7`, `screen door ×7`, `balustrade ×5` — same objects re-detected
- v2: `window ×5`, `cushion ×3`, `armchair ×2` — more diverse and meaningful

v2 has 22 unique categories vs v1's 19, despite fewer total objects.

### Known issues (resolved in Experiment 6)

~~1. **`pen` over-detection** (18/314 = 5.7%)~~ → Replaced with `remote control` in vocab
~~2. **`column` over-detection** (15/314 = 4.8%)~~ → Replaced with `pillar` in vocab

3. **Processing speed**: Hybrid mode runs at ~4s/frame (vs ~1s for OD-only) due to 9 additional grounding passes per frame. Total for 5 scenes: ~40 min (stride=5, 120 frames/scene).

## Experiment 6: Vocabulary refinement (FINAL)

**Changes**: `pen` → `remote control`, `column` → `pillar` in vocabulary batches.

**Rationale**: "pen" triggered false positives on small elongated shapes in non-office scenes. "remote control" is more distinctive. "pillar" is more specific than "column" and matches v1's terminology.

**Final results** (5 scenes, stride=5):

| Metric | RAM+GDINO (v1) | Florence-2 Hybrid |
|--------|----------------|-------------------|
| Total objects | 229 | **312** (+36%) |
| Unique categories | 87 | **77** |
| Junk labels | 3 | **1** (`light`) |
| Junk rate | 3.1% | **0.3%** |

Per-scene:

| Scene | v1 objects | v2 hybrid | v1 cats | v2 cats |
|-------|-----------|-----------|---------|---------|
| 002 (kitchen) | 49 | **72** | 23 | **32** |
| 003 (office) | 63 | **90** | 28 | **54** |
| 012 (living) | 37 | **72** | 18 | **34** |
| 013 (bedroom) | 29 | **39** | 17 | **28** |
| 014 (open living) | 51 | 39 | 19 | **23** |

Scene 014 (39 vs 51) has fewer objects but higher quality: v1 had repetitive noisy labels (`footrest ×7`, `screen door ×7`), v2 has diverse meaningful categories (23 unique vs 19).

Top-20 categories are now clean:
```
window ×23, curtain ×14, outlet ×13, bottle ×13, screen ×10,
shoe ×10, picture frame ×10, bowl ×9, countertop ×9, pillar ×9,
drawer ×8, phone ×8, carpet ×7, speaker ×7, desk ×7, cushion ×7,
keyboard ×6, towel ×6, book ×5, cabinetry ×5
```

## Technical fixes applied

1. **SAM import**: Made mandatory — removed try/except fallback that silently degraded to rectangle masks
2. **Python path**: `.venv/bin/python` was shadowing conda's python, causing torch import to fail. Now using explicit `/home/ysh/miniconda3/envs/conceptgraph/bin/python`
3. **to_scalar()**: Added `np.generic` support for numpy scalar types
4. **pipeline.py**: Fixed `BG_CLASSES` → `bg_classes` kwarg mismatch
5. **Hydra config**: Created missing `conceptgraph/configs/slam_pipeline/base.yaml`
6. **datasets_common.py**: Created missing module needed by SLAM pipeline

## Future improvements (if needed)

1. **Scene-adaptive vocabulary**: Extract nouns from `<MORE_DETAILED_CAPTION>` per-image and add to grounding vocabulary
2. **Confidence scoring**: Use CLIP image-text similarity as proxy confidence for grounding detections
3. **Tiled detection**: Run detection on image crops/tiles for small object recall
4. **Florence-2-large vs base**: Compare model sizes for precision/recall tradeoff

## Conclusion

The hybrid Florence-2 pipeline (OD + vocabulary grounding + NMS) achieves **36% more objects** (312 vs 229) than RAM+GDINO while reducing junk from 3.1% to 0.3%. The key insight: Florence-2's `<OD>` head alone is too conservative, but `<CAPTION_TO_PHRASE_GROUNDING>` with a broad indoor vocabulary dramatically increases recall. Batching the vocabulary into groups of ~10 avoids Florence-2's token-limit concatenation bugs. Vocabulary refinement (replacing `pen`/`column` with more specific terms) eliminated systematic false positives.
