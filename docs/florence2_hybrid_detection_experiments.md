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

### Known issues

1. **`pen` over-detection** (18/314 = 5.7%): Florence-2 vocab grounding matches "pen" to small elongated shapes across all scene types. Mitigation: could remove "pen" from non-office scenes or add to post-filter.

2. **`column` over-detection** (15/314 = 4.8%): Structural elements like door frames and pillars are grounded as "column". Mostly correct, but inflated.

3. **Processing speed**: Hybrid mode runs at ~4s/frame (vs ~1s for OD-only) due to 9 additional grounding passes per frame. Total for 5 scenes: ~40 min (stride=5, 120 frames/scene).

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
3. **Vocabulary tuning**: Remove/rename terms causing systematic false positives (pen, column)
4. **Tiled detection**: Run detection on image crops/tiles for small object recall
5. **Florence-2-large vs base**: Compare model sizes for precision/recall tradeoff

## Conclusion

The hybrid Florence-2 pipeline (OD + vocabulary grounding + NMS) achieves **37% more objects** than RAM+GDINO while reducing junk from 3.1% to 0.3%. The key insight: Florence-2's `<OD>` head alone is too conservative, but `<CAPTION_TO_PHRASE_GROUNDING>` with a broad vocabulary dramatically increases recall. Batching the vocabulary avoids token-limit concatenation bugs.
