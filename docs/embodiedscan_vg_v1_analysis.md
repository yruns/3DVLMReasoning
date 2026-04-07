# EmbodiedScan VG v1 Evaluation Analysis

**Date:** 2026-04-07
**Branch:** `feat/embodiedscan-grounding`
**Model:** gpt-5.4-2026-03-05 (Stage 2), gemini-2.5-pro (Stage 1 parsing)

## Evaluation Setup

- **Dataset:** EmbodiedScan val split, ScanNet source only
- **Sampling:** Diverse — 2 samples/scene across 40 scenes (80 total)
- **Pipeline:** Stage 1 keyframe selection (k=5) → Stage 2 VLM grounding
- **Metric:** 9-DOF oriented 3D IoU (Acc@0.25, Acc@0.50, Mean IoU)

## Results

| Metric | Value |
|--------|-------|
| Acc@0.25 | **6.2%** (5/80) |
| Acc@0.50 | **0.0%** (0/80) |
| Mean IoU | **0.027** |
| Samples with bbox prediction | 56/80 (70%) |
| Mean centroid distance | 1.32m |

## Failure Breakdown

Of 75 failures (IoU < 0.25):

| Category | Count | % Total | % Failures | Description |
|----------|-------|---------|------------|-------------|
| **(b) Wrong object** | 44 | 55.0% | 58.7% | VLM selected an object but wrong instance |
| **(a) No prediction** | 24 | 30.0% | 32.0% | VLM returned no bbox (insufficient evidence) |
| **(c) Category missing** | 7 | 8.8% | 9.3% | GT category not in ConceptGraph scene graph |

### Centroid Distance (56 samples with predictions)

| Distance | Count | Percentage |
|----------|-------|------------|
| < 0.5m | 13 | 23.2% |
| < 1.0m | 28 | 50.0% |
| < 2.0m | 46 | 82.1% |
| < 3.0m | 52 | 92.9% |

50% of predictions are within 1m of the GT center, indicating the VLM often picks an object in the right area but the wrong instance.

## Bugs Fixed During Evaluation

1. **Coordinate system mismatch** (critical) — ConceptGraph centroids were in ScanNet raw frame, not EmbodiedScan aligned frame. Applied `axis_align_matrix` transform.
2. **bbox_3d type safety** — VLM output could be string/None/incomplete; added robust parsing in `_parse_bbox_3d`.
3. **Image path resolution** — `_set_image_paths` now falls back to `raw/*-rgb.jpg` for EmbodiedScan scenes.
4. **Stride mismatch** — Visibility index built with stride=1, selector used stride=5; fixed to stride=1.
5. **VG candidates not in prompt** — `_format_vg_candidates` was defined but never called in `build_user_message`.
6. **Empty candidate list** — `build_vg_candidates` required `bbox_extent` which ConceptGraph objects lack; now uses default 0.3 size.

## Top 3 Improvement Directions

### 1. Real 3D bbox extents (highest impact)

All predicted bboxes use default 0.3x0.3x0.3 size. With 50% of centroids within 1m of GT, using real object extents would significantly increase IoU overlap.

**Action:** Compute per-object extent from ConceptGraph point clouds (stored in PCD saves) or from EmbodiedScan instance annotations. The PKL metadata contains instance-level bboxes that could be used as candidate sizes.

### 2. Spatial reasoning for disambiguation (55% of failures)

The VLM often picks the right category but wrong instance (e.g., selects wrong curtain when query says "closer to desk"). The VLM needs explicit spatial verification.

**Action:** Pre-compute pairwise distances between candidates and anchor objects mentioned in the query. Include these distances in the VG candidates list so the VLM can directly compare. Alternatively, add a post-selection verification step that checks spatial constraints.

### 3. Reduce "no prediction" rate (30% of failures)

The VLM reports `insufficient_evidence` too often due to: (a) target category not recognized in ConceptGraph labels (e.g., "bag" → "wastebasket"), (b) VLM cannot visually confirm objects from keyframes.

**Action:** Add a category synonym mapping (EmbodiedScan vocabulary → ConceptGraph vocabulary). Force VG tasks to always output a best-guess prediction even at low confidence, since a wrong prediction is more useful than no prediction for evaluation.
