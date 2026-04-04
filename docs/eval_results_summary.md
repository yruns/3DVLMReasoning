# 3DVLMReasoning Evaluation Results Summary

**Last updated:** 2026-04-04
**Branch:** `feat/v11-evidence-seeking-fixes`
**Benchmark:** OpenEQA ScanNet Episodic Memory (EM-EQA)
**Scoring:** LLM-Match (1-5 raw scale), reported as **MNAS** (0-100 normalized)
**MNAS formula:** `MNAS = (raw_mean - 1) / 4 * 100`
**Judge LLM:** Gemini 2.5 Pro (note: original paper uses GPT-4)

---

## 1. Version History (100-query subset: 20 scenes x 5 questions)

| Version | Raw Mean | **MNAS** | Score=1 | Score>=4 | Failed | Key Change |
|---------|:--------:|:--------:|:-------:|:--------:|:------:|------------|
| v9 | 2.860 | **46.5** | 44% | 42% | 0 | Baseline (no enrichment, no callbacks) |
| v10 | 3.214 | **55.4** | 34% | 50% | 2 | +LLM object enrichment |
| v11.1 | 3.505 | **62.6** | 29% | 61% | 3 | +Tool callbacks, min keyframes, open-ended mode |
| v12 | 3.602 | **65.0** | 22% | 62% | 2 | +Tool prompt rules, regression fix, between-midpoint, GPT-5.4 |
| v13 | 3.547 | **63.7** | 24% | 65% | 5 | +Mandatory crops, overconfidence calibration, bias correction |

### Cumulative Improvement (100-query)

| Transition | MNAS Delta | Relative |
|------------|:----------:|:--------:|
| v9 → v10 (enrichment) | +8.9 | +19.1% |
| v10 → v11.1 (callbacks+keyframes) | +7.2 | +13.0% |
| v11.1 → v12 (tool prompt+GPT-5.4) | +2.4 | +3.8% |
| **v9 → v12 (total)** | **+18.5** | **+39.8%** |

---

## 2. Full OpenEQA Evaluation (1050 questions)

### v14 (Latest) — Enrichment Inventory in System Prompt

| Metric | Value |
|--------|:-----:|
| **MNAS** | **73.1** |
| Raw E2E mean | 3.926 |
| Questions completed | 1050/1200 (87.5%) |
| Failed | 29 |
| Score=5 | 57.1% |
| Score>=4 | 71.8% |
| Score<=2 | 19.4% |
| Tool calls | 1401 (1.33/Q) |
| VLM backend | GPT-5.4-2026-03-05 |

### Per-Category MNAS (v14 full, 1050Q)

| Category | **v14 MNAS** | v13 MNAS | Delta | vs Human | vs GPT-4V |
|----------|:-----------:|:--------:|:-----:|:--------:|:---------:|
| Object State Recognition | **84.7** | 82.8 | +1.9 | -14.0 | +21.5 |
| Object Localization | **78.0** | 79.6 | -1.6 | +0.7 | +36.0 |
| Attribute Recognition | **77.5** | 74.0 | +3.4 | -10.4 | +20.3 |
| Functional Reasoning | **76.9** | 74.3 | +2.6 | -4.9 | +19.5 |
| World Knowledge | **67.2** | 65.3 | +1.9 | -20.0 | +16.5 |
| Object Recognition | **66.0** | 64.2 | +1.8 | -21.9 | +22.6 |
| Spatial Understanding | **60.4** | 57.8 | +2.6 | -26.3 | +26.8 |
| **Overall** | **73.1** | **71.4** | **+1.8** | **-14.6** | **+21.8** |

### v13 → v14 Improvement

| Metric | v13 | v14 | Delta |
|--------|:---:|:---:|:-----:|
| MNAS | 71.4 | **73.1** | **+1.8** |
| Score=5 | 573 | **600** | +27 |
| Score=1 | 195 | **178** | -17 |
| Score<=2 | 222 | **204** | -18 |

Key change: Injected enrichment object inventory (category + description) directly into Stage 2 system prompt. Previously agents had to call `retrieve_object_context()` tool; now they see all objects upfront. Biggest gains in Attribute Recognition (+3.4) and Spatial Understanding (+2.6).

---

## 3. Comparison with Published Baselines

### OpenEQA Paper (CVPR 2024) — ScanNet EM-EQA Split

| Rank | Method | MNAS | Source |
|:----:|--------|:----:|--------|
| - | Human | 87.7 | Paper |
| **1** | **Ours (v14)** | **73.1** | **This work** |
| 2 | Gemini 1.5 Flash | 72.5 | Community |
| 3 | Ours (v13) | 71.4 | This work |
| 4 | GLM-4.6V + Chain-of-View | 67.0 | CoV paper |
| 5 | CoV (Qwen3-VL) | 58.8 | CoV paper |
| 6 | GraphPad (Gemini 2.0 Flash) | 55.3 | GraphPad |
| 7 | GPT-4V (500Q subset) | 51.3 | Paper |
| 8 | GPT-4 + LLaVA-1.5 captions | 45.4 | Paper |
| 9 | GPT-4 + Sparse Voxel Maps | 40.9 | Paper |
| 10 | GPT-4 + ConceptGraphs | 37.8 | Paper |
| 11 | GPT-4 (blind, text-only) | 32.5 | Paper |

### Key Findings

1. **#1 on OpenEQA ScanNet** — MNAS 73.1, surpassing Gemini 1.5 Flash (72.5)
2. **+21.8 MNAS over GPT-4V** (73.1 vs 51.3) — the largest gap among all methods
3. **Object Localization (78.0) exceeds Human (77.3)** — evidence-seeking + 3D scene graph advantage
4. **Object State Recognition (84.7)** — approaching human (98.7), best non-human category
5. **Spatial Understanding remains weakest** (60.4 MNAS, -26.3 vs Human) but improved +2.6 from v13

### Caveats

- Judge LLM differs: paper uses GPT-4, we use Gemini 2.5 Pro (may cause systematic bias)
- Coverage: we evaluated 1050/1200 (87.5%), GPT-4V only 500 questions
- Some community baselines (Gemini 1.5 Flash 72.5) use different protocols

---

## 4. Architecture Summary

```
Question → [Stage 1: Query-Driven Keyframe Retrieval]
              ├── LLM query parser → structured hypothesis
              ├── ConceptGraph scene graph matching
              ├── LLM-enriched object labels (5097 objects, Gemini 2.5 Pro)
              ├── Spatial midpoint retrieval for "between" queries
              └── Min 3 keyframes with visibility-based padding
           → [Stage 2: VLM Reasoning with Evidence-Seeking]
              ├── GPT-5.4 VLM examines keyframes
              ├── Tool callbacks: request_more_views, request_crops
              ├── Scene object inventory injected into prompt (v14)
              ├── Mandatory crops for color/attribute questions
              ├── Self-check: answer must not contradict question premise
              └── Prominent-object bias warnings
           → [E2E: Extended reasoning if Stage 2 low-confidence]
              └── Additional tool-based evidence gathering
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| Object enrichment | `src/scripts/enrich_objects.py` | Gemini vision → category/description/color/location per object |
| Enrichment loading | `keyframe_selector.py:_load_enrichment()` | Populates SceneObject fields from enriched_objects.json |
| Min keyframes | `keyframe_selector.py:_pad_keyframes_to_minimum()` | Visibility-based padding to min 3 frames |
| Open-ended queries | `keyframe_selector.py` + `query_executor.py` | UNKNOW target → return all objects for spatial filtering |
| Between midpoint | `keyframe_selector.py` | Geometric centroid between anchors for view selection |
| Scene inventory | `runtime/base.py:_format_scene_inventory()` | Injects enrichment category+description into system prompt |
| Tool prompt rules | `runtime/base.py:build_system_prompt()` | Mandatory crops, self-check, bias correction |
| Scoring retry | `llm_client.py:invoke_with_full_retry()` | Multi-round retry with key rotation |
| Concurrent scoring | `openeqa_official_eval.py` | 12-worker parallel scoring with resume |

---

## 5. Remaining Failure Modes (from v13 full analysis, 96 cases sampled)

| Root Cause | % of failures | Mitigation Status |
|------------|:-------------:|:-----------------:|
| VLM Perception Error | 35% | Partially mitigated by enrichment injection (v14) |
| Retrieval Failure | 24% | Addressed by midpoint + min keyframes |
| Visibility/Resolution | 12% | Needs higher-res crops or OCR |
| Spatial Reasoning | 8% | Improved +2.6 MNAS in v14 |
| State Recognition | 7% | Hard (inherent from still images) |
| World Knowledge | 5% | Hard (needs external knowledge) |
| GT/Label Issues | 5% | Not fixable |
| Missing from Scene Graph | 4% | Needs better upstream detection |

### Top Improvement Opportunities

1. **Spatial understanding** (MNAS 60.4, -26.3 vs Human) — biggest gap, needs BEV + multi-view reasoning
2. **Object recognition** (MNAS 66.0, -21.9 vs Human) — small object detection pipeline gap
3. **World knowledge** (MNAS 67.2, -20.0 vs Human) — needs external knowledge grounding

### Estimated Ceiling

With all addressable fixes (Tier 1 + Tier 2): **MNAS ~78** (vs current 73.1, vs Human 87.7).
Remaining ~10 MNAS gap is from: state recognition from stills, world knowledge, GT quality.

---

## 6. Full Version Progression (1050Q)

| Version | MNAS | Score<=2 | Score>=4 | Key Change |
|---------|:----:|:--------:|:--------:|------------|
| v13 | 71.4 | 222 (21.1%) | 741 (70.6%) | Mandatory crops, overconfidence calibration |
| **v14** | **73.1** | **204 (19.4%)** | **754 (71.8%)** | **+Enrichment inventory in system prompt** |
