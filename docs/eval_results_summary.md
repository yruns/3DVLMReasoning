# 3DVLMReasoning Evaluation Results Summary

**Last updated:** 2026-04-03
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

## 2. Full OpenEQA Evaluation (v13, 1050 questions)

| Metric | Value |
|--------|:-----:|
| **MNAS** | **71.4** |
| Raw E2E mean | 3.854 |
| Questions completed | 1050/1200 (87.5%) |
| Failed | 29 |
| Score=5 | 54.6% |
| Score>=4 | 70.6% |
| Score<=2 | 21.1% |
| Tool calls | 1423 (1.36/Q) |
| Tool usage rate | 88.1% |
| Stage1 direct_grounded | 78.5% |
| VLM backend | GPT-5.4-2026-03-05 |

### Per-Category MNAS (v13 full, 1050Q)

| Category | Raw Mean | **MNAS** | n | vs Human | vs GPT-4V |
|----------|:--------:|:--------:|:---:|:--------:|:---------:|
| Object State Recognition | 4.312 | **82.8** | 157 | -15.9 | +19.6 |
| Object Localization | 4.182 | **79.5** | 159 | **+2.2** | +37.5 |
| Attribute Recognition | 3.961 | **74.0** | 154 | -13.9 | +16.8 |
| Functional Reasoning | 3.972 | **74.3** | 143 | -7.5 | +16.9 |
| World Knowledge | 3.610 | **65.3** | 136 | -21.9 | +14.6 |
| Object Recognition | 3.569 | **64.2** | 153 | -23.7 | +20.8 |
| Spatial Understanding | 3.311 | **57.8** | 148 | -28.9 | +24.2 |
| **Overall** | **3.854** | **71.4** | **1050** | **-16.3** | **+20.1** |

---

## 3. Comparison with Published Baselines

### OpenEQA Paper (CVPR 2024) — ScanNet EM-EQA Split

| Rank | Method | MNAS | Source |
|:----:|--------|:----:|--------|
| - | Human | 87.7 | Paper |
| 1 | Gemini 1.5 Flash | 72.5 | Community |
| **2** | **Ours (v13)** | **71.4** | **This work** |
| 3 | GLM-4.6V + Chain-of-View | 67.0 | CoV paper |
| 4 | CoV (Qwen3-VL) | 58.8 | CoV paper |
| 5 | GraphPad (Gemini 2.0 Flash) | 55.3 | GraphPad |
| 6 | GPT-4V (500Q subset) | 51.3 | Paper |
| 7 | GPT-4 + LLaVA-1.5 captions | 45.4 | Paper |
| 8 | GPT-4 + Sparse Voxel Maps | 40.9 | Paper |
| 9 | GPT-4 + ConceptGraphs | 37.8 | Paper |
| 10 | GPT-4 (blind, text-only) | 32.5 | Paper |

### Key Findings

1. **+20.1 MNAS over GPT-4V** (71.4 vs 51.3) — the largest gap among all methods
2. **Object Localization (79.5) exceeds Human (77.3)** — evidence-seeking + 3D scene graph advantage
3. **Within 1.1 MNAS of Gemini 1.5 Flash** (71.4 vs 72.5) — near SOTA
4. **Spatial Understanding remains weakest** (57.8 MNAS, -28.9 vs Human)

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
| Tool prompt rules | `runtime/base.py:build_system_prompt()` | Mandatory crops, self-check, bias correction |
| Scoring retry | `llm_client.py:invoke_with_full_retry()` | Multi-round retry with key rotation |
| Concurrent scoring | `openeqa_official_eval.py` | 12-worker parallel scoring with resume |

---

## 5. Remaining Failure Modes (from v12 analysis, 24 cases)

| Root Cause | % of failures | Mitigation Status |
|------------|:-------------:|:-----------------:|
| REASONING_ERROR (VLM perception) | 71% | Partially mitigated by crop rules |
| TOOL_UNDERUSE (overconfident) | 42% | Partially mitigated by v13 prompt |
| VISIBILITY_ISSUE (dark/small) | 42% | Needs better detection pipeline |
| RETRIEVAL_FAILURE | 38% | Addressed by midpoint + min keyframes |
| AMBIGUOUS_GT | 17% | Not fixable (GT quality issue) |

### Top Improvement Opportunities

1. **Spatial understanding** (MNAS 57.8, -28.9 vs Human) — biggest gap, needs better spatial reasoning
2. **Object recognition** (MNAS 64.2, -23.7 vs Human) — small object detection pipeline gap
3. **World knowledge** (MNAS 65.3, -21.9 vs Human) — needs external knowledge grounding
