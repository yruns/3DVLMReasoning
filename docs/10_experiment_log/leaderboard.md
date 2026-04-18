# OpenEQA EM-EQA Leaderboard

**Metric:** LLM-Match / MNAS (0-100)
**Scope:** Only entries from published papers citing OpenEQA (CVPR 2024)
**Exclusions:** AlanaVLM (paper authenticity disputed)

## Important Notes

1. **Our judge is Gemini 2.5 Pro**; all other entries use **GPT-4** as judge (per OpenEQA paper). Scores not directly comparable.
2. Some methods only report ALL (ScanNet+HM3D combined), not ScanNet-specific scores.
3. `*` = arXiv revision of OpenEQA paper, not original CVPR Table 2.

---

## ScanNet EM-EQA Split

Only methods reporting ScanNet-specific scores:

| # | Method | ScanNet MNAS | Paper | Venue | Judge |
|---|--------|:-----------:|-------|-------|:-----:|
| - | Human | 87.7 | OpenEQA | CVPR 2024 | - |
| **1** | **Ours (v14)** | **73.1** | **This work** | **-** | **Gemini 2.5 Pro** |
| 2 | GPT-4V (500Q subset) | 51.3 | OpenEQA | CVPR 2024 | GPT-4 |
| 3 | GPT-4 + LLaVA-1.5 | 45.4 | OpenEQA | CVPR 2024 | GPT-4 |
| 4 | GPT-4 + Sparse Voxel Maps | 40.9 | OpenEQA | CVPR 2024 | GPT-4 |
| 5 | LLaMA-2 + LLaVA-1.5 | 39.6 | OpenEQA | CVPR 2024 | GPT-4 |
| 6 | GPT-4 + ConceptGraphs | 37.8 | OpenEQA | CVPR 2024 | GPT-4 |
| 7 | LLaMA-2 + Sparse Voxel Maps | 36.0 | OpenEQA | CVPR 2024 | GPT-4 |
| 8 | GPT-4 (blind) | 32.5 | OpenEQA | CVPR 2024 | GPT-4 |
| 9 | LLaMA-2 + ConceptGraphs | 31.0 | OpenEQA | CVPR 2024 | GPT-4 |
| 10 | LLaMA-2 (blind) | 27.9 | OpenEQA | CVPR 2024 | GPT-4 |

---

## ALL (ScanNet + HM3D) Combined

Methods reporting combined scores (not directly comparable to ScanNet-only):

| # | Method | ALL MNAS | Paper | Venue | Judge | Notes |
|---|--------|:-------:|-------|-------|:-----:|-------|
| - | Human | 86.8 | OpenEQA | CVPR 2024 | - | |
| 1 | GLM-4.6V + CoV | 67.7 | CoV | arXiv 2026 | GPT-4 (assumed) | No ScanNet breakdown |
| 2 | GLM-4.6V (baseline) | 62.4 | CoV | arXiv 2026 | GPT-4 (assumed) | |
| 3 | Qwen3-VL-Flash + CoV | 59.8 | CoV | arXiv 2026 | GPT-4 (assumed) | |
| 4 | Gemini 2.5 Flash + CoV | 59.2 | CoV | arXiv 2026 | GPT-4 (assumed) | |
| 5 | 3D-Mem (GPT-4V) | 57.2 | 3D-Mem | CVPR 2025 | GPT-4 | No ScanNet breakdown |
| 6 | GraphPad (Gemini 2.0 Flash) | 55.3 | GraphPad | arXiv 2025 | GPT-4 | No ScanNet breakdown |
| 7 | Qwen3-VL-Flash (baseline) | 52.7 | CoV | arXiv 2026 | GPT-4 (assumed) | |
| 8 | Gemini 2.0 Flash (25f) | 52.3 | GraphPad | arXiv 2025 | GPT-4 | |
| 9 | GPT-4o-Mini + CoV | 51.6 | CoV | arXiv 2026 | GPT-4 (assumed) | |
| 10 | GPT-4V (ALL, 500Q) | 49.6 | OpenEQA | CVPR 2024 | GPT-4 | |
| 11 | R-EQA + Qwen2.5-VL | 46.0 | R-EQA | Workshop 2025 | GPT-4 | ScanNet: 49.1 |
| 12 | GPT-4 + LLaVA-1.5 | 43.6 | OpenEQA | CVPR 2024 | GPT-4 | |
| 13 | GPT-4 + Sparse Voxel Maps | 38.9 | OpenEQA | CVPR 2024 | GPT-4 | |
| 14 | GPT-4 + ConceptGraphs | 36.5 | OpenEQA | CVPR 2024 | GPT-4 | |
| 15 | GPT-4 (blind) | 33.5 | OpenEQA | CVPR 2024 | GPT-4 | |

---

## Paper References

| Short Name | Full Title | arXiv/Venue |
|------------|-----------|-------------|
| OpenEQA | OpenEQA: Embodied Question Answering in the Era of Foundation Models | CVPR 2024, arXiv:2312.15857 |
| CoV | Chain-of-View Prompting for Spatial Reasoning | arXiv:2601.05172 (preprint) |
| 3D-Mem | 3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning | CVPR 2025, arXiv:2411.17735 |
| GraphPad | GraphPad: Inference-Time 3D Scene Graph Updates for EQA | arXiv:2506.01174 (preprint) |
| R-EQA | R-EQA: Retrieval-Augmented Generation for Embodied Question Answering | Embodied AI Workshop 2025 |

---

## Our Position

- **ScanNet-only**: 73.1 MNAS — **#1 among methods reporting ScanNet scores** (excluding AlanaVLM)
- **vs OpenEQA paper baselines**: +21.8 over GPT-4V (51.3), +35.3 over GPT-4+ConceptGraphs (37.8)
- **Not comparable to CoV/3D-Mem/GraphPad**: They report ALL, we report ScanNet-only
- **Judge caveat**: Our Gemini 2.5 Pro judge vs others' GPT-4 judge

## To-Do for Fair Comparison

1. Re-evaluate with GPT-4 as judge (same protocol as all baselines)
2. Run on HM3D scenes to produce ALL (ScanNet+HM3D) combined score
