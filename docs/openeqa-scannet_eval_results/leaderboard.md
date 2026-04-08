# OpenEQA EM-EQA Leaderboard

**Metric:** LLM-Match / MNAS (0-100)
**All entries from published papers citing OpenEQA (CVPR 2024)**

## Important Notes

1. **Our judge is Gemini 2.5 Pro**, all other entries use **GPT-4** as judge (per OpenEQA paper). Scores are NOT directly comparable due to judge difference.
2. Scores marked with `*` are from arXiv revision of OpenEQA, not the original CVPR Table 2.
3. "ALL" = ScanNet + HM3D combined. Some methods only report ALL.

---

## ScanNet EM-EQA Split

| # | Method | ScanNet | Paper | Venue | Judge |
|---|--------|:-------:|-------|-------|:-----:|
| - | Human | 87.7 | OpenEQA | CVPR 2024 | - |
| 1 | Gemini 1.5 Flash (50f) | **74.0** | AlanaVLM | arXiv 2024 | GPT-4 |
| **2** | **Ours (v14)** | **73.1** | **This work** | **-** | **Gemini 2.5 Pro** |
| 3 | Gemini 1.5 Pro (50f) | 66.9 | AlanaVLM | arXiv 2024 | GPT-4 |
| 4 | GPT-4V (50f, full eval)* | 57.4 | AlanaVLM | arXiv 2024 | GPT-4 |
| 5 | GPT-4V (50f, 500Q subset) | 51.3 | OpenEQA | CVPR 2024 | GPT-4 |
| 6 | R-EQA w/ Qwen2.5-VL | 49.1 | R-EQA | Workshop 2025 | GPT-4 |
| 7 | AlanaVLM (7B, 50f) | 47.8 | AlanaVLM | arXiv 2024 | GPT-4 |
| 8 | GPT-4 + LLaVA-1.5 | 45.4 | OpenEQA | CVPR 2024 | GPT-4 |
| 9 | Chat-UniVi (50f) | 43.4 | AlanaVLM | arXiv 2024 | GPT-4 |
| 10 | GPT-4 + Sparse Voxel Maps | 40.9 | OpenEQA | CVPR 2024 | GPT-4 |
| 11 | LLaMA-2 + LLaVA-1.5 | 39.6 | OpenEQA | CVPR 2024 | GPT-4 |
| 12 | GPT-4 + ConceptGraphs | 37.8 | OpenEQA | CVPR 2024 | GPT-4 |
| 13 | LLaMA-2 + Sparse Voxel Maps | 36.0 | OpenEQA | CVPR 2024 | GPT-4 |
| 14 | GPT-4 (blind) | 32.5 | OpenEQA | CVPR 2024 | GPT-4 |
| 15 | LLaMA-2 + ConceptGraphs | 31.0 | OpenEQA | CVPR 2024 | GPT-4 |
| 16 | LLaMA-2 (blind) | 27.9 | OpenEQA | CVPR 2024 | GPT-4 |

---

## ALL (ScanNet + HM3D) Combined

| # | Method | ALL | ScanNet | HM3D | Paper | Venue |
|---|--------|:---:|:-------:|:----:|-------|-------|
| - | Human | 86.8 | 87.7 | 85.1 | OpenEQA | CVPR 2024 |
| 1 | Gemini 1.5 Flash (50f) | 72.5 | 74.0 | 69.7 | AlanaVLM | arXiv 2024 |
| 2 | GLM-4.6V + CoV | 67.7 | - | - | CoV | arXiv 2026 |
| 3 | Gemini 1.5 Pro (50f) | 64.9 | 66.9 | 61.0 | AlanaVLM | arXiv 2024 |
| 4 | GLM-4.6V (baseline) | 62.4 | - | - | CoV | arXiv 2026 |
| 5 | Qwen3-VL-Flash + CoV | 59.8 | - | - | CoV | arXiv 2026 |
| 6 | Gemini 2.5 Flash + CoV | 59.2 | - | - | CoV | arXiv 2026 |
| 7 | 3D-Mem (GPT-4V) | 57.2 | - | - | 3D-Mem | CVPR 2025 |
| 8 | GPT-4V (full eval)* | 55.3 | 57.4 | 51.3 | AlanaVLM | arXiv 2024 |
| 9 | GraphPad (Gemini 2.0 Flash) | 55.3 | - | - | GraphPad | arXiv 2025 |
| 10 | Qwen3-VL-Flash (baseline) | 52.7 | - | - | CoV | arXiv 2026 |
| 11 | Gemini 2.0 Flash (25f) | 52.3 | - | - | GraphPad | arXiv 2025 |
| 12 | GPT-4o-Mini + CoV | 51.6 | - | - | CoV | arXiv 2026 |
| 13 | GPT-4V (500Q subset) | 49.6 | 51.3 | 46.6 | OpenEQA | CVPR 2024 |
| 14 | AlanaVLM (7B, 50f) | 46.7 | 47.8 | 44.8 | AlanaVLM | arXiv 2024 |
| 15 | R-EQA w/ Qwen2.5-VL | 46.0 | 49.1 | 42.8 | R-EQA | Workshop 2025 |
| 16 | GPT-4o-Mini (baseline) | 45.9 | - | - | CoV | arXiv 2026 |
| 17 | Gemini 1.0 Pro Vision (15f) | 44.9 | - | - | AlanaVLM | arXiv 2024 |
| 18 | GPT-4 + LLaVA-1.5 | 43.6 | 45.4 | 40.0 | OpenEQA | CVPR 2024 |
| 19 | Chat-UniVi (50f) | 42.3 | 43.4 | 40.4 | AlanaVLM | arXiv 2024 |
| 20 | R-EQA w/ Ferret | 40.4 | 42.7 | 38.3 | R-EQA | Workshop 2025 |
| 21 | GPT-4 + Sparse Voxel Maps | 38.9 | 40.9 | 35.0 | OpenEQA | CVPR 2024 |
| 22 | LLaMA-2 + LLaVA-1.5 | 36.8 | 39.6 | 31.1 | OpenEQA | CVPR 2024 |
| 23 | GPT-4 + ConceptGraphs | 36.5 | 37.8 | 34.0 | OpenEQA | CVPR 2024 |
| 24 | Claude 3 (20f) | 36.3 | - | - | AlanaVLM | arXiv 2024 |
| 25 | GPT-4 (blind) | 33.5 | 32.5 | 35.5 | OpenEQA | CVPR 2024 |
| 26 | LLaMA-2 (blind) | 28.3 | 27.9 | 29.0 | OpenEQA | CVPR 2024 |

---

## Paper References

| Short Name | Full Title | arXiv/Venue |
|------------|-----------|-------------|
| OpenEQA | OpenEQA: Embodied Question Answering in the Era of Foundation Models | CVPR 2024, arXiv:2312.15857 |
| AlanaVLM | AlanaVLM: A Multimodal Embodied AI Foundation Model for Egocentric Video Understanding | arXiv:2406.13807 |
| CoV | Chain-of-View Prompting for Spatial Reasoning | arXiv:2601.05172 |
| 3D-Mem | 3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning | CVPR 2025, arXiv:2411.17735 |
| GraphPad | GraphPad: Inference-Time 3D Scene Graph Updates for EQA | arXiv:2506.01174 |
| R-EQA | R-EQA: Retrieval-Augmented Generation for Embodied Question Answering | Embodied AI Workshop 2025 |

---

## Our Position

- **ScanNet**: 73.1 MNAS (#2, behind Gemini 1.5 Flash 74.0)
- **Caveat**: Our judge is Gemini 2.5 Pro, all others use GPT-4. Not directly comparable.
- **To make fair comparison**: Re-evaluate with GPT-4 as judge, or evaluate on ALL (ScanNet+HM3D).
