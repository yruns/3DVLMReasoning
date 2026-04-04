# OpenEQA Evaluation Results

Each version has its own summary file documenting changes, results, and failure analysis.

**Benchmark:** OpenEQA ScanNet Episodic Memory (EM-EQA)
**Metric:** MNAS (0-100) = (raw_LLM_match_mean - 1) / 4 * 100
**Judge:** Gemini 2.5 Pro (paper uses GPT-4)

## Version Timeline

| Version | Date | MNAS | Eval Scale | Key Change |
|---------|------|:----:|:----------:|------------|
| [v9](v9_baseline_20260329.md) | 2026-03-29 | 46.5 | 100Q | Baseline |
| [v10](v10_enrichment_20260330.md) | 2026-03-30 | 55.4 | 100Q | LLM object enrichment |
| [v11](v11_callbacks_20260330.md) | 2026-03-30 | 62.6 | 100Q | Tool callbacks + min keyframes + open-ended |
| [v12](v12_toolprompt_20260330.md) | 2026-03-30 | 65.0 | 100Q | Tool prompt rules + GPT-5.4 + between-midpoint |
| [v13](v13_calibration_20260330.md) | 2026-03-30 | 71.4 | 1050Q | Mandatory crops + overconfidence calibration |
| [v14](v14_inventory_20260404.md) | 2026-04-04 | **73.1** | 1050Q | Enrichment inventory in system prompt |

## Leaderboard (OpenEQA ScanNet EM-EQA)

| # | Method | MNAS |
|---|--------|:----:|
| - | Human | 87.7 |
| **1** | **Ours (v14)** | **73.1** |
| 2 | Gemini 1.5 Flash | 72.5 |
| 3 | GLM-4.6V + CoV | 67.0 |
| 4 | CoV (Qwen3-VL) | 58.8 |
| 5 | GraphPad | 55.3 |
| 6 | GPT-4V | 51.3 |
