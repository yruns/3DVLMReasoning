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

See [leaderboard.md](leaderboard.md) for the full list with paper references.

| # | Method | ScanNet MNAS | Paper | Judge |
|---|--------|:----:|-------|:-----:|
| - | Human | 87.7 | OpenEQA (CVPR 2024) | - |
| 1 | Gemini 1.5 Flash (50f) | 74.0 | AlanaVLM (arXiv 2024) | GPT-4 |
| **2** | **Ours (v14)** | **73.1** | **This work** | **Gemini 2.5 Pro** |
| 3 | Gemini 1.5 Pro (50f) | 66.9 | AlanaVLM (arXiv 2024) | GPT-4 |
| 4 | GPT-4V (50f, full eval) | 57.4 | AlanaVLM (arXiv 2024) | GPT-4 |
| 5 | GPT-4V (500Q subset) | 51.3 | OpenEQA (CVPR 2024) | GPT-4 |
| 6 | R-EQA + Qwen2.5-VL | 49.1 | R-EQA (Workshop 2025) | GPT-4 |
| 7 | GPT-4 + LLaVA-1.5 | 45.4 | OpenEQA (CVPR 2024) | GPT-4 |
| 8 | GPT-4 + ConceptGraphs | 37.8 | OpenEQA (CVPR 2024) | GPT-4 |
| 9 | GPT-4 (blind) | 32.5 | OpenEQA (CVPR 2024) | GPT-4 |

**Note:** Our judge is Gemini 2.5 Pro; all other entries use GPT-4. Scores not directly comparable.

## Evaluation Commands

### Full OpenEQA Run (inference + LLM judge scoring)

```bash
python -m src.agents.examples.openeqa_official_question_pilot \
    --json-path data/open-eqa-v0.json \
    --data-root data/OpenEQA/scannet \
    --max-samples 1200 \
    --workers 6 \
    --evaluate \
    --output-root tmp/openeqa_eval_v13_full
```

### Key Flags

| Flag | Description |
|------|-------------|
| `--json-path` | Official OpenEQA question set |
| `--data-root` | Root of prepared ScanNet scenes |
| `--max-samples` | Total questions to run (1200 = all available) |
| `--workers` | Parallel inference workers (default 6) |
| `--evaluate` | Enable LLM-match judge scoring after inference |
| `--eval-model` | Judge model (default `gemini-2.5-pro`, uses GeminiClientPool) |
| `--output-root` | Results directory |
| `--num-scenes` / `--questions-per-scene` | Alternative to --max-samples for subset runs (e.g., `--num-scenes 20 --questions-per-scene 5` = 100Q) |

### Re-run Scoring Only (predictions already saved)

```python
from benchmarks.openeqa_official_eval import evaluate_predictions_with_official_llm_match
evaluate_predictions_with_official_llm_match(
    dataset_items=dataset,
    predictions=e2e_preds,
    output_path=Path("tmp/.../official_predictions_e2e-metrics.json"),
    eval_model="gemini-2.5-pro",
    max_workers=12,
    max_retries=10,
)
```
