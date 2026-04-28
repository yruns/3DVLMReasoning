# OpenEQA Evaluation Results

Each version has its own summary file documenting changes, results, and failure analysis.

**Benchmark:** OpenEQA ScanNet Episodic Memory (EM-EQA)
**Metric:** MNAS (0-100) = (raw_LLM_match_mean - 1) / 4 * 100
**Judge:** Gemini 2.5 Pro (paper uses GPT-4)

## Version Timeline

MNAS column is the headline number reported in each version's doc — `v9..v14` use the **e2e** column on their respective fold; `v15` reports **stage2** because the e2e judge column was only partially scored (see `v15_trajectory_aware_20260420.md` for cross-version comparison).

| Version | Date | MNAS | Column | Eval Scale | Key Change |
|---------|------|:----:|:------:|:----------:|------------|
| [v9](v9_baseline_20260329.md) | 2026-03-29 | 46.5 | e2e | 100Q | Baseline |
| [v10](v10_enrichment_20260330.md) | 2026-03-30 | 55.4 | e2e | 100Q | LLM object enrichment |
| [v11](v11_callbacks_20260330.md) | 2026-03-30 | 62.6 | e2e | 100Q | Tool callbacks + min keyframes + open-ended |
| [v12](v12_toolprompt_20260330.md) | 2026-03-30 | 65.0 | e2e | 100Q | Tool prompt rules + GPT-5.4 + between-midpoint |
| [v13](v13_calibration_20260330.md) | 2026-03-30 | 71.4 | e2e | 1050Q | Mandatory crops + overconfidence calibration |
| [v14](v14_inventory_20260404.md) | 2026-04-04 | 73.1 | e2e | 1050Q | Enrichment inventory in system prompt |
| [v15](v15_trajectory_aware_20260420.md) | 2026-04-20 | **74.33** | stage2 | 1050Q (frozen) | Trajectory-aware Stage 1 (pose-aware + L1 frustum) |
| [v15-chassis-repro](v15_chassis_repro_20260427.md) | 2026-04-27 | 73.76 | stage2 | 1047Q (frozen) | Plan B QA pack migration on top of v15 — chassis trio + skill catalog |
| [postfix-mini35](postfix_mini35_20260428.md) | 2026-04-28 | 80.00¹ | stage2 | 35Q (mini frozen) | Fix 1+2+3 verification — switch_or_expand_hypothesis bug, QA opts out of chassis, per-qid LLM-call binding |
| [postfix-full1050](postfix_full1050_20260428.md) | 2026-04-28 | 73.09 | stage2 | 1047Q (frozen) | Fix 1+2+3 full-scale rerun — net **noisily negative** vs v15 and pre-fix chassis; the mini35 result was n=35 noise |

> v15 vs v14 on the matched **stage2** column: 74.33 vs 71.60 (`v14_full`) / 72.29 (`v14_rerun`) → **+2.04 to +2.73 MNAS**.
>
> v15-chassis-repro vs v15_s1_l1 on matched stage2: 73.76 vs 74.33 → **−0.57 MNAS** (within noise, 71.5% per-question agreement). Plan B QA pack migration ships safely on OpenEQA.
>
> ¹ postfix-mini35 (80.00) is on a 35-Q mini subset and turned out to be **small-sample noise**. Full 1050Q rerun (`postfix-full1050`) shows net −1.24 MNAS vs v15 — the 38 original chassis regressions were 47% recovered but ~23 new severe regressions appeared on different qids. Lesson: **never claim a benchmark improvement on n<200 questions**.

## Leaderboard (OpenEQA ScanNet EM-EQA)

See [leaderboard.md](leaderboard.md) for the full list with paper references.

| # | Method | ScanNet MNAS | Paper | Judge |
|---|--------|:----:|-------|:-----:|
| - | Human | 87.7 | OpenEQA (CVPR 2024) | - |
| 1 | Gemini 1.5 Flash (50f) | 74.0 | AlanaVLM (arXiv 2024) | GPT-4 |
| **2** | **Ours (v15, `s1_l1` stage2)** | **74.33** | **This work** | **Gemini 2.5 Pro** |
| 3 | Ours (v14, e2e) | 73.1 | This work | Gemini 2.5 Pro |
| 4 | Gemini 1.5 Pro (50f) | 66.9 | AlanaVLM (arXiv 2024) | GPT-4 |
| 5 | GPT-4V (50f, full eval) | 57.4 | AlanaVLM (arXiv 2024) | GPT-4 |
| 6 | GPT-4V (500Q subset) | 51.3 | OpenEQA (CVPR 2024) | GPT-4 |
| 7 | R-EQA + Qwen2.5-VL | 49.1 | R-EQA (Workshop 2025) | GPT-4 |
| 8 | GPT-4 + LLaVA-1.5 | 45.4 | OpenEQA (CVPR 2024) | GPT-4 |
| 9 | GPT-4 + ConceptGraphs | 37.8 | OpenEQA (CVPR 2024) | GPT-4 |
| 10 | GPT-4 (blind) | 32.5 | OpenEQA (CVPR 2024) | GPT-4 |

**Note:** Our judge is Gemini 2.5 Pro; all other entries use GPT-4. Scores not directly comparable. The v14/v15 cross-version comparison is internally consistent (same judge, same frozen 1050Q set when `--force-selection` is used).

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
