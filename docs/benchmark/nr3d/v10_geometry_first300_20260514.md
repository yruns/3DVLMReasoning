# v10 Geometry Ranking First300 - 2026-05-14

NR3D rerun on the Transcrib3D `nr3d_first300_valid` matched fold after adding
the deterministic geometry-ranking VG helper.

## Run Identity

- Branch: `feat/nr3d-transcrib3d-first300`
- Tip commit at run time: `5b53037`
- Internal version: `v10_geometry_first300`
- Run ID: `v10_geometry_first300_20260514`
- Stage 2 backend: `gpt-5.4-2026-03-05` via internal ModelHub
- Judge: none; scoring is programmatic target-id classification
- SQLite row:
  `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v10_geometry_first300_20260514'`

## Fold And Pack

- Split: Transcrib3D `nr3d_first300_valid`
- Fold size: 281 valid samples from Transcrib3D's first 300 NR3D rows
- Selection file:
  `docs/benchmark/nr3d/assets/transcrib3d_first300_valid_sample_ids_20260514.json`
- Prepared pack:
  `data/nr3d/scannet/<scene>/pack_nr3d_v8_transcrib3d_first300_baseline/`
- Canonical filter after restriction: `n_full=281`, `n_filtered=281`
- Matched baseline:
  `/Users/bytedance/project/Transcrib3D/runs/modelhub_sharded/20260513-175704_nr3d_nr3d_first300_valid/`

## What Changed From v9

v10 keeps the v9 `Scene Proposal Inventory` prompt prior and adds
`rank_proposals_by_geometry(candidate_ids, criterion)` for deterministic
same-category size and height comparisons. Supported criteria are
`largest`, `smallest`, `tallest`, `shortest`, `highest`, `lowest`,
`widest`, and `narrowest`.

No fold, pack, visibility index, callback wiring, or final-decision guard was
changed.

## Raw Artifacts

- Eval output:
  `tmp/nr3d_eval_v10_geometry_first300_20260514/`
- Per-sample checkpoints:
  `tmp/nr3d_eval_v10_geometry_first300_20260514/per_sample/pack_nr3d_v8_transcrib3d_first300_baseline/*.json`
- Initial failed sentinel backup:
  `tmp/nr3d_eval_v10_geometry_first300_20260514/failed_invalid_prompt_attempt1/`
- Failed-rerun sample ids:
  `tmp/nr3d_artifacts/v10_geometry_first300_failed_invalid_prompt_20260514.json`
- Side-by-side output:
  `tmp/nr3d_eval_v10_geometry_first300_20260514/side_by_side.json`
- Leaderboard metrics:
  `tmp/nr3d_eval_v10_geometry_first300_20260514/leaderboard_metrics.json`
- Comparison:
  `tmp/nr3d_eval_v10_geometry_first300_20260514/comparison_vs_transcrib3d.json`
- Logs:
  - `tmp/nr3d_eval_v10_geometry_first300_20260514_w100.log`
  - `tmp/nr3d_eval_v10_geometry_first300_failed10_20260514.log`
  - `tmp/nr3d_eval_v10_geometry_first300_20260514_assemble.log`
  - `tmp/nr3d_eval_v10_geometry_first300_20260514_metrics.log`

## Commands

Initial 100-worker checkpoint pass:

```bash
tmux new-session -d -s nr3d_v10_geometry_first300_20260514 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids docs/benchmark/nr3d/assets/transcrib3d_first300_valid_sample_ids_20260514.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v8_transcrib3d_first300_baseline --output-dir tmp/nr3d_eval_v10_geometry_first300_20260514 --workers 100 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v10_geometry_first300_20260514_w100.log'
```

The initial pass produced 271 completed checkpoints and 10 `invalid_prompt`
failed sentinels. Those failed checkpoints were moved aside and rerun:

```bash
tmux new-session -d -s nr3d_v10_geometry_first300_failed10_20260514 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v10_geometry_first300_failed_invalid_prompt_20260514.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v8_transcrib3d_first300_baseline --output-dir tmp/nr3d_eval_v10_geometry_first300_20260514 --workers 10 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v10_geometry_first300_failed10_20260514.log'
```

Assemble and score:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids docs/benchmark/nr3d/assets/transcrib3d_first300_valid_sample_ids_20260514.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v8_transcrib3d_first300_baseline \
  --output-dir tmp/nr3d_eval_v10_geometry_first300_20260514 \
  --workers 100 \
  --max-new-samples 0 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v10_geometry_first300_20260514/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids docs/benchmark/nr3d/assets/transcrib3d_first300_valid_sample_ids_20260514.json \
  --output tmp/nr3d_eval_v10_geometry_first300_20260514/leaderboard_metrics.json \
  --canonical-filter true
```

Ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v10_geometry_first300_20260514 \
  --run-id v10_geometry_first300_20260514 \
  --branch feat/nr3d-transcrib3d-first300 \
  --commit 5b53037 \
  --backend pack_v1 \
  --judge-model none \
  --notes "v10 deterministic geometry ranking tool on Transcrib3D first300-valid matched fold; reused pack_nr3d_v8_transcrib3d_first300_baseline; workers=100 with 15GB RSS guard and 60s checks; initial 10 invalid_prompt failed sentinels moved aside and rerun with workers=10; final 281/281 completed; TADG + no-match + evidence-frame guards; no callback-not-configured; no rss_guard; 216/281 vs Transcrib3D GPT-4o 208/281" \
  --leaderboard-metrics tmp/nr3d_eval_v10_geometry_first300_20260514/leaderboard_metrics.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

## Metrics

### Leaderboard-Track Classification

| Metric | Transcrib3D GPT-4o | v8 baseline | v9 inventory | v10 geometry |
|---|---:|---:|---:|---:|
| Overall | 74.02 | 74.38 | 75.09 | 76.87 |
| Easy | 78.72 | 78.72 | 80.14 | 82.27 |
| Hard | 69.29 | 70.00 | 70.00 | 71.43 |
| View-Dep | 57.47* | 59.76 | 62.20 | 65.85 |
| View-Indep | 81.44* | 80.40 | 80.40 | 81.41 |

`*` Transcrib3D uses its own per-row view-dependence tags. The v8-v10 local
columns use the local `nr3d_leaderboard_metrics.py` split on the same sample
ids.

### Matched Counts

| Comparison | Correct | Delta vs v10 |
|---|---:|---:|
| v10 geometry | 216 / 281 | 0 |
| v9 inventory | 211 / 281 | +5 for v10 |
| v8 baseline | 209 / 281 | +7 for v10 |
| Transcrib3D GPT-4o | 208 / 281 | +8 for v10 |

Against Transcrib3D:

- v10-only correct: 44
- Transcrib3D-only correct: 36
- both correct: 172
- both wrong: 29

Against v9:

- v10-only correct: 23
- v9-only correct: 18
- net: +5 samples

### Slice Comparison vs Transcrib3D

| Slice | n | v10 correct | Transcrib3D correct | Delta |
|---|---:|---:|---:|---:|
| Easy | 141 | 116 | 111 | +5 |
| Hard | 140 | 100 | 97 | +3 |
| Local View-Dep | 82 | 54 | 47 | +7 |
| Local View-Indep | 199 | 162 | 161 | +1 |
| Transcrib3D spatial | 242 | 183 | 179 | +4 |
| Transcrib3D color | 70 | 48 | 52 | -4 |
| Transcrib3D shape | 41 | 35 | 28 | +7 |

## Runtime Notes

- Initial 100-worker pass: 271 completed, 10 `invalid_prompt` failed sentinels.
- Targeted failed rerun: 10/10 completed at `workers=10`.
- Final checkpoint audit: 281/281 completed, 0 failed sentinels.
- Combined log counts:
  - retryable `429`: 231
  - retryable `503`: 0
  - attempts at 3/5 or higher: 0
  - `callback is not configured`: 0
  - `rss_guard`: 0
  - `Traceback`: 0
  - `Loading CLIP model`: 9
- Tool trace counts:
  - `rank_proposals_by_geometry`: 40 calls across 38 samples
  - rank-tool subset accuracy: 29/38 = 76.32

## Interpretation

v10 is a clear matched-fold improvement over the Transcrib3D GPT-4o first300
baseline: +8 samples and +2.85 percentage points overall. The gain is strongest
on local view-dependent and shape-tagged rows, which matches the intended
geometry/superlative transfer target.

This is still not a full NR3D result. The same v10 code regressed on the fixed
random100 fold, so the geometry helper should be treated as useful but not yet
robust. Its policy needs tighter gating before scaling to the full test set.

## Caveats

- The fold is Transcrib3D's first300-valid subset, not the full NR3D test set.
- Transcrib3D's public full-valid result is 77.61 on 6910 valid rows and is not
  directly comparable to this 281-sample matched fold.
- No selector frame-overlap NMS was applied.
- The run uses a GT segmented/object-proposal candidate pool for NR3D-style
  selection; the target id is only used for scoring.
