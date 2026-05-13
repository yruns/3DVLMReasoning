# v8 Transcrib3D First300 Baseline - 2026-05-14

NR3D evaluation on the Transcrib3D `nr3d_first300_valid` comparison fold. This
is the first apples-to-apples baseline before porting Transcrib3D-style
text/geometry reasoning into the 3DVLMReasoning agent.

## Run Identity

- Branch: `feat/nr3d-transcrib3d-first300`
- Tip commit at Stage2 run time: `f90be6d`
- Internal version: `v8_transcrib3d_first300_baseline`
- Run ID: `v8_transcrib3d_first300_baseline_20260514`
- Stage2 backend: `gpt-5.4-2026-03-05` via internal ModelHub
- Judge: none; scoring is programmatic target-id classification
- SQLite row:
  `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v8_transcrib3d_first300_baseline_20260514'`

## Fold

- Source comparison fold:
  `/Users/bytedance/project/Transcrib3D/runs/modelhub_sharded/20260513-175704_nr3d_nr3d_first300_valid/`
- Transcrib3D protocol: lines 2..301 of
  `/Users/bytedance/project/Transcrib3D/data/referit3d/nr3d_test_sampled1000.csv`
  after `correct_guess=true && mentions_target_class=true`
- Local sample-id mapping:
  `docs/benchmark/nr3d/assets/transcrib3d_first300_valid_sample_ids_20260514.json`
- Fold size: 281 samples
- Mapping validation: 281/281 loaded by local `Nr3dDataset` with
  `bbox_source="phase8_gt_cg"`

## Pack

- Pack name: `pack_nr3d_v8_transcrib3d_first300_baseline`
- Pack path:
  `data/nr3d/scannet/<scene>/pack_nr3d_v8_transcrib3d_first300_baseline/`
- Keyframe mode: `query_driven`
- `keyframe_selection_uses_gt_target`: 0/281
- Keyframes per sample: 3/281
- `keyframe_selection_used_fallback`: 79/281
- Pack prep logs:
  - `tmp/nr3d_v8_transcrib3d_first300_prep4_20260514.log`
  - `tmp/nr3d_v8_transcrib3d_first300_prep_shard_{00..03}_20260514.log`
- Prep log checks: 0 traceback, 0 RSS guard, 0 retryable 429/503, 0 missing
  depth-aware visibility, 0 projection-fallback visibility.

## Commands

Pack prep used four scene-disjoint shards under one 15GB RSS guard:

```bash
./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- \
  bash tmp/nr3d_artifacts/run_v8_first300_prep_shards.sh
```

Stage2 checkpoints:

```bash
tmux new-session -d -s nr3d_v8_t3d_first300_eval 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids docs/benchmark/nr3d/assets/transcrib3d_first300_valid_sample_ids_20260514.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v8_transcrib3d_first300_baseline --output-dir tmp/nr3d_eval_v8_transcrib3d_first300_baseline_20260514 --workers 100 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v8_transcrib3d_first300_baseline_20260514_w100.log'
```

The initial pass produced 279 completed checkpoints and 2 `invalid_prompt`
failed sentinels. The failed checkpoint files were backed up under
`failed_invalid_prompt_attempt1/`, then rerun:

```bash
tmux new-session -d -s nr3d_v8_t3d_first300_failed2 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v8_t3d_first300_failed_invalid_prompt_20260514.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v8_transcrib3d_first300_baseline --output-dir tmp/nr3d_eval_v8_transcrib3d_first300_baseline_20260514 --workers 2 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v8_transcrib3d_first300_baseline_failed2_20260514.log'
```

Assemble and score:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids docs/benchmark/nr3d/assets/transcrib3d_first300_valid_sample_ids_20260514.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v8_transcrib3d_first300_baseline \
  --output-dir tmp/nr3d_eval_v8_transcrib3d_first300_baseline_20260514 \
  --workers 100 \
  --max-new-samples 0 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v8_transcrib3d_first300_baseline_20260514/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids docs/benchmark/nr3d/assets/transcrib3d_first300_valid_sample_ids_20260514.json \
  --output tmp/nr3d_eval_v8_transcrib3d_first300_baseline_20260514/leaderboard_metrics.json \
  --canonical-filter true
```

Ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v8_transcrib3d_first300_baseline_20260514 \
  --run-id v8_transcrib3d_first300_baseline_20260514 \
  --branch feat/nr3d-transcrib3d-first300 \
  --commit f90be6d \
  --backend pack_v1 \
  --judge-model none \
  --leaderboard-metrics tmp/nr3d_eval_v8_transcrib3d_first300_baseline_20260514/leaderboard_metrics.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

## Metrics

### Matched First300 Valid Comparison

| Method | Model | Fold | Correct | Overall | Easy | Hard | View-dep | View-indep |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Transcrib3D | GPT-4o text-only | 281 | 208 | 74.02 | 78.72 | 69.29 | 57.47 | 81.44 |
| Ours v8 baseline | GPT-5.4 + RGB agent | 281 | 209 | 74.38 | 78.72 | 70.00 | 59.76 | 80.40 |
| Delta | - | 0 | +1 | +0.36 pp | +0.00 pp | +0.71 pp | +2.29 pp | -1.04 pp |

This is a narrow +1-sample win, not a clear superiority result.

### Runtime / Reliability

| Check | Value |
|---|---:|
| Initial completed checkpoints | 279 |
| Initial `invalid_prompt` sentinels | 2 |
| Final completed checkpoints | 281 |
| Final failed sentinels | 0 |
| retryable 429 | 288 |
| retryable 503 | 0 |
| attempts at 3/5 or higher | 0 |
| `callback is not configured` | 0 |
| `rss_guard` | 0 |
| `Traceback` | 0 |

### Per-Sample Agreement vs Transcrib3D

Raw comparison:

`tmp/nr3d_eval_v8_transcrib3d_first300_baseline_20260514/comparison_vs_transcrib3d.json`

| Bucket | Count |
|---|---:|
| Both correct | 166 |
| Ours correct, Transcrib3D wrong | 43 |
| Transcrib3D correct, ours wrong | 42 |
| Both wrong | 30 |

Representative Transcrib3D-correct / ours-wrong cases:

- line 5, `scannet/scene0221_00::3::23651`: larger chair near octagonal table
- line 6, `scannet/scene0353_00::14::19411`: book closest to grey couch
- line 18, `scannet/scene0030_00::26::30435`: left window near green chalkboard
- line 53, `scannet/scene0653_00::44::18781`: filing cabinet under desk
- line 85, `scannet/scene0549_00::5::9189`: end table with vase and flowers
- line 145/146, `scannet/scene0565_00::24::*`: green chair beside grey chair

These failures are mostly category-compatible candidate ranking, metric
spatial relation, color/size disambiguation, and relative-viewpoint language.
They match the Transcrib3D transfer hypotheses: stronger text-first candidate
tables and deterministic geometry/color helpers are likely more useful than
more raw image turns.

## Caveats

- This is a 281-sample comparison fold, not the full NR3D benchmark.
- The fold is intentionally matched to the Transcrib3D first300-valid protocol;
  its split counts differ slightly from our local canonical view-dep token set.
- Since the margin is only one sample, this baseline does not satisfy the
  "clearly surpass Transcrib3D" goal.
