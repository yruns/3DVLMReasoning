# ScanRefer v3.10 - terminal latch and guard cleanup

**Branch**: `feat/scanrefer-v3p10-stage-audit`
**Tip commit at harvest time**: `8acd3da` (working tree dirty; v3.10 code
changes not yet committed)
**Run ID** (SQLite): `v3p10_terminal_latch_random100_20260508`
**Date harvested**: 2026-05-08 GMT+8

## Headline

Random100 fold, frozen seed=20260503:
`tmp/scanrefer_artifacts/random100_sample_ids.json`.

| Variant | Acc@0.25 | Acc@0.50 | mean IoU | failed after retry |
|---|---:|---:|---:|---:|
| v3.8 relation-ranking fresh | 49.0% | 43.0% | 0.4120 | 0 |
| v3.9 anchor/evidence guards | 53.0% | 46.0% | 0.4424 | 0 |
| **v3.10 terminal latch** | **54.0%** | **48.0%** | **0.4550** | **0** |

Per-column leaderboard slices after aggregation-GT rescoring:

| | Unique | Multiple | Overall |
|---|---:|---:|---:|
| Acc@0.25 | 82.14 | 43.06 | **54.00** |
| Acc@0.50 | 82.14 | 34.72 | **48.00** |

Interpretation: v3.10 is the best honest query-driven random100 result so far:
+1pp/+2pp over v3.9 and +5pp/+5pp over v3.8 fresh. It is still below Z3D's
zero-shot Mask3D-pool reference of 58.9 / 52.7, so this is not a solved state.

## What Changed

v3.10 targeted the v3.9 regressions with no GT at inference:

- EFG now allows proximity relations when the latest spatial comparison ranks
  the submitted target first, and no longer treats "red chair to the left of it"
  as "target must be leftmost".
- TADG now covers `near`, `next_to`, `left_of`, and `right_of`, adds inverse
  pronoun relation matching, blocks incomplete same-category spatial compares,
  and rejects anchor-self overrides.
- `submit_final` is latched: once a terminal answer is accepted in a
  DeepAgents turn, later same-turn `submit_final` calls are recorded as
  `ALREADY_SUBMITTED` and cannot overwrite the accepted final before the outer
  run loop observes it.

## Commands

Main random100 gate:

```bash
tmux new-session -d -s v3p10_latch_random100 \
  "cd /Users/bytedance/project/3DVLMReasoning && \
  CVRA_DISABLE=1 PYTHONPATH=src .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p10_terminal_latch_random100_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 4 \
  --sample-retries 4 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard \
  2>&1 | tee tmp/v3p10_terminal_latch_random100_eval.log"
```

One model-level failed sample was retried with the same raw query and no GT at
inference:

```bash
tmux new-session -d -s v3p10_latch_retry_failed \
  "cd /Users/bytedance/project/3DVLMReasoning && \
  CVRA_DISABLE=1 PYTHONPATH=src .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/v3p10_latch_failed_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p10_terminal_latch_retry_failed_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 1 \
  --sample-retries 4 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard \
  2>&1 | tee tmp/v3p10_terminal_latch_retry_failed_eval.log"
```

The retry replaced only `scannet/scene0474_00::15::3`, producing:
`tmp/v3p10_terminal_latch_random100_eval_retry_merged/side_by_side.json`.

Aggregation-GT rescore:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/v3p10_terminal_latch_random100_eval_retry_merged/side_by_side.json \
  --output-dir tmp/v3p10_terminal_latch_random100_eval_retry_merged_agg_gt
```

Leaderboard slicing:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.scanrefer_leaderboard_metrics \
  --side-by-side tmp/v3p10_terminal_latch_random100_eval_retry_merged_agg_gt/side_by_side.json \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --output tmp/v3p10_terminal_latch_random100_eval_retry_merged_agg_gt/leaderboard_metrics.json
```

SQLite ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/v3p10_terminal_latch_random100_eval_retry_merged_agg_gt \
  --run-id v3p10_terminal_latch_random100_20260508 \
  --branch feat/scanrefer-v3p10-stage-audit \
  --commit 8acd3da \
  --leaderboard-metrics tmp/v3p10_terminal_latch_random100_eval_retry_merged_agg_gt/leaderboard_metrics.json \
  --db docs/benchmark/scanrefer/runs.sqlite \
  --keyframe-mode query_driven \
  --notes "v3.10 terminal-latch/guard cleanup fresh random100: no GT at inference; EFG proximity and anchor-relative fixes, TADG candidate coverage/inverse relation/strict anchor checks, submit_final terminal latch; one model-level failed sample retried and merged before aggregation-GT rescore; result 54/48 on frozen random100."
```

## Raw Artifacts

- Agent output: `tmp/v3p10_terminal_latch_random100_eval/`
- Failed-sample retry: `tmp/v3p10_terminal_latch_retry_failed_eval/`
- Merged raw output: `tmp/v3p10_terminal_latch_random100_eval_retry_merged/`
- Aggregation-GT rescore: `tmp/v3p10_terminal_latch_random100_eval_retry_merged_agg_gt/`
- Frozen fold: `tmp/scanrefer_artifacts/random100_sample_ids.json`
- SQLite DB: `docs/benchmark/scanrefer/runs.sqlite`

## Evidence

Raw side-by-side after retry merge, before aggregation-GT rescore:

```text
n=100 mean_iou=0.2297 acc25=0.4800 acc50=0.1700
failed=[]
```

Aggregation-GT rescore:

```text
n=100 mean_iou=0.4550 acc25=0.5400 acc50=0.4800
stats: missing_scene=0 missing_target=0 no_prediction=0 completed_new_gt=100
```

Leaderboard slicing:

```text
n_total=100 (Unique=28, Multiple=72)
acc25  overall=0.5400 unique=0.8214 multiple=0.4306
acc50  overall=0.4800 unique=0.8214 multiple=0.3472
```

SQLite reproduction:

```sql
SELECT run_id, n_total,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', mean_iou_overall) AS miou,
       printf('%.4f', acc25_unique) AS u25,
       printf('%.4f', acc50_unique) AS u50,
       printf('%.4f', acc25_multiple) AS m25,
       printf('%.4f', acc50_multiple) AS m50,
       keyframe_mode
FROM runs
WHERE run_id IN (
  'v3p8_relation_ranking_fresh_random100_20260508',
  'v3p9_anchor_guard_v18_20260508',
  'v3p10_terminal_latch_random100_20260508'
)
ORDER BY run_id;
```

Expected:

```text
v3p8_relation_ranking_fresh_random100_20260508|100|0.4900|0.4300|0.4120|0.7500|0.7500|0.3889|0.3056|query_driven
v3p9_anchor_guard_v18_20260508|100|0.5300|0.4600|0.4424|0.7857|0.7857|0.4306|0.3333|query_driven
v3p10_terminal_latch_random100_20260508|100|0.5400|0.4800|0.4550|0.8214|0.8214|0.4306|0.3472|query_driven
```

Tool trace durability: v3.10 ingested 2584 rows into `tool_calls`, and all
100 samples are `status='completed'` in SQLite after the failed-sample retry
merge.

## Delta vs v3.9

| Slice | v3.9 | v3.10 | Delta |
|---|---:|---:|---:|
| mean IoU | 0.4424 | 0.4550 | +0.0126 |
| Overall Acc@0.25 | 53.00 | 54.00 | +1.00 |
| Overall Acc@0.50 | 46.00 | 48.00 | +2.00 |
| Unique Acc@0.25 | 78.57 | 82.14 | +3.57 |
| Unique Acc@0.50 | 78.57 | 82.14 | +3.57 |
| Multiple Acc@0.25 | 43.06 | 43.06 | +0.00 |
| Multiple Acc@0.50 | 33.33 | 34.72 | +1.39 |

Acc@0.25 flips vs v3.9: 7 up, 6 down. Acc@0.50 flips: 6 up, 4 down.

Notable recoveries:

| Sample | v3.9 IoU | v3.10 IoU |
|---|---:|---:|
| `scannet/scene0203_00::14::3` | 0.0000 | 0.9759 |
| `scannet/scene0697_00::11::2` | 0.0000 | 0.9726 |
| `scannet/scene0474_00::15::3` | 0.0000 | 0.8727 |
| `scannet/scene0699_00::11::3` | 0.0000 | 0.8333 |
| `scannet/scene0565_00::12::2` | 0.0000 | 0.6417 |
| `scannet/scene0100_00::26::0` | 0.0858 | 0.7175 |
| `scannet/scene0149_00::21::2` | 0.1131 | 0.3932 |

Notable regressions:

| Sample | v3.9 IoU | v3.10 IoU |
|---|---:|---:|
| `scannet/scene0660_00::4::4` | 1.0000 | 0.0000 |
| `scannet/scene0377_00::3::1` | 0.8658 | 0.0000 |
| `scannet/scene0645_00::29::3` | 0.6287 | 0.0000 |
| `scannet/scene0025_00::9::0` | 0.5524 | 0.0056 |
| `scannet/scene0678_00::11::0` | 0.4492 | 0.0000 |
| `scannet/scene0574_00::25::4` | 0.4364 | 0.0169 |

## Caveats

- This is the frozen random100 development fold, not full 9508 val.
- No GT is used during inference. GT appears only in post-run aggregation-GT
  IoU scoring and in this diagnosis.
- One model-level no-match failure (`scene0474_00::15::3`) was retried with the
  same query and no GT at inference, then merged before rescore. The retry was
  necessary because the first pass ended in a guarded `proposal_id=-1` failure,
  not because of an infrastructure exception.
- The run is a small development-fold improvement, not a zero-shot SOTA claim.
  It remains 4.9pp / 4.7pp below Z3D's 58.9 / 52.7 reference.
- Scene0354 and scene0690 remain unresolved hard regressions; scene0354 is
  highly stochastic under same-category chair/anchor reasoning, and scene0690
  still needs more robust inverse-relation candidate discipline.
