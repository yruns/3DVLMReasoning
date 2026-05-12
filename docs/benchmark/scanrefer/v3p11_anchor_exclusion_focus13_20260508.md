# ScanRefer v3.11 - same-category anchor-exclusion focus13 ablation

**Branch**: `feat/scanrefer-v3p10-stage-audit`
**Tip commit at harvest time**: `8acd3da` (working tree dirty; v3.11 code
changes not yet committed)
**Run IDs** (SQLite):
`v3p11_tadg_anchor_exclusion_focus13_20260508`,
`v3p11_efg_2d_mark_focus13_20260508`
**Date harvested**: 2026-05-08 GMT+8

## Headline

This is a focused 13-sample ablation over the paired v3.10-vs-v3.9 flip set,
not a new random100 gate. It uses query-driven evidence only; GT is used only
for offline aggregation-GT rescoring.

| Variant | n | Acc@0.25 | Acc@0.50 | mean IoU |
|---|---:|---:|---:|---:|
| v3.9 anchor/evidence guards | 13 | 46.15 | 30.77 | 0.3178 |
| v3.10 terminal latch | 13 | 53.85 | 46.15 | 0.4176 |
| v3.11 EFG 2D-mark experiment (rejected) | 13 | 53.85 | 38.46 | 0.3482 |
| v3.11 TADG anchor-exclusion only | 13 | 53.85 | 46.15 | 0.4488 |

Interpretation: the TADG-only patch fixes the concrete scene0660 regression
and improves mean IoU on this focused set, but it does not improve Acc@0.25 or
Acc@0.50 over v3.10 on the matched 13 samples. The EFG 2D-mark experiment was
reverted because it reduced mean IoU and Acc@0.50 on the same slice.

## What Changed

TADG candidate coverage now ignores the current spatial anchor when checking
whether same-category candidates were omitted from `compare_proposals_spatial`.
This handles traces such as "chair next to another same chair", where the
anchor chair is intentionally excluded from candidate_ids and should not cause
`TADG_CANDIDATE_COVERAGE`.

A separate EFG experiment blocked cited visible proposals that lacked a
`boxes_2d` mark in the cited frame. It was tested and rejected after focus13
because it did not reproduce the scene0660 win and lowered focus-set Acc@0.50.

## Commands

TADG-only focus13 run:

```bash
tmux new-session -d -s v3p11_tadg_focus13 \
  "cd /Users/bytedance/project/3DVLMReasoning && \
  CVRA_DISABLE=1 PYTHONPATH=src .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/v3p11_regression_recovery13_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p11_tadg_anchor_exclusion_regression_recovery13_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 2 \
  --sample-retries 4 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard \
  2>&1 | tee tmp/v3p11_tadg_anchor_exclusion_regression_recovery13_eval.log"
```

Aggregation-GT rescore:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/v3p11_tadg_anchor_exclusion_regression_recovery13_eval/side_by_side.json \
  --output-dir tmp/v3p11_tadg_anchor_exclusion_regression_recovery13_eval_agg_gt
```

Leaderboard slicing:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.scanrefer_leaderboard_metrics \
  --side-by-side tmp/v3p11_tadg_anchor_exclusion_regression_recovery13_eval_agg_gt/side_by_side.json \
  --sample-ids tmp/v3p11_regression_recovery13_sample_ids.json \
  --output tmp/v3p11_tadg_anchor_exclusion_regression_recovery13_eval_agg_gt/leaderboard_metrics.json
```

SQLite ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/v3p11_tadg_anchor_exclusion_regression_recovery13_eval_agg_gt \
  --run-id v3p11_tadg_anchor_exclusion_focus13_20260508 \
  --branch feat/scanrefer-v3p10-stage-audit \
  --commit 8acd3da \
  --leaderboard-metrics tmp/v3p11_tadg_anchor_exclusion_regression_recovery13_eval_agg_gt/leaderboard_metrics.json \
  --db docs/benchmark/scanrefer/runs.sqlite \
  --keyframe-mode query_driven \
  --notes "v3.11 focused 13-case ablation after reverting unsupported EFG 2D-mark experiment; TADG candidate-coverage ignores the current anchor when same-category compare candidate_ids intentionally exclude anchor; no GT at inference; aggregation-GT focus13 result mean_iou=0.4488 acc25=0.5385 acc50=0.4615."
```

Rejected EFG 2D-mark experiment artifacts were also rescored and ingested under
`v3p11_efg_2d_mark_focus13_20260508`.

## Raw Artifacts

- Focus sample list: `tmp/v3p11_regression_recovery13_sample_ids.json`
- TADG-only raw output:
  `tmp/v3p11_tadg_anchor_exclusion_regression_recovery13_eval/`
- TADG-only aggregation-GT rescore:
  `tmp/v3p11_tadg_anchor_exclusion_regression_recovery13_eval_agg_gt/`
- Rejected EFG experiment raw output:
  `tmp/v3p11_anchor_exclusion_regression_recovery13_eval/`
- Rejected EFG experiment aggregation-GT rescore:
  `tmp/v3p11_anchor_exclusion_regression_recovery13_eval_agg_gt/`
- SQLite DB: `docs/benchmark/scanrefer/runs.sqlite`

## Evidence

TADG-only aggregation-GT rescore:

```text
n=13 mean_iou=0.4488 acc25=0.5385 acc50=0.4615
stats: missing_scene=0 missing_target=0 no_prediction=0 completed_new_gt=13
```

Rejected EFG experiment aggregation-GT rescore:

```text
n=13 mean_iou=0.3482 acc25=0.5385 acc50=0.3846
stats: missing_scene=0 missing_target=0 no_prediction=0 completed_new_gt=13
```

Per-sample deltas vs v3.10 after aggregation-GT rescoring:

| Sample | v3.10 pid / IoU | TADG-only pid / IoU | Delta |
|---|---:|---:|---:|
| `scannet/scene0660_00::4::4` | 9 / 0.0000 | 1 / 1.0000 | +1.0000 |
| `scannet/scene0377_00::3::1` | 41 / 0.0000 | 41 / 0.0000 | +0.0000 |
| `scannet/scene0645_00::29::3` | 4 / 0.0000 | 62 / 0.6287 | +0.6287 |
| `scannet/scene0025_00::9::0` | 20 / 0.0056 | 20 / 0.0056 | +0.0000 |
| `scannet/scene0678_00::11::0` | 30 / 0.0000 | 15 / 0.0000 | +0.0000 |
| `scannet/scene0574_00::25::4` | 6 / 0.0169 | 4 / 0.4364 | +0.4195 |
| `scannet/scene0203_00::14::3` | 2 / 0.9759 | 2 / 0.9759 | +0.0000 |
| `scannet/scene0697_00::11::2` | 8 / 0.9726 | 8 / 0.9726 | +0.0000 |
| `scannet/scene0474_00::15::3` | 6 / 0.8727 | 6 / 0.8727 | +0.0000 |
| `scannet/scene0699_00::11::3` | 6 / 0.8333 | 6 / 0.8333 | +0.0000 |
| `scannet/scene0565_00::12::2` | 29 / 0.6417 | 30 / 0.0000 | -0.6417 |
| `scannet/scene0100_00::26::0` | 18 / 0.7175 | 34 / 0.0858 | -0.6317 |
| `scannet/scene0149_00::21::2` | 56 / 0.3932 | 27 / 0.0241 | -0.3691 |

SQLite reproduction:

```sql
SELECT run_id, n_total,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', mean_iou_overall) AS miou,
       keyframe_mode
FROM runs
WHERE run_id IN (
  'v3p11_tadg_anchor_exclusion_focus13_20260508',
  'v3p11_efg_2d_mark_focus13_20260508'
)
ORDER BY run_id;
```

Expected:

```text
v3p11_efg_2d_mark_focus13_20260508|13|0.5385|0.3846|0.3482|query_driven
v3p11_tadg_anchor_exclusion_focus13_20260508|13|0.5385|0.4615|0.4488|query_driven
```

## Caveats

- This is a focus13 ablation selected from v3.9/v3.10 paired flips, not the
  frozen random100 fold. It cannot replace the v3.10 headline.
- Acc@0.25 and Acc@0.50 do not improve over v3.10 on this focus set; only mean
  IoU improves. A random100 gate is still required before quoting v3.11 as a
  new development-fold result.
- The TADG change is trace-only and does not use GT at inference. The GT-derived
  aggregation bboxes are used only in the offline rescore and documentation.
