# ScanRefer v3.8 - relation-ranking tool

**Branch**: `feat/scanrefer-v3p7-final-selection`
**Tip commit at harvest time**: `8acd3da` (working tree dirty; v3.8 code
change not yet committed)
**Run ID** (SQLite, fresh gate): `v3p8_relation_ranking_fresh_random100_20260508`
**Cached estimate run ID**: `v3p8_relation_ranking_cached_random100_20260508`
**Date harvested**: 2026-05-08 GMT+8

## Headline

Random100 fold, frozen seed=20260503:
`tmp/scanrefer_artifacts/random100_sample_ids.json`.

| Variant | Acc@0.25 | Acc@0.50 | mean IoU | no-prediction / failed |
|---|---:|---:|---:|---:|
| v3.3 vertical spatial | 48.0% | 42.0% | 0.4130 | 5 |
| v3.7 final-selection guards | 46.0% | 42.0% | 0.4024 | 0 |
| v3.8 relation-ranking cached estimate | 51.0% | 45.0% | 0.4357 | 0 |
| **v3.8 relation-ranking fresh gate** | **49.0%** | **43.0%** | **0.4120** | **0** |

Per-column leaderboard slices after aggregation-GT rescoring:

| | Unique | Multiple | Overall |
|---|---:|---:|---:|
| Acc@0.25 | 75.00 | 38.89 | **49.00** |
| Acc@0.50 | 75.00 | 30.56 | **43.00** |

Interpretation: the cached 93+7 estimate was optimistic. The fresh gate still
moves the query-driven random100 result above v3.3 by +1pp/+1pp and eliminates
no-prediction failures, but the lift is small and remains far below Z3D's
58.9 / 52.7 zero-shot reference.

## Hypothesis Tested

v3.7 removed no-prediction failures but left multi-distractor spatial ranking
unstable. Audits showed several wrong final picks on descriptions whose decisive
signal was not just vertical `above` / `below`, but horizontal or anchor
relations such as left/right/near/next-to.

v3.8 extends the existing tool surface without adding GT at inference:

- `compare_proposals_spatial` now accepts `left_of` and `right_of`, ranking by
  co-viewed 2D marked-frame geometry between candidate and anchor proposals.
- The same tool now accepts `near` and `next_to`, ranking by floor-plane
  proposal-center distance.
- Tool output exposes supporting and contradicting frame counts so the agent can
  see whether a left/right relation is visually supported or view-ambiguous.
- TADG aliases were updated so final answers that disagree with the new relation
  calls remain soft-blocked.
- `chassis_tools_version` was bumped from 8 to 9 to avoid prompt-cache reuse of
  the old tool contract.
- During the fresh gate, two abrupt exits at 38/100 and 76/100 exposed a
  duplicate CLIP lazy-load race under concurrent callbacks. A test-first
  `_clip_model_lock` fix was added, after which the run completed to 100/100.

## Commands

The focused relation-regression run used seven v3.7 Acc@0.25 regressions:
`tmp/scanrefer_artifacts/v3p8_relation_regression7_sample_ids.json`.

```bash
CVRA_DISABLE=1 PYTHONPATH=src .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/scanrefer_artifacts/v3p8_relation_regression7_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p8_relation_regression7_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 1 \
  --sample-retries 4 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard
```

Focused aggregation-GT rescore:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/v3p8_relation_regression7_eval/side_by_side.json \
  --output-dir tmp/v3p8_relation_regression7_eval_agg_gt
```

The cached random100 estimate was created by copying the v3.7 random100 output
and replacing only those seven per-sample JSON files, then resuming the runner:

```bash
CVRA_DISABLE=1 PYTHONPATH=src .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p8_relation_ranking_random100_cached_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 1 \
  --sample-retries 4 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard
```

Cached random100 aggregation-GT rescore:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/v3p8_relation_ranking_random100_cached_eval/side_by_side.json \
  --output-dir tmp/v3p8_relation_ranking_random100_cached_eval_agg_gt
```

Leaderboard slicing:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.scanrefer_leaderboard_metrics \
  --side-by-side tmp/v3p8_relation_ranking_random100_cached_eval_agg_gt/side_by_side.json \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --output tmp/v3p8_relation_ranking_random100_cached_eval_agg_gt/leaderboard_metrics.json
```

SQLite ingest:

```bash
.venv/bin/python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/v3p8_relation_ranking_random100_cached_eval_agg_gt \
  --run-id v3p8_relation_ranking_cached_random100_20260508 \
  --branch feat/scanrefer-v3p7-final-selection \
  --commit 8acd3da \
  --leaderboard-metrics tmp/v3p8_relation_ranking_random100_cached_eval_agg_gt/leaderboard_metrics.json \
  --db docs/benchmark/scanrefer/runs.sqlite \
  --keyframe-mode query_driven \
  --notes "v3.8 relation-ranking cached random100 estimate: v3.7 random100 reused except seven relation-regression samples rerun with no GT at inference; compare_proposals_spatial now supports left_of/right_of via co-viewed 2D marked geometry and near/next_to via floor-plane distance; TADG relation aliases updated; cached estimate improves Acc@0.25/0.50 to 51/45 but is not a fresh full random100 run"
```

Fresh random100 gate used the same runner command with a fresh output dir:

```bash
CVRA_DISABLE=1 PYTHONPATH=src .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p8_relation_ranking_fresh_random100_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 1 \
  --sample-retries 4 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard
```

The fresh gate resumed twice from durable checkpoints:

- `v3p8_fresh_random100_eval`: exited at 38/100 with no traceback.
- `v3p8_fresh_random100_resume1`: exited at 76/100 after duplicate CLIP lazy-load
  logs on `scene0645_00`.
- `v3p8_fresh_random100_resume2`: completed 100/100 after the CLIP-load lock fix.

Fresh aggregation-GT rescore:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/v3p8_relation_ranking_fresh_random100_eval/side_by_side.json \
  --output-dir tmp/v3p8_relation_ranking_fresh_random100_eval_agg_gt
```

Fresh leaderboard slicing:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.scanrefer_leaderboard_metrics \
  --side-by-side tmp/v3p8_relation_ranking_fresh_random100_eval_agg_gt/side_by_side.json \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --output tmp/v3p8_relation_ranking_fresh_random100_eval_agg_gt/leaderboard_metrics.json
```

Fresh SQLite ingest:

```bash
.venv/bin/python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/v3p8_relation_ranking_fresh_random100_eval_agg_gt \
  --run-id v3p8_relation_ranking_fresh_random100_20260508 \
  --branch feat/scanrefer-v3p7-final-selection \
  --commit 8acd3da \
  --leaderboard-metrics tmp/v3p8_relation_ranking_fresh_random100_eval_agg_gt/leaderboard_metrics.json \
  --db docs/benchmark/scanrefer/runs.sqlite \
  --keyframe-mode query_driven \
  --notes "v3.8 relation-ranking fresh random100 gate: no GT at inference; left_of/right_of co-viewed 2D relation ranking, near/next_to floor-distance ranking, TADG aliases, tool version 9, and CLIP lazy-load lock after two abrupt duplicate-load exits; fresh result 49/43, cached 51/45 was optimistic"
```

## Raw Artifacts

- Focused agent output: `tmp/v3p8_relation_regression7_eval/`
- Focused aggregation-GT rescore: `tmp/v3p8_relation_regression7_eval_agg_gt/`
- Cached random100 agent output:
  `tmp/v3p8_relation_ranking_random100_cached_eval/`
- Cached random100 aggregation-GT rescore:
  `tmp/v3p8_relation_ranking_random100_cached_eval_agg_gt/`
- Fresh random100 agent output:
  `tmp/v3p8_relation_ranking_fresh_random100_eval/`
- Fresh random100 aggregation-GT rescore:
  `tmp/v3p8_relation_ranking_fresh_random100_eval_agg_gt/`
- Frozen fold: `tmp/scanrefer_artifacts/random100_sample_ids.json`
- Focused sample ids:
  `tmp/scanrefer_artifacts/v3p8_relation_regression7_sample_ids.json`
- SQLite DB: `docs/benchmark/scanrefer/runs.sqlite`

## Evidence

Focused seven-case aggregation-GT rescore:

```text
n=7 mean_iou=0.4908 acc25=0.7143 acc50=0.4286
```

Cached random100 leaderboard slicing:

```text
n_total=100 (Unique=28, Multiple=72)
acc25  overall=0.5100 unique=0.7500 multiple=0.4167
acc50  overall=0.4500 unique=0.7500 multiple=0.3333
```

Fresh random100 aggregation-GT rescore:

```text
n=100 mean_iou=0.4120 acc25=0.4900 acc50=0.4300
stats: missing_scene=0 missing_target=0 no_prediction=0 completed_new_gt=100
```

Fresh random100 leaderboard slicing:

```text
n_total=100 (Unique=28, Multiple=72)
acc25  overall=0.4900 unique=0.7500 multiple=0.3889
acc50  overall=0.4300 unique=0.7500 multiple=0.3056
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
  'v3p3_vertical_spatial_random100_20260504',
  'v3p7_final_selection_random100_20260508',
  'v3p8_relation_ranking_cached_random100_20260508',
  'v3p8_relation_ranking_fresh_random100_20260508'
)
ORDER BY run_id;
```

Expected:

```text
v3p3_vertical_spatial_random100_20260504|100|0.4800|0.4200|0.4130|0.7500|0.7500|0.3750|0.2917|mask3d_query_driven
v3p7_final_selection_random100_20260508|100|0.4600|0.4200|0.4024|0.7500|0.7500|0.3472|0.2917|query_driven
v3p8_relation_ranking_cached_random100_20260508|100|0.5100|0.4500|0.4357|0.7500|0.7500|0.4167|0.3333|query_driven
v3p8_relation_ranking_fresh_random100_20260508|100|0.4900|0.4300|0.4120|0.7500|0.7500|0.3889|0.3056|query_driven
```

Tool trace durability: the fresh v3.8 run ingested 2480 rows into `tool_calls`,
and all 100 samples are `status='completed'` in SQLite.

## Seven-Case Delta

| Sample | cached v3.8 / IoU | fresh v3.8 / IoU | Fresh result |
|---|---:|---:|---|
| `scannet/scene0149_00::21::2` | 33 / 0.1131 | 33 / 0.1131 | still below Acc@0.25 |
| `scannet/scene0257_00::34::0` | 23 / 0.7239 | 23 / 0.7239 | stable Acc@0.50 |
| `scannet/scene0377_00::9::2` | 27 / 0.9844 | 15 / 0.0000 | fresh regression |
| `scannet/scene0474_00::15::3` | 6 / 0.8727 | 6 / 0.8727 | stable Acc@0.50 |
| `scannet/scene0591_00::5::2` | 24 / 0.2926 | 24 / 0.2926 | stable Acc@0.25 |
| `scannet/scene0645_00::19::3` | 42 / 0.0000 | 78 / 1.0000 | fresh recovery |
| `scannet/scene0678_00::11::0` | 13 / 0.4492 | 13 / 0.4492 | stable Acc@0.25 |

Two cases remain useful next-step diagnostics. On `scene0149_00::21::2`, the
agent visually cited the selected kitchen-cabinet proposal but handed
`compare_proposals_spatial` an incomplete candidate list, omitting exact-category
candidates that should have stayed in the comparison. Fresh randomness also
exposed instability: `scene0377_00::9::2` regressed from a cached correct pick
to IoU 0, while `scene0645_00::19::3` recovered from IoU 0 to exact GT. This
points to final-selection variance and candidate-set discipline rather than
Stage 1 keyframe coverage alone.

## Caveats

- This is the frozen random100 development fold, not full 9508 val.
- The fresh gate is the official v3.8 process result. The cached 51/45 estimate
  is retained only as pre-gate directional evidence.
- No GT is used during inference. GT appears only in
  post-run aggregation-GT IoU scoring and in this diagnosis.
- The result remains below Z3D's Mask3D-pool zero-shot reference
  (58.9 / 52.7 overall), so the task is not solved.
- The next step should audit remaining Multiple failures and the fresh
  `scene0377_00::9::2` regression before scaling to full 9508 validation.
