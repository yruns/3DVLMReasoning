# ScanRefer v3.7 - final-selection guards

**Branch**: `feat/scanrefer-v3p7-final-selection`
**Tip commit at harvest time**: `8acd3da` (working tree dirty; v3.7 code
change not yet committed)
**Run ID** (SQLite): `v3p7_final_selection_random100_20260508`
**Date harvested**: 2026-05-08 GMT+8

## Headline

Random100 fold, frozen seed=20260503:
`tmp/scanrefer_artifacts/random100_sample_ids.json`.

| Variant | Acc@0.25 | Acc@0.50 | mean IoU | no-prediction / failed |
|---|---:|---:|---:|---:|
| v3.3 vertical spatial | **48.0%** | 42.0% | **0.4130** | 5 |
| v3.6 TADG completed-only subset | 46.94% | 42.86% | 0.4106 | 0 / 98 rows |
| **v3.7 final-selection guards** | 46.0% | **42.0%** | 0.4024 | **0** |

Per-column leaderboard slices after aggregation-GT rescoring:

| | Unique | Multiple | Overall |
|---|---:|---:|---:|
| Acc@0.25 | 75.00 | 34.72 | **46.00** |
| Acc@0.50 | 75.00 | 29.17 | **42.00** |

Interpretation: v3.7 is a mixed durability result, not the new headline.
It eliminates no-prediction failures on the cached random100 fold, but it
does not improve the metric headline over v3.3. The v3.6 row above is the
98-row completed-only aggregation-GT subset under
`tmp/v3p6_tadg_random100_eval_ignore2_agg_gt/`, included only to show that
the two newly completed v3.7 rows did not lift the metric.

## Hypothesis Tested

v3.6's two random100 failures were not Stage 1 coverage failures. Audits showed
the target proposal was visible or reachable, but Stage 2 either submitted
`proposal_id=-1` too early or selected a proposal inconsistent with the marked
frame it cited.

v3.7 adds final-selection pressure without using GT at inference:

- `NO_MATCH_GUARD` blocks `submit_final(proposal_id=-1)` while the agent's own
  tool trace still contains category candidates or viewed-but-uninspected marked
  proposals.
- The guard was tightened after the first cached rerun: repeated no-match is no
  longer accepted while category candidates remain.
- `EVIDENCE_FRAME_GUARD` blocks a final proposal when the rationale cites a
  marked frame where that proposal is absent.
- `view_keyframe_marked` now exposes `left_to_right` and `boxes_2d`, backed by
  ConceptGraph frame-view enrichment, so the guard can catch simple left/right
  conflicts in cited marked frames.
- Stage 2 direct structured responses are forced to continue when a guard
  blocked the corresponding `submit_final`.

## Commands

The official v3.7 artifact is a cached run: 98 existing v3.6 TADG-success
checkpoints were reused, and only the two failed samples were rerun. The cache
directory was created from `tmp/v3p6_tadg_random100_eval/`; the two failed
per-sample JSON files were removed before resuming.

Final runner command shape:

```bash
CVRA_DISABLE=1 PYTHONPATH=src .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p7_final_selection_random100_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 1 \
  --sample-retries 4 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard
```

Rerun history:

- `tmp/v3p7_final_selection_random100_eval.log`: initial cached resume; reran
  `scene0474_00::15::3` and `scene0678_00::34::3`, but `scene0678` still ended
  as failed no-match.
- `tmp/v3p7_final_selection_random100_eval_rerun_scene0678.log`: stricter
  no-match guard; reran only `scene0678`, but exhausted on a retryable
  `504/-4307`.
- `tmp/v3p7_final_selection_random100_eval_rerun_scene0678_b.log`: final
  scene0678 rerun with `--sample-retries 4`; completed and rewrote the 100-row
  `side_by_side.json`.

Aggregation-GT rescore:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/v3p7_final_selection_random100_eval/side_by_side.json \
  --output-dir tmp/v3p7_final_selection_random100_eval_agg_gt
```

Leaderboard slicing:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.scanrefer_leaderboard_metrics \
  --side-by-side tmp/v3p7_final_selection_random100_eval_agg_gt/side_by_side.json \
  --scanrefer-data-root data/scanrefer \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --output tmp/v3p7_final_selection_random100_eval_agg_gt/leaderboard_metrics.json
```

SQLite ingest:

```bash
.venv/bin/python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/v3p7_final_selection_random100_eval_agg_gt \
  --run-id v3p7_final_selection_random100_20260508 \
  --branch feat/scanrefer-v3p7-final-selection \
  --commit 8acd3da \
  --leaderboard-metrics tmp/v3p7_final_selection_random100_eval_agg_gt/leaderboard_metrics.json \
  --db docs/benchmark/scanrefer/runs.sqlite \
  --keyframe-mode query_driven \
  --notes "v3.7 final-selection cached random100: v3.6 TADG successes reused; failed no-match samples rerun with no-match guard, stricter no-match category blocking, evidence-frame guard, ConceptGraph frame_views, and left/right mark-geometry consistency; no GT used during inference; mixed result, failed_count eliminated but hard cases remain wrong"
```

## Raw Artifacts

- Agent output: `tmp/v3p7_final_selection_random100_eval/`
- Aggregation-GT rescore: `tmp/v3p7_final_selection_random100_eval_agg_gt/`
- Logs:
  - `tmp/v3p7_final_selection_random100_eval.log`
  - `tmp/v3p7_final_selection_random100_eval_rerun_scene0678.log`
  - `tmp/v3p7_final_selection_random100_eval_rerun_scene0678_b.log`
  - `tmp/v3p7_final_selection_random100_eval_agg_gt.log`
  - `tmp/v3p7_final_selection_random100_eval_agg_gt_leaderboard.log`
- Frozen fold: `tmp/scanrefer_artifacts/random100_sample_ids.json`
- SQLite DB: `docs/benchmark/scanrefer/runs.sqlite`

## Evidence

Aggregation-GT rescore:

```text
n=100 mean_iou=0.4024 acc25=0.4600 acc50=0.4200
stats: missing_scene=0 missing_target=0 no_prediction=0 completed_new_gt=100
```

Leaderboard slicing:

```text
n_total=100 (Unique=28, Multiple=72)
acc25  overall=0.4600 unique=0.7500 multiple=0.3472
acc50  overall=0.4200 unique=0.7500 multiple=0.2917
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
       printf('%.4f', acc50_multiple) AS m50
FROM runs
WHERE run_id IN (
  'v3p3_vertical_spatial_random100_20260504',
  'v3p7_final_selection_random100_20260508'
)
ORDER BY run_id;
```

Expected:

```text
v3p3_vertical_spatial_random100_20260504|100|0.4800|0.4200|0.4130|0.7500|0.7500|0.3750|0.2917
v3p7_final_selection_random100_20260508|100|0.4600|0.4200|0.4024|0.7500|0.7500|0.3472|0.2917
```

Tool trace durability: v3.7 ingested 2140 rows into `tool_calls`, and all
100 samples are `status='completed'` in SQLite.

## Hard Cases

The two original failed v3.6 samples are no longer no-prediction failures, but
both remain wrong after aggregation-GT rescoring:

| Sample | v3.7 selected | aggregation-GT IoU | Note |
|---|---:|---:|---|
| `scannet/scene0474_00::15::3` | 52 | 0.0000 | Focused r13 once selected pid 6 at IoU 0.3538, but the cached random100 rerun selected generic object 52. |
| `scannet/scene0678_00::34::3` | 7 | 0.0000 | Focused r13 once selected pid 10 at IoU 0.7851; final cached rerun selected pid 7 after exhausting evidence. |

This confirms that the guard mechanisms are useful but not stable enough to
claim a metric improvement.

## Caveats

- This is the frozen random100 development fold, not full 9508 val.
- This is a cached 98+2 process, not a fresh 100-sample rerun. It is valid as a
  final-selection durability check but should not be treated as a variance
  estimate.
- v3.7 does not use GT during inference. GT appears only in post-run IoU
  scoring and in the diagnosis above.
- v3.7 eliminates `proposal_id=-1` failures, but Acc@0.25 drops 2pp vs v3.3.
  v3.3 remains the honest query-driven headline on the random100 fold.
- The LLM is non-deterministic. Focused two-case r13 showed both target
  proposals are reachable, but the cached random100 rerun did not reproduce
  those correct final selections.
