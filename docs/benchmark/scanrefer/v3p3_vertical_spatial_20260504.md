# ScanRefer v3.3 - vertical spatial relations

**Branch**: `feat/scanrefer-v3-query-driven`
**Tip commit at harvest time**: `c36c803` (working tree dirty; v3.3 code change
not yet committed)
**Run ID** (SQLite): `v3p3_vertical_spatial_random100_20260504`
**Date harvested**: 2026-05-04 00:52 GMT+8

## Headline

Random100 fold, frozen seed=20260503:
`tmp/scanrefer_artifacts/random100_sample_ids.json`.

| Variant | Acc@0.25 | Acc@0.50 | mean IoU |
|---|---:|---:|---:|
| v3.1 corrected aggregation-GT baseline | 45.0% | 39.0% | 0.3820 |
| v3.2 callback-durable | 46.0% | 40.0% | 0.3857 |
| **v3.3 vertical spatial** | **48.0%** | **42.0%** | **0.4130** |

Per-column leaderboard slices:

| | Unique | Multiple | Overall |
|---|---:|---:|---:|
| Acc@0.25 | 75.00 | 37.50 | **48.00** |
| Acc@0.50 | 75.00 | 29.17 | **42.00** |

Net vs v3.2: +2pp Acc@0.25, +2pp Acc@0.50, +0.0273 mean IoU.

## Hypothesis Tested

Several random100 failures used vertical language such as "above" or "below",
but the only geometric tool relation was `closest_to` / `farthest_from`. In
`scene0011_00::20::2`, for example, the agent interpreted "cabinet above a
refrigerator" through `closest_to`, so the tool ranked the cabinet closest to
the refrigerator rather than the cabinet with positive z offset.

v3.3 adds `relation="above"` and `relation="below"` to
`compare_proposals_spatial`, ranks by bbox-center z offset relative to the
anchor, and updates the VG spatial skill docs so the model can use these
relations instead of forcing vertical language through `closest_to`.

## Commands

Agent run, first pass:

```bash
PYTHONPATH=src python -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p3_vertical_spatial_random100_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 4 \
  --sample-retries 1
```

The first tmux run exited after 75/100 checkpoints without writing
`side_by_side.json`; the same command was resumed against the same output
directory and reused the completed checkpoints. The resumed run reached 100/100
and wrote `side_by_side.json`.

Aggregation-GT rescore:

```bash
PYTHONPATH=src python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/v3p3_vertical_spatial_random100_eval/side_by_side.json \
  --output-dir tmp/v3p3_vertical_spatial_random100_eval_agg_gt \
  --scannet-aux-root data/nr3d/scannet_aux \
  --mesh-root data/nr3d/scannet_aux_meshes
```

Leaderboard slicing:

```bash
PYTHONPATH=src python -m evaluation.scripts.scanrefer_leaderboard_metrics \
  --side-by-side tmp/v3p3_vertical_spatial_random100_eval_agg_gt/side_by_side.json \
  --scanrefer-data-root data/scanrefer \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --output tmp/v3p3_vertical_spatial_random100_eval_agg_gt/leaderboard_metrics.json
```

SQLite ingest:

```bash
.venv/bin/python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/v3p3_vertical_spatial_random100_eval_agg_gt \
  --run-id v3p3_vertical_spatial_random100_20260504 \
  --branch feat/scanrefer-v3-query-driven \
  --commit c36c803 \
  --backend pack_v1 \
  --keyframe-mode mask3d_query_driven \
  --leaderboard-metrics tmp/v3p3_vertical_spatial_random100_eval_agg_gt/leaderboard_metrics.json \
  --notes "v3.3 vertical spatial relations in compare_proposals_spatial (above/below), updated VG spatial skills; aggregation-GT random100 rescore"
```

## Raw Artifacts

- Agent output: `tmp/v3p3_vertical_spatial_random100_eval/`
- Aggregation-GT rescore: `tmp/v3p3_vertical_spatial_random100_eval_agg_gt/`
- First-pass log: `/tmp/v3p_logs/v3p3_vertical_spatial_random100_agent.log`
- Resume log: `/tmp/v3p_logs/v3p3_vertical_spatial_random100_resume.log`
- Frozen fold: `tmp/scanrefer_artifacts/random100_sample_ids.json`
- SQLite DB: `docs/benchmark/scanrefer/runs.sqlite`

## Evidence

Rescore output:

```text
n=100 mean_iou=0.4130 acc25=0.4800 acc50=0.4200
stats: missing_scene=0 missing_target=0 no_prediction=5 completed_new_gt=95
```

SQLite:

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
  'v3p2_callbacks_durable_random100_20260504',
  'v3p3_vertical_spatial_random100_20260504'
)
ORDER BY run_id;
```

Expected:

```text
v3p2_callbacks_durable_random100_20260504|100|0.4600|0.4000|0.3857|0.7143|0.7143|0.3611|0.2778
v3p3_vertical_spatial_random100_20260504|100|0.4800|0.4200|0.4130|0.7500|0.7500|0.3750|0.2917
```

Tool trace durability: v3.3 ingested 1891 rows into `tool_calls`.

Vertical relation use in v3.3 tool traces:

```text
closest_to: 50
farthest_from: 8
above: 4
below: 2
```

Acc@0.50 movements vs v3.2:

- Gains: 7 samples crossed from `<0.50` to `>=0.50`.
- Losses: 5 samples crossed from `>=0.50` to `<0.50`.
- Net: +2 Acc@0.50 samples.

Not all vertical calls were beneficial. The original `scene0011_00::20::2`
"cabinet above refrigerator" case switched to the `above` relation and selected
proposal 41, but aggregation-GT IoU became 0.0 instead of v3.2's 0.343. The
overall gain came from the broader rerun plus the new tool affordance, not from
that single motivating case.

## Caveats

- This is still the frozen random100 development fold, not full 9508 val.
- The LLM is non-deterministic; v3.3 is one measured run, not a variance study.
- v3.3 has 5 no-prediction samples after retries, worse than v3.2's 2, yet the
  aggregate metric still improved.
- The remaining gap is still multi-distractor ranking. Multiple Acc@0.50 is
  29.17, far below the v2 GT-view-oracle random100 upper bound.
