# ScanRefer v3.2 - callback-image durability + aggregation-GT rescore

**Branch**: `feat/scanrefer-v3-query-driven`
**Tip commit at harvest time**: `c36c803` (working tree dirty; this run uses the
runtime and evaluator fixes listed below)
**Run ID** (SQLite): `v3p2_callbacks_durable_random100_20260504`
**Date harvested**: 2026-05-04 00:04 GMT+8

## Headline

Random100 fold, frozen seed=20260503:
`tmp/scanrefer_artifacts/random100_sample_ids.json`.

| Variant | GT used for IoU | Acc@0.25 | Acc@0.50 | mean IoU |
|---|---|---:|---:|---:|
| v3.1 smoke, original quoted number | Phase8 GT-CG bbox | 39.0% | 15.0% | 0.1981 |
| **v3.1 corrected baseline** | **ScanNet aggregation GT** | **45.0%** | **39.0%** | **0.3820** |
| **v3.2 callback-durable run** | **ScanNet aggregation GT** | **46.0%** | **40.0%** | **0.3857** |

Per-column leaderboard slices for v3.2:

| | Unique | Multiple | Overall |
|---|---:|---:|---:|
| Acc@0.25 | 71.43 | 36.11 | **46.00** |
| Acc@0.50 | 71.43 | 27.78 | **40.00** |

Verdict: the largest issue in the diagnosis doc was an evaluator/GT-source
mistake, not a true 39/15 model state. After rescoring against ScanNet
aggregation GT, v3.1 is 45/39. The v3.2 runtime fixes make a small real move
to 46/40 and add durable tool traces, but they do not close the v2 GT-view
oracle gap (v2 random100 was 67/59).

## What changed

Code-level fixes in the working tree used for this run:

- `view_keyframe_marked` and Stage 1 callback images are now injected before
  accepting a same-turn `submit_final`. The agent loop defers the finalizer
  when pending images were queued, appends the new visual evidence, and
  continues.
- The evidence-update prompt explicitly tells the model that a final answer
  submitted before seeing newly injected images was premature and was not
  accepted.
- ScanRefer callback selectors now load the visibility index at `stride=1`.
  The previous `stride=10` missed the local ScanRefer/NR3D visibility files.
- Raw keyframe path resolution now accepts local `*-rgb.png` frames in addition
  to the older `*-rgb.jpg` convention.
- The side-by-side extractor now resolves structured finalizer payloads with
  `proposal_id` through the stored `vg_proposal_pool` before declaring a sample
  failed.
- Per-sample `tool_trace` is written to checkpoints and side-by-side output,
  and `scripts/ingest_scanrefer_run.py` imports it into SQLite `tool_calls`.
- The ScanRefer leaderboard aggregator supports `--sample-ids`, so random100
  slicing is explicit instead of relying on full-val defaults.

## Commands

The pack was the existing random100 `pack_scanrefer_v3p_iterative` pack. The
agent run was resumed after targeted fixes using the same runner and output
directory:

```bash
PYTHONPATH=src python -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p_runtimefix_stride1_png_random100_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 4 \
  --sample-retries 1
```

After the extractor fix, the five failed `prediction missing status` checkpoints
were deleted and the same command was resumed with `--workers 2`, which reused
completed per-sample checkpoints and filled the missing predictions.

Paper-comparable rescore:

```bash
PYTHONPATH=src python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/v3p_runtimefix_stride1_png_random100_eval/side_by_side.json \
  --output-dir tmp/v3p_runtimefix_stride1_png_random100_eval_agg_gt \
  --scannet-aux-root data/nr3d/scannet_aux \
  --mesh-root data/nr3d/scannet_aux_meshes
```

Leaderboard slicing:

```bash
PYTHONPATH=src python -m evaluation.scripts.scanrefer_leaderboard_metrics \
  --side-by-side tmp/v3p_runtimefix_stride1_png_random100_eval_agg_gt/side_by_side.json \
  --scanrefer-data-root data/scanrefer \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --output tmp/v3p_runtimefix_stride1_png_random100_eval_agg_gt/leaderboard_metrics.json
```

SQLite ingest:

```bash
python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/v3p_runtimefix_stride1_png_random100_eval_agg_gt \
  --run-id v3p2_callbacks_durable_random100_20260504 \
  --branch feat/scanrefer-v3-query-driven \
  --commit c36c803 \
  --backend pack_v1 \
  --keyframe-mode mask3d_query_driven \
  --leaderboard-metrics tmp/v3p_runtimefix_stride1_png_random100_eval_agg_gt/leaderboard_metrics.json \
  --notes "v3.2 callback-image durability, stride=1 ScanRefer selector, PNG raw-frame support, structured proposal payload extraction; aggregation-GT random100 rescore"
```

## Raw artifacts

- Agent output: `tmp/v3p_runtimefix_stride1_png_random100_eval/`
- Aggregation-GT rescore: `tmp/v3p_runtimefix_stride1_png_random100_eval_agg_gt/`
- Main console log: `/tmp/v3p_logs/runtimefix_stride1_png_random100_agent.log`
- Frozen fold: `tmp/scanrefer_artifacts/random100_sample_ids.json`
- SQLite DB: `docs/benchmark/scanrefer/runs.sqlite`

## SQLite checks

```sql
SELECT run_id, n_total,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', mean_iou_overall) AS miou,
       keyframe_mode
FROM runs
WHERE run_id IN (
  'v3p_agg_gt_random100_20260503',
  'v3p2_callbacks_durable_random100_20260504'
)
ORDER BY run_id;
```

Expected:

```text
v3p2_callbacks_durable_random100_20260504|100|0.4600|0.4000|0.3857|mask3d_query_driven
v3p_agg_gt_random100_20260503|100|0.4500|0.3900|0.3820|mask3d_query_driven
```

Tool-trace durability:

```sql
SELECT run_id, count(*)
FROM tool_calls
WHERE run_id = 'v3p2_callbacks_durable_random100_20260504'
GROUP BY run_id;
```

Expected: `1899` tool calls.

## Caveats

- This is still a 100-utterance development fold, not the full 9508-val run.
- The improvement over corrected v3.1 is small (+1pp Acc@0.25, +1pp Acc@0.50).
  Treat the runtime fixes as correctness and observability fixes, not a solved
  ScanRefer ranking improvement.
- v2 remains a GT-view-oracle upper bound. It uses ScanNet aggregation GT for
  IoU but uses the GT target's visibility to choose initial keyframes.
- `llm_calls` is still empty for this ScanRefer run. `tool_calls` is now durable
  at the per-sample level.
- Two v3.2 samples still have no prediction after retries and count as zero.

## Interpretation

The ranking bottleneck remains the same after the evaluator correction: unique
objects are mostly fine (71.43/71.43), while multiple-object references stay at
36.11/27.78. The next high-leverage work should target multi-distractor
proposal ranking rather than more initial-keyframe coverage.
