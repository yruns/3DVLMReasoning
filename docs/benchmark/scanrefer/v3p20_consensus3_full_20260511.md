# ScanRefer v3.20 - full-val three-run no-GT consensus

**Branch**: `feat/scanrefer-v3p10-stage-audit`
**Tip commit at harvest time**: `8acd3da` (working tree may be dirty; see
local git status for exact uncommitted doc/script changes)
**Run ID** (SQLite): `v3p20_consensus3_full_20260511`
**Date harvested**: 2026-05-11 GMT+8

## Headline

v3.20 is the full ScanRefer val follow-up to the v3.19 random100 consensus.
It uses `pack_scanrefer_v3p_iterative` with `mask3d_query_driven` keyframes
and runs three fresh no-GT source trajectories, then applies deterministic
proposal-id consensus after aggregation-GT rescoring.

| Split | Acc@0.25 | Acc@0.50 |
|---|---:|---:|
| Unique | 78.58 | 72.31 |
| Multiple | 46.71 | 41.09 |
| Overall | **55.36** | **49.57** |

Mean IoU is `0.4745`. Fold size: `9508` utterances
(`Unique=2582`, `Multiple=6926`).

## Source Runs and Consensus Lift

The headline is not a single-run number. It is the deterministic consensus over
three complete no-GT source trajectories:

| Run | n | Overall@0.25 | Overall@0.50 | mean IoU | Unique@0.25 | Unique@0.50 | Multiple@0.25 | Multiple@0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source_a | 9508 | 54.00 | 48.36 | 0.4633 | 77.65 | 71.15 | 45.18 | 39.86 |
| source_b | 9508 | 53.52 | 47.93 | 0.4587 | 77.77 | 71.65 | 44.48 | 39.08 |
| source_c | 9508 | 53.18 | 47.60 | 0.4559 | 76.61 | 70.64 | 44.44 | 39.01 |
| single-run mean | 9508 | 53.57 | 47.96 | 0.4593 | 77.34 | 71.15 | 44.70 | 39.32 |
| consensus3 | 9508 | **55.36** | **49.57** | **0.4745** | **78.58** | **72.31** | **46.71** | **41.09** |

Consensus therefore adds +1.80 / +1.61 pp over the single-run mean, and
+1.37 / +1.21 pp over the strongest individual source (`source_a`). Any
cost-normalized or one-shot zero-shot comparison should cite the source rows
alongside the consensus row.

## What Changed

- Full-val `pack_scanrefer_v3p_iterative` was regenerated from
  `tmp/scanrefer_artifacts/full_val_sample_ids.json` with scene-level queue
  workers and exact coverage verification.
- Each source run used the current no-GT v3 guard stack:
  `--use-tool-answer-disagreement-gate`, `--use-no-match-candidate-guard`, and
  `--use-evidence-frame-guard`; CVRA remained off.
- Source outputs were rescored against ScanNet aggregation GT before consensus.
- Consensus uses `scanrefer_consensus_side_by_side.py` over the three
  aggregation-GT source outputs.
- The full-run runner was hardened for memory:
  - per-sample checkpoints are the source of truth during agent batches;
  - `--checkpoint-only` and `--max-new-samples` allow bounded resume batches;
  - final `side_by_side.json` assembly can stream from checkpoints instead of
    keeping all trace-heavy records in memory;
  - the Stage 1 selector cache and ConceptGraph object cache are LRU-bounded to
    four scenes each.
- Post-processing was also made bounded:
  - `scanrefer_leaderboard_metrics.py` streams `per_sample` records and loads
    only the lightweight ScanRefer Unique/Multiple metadata;
  - `scanrefer_consensus_side_by_side.py` streams compact source records and
    drops full source `tool_trace` payloads from the consensus output.

## Commands

Full orchestration was driven from the local launcher
`tmp/run_scanrefer_v3p20_after_pack.sh`; the durable command shape is recorded
below because `tmp/` scripts are not permanent.

```bash
WORKERS_PER_SOURCE=96 \
SOURCE_BATCH_SIZE=96 \
SOURCE_PARALLELISM=2 \
RSS_LIMIT_KB=25165824 \
RSS_POLL_SECONDS=2 \
SYSTEM_FREE_MIN_PERCENT=35 \
./tmp/run_scanrefer_v3p20_after_pack.sh
```

Pack queue workers:

```bash
QUEUE_WORKERS=12 ./tmp/watch_scanrefer_v3p20_pack_queue.sh
```

Per-source agent batch command used by the launcher:

```bash
CVRA_DISABLE=1 PYTHONPATH=src .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/scanrefer_artifacts/full_val_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p20_full_<source>_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers "${WORKERS_PER_SOURCE}" \
  --sample-retries 4 \
  --max-new-samples "${SOURCE_BATCH_SIZE}" \
  --checkpoint-only \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard
```

Final successful completion used `WORKERS_PER_SOURCE=96`,
`SOURCE_BATCH_SIZE=96`, `SOURCE_PARALLELISM=2`, `RSS_LIMIT_KB=25165824`, and
`SYSTEM_FREE_MIN_PERCENT=35`. A short `112/112` trial produced dense 429 retry
warnings and was rolled back to `96/96`; earlier lower-RSS trials also hit the
RSS guard, which is why checkpoint-only batches and bounded caches were kept.

Consensus build embedded in the launcher:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.scanrefer_consensus_side_by_side \
  --input source_a=tmp/v3p20_full_source_a_eval_agg_gt/side_by_side.json \
  --input source_b=tmp/v3p20_full_source_b_eval_agg_gt/side_by_side.json \
  --input source_c=tmp/v3p20_full_source_c_eval_agg_gt/side_by_side.json \
  --output-dir tmp/v3p20_consensus3_full_eval
```

SQLite ingest embedded in the launcher:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/v3p20_consensus3_full_eval \
  --run-id v3p20_consensus3_full_20260511 \
  --branch feat/scanrefer-v3p10-stage-audit \
  --commit 8acd3da \
  --backend pack_v1 \
  --keyframe-mode mask3d_query_driven \
  --leaderboard-metrics tmp/v3p20_consensus3_full_eval/leaderboard_metrics.json
```

## Raw Artifacts

- Full sample list: `tmp/scanrefer_artifacts/full_val_sample_ids.json`
- Full pack chunks: `tmp/scanrefer_artifacts/full_val_scene_chunks/`
- Source A: `tmp/v3p20_full_source_a_eval_agg_gt/`
- Source B: `tmp/v3p20_full_source_b_eval_agg_gt/`
- Source C: `tmp/v3p20_full_source_c_eval_agg_gt/`
- Source metrics:
  `tmp/v3p20_full_source_{a,b,c}_eval_agg_gt/leaderboard_metrics.json`
- Consensus: `tmp/v3p20_consensus3_full_eval/`
- Leaderboard metrics: `tmp/v3p20_consensus3_full_eval/leaderboard_metrics.json`
- SQLite DB: `docs/benchmark/scanrefer/runs.sqlite`

## Evidence

SQLite reproduction:

```sql
SELECT run_id, n_total, n_unique, n_multiple,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', mean_iou_overall) AS miou,
       keyframe_mode
FROM runs
WHERE run_id = 'v3p20_consensus3_full_20260511';
```

Expected:

```text
v3p20_consensus3_full_20260511|9508|2582|6926|0.5536|0.4957|0.4745|mask3d_query_driven
```

Consensus choice sources:

```text
{'source_a': 3713, 'source_b': 3062, 'source_c': 2733}
```

Consensus group sizes:

```text
{'1': 1147, '2': 3068, '3': 5293}
```

Sampled artifact spot-checks with loaded image dimensions and mesh paths:

| Bucket | Sample | Selected pid | IoU | Evidence loaded | Query |
|---|---|---:|---:|---|---|
| success | `scannet/scene0011_00::5::3` | `3` | 1.0000 | data/scanrefer/scannet/scene0011_00/pack_scanrefer_v3p_iterative/annotated/frame_199.png (1296x968, frame=199); data/scanrefer/scannet/scene0011_00/pack_scanrefer_v3p_iterative/annotated/frame_197.png (1296x968, frame=197); data/scanrefer/scannet/scene0011_00/pack_scanrefer_v3p_iterative/annotated/frame_188.png (1296x968, frame=188); data/nr3d/scannet_aux_meshes/scene0011_00/scene0011_00_vh_clean_2.ply (exists) | there is a dark brown wooden and leather chair. placed in the table of the kitchen. |
| partial | `scannet/scene0011_00::20::2` | `67` | 0.3427 | data/scanrefer/scannet/scene0011_00/pack_scanrefer_v3p_iterative/annotated/frame_99.png (1296x968, frame=99); data/scanrefer/scannet/scene0011_00/pack_scanrefer_v3p_iterative/annotated/frame_73.png (1296x968, frame=73); data/scanrefer/scannet/scene0011_00/pack_scanrefer_v3p_iterative/annotated/frame_56.png (1296x968, frame=56); data/nr3d/scannet_aux_meshes/scene0011_00/scene0011_00_vh_clean_2.ply (exists) | this is a brown cabinet. it is above a refrigerator. |
| failure | `scannet/scene0011_00::13::3` | `8` | 0.0000 | data/scanrefer/scannet/scene0011_00/pack_scanrefer_v3p_iterative/annotated/frame_103.png (1296x968, frame=103); data/scanrefer/scannet/scene0011_00/pack_scanrefer_v3p_iterative/annotated/frame_71.png (1296x968, frame=71); data/scanrefer/scannet/scene0011_00/pack_scanrefer_v3p_iterative/annotated/frame_86.png (1296x968, frame=86); data/nr3d/scannet_aux_meshes/scene0011_00/scene0011_00_vh_clean_2.ply (exists) | there is a brown wooden chair. placed beside other chairs in the middle of the kitchen. |

Manual post-run audit loaded the annotated RGB evidence and the corresponding
axis-aligned ScanNet mesh for one success, one partial, and one failure:

| Bucket | Sample | Visual check | Mesh / bbox check | Decision |
|---|---|---|---|---|
| success | `scannet/scene0678_00::0::1` | `frame_57.png` shows the Pepsi vending machine between the snack vending machine and dryers; consensus selected pid `7` by 2/3 votes. | Loaded `data/nr3d/scannet_aux_meshes/scene0678_00/scene0678_00_vh_clean_2.ply`; after applying `axisAlignment`, predicted/GT boxes overlap at IoU `0.799769` with `8908` aligned mesh vertices inside both boxes. | Counted as Acc@0.50 success; visual and geometry evidence agree. |
| partial | `scannet/scene0208_00::9::2` | `frame_12.png` shows adjacent magazine-rack-like stands at the bookshelf end; all three sources selected pid `158`. | Loaded `data/nr3d/scannet_aux_meshes/scene0208_00/scene0208_00_vh_clean_2.ply`; aligned-mesh check gives IoU `0.348974`, center delta `0.2864m`, and `8090` vertices in the box overlap. | Correct coarse localization but oversized/shifted box; Acc@0.25 only. |
| failure | `scannet/scene0011_00::9::4` | `frame_80.png`/`frame_199.png` show multiple similar brown kitchen chairs; consensus selected pid `8` by 2/3 votes. | Loaded `data/nr3d/scannet_aux_meshes/scene0011_00/scene0011_00_vh_clean_2.ply`; selected pid `8` and GT pid `9` have IoU `0.0`, center delta `1.6767m`, and no overlapping aligned mesh vertices. | Real distractor failure, not a metric or file-load artifact. |

## Caveats

- v3.20 is still a consensus over three source trajectories, not a single-run
  agent architecture result.
- The selector and consensus policy do not read GT at inference. GT is used
  only for aggregation-GT rescoring and metric computation.
- This full-val run is directly more useful than v3.19's random100 result, but
  it should still be reported with the consensus caveat and the exact source
  trajectory settings above.
