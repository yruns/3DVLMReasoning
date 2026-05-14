# ScanRefer v3.21 - text-first policy random100 negative gate

**Branch**: `feat/nr3d-transcrib3d-first300`  
**Tip commit at harvest time**: `4ee6167`  
**Run ID** (SQLite): `v3p21_textfirst_random100_20260514`  
**Date harvested**: 2026-05-14 GMT+8

## Headline

v3.21 tests the v11 VG text-first policy on the frozen ScanRefer random100
development fold. The policy was designed to be more Transcrib3D-like: first
filter candidates using the Scene Proposal Inventory, category lookup,
proposal metadata, deterministic geometry ranking, and spatial comparison;
only then spend turns on marked frames or crops.

This is a **negative ScanRefer gate**. On the detector-pool random100 fold it
drops below the v3.11 single-run baseline.

| Split | Acc@0.25 | Acc@0.50 |
|---|---:|---:|
| Unique | 71.43 | 71.43 |
| Multiple | 36.11 | 25.00 |
| Overall | 46.00 | 38.00 |

Mean IoU is `0.3792`. Fold size is `100` (`Unique=28`, `Multiple=72`).

## What Changed

The ScanRefer pack and runtime guards stayed the same as the current
`pack_scanrefer_v3p_iterative` random100 path. The only intentional behavior
change is the v11 prompt/playbook policy:

- read the Scene Proposal Inventory before visual exploration;
- call `find_proposals_by_category` for the focal class;
- inspect candidate proposals before finalizing;
- use `rank_proposals_by_geometry` for pure size/height superlatives;
- use `compare_proposals_spatial` for anchor relations;
- require visual confirmation for ScanRefer/Mask3D detector-pool cases before
  final submission.

No consensus was used. This is a single agent trajectory.

## Commands

Fresh random100 run:

```bash
tmux new-session -d -s scanrefer_v3p21_textfirst_random100_20260514 \
  'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src CVRA_DISABLE=1 && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python -m evaluation.scripts.run_scanrefer_vg_side_by_side --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json --data-root data/scanrefer/scannet --output-dir tmp/scanrefer_v3p21_textfirst_random100_20260514_eval --pack-name pack_scanrefer_v3p_iterative --workers 8 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/scanrefer_v3p21_textfirst_random100_20260514_eval.log'
```

Assemble:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src CVRA_DISABLE=1 .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/scanrefer_v3p21_textfirst_random100_20260514_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 8 \
  --max-new-samples 0 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard
```

Aggregation-GT rescore:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/scanrefer_v3p21_textfirst_random100_20260514_eval/side_by_side.json \
  --output-dir tmp/scanrefer_v3p21_textfirst_random100_20260514_eval_agg_gt
```

Leaderboard slicing:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.scanrefer_leaderboard_metrics \
  --side-by-side tmp/scanrefer_v3p21_textfirst_random100_20260514_eval_agg_gt/side_by_side.json \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --output tmp/scanrefer_v3p21_textfirst_random100_20260514_eval_agg_gt/leaderboard_metrics.json
```

SQLite ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/scanrefer_v3p21_textfirst_random100_20260514_eval_agg_gt \
  --run-id v3p21_textfirst_random100_20260514 \
  --branch feat/nr3d-transcrib3d-first300 \
  --commit 4ee6167 \
  --backend pack_v1 \
  --judge-model none \
  --db docs/benchmark/scanrefer/runs.sqlite \
  --leaderboard-metrics tmp/scanrefer_v3p21_textfirst_random100_20260514_eval_agg_gt/leaderboard_metrics.json \
  --keyframe-mode mask3d_query_driven \
  --notes "v3.21 ScanRefer random100 single run with v11 VG text-first policy; pack_scanrefer_v3p_iterative; workers=8 under 15GB RSS guard; 100/100 completed; aggregation-GT rescored; CVRA disabled; TADG + no-match + evidence-frame guards."
```

## Raw Artifacts

- Frozen sample list: `tmp/scanrefer_artifacts/random100_sample_ids.json`
- Raw output: `tmp/scanrefer_v3p21_textfirst_random100_20260514_eval/`
- Aggregation-GT rescore:
  `tmp/scanrefer_v3p21_textfirst_random100_20260514_eval_agg_gt/`
- Leaderboard metrics:
  `tmp/scanrefer_v3p21_textfirst_random100_20260514_eval_agg_gt/leaderboard_metrics.json`
- Logs:
  - `tmp/scanrefer_v3p21_textfirst_random100_20260514_eval.log`
  - `tmp/scanrefer_v3p21_textfirst_random100_20260514_assemble.log`
  - `tmp/scanrefer_v3p21_textfirst_random100_20260514_agg_gt.log`
  - `tmp/scanrefer_v3p21_textfirst_random100_20260514_metrics.log`
- SQLite DB: `docs/benchmark/scanrefer/runs.sqlite`

SQLite note: `tool_calls.response_text` for this run is compacted to keep the
tracked database below the normal git single-file size risk. Tool names,
inputs, counts, run/sample rows, and metrics remain queryable; full raw traces
remain in the side-by-side JSON under the raw artifact directory above.

## Evidence

Aggregation-GT rescore:

```text
n=100 mean_iou=0.3792 acc25=0.4600 acc50=0.3800
stats: missing_scene=0 missing_target=0 no_prediction=0 completed_new_gt=100
```

Leaderboard metrics:

```text
n_total=100 (Unique=28, Multiple=72)
acc25  overall=0.4600 unique=0.7143 multiple=0.3611
acc50  overall=0.3800 unique=0.7143 multiple=0.2500
```

SQLite reproduction:

```sql
SELECT run_id, n_total, n_unique, n_multiple,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', mean_iou_overall) AS miou,
       keyframe_mode
FROM runs
WHERE run_id='v3p21_textfirst_random100_20260514';
```

Expected:

```text
v3p21_textfirst_random100_20260514|100|28|72|0.4600|0.3800|0.3792|mask3d_query_driven
```

## Comparison

| Run | Acc@0.25 | Acc@0.50 | Mean IoU | Notes |
|---|---:|---:|---:|---|
| v3.11 single run | 54.00 | 50.00 | 0.4615 | Best single-run random100 baseline |
| v3.19 consensus3 | 61.00 | 54.00 | 0.5080 | No-GT consensus over v3.10/v3.11/v3.13 |
| v3.21 text-first | 46.00 | 38.00 | 0.3792 | This run |

Paired drift vs v3.11:

| Threshold | v3.21 recoveries | v3.21 regressions | Net |
|---|---:|---:|---:|
| Acc@0.25 | 5 | 13 | -8 |
| Acc@0.50 | 1 | 13 | -12 |

Only one Acc@0.50 recovery over v3.11 was observed:

- `scannet/scene0100_00::26::0`

Common Acc@0.50 regressions from v3.11 include:

- `scannet/scene0050_00::10::4`
- `scannet/scene0025_00::9::0`
- `scannet/scene0629_00::3::3`
- `scannet/scene0377_00::3::1`
- `scannet/scene0050_00::14::0`
- `scannet/scene0574_00::24::3`
- `scannet/scene0660_00::4::4`
- `scannet/scene0645_00::19::3`
- `scannet/scene0565_00::12::2`
- `scannet/scene0535_00::1::0`

## Tool / Runtime Notes

| Tool | v3.11 calls | v3.21 calls |
|---|---:|---:|
| `inspect_proposal` | 732 | 566 |
| `view_keyframe_marked` | 458 | 325 |
| `find_proposals_by_category` | 250 | 273 |
| `compare_proposals_spatial` | 141 | 133 |
| `request_more_views` | 77 | 54 |
| `request_crops` | 73 | 80 |
| Total tool calls | 2474 | 2063 |

Log counts:

- completed checkpoints: 100/100
- retryable `429`: 0
- retryable `503`: 0
- `callback is not configured`: 0
- `rss_guard`: 0
- `Traceback`: 0
- `Loading CLIP model`: 29
- SQLite tool rows: 2063 rows; `response_text` compacted from about 25.8MB to
  about 1.1MB before `VACUUM`

The run did not hit the RSS guard, but the log reproduces the same crop-path
memory risk seen in NR3D v11 first300. Examples include `request_more_views`
or `request_crops` terms such as `cabinet 24` and `trash can 23`; those miss
string matching and fall through to `selector.find_objects()` CLIP fallback.

## Interpretation

The text-first policy does not transfer cleanly to ScanRefer's noisy Mask3D
detector pool. For NR3D, where the candidate pool is clean and category labels
are reliable object identities, structured metadata can be a strong first
signal. For ScanRefer, the proposal labels and large detector boxes are weak
priors; reducing visual/keyframe inspection and emphasizing metadata appears
to hurt multi-distractor selection.

Do not promote v3.21. Keep v3.11 as the best single-run random100 baseline and
v3.19/v3.20 as the consensus development/full-val headlines. The next
ScanRefer improvement should be detector-aware: explicit proposal-id crop
resolution, no CLIP fallback for numeric proposal terms, and co-visible visual
confirmation rather than stronger text-first bias.

## Caveats

- This is the frozen random100 development fold, not full 9508-val.
- It is a single-run negative result; no consensus was applied.
- The run used `workers=8`, while v3.11 used `workers=4`; concurrency should
  not change scoring logic but may contribute to normal LLM trajectory drift.
- The result is aggregation-GT rescored and paper-comparable within the local
  random100 development protocol.
