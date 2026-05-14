# v8 Frame NMS Random100 - 2026-05-14

NR3D fixed random100 pilot for selector-level frame overlap NMS. This is a
partial internal ablation, not a public leaderboard row.

The matched pre-NMS ablation is included in this record because it is the
only fair way to isolate NMS: v8 reruns Stage1 query-driven keyframe selection
on rebuilt depth-aware visibility, so comparing v8 directly to v7 would mix
Stage1 rerun drift with the NMS decision itself.

## Run Identity

- Branch: `feat/nr3d-v4-agent-guards-fair-views`
- Run-time base commit: `101acaa` plus uncommitted frame-NMS implementation
  and docs changes recorded in this branch.
- Internal version: `v8_frame_nms_random100`
- Main run ID:
  `v8_frame_nms_random100_20260514`
- Matched ablation run ID:
  `v8_pre_nms_matched_random100_20260514`
- Stage 2 backend: `gpt-5.4-2026-03-05` via internal ModelHub
- Judge: none; scoring is programmatic target-id classification
- SQLite rows:
  - `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v8_frame_nms_random100_20260514'`
  - `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v8_pre_nms_matched_random100_20260514'`

## What Changed vs v7

- Added hard ordered frame NMS in `src/query_scene/frame_nms.py`.
- Wired selector flags:
  - `frame_nms`
  - `frame_nms_overlap_threshold`
  - `frame_nms_candidate_multiplier`
  - `frustum_method`
- NMS expands the Stage1 candidate pool to `k * candidate_multiplier`, then
  keeps views whose maximum symmetric frustum overlap is at or below the
  threshold. A relaxed backfill path fills the keyframe budget if strict NMS
  would return fewer than `k` views.
- `KeyframeSelector` now loads Phase8 raw-layout poses from
  `raw/scene_info.json::kept_frame_ids` and `raw/<frame>.txt` when
  `conceptgraph/traj.txt` is absent.
- NR3D pack prep exposes the NMS flags and records
  `keyframe_selection_metadata.frame_nms` in each sample artifact.
- Density fallback samples also pass through the same NMS logic, so every v8
  sample records comparable `frame_nms` metadata.
- Added visual spotcheck HTML:
  [frame_nms_spotcheck_20260514.html](frame_nms_spotcheck_20260514.html).

## Fold

- Split: NR3D `test`
- Fold size: 100 samples
- Selection file:
  `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json`
- Selection mechanism: same deterministic v4/v6/v7 random100 fold.
- Canonical filter after restriction: `n_full=100`, `n_filtered=100`

## Prepared Packs

- NMS pack:
  `data/nr3d/scannet/<scene>/pack_nr3d_v8_frame_nms_l1_t075/`
- Matched pre-NMS pack:
  `data/nr3d/scannet/<scene>/pack_nr3d_v8_pre_nms_l1_t075/`
- Pack audit after rerunning fallback samples:
  - samples: 100
  - missing `frame_nms`: 0
  - fallback samples: 24
  - fallback samples with NMS metadata: 24
  - samples where pre-NMS top3 differs from NMS selected top3: 72
  - suppressed candidates: 381
  - relaxed backfilled candidates: 26

The pre-NMS pack reuses the same proposals, visibility, and annotated frames
as the NMS pack. It changes only `keyframes` to
`frame_nms.pre_nms_keyframe_indices[:3]` for each sample.

## Raw Artifacts

- NMS eval output:
  `tmp/nr3d_eval_v8_frame_nms_random100_20260514/`
- NMS per-sample checkpoints:
  `tmp/nr3d_eval_v8_frame_nms_random100_20260514/per_sample/pack_nr3d_v8_frame_nms_l1_t075/*.json`
- NMS leaderboard metrics:
  `tmp/nr3d_eval_v8_frame_nms_random100_20260514/leaderboard_metrics.json`
- Pre-NMS eval output:
  `tmp/nr3d_eval_v8_pre_nms_random100_20260514/`
- Pre-NMS per-sample checkpoints:
  `tmp/nr3d_eval_v8_pre_nms_random100_20260514/per_sample/pack_nr3d_v8_pre_nms_l1_t075/*.json`
- Pre-NMS leaderboard metrics:
  `tmp/nr3d_eval_v8_pre_nms_random100_20260514/leaderboard_metrics.json`
- Logs:
  - `tmp/nr3d_eval_v8_frame_nms_random100_20260514_w100.log`
  - `tmp/nr3d_eval_v8_frame_nms_random100_20260514_assemble.log`
  - `tmp/nr3d_eval_v8_frame_nms_random100_20260514_metrics.log`
  - `tmp/nr3d_eval_v8_pre_nms_random100_20260514_w100.log`
  - `tmp/nr3d_eval_v8_pre_nms_random100_20260514_assemble.log`
  - `tmp/nr3d_eval_v8_pre_nms_random100_20260514_metrics.log`
- Pack-prep shard scripts:
  - `tmp/nr3d_artifacts/run_v8_frame_nms_random100_prep_shards.sh`
  - `tmp/nr3d_artifacts/run_v8_frame_nms_missing_nms_prep_shards.sh`
- Pack-prep shard definitions:
  - `tmp/nr3d_artifacts/v8_frame_nms_random100_scene_shards/`
  - `tmp/nr3d_artifacts/v8_frame_nms_random100_missing_nms_scene_shards/`

## Commands

Build the NMS pack. The first command built the 100-sample pack in four scene
shards; the second command reran the 29 samples that initially lacked NMS
metadata because they went through the density fallback path before that path
was patched.

```bash
tmux new-session -d -s nr3d_v8_frame_nms_prep_random100 \
  'cd /Users/bytedance/project/3DVLMReasoning && \
   ./scripts/run_with_rss_guard.sh --rss-limit-mb 30000 --check-interval-sec 60 -- \
   bash tmp/nr3d_artifacts/run_v8_frame_nms_random100_prep_shards.sh'

tmux new-session -d -s nr3d_v8_frame_nms_missing_nms_prep \
  'cd /Users/bytedance/project/3DVLMReasoning && \
   ./scripts/run_with_rss_guard.sh --rss-limit-mb 30000 --check-interval-sec 60 -- \
   bash tmp/nr3d_artifacts/run_v8_frame_nms_missing_nms_prep_shards.sh'
```

Build the matched pre-NMS ablation pack:

```bash
.venv/bin/python scripts/build_nr3d_pre_nms_ablation_pack.py \
  --data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --source-pack pack_nr3d_v8_frame_nms_l1_t075 \
  --dest-pack pack_nr3d_v8_pre_nms_l1_t075 \
  --asset-mode symlink \
  --force
```

Run the NMS Stage 2 checkpoints:

```bash
tmux new-session -d -s nr3d_v8_frame_nms_stage2_random100 \
  'cd /Users/bytedance/project/3DVLMReasoning && \
   ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- \
   env PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
   src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
   --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
   --data-root data/nr3d/scannet \
   --pack-name pack_nr3d_v8_frame_nms_l1_t075 \
   --output-dir tmp/nr3d_eval_v8_frame_nms_random100_20260514 \
   --workers 100 \
   --checkpoint-only \
   --sample-retries 2 \
   --use-tool-answer-disagreement-gate \
   --use-no-match-candidate-guard \
   --use-evidence-frame-guard \
   2>&1 | tee tmp/nr3d_eval_v8_frame_nms_random100_20260514_w100.log'
```

Run the matched pre-NMS Stage 2 checkpoints:

```bash
tmux new-session -d -s nr3d_v8_pre_nms_stage2_random100 \
  'cd /Users/bytedance/project/3DVLMReasoning && \
   ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- \
   env PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
   src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
   --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
   --data-root data/nr3d/scannet \
   --pack-name pack_nr3d_v8_pre_nms_l1_t075 \
   --output-dir tmp/nr3d_eval_v8_pre_nms_random100_20260514 \
   --workers 100 \
   --checkpoint-only \
   --sample-retries 2 \
   --use-tool-answer-disagreement-gate \
   --use-no-match-candidate-guard \
   --use-evidence-frame-guard \
   2>&1 | tee tmp/nr3d_eval_v8_pre_nms_random100_20260514_w100.log'
```

Assemble, score, and ingest NMS:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v8_frame_nms_l1_t075 \
  --output-dir tmp/nr3d_eval_v8_frame_nms_random100_20260514 \
  --workers 100 \
  --max-new-samples 0 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v8_frame_nms_random100_20260514/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --output tmp/nr3d_eval_v8_frame_nms_random100_20260514/leaderboard_metrics.json \
  --canonical-filter true

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v8_frame_nms_random100_20260514 \
  --run-id v8_frame_nms_random100_20260514 \
  --branch feat/nr3d-v4-agent-guards-fair-views \
  --commit 101acaa+dirty \
  --backend pack_v1 \
  --judge-model none \
  --leaderboard-metrics tmp/nr3d_eval_v8_frame_nms_random100_20260514/leaderboard_metrics.json \
  --notes "Depth-aware query-driven Stage1 rerun with hard frame NMS l1 threshold 0.75 on fixed random100; 100/100 completed."
```

Assemble, score, and ingest pre-NMS:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v8_pre_nms_l1_t075 \
  --output-dir tmp/nr3d_eval_v8_pre_nms_random100_20260514 \
  --workers 100 \
  --max-new-samples 0 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v8_pre_nms_random100_20260514/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --output tmp/nr3d_eval_v8_pre_nms_random100_20260514/leaderboard_metrics.json \
  --canonical-filter true

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v8_pre_nms_random100_20260514 \
  --run-id v8_pre_nms_matched_random100_20260514 \
  --branch feat/nr3d-v4-agent-guards-fair-views \
  --commit 101acaa+dirty \
  --backend pack_v1 \
  --judge-model none \
  --leaderboard-metrics tmp/nr3d_eval_v8_pre_nms_random100_20260514/leaderboard_metrics.json \
  --notes "Matched pre-NMS top3 ablation for v8 frame-NMS candidate pools on fixed random100; 100/100 completed."
```

## Metrics

### Leaderboard-Track Classification

| Metric | v7 callbacks no-NMS | v8 matched pre-NMS | v8 frame NMS | NMS delta vs pre-NMS | v8 NMS delta vs v7 |
|---|---:|---:|---:|---:|---:|
| Overall | 73.00 | 69.00 | 65.00 | -4.00 pp | -8.00 pp |
| Easy | 82.93 | 75.61 | 75.61 | +0.00 pp | -7.32 pp |
| Hard | 66.10 | 64.41 | 57.63 | -6.78 pp | -8.47 pp |
| View-Dep | 70.59 | 61.76 | 58.82 | -2.94 pp | -11.76 pp |
| View-Indep | 74.24 | 72.73 | 68.18 | -4.55 pp | -6.06 pp |

### IoU Proxy on GT Pool

| Metric | v8 matched pre-NMS | v8 frame NMS | Delta |
|---|---:|---:|---:|
| n | 100 | 100 | 0 |
| mean IoU | 0.6948 | 0.6551 | -0.0397 |
| Acc@0.25 | 0.6900 | 0.6500 | -0.0400 |
| Acc@0.50 | 0.6900 | 0.6500 | -0.0400 |
| failed sentinels | 0 | 0 | 0 |
| uncaught sample errors | 0 | 0 | 0 |

### Per-Sample Movement: Pre-NMS -> NMS

| Check | Count |
|---|---:|
| Wrong -> correct | 10 |
| Correct -> wrong | 14 |
| Net movement | -4 |

Wrong -> correct sample IDs:

- `scannet/scene0565_00::14::19228`
- `scannet/scene0578_00::10::5147`
- `scannet/scene0095_00::28::8465`
- `scannet/scene0644_00::40::39251`
- `scannet/scene0651_00::8::26207`
- `scannet/scene0490_00::14::39189`
- `scannet/scene0648_00::15::24119`
- `scannet/scene0629_00::6::19188`
- `scannet/scene0578_00::18::21381`
- `scannet/scene0618_00::26::28943`

Correct -> wrong sample IDs:

- `scannet/scene0643_00::6::10715`
- `scannet/scene0187_00::13::24713`
- `scannet/scene0300_00::8::22495`
- `scannet/scene0095_00::46::33663`
- `scannet/scene0025_00::32::23687`
- `scannet/scene0686_00::24::10791`
- `scannet/scene0474_00::12::7632`
- `scannet/scene0629_00::31::23523`
- `scannet/scene0574_00::5::13160`
- `scannet/scene0643_00::22::25584`
- `scannet/scene0645_00::34::34611`
- `scannet/scene0704_00::0::16937`
- `scannet/scene0435_00::60::25927`
- `scannet/scene0704_00::5::8434`

## Runtime Notes

- Both Stage 2 runs completed 100/100 checkpoints.
- Both runs used `workers=100` with the 15GB RSS guard and 60-second RSS check.
- No RSS guard trigger, traceback, uncaught sample error, or
  `callback is not configured` message was observed in either run.
- NMS run log:
  - retryable `429`: 96 lines
  - `Loading CLIP model`: 5 lines
- Pre-NMS run log:
  - retryable `429`: 66 lines
  - `Loading CLIP model`: 3 lines
- The CLIP loads come from crop/hypothesis text matching paths, not from the
  `request_more_views` object-term fallback disabled in v7.

## Interpretation

Hard frame NMS at `l1` overlap threshold `0.75` should **not** be enabled by
default for NR3D. On the matched candidate pool it drops overall accuracy from
69.00 to 65.00 and hurts especially on Hard and View-Indep subsets.

The likely failure mode is that the first three selector-ranked views often
carry redundant but useful close-up evidence. Removing one highly overlapping
frame can replace a discriminative close-up with a broader or off-angle frame;
the LLM then loses local details needed to choose among same-category objects.
This is visible in the net 14 correct-to-wrong losses versus 10 wrong-to-correct
gains.

The feature remains useful as an optional diagnostic and may still be worth
trying with softer rules, for example reranking redundant frames after visual
quality scoring instead of hard suppression, or applying NMS only when the
first `k` views are near-identical and object coverage is unchanged.

## Caveats

- This is a 100-sample internal ablation, not a full NR3D row.
- Direct v7 -> v8 comparison is not a pure NMS comparison. v7 used
  `pack_nr3d_v6_inline_labels_depth_visible` and preserved earlier keyframe
  frame IDs; v8 reruns Stage1 query-driven keyframe selection on depth-aware
  visibility. The pure NMS effect is the v8 pre-NMS -> v8 NMS comparison.
- The run-time commit is recorded as `101acaa+dirty` because the code changes
  were intentionally evaluated before being committed. The committed branch
  state after this doc contains the implementation and this record.
