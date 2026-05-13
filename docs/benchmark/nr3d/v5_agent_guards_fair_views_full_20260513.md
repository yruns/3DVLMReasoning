# v5 Agent Guards + Fair Views Full - 2026-05-13

> Superseded for fair-view headline reporting by
> [v5p1_failed_rerun_full_20260513.md](v5p1_failed_rerun_full_20260513.md),
> which reruns all 241 failed v5 sentinels and raises the filtered overall
> from 66.53 to 68.48.

Full NR3D test run using the v4 fair-view evidence path and ScanRefer-derived
agent guard stack. This is the first full-test NR3D row without the historical
GT-target-visible keyframe shortcut.

## Run Identity

- Branch: `feat/nr3d-v4-agent-guards-fair-views`
- Tip commit at run time: `7c996ad`
- Working tree: `/Users/bytedance/project/3DVLMReasoning` (Mac, `.venv`, Python 3.12)
- Internal version: `v5_agent_guards_fair_views_full`
- Run ID: `v5_agent_guards_fair_views_full_20260513`
- Stage 1 keyframe parser model: `gemini-2.5-pro`
- Stage 2 backend: `gpt-5.4-2026-03-05` via internal ModelHub
- Judge: none; scoring is programmatic
- SQLite row: `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v5_agent_guards_fair_views_full_20260513'`

## What Changed vs v4

v5 scales the v4 random100 design to the full NR3D test set:

- Full fold: 8584 NR3D test utterances from
  `tmp/nr3d_artifacts/full_test_sample_ids.json`.
- Query-driven keyframes for every sample, with non-GT scene-density fallback.
- Stage 2 guard flags enabled:
  - `--use-tool-answer-disagreement-gate`
  - `--use-no-match-candidate-guard`
  - `--use-evidence-frame-guard`
- Prep was ramped through 30/40/60/100 scene workers under a 30GB total RSS
  monitor, then fixed the remaining 30 samples with 30 one-sample workers.
- Stage 2 used `--workers 100`, `--checkpoint-only`, `--sample-retries 2`,
  and a 15GB RSS guard.

The candidate pool remains the NR3D ScanNet GT bbox pool. The target id and GT
bbox are used for scoring only, not for keyframe selection.

## Fold

- Split: NR3D `test`
- Full fold size: 8584 samples
- Canonical filtered size: 7805 samples
- Selection file: `tmp/nr3d_artifacts/full_test_sample_ids.json`
- Selection mechanism: complete local NR3D test split with Phase 8 GT-CG
  bbox availability.
- Candidate scenes: 130

## Fair-View Prep Audit

Audit artifact: `tmp/nr3d_artifacts/full_v4_pack_audit_20260513.json`

| Check | Value |
|---|---:|
| Prepared samples | 8584 |
| Unique sample ids | 8584 |
| Missing sample artifacts | 0 |
| Problem count | 0 |
| `keyframe_mode=query_driven` | 8584 |
| `keyframe_selection_uses_gt_target=false` | 8584 |
| Non-GT fallback used | 2212 |
| Samples with 3 keyframes | 8582 |
| Samples with 2 keyframes | 2 |

The 2212 fallback samples used the non-GT scene-density fallback path after
query-driven selection failed to produce executable evidence. They did not use
`target_id`, GT bbox, or target visibility.

## Raw Artifacts

- Prepared pack:
  `data/nr3d/scannet/<scene>/pack_nr3d_v4_agent_guards_fair_views/`
- Eval output:
  `tmp/nr3d_eval_v4_agent_guards_fair_views_full/`
- Per-sample checkpoints:
  `tmp/nr3d_eval_v4_agent_guards_fair_views_full/per_sample/pack_nr3d_v4_agent_guards_fair_views/*.json`
- Side-by-side output:
  `tmp/nr3d_eval_v4_agent_guards_fair_views_full/side_by_side.json`
- Leaderboard metrics:
  `tmp/nr3d_eval_v4_agent_guards_fair_views_full/leaderboard_metrics.json`
- Checkpoint audit:
  `tmp/nr3d_eval_v4_agent_guards_fair_views_full/checkpoint_audit_20260513.json`
- Stage 2 log:
  `tmp/nr3d_eval_v4_agent_guards_fair_views_full_w100.log`
- Prep RSS logs:
  - `tmp/nr3d_v4_prep_total_rss_guard_w30.log`
  - `tmp/nr3d_v4_prep_total_rss_guard_w40.log`
  - `tmp/nr3d_v4_prep_total_rss_guard_w60.log`
  - `tmp/nr3d_v4_prep_total_rss_guard_w100.log`
  - `tmp/nr3d_v4_prep_total_rss_guard_fix30.log`
- Post-run lightweight cache log:
  `tmp/nr3d_light_cache_20260513.log`

## Commands

Prepare full query-driven packs. The run was split across scene-worker batches
to keep the workstation stable while maximizing concurrency:

```bash
source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH=src

python src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
  --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v4_agent_guards_fair_views \
  --split test \
  --keyframe-mode query_driven \
  --keyframe-llm-model gemini-2.5-pro \
  --max-selector-cache-size 1 \
  --max-scene-artifact-cache-size 1
```

Run Stage 2 full sweep:

```bash
tmux new-session -d -s nr3d_v4_stage2_full_w100 'cd /Users/bytedance/project/3DVLMReasoning && \
  source .venv/bin/activate && \
  export PYTHONUNBUFFERED=1 PYTHONPATH=src && \
  ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- \
    python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
      --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
      --data-root data/nr3d/scannet \
      --pack-name pack_nr3d_v4_agent_guards_fair_views \
      --output-dir tmp/nr3d_eval_v4_agent_guards_fair_views_full \
      --workers 100 \
      --checkpoint-only \
      --sample-retries 2 \
      --use-tool-answer-disagreement-gate \
      --use-no-match-candidate-guard \
      --use-evidence-frame-guard \
    2>&1 | tee tmp/nr3d_eval_v4_agent_guards_fair_views_full_w100.log'
```

Assemble and score without re-running samples:

```bash
PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v4_agent_guards_fair_views \
  --output-dir tmp/nr3d_eval_v4_agent_guards_fair_views_full \
  --workers 100 \
  --max-new-samples 0 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v4_agent_guards_fair_views_full/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
  --output tmp/nr3d_eval_v4_agent_guards_fair_views_full/leaderboard_metrics.json \
  --canonical-filter true
```

Ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v4_agent_guards_fair_views_full \
  --run-id v5_agent_guards_fair_views_full_20260513 \
  --branch feat/nr3d-v4-agent-guards-fair-views \
  --commit 7c996ad \
  --backend pack_v1 \
  --judge-model none \
  --notes "full NR3D v5/v4-code fair-view run; query-driven keyframes; TADG + no-match + evidence-frame guards; prep max 100 workers with 30GB total guard; Stage2 workers=100 with 15GB RSS guard; 8584 checkpoints, 241 failed sentinels (2.81%), no rss_guard/traceback" \
  --leaderboard-metrics tmp/nr3d_eval_v4_agent_guards_fair_views_full/leaderboard_metrics.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

Post-run memory hardening:

```bash
tmux new-session -d -s nr3d_light_cache 'cd /Users/bytedance/project/3DVLMReasoning && \
  source .venv/bin/activate && \
  export PYTHONUNBUFFERED=1 PYTHONPATH=src && \
  ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- \
    python src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
      --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
      --data-root data/nr3d/scannet \
      --build-lightweight-cache-only \
    2>&1 | tee tmp/nr3d_light_cache_20260513.log'
```

Future high-concurrency prep should use `--ensure-lightweight-cache`. Missing
sidecar caches are built under a cross-process lock and the sample continues;
cache absence is not allowed to change the final benchmark set.

## Metrics

### Leaderboard-Track Classification

| Metric | v3 GT-visible full | v5 fair-view full | Delta |
|---|---:|---:|---:|
| Overall | 80.79 | 66.53 | -14.26 pp |
| Easy | 86.06 | 76.09 | -9.97 pp |
| Hard | 75.87 | 57.59 | -18.28 pp |
| View-Dep | 72.46 | 55.74 | -16.72 pp |
| View-Indep | 85.34 | 72.41 | -12.93 pp |

### Full-Fold IoU Proxy on GT Pool

| Metric | Value |
|---|---:|
| n_full | 8584 |
| n_filtered | 7805 |
| classification_acc_full | 63.54 |
| classification_acc_filtered | 66.53 |
| mean IoU | 0.6391 |
| Acc@0.25 | 63.56 |
| Acc@0.50 | 63.54 |

### Completion and Service Health

| Check | Value |
|---|---:|
| Per-sample checkpoints | 8584 |
| Completed checkpoints | 8343 |
| Failed sentinels | 241 |
| Final failure rate | 2.81% |
| Records with `error` | 235 |
| Log `Traceback` count | 0 |
| Log `[rss_guard]` count | 0 |
| Retryable 429 warnings | 30997 |
| Retryable 503 warnings | 174 |
| Transport failures logged | 255 |

Failed sentinel breakdown:

| Error type | Count |
|---|---:|
| `BadRequestError` | 232 |
| `InternalServerError` | 1 |
| `APIConnectionError` | 2 |
| Failed without `error` field | 6 |

## Throughput and Memory

Stage 2 ran from 2026-05-13 03:28:51 to 10:53:30 CST, about 7h25m for
8584 samples at 100 workers. Thirty-minute progress checks were used during
the run. The largest observed Stage 2 process RSS checkpoint was about 2.68GB,
well below the 15GB guard.

Prep initially exposed the real high-concurrency memory risk. The raw
ConceptGraph pkl files are small on disk but contain huge per-detection `mask`
arrays:

| Scene | Compressed pkl | Expanded `mask` bytes | Useful arrays |
|---|---:|---:|---:|
| `scene0231_00` | 23.6MB | 7732.4MB | 17.6MB |
| `scene0645_00` | 21.2MB | 7161.7MB | 15.3MB |
| `scene0208_00` | 19.4MB | 5814.6MB | 18.0MB |
| `scene0030_00` | 17.1MB | 5678.2MB | 12.9MB |
| `scene0653_00` | 16.7MB | 5198.4MB | 12.4MB |

Across local scenes, the expanded mask payload is about 258.9GB. The prep code
was only using bbox/class/centroid/CLIP metadata, but `pickle.load()` had to
materialize the entire object, including masks, before the unused fields could
be ignored. High process concurrency therefore multiplied the temporary mask
peak.

Post-run hardening in this branch adds stripped
`full_pcd_gt_axisaligned_post.light.pkl.gz` sidecar caches and a
`--ensure-lightweight-cache` path. Missing sidecars are auto-built under a
cross-process lock instead of failing a sample. For the largest measured scene,
`KeyframeSelector.from_scene_path(..., ensure_lightweight_pcd=True)` dropped
from about 8.35GB max RSS to about 110MB max RSS. The full test split now has
130 lightweight sidecar caches totaling about 31MB.

## SQLite Reproduction Query

```sql
SELECT run_id, n, n_filtered,
       printf('%.4f', classification_acc_filtered) AS overall,
       printf('%.4f', acc_easy) AS easy,
       printf('%.4f', acc_hard) AS hard,
       printf('%.4f', acc_view_dep) AS vdep,
       printf('%.4f', acc_view_indep) AS vind
FROM runs
WHERE run_id='v5_agent_guards_fair_views_full_20260513';
-- v5_agent_guards_fair_views_full_20260513|8584|7805|0.6653|0.7609|0.5759|0.5574|0.7241
```

## Interpretation

v5 is lower than the historical v3 full row because v3 inherited the older
GT-target-visible keyframe selection path. v5 removes that shortcut and is the
cleaner fair-view diagnostic, but it is not the highest historical number.

The main regression is on Hard and View-Dep splits, which is consistent with
the harder evidence-selection problem after removing target-visible oracle
views. The Stage 2 guard stack itself stayed operational: the final failed
sentinel rate was 2.81%, far below the 25% threshold used for concurrency
ramping, and no RSS guard or traceback was recorded in the full Stage 2 log.
