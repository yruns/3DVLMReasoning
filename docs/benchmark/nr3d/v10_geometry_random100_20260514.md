# v10 Geometry Ranking Random100 - 2026-05-14

NR3D random100 rerun with the v10 VG tool surface. This run isolates the
deterministic geometry-ranking helper on the same fixed depth-aware random100
fold and pack used by v7.1 and v9.

## Run Identity

- Branch: `feat/nr3d-transcrib3d-first300`
- Tip commit at run time: `5b53037`
- Internal version: `v10_geometry_random100`
- Run ID: `v10_geometry_random100_20260514`
- Stage 2 backend: `gpt-5.4-2026-03-05` via internal ModelHub
- Judge: none; scoring is programmatic target-id classification
- SQLite row:
  `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v10_geometry_random100_20260514'`

## Fold And Pack

- Split: NR3D `test`
- Fold size: 100 samples
- Selection file:
  `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json`
- Prepared pack:
  `data/nr3d/scannet/<scene>/pack_nr3d_v6_inline_labels_depth_visible/`
- Canonical filter after restriction: `n_full=100`, `n_filtered=100`
- Evidence setup:
  - depth-aware `visibility_index.pkl`
  - rerendered marked frames with inline `#proposal_id category` labels
  - preserved v4/v6/v7/v9 keyframe frame IDs
  - no selector frame-overlap NMS

## What Changed From v9

v10 keeps the v9 compact `Scene Proposal Inventory` prompt prior and adds one
deterministic VG helper:

- `rank_proposals_by_geometry(candidate_ids, criterion)`
- supported criteria:
  `largest`, `smallest`, `tallest`, `shortest`, `highest`, `lowest`,
  `widest`, `narrowest`
- returned evidence:
  ranked ids, volumes, footprint areas, heights, center-z values, sizes,
  centers, and per-proposal geometry rows

The VG playbook now asks the agent to use this helper for same-category
size/height superlatives after it builds a category-compatible candidate set.
`chassis_tools_version` was bumped to `20`.

No pack, fold, visibility, callback wiring, or final-decision guard was changed
for this run.

## Raw Artifacts

- Eval output:
  `tmp/nr3d_eval_v10_geometry_random100_20260514/`
- Per-sample checkpoints:
  `tmp/nr3d_eval_v10_geometry_random100_20260514/per_sample/pack_nr3d_v6_inline_labels_depth_visible/*.json`
- Failed sentinel backups:
  - `tmp/nr3d_eval_v10_geometry_random100_20260514/failed_invalid_prompt_attempt1/`
  - `tmp/nr3d_eval_v10_geometry_random100_20260514/failed_invalid_prompt_attempt2/`
- Rerun sample ids:
  - `tmp/nr3d_artifacts/v10_geometry_random100_remaining_after_rss_guard_20260514.json`
  - `tmp/nr3d_artifacts/v10_geometry_random100_failed_invalid_prompt_attempt2_20260514.json`
- Side-by-side output:
  `tmp/nr3d_eval_v10_geometry_random100_20260514/side_by_side.json`
- Leaderboard metrics:
  `tmp/nr3d_eval_v10_geometry_random100_20260514/leaderboard_metrics.json`
- Comparison:
  `tmp/nr3d_eval_v10_geometry_random100_20260514/comparison_vs_v7p1_v9.json`
- Logs:
  - `tmp/nr3d_eval_v10_geometry_random100_20260514_w100.log`
  - `tmp/nr3d_eval_v10_geometry_random100_remaining89_w50_20260514.log`
  - `tmp/nr3d_eval_v10_geometry_random100_failed3_20260514.log`
  - `tmp/nr3d_eval_v10_geometry_random100_20260514_assemble.log`
  - `tmp/nr3d_eval_v10_geometry_random100_20260514_metrics.log`

## Commands

Initial 100-worker checkpoint pass:

```bash
tmux new-session -d -s nr3d_v10_geometry_random100_20260514 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v6_inline_labels_depth_visible --output-dir tmp/nr3d_eval_v10_geometry_random100_20260514 --workers 100 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v10_geometry_random100_20260514_w100.log'
```

The 100-worker pass hit the 15GB RSS guard after 12 checkpoints
(11 completed, 1 `invalid_prompt` sentinel). The sentinel was moved aside and
the remaining 89 samples were rerun at `workers=50`:

```bash
tmux new-session -d -s nr3d_v10_geometry_random100_remaining89_w50_20260514 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v10_geometry_random100_remaining_after_rss_guard_20260514.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v6_inline_labels_depth_visible --output-dir tmp/nr3d_eval_v10_geometry_random100_20260514 --workers 50 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v10_geometry_random100_remaining89_w50_20260514.log'
```

The 50-worker pass produced 97 completed checkpoints and 3 remaining
`invalid_prompt` sentinels. Those three were moved aside and rerun:

```bash
tmux new-session -d -s nr3d_v10_geometry_random100_failed3_20260514 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v10_geometry_random100_failed_invalid_prompt_attempt2_20260514.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v6_inline_labels_depth_visible --output-dir tmp/nr3d_eval_v10_geometry_random100_20260514 --workers 3 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v10_geometry_random100_failed3_20260514.log'
```

Assemble and score:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v6_inline_labels_depth_visible \
  --output-dir tmp/nr3d_eval_v10_geometry_random100_20260514 \
  --workers 50 \
  --max-new-samples 0 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v10_geometry_random100_20260514/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --output tmp/nr3d_eval_v10_geometry_random100_20260514/leaderboard_metrics.json \
  --canonical-filter true
```

Ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v10_geometry_random100_20260514 \
  --run-id v10_geometry_random100_20260514 \
  --branch feat/nr3d-transcrib3d-first300 \
  --commit 5b53037 \
  --backend pack_v1 \
  --judge-model none \
  --notes "v10 geometry-ranking tool rerun on same v4/v6/v7/v9 fixed random100 fold and pack_nr3d_v6_inline_labels_depth_visible; initial workers=100 hit 15GB RSS guard after 12 checkpoints because concurrent request_crops triggered CLIP loads; preserved completed checkpoints, moved invalid_prompt sentinels aside, completed remaining 89 with workers=50 and final 3 failed sentinels with workers=3; final 100/100 completed; TADG + no-match + evidence-frame guards; no callback-not-configured; 69/100 overall" \
  --leaderboard-metrics tmp/nr3d_eval_v10_geometry_random100_20260514/leaderboard_metrics.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

## Metrics

### Leaderboard-Track Classification

| Metric | v7.1 rerun | v9 inventory | v10 geometry | Delta vs v9 |
|---|---:|---:|---:|---:|
| Overall | 71.00 | 71.00 | 69.00 | -2.00 pp |
| Easy | 82.93 | 85.37 | 85.37 | +0.00 pp |
| Hard | 62.71 | 61.02 | 57.63 | -3.39 pp |
| View-Dep | 61.76 | 61.76 | 50.00 | -11.76 pp |
| View-Indep | 75.76 | 75.76 | 78.79 | +3.03 pp |

### IoU Proxy on GT Pool

| Metric | v9 inventory | v10 geometry | Delta |
|---|---:|---:|---:|
| n | 100 | 100 | 0 |
| mean IoU | 0.7123 | 0.6935 | -0.0188 |
| Acc@0.25 | 0.7100 | 0.6900 | -0.0200 |
| Acc@0.50 | 0.7100 | 0.6900 | -0.0200 |
| final failed sentinels | 0 | 0 | 0 |

### Per-Sample Drift

| Comparison | v10-only correct | Baseline-only correct | Both correct | Both wrong | Net |
|---|---:|---:|---:|---:|---:|
| vs v7.1 | 5 | 7 | 64 | 24 | -2 |
| vs v9 | 6 | 8 | 63 | 23 | -2 |

v10 improvements over v9:

- `scannet/scene0249_00::36::39076`
- `scannet/scene0500_00::27::34282`
- `scannet/scene0565_00::20::35581`
- `scannet/scene0629_00::6::19188`
- `scannet/scene0697_00::19::35451`
- `scannet/scene0704_00::5::8434`

v10 regressions from v9:

- `scannet/scene0025_00::32::23687`
- `scannet/scene0300_00::8::22495`
- `scannet/scene0474_00::12::7632`
- `scannet/scene0565_00::14::19228`
- `scannet/scene0578_00::10::5147`
- `scannet/scene0618_00::26::28943`
- `scannet/scene0643_00::22::25584`
- `scannet/scene0652_00::13::34162`

Slice counts on the same 100 samples:

| Slice | n | v7.1 correct | v9 correct | v10 correct |
|---|---:|---:|---:|---:|
| Easy | 41 | 34 | 35 | 35 |
| Hard | 59 | 37 | 36 | 34 |
| View-Dep | 34 | 21 | 21 | 17 |
| View-Indep | 66 | 50 | 50 | 52 |

## Runtime Notes

- Initial 100-worker pass:
  - 11 completed checkpoints
  - 1 `invalid_prompt` sentinel
  - RSS guard fired at `15744MB > 15000MB`
- 50-worker remaining pass:
  - recovered to 97 completed checkpoints
  - 3 `invalid_prompt` sentinels
  - no RSS guard
- 3-worker failed rerun:
  - final checkpoint audit: 100/100 completed, 0 failed sentinels
- Combined log counts:
  - retryable `429`: 99
  - retryable `503`: 0
  - attempts at 3/5 or higher: 0
  - `callback is not configured`: 0
  - `rss_guard`: 1
  - `Traceback`: 0
  - `Loading CLIP model`: 8
- Tool trace counts:
  - `rank_proposals_by_geometry`: 14 calls across 14 samples
  - rank-tool subset accuracy: 11/14 = 78.57
  - non-rank subset accuracy: 58/86 = 67.44

The 100-worker memory trip is not an LLM failure-rate issue. The log shows
several concurrent `request_crops` calls entering `selector.find_objects()`,
which lazily loads CLIP models in multiple workers. `workers=50` completed the
remaining 89 samples under the same 15GB guard without another RSS trip.

## Interpretation

v10 geometry ranking helped some geometry/superlative cases and the rank-tool
subset itself was relatively strong, but it did not improve the fixed
random100 headline. The regression is concentrated in harder and view-dependent
samples: v10 gained +2 correct view-independent samples versus v9, but lost
4 correct view-dependent samples.

This means v10's first300 gain should not be generalized to random100. The
geometry helper is useful, but the playbook/tool policy likely overuses or
misapplies deterministic geometry in some visually grounded spatial cases.
The next fix should make the helper conditional: use it for pure
size/height/superlative shortlists, but require visual/spatial evidence for
view-dependent relations before final submission.

Operationally, `workers=100` is unsafe for this pack/runtime when many crop
callbacks trigger CLIP matching at once. Until CLIP matching is cached/shared or
disabled in `request_crops`, `workers=50` is the observed stable ceiling for
this run under a 15GB RSS guard.

## Caveats

- This is a 100-sample internal ablation, not a full NR3D row.
- The fold has visible LLM nondeterminism: v7 and v7.1 differed by 2pp under
  the same runtime code path.
- The run still has no selector frame-overlap NMS.
- The initial 100-worker pass was intentionally preserved rather than
  discarded; its completed checkpoints are valid, and all failed/missing cases
  were rerun to completion before scoring.
