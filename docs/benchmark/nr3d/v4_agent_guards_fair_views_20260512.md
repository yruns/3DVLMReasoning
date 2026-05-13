# v4 Agent Guards + Fair Views Random100 - 2026-05-12

NR3D pilot run using the optimized ScanRefer-style VG agent framework while
removing the previous GT-target-visible keyframe shortcut. This is a
100-sample partial pilot, not a full public leaderboard row.

**Status update on 2026-05-13:** invalidated as a fair-view diagnostic because
the underlying NR3D object-frame visibility source was built with
`metadata.use_depth=false`.

## Run Identity

- Branch: `feat/nr3d-v4-agent-guards-fair-views`
- Tip commit at run time: `3f1c6d8`
- Working tree: `/Users/bytedance/project/3DVLMReasoning` (Mac, `.venv`, Python 3.12)
- Internal version: `v4_agent_guards_fair_views_random100`
- Run ID: `v4_agent_guards_fair_views_random100_20260512`
- Stage 1 keyframe parser model: `gemini-2.5-pro`
- Stage 2 backend: `gpt-5.4-2026-03-05` via internal ModelHub
- Judge: none; scoring is programmatic
- SQLite row: `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v4_agent_guards_fair_views_random100_20260512'`

## What Changed vs v3

This pilot changes the inference-time evidence path and the agent guard set:

- `prepare_pack_v1_inputs_nr3d.py` used `--keyframe-mode query_driven`.
- Keyframes are selected from the query and scene evidence, not from
  `visibility.object_to_views[target_id]`.
- Stage 2 guard flags were enabled:
  - `--use-tool-answer-disagreement-gate`
  - `--use-no-match-candidate-guard`
  - `--use-evidence-frame-guard`
- Runner execution used per-sample checkpoints and final streaming
  `side_by_side.json` assembly to avoid retaining all results in memory.
- Concurrency was ramped by checkpoint batch: workers `30 -> 60 -> 100`.
  The stopping rule was LLM/service failure rate greater than 25%; observed
  failure rate stayed at 0%.

The candidate pool remains the NR3D ScanNet GT bbox pool. The target id and
GT bbox are still used for scoring, but not for keyframe selection.

## Fold

- Split: NR3D `test`
- Fold size: 100 samples
- Selection file:
  `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json`
- Summary file:
  `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_summary.json`
- Selection mechanism:
  deterministic SHA1 ordering over the canonical filtered fold intersected
  with existing v1/v2 full-run predictions.
- Selection salt:
  `nr3d_v4_agent_guards_fair_views_20260512`
- Candidate count before selecting 100: 7805
- Duplicate check: 100 rows, 100 unique `sample_id`s
- Canonical filter after restriction: `n_full=100`, `n_filtered=100`

## Fair-View Audit

Prepared sample audit:

| Check | Value |
|---|---:|
| Prepared samples | 100 |
| Missing sample artifacts | 0 |
| `keyframe_mode != query_driven` | 0 |
| `keyframe_selection_uses_gt_target != false` | 0 |
| Non-GT fallback used | 29 |

The 29 fallback samples used the non-GT scene-density fallback path after
query-driven selection failed to produce executable evidence. They did not use
`target_id`, GT bbox, or target visibility.

## Raw Artifacts

- Prepared pack:
  `data/nr3d/scannet/<scene>/pack_nr3d_v4_agent_guards_fair_views/`
- Eval output:
  `tmp/nr3d_eval_v4_agent_guards_fair_views_random100/`
- Per-sample checkpoints:
  `tmp/nr3d_eval_v4_agent_guards_fair_views_random100/per_sample/pack_nr3d_v4_agent_guards_fair_views/*.json`
- Side-by-side output:
  `tmp/nr3d_eval_v4_agent_guards_fair_views_random100/side_by_side.json`
- Leaderboard metrics:
  `tmp/nr3d_eval_v4_agent_guards_fair_views_random100/leaderboard_metrics.json`
- Same-fold baseline metrics:
  `tmp/nr3d_eval_v4_baseline_random100/leaderboard_metrics.json`
- Logs:
  - `tmp/nr3d_eval_v4_prep_random100.log`
  - `tmp/nr3d_eval_v4_agent_guards_fair_views_random100_w30.log`
  - `tmp/nr3d_eval_v4_agent_guards_fair_views_random100_w60.log`
  - `tmp/nr3d_eval_v4_agent_guards_fair_views_random100_w100.log`

## Commands

Build the fixed fold:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/build_nr3d_v4_random100_fold.py
```

Prepare query-driven packs:

```bash
tmux new-session -d -s nr3d_v4_prep 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && PYTHONUNBUFFERED=1 PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v4_agent_guards_fair_views \
  --split test \
  --keyframe-mode query_driven \
  --keyframe-llm-model gemini-2.5-pro \
  2>&1 | tee tmp/nr3d_eval_v4_prep_random100.log'
```

Run checkpoint batches:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v4_agent_guards_fair_views \
  --output-dir tmp/nr3d_eval_v4_agent_guards_fair_views_random100 \
  --workers 30 \
  --max-new-samples 30 \
  --checkpoint-only \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONUNBUFFERED=1 PYTHONPATH=src python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v4_agent_guards_fair_views \
  --output-dir tmp/nr3d_eval_v4_agent_guards_fair_views_random100 \
  --workers 60 \
  --max-new-samples 50 \
  --checkpoint-only \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONUNBUFFERED=1 PYTHONPATH=src python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v4_agent_guards_fair_views \
  --output-dir tmp/nr3d_eval_v4_agent_guards_fair_views_random100 \
  --workers 100 \
  --checkpoint-only \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard
```

Assemble without re-running samples:

```bash
PYTHONPATH=src python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v4_agent_guards_fair_views \
  --output-dir tmp/nr3d_eval_v4_agent_guards_fair_views_random100 \
  --workers 100 \
  --max-new-samples 0 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard
```

Compute metrics and ingest:

```bash
PYTHONPATH=src python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v4_agent_guards_fair_views_random100/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --output tmp/nr3d_eval_v4_agent_guards_fair_views_random100/leaderboard_metrics.json \
  --canonical-filter true

PYTHONPATH=src python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v4_agent_guards_fair_views_random100 \
  --run-id v4_agent_guards_fair_views_random100_20260512 \
  --branch feat/nr3d-v4-agent-guards-fair-views \
  --commit 3f1c6d8 \
  --backend pack_v1 \
  --judge-model none \
  --notes "v4 random100; query-driven fair keyframes; TADG + no-match + evidence-frame guards; workers ramp 30/60/100; no LLM service failures" \
  --leaderboard-metrics tmp/nr3d_eval_v4_agent_guards_fair_views_random100/leaderboard_metrics.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

## Metrics

### Leaderboard-Track Classification

| Metric | v1/v3 same-fold baseline | v4 fair-view + guards | Delta |
|---|---:|---:|---:|
| Overall | 80.00 | 71.00 | -9.00 pp |
| Easy | 85.37 | 82.93 | -2.44 pp |
| Hard | 76.27 | 62.71 | -13.56 pp |
| View-Dep | 61.76 | 67.65 | +5.88 pp |
| View-Indep | 89.39 | 72.73 | -16.67 pp |

### IoU Proxy on GT Pool

| Metric | Value |
|---|---:|
| n | 100 |
| mean IoU | 0.7124 |
| Acc@0.25 | 0.7100 |
| Acc@0.50 | 0.7100 |
| failed sentinels | 0 |
| LLM/service errors | 0 |

Under the GT-pool setup, Acc@0.50 is equivalent to target-instance
classification accuracy for completed samples.

## Per-Sample Delta Summary

| Transition | Count |
|---|---:|
| baseline correct, v4 correct | 61 |
| baseline correct, v4 wrong | 19 |
| baseline wrong, v4 correct | 10 |
| baseline wrong, v4 wrong | 10 |

The net change is `10 - 19 = -9` samples, matching the -9.00 pp overall
delta. The main regression is on Hard and View-Indep examples, while
View-Dep improves on this fixed subset.

## SQLite Reproduction Query

```sql
SELECT run_id, n,
       printf('%.4f', classification_acc_filtered) AS overall,
       printf('%.4f', acc_easy) AS easy,
       printf('%.4f', acc_hard) AS hard,
       printf('%.4f', acc_view_dep) AS vdep,
       printf('%.4f', acc_view_indep) AS vind,
       n_filtered
FROM runs
WHERE run_id='v4_agent_guards_fair_views_random100_20260512';
-- v4_agent_guards_fair_views_random100_20260512|100|0.7100|0.8293|0.6271|0.6765|0.7273|100
```

## Interpretation

The fair-view pilot is lower than the same-fold v1/v3 baseline overall
because the old baseline used GT-target-visible keyframe selection. Removing
that shortcut makes the evidence problem harder. The guard stack did not cause
service instability: all 100 samples completed and no LLM/service failures were
recorded even after ramping workers to 100.

The result should not replace the v3 full-test leaderboard row. It is a
diagnostic pilot showing that the previous full-run score partially depended on
GT-assisted evidence selection, and that the optimized agent framework can run
NR3D with fair query-driven keyframes under high concurrency.

## Caveats

- Partial fold only: 100 / 7805 canonical filtered samples.
- The same-fold baseline is post-aggregated from the earlier full run, whose
  keyframes used GT-target visibility.
- v4 changes two factors at once: fair query-driven keyframes and ScanRefer
  guard constraints. It is not an isolated guard ablation.
- 29 / 100 prepared samples used non-GT fallback keyframes after query-driven
  selection could not produce evidence.
- The run still uses GT bbox candidates, as expected for the NR3D
  classification/GT-pool protocol. GT is not exposed to keyframe selection or
  final selection.
