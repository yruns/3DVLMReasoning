# v7.1 Callbacks No-CLIP Random100 Rerun - 2026-05-14

NR3D random100 rerun of the current checkout after the v7 callback/no-CLIP
hardening. This is a repeatability check on the same depth-aware fixed fold and
pack, not a new full benchmark row.

## Run Identity

- Branch: `feat/nr3d-transcrib3d-first300`
- Tip commit at run time: `101acaa`
- Code lineage: same Stage2 runtime code as v7 (`690cbf6`); `101acaa` is the
  v7 documentation commit on top.
- Internal version: `v7p1_callbacks_noclip_random100_rerun`
- Run ID: `v7p1_callbacks_noclip_random100_rerun_20260514`
- Stage 2 backend: `gpt-5.4-2026-03-05` via internal ModelHub
- Judge: none; scoring is programmatic target-id classification
- SQLite row:
  `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v7p1_callbacks_noclip_random100_rerun_20260514'`

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
  - preserved v4/v6 keyframe frame IDs
  - no selector frame-overlap NMS

## What This Rerun Tested

- Reused the v7 Stage2 agent configuration:
  - `request_more_views`, `request_crops`, and `switch_or_expand_hypothesis`
    callbacks wired for NR3D
  - `request_more_views` does not use the per-selector CLIP object-term
    fallback, avoiding high-concurrency CLIP memory spikes
  - TADG, no-match candidate guard, and evidence-frame guard enabled
- Reran the same 100Q fold to measure nondeterministic drift from LLM calls and
  callback use.

## Raw Artifacts

- Eval output:
  `tmp/nr3d_eval_v7_callbacks_noclip_random100_rerun_20260514/`
- Per-sample checkpoints:
  `tmp/nr3d_eval_v7_callbacks_noclip_random100_rerun_20260514/per_sample/pack_nr3d_v6_inline_labels_depth_visible/*.json`
- Initial failed sentinel backup:
  `tmp/nr3d_eval_v7_callbacks_noclip_random100_rerun_20260514/failed_invalid_prompt_attempt1/`
- Side-by-side output:
  `tmp/nr3d_eval_v7_callbacks_noclip_random100_rerun_20260514/side_by_side.json`
- Leaderboard metrics:
  `tmp/nr3d_eval_v7_callbacks_noclip_random100_rerun_20260514/leaderboard_metrics.json`
- Logs:
  - `tmp/nr3d_eval_v7_callbacks_noclip_random100_rerun_20260514_w100.log`
  - `tmp/nr3d_eval_v7_callbacks_noclip_random100_rerun_failed3_20260514.log`
  - `tmp/nr3d_eval_v7_callbacks_noclip_random100_rerun_20260514_assemble.log`
  - `tmp/nr3d_eval_v7_callbacks_noclip_random100_rerun_20260514_metrics.log`

## Commands

Initial 100-worker checkpoint pass:

```bash
tmux new-session -d -s nr3d_v7_rerun_random100_20260514 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v6_inline_labels_depth_visible --output-dir tmp/nr3d_eval_v7_callbacks_noclip_random100_rerun_20260514 --workers 100 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v7_callbacks_noclip_random100_rerun_20260514_w100.log'
```

The initial pass produced 97 completed checkpoints and 3 `invalid_prompt`
failed sentinels. Those failed checkpoint files were moved aside and the three
samples were rerun only:

```bash
tmux new-session -d -s nr3d_v7_rerun_random100_failed3_20260514 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v7_rerun_failed_invalid_prompt_20260514.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v6_inline_labels_depth_visible --output-dir tmp/nr3d_eval_v7_callbacks_noclip_random100_rerun_20260514 --workers 3 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v7_callbacks_noclip_random100_rerun_failed3_20260514.log'
```

Assemble and score:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v6_inline_labels_depth_visible \
  --output-dir tmp/nr3d_eval_v7_callbacks_noclip_random100_rerun_20260514 \
  --workers 100 \
  --max-new-samples 0 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v7_callbacks_noclip_random100_rerun_20260514/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --output tmp/nr3d_eval_v7_callbacks_noclip_random100_rerun_20260514/leaderboard_metrics.json \
  --canonical-filter true
```

Ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v7_callbacks_noclip_random100_rerun_20260514 \
  --run-id v7p1_callbacks_noclip_random100_rerun_20260514 \
  --branch feat/nr3d-transcrib3d-first300 \
  --commit 101acaa \
  --backend pack_v1 \
  --judge-model none \
  --notes "v7.1 random100 rerun on same v4/v6/v7 fold and pack_nr3d_v6_inline_labels_depth_visible at current tip; workers=100 with 15GB RSS guard; initial 3 invalid_prompt failed sentinels were moved aside and rerun with workers=3; final 100/100 completed; TADG + no-match + evidence-frame guards; no callback-not-configured; no rss_guard" \
  --leaderboard-metrics tmp/nr3d_eval_v7_callbacks_noclip_random100_rerun_20260514/leaderboard_metrics.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

## Metrics

### Leaderboard-Track Classification

| Metric | v7 2026-05-13 | v7.1 rerun 2026-05-14 | Delta |
|---|---:|---:|---:|
| Overall | 73.00 | 71.00 | -2.00 pp |
| Easy | 82.93 | 82.93 | +0.00 pp |
| Hard | 66.10 | 62.71 | -3.39 pp |
| View-Dep | 70.59 | 61.76 | -8.82 pp |
| View-Indep | 74.24 | 75.76 | +1.52 pp |

### IoU Proxy on GT Pool

| Metric | v7 2026-05-13 | v7.1 rerun 2026-05-14 | Delta |
|---|---:|---:|---:|
| n | 100 | 100 | 0 |
| mean IoU | 0.7327 | 0.7147 | -0.0180 |
| Acc@0.25 | 0.7300 | 0.7100 | -0.0200 |
| Acc@0.50 | 0.7300 | 0.7100 | -0.0200 |
| final failed sentinels | 0 | 0 | 0 |

### Per-Sample Drift vs v7

| Check | Count |
|---|---:|
| Samples with changed selected proposal id or correctness | 17 |
| Wrong -> correct | 5 |
| Correct -> wrong | 7 |
| Correct -> correct id changes | 0 |
| Wrong -> wrong id changes | 5 |
| Net overall movement | -2 |

Wrong -> correct sample IDs:

- `scannet/scene0490_00::14::39189`
- `scannet/scene0500_00::27::34282`
- `scannet/scene0618_00::18::36850`
- `scannet/scene0629_00::6::19188`
- `scannet/scene0645_00::38::39739`

Correct -> wrong sample IDs:

- `scannet/scene0474_00::12::7632`
- `scannet/scene0618_00::26::28943`
- `scannet/scene0643_00::22::25584`
- `scannet/scene0648_00::15::24119`
- `scannet/scene0652_00::13::34162`
- `scannet/scene0653_00::16::3852`
- `scannet/scene0704_00::0::16937`

The three initial `invalid_prompt` sentinels were not responsible for the
metric drop. After the targeted rerun, all three were completed and all three
matched the old v7 correctness:

| Sample | Old v7 | v7.1 rerun |
|---|---:|---:|
| `scannet/scene0329_00::7::11351` | correct | correct |
| `scannet/scene0025_00::30::25001` | correct | correct |
| `scannet/scene0353_00::52::27519` | correct | correct |

## Runtime Notes

- Initial 100-worker pass: 97 completed, 3 `invalid_prompt` failed sentinels.
- Targeted failed rerun: 3/3 completed at `workers=3`.
- Final checkpoint audit: 100/100 completed, 0 failed sentinels.
- Combined log counts:
  - retryable `429`: 68
  - retryable `503`: 1
  - attempts at 3/5 or higher: 0
  - `callback is not configured`: 0
  - `rss_guard`: 0
  - `Loading CLIP model`: 3, from text-only Stage1 re-query path

## Caveats

- This is still a 100-sample internal ablation, not a full NR3D row.
- The rerun demonstrates about 2pp negative drift on this fold under the same
  code path, so single random100 runs should be interpreted as noisy.
- No selector frame-overlap NMS or Transcrib3D-derived reasoning changes are
  included here.
