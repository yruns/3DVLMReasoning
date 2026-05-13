# v9 Proposal Inventory Random100 - 2026-05-14

NR3D random100 rerun with the v9 VG prompt/runtime change. This isolates the
new Transcrib3D-style compact proposal inventory on the same fixed depth-aware
random100 fold and pack used by v7/v7.1.

## Run Identity

- Branch: `feat/nr3d-transcrib3d-first300`
- Tip commit at run time: `db95169`
- Internal version: `v9_inventory_random100`
- Run ID: `v9_inventory_random100_20260514`
- Stage 2 backend: `gpt-5.4-2026-03-05` via internal ModelHub
- Judge: none; scoring is programmatic target-id classification
- SQLite row:
  `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v9_inventory_random100_20260514'`

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
  - preserved v4/v6/v7 keyframe frame IDs
  - no selector frame-overlap NMS

## What Changed From v7.1

The v9 code injects a compact `Scene Proposal Inventory` into the VG user
message from the proposal pool. Each proposal row includes:

- proposal id
- category label
- 3D center
- 3D size
- visible-view count

The VG system/playbook also tells the agent to use this table as a text-first
candidate prior before spending turns on more images, mirroring Transcrib3D's
object-table candidate discipline. `chassis_tools_version` was bumped to `19`
so prompt-cache keys do not mix v8/v9 prompts.

No pack, fold, visibility, callback wiring, or final-decision guard was changed
for this run.

## Raw Artifacts

- Eval output:
  `tmp/nr3d_eval_v9_inventory_random100_20260514/`
- Per-sample checkpoints:
  `tmp/nr3d_eval_v9_inventory_random100_20260514/per_sample/pack_nr3d_v6_inline_labels_depth_visible/*.json`
- Initial failed sentinel backup:
  `tmp/nr3d_eval_v9_inventory_random100_20260514/failed_invalid_prompt_attempt1/`
- Failed-rerun sample ids:
  `tmp/nr3d_artifacts/v9_inventory_random100_failed_invalid_prompt_20260514.json`
- Side-by-side output:
  `tmp/nr3d_eval_v9_inventory_random100_20260514/side_by_side.json`
- Leaderboard metrics:
  `tmp/nr3d_eval_v9_inventory_random100_20260514/leaderboard_metrics.json`
- Logs:
  - `tmp/nr3d_eval_v9_inventory_random100_20260514_w100.log`
  - `tmp/nr3d_eval_v9_inventory_random100_failed2_20260514.log`
  - `tmp/nr3d_eval_v9_inventory_random100_20260514_assemble.log`
  - `tmp/nr3d_eval_v9_inventory_random100_20260514_metrics.log`

## Commands

Initial 100-worker checkpoint pass:

```bash
tmux new-session -d -s nr3d_v9_inventory_random100_20260514 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v6_inline_labels_depth_visible --output-dir tmp/nr3d_eval_v9_inventory_random100_20260514 --workers 100 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v9_inventory_random100_20260514_w100.log'
```

The initial pass produced 98 completed checkpoints and 2 `invalid_prompt`
failed sentinels. Those failed checkpoint files were moved aside and rerun:

```bash
tmux new-session -d -s nr3d_v9_inventory_random100_failed2_20260514 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v9_inventory_random100_failed_invalid_prompt_20260514.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v6_inline_labels_depth_visible --output-dir tmp/nr3d_eval_v9_inventory_random100_20260514 --workers 2 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v9_inventory_random100_failed2_20260514.log'
```

Assemble and score:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v6_inline_labels_depth_visible \
  --output-dir tmp/nr3d_eval_v9_inventory_random100_20260514 \
  --workers 100 \
  --max-new-samples 0 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v9_inventory_random100_20260514/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --output tmp/nr3d_eval_v9_inventory_random100_20260514/leaderboard_metrics.json \
  --canonical-filter true
```

Ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v9_inventory_random100_20260514 \
  --run-id v9_inventory_random100_20260514 \
  --branch feat/nr3d-transcrib3d-first300 \
  --commit db95169 \
  --backend pack_v1 \
  --judge-model none \
  --notes "v9 proposal-inventory prompt rerun on same random100 fold and pack_nr3d_v6_inline_labels_depth_visible; workers=100 with 15GB RSS guard and 60s checks; initial 2 invalid_prompt failed sentinels moved aside and rerun with workers=2; final 100/100 completed; TADG + no-match + evidence-frame guards; no callback-not-configured; no rss_guard" \
  --leaderboard-metrics tmp/nr3d_eval_v9_inventory_random100_20260514/leaderboard_metrics.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

## Metrics

### Leaderboard-Track Classification

| Metric | v7 original | v7.1 rerun | v9 inventory | Delta vs v7.1 |
|---|---:|---:|---:|---:|
| Overall | 73.00 | 71.00 | 71.00 | +0.00 pp |
| Easy | 82.93 | 82.93 | 85.37 | +2.44 pp |
| Hard | 66.10 | 62.71 | 61.02 | -1.69 pp |
| View-Dep | 70.59 | 61.76 | 61.76 | +0.00 pp |
| View-Indep | 74.24 | 75.76 | 75.76 | +0.00 pp |

### IoU Proxy on GT Pool

| Metric | v7.1 rerun | v9 inventory | Delta |
|---|---:|---:|---:|
| n | 100 | 100 | 0 |
| mean IoU | 0.7147 | 0.7123 | -0.0024 |
| Acc@0.25 | 0.7100 | 0.7100 | +0.0000 |
| Acc@0.50 | 0.7100 | 0.7100 | +0.0000 |
| final failed sentinels | 0 | 0 | 0 |

### Per-Sample Drift vs v7.1

| Check | Count |
|---|---:|
| Samples with changed selected proposal id or correctness | 21 |
| Wrong -> correct | 8 |
| Correct -> wrong | 8 |
| Correct -> correct id changes | 0 |
| Wrong -> wrong id changes | 5 |
| Net overall movement | 0 |

Wrong -> correct sample IDs:

- `scannet/scene0652_00::13::34162`
- `scannet/scene0474_00::12::7632`
- `scannet/scene0704_00::0::16937`
- `scannet/scene0578_00::10::5147`
- `scannet/scene0618_00::26::28943`
- `scannet/scene0643_00::22::25584`
- `scannet/scene0663_00::3::36534`
- `scannet/scene0653_00::16::3852`

Correct -> wrong sample IDs:

- `scannet/scene0187_00::13::24713`
- `scannet/scene0084_00::35::19742`
- `scannet/scene0651_00::8::37347`
- `scannet/scene0629_00::6::19188`
- `scannet/scene0500_00::27::34282`
- `scannet/scene0249_00::36::39076`
- `scannet/scene0697_00::19::35451`
- `scannet/scene0645_00::34::34611`

Wrong -> wrong id changes:

- `scannet/scene0704_00::5::8434`
- `scannet/scene0643_00::6::10715`
- `scannet/scene0644_00::40::39251`
- `scannet/scene0568_00::19::17227`
- `scannet/scene0095_00::28::8465`

## Runtime Notes

- Initial 100-worker pass: 98 completed, 2 `invalid_prompt` failed sentinels.
- Targeted failed rerun: 2/2 completed at `workers=2`.
- Final checkpoint audit: 100/100 completed, 0 failed sentinels.
- Peak observed Python RSS during monitoring was about 5.4GB, below the 15GB
  RSS guard.
- Combined log counts:
  - retryable `429`: 62
  - retryable `503`: 0
  - attempts at 3/5 or higher: 0
  - `callback is not configured`: 0
  - `rss_guard`: 0
  - `Traceback`: 0
  - `Loading CLIP model`: 7

## Interpretation

The compact proposal inventory did not improve this random100 fold overall. It
changed decisions in both directions: eight v7.1 misses became correct, but
eight v7.1 correct samples regressed. The same 71.00 overall as v7.1 means this
change should not be treated as a random100 win by itself.

The result is still useful: the prompt/runtime change is stable under 100-way
concurrency, stays far below the memory guard, and does not reintroduce the
callback-not-configured issue. More targeted transfer work should focus on
deterministic candidate ranking helpers or stronger relation-specific prompts,
because merely exposing the object table relies on the LLM to use geometry
consistently.

## Caveats

- This is a 100-sample internal ablation, not a full NR3D row.
- The fold has visible LLM nondeterminism: v7 and v7.1 differed by 2pp under
  the same runtime code path.
- The run still has no selector frame-overlap NMS.
- The proposal inventory uses detector categories as weak priors; it does not
  add new visual evidence or change the proposal pool.
