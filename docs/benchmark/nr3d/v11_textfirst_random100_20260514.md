# v11 Text-First Policy Random100 - 2026-05-14

NR3D random100 rerun after changing the VG prompt/playbook to enforce a
Transcrib3D-like structured-first pass: use the proposal inventory, category
lookup, proposal metadata, and deterministic geometry/spatial tools before
spending turns on extra images.

## Run Identity

- Branch: `feat/nr3d-transcrib3d-first300`
- Tip commit at run time: `4ee6167`
- Internal version: `v11_textfirst_random100`
- Run ID: `v11_textfirst_random100_20260514`
- Stage 2 backend: `gpt-5.4-2026-03-05` via internal ModelHub
- Judge: none; scoring is programmatic target-id classification
- SQLite row:
  `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v11_textfirst_random100_20260514'`

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
  - preserved v4/v6/v7/v9/v10 keyframe frame ids
  - no selector frame-overlap NMS

## What Changed From v10

v11 changes policy, not the proposal pool or fold:

- VG system prompt now requires a "Structured-first VG pass".
- `vg-grounding-playbook` now asks the agent to:
  - read the Scene Proposal Inventory first,
  - identify focal object category and synonyms,
  - call `find_proposals_by_category`,
  - inspect candidates with `inspect_proposal`,
  - use `rank_proposals_by_geometry` for pure size/height superlatives,
  - use `compare_proposals_spatial` for anchor relations,
  - request images/crops only after structured evidence leaves ambiguity or
    visual confirmation is required.
- Prompt explicitly distinguishes clean NR3D-style GT/object-pool VG from
  noisy ScanRefer/Mask3D detector-pool VG.
- `chassis_tools_version` was bumped to `21`.

No pack, fold, visibility, callback wiring, rank-tool implementation, or final
decision guard changed.

## Raw Artifacts

- Eval output:
  `tmp/nr3d_eval_v11_textfirst_random100_20260514/`
- Per-sample checkpoints:
  `tmp/nr3d_eval_v11_textfirst_random100_20260514/per_sample/pack_nr3d_v6_inline_labels_depth_visible/*.json`
- Side-by-side output:
  `tmp/nr3d_eval_v11_textfirst_random100_20260514/side_by_side.json`
- Leaderboard metrics:
  `tmp/nr3d_eval_v11_textfirst_random100_20260514/leaderboard_metrics.json`
- Logs:
  - `tmp/nr3d_eval_v11_textfirst_random100_20260514_w50.log`
  - `tmp/nr3d_eval_v11_textfirst_random100_20260514_assemble.log`
  - `tmp/nr3d_eval_v11_textfirst_random100_20260514_metrics.log`

## Commands

Checkpoint run:

```bash
tmux new-session -d -s nr3d_v11_textfirst_random100_20260514 \
  'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v6_inline_labels_depth_visible --output-dir tmp/nr3d_eval_v11_textfirst_random100_20260514 --workers 50 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v11_textfirst_random100_20260514_w50.log'
```

Assemble and score:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v6_inline_labels_depth_visible \
  --output-dir tmp/nr3d_eval_v11_textfirst_random100_20260514 \
  --workers 50 \
  --max-new-samples 0 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v11_textfirst_random100_20260514/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --output tmp/nr3d_eval_v11_textfirst_random100_20260514/leaderboard_metrics.json \
  --canonical-filter true
```

Ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v11_textfirst_random100_20260514 \
  --run-id v11_textfirst_random100_20260514 \
  --branch feat/nr3d-transcrib3d-first300 \
  --commit 4ee6167 \
  --backend pack_v1 \
  --judge-model none \
  --notes "v11 VG text-first policy on fixed depth-aware random100 fold; same pack_nr3d_v6_inline_labels_depth_visible as v9/v10; workers=50 under 15GB RSS guard; final 100/100 completed; structured-first prompt increased category/proposal tool use; no callback-not-configured; no rss_guard; 72/100 overall." \
  --leaderboard-metrics tmp/nr3d_eval_v11_textfirst_random100_20260514/leaderboard_metrics.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

SQLite reproduction:

```sql
SELECT run_id, n,
       printf('%.4f', classification_acc_full) AS acc,
       printf('%.4f', acc_easy) AS easy,
       printf('%.4f', acc_hard) AS hard,
       printf('%.4f', acc_view_dep) AS view_dep,
       printf('%.4f', acc_view_indep) AS view_indep
FROM runs
WHERE run_id='v11_textfirst_random100_20260514';
```

Expected:

```text
v11_textfirst_random100_20260514|100|0.7200|0.8293|0.6441|0.5882|0.7879
```

## Metrics

### Leaderboard-Track Classification

| Metric | v9 inventory | v10 geometry | v11 text-first | Delta vs v10 |
|---|---:|---:|---:|---:|
| Overall | 71.00 | 69.00 | 72.00 | +3.00 pp |
| Easy | 85.37 | 85.37 | 82.93 | -2.44 pp |
| Hard | 61.02 | 57.63 | 64.41 | +6.78 pp |
| View-Dep | 61.76 | 50.00 | 58.82 | +8.82 pp |
| View-Indep | 75.76 | 78.79 | 78.79 | +0.00 pp |

### Per-Sample Drift

| Comparison | v11-only correct | Baseline-only correct | Net |
|---|---:|---:|---:|
| vs v10 geometry | 8 | 5 | +3 |
| vs v9 inventory | 9 | 8 | +1 |

v11 improvements over v10:

- `scannet/scene0300_00::8::22495`
- `scannet/scene0690_00::15::4351`
- `scannet/scene0645_00::34::34611`
- `scannet/scene0435_00::60::25927`
- `scannet/scene0643_00::6::10715`
- `scannet/scene0474_00::12::7632`
- `scannet/scene0618_00::26::28943`
- `scannet/scene0553_00::10::27475`

v11 regressions from v10:

- `scannet/scene0704_00::5::8434`
- `scannet/scene0490_00::14::39189`
- `scannet/scene0565_00::20::35581`
- `scannet/scene0549_00::31::29364`
- `scannet/scene0618_00::18::36850`

### Tool Policy Shift

| Tool | v9 calls | v10 calls | v11 calls |
|---|---:|---:|---:|
| `find_proposals_by_category` | 126 | 126 | 181 |
| `inspect_proposal` | 356 | 300 | 381 |
| `view_keyframe_marked` | 159 | 183 | 205 |
| `compare_proposals_spatial` | 79 | 86 | 92 |
| `rank_proposals_by_geometry` | 0 | 14 | 16 |
| `request_crops` | 21 | 17 | 31 |
| `request_more_views` | 33 | 30 | 25 |

The prompt did move the agent toward structured candidate filtering. It also
slightly increased visual/crop demand, which matters for memory.

## Runtime Notes

- Initial pass: 100/100 completed, 0 failed sentinels.
- Log counts:
  - retryable `429`: 48
  - retryable `503`: 0
  - attempts at 2/5 or higher: 0
  - `callback is not configured`: 0
  - `rss_guard`: 0
  - `Traceback`: 0
  - `Loading CLIP model`: 4

## Interpretation

v11 recovers the random100 regression seen in v10 and reaches the best current
same-fold rerun on this branch: 72/100. The gain comes mostly from Hard and
view-dependent slices relative to v10, which is consistent with a policy that
forces candidate/anchor reasoning before final submission.

This is a small 100-sample development fold. The same v11 policy drops the
Transcrib3D first300-valid fold from 216/281 to 214/281, so it should not be
promoted as an unconditional improvement.

## Caveats

- This is a 100-sample internal ablation, not a full NR3D row.
- The fold has known LLM nondeterminism; v7 and v7.1 differed by 2pp under the
  same runtime path.
- No selector frame-overlap NMS was applied.
- This run did not hit the RSS guard, but v11 first300 did; the crop callback's
  CLIP fallback remains a memory-risk path at high concurrency.
