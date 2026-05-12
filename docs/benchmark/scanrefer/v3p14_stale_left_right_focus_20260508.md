# ScanRefer v3.14 - stale left/right focus gate (REJECTED)

**Branch**: `feat/scanrefer-v3p10-stage-audit`
**Tip commit at harvest time**: `8acd3da` (working tree dirty; candidate code
was later reverted from the default path)
**Run ID** (SQLite): `v3p14_stale_left_right_focus1_20260508`
**Date harvested**: 2026-05-08 GMT+8

## Headline

v3.14 was a one-sample focus gate, not a promoted random100 result. It tested
a narrow evidence-frame guard for stale zero-support `left_of` / `right_of`
spatial ranks on `scannet/scene0552_00::14::0`, a v3.11 miss that v3.13
recovered stochastically. The guard did not fire on the fresh trace and the
sample still selected proposal `24` with IoU `0.0000`.

| Split | Acc@0.25 | Acc@0.50 |
|---|---:|---:|
| Unique | 0.00 | 0.00 |
| Multiple | 0.00 | 0.00 |
| Overall | 0.00 | 0.00 |

Mean IoU is `0.0000` on `n=1`. This result is excluded from the public
leaderboard table because it is only a focus gate. v3.11 remains the current
honest query-driven random100 headline.

## What Changed

The rejected candidate attempted to block this pattern inside
`src/agents/skills/evidence_frame_guard.py`:

- latest `compare_proposals_spatial` was `left_of` / `right_of`;
- submitted proposal was rank-1;
- all `supporting_frame_counts` were zero;
- later viewed marked frames showed the spatial anchor plus an alternative
  candidate while the submitted proposal was absent.

The matching unit test passed locally, but the fresh focus trace did not
exhibit the same condition. The agent viewed frame `8`, where both candidate
cabinet proposals `24` and `6` plus trash-can anchor `23` were listed, then
cited frame `57`, where submitted proposal `24` and anchor `23` were listed.
Therefore the "submitted absent, alternative present" branch never fired.

The candidate code, test, and prompt-cache bump were removed after the failed
focus gate. `chassis_tools_version` was restored to `18`.

## Commands

Fresh focus run:

```bash
tmux new-session -d -s v3p14_focus1 \
  'cd /Users/bytedance/project/3DVLMReasoning && \
  CVRA_DISABLE=1 PYTHONPATH=src .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/v3p14_stale_left_right_focus1_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p14_stale_left_right_focus1_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 1 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard \
  2>&1 | tee tmp/v3p14_stale_left_right_focus1_eval.log'
```

Aggregation-GT rescore:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/v3p14_stale_left_right_focus1_eval/side_by_side.json \
  --output-dir tmp/v3p14_stale_left_right_focus1_eval_agg_gt
```

Leaderboard slicing:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.scanrefer_leaderboard_metrics \
  --side-by-side tmp/v3p14_stale_left_right_focus1_eval_agg_gt/side_by_side.json \
  --sample-ids tmp/v3p14_stale_left_right_focus1_sample_ids.json \
  --output tmp/v3p14_stale_left_right_focus1_eval_agg_gt/leaderboard_metrics.json
```

SQLite ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/v3p14_stale_left_right_focus1_eval_agg_gt \
  --run-id v3p14_stale_left_right_focus1_20260508 \
  --branch feat/scanrefer-v3p10-stage-audit \
  --commit 8acd3da \
  --leaderboard-metrics tmp/v3p14_stale_left_right_focus1_eval_agg_gt/leaderboard_metrics.json \
  --db docs/benchmark/scanrefer/runs.sqlite \
  --keyframe-mode query_driven \
  --notes "v3.14 REJECTED focus gate: attempted stale zero-support left/right EFG branch; one-sample scene0552 focus still selected proposal 24 with IoU 0.0 because the trace did not match the narrow absent-submitted condition; candidate reverted from default path."
```

## Raw Artifacts

- Focus sample list: `tmp/v3p14_stale_left_right_focus1_sample_ids.json`
- Raw output: `tmp/v3p14_stale_left_right_focus1_eval/`
- Aggregation-GT rescore:
  `tmp/v3p14_stale_left_right_focus1_eval_agg_gt/`
- Console log: `tmp/v3p14_stale_left_right_focus1_eval.log`
- SQLite DB: `docs/benchmark/scanrefer/runs.sqlite`

## Evidence

Aggregation-GT rescore:

```text
n=1 mean_iou=0.0000 acc25=0.0000 acc50=0.0000
stats: missing_scene=0 missing_target=0 no_prediction=0 completed_new_gt=1
```

Leaderboard metrics:

```text
n_total=1 (Unique=1, Multiple=0)
acc25 overall=0.0000 unique=0.0000 multiple=0.0000
acc50 overall=0.0000 unique=0.0000 multiple=0.0000
```

Key trace:

```text
compare_proposals_spatial(candidate_ids=[6,24], anchor_id=23, relation=right_of)
  ranked_ids=[24,6]
  shared_frame_counts=[0,0]
  supporting_frame_counts=[0,0]

view_keyframe_marked(frame_id=8)
  visible_proposals=[15,17,14,23,2,21,24,6]

view_keyframe_marked(frame_id=57)
  visible_proposals=[8,24,15,2,7,42,4,14,17,3,31,23,29,25]

submit_final(proposal_id=24)
  evidence_frame_guard_blocked=false
  selected_object_id=24
  IoU=0.0000
```

SQLite reproduction:

```sql
SELECT run_id, n_total, n_unique, n_multiple,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', mean_iou_overall) AS miou,
       keyframe_mode
FROM runs
WHERE run_id='v3p14_stale_left_right_focus1_20260508';
```

Expected:

```text
v3p14_stale_left_right_focus1_20260508|1|1|0|0.0000|0.0000|0.0000|query_driven
```

Tool calls ingested: `17`.

## Decision

Do not promote v3.14. The patch was too trace-specific: it matched the v3.11
failure simulation, but not the fresh focus trajectory. The default path has
been restored to v3.11/v3.13-era behavior (`chassis_tools_version=18`).

The useful lesson is that zero-support left/right 3D ranking remains risky,
but the fix should not be another narrow after-the-fact guard. A stronger next
candidate should bind the final answer to explicit relation evidence that
contains both target and anchor with usable 2D boxes, or revise the spatial
tool so zero-support left/right ranks are treated as unverified rather than
ranked evidence.
