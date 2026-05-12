# ScanRefer v3.17 - final-revision guard focus gate (REJECTED)

**Branch**: `feat/scanrefer-v3p10-stage-audit`
**Tip commit at harvest time**: `8acd3da` (working tree dirty; candidate code
was later reverted from the default path)
**Run ID** (SQLite): `v3p17_final_revision_guard_focus12_20260508`
**Date harvested**: 2026-05-08 GMT+8

## Headline

v3.17 tested a runtime `submit_final` guard for a real v3.11 failure mode:
after one final proposal has already been accepted, newly injected deferred
visual evidence can cause the agent to submit a different proposal without an
explicit `tool_override_reason`.

The focus12 gate is rejected. The guard fired, but the agent usually resubmitted
the changed proposal with a rationale, so the guard did not stabilize final
selection. It also regressed a v3.11 Acc@0.50 success on this slice.

| Run | n | Acc@0.25 | Acc@0.50 | Mean IoU |
|---|---:|---:|---:|---:|
| focus12 | 12 | 0.4167 | 0.2500 | 0.2987 |

No random100 gate was run.

## What Changed

The rejected candidate added `src/agents/skills/final_revision_guard.py` and a
runtime flag `use_final_revision_guard`. When enabled, the guard tracked
accepted `submit_final` calls in the current runtime and soft-blocked a later
submission that changed `proposal_id` without an explicit
`tool_override_reason`.

The guard was intended to catch cases where the first accepted final answer was
well-supported, but the final run result changed after deferred evidence was
injected. It was deliberately placed before the pack validator and used a
bounded repeat counter so the agent could still proceed after repeated
resubmissions.

Unit tests and CLI-wiring tests passed, but the focus gate showed the rule was
not useful enough as a promotion candidate. The guard did fire on six samples,
yet the final outcomes still trailed both v3.11 and v3.13 on the same focus12
slice.

## Commands

Focus12:

```bash
tmux new-session -d -s v3p17_focus12 \
  'cd /Users/bytedance/project/3DVLMReasoning && \
  CVRA_DISABLE=1 PYTHONPATH=src .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/v3p17_final_revision_guard_focus12_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p17_final_revision_guard_focus12_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 3 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard \
  --use-final-revision-guard \
  2>&1 | tee tmp/v3p17_final_revision_guard_focus12_eval.log'
```

Aggregation-GT rescoring:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/v3p17_final_revision_guard_focus12_eval/side_by_side.json \
  --output-dir tmp/v3p17_final_revision_guard_focus12_eval_agg_gt
```

Leaderboard slicing:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.scanrefer_leaderboard_metrics \
  --side-by-side tmp/v3p17_final_revision_guard_focus12_eval_agg_gt/side_by_side.json \
  --sample-ids tmp/v3p17_final_revision_guard_focus12_sample_ids.json \
  --output tmp/v3p17_final_revision_guard_focus12_eval_agg_gt/leaderboard_metrics.json
```

SQLite ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/v3p17_final_revision_guard_focus12_eval_agg_gt \
  --run-id v3p17_final_revision_guard_focus12_20260508 \
  --branch feat/scanrefer-v3p10-stage-audit \
  --commit 8acd3da \
  --backend pack_v1 \
  --judge-model none \
  --db docs/benchmark/scanrefer/runs.sqlite \
  --leaderboard-metrics tmp/v3p17_final_revision_guard_focus12_eval_agg_gt/leaderboard_metrics.json \
  --keyframe-mode query_driven \
  --notes "v3.17 focus12 final-revision guard; rejected: focus12 below v3.11/v3.13"
```

## Raw Artifacts

- Focus12 sample list:
  `tmp/v3p17_final_revision_guard_focus12_sample_ids.json`
- Focus12 raw output: `tmp/v3p17_final_revision_guard_focus12_eval/`
- Focus12 aggregation-GT rescore:
  `tmp/v3p17_final_revision_guard_focus12_eval_agg_gt/`
- Focus12 console log: `tmp/v3p17_final_revision_guard_focus12_eval.log`
- SQLite DB: `docs/benchmark/scanrefer/runs.sqlite`

## Evidence

Aggregation-GT rescoring:

```text
focus12: n=12 mean_iou=0.2987 acc25=0.4167 acc50=0.2500
```

Leaderboard slicing:

```text
n_total=12 Unique=3 Multiple=9
acc25 overall=0.4167 unique=0.3333 multiple=0.4444
acc50 overall=0.2500 unique=0.3333 multiple=0.2222
```

Matched comparison against existing v3.11 / v3.13 artifacts:

| Sample | v3.11 pid / IoU | v3.13 pid / IoU | v3.17 pid / IoU |
|---|---:|---:|---:|
| `scene0203_00::14::3` | 73 / 0.0000 | 2 / 0.9759 | 2 / 0.9759 |
| `scene0342_00::17::2` | 8 / 0.0000 | 13 / 0.0000 | 18 / 0.0000 |
| `scene0377_00::3::1` | 4 / 0.8658 | 4 / 0.8658 | 37 / 0.0000 |
| `scene0414_00::5::0` | 5 / 0.0000 | 15 / 0.8160 | 5 / 0.0000 |
| `scene0426_00::3::2` | 2 / 0.9648 | 2 / 0.9648 | 2 / 0.9648 |
| `scene0565_00::31::0` | 31 / 0.0000 | 22 / 0.4833 | 22 / 0.4833 |
| `scene0593_00::15::3` | 11 / 0.8697 | 11 / 0.8697 | 11 / 0.8697 |
| `scene0595_00::1::3` | 12 / 0.2907 | 9 / 0.2901 | 12 / 0.2907 |
| `scene0645_00::16::4` | 5 / 0.0000 | 5 / 0.0000 | 5 / 0.0000 |
| `scene0690_00::18::2` | 11 / 0.0396 | 20 / 0.0000 | 8 / 0.0000 |
| `scene0693_00::0::2` | 34 / 0.0000 | 6 / 0.0000 | 6 / 0.0000 |
| `scene0699_00::11::3` | 6 / 0.8333 | 42 / 0.0000 | 42 / 0.0000 |

Same-slice aggregates:

```text
v3.11: mean_iou=0.3220 acc25=0.4167 acc50=0.3333
v3.13: mean_iou=0.4388 acc25=0.5833 acc50=0.4167
v3.17: mean_iou=0.2987 acc25=0.4167 acc50=0.2500
```

Trigger audit:

```text
scannet/scene0203_00::14::3: block 71 -> 2
scannet/scene0377_00::3::1: block 41 -> 37
scannet/scene0414_00::5::0: block 10 -> 5
scannet/scene0426_00::3::2: block 35 -> 2
scannet/scene0593_00::15::3: block 39 -> 11
scannet/scene0595_00::1::3: block 18 -> 12
```

The guard was therefore reachable, but not decisive. On `scene0203_00` it
still ended with the v3.13/v3.17 correct proposal `2`, which is useful. On
`scene0377_00`, however, v3.11/v3.13 were already correct with proposal `4`,
while v3.17 ended at proposal `37`, producing IoU 0.0. The net slice result is
negative.

SQLite reproduction:

```sql
SELECT run_id, n_total, n_unique, n_multiple,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', mean_iou_overall) AS miou,
       keyframe_mode
FROM runs
WHERE run_id = 'v3p17_final_revision_guard_focus12_20260508';
```

Expected:

```text
v3p17_final_revision_guard_focus12_20260508|12|3|9|0.4167|0.2500|0.2987|query_driven
```

## Decision

Reject and revert. v3.17 shows that a late "are you sure you changed the
answer?" guard is too weak: if the agent can satisfy the guard with a short
explanation, the same unstable final-selection behavior remains. The next
candidate should move the contract earlier: produce an explicit final evidence
ledger that binds the winner, compared neighbors, inspected frames, spatial
relation evidence, and reject reasons before any `submit_final` call.
