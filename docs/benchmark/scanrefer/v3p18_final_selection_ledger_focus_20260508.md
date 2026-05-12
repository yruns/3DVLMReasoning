# ScanRefer v3.18 - final-selection ledger focus gate (REJECTED)

**Branch**: `feat/scanrefer-v3p10-stage-audit`
**Tip commit at harvest time**: `8acd3da` (working tree dirty; candidate code
was later reverted from the default path)
**Run ID** (SQLite): `v3p18_final_selection_ledger_focus12_20260508`
**Date harvested**: 2026-05-08 GMT+8

## Headline

v3.18 tested a pre-submit final-selection ledger: the agent had to call
`record_final_selection(...)` before `submit_final`, binding the winner,
candidate set, evidence frame ids, and reject reasons.

The focus12 gate is rejected. The agent complied with the ledger tool on every
sample, but the explicit ledger did not improve the actual selection. The slice
landed far below v3.11, v3.13, and v3.17.

| Run | n | Acc@0.25 | Acc@0.50 | Mean IoU |
|---|---:|---:|---:|---:|
| focus12 | 12 | 0.1667 | 0.1667 | 0.1564 |

No random100 gate was run.

## What Changed

The rejected candidate added:

- a VG pack tool `record_final_selection(...)`;
- config flag `use_final_selection_ledger`;
- a `submit_final` pre-check requiring the latest ledger winner to match the
  submitted `proposal_id`;
- prompt/skill guidance asking the agent to record a winner, candidate set,
  evidence frames, and reject reasons immediately before final submission.

This was meant to move final evidence binding before `submit_final`, rather
than asking for a late confirmation after a wrong answer had already been
submitted. The tool worked mechanically, but did not improve the model's
ranking behavior.

## Commands

Focus12:

```bash
tmux new-session -d -s v3p18_focus12 \
  'cd /Users/bytedance/project/3DVLMReasoning && \
  CVRA_DISABLE=1 PYTHONPATH=src .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/v3p17_final_revision_guard_focus12_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p18_final_selection_ledger_focus12_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 3 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard \
  --use-final-selection-ledger \
  2>&1 | tee tmp/v3p18_final_selection_ledger_focus12_eval.log'
```

Aggregation-GT rescoring:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/v3p18_final_selection_ledger_focus12_eval/side_by_side.json \
  --output-dir tmp/v3p18_final_selection_ledger_focus12_eval_agg_gt
```

Leaderboard slicing:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.scanrefer_leaderboard_metrics \
  --side-by-side tmp/v3p18_final_selection_ledger_focus12_eval_agg_gt/side_by_side.json \
  --sample-ids tmp/v3p17_final_revision_guard_focus12_sample_ids.json \
  --output tmp/v3p18_final_selection_ledger_focus12_eval_agg_gt/leaderboard_metrics.json
```

SQLite ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/v3p18_final_selection_ledger_focus12_eval_agg_gt \
  --run-id v3p18_final_selection_ledger_focus12_20260508 \
  --branch feat/scanrefer-v3p10-stage-audit \
  --commit 8acd3da \
  --backend pack_v1 \
  --judge-model none \
  --db docs/benchmark/scanrefer/runs.sqlite \
  --leaderboard-metrics tmp/v3p18_final_selection_ledger_focus12_eval_agg_gt/leaderboard_metrics.json \
  --keyframe-mode query_driven \
  --notes "v3.18 focus12 final-selection ledger tool/guard; rejected: focus12 16.67/16.67 below v3.11/v3.13/v3.17"
```

## Raw Artifacts

- Focus12 sample list:
  `tmp/v3p17_final_revision_guard_focus12_sample_ids.json`
- Focus12 raw output: `tmp/v3p18_final_selection_ledger_focus12_eval/`
- Focus12 aggregation-GT rescore:
  `tmp/v3p18_final_selection_ledger_focus12_eval_agg_gt/`
- Focus12 console log: `tmp/v3p18_final_selection_ledger_focus12_eval.log`
- SQLite DB: `docs/benchmark/scanrefer/runs.sqlite`

## Evidence

Aggregation-GT rescoring:

```text
focus12: n=12 mean_iou=0.1564 acc25=0.1667 acc50=0.1667
```

Leaderboard slicing:

```text
n_total=12 Unique=3 Multiple=9
acc25 overall=0.1667 unique=0.3333 multiple=0.1111
acc50 overall=0.1667 unique=0.3333 multiple=0.1111
```

Matched comparison:

| Sample | v3.11 pid / IoU | v3.13 pid / IoU | v3.17 pid / IoU | v3.18 pid / IoU |
|---|---:|---:|---:|---:|
| `scene0203_00::14::3` | 73 / 0.0000 | 2 / 0.9759 | 2 / 0.9759 | 2 / 0.9759 |
| `scene0342_00::17::2` | 8 / 0.0000 | 13 / 0.0000 | 18 / 0.0000 | 13 / 0.0000 |
| `scene0377_00::3::1` | 4 / 0.8658 | 4 / 0.8658 | 37 / 0.0000 | 41 / 0.0000 |
| `scene0414_00::5::0` | 5 / 0.0000 | 15 / 0.8160 | 5 / 0.0000 | 49 / 0.0000 |
| `scene0426_00::3::2` | 2 / 0.9648 | 2 / 0.9648 | 2 / 0.9648 | 13 / 0.0000 |
| `scene0565_00::31::0` | 31 / 0.0000 | 22 / 0.4833 | 22 / 0.4833 | 55 / 0.0315 |
| `scene0593_00::15::3` | 11 / 0.8697 | 11 / 0.8697 | 11 / 0.8697 | 11 / 0.8697 |
| `scene0595_00::1::3` | 12 / 0.2907 | 9 / 0.2901 | 12 / 0.2907 | 18 / 0.0000 |
| `scene0645_00::16::4` | 5 / 0.0000 | 5 / 0.0000 | 5 / 0.0000 | 5 / 0.0000 |
| `scene0690_00::18::2` | 11 / 0.0396 | 20 / 0.0000 | 8 / 0.0000 | 8 / 0.0000 |
| `scene0693_00::0::2` | 34 / 0.0000 | 6 / 0.0000 | 6 / 0.0000 | 6 / 0.0000 |
| `scene0699_00::11::3` | 6 / 0.8333 | 42 / 0.0000 | 42 / 0.0000 | 42 / 0.0000 |

Same-slice aggregates:

```text
v3.11: mean_iou=0.3220 acc25=0.4167 acc50=0.3333
v3.13: mean_iou=0.4388 acc25=0.5833 acc50=0.4167
v3.17: mean_iou=0.2987 acc25=0.4167 acc50=0.2500
v3.18: mean_iou=0.1564 acc25=0.1667 acc50=0.1667
```

Ledger audit:

```text
All 12 samples called record_final_selection.
Each sample had 2-4 ledger calls.
FINAL_SELECTION_LEDGER_* submit blocks: 0.
```

The ledger therefore changed the reasoning trajectory, not just the final
chassis gate. It made the answer more verbose and auditable, but it did not
make the winner more correct.

SQLite reproduction:

```sql
SELECT run_id, n_total, n_unique, n_multiple,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', mean_iou_overall) AS miou,
       keyframe_mode
FROM runs
WHERE run_id = 'v3p18_final_selection_ledger_focus12_20260508';
```

Expected:

```text
v3p18_final_selection_ledger_focus12_20260508|12|3|9|0.1667|0.1667|0.1564|query_driven
```

## Decision

Reject and revert. A generic ledger prompt is not enough: it makes the agent
explain its current choice, but does not supply a better scoring function. The
next useful direction should be code-side ranking/evidence computation for
ambiguous same-category candidates, not another generic reflection tool.
