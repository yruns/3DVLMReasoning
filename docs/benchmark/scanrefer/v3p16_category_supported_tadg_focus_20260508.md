# ScanRefer v3.16 - category-supported TADG focus gate (REJECTED)

**Branch**: `feat/scanrefer-v3p10-stage-audit`
**Tip commit at harvest time**: `8acd3da` (working tree dirty; candidate code
was later reverted from the default path)
**Run ID** (SQLite): `v3p16_category_supported_tadg_focus6_20260508`
**Date harvested**: 2026-05-08 GMT+8

## Headline

v3.16 tested a narrow TADG rule for `not_in_candidates` overrides: if a
relation-ranked proposal appeared in a non-empty category lookup, but the
submitted proposal never appeared in any non-empty category lookup in the
agent's own trace, reject the override as an unsupported target-category
switch.

The focus6 gate is rejected. It tied v3.11 on threshold accuracy for this
slice but lowered mean IoU, and the new branch did not fire on the motivating
fresh trace.

| Run | n | Acc@0.25 | Acc@0.50 | Mean IoU |
|---|---:|---:|---:|---:|
| focus6 | 6 | 0.50 | 0.50 | 0.4294 |

No random100 gate was run.

## What Changed

The rejected candidate added a TADG branch in `src/agents/skills/tadg.py`:

- scan the recent `find_proposals_by_category(...)` trace;
- treat non-empty category lookup results as the agent's own category-support
  evidence;
- when a final answer uses a `not_in_candidates` override, block it if the
  submitted proposal was absent from every non-empty category lookup while the
  relation-ranked top proposal appeared in one.

This was motivated by the v3.11 trace for `scannet/scene0149_00::21::2`, where
the agent submitted proposal `33` for black base cabinets left of the sink
while the relation-ranked candidate was proposal `56`. In the fresh v3.16
trace, however, the existing `TADG_STRICT` strong-2D branch fired first and
pushed the answer to proposal `27`; the new category-support branch did not
match.

The candidate unit tests passed, but the focus gate did not provide a positive
signal. The candidate helper, tests, and `chassis_tools_version=19` bump were
removed after harvest. `chassis_tools_version` is restored to `18`.

## Commands

Focus6:

```bash
tmux new-session -d -s v3p16_focus6 \
  'cd /Users/bytedance/project/3DVLMReasoning && \
  CVRA_DISABLE=1 PYTHONPATH=src .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/v3p16_category_supported_tadg_focus6_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p16_category_supported_tadg_focus6_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 3 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard \
  2>&1 | tee tmp/v3p16_category_supported_tadg_focus6_eval.log'
```

Aggregation-GT rescoring:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/v3p16_category_supported_tadg_focus6_eval/side_by_side.json \
  --output-dir tmp/v3p16_category_supported_tadg_focus6_eval_agg_gt
```

Leaderboard slicing:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.scanrefer_leaderboard_metrics \
  --side-by-side tmp/v3p16_category_supported_tadg_focus6_eval_agg_gt/side_by_side.json \
  --sample-ids tmp/v3p16_category_supported_tadg_focus6_sample_ids.json \
  --output tmp/v3p16_category_supported_tadg_focus6_eval_agg_gt/leaderboard_metrics.json
```

SQLite ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/v3p16_category_supported_tadg_focus6_eval_agg_gt \
  --run-id v3p16_category_supported_tadg_focus6_20260508 \
  --branch feat/scanrefer-v3p10-stage-audit \
  --commit 8acd3da \
  --leaderboard-metrics tmp/v3p16_category_supported_tadg_focus6_eval_agg_gt/leaderboard_metrics.json \
  --db docs/benchmark/scanrefer/runs.sqlite \
  --keyframe-mode query_driven \
  --notes "v3.16 REJECTED focus6: category-supported TADG not-in-candidates override guard did not fire on the motivating fresh trace; focus6 tied v3.11 at 50/50 but lowered mean IoU, candidate reverted from default path."
```

## Raw Artifacts

- Focus6 sample list:
  `tmp/v3p16_category_supported_tadg_focus6_sample_ids.json`
- Focus6 raw output: `tmp/v3p16_category_supported_tadg_focus6_eval/`
- Focus6 aggregation-GT rescore:
  `tmp/v3p16_category_supported_tadg_focus6_eval_agg_gt/`
- Focus6 console log: `tmp/v3p16_category_supported_tadg_focus6_eval.log`
- SQLite DB: `docs/benchmark/scanrefer/runs.sqlite`

## Evidence

Aggregation-GT rescoring:

```text
focus6: n=6 mean_iou=0.4294 acc25=0.5000 acc50=0.5000
```

Leaderboard slicing:

```text
n_total=6 Unique=1 Multiple=5
acc25 overall=0.5000 unique=0.0000 multiple=0.6000
acc50 overall=0.5000 unique=0.0000 multiple=0.6000
```

Matched comparison against existing v3.11 / v3.13 artifacts:

| Sample | v3.11 pid / IoU | v3.13 pid / IoU | v3.16 pid / IoU |
|---|---:|---:|---:|
| `scene0149_00::21::2` | 33 / 0.1131 | 27 / 0.0241 | 27 / 0.0241 |
| `scene0025_00::9::0` | 27 / 0.5524 | 34 / 0.0586 | 27 / 0.5524 |
| `scene0249_00::38::4` | 102 / 0.0000 | 42 / 0.0000 | 64 / 0.0000 |
| `scene0412_00::13::0` | 1 / 0.0479 | 3 / 1.0000 | 3 / 1.0000 |
| `scene0660_00::4::4` | 1 / 1.0000 | 5 / 0.0000 | 1 / 1.0000 |
| `scene0629_00::3::3` | 11 / 0.9643 | 34 / 0.0000 | 30 / 0.0000 |

Same-slice aggregates:

```text
v3.11: mean_iou=0.4463 acc25=0.5000 acc50=0.5000
v3.13: mean_iou=0.1805 acc25=0.1667 acc50=0.1667
v3.16: mean_iou=0.4294 acc25=0.5000 acc50=0.5000
```

Trigger audit:

```text
TADG_STRICT: override rejected for relation 'left_of' against proposal 7;
proposal 27 is the relation-ranked target and proposal 33 should not bypass
that result. The top-ranked proposal has strong 2D shared-frame support
(48 supporting vs 3 contradicting frames), while the submitted proposal has
only 11 supporting left/right frames for this anchor.
```

The new category-support branch did not trigger. The fresh trajectory changed
the failure from "unsupported not-in-candidates override" into an existing
strong-2D TADG path that still chose the wrong cabinet proposal for the
referring expression.

SQLite reproduction:

```sql
SELECT run_id, n_total, n_unique, n_multiple,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', mean_iou_overall) AS miou,
       keyframe_mode
FROM runs
WHERE run_id = 'v3p16_category_supported_tadg_focus6_20260508';
```

Expected:

```text
v3p16_category_supported_tadg_focus6_20260508|6|1|5|0.5000|0.5000|0.4294|query_driven
```

## Decision

Reject and revert. v3.16 confirms that another narrow post-hoc TADG guard is
not enough. The next candidate should move toward an explicit final-selection
ledger or relation contract that binds the final proposal to the actual
winner, candidate set, spatial evidence, and reject reasons before submission.
