# ScanRefer v3.12 - vertical-relation TADG guard (NEGATIVE)

**Branch**: `feat/scanrefer-v3p10-stage-audit`
**Tip commit at harvest time**: `8acd3da` (working tree dirty; v3.12 code
changes not yet committed)
**Run ID** (SQLite): `v3p12_vertical_relation_guard_random100_20260508`
**Date harvested**: 2026-05-08 GMT+8

## Headline

v3.12 is a negative frozen-random100 gate. It adds a TADG guard for target
`above` / `below` constraints, but the full random100 result regresses against
v3.11 on Acc@0.50 and mean IoU. It is retained as process evidence and should
not replace v3.11 as the current query-driven headline.

| Split | Acc@0.25 | Acc@0.50 |
|---|---:|---:|
| Unique | 71.43 | 71.43 |
| Multiple | 47.22 | 38.89 |
| Overall | 54.00 | 48.00 |

Mean IoU is `0.4559`. Compared with v3.11 on the same frozen random100 fold,
v3.12 ties Overall@0.25, lowers Overall@0.50 by -2pp, and lowers mean IoU by
`-0.0056`. v3.11 remains the current honest query-driven random100 headline.

## What Changed

TADG now detects reference text where the target itself is constrained by a
vertical relation, such as "cabinet below the countertop" or "object above the
washer". If the agent tries to submit a final proposal before running a
matching `compare_proposals_spatial` vertical comparison over a usable ambiguous
candidate set, `submit_final` is soft-blocked with
`TADG_VERTICAL_RELATION_UNTESTED`.

Implementation notes:

- `src/agents/skills/tadg.py` adds target vertical-pattern detection,
  submitted-candidate context extraction, and missing vertical-compare gap
  detection.
- Broad category lookups with too many ids no longer mask a narrower
  `compare_proposals_spatial` candidate set.
- Anchor-under-target phrasing such as "underneath it" is excluded from the
  target-vertical guard.
- `src/agents/core/agent_config.py` bumps `chassis_tools_version` to `19`.
- Unit tests cover no-category, broad-category, target-vertical, and
  anchor-under-target cases.

The patch was motivated by `scene0328_00::15::0`, where v3.11 selected a corner
cabinet without checking the query's "below the countertop" constraint. The
guard does fire on that trace, but the downstream `below` ranking still puts the
wrong proposal first; forcing a vertical compare is not sufficient to solve the
corner/anchor reasoning failure.

## Commands

Fresh random100 run:

```bash
tmux new-session -d -s v3p12_random100 \
  "cd /Users/bytedance/project/3DVLMReasoning && \
  CVRA_DISABLE=1 PYTHONPATH=src .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p12_vertical_guard3_random100_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 4 \
  --sample-retries 4 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard \
  2>&1 | tee tmp/v3p12_vertical_guard3_random100_eval.log"
```

Aggregation-GT rescore:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/v3p12_vertical_guard3_random100_eval/side_by_side.json \
  --output-dir tmp/v3p12_vertical_guard3_random100_eval_agg_gt
```

Leaderboard slicing:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.scanrefer_leaderboard_metrics \
  --side-by-side tmp/v3p12_vertical_guard3_random100_eval_agg_gt/side_by_side.json \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --output tmp/v3p12_vertical_guard3_random100_eval_agg_gt/leaderboard_metrics.json
```

SQLite ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/v3p12_vertical_guard3_random100_eval_agg_gt \
  --run-id v3p12_vertical_relation_guard_random100_20260508 \
  --branch feat/scanrefer-v3p10-stage-audit \
  --commit 8acd3da \
  --leaderboard-metrics tmp/v3p12_vertical_guard3_random100_eval_agg_gt/leaderboard_metrics.json \
  --db docs/benchmark/scanrefer/runs.sqlite \
  --keyframe-mode query_driven \
  --notes "v3.12 NEGATIVE: TADG target above/below missing-vertical-compare guard; no GT at inference; random100 regressed vs v3.11."
```

## Raw Artifacts

- Frozen sample list: `tmp/scanrefer_artifacts/random100_sample_ids.json`
- Raw output: `tmp/v3p12_vertical_guard3_random100_eval/`
- Aggregation-GT rescore:
  `tmp/v3p12_vertical_guard3_random100_eval_agg_gt/`
- Console log: `tmp/v3p12_vertical_guard3_random100_eval.log`
- SQLite DB: `docs/benchmark/scanrefer/runs.sqlite`
- Pre-gate vertical focus set: `tmp/v3p12_vertical12_sample_ids.json`

## Evidence

Aggregation-GT rescore:

```text
n=100 mean_iou=0.4559 acc25=0.5400 acc50=0.4800
stats: missing_scene=0 missing_target=0 no_prediction=0 completed_new_gt=100
```

Leaderboard metrics:

```text
n_total=100 (Unique=28, Multiple=72)
acc25  overall=0.5400 unique=0.7143 multiple=0.4722
acc50  overall=0.4800 unique=0.7143 multiple=0.3889
```

SQLite reproduction:

```sql
SELECT run_id, n_total,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', mean_iou_overall) AS miou,
       keyframe_mode
FROM runs
WHERE run_id IN (
  'v3p10_terminal_latch_random100_20260508',
  'v3p11_tadg_anchor_exclusion_random100_20260508',
  'v3p12_vertical_relation_guard_random100_20260508'
)
ORDER BY run_id;
```

Expected:

```text
v3p10_terminal_latch_random100_20260508|100|0.5400|0.4800|0.4550|query_driven
v3p11_tadg_anchor_exclusion_random100_20260508|100|0.5400|0.5000|0.4615|query_driven
v3p12_vertical_relation_guard_random100_20260508|100|0.5400|0.4800|0.4559|query_driven
```

Paired random100 deltas vs v3.11:

```text
Acc@0.25 recoveries=7, regressions=7
Acc@0.50 recoveries=4, regressions=6
Mean IoU delta=-0.0056
```

For context, paired deltas vs v3.10 are neutral at the threshold level:

```text
Acc@0.25 recoveries=9, regressions=9
Acc@0.50 recoveries=7, regressions=7
Mean IoU delta=+0.0009
```

Largest IoU recoveries vs v3.11:

| Sample | v3.11 pid / IoU | v3.12 pid / IoU | Delta |
|---|---:|---:|---:|
| `scannet/scene0203_00::14::3` | 73 / 0.0000 | 2 / 0.9759 | +0.9759 |
| `scannet/scene0618_00::4::0` | 3 / 0.0000 | 7 / 0.8435 | +0.8435 |
| `scannet/scene0653_00::31::0` | 9 / 0.0000 | 32 / 0.8104 | +0.8104 |
| `scannet/scene0100_00::26::0` | 34 / 0.0858 | 18 / 0.7175 | +0.6317 |
| `scannet/scene0565_00::31::0` | 31 / 0.0000 | 22 / 0.4833 | +0.4833 |

Largest IoU regressions vs v3.11:

| Sample | v3.11 pid / IoU | v3.12 pid / IoU | Delta |
|---|---:|---:|---:|
| `scannet/scene0660_00::4::4` | 1 / 1.0000 | 2 / 0.0000 | -1.0000 |
| `scannet/scene0629_00::3::3` | 11 / 0.9643 | 30 / 0.0000 | -0.9643 |
| `scannet/scene0474_00::15::3` | 6 / 0.8727 | 52 / 0.0000 | -0.8727 |
| `scannet/scene0377_00::3::1` | 4 / 0.8658 | 41 / 0.0000 | -0.8658 |
| `scannet/scene0699_00::11::3` | 6 / 0.8333 | 42 / 0.0000 | -0.8333 |

## Focus-Set Pre-Gate

Before the random100 gate, the patch was checked on a 12-sample vertical
relation slice:

```text
n=12 mean_iou=0.3559 acc25=0.5000 acc50=0.3333
```

On the same 12 ids, v3.11 scored Acc@0.25 / Acc@0.50 = `0.4167 / 0.3333`,
mean IoU `0.3206`; v3.10 scored `0.6667 / 0.5000`, mean IoU `0.5119`. The
focus set showed a small lift over v3.11 but remained below v3.10, so it was
only a candidate patch. The full random100 gate rejected it as a headline
change.

## Decision

Do not promote v3.12. The guard is logically defensible as trace hygiene, but
it perturbs the agent enough to lose more high-threshold successes than it
gains against v3.11. The next iteration should target the actual ranking
failure inside `compare_proposals_spatial` / final selection rather than only
requiring the agent to call the vertical relation tool.

## Caveats

- This is still the frozen random100 development fold, not full 9508-val.
- The run used query-driven Stage 1 and the Mask3D proposal pool; no GT was
  used at inference. GT is used only after final submission for aggregation-GT
  rescoring.
- The tip commit is unchanged from v3.11 because these are dirty-tree
  experimental changes. The per-version doc and SQLite row are therefore the
  durable audit trail for this negative result.
- The result demonstrates that guard trigger coverage is not enough. On
  `scene0328_00::15::0`, the new guard forces a `below` comparison, but the
  relation ranking still prefers the wrong cabinet. The remaining bottleneck is
  proposal ranking under compound spatial and visual constraints.
