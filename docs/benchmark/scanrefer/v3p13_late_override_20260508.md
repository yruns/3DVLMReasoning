# ScanRefer v3.13 - late left/right override guard (NEGATIVE)

**Branch**: `feat/scanrefer-v3p10-stage-audit`
**Tip commit at harvest time**: `8acd3da` (working tree dirty; v3.13 code
changes not committed and later reverted from the default path)
**Run ID** (SQLite): `v3p13_late_left_right_override_random100_20260508`
**Date harvested**: 2026-05-08 GMT+8

## Headline

v3.13 is a negative frozen-random100 gate. It attempted to stop late
left/right TADG overrides from replacing an already accepted relation-ranked
answer, but the new branch did not trigger on the random100 run. The full gate
ties v3.11 on Acc@0.25, lowers Acc@0.50 by -2pp, and slightly lowers mean IoU.
v3.11 remains the current honest query-driven random100 headline.

| Split | Acc@0.25 | Acc@0.50 |
|---|---:|---:|
| Unique | 75.00 | 75.00 |
| Multiple | 45.83 | 37.50 |
| Overall | 54.00 | 48.00 |

Mean IoU is `0.4601`. Compared with v3.11 on the same frozen random100 fold,
v3.13 has mean IoU delta `-0.0014`, Acc@0.25 recoveries/regressions `8/8`,
and Acc@0.50 recoveries/regressions `6/8`.

## What Changed

The candidate patch added a narrow TADG rule for `left_of` / `right_of`
rank-mismatch overrides: if the relation-ranked proposal had already been
accepted by `submit_final`, a later override should not replace it without new
relation-ranked evidence.

Implementation notes:

- `src/agents/skills/tadg.py` added a helper to inspect prior successful
  `submit_final` calls in `runtime.tool_trace`.
- `_should_reject_override` blocked late left/right rank-mismatch overrides
  when the current rank-1 proposal had already been submitted successfully.
- `src/agents/core/agent_config.py` temporarily bumped `chassis_tools_version`
  to `20`.
- Unit coverage was added for the late-left/right override case.

The patch was motivated by `scene0412_00::13::0`, where v3.11 accepted the
correct relation-ranked trash can early and later overwrote it with a worse
proposal after deferred visual evidence. In the full random100 run, however,
the new "already accepted" branch triggered `0` times, so observed metric
changes are stochastic/runtime trajectory effects rather than proof that the
new rule helps.

## Commands

Fresh random100 run:

```bash
tmux new-session -d -s v3p13_random100 \
  "cd /Users/bytedance/project/3DVLMReasoning && \
  CVRA_DISABLE=1 PYTHONPATH=src .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p13_late_override_random100_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 4 \
  --sample-retries 4 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard \
  2>&1 | tee tmp/v3p13_late_override_random100_eval.log"
```

Aggregation-GT rescore:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/v3p13_late_override_random100_eval/side_by_side.json \
  --output-dir tmp/v3p13_late_override_random100_eval_agg_gt
```

Leaderboard slicing:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.scanrefer_leaderboard_metrics \
  --side-by-side tmp/v3p13_late_override_random100_eval_agg_gt/side_by_side.json \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --output tmp/v3p13_late_override_random100_eval_agg_gt/leaderboard_metrics.json
```

SQLite ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/v3p13_late_override_random100_eval_agg_gt \
  --run-id v3p13_late_left_right_override_random100_20260508 \
  --branch feat/scanrefer-v3p10-stage-audit \
  --commit 8acd3da \
  --leaderboard-metrics tmp/v3p13_late_override_random100_eval_agg_gt/leaderboard_metrics.json \
  --db docs/benchmark/scanrefer/runs.sqlite \
  --keyframe-mode query_driven \
  --notes "v3.13 NEGATIVE: attempted to reject late left/right overrides after relation-ranked proposal was already accepted; new branch did not trigger on random100; no GT at inference; regressed vs v3.11."
```

## Raw Artifacts

- Frozen sample list: `tmp/scanrefer_artifacts/random100_sample_ids.json`
- Raw output: `tmp/v3p13_late_override_random100_eval/`
- Aggregation-GT rescore:
  `tmp/v3p13_late_override_random100_eval_agg_gt/`
- Console log: `tmp/v3p13_late_override_random100_eval.log`
- SQLite DB: `docs/benchmark/scanrefer/runs.sqlite`
- Focus set used before the full gate:
  `tmp/v3p13_late_override_focus6_sample_ids.json`

## Evidence

Aggregation-GT rescore:

```text
n=100 mean_iou=0.4601 acc25=0.5400 acc50=0.4800
stats: missing_scene=0 missing_target=0 no_prediction=0 completed_new_gt=100
```

Leaderboard metrics:

```text
n_total=100 (Unique=28, Multiple=72)
acc25  overall=0.5400 unique=0.7500 multiple=0.4583
acc50  overall=0.4800 unique=0.7500 multiple=0.3750
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
  'v3p11_tadg_anchor_exclusion_random100_20260508',
  'v3p12_vertical_relation_guard_random100_20260508',
  'v3p13_late_left_right_override_random100_20260508'
)
ORDER BY run_id;
```

Expected:

```text
v3p11_tadg_anchor_exclusion_random100_20260508|100|0.5400|0.5000|0.4615|query_driven
v3p12_vertical_relation_guard_random100_20260508|100|0.5400|0.4800|0.4559|query_driven
v3p13_late_left_right_override_random100_20260508|100|0.5400|0.4800|0.4601|query_driven
```

Trigger check:

```text
new v3.13 "already accepted" strict branch trigger count: 0 samples
all TADG_STRICT trigger count in v3.13 traces: 6 samples
tool_calls ingested: 2628
```

Paired random100 deltas vs v3.11:

```text
Acc@0.25 recoveries=8, regressions=8
Acc@0.50 recoveries=6, regressions=8
Mean IoU delta=-0.0014
```

For context, paired deltas vs v3.12:

```text
Acc@0.25 recoveries=9, regressions=9
Acc@0.50 recoveries=7, regressions=7
Mean IoU delta=+0.0042
```

Largest IoU recoveries vs v3.11:

| Sample | v3.11 IoU | v3.13 IoU | Delta |
|---|---:|---:|---:|
| `scannet/scene0354_00::5::1` | 0.0000 | 1.0000 | +1.0000 |
| `scannet/scene0552_00::14::0` | 0.0000 | 0.9990 | +0.9990 |
| `scannet/scene0203_00::14::3` | 0.0000 | 0.9759 | +0.9759 |
| `scannet/scene0412_00::13::0` | 0.0479 | 1.0000 | +0.9521 |
| `scannet/scene0414_00::5::0` | 0.0000 | 0.8160 | +0.8160 |

Largest IoU regressions vs v3.11:

| Sample | v3.11 IoU | v3.13 IoU | Delta |
|---|---:|---:|---:|
| `scannet/scene0660_00::4::4` | 1.0000 | 0.0000 | -1.0000 |
| `scannet/scene0704_00::8::3` | 0.9961 | 0.0000 | -0.9961 |
| `scannet/scene0629_00::3::3` | 0.9643 | 0.0000 | -0.9643 |
| `scannet/scene0050_00::10::4` | 0.8575 | 0.0000 | -0.8575 |
| `scannet/scene0699_00::11::3` | 0.8333 | 0.0000 | -0.8333 |

## Focus-Set Pre-Gate

Before the random100 gate, the patch was checked on a 6-sample late-override
focus/control set:

```text
n=6 mean_iou=0.2472 acc25=0.3333 acc50=0.1667
```

On the same 6 ids, v3.11 scored Acc@0.25 / Acc@0.50 = `0.3333 / 0.3333`,
mean IoU `0.3135`; v3.12 scored `0.3333 / 0.1667`, mean IoU `0.2552`.
The focus run recovered `scene0412_00::13::0`, but the new strict branch did
not fire there either, and stochastic/control regressions erased the gain.

## Decision

Do not promote v3.13. The intended branch did not trigger on the full frozen
random100 gate, and the observed random100 result is worse than v3.11 on
Acc@0.50 and mean IoU. The code change should not remain in the default path.

The useful lesson is narrower: late overwrite can be a real failure mode, but
detecting it by scanning prior `submit_final` calls is too brittle. Future work
should make the final evidence ledger explicit and require the final answer to
bind to the latest relation-ranked winner/reject-reason evidence, instead of
adding another opportunistic guard after the fact.

## Caveats

- This is still the frozen random100 development fold, not full 9508-val.
- The run used query-driven Stage 1 and the Mask3D proposal pool; no GT was
  used at inference. GT is used only after final submission for aggregation-GT
  rescoring.
- The tip commit is unchanged from v3.11/v3.12 because these are dirty-tree
  experimental changes. The per-version doc and SQLite row are therefore the
  durable audit trail for this negative result.
- Since the new branch triggered `0` times, the metric deltas should not be
  attributed to the intended guard. They are evidence that the patch is not
  worth keeping, not evidence that the late-overwrite failure mode is solved.
