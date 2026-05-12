# ScanRefer v3.11 - TADG same-category anchor-exclusion

**Branch**: `feat/scanrefer-v3p10-stage-audit`
**Tip commit at harvest time**: `8acd3da` (working tree dirty; v3.11 code
changes not yet committed)
**Run ID** (SQLite): `v3p11_tadg_anchor_exclusion_random100_20260508`
**Date harvested**: 2026-05-08 GMT+8

## Headline

v3.11 is a fresh frozen-random100 query-driven ScanRefer gate. It uses the
Mask3D/ConceptGraph proposal pool and no GT at inference; GT is used only for
offline aggregation-GT rescoring after the agent submits a proposal id.

| Split | Acc@0.25 | Acc@0.50 |
|---|---:|---:|
| Unique | 82.14 | 82.14 |
| Multiple | 43.06 | 37.50 |
| Overall | 54.00 | 50.00 |

Mean IoU is `0.4615`. Compared with v3.10 on the same frozen random100 fold,
v3.11 ties Overall@0.25, improves Overall@0.50 by +2pp, and improves mean IoU
by +0.0065. It remains below the Z3D zero-shot reference of 58.9 / 52.7.

## What Changed

TADG candidate coverage now ignores the current spatial anchor when checking
whether same-category candidates were omitted from `compare_proposals_spatial`.
This handles traces such as "chair next to another same chair", where the
anchor chair is intentionally excluded from `candidate_ids` and should not
trigger `TADG_CANDIDATE_COVERAGE`.

This patch was first tested on a 13-case paired v3.9/v3.10 flip set. The
TADG-only focus result improved mean IoU and fixed `scene0660_00::4::4`; a
separate EFG 2D-mark experiment was rejected and reverted because it lowered
focus-set Acc@0.50 and mean IoU. See
[`v3p11_anchor_exclusion_focus13_20260508.md`](v3p11_anchor_exclusion_focus13_20260508.md).

## Commands

Fresh random100 run:

```bash
tmux new-session -d -s v3p11_tadg_random100 \
  "cd /Users/bytedance/project/3DVLMReasoning && \
  CVRA_DISABLE=1 PYTHONPATH=src .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p11_tadg_anchor_exclusion_random100_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 4 \
  --sample-retries 4 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard \
  2>&1 | tee tmp/v3p11_tadg_anchor_exclusion_random100_eval.log"
```

Aggregation-GT rescore:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/v3p11_tadg_anchor_exclusion_random100_eval/side_by_side.json \
  --output-dir tmp/v3p11_tadg_anchor_exclusion_random100_eval_agg_gt
```

Leaderboard slicing:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.scanrefer_leaderboard_metrics \
  --side-by-side tmp/v3p11_tadg_anchor_exclusion_random100_eval_agg_gt/side_by_side.json \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --output tmp/v3p11_tadg_anchor_exclusion_random100_eval_agg_gt/leaderboard_metrics.json
```

SQLite ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/v3p11_tadg_anchor_exclusion_random100_eval_agg_gt \
  --run-id v3p11_tadg_anchor_exclusion_random100_20260508 \
  --branch feat/scanrefer-v3p10-stage-audit \
  --commit 8acd3da \
  --leaderboard-metrics tmp/v3p11_tadg_anchor_exclusion_random100_eval_agg_gt/leaderboard_metrics.json \
  --db docs/benchmark/scanrefer/runs.sqlite \
  --keyframe-mode query_driven \
  --notes "v3.11 TADG same-category anchor-exclusion random100 gate; no GT at inference; TADG-only after rejected EFG 2D-mark experiment."
```

## Raw Artifacts

- Frozen sample list: `tmp/scanrefer_artifacts/random100_sample_ids.json`
- Raw output: `tmp/v3p11_tadg_anchor_exclusion_random100_eval/`
- Aggregation-GT rescore:
  `tmp/v3p11_tadg_anchor_exclusion_random100_eval_agg_gt/`
- Console log: `tmp/v3p11_tadg_anchor_exclusion_random100_eval.log`
- SQLite DB: `docs/benchmark/scanrefer/runs.sqlite`

## Evidence

Aggregation-GT rescore:

```text
n=100 mean_iou=0.4615 acc25=0.5400 acc50=0.5000
stats: missing_scene=0 missing_target=0 no_prediction=0 completed_new_gt=100
```

Leaderboard metrics:

```text
n_total=100 (Unique=28, Multiple=72)
acc25  overall=0.5400 unique=0.8214 multiple=0.4306
acc50  overall=0.5000 unique=0.8214 multiple=0.3750
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
  'v3p11_tadg_anchor_exclusion_random100_20260508'
)
ORDER BY run_id;
```

Expected:

```text
v3p10_terminal_latch_random100_20260508|100|0.5400|0.4800|0.4550|query_driven
v3p11_tadg_anchor_exclusion_random100_20260508|100|0.5400|0.5000|0.4615|query_driven
```

Paired random100 deltas vs v3.10:

```text
Acc@0.25 recoveries=8, regressions=8
Acc@0.50 recoveries=8, regressions=6
Mean IoU delta=+0.0065
```

Largest IoU recoveries:

| Sample | v3.10 pid / IoU | v3.11 pid / IoU | Delta |
|---|---:|---:|---:|
| `scannet/scene0660_00::4::4` | 9 / 0.0000 | 1 / 1.0000 | +1.0000 |
| `scannet/scene0377_00::9::2` | 15 / 0.0000 | 27 / 0.9844 | +0.9844 |
| `scannet/scene0426_00::3::2` | 77 / 0.0000 | 2 / 0.9648 | +0.9648 |
| `scannet/scene0629_00::3::3` | 19 / 0.0000 | 11 / 0.9643 | +0.9643 |
| `scannet/scene0377_00::3::1` | 41 / 0.0000 | 4 / 0.8658 | +0.8658 |

Largest IoU regressions:

| Sample | v3.10 pid / IoU | v3.11 pid / IoU | Delta |
|---|---:|---:|---:|
| `scannet/scene0328_00::15::0` | 23 / 1.0000 | 4 / 0.0000 | -1.0000 |
| `scannet/scene0552_00::14::0` | 6 / 0.9990 | 24 / 0.0000 | -0.9990 |
| `scannet/scene0203_00::14::3` | 2 / 0.9759 | 73 / 0.0000 | -0.9759 |
| `scannet/scene0412_00::13::0` | 3 / 1.0000 | 1 / 0.0479 | -0.9521 |
| `scannet/scene0648_00::23::1` | 7 / 0.7745 | 22 / 0.0000 | -0.7745 |

## Caveats

- This is still the frozen random100 development fold, not full 9508-val.
- The improvement is modest and partly stochastic: Overall@0.25 is unchanged,
  Acc@0.50 improves through fewer high-threshold regressions than recoveries.
- Multiple@0.50 is still only 37.50, so same-category / spatial-anchor final
  selection remains the main bottleneck.
- v3.11 remains below Z3D's 58.9 / 52.7 zero-shot reference, so this is a
  new development headline, not a solved-state claim.
