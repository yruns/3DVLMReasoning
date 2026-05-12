# ScanRefer v3.15 - cited-frame inspected-visibility focus gate (REJECTED)

**Branch**: `feat/scanrefer-v3p10-stage-audit`
**Tip commit at harvest time**: `8acd3da` (working tree dirty; candidate code
was later reverted from the default path)
**Run IDs** (SQLite): `v3p15_cited_frame_guard_focus1_20260508`,
`v3p15_cited_frame_guard_focus10_20260508`
**Date harvested**: 2026-05-08 GMT+8

## Headline

v3.15 tested a narrow EFG branch for stale final rationales that cite a frame
where the submitted proposal's own inspected `frames_appeared` never places
that proposal, while already-inspected same-category alternatives do appear in
that cited frame.

It recovered the motivating `scene0203_00::14::3` pillow sample, but failed the
10-case focused control. The candidate is rejected and excluded from the
leaderboard table.

| Run | n | Acc@0.25 | Acc@0.50 | Mean IoU |
|---|---:|---:|---:|---:|
| focus1 | 1 | 1.00 | 1.00 | 0.9759 |
| focus10 | 10 | 0.20 | 0.20 | 0.1941 |

On the same focus10 set, v3.11 was also `0.20 / 0.20`, while the existing v3.13
random100 trace was `0.50 / 0.40`. This made the patch too fragile to justify a
random100 gate.

## What Changed

The rejected candidate added an EFG branch in
`src/agents/skills/evidence_frame_guard.py`:

- parse `inspect_proposal(...)` trace records into proposal `frames_appeared`;
- when a final rationale cites frame ids, block the submitted proposal if none
  of those cited frames appear in its inspected metadata;
- only fire if a same-category candidate from the agent's own
  `find_proposals_by_category(...)` trace was also inspected and does appear in
  one of the cited frames.

This was motivated by `scannet/scene0203_00::14::3`. v3.11 selected pillow
proposal `73` with IoU `0.0` while citing "frame 144" evidence; inspected
metadata for proposal `73` did not include frame `144`, while pillow proposals
`2` and `39` did.

The candidate unit test passed and the one-sample focus run recovered the case,
but the 10-case focus gate was not positive. The candidate code, test, and
`chassis_tools_version=19` bump were removed after the failed focus gate.
`chassis_tools_version` is restored to `18`.

## Commands

Focus1:

```bash
tmux new-session -d -s v3p15_focus_scene0203 \
  'cd /Users/bytedance/project/3DVLMReasoning && \
  CVRA_DISABLE=1 PYTHONPATH=src .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/v3p15_cited_frame_focus1_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p15_cited_frame_focus1_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 1 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard \
  2>&1 | tee tmp/v3p15_cited_frame_focus1_eval.log'
```

Focus10:

```bash
tmux new-session -d -s v3p15_focus10 \
  'cd /Users/bytedance/project/3DVLMReasoning && \
  CVRA_DISABLE=1 PYTHONPATH=src .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/v3p15_cited_frame_focus10_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p15_cited_frame_focus10_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 4 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard \
  2>&1 | tee tmp/v3p15_cited_frame_focus10_eval.log'
```

Aggregation-GT rescoring:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/v3p15_cited_frame_focus1_eval/side_by_side.json \
  --output-dir tmp/v3p15_cited_frame_focus1_eval_agg_gt

PYTHONPATH=src .venv/bin/python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/v3p15_cited_frame_focus10_eval/side_by_side.json \
  --output-dir tmp/v3p15_cited_frame_focus10_eval_agg_gt
```

Leaderboard slicing:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.scanrefer_leaderboard_metrics \
  --side-by-side tmp/v3p15_cited_frame_focus1_eval_agg_gt/side_by_side.json \
  --sample-ids tmp/v3p15_cited_frame_focus1_sample_ids.json \
  --output tmp/v3p15_cited_frame_focus1_eval_agg_gt/leaderboard_metrics.json

PYTHONPATH=src .venv/bin/python -m evaluation.scripts.scanrefer_leaderboard_metrics \
  --side-by-side tmp/v3p15_cited_frame_focus10_eval_agg_gt/side_by_side.json \
  --sample-ids tmp/v3p15_cited_frame_focus10_sample_ids.json \
  --output tmp/v3p15_cited_frame_focus10_eval_agg_gt/leaderboard_metrics.json
```

SQLite ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/v3p15_cited_frame_focus1_eval_agg_gt \
  --run-id v3p15_cited_frame_guard_focus1_20260508 \
  --branch feat/scanrefer-v3p10-stage-audit \
  --commit 8acd3da \
  --leaderboard-metrics tmp/v3p15_cited_frame_focus1_eval_agg_gt/leaderboard_metrics.json \
  --db docs/benchmark/scanrefer/runs.sqlite \
  --keyframe-mode query_driven \
  --notes "v3.15 REJECTED focus1: inspected-visibility cited-frame guard rescued scene0203 from stale pid 73 to correct pillow pid; not promoted because focus10 control was fragile/negative and candidate was reverted from default path."

PYTHONPATH=src .venv/bin/python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/v3p15_cited_frame_focus10_eval_agg_gt \
  --run-id v3p15_cited_frame_guard_focus10_20260508 \
  --branch feat/scanrefer-v3p10-stage-audit \
  --commit 8acd3da \
  --leaderboard-metrics tmp/v3p15_cited_frame_focus10_eval_agg_gt/leaderboard_metrics.json \
  --db docs/benchmark/scanrefer/runs.sqlite \
  --keyframe-mode query_driven \
  --notes "v3.15 REJECTED focus10: inspected-visibility cited-frame guard fixed scene0203 but focus10 landed 20/20, tied v3.11 on this slice and far below v3.13 slice; candidate reverted from default path."
```

## Raw Artifacts

- Focus1 sample list: `tmp/v3p15_cited_frame_focus1_sample_ids.json`
- Focus1 raw output: `tmp/v3p15_cited_frame_focus1_eval/`
- Focus1 aggregation-GT rescore:
  `tmp/v3p15_cited_frame_focus1_eval_agg_gt/`
- Focus1 console log: `tmp/v3p15_cited_frame_focus1_eval.log`
- Focus10 sample list: `tmp/v3p15_cited_frame_focus10_sample_ids.json`
- Focus10 raw output: `tmp/v3p15_cited_frame_focus10_eval/`
- Focus10 aggregation-GT rescore:
  `tmp/v3p15_cited_frame_focus10_eval_agg_gt/`
- Focus10 console log: `tmp/v3p15_cited_frame_focus10_eval.log`
- SQLite DB: `docs/benchmark/scanrefer/runs.sqlite`

## Evidence

Aggregation-GT rescoring:

```text
focus1:  n=1  mean_iou=0.9759  acc25=1.0000  acc50=1.0000
focus10: n=10 mean_iou=0.1941  acc25=0.2000  acc50=0.2000
```

Leaderboard slicing:

```text
focus1:  n_total=1  Unique=0  Multiple=1  overall=1.0000 / 1.0000
focus10: n_total=10 Unique=1  Multiple=9  overall=0.2000 / 0.2000
```

Focus10 matched comparison against existing v3.11 / v3.13 artifacts:

| Sample | v3.11 pid / IoU | v3.13 pid / IoU | v3.15 pid / IoU |
|---|---:|---:|---:|
| `scene0552_00::14::0` | 24 / 0.0000 | 6 / 0.9990 | 24 / 0.0000 |
| `scene0678_00::34::3` | 30 / 0.0000 | 30 / 0.0000 | 7 / 0.0000 |
| `scene0203_00::14::3` | 73 / 0.0000 | 2 / 0.9759 | 2 / 0.9759 |
| `scene0693_00::0::2` | 34 / 0.0000 | 6 / 0.0000 | 6 / 0.0000 |
| `scene0618_00::4::0` | 3 / 0.0000 | 1 / 0.0000 | 3 / 0.0000 |
| `scene0426_00::3::2` | 2 / 0.9648 | 2 / 0.9648 | 2 / 0.9648 |
| `scene0095_00::25::1` | 18 / 0.0000 | 18 / 0.0000 | 18 / 0.0000 |
| `scene0342_00::17::2` | 8 / 0.0000 | 13 / 0.0000 | 18 / 0.0000 |
| `scene0558_00::33::2` | 15 / 0.0069 | 35 / 0.3311 | 34 / 0.0000 |
| `scene0377_00::9::2` | 27 / 0.9844 | 27 / 0.9844 | 15 / 0.0000 |

The new inspected-visibility branch fired on `scene0203_00::14::3`:

```text
EVIDENCE_FRAME_GUARD: rationale cites frame(s) 144 as final evidence,
but inspected metadata for submitted proposal 71 does not include any of those
frame(s) in `frames_appeared`. Already-inspected same-category proposal(s)
do appear in the cited frame(s): 2:pillow, 39:pillow.
```

The existing broader frame-citation EFG also fired on `scene0678_00::34::3`,
but that sample remained IoU `0.0`.

SQLite reproduction:

```sql
SELECT run_id, n_total, n_unique, n_multiple,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', mean_iou_overall) AS miou,
       keyframe_mode
FROM runs
WHERE run_id IN (
  'v3p15_cited_frame_guard_focus1_20260508',
  'v3p15_cited_frame_guard_focus10_20260508'
)
ORDER BY run_id;
```

Expected:

```text
v3p15_cited_frame_guard_focus10_20260508|10|1|9|0.2000|0.2000|0.1941|query_driven
v3p15_cited_frame_guard_focus1_20260508|1|0|1|1.0000|1.0000|0.9759|query_driven
```

Tool calls ingested:

```text
v3p15_cited_frame_guard_focus10_20260508|342
v3p15_cited_frame_guard_focus1_20260508|34
```

## Decision

Do not promote v3.15. The one-sample recovery is real, trace-only, and
non-GT, but it was already recovered in the existing v3.13 random100 artifact
and did not carry a positive 10-case control. The correct next direction is not
another after-the-fact EFG branch; it is a more explicit final evidence ledger
or a relation tool contract that distinguishes verified relation evidence from
proposal-only appearance evidence before final selection.
