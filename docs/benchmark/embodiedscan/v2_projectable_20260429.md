# v2 projectable pack_v1 smoke

**Date:** 2026-04-29
**Branch:** `feat/explore_3dbbox`
**Run-time tip:** `370b12e` plus dirty EmbodiedScan prep/runner fixes
**Eval scale:** 70 frozen samples requested, 67 prepared and run
**Proposal source:** EmbodiedScan GT instances from `embodiedscan_infos_val.pkl`
**Judge:** none; programmatic oriented 9-DoF 3D IoU
**Raw artifacts:** `tmp/embodiedscan_eval_projectable_w12_20260429_150000/`
**Run log:** `/tmp/embodiedscan_eval_projectable_w12_20260429_150000.log`
**Frozen list:** `tmp/embodiedscan_artifacts/v1_prepared_projectable_20260429_sample_ids.json`
**Prep validation:** `tmp/embodiedscan_artifacts/prepare_validation_projectable_20260429.json`
**Visual validation:** `tmp/embodiedscan_visual_validation_projectable_20260429/`
**SQLite:** `docs/benchmark/embodiedscan/runs.sqlite`, run_id `v2_projectable_w12_67`

## What Changed

This version is the corrected projectable-only rerun of the v1 GT-pool smoke:

- `visible_instance_ids` are interpreted as indices into the official
  `instances` list, not as `bbox_id` values.
- Duplicate `bbox_id` targets are filtered out instead of being overwritten.
- Prepared samples keep only keyframes where the target is visible and
  projectable by the pack-v1 2D marking path.
- Stale per-sample artifacts are removed when a requested target is skipped.
- The side-by-side runner records per-sample backend exceptions as `status=error`
  rows and continues the batch.
- The evaluation was launched with `--workers 12`; all 67 samples completed with
  no recorded sample errors.

## Commands

Preparation:

```bash
PYTHONPATH=src python -m evaluation.scripts.prepare_pack_v1_inputs \
    --sample-ids tmp/embodiedscan_artifacts/v1_70_sample_ids.json \
    --data-root data/embodiedscan \
    --split val 2>&1 | tee /tmp/prepare_embodiedscan_pack_v1_projectable_20260429.log
```

Run:

```bash
tmux new-session -d -s vg-projectable-w12 \
  "cd /Users/bytedance/project/3DVLMReasoning && \
   source .venv/bin/activate && \
   PYTHONPATH=src python -m evaluation.scripts.run_embodiedscan_vg_side_by_side \
     --sample-ids tmp/embodiedscan_artifacts/v1_prepared_projectable_20260429_sample_ids.json \
     --data-root data/embodiedscan \
     --output-dir tmp/embodiedscan_eval_projectable_w12_20260429_150000 \
     --workers 12 \
     --sample-retries 2 \
     2>&1 | tee /tmp/embodiedscan_eval_projectable_w12_20260429_150000.log"
```

Ingest:

```bash
PYTHONPATH=src python scripts/ingest_embodiedscan_run.py \
    --output-dir tmp/embodiedscan_eval_projectable_w12_20260429_150000 \
    --run-id v2_projectable_w12_67 \
    --branch feat/explore_3dbbox \
    --commit 370b12e \
    --notes "Projectable-only pack_v1 smoke; corrected visible-instance mapping, duplicate bbox filtering, and 12-worker run with sample error isolation" \
    --db docs/benchmark/embodiedscan/runs.sqlite
```

## Headline Metrics

| n | mean IoU | Acc@0.25 | Acc@0.50 | completed | error | misses |
|--:|---------:|---------:|---------:|----------:|------:|-------:|
| 67 | 94.08 | 94.03 | 94.03 | 67 | 0 | 4 |

SQLite reproduction:

```sql
SELECT run_id, n,
       printf('%.4f', mean_iou) AS mean_iou,
       printf('%.4f', acc25) AS acc25,
       printf('%.4f', acc50) AS acc50
FROM runs
WHERE run_id='v2_projectable_w12_67';
```

Misses below Acc@0.25:

| sample_id | selected | IoU | query |
|-----------|---------:|----:|-------|
| `scene0111_00::11` | 9 | 0.0000 | choose the cabinet that is next to the stove |
| `scene0509_01::16` | 14 | 0.0000 | facing the front of the toothbrush, select the towel that is on the left of it |
| `scene0006_01::36` | 43 | 0.0000 | facing the front of the desk, select the clothes that are on the right of it |
| `scene0614_01::19` | 32 | 0.0343 | facing the front of the blackboard, select the cabinet that is on the right of it |

## Prep Validation

The projectable preparation audit reported:

| requested | prepared | skipped | scenes | projection bad | errors |
|----------:|---------:|--------:|-------:|---------------:|-------:|
| 70 | 67 | 3 | 47 | 0 | 0 |

Skipped targets:

- `scene0143_02::35`: no valid EmbodiedScan VG sample after duplicate/ambiguous
  target filtering.
- `scene0236_00::23`: no valid EmbodiedScan VG sample after duplicate/ambiguous
  target filtering.
- `scene0451_00::5`: no visible frames for the target.

Visual validation artifacts:

- `tmp/embodiedscan_visual_validation_projectable_20260429/target_highlight_grid.png`
- `tmp/embodiedscan_visual_validation_projectable_20260429/raw_vs_annotated_pairs_page1.png`
- `tmp/embodiedscan_visual_validation_projectable_20260429/raw_vs_annotated_pairs_page2.png`
- `tmp/embodiedscan_visual_validation_projectable_20260429/raw_vs_annotated_pairs_page3.png`
- `tmp/embodiedscan_visual_validation_projectable_20260429/manifest.json`

Manual inspection confirmed the target-highlight pages had visible yellow
`TARGET <id>` overlays and no null-projection panels.

## Cross-Version Comparison

| Version | Fold | n | mean IoU | Acc@0.25 | Acc@0.50 | Notes |
|---------|------|--:|---------:|---------:|---------:|-------|
| v1 | original prepared smoke | 68 | 91.18 | 91.18 | 91.18 | included stale/incorrectly prepared targets |
| v2 | projectable-only corrected smoke | 67 | 94.08 | 94.03 | 94.03 | corrected visibility/projection/filtering |

These are not a matched-fold comparison because v2 filters three targets from
the original 70 and v1 used an earlier prepared list.

## Caveats

- This remains a smoke run, not a benchmark claim; n=67 is below the repo's
  n>=200 threshold for quality claims.
- The GT proposal pool removes detector recall/precision as a variable. The
  metric is top-1 candidate selection accuracy over oracle annotations.
- `tool_calls` and `llm_calls` are still not populated in SQLite; the current
  side-by-side JSON stores per-sample predictions and aggregate metrics only.
- The run used a dirty working tree containing the fixes listed above. Preserve
  those source changes with the run record before treating this as reproducible
  from a clean commit.

## Next

- Add incremental per-sample checkpoint output before increasing concurrency
  beyond 12 workers, so interrupted runs can resume without duplicating LLM
  spend.
- Run a larger frozen set of at least 200 projectable samples before quoting a
  headline EmbodiedScan number.
- Persist per-sample tool traces and token usage so SQLite can support deeper
  regression analysis.
