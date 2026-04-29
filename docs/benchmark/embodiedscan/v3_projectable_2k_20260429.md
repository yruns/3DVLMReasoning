# v3 projectable 2k pack_v1 smoke

**Date:** 2026-04-29
**Branch:** `feat/explore_3dbbox`
**Run-time tip:** `370b12e` plus dirty EmbodiedScan prep/runner/checkpoint fixes
**Eval scale:** 2000 frozen projectable unique-target samples
**Proposal source:** EmbodiedScan GT instances from `embodiedscan_infos_val.pkl`
**Judge:** none; programmatic oriented 9-DoF 3D IoU
**Raw artifacts:** `tmp/embodiedscan_eval_v3_projectable_2k_w32_20260429_1730/`
**Run logs:** `/tmp/embodiedscan_eval_v3_projectable_2k_w32_20260429_1730.log`, `/tmp/embodiedscan_eval_v3_projectable_2k_w16_resume_20260429_1920.log`, `/tmp/embodiedscan_eval_v3_projectable_2k_w24_resume_20260429_1930.log`, `/tmp/embodiedscan_eval_v3_projectable_2k_w20_resume_20260429_2030.log`
**Frozen list:** `tmp/embodiedscan_artifacts/v3_projectable_2k_20260429_sample_ids.json`
**Prep validation:** `tmp/embodiedscan_artifacts/prepare_validation_v3_projectable_2k_20260429.json`
**Visual validation:** `tmp/embodiedscan_visual_validation_v3_projectable_2k_20260429/`
**SQLite:** `docs/benchmark/embodiedscan/runs.sqlite`, run_id `v3_projectable_2k_adaptive`

## What Changed

This version scales the corrected v2 projectable-only GT-pool smoke from 67
samples to 2000 samples and adds per-sample checkpoint/resume support to the
side-by-side runner.

- Prepared a local unique-target clean candidate list from 220 local ScanNet
  scenes after excluding one corrupt scene.
- Prepared 2013 projectable samples from 2015 clean candidates, then froze the
  first 2000 for this run.
- Added strict per-sample JSON checkpointing under
  `output_dir/per_sample/pack_v1/` so interrupted or throttled runs can resume.
- Started at 32 workers, then adapted concurrency after backend/network
  failures: 32 -> 16 -> 24 -> 20.
- Quarantined transient failed checkpoints before retrying:
  - 179 `APIConnectionError` checkpoints from the 32-worker phase.
  - 7 `BadRequestError` `invalid_prompt` checkpoints from the 24-worker phase.
- Final completed run has 2000 checkpoints, 0 `status=error` rows, and 10
  `status=failed` model outputs.

## Commands

Preparation helper:

```bash
source .venv/bin/activate
PYTHONPATH=src python -u tmp/embodiedscan_artifacts/prepare_v3_projectable_2k.py \
    2>&1 | tee /tmp/prepare_embodiedscan_pack_v1_v3_projectable_2k_20260429.log
```

Final frozen list:

```text
tmp/embodiedscan_artifacts/v3_projectable_2k_20260429_sample_ids.json
```

Visual validation helper:

```bash
source .venv/bin/activate
PYTHONPATH=src python -u tmp/embodiedscan_artifacts/visual_validate_v3_projectable_2k.py
```

Initial run:

```bash
tmux new-session -d -s embodiedscan-v3-2k-w32 \
  "cd /Users/bytedance/project/3DVLMReasoning && \
   source .venv/bin/activate && \
   PYTHONPATH=src python -m evaluation.scripts.run_embodiedscan_vg_side_by_side \
     --sample-ids tmp/embodiedscan_artifacts/v3_projectable_2k_20260429_sample_ids.json \
     --data-root data/embodiedscan \
     --output-dir tmp/embodiedscan_eval_v3_projectable_2k_w32_20260429_1730 \
     --workers 32 \
     --sample-retries 2 \
     2>&1 | tee /tmp/embodiedscan_eval_v3_projectable_2k_w32_20260429_1730.log"
```

Resumes used the same `--sample-ids`, `--data-root`, and `--output-dir`, with
`--workers 16`, then `--workers 24`, then `--workers 20`.

Ingest:

```bash
PYTHONPATH=src python scripts/ingest_embodiedscan_run.py \
    --output-dir tmp/embodiedscan_eval_v3_projectable_2k_w32_20260429_1730 \
    --run-id v3_projectable_2k_adaptive \
    --branch feat/explore_3dbbox \
    --commit 370b12e \
    --backend pack_v1 \
    --judge-model gpt-5.4-2026-03-05 \
    --notes "Projectable GT-pool pack_v1 2k unique-target smoke. Prepared 2013 projectable samples from 2015 clean candidates; ran 2000. Started at 32 workers, quarantined 179 transient APIConnectionError checkpoints, retried; quarantined and retried 7 invalid_prompt BadRequest checkpoints; completed with adaptive 20-worker finish, 1990 completed, 10 model failed, 0 error." \
    --db docs/benchmark/embodiedscan/runs.sqlite
```

## Headline Metrics

| n | mean IoU | Acc@0.25 | Acc@0.50 | completed | failed | error | misses |
|--:|---------:|---------:|---------:|----------:|-------:|------:|-------:|
| 2000 | 89.20 | 89.15 | 89.15 | 1990 | 10 | 0 | 217 |

SQLite reproduction:

```sql
SELECT run_id, n,
       printf('%.4f', mean_iou) AS mean_iou,
       printf('%.4f', acc25) AS acc25,
       printf('%.4f', acc50) AS acc50
FROM runs
WHERE run_id='v3_projectable_2k_adaptive';
```

Failed samples:

| sample_id | query |
|-----------|-------|
| `scene0129_00::17` | choose the window that is farthest from the board |
| `scene0129_00::32` | select the door that is closer to the board |
| `scene0226_00::27` | find the socket that is in front of the laptop |
| `scene0236_01::39` | select the book that is in front of the hair dryer |
| `scene0262_00::12` | find the chair that is in front of the bag |
| `scene0264_02::31` | choose the mouse that is in front of the clock |
| `scene0362_02::18` | facing the front of the cup , pick the towel that is on the right of the cup |
| `scene0449_00::36` | looking at the front of the bin , pick the switch that is on the right of the bin |
| `scene0510_02::22` | facing the front of the bar , pick the mirror that is on the right of the bar |
| `scene0656_02::32` | find the picture that is closer to the shelf |

## Prep Validation

| requested | prepared | run | skipped | scenes | excluded corrupt scenes |
|----------:|---------:|----:|--------:|-------:|------------------------:|
| 2015 | 2013 | 2000 | 2 | 220 | 1 |

Skipped targets:

- `scene0451_00::5`
- `scene0072_01::48`

Excluded corrupt scene:

- `scene0006_02`: `raw/000017-rgb.jpg` is a zero-byte image.

Visual validation artifacts:

- `tmp/embodiedscan_visual_validation_v3_projectable_2k_20260429/target_highlight_grid.png`
- `tmp/embodiedscan_visual_validation_v3_projectable_2k_20260429/raw_vs_annotated_pairs_page1.png`
- `tmp/embodiedscan_visual_validation_v3_projectable_2k_20260429/raw_vs_annotated_pairs_page2.png`
- `tmp/embodiedscan_visual_validation_v3_projectable_2k_20260429/raw_vs_annotated_pairs_page3.png`
- `tmp/embodiedscan_visual_validation_v3_projectable_2k_20260429/raw_vs_annotated_pairs_page4.png`
- `tmp/embodiedscan_visual_validation_v3_projectable_2k_20260429/manifest.json`

Manual inspection confirmed the 48-sample target-highlight grid and four
raw-vs-annotated pages had visible yellow target boxes and no null-projection
panels. Some boxes are clipped at image edges, which is expected after
projection/clamping.

## Cross-Version Comparison

| Version | Fold | n | mean IoU | Acc@0.25 | Acc@0.50 | Notes |
|---------|------|--:|---------:|---------:|---------:|-------|
| v1 | original prepared smoke | 68 | 91.18 | 91.18 | 91.18 | included stale/incorrectly prepared targets |
| v2 | projectable-only corrected smoke | 67 | 94.08 | 94.03 | 94.03 | corrected visibility/projection/filtering |
| v3 | projectable-only 2k smoke | 2000 | 89.20 | 89.15 | 89.15 | larger unique-target sweep with checkpointed adaptive concurrency |

These are not matched-fold comparisons. v3 is the first run large enough to be
useful as a broad smoke, but it still uses oracle GT proposals.

## Caveats

- This remains a GT-pool oracle evaluation. It measures top-1 selection over
  annotated candidate boxes, not detector recall or detector localization.
- The run used a dirty working tree containing prep, loader, test, runner, and
  doc changes. Preserve those source changes before treating this as
  reproducible from a clean commit.
- The run required adaptive concurrency because 32 workers produced transient
  `APIConnectionError` rows, and 24 workers produced a burst of
  `invalid_prompt` BadRequest rows. Those checkpoint files are preserved under
  quarantine directories in the raw artifact tree and were retried before the
  final aggregate was written.
- `tool_calls` and `llm_calls` are still not populated in SQLite; the current
  side-by-side JSON stores per-sample predictions and aggregate metrics only.
