# v11 Text-First Policy First300 - 2026-05-14

NR3D Transcrib3D first300-valid rerun after changing the VG prompt/playbook to
enforce structured candidate filtering before visual inspection.

## Run Identity

- Branch: `feat/nr3d-transcrib3d-first300`
- Tip commit at run time: `4ee6167`
- Internal version: `v11_textfirst_first300`
- Run ID: `v11_textfirst_first300_20260514`
- Stage 2 backend: `gpt-5.4-2026-03-05` via internal ModelHub
- Judge: none; scoring is programmatic target-id classification
- SQLite row:
  `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v11_textfirst_first300_20260514'`

## Fold And Pack

- Split: Transcrib3D `nr3d_first300_valid`
- Fold size: 281 valid samples from Transcrib3D's first 300 NR3D rows
- Selection file:
  `docs/benchmark/nr3d/assets/transcrib3d_first300_valid_sample_ids_20260514.json`
- Prepared pack:
  `data/nr3d/scannet/<scene>/pack_nr3d_v8_transcrib3d_first300_baseline/`
- Canonical filter after restriction: `n_full=281`, `n_filtered=281`
- Matched baseline:
  `/Users/bytedance/project/Transcrib3D/runs/modelhub_sharded/20260513-175704_nr3d_nr3d_first300_valid/`

## What Changed From v10

Same v11 policy as the random100 run:

- VG system prompt requires a "Structured-first VG pass".
- `vg-grounding-playbook` prioritizes Scene Proposal Inventory,
  `find_proposals_by_category`, `inspect_proposal`,
  `rank_proposals_by_geometry`, and `compare_proposals_spatial` before
  additional marked frames/crops.
- The playbook distinguishes clean NR3D-style object-pool VG from noisy
  ScanRefer-style detector-pool VG.
- `chassis_tools_version` is `21`.

No fold, pack, visibility index, callback wiring, rank-tool implementation, or
final-decision guard changed.

## Raw Artifacts

- Eval output:
  `tmp/nr3d_eval_v11_textfirst_first300_20260514/`
- Per-sample checkpoints:
  `tmp/nr3d_eval_v11_textfirst_first300_20260514/per_sample/pack_nr3d_v8_transcrib3d_first300_baseline/*.json`
- Remaining-after-RSS sample ids:
  `tmp/nr3d_artifacts/v11_textfirst_first300_remaining59_after_rss_guard_20260514.json`
- Side-by-side output:
  `tmp/nr3d_eval_v11_textfirst_first300_20260514/side_by_side.json`
- Leaderboard metrics:
  `tmp/nr3d_eval_v11_textfirst_first300_20260514/leaderboard_metrics.json`
- Logs:
  - `tmp/nr3d_eval_v11_textfirst_first300_20260514_w50.log`
  - `tmp/nr3d_eval_v11_textfirst_first300_remaining59_w25_20260514.log`
  - `tmp/nr3d_eval_v11_textfirst_first300_20260514_assemble.log`
  - `tmp/nr3d_eval_v11_textfirst_first300_20260514_metrics.log`

## Commands

Initial 50-worker checkpoint pass:

```bash
tmux new-session -d -s nr3d_v11_textfirst_first300_20260514 \
  'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids docs/benchmark/nr3d/assets/transcrib3d_first300_valid_sample_ids_20260514.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v8_transcrib3d_first300_baseline --output-dir tmp/nr3d_eval_v11_textfirst_first300_20260514 --workers 50 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v11_textfirst_first300_20260514_w50.log'
```

The initial pass hit the 15GB RSS guard after 222 completed checkpoints:

```text
[rss_guard] RSS 16163MB exceeded limit 15000MB; terminating pid 87129
```

The completed checkpoints were preserved. The remaining 59 sample ids were
written to:

```text
tmp/nr3d_artifacts/v11_textfirst_first300_remaining59_after_rss_guard_20260514.json
```

Remaining-sample rerun:

```bash
tmux new-session -d -s nr3d_v11_textfirst_first300_remaining59_20260514 \
  'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v11_textfirst_first300_remaining59_after_rss_guard_20260514.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v8_transcrib3d_first300_baseline --output-dir tmp/nr3d_eval_v11_textfirst_first300_20260514 --workers 25 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v11_textfirst_first300_remaining59_w25_20260514.log'
```

Assemble and score:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids docs/benchmark/nr3d/assets/transcrib3d_first300_valid_sample_ids_20260514.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v8_transcrib3d_first300_baseline \
  --output-dir tmp/nr3d_eval_v11_textfirst_first300_20260514 \
  --workers 50 \
  --max-new-samples 0 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v11_textfirst_first300_20260514/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids docs/benchmark/nr3d/assets/transcrib3d_first300_valid_sample_ids_20260514.json \
  --output tmp/nr3d_eval_v11_textfirst_first300_20260514/leaderboard_metrics.json \
  --canonical-filter true
```

Ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v11_textfirst_first300_20260514 \
  --run-id v11_textfirst_first300_20260514 \
  --branch feat/nr3d-transcrib3d-first300 \
  --commit 4ee6167 \
  --backend pack_v1 \
  --judge-model none \
  --notes "v11 VG text-first policy on Transcrib3D first300-valid matched fold; same pack_nr3d_v8_transcrib3d_first300_baseline as v8-v10; initial workers=50 hit 15GB RSS guard at 222 completed due concurrent request_crops CLIP fallback; remaining 59 rerun at workers=25; final 281/281 completed; no failed sentinels; 214/281 overall." \
  --leaderboard-metrics tmp/nr3d_eval_v11_textfirst_first300_20260514/leaderboard_metrics.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

SQLite reproduction:

```sql
SELECT run_id, n,
       printf('%.4f', classification_acc_full) AS acc,
       printf('%.4f', acc_easy) AS easy,
       printf('%.4f', acc_hard) AS hard,
       printf('%.4f', acc_view_dep) AS view_dep,
       printf('%.4f', acc_view_indep) AS view_indep
FROM runs
WHERE run_id='v11_textfirst_first300_20260514';
```

Expected:

```text
v11_textfirst_first300_20260514|281|0.7616|0.8014|0.7214|0.6463|0.8090
```

## Metrics

### Leaderboard-Track Classification

| Metric | Transcrib3D GPT-4o | v8 baseline | v9 inventory | v10 geometry | v11 text-first |
|---|---:|---:|---:|---:|---:|
| Overall | 74.02 | 74.38 | 75.09 | 76.87 | 76.16 |
| Easy | 78.72 | 78.72 | 80.14 | 82.27 | 80.14 |
| Hard | 69.29 | 70.00 | 70.00 | 71.43 | 72.14 |
| View-Dep | 57.47* | 59.76 | 62.20 | 65.85 | 64.63 |
| View-Indep | 81.44* | 80.40 | 80.40 | 81.41 | 80.90 |

`*` Transcrib3D uses its own per-row view-dependence tags. The local columns
use the local `nr3d_leaderboard_metrics.py` split on the same sample ids.

### Matched Counts

| Comparison | Correct | Delta vs v11 |
|---|---:|---:|
| v11 text-first | 214 / 281 | 0 |
| v10 geometry | 216 / 281 | -2 for v11 |
| v9 inventory | 211 / 281 | +3 for v11 |
| v8 baseline | 209 / 281 | +5 for v11 |
| Transcrib3D GPT-4o | 208 / 281 | +6 for v11 |

Against v10:

- v11-only correct: 20
- v10-only correct: 22
- net: -2 samples

Against v9:

- v11-only correct: 18
- v9-only correct: 15
- net: +3 samples

## Tool Policy Shift

| Tool | v9 calls | v10 calls | v11 calls |
|---|---:|---:|---:|
| `find_proposals_by_category` | 338 | 347 | 542 |
| `inspect_proposal` | 870 | 882 | 1016 |
| `view_keyframe_marked` | 437 | 423 | 495 |
| `compare_proposals_spatial` | 198 | 202 | 210 |
| `rank_proposals_by_geometry` | 0 | 40 | 39 |
| `request_crops` | 40 | 41 | 64 |
| `request_more_views` | 74 | 68 | 55 |

v11 clearly pushed the agent into more structured candidate lookup and proposal
inspection. It also increased crop usage, which exposed the remaining CLIP
fallback memory boundary at high concurrency.

## Runtime Notes

- Initial `workers=50` pass:
  - 222 completed checkpoints
  - 0 failed sentinels
  - `rss_guard` fired once at 16163MB > 15000MB
- Remaining `workers=25` pass:
  - recovered the remaining 59 samples
  - no RSS guard
- Final checkpoint audit: 281/281 completed, 0 failed sentinels.
- Combined log counts:
  - retryable `429`: 152
  - retryable `503`: 0
  - attempts at 2/5 or higher: 0
  - `callback is not configured`: 0
  - `Traceback`: 0
  - `Loading CLIP model`: 19

The RSS guard was triggered by concurrent crop callbacks entering
`selector.find_objects()` and lazily loading per-process CLIP models. One bad
pattern in the log is `request_crops` with terms like `proposal 15` /
`proposal 16`; because these do not string-match object labels, the callback
falls through to CLIP semantic matching. That path is both semantically weak
for numeric proposal references and memory-expensive.

## Interpretation

v11 remains a matched-fold improvement over the Transcrib3D GPT-4o baseline:
214/281 vs 208/281. However, it is not better than v10 geometry on this fold
and therefore is not the new first300 headline. The main positive signal is
that it improves over v9 while preserving most of v10's gain.

The memory signal is actionable: the structured-first policy increases crop
requests, so the crop callback should resolve explicit proposal ids directly
and should avoid CLIP fallback unless explicitly enabled. That fix should be a
separate version because it changes runtime behavior.

## Caveats

- The fold is Transcrib3D's first300-valid subset, not the full NR3D test set.
- Transcrib3D's public full-valid result is 77.61 on 6910 valid rows and is not
  directly comparable to this 281-sample matched fold.
- No selector frame-overlap NMS was applied.
- The initial `workers=50` pass exceeded the 15GB RSS guard; final metrics use
  completed checkpoints plus a lower-concurrency rerun of every missing sample,
  not a partial fold.
