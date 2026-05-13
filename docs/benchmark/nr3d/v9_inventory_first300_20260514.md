# v9 Proposal Inventory First300 - 2026-05-14

NR3D Transcrib3D first300-valid matched-fold rerun with the v9 compact proposal
inventory prompt. This run reuses the v8 prepared pack and only reruns Stage 2
so the measured change is the prompt/playbook transfer step.

## Run Identity

- Branch: `feat/nr3d-transcrib3d-first300`
- Tip commit at run time: `9eaf05d`
- Code change commit: `db95169` (`Add VG proposal inventory prompt`)
- Internal version: `v9_inventory_first300`
- Run ID: `v9_inventory_first300_20260514`
- Stage 2 backend: `gpt-5.4-2026-03-05` via internal ModelHub
- Judge: none; scoring is programmatic target-id classification
- SQLite row:
  `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v9_inventory_first300_20260514'`

## Fold And Pack

- Protocol target: Transcrib3D `nr3d_first300_valid`
- Fold size after Transcrib3D valid filtering and local sample mapping: 281
- Selection file:
  `docs/benchmark/nr3d/assets/transcrib3d_first300_valid_sample_ids_20260514.json`
- Prepared pack:
  `data/nr3d/scannet/<scene>/pack_nr3d_v8_transcrib3d_first300_baseline/`
- Canonical local filter after restriction: `n_full=281`, `n_filtered=281`
- Evidence setup:
  - query-driven Stage 1 pack prepared by v8
  - depth-aware visibility and inline `#proposal_id category` marked frames
  - no GT target used for keyframe selection
  - no selector frame-overlap NMS

## What Changed From v8

v8 already wired NR3D callbacks and used the depth-aware query-driven pack. v9
adds the Transcrib3D-style candidate prior:

- inject `Scene Proposal Inventory` into the VG user prompt
- list every submit-able proposal id with detector category, 3D center, 3D
  size, and visible-view count
- tell the agent to form category-compatible candidate sets and use center/size
  comparisons before spending more turns on images
- bump `chassis_tools_version` to `19` to avoid prompt-cache reuse across the
  prompt-surface change

No proposal pool, keyframe pack, visibility, callback wiring, or final-decision
guard changed.

## Raw Artifacts

- Eval output:
  `tmp/nr3d_eval_v9_inventory_first300_20260514/`
- Per-sample checkpoints:
  `tmp/nr3d_eval_v9_inventory_first300_20260514/per_sample/pack_nr3d_v8_transcrib3d_first300_baseline/*.json`
- Initial failed sentinel backup:
  `tmp/nr3d_eval_v9_inventory_first300_20260514/failed_invalid_prompt_attempt1/`
- Failed-rerun sample ids:
  `tmp/nr3d_artifacts/v9_inventory_first300_failed_invalid_prompt_20260514.json`
- Side-by-side output:
  `tmp/nr3d_eval_v9_inventory_first300_20260514/side_by_side.json`
- Leaderboard metrics:
  `tmp/nr3d_eval_v9_inventory_first300_20260514/leaderboard_metrics.json`
- Transcrib3D comparison:
  `tmp/nr3d_eval_v9_inventory_first300_20260514/comparison_vs_transcrib3d.json`
- Logs:
  - `tmp/nr3d_eval_v9_inventory_first300_20260514_w100.log`
  - `tmp/nr3d_eval_v9_inventory_first300_failed1_20260514.log`
  - `tmp/nr3d_eval_v9_inventory_first300_20260514_assemble.log`
  - `tmp/nr3d_eval_v9_inventory_first300_20260514_metrics.log`

## Commands

Initial 100-worker checkpoint pass:

```bash
tmux new-session -d -s nr3d_v9_inventory_first300_20260514 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids docs/benchmark/nr3d/assets/transcrib3d_first300_valid_sample_ids_20260514.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v8_transcrib3d_first300_baseline --output-dir tmp/nr3d_eval_v9_inventory_first300_20260514 --workers 100 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v9_inventory_first300_20260514_w100.log'
```

The initial pass produced 280 completed checkpoints and 1 failed sentinel:
`scannet/scene0565_00::4::14021`. The failed checkpoint was moved aside and
rerun:

```bash
tmux new-session -d -s nr3d_v9_inventory_first300_failed1_20260514 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v9_inventory_first300_failed_invalid_prompt_20260514.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v8_transcrib3d_first300_baseline --output-dir tmp/nr3d_eval_v9_inventory_first300_20260514 --workers 1 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v9_inventory_first300_failed1_20260514.log'
```

Assemble and score:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids docs/benchmark/nr3d/assets/transcrib3d_first300_valid_sample_ids_20260514.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v8_transcrib3d_first300_baseline \
  --output-dir tmp/nr3d_eval_v9_inventory_first300_20260514 \
  --workers 100 \
  --max-new-samples 0 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v9_inventory_first300_20260514/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids docs/benchmark/nr3d/assets/transcrib3d_first300_valid_sample_ids_20260514.json \
  --output tmp/nr3d_eval_v9_inventory_first300_20260514/leaderboard_metrics.json \
  --canonical-filter true
```

Ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v9_inventory_first300_20260514 \
  --run-id v9_inventory_first300_20260514 \
  --branch feat/nr3d-transcrib3d-first300 \
  --commit 9eaf05d \
  --backend pack_v1 \
  --judge-model none \
  --notes "v9 compact proposal-inventory prompt on Transcrib3D first300-valid matched fold; run-time tip=9eaf05d, code change commit=db95169; reused pack_nr3d_v8_transcrib3d_first300_baseline; workers=100 with 15GB RSS guard and 60s checks; initial 1 invalid_prompt failed sentinel moved aside and rerun with workers=1; final 281/281 completed; TADG + no-match + evidence-frame guards; no callback-not-configured; no rss_guard; 211/281 vs Transcrib3D GPT-4o 208/281" \
  --leaderboard-metrics tmp/nr3d_eval_v9_inventory_first300_20260514/leaderboard_metrics.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

## Metrics

### Local Leaderboard-Track Classification

| Metric | v8 baseline | v9 inventory | Delta |
|---|---:|---:|---:|
| Overall | 74.38 | 75.09 | +0.71 pp |
| Easy | 78.72 | 80.14 | +1.42 pp |
| Hard | 70.00 | 70.00 | +0.00 pp |
| View-Dep | 59.76 | 62.20 | +2.44 pp |
| View-Indep | 80.40 | 80.40 | +0.00 pp |
| Correct / total | 209 / 281 | 211 / 281 | +2 |

### Against Transcrib3D GPT-4o First300-Valid

Transcrib3D baseline source:
`/Users/bytedance/project/Transcrib3D/runs/modelhub_sharded/20260513-175704_nr3d_nr3d_first300_valid/metrics.json`

| Metric | Transcrib3D GPT-4o | Ours v9 | Delta |
|---|---:|---:|---:|
| Overall | 74.02 | 75.09 | +1.07 pp / +3 samples |
| Easy | 78.72 | 80.14 | +1.42 pp / +2 samples |
| Hard | 69.29 | 70.00 | +0.71 pp / +1 sample |
| View-Dep | 57.47 | 62.20 | caveat: split denominator differs |
| View-Indep | 81.44 | 80.40 | caveat: split denominator differs |

The view-dependent rows are not directly apples-to-apples because
Transcrib3D's run reports 87 view-dependent / 194 view-independent samples,
while the local canonical NR3D metric script reports 82 / 199 on the same
sample ids.

Using the same local sample rows and Transcrib3D per-row correctness:

| Slice | n | Ours v9 | Transcrib3D | Correct-count delta |
|---|---:|---:|---:|---:|
| Easy | 141 | 113 / 80.14 | 111 / 78.72 | +2 |
| Hard | 140 | 98 / 70.00 | 97 / 69.29 | +1 |
| Local View-Dep | 82 | 51 / 62.20 | 47 / 57.32 | +4 |
| Local View-Indep | 199 | 160 / 80.40 | 161 / 80.90 | -1 |
| Transcrib3D spatial flag | 242 | 181 / 74.79 | 179 / 73.97 | +2 |
| Transcrib3D color flag | 70 | 50 / 71.43 | 52 / 74.29 | -2 |
| Transcrib3D shape flag | 41 | 30 / 73.17 | 28 / 68.29 | +2 |

### Agreement With Transcrib3D

| Bucket | Count |
|---|---:|
| Ours v9 only correct | 39 |
| Transcrib3D only correct | 36 |
| Both correct | 172 |
| Both wrong | 34 |

### v9 Drift vs v8

| Bucket | Count |
|---|---:|
| v9 only correct | 23 |
| v8 only correct | 21 |
| Both correct | 188 |
| Both wrong | 49 |
| Net movement | +2 |

Representative v9-only wins vs Transcrib3D:

- line 11 `scannet/scene0077_00::20::16367`: door leading outside
- line 29 `scannet/scene0084_00::35::7464`: toilet paper roll next to toilet
- line 30 `scannet/scene0329_00::37::24849`: black chair closest to whiteboard
- line 33 `scannet/scene0329_00::34::16819`: middle chair
- line 39 `scannet/scene0030_00::51::24881`: table in top-right corner

Representative Transcrib3D-only wins vs v9:

- line 5 `scannet/scene0221_00::3::23651`: larger chair near octagonal table
- line 15 `scannet/scene0203_00::14::18184`: white pillow on the chair
- line 38 `scannet/scene0629_00::24::7584`: thin white picture next to door
- line 46 `scannet/scene0139_00::15::28907`: lower set of pipes
- line 53 `scannet/scene0653_00::44::18781`: filing cabinet under the desk

## Runtime Notes

- Initial 100-worker pass: 280 completed, 1 failed sentinel.
- Targeted failed rerun: 1/1 completed at `workers=1`.
- Final checkpoint audit: 281/281 completed, 0 failed sentinels.
- Peak observed Python RSS during monitoring was about 5.7GB, below the 15GB
  RSS guard.
- Combined log counts:
  - retryable `429`: 245
  - retryable `503`: 0
  - attempts at 3/5 or higher: 0
  - `callback is not configured`: 0
  - `rss_guard`: 0
  - `Traceback`: 0
  - `Loading CLIP model`: 9

## Interpretation

v9 is the first local run on the Transcrib3D first300-valid fold that beats the
recorded Transcrib3D GPT-4o text-only baseline by more than one sample:
211/281 vs 208/281. It also improves v8 by 2 samples.

The margin is still modest. This is a valid improvement and a useful transfer
step, but it is not yet a decisive win over Transcrib3D. The remaining
Transcrib3D-only wins still cluster around cases where deterministic geometry
or attribute ranking should help: larger/farthest superlatives, under/above
relations, left/right/ordinal relations, and color/attribute disambiguation.

The next transfer step should add a deterministic proposal-ranking helper or
prompt/tool path for those relation families instead of relying on the LLM to
derive all comparisons from the inventory table.

## Caveats

- This is a 281-sample matched-fold ablation, not a full NR3D result.
- The result uses the v8 prepared pack, so it has no selector frame-overlap NMS.
- The prompt inventory exposes geometry but does not enforce geometry use; the
  sample-level drift shows the LLM still applies the table inconsistently.
- Transcrib3D and local view-dependent splits use different denominators on
  this fold; compare overall/easy/hard directly and treat view-dep/view-indep
  as directional only.
