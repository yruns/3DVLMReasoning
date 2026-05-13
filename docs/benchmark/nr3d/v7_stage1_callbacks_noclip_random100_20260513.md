# v7 Stage1 Callbacks No-CLIP Random100 - 2026-05-13

NR3D random100 rerun on the same depth-aware v6 pack after wiring the Stage1
callback tools into the NR3D runner. This is a partial pilot, not a public
leaderboard row.

The run intentionally does **not** include selector frame-overlap NMS. It
isolates the effect of making the already-advertised callback tools actually
available at runtime for NR3D.

## Run Identity

- Branch: `feat/nr3d-v4-agent-guards-fair-views`
- Tip commit at run time: `690cbf6`
- Relevant commits:
  - `2c7bda8` wires NR3D Stage1 callbacks into `Stage2DeepResearchAgent`.
  - `690cbf6` disables per-selector CLIP object-term fallback for NR3D
    `request_more_views`, preventing high-concurrency CLIP memory spikes.
- Working tree: `/Users/bytedance/project/3DVLMReasoning` (Mac, `.venv`,
  Python 3.12)
- Internal version: `v7_stage1_callbacks_noclip_random100`
- Run ID: `v7_stage1_callbacks_noclip_random100_20260513`
- Stage 2 backend: `gpt-5.4-2026-03-05` via internal ModelHub
- Judge: none; scoring is programmatic
- SQLite row:
  `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v7_stage1_callbacks_noclip_random100_20260513'`

## What Changed vs v6

- Reused the same random100 fold and the same prepared pack:
  `pack_nr3d_v6_inline_labels_depth_visible`.
- NR3D now passes real callbacks into the agent:
  - `request_more_views`
  - `request_crops`
  - `switch_or_expand_hypothesis`
- `switch_or_expand_hypothesis` keeps `use_visual_context=False`, so Stage1
  re-query is text-only and does not inject target visual context.
- `request_more_views` disables `selector.find_objects()` CLIP fallback in
  NR3D high-concurrency runs. It still uses existing hypothesis categories,
  object categories, visibility, pinned frame IDs, and exploration mode.
- Stage 2 guard flags remain enabled:
  - `--use-tool-answer-disagreement-gate`
  - `--use-no-match-candidate-guard`
  - `--use-evidence-frame-guard`

## Fold

- Split: NR3D `test`
- Fold size: 100 samples
- Selection file:
  `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json`
- Selection mechanism: same deterministic v4/v6 random100 fold.
- Canonical filter after restriction: `n_full=100`, `n_filtered=100`

## Raw Artifacts

- Prepared pack:
  `data/nr3d/scannet/<scene>/pack_nr3d_v6_inline_labels_depth_visible/`
- Eval output:
  `tmp/nr3d_eval_v7_callbacks_noclip_random100_20260513/`
- Per-sample checkpoints:
  `tmp/nr3d_eval_v7_callbacks_noclip_random100_20260513/per_sample/pack_nr3d_v6_inline_labels_depth_visible/*.json`
- Side-by-side output:
  `tmp/nr3d_eval_v7_callbacks_noclip_random100_20260513/side_by_side.json`
- Leaderboard metrics:
  `tmp/nr3d_eval_v7_callbacks_noclip_random100_20260513/leaderboard_metrics.json`
- Same-fold comparison vs v6:
  `tmp/nr3d_eval_v7_callbacks_noclip_random100_20260513/comparison_vs_v6.json`
- Logs:
  - `tmp/nr3d_eval_v7_callbacks_noclip_random100_20260513_w100.log`
  - `tmp/nr3d_eval_v7_callbacks_noclip_random100_20260513_assemble.log`
  - `tmp/nr3d_eval_v7_callbacks_noclip_random100_20260513_metrics.log`
- Aborted pre-fix attempt, not used for metrics:
  `tmp/nr3d_eval_v7_stage1_callbacks_random100_20260513/`

## Commands

Run Stage 2 checkpoints:

```bash
tmux new-session -d -s nr3d_v7_callbacks_noclip_random100 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v6_inline_labels_depth_visible --output-dir tmp/nr3d_eval_v7_callbacks_noclip_random100_20260513 --workers 100 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v7_callbacks_noclip_random100_20260513_w100.log'
```

Assemble without re-running samples:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v6_inline_labels_depth_visible \
  --output-dir tmp/nr3d_eval_v7_callbacks_noclip_random100_20260513 \
  --workers 100 \
  --max-new-samples 0 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard
```

Compute metrics and ingest:

```bash
PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v7_callbacks_noclip_random100_20260513/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --output tmp/nr3d_eval_v7_callbacks_noclip_random100_20260513/leaderboard_metrics.json \
  --canonical-filter true

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v7_callbacks_noclip_random100_20260513 \
  --run-id v7_stage1_callbacks_noclip_random100_20260513 \
  --branch feat/nr3d-v4-agent-guards-fair-views \
  --commit 690cbf6 \
  --backend pack_v1 \
  --judge-model none \
  --notes "v7 random100 rerun on same v4/v6 fold and pack_nr3d_v6_inline_labels_depth_visible; NR3D runner wires Stage1 callbacks; request_more_views disables per-selector CLIP object-term fallback to avoid high-concurrency memory spike; TADG + no-match + evidence-frame guards; workers=100 with 15GB RSS guard; 100/100 checkpoints; 0 failed sentinels" \
  --leaderboard-metrics tmp/nr3d_eval_v7_callbacks_noclip_random100_20260513/leaderboard_metrics.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

## Metrics

### Leaderboard-Track Classification

| Metric | v6 no-NMS rerender | v7 callbacks no-CLIP | Delta |
|---|---:|---:|---:|
| Overall | 71.00 | 73.00 | +2.00 pp |
| Easy | 78.05 | 82.93 | +4.88 pp |
| Hard | 66.10 | 66.10 | +0.00 pp |
| View-Dep | 70.59 | 70.59 | +0.00 pp |
| View-Indep | 71.21 | 74.24 | +3.03 pp |

### IoU Proxy on GT Pool

| Metric | v6 | v7 | Delta |
|---|---:|---:|---:|
| n | 100 | 100 | 0 |
| mean IoU | 0.7131 | 0.7327 | +0.0196 |
| Acc@0.25 | 0.7100 | 0.7300 | +0.0200 |
| Acc@0.50 | 0.7100 | 0.7300 | +0.0200 |
| failed sentinels | 0 | 0 | 0 |
| uncaught sample errors | 0 | 0 | 0 |

### Per-Sample Movement vs v6

| Check | Count |
|---|---:|
| Samples with changed selected proposal id | 22 |
| Wrong -> correct | 10 |
| Correct -> wrong | 8 |
| Net overall movement | +2 |

Wrong -> correct sample IDs:

- `scannet/scene0025_00::32::23687`
- `scannet/scene0187_00::13::24713`
- `scannet/scene0629_00::31::23523`
- `scannet/scene0648_00::15::24119`
- `scannet/scene0686_00::24::10791`
- `scannet/scene0568_00::15::20548`
- `scannet/scene0643_00::22::25584`
- `scannet/scene0249_00::36::39076`
- `scannet/scene0329_00::36::39361`
- `scannet/scene0653_00::16::3852`

Correct -> wrong sample IDs:

- `scannet/scene0645_00::38::39739`
- `scannet/scene0629_00::6::19188`
- `scannet/scene0328_00::28::19096`
- `scannet/scene0500_00::27::34282`
- `scannet/scene0568_00::19::17227`
- `scannet/scene0490_00::14::39189`
- `scannet/scene0663_00::3::36534`
- `scannet/scene0618_00::18::36850`

## Tool Trace / Runtime Notes

- Stage 2 completed 100/100 checkpoints with `workers=100`.
- RSS guard limit was 15GB; observed RSS after disabling the CLIP fallback was
  about 1.4GB early and about 0.6GB near 97/100.
- The log contains 455 retryable `429` lines, 6 retryable `503` lines, and 17
  retry attempts at attempt 3 or above. All samples completed.
- Per-sample tool traces include:
  - `request_more_views`: 43 calls
  - `request_crops`: 17 calls
  - `switch_or_expand_hypothesis`: 3 calls
  - `view_keyframe_marked`: 232 calls
- The log contains no `callback is not configured` messages.
- The log contains 3 `Loading CLIP model` messages from the text-only Stage1
  re-query path, not from `request_more_views` object-term fallback.

## Aborted Pre-Fix Attempt

The first v7 attempt used commit `2c7bda8` with callbacks wired but without
the no-CLIP guard. It aborted at 14/100 checkpoints:

- Output: `tmp/nr3d_eval_v7_stage1_callbacks_random100_20260513/`
- Log: `tmp/nr3d_eval_v7_stage1_callbacks_random100_20260513_w100.log`
- Symptom: process exited without Python traceback after multiple
  `Loading CLIP model` events and RSS around 10GB.
- Root cause: each scene selector owns a lazy CLIP model; 100-worker callbacks
  could trigger many selector-local CLIP loads concurrently.
- Fix: commit `690cbf6` adds `use_clip_object_terms=False` for NR3D
  `request_more_views` while preserving category/hypothesis/visibility based
  callback retrieval.

## Caveats

- This is a 100-sample internal ablation, not a full NR3D row.
- It still preserves v4 keyframe frame IDs via the v6 rerender pack. It does
  not test selector NMS or a full Stage1 re-selection pass.
- The callback-enabled agent has more evidence-seeking degrees of freedom than
  v6, so the +2pp result should be treated as a Stage2 runtime ablation on the
  fixed fold, not a final benchmark claim.
