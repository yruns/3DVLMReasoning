# v6 Inline Labels + Depth-Visible Marks Random100 - 2026-05-13

NR3D random100 rerun using the current VG agent runtime after the depth-aware
visibility rebuild and inline marked-box label update. This is a partial pilot,
not a public leaderboard row.

This run intentionally does **not** include selector frame-overlap NMS. The
NMS idea is deferred to a separate follow-up version so its impact can be
measured independently.

## Run Identity

- Branch: `feat/nr3d-v4-agent-guards-fair-views`
- Tip commit at run time: `41253ad`
- Working tree: `/Users/bytedance/project/3DVLMReasoning` (Mac, `.venv`, Python 3.12)
- Internal version: `v6_inline_labels_depth_visible_random100`
- Run ID: `v6_inline_labels_depth_visible_random100_20260513`
- Stage 2 backend: `gpt-5.4-2026-03-05` via internal ModelHub
- Judge: none; scoring is programmatic
- SQLite row: `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v6_inline_labels_depth_visible_random100_20260513'`

## What Changed vs v4

- Re-rendered the same v4 random100 samples into
  `pack_nr3d_v6_inline_labels_depth_visible`.
- Preserved the v4 per-sample keyframe `frame_id`s instead of rerunning Stage 1.
  This isolates the visual evidence rendering / prompt-label change from future
  selector NMS changes.
- Marked frames now use depth-visible point projection and render only proposals
  visible in the current depth-aware frame visibility index.
- Box labels are inline high-contrast `#proposal_id category` labels; there is
  no bottom legend.
- Stage 2 prompt/playbook explicitly tells the agent that marked RGB images
  contain colored 2D boxes labeled by proposal id and category.
- Stage 2 guard flags remain enabled:
  - `--use-tool-answer-disagreement-gate`
  - `--use-no-match-candidate-guard`
  - `--use-evidence-frame-guard`

## Deferred Task Marker

Selector frame-overlap NMS is **not included** in this run. The intended next
ablation is:

- compute frame overlap from camera pose + frustum geometry;
- suppress near-duplicate frames above a fixed overlap threshold;
- visualize before/after keyframes plus the overlap matrix;
- rerun the same random100 fold as a separate version so changes can be
  attributed to NMS only.

## Fold

- Split: NR3D `test`
- Fold size: 100 samples
- Selection file:
  `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json`
- Selection mechanism:
  same deterministic v4 random100 fold.
- Canonical filter after restriction: `n_full=100`, `n_filtered=100`

## Raw Artifacts

- Prepared pack:
  `data/nr3d/scannet/<scene>/pack_nr3d_v6_inline_labels_depth_visible/`
- Eval output:
  `tmp/nr3d_eval_v6_inline_labels_depth_visible_random100_20260513/`
- Per-sample checkpoints:
  `tmp/nr3d_eval_v6_inline_labels_depth_visible_random100_20260513/per_sample/pack_nr3d_v6_inline_labels_depth_visible/*.json`
- Side-by-side output:
  `tmp/nr3d_eval_v6_inline_labels_depth_visible_random100_20260513/side_by_side.json`
- Leaderboard metrics:
  `tmp/nr3d_eval_v6_inline_labels_depth_visible_random100_20260513/leaderboard_metrics.json`
- Same-fold comparison vs v4:
  `tmp/nr3d_eval_v6_inline_labels_depth_visible_random100_20260513/comparison_vs_v4.json`
- Logs:
  - `tmp/nr3d_v6_inline_prep_random100_20260513.log`
  - `tmp/nr3d_eval_v6_inline_labels_depth_visible_random100_20260513_w100.log`
  - `tmp/nr3d_eval_v6_inline_labels_depth_visible_random100_20260513_assemble.log`
  - `tmp/nr3d_eval_v6_inline_labels_depth_visible_random100_20260513_metrics.log`

## Commands

Re-render the pack while preserving v4 keyframe choices:

```bash
tmux new-session -d -s nr3d_v6_inline_prep_random100 'cd /Users/bytedance/project/3DVLMReasoning && PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python scripts/prepare_nr3d_rerender_pack_from_existing.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --source-pack pack_nr3d_v4_agent_guards_fair_views \
  --out-pack pack_nr3d_v6_inline_labels_depth_visible \
  2>&1 | tee tmp/nr3d_v6_inline_prep_random100_20260513.log'
```

Run Stage 2 checkpoints:

```bash
tmux new-session -d -s nr3d_v6_inline_stage2_random100 'cd /Users/bytedance/project/3DVLMReasoning && PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v6_inline_labels_depth_visible \
  --output-dir tmp/nr3d_eval_v6_inline_labels_depth_visible_random100_20260513 \
  --workers 100 \
  --checkpoint-only \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard \
  2>&1 | tee tmp/nr3d_eval_v6_inline_labels_depth_visible_random100_20260513_w100.log'
```

Assemble without re-running samples:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v6_inline_labels_depth_visible \
  --output-dir tmp/nr3d_eval_v6_inline_labels_depth_visible_random100_20260513 \
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
  --side-by-side tmp/nr3d_eval_v6_inline_labels_depth_visible_random100_20260513/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --output tmp/nr3d_eval_v6_inline_labels_depth_visible_random100_20260513/leaderboard_metrics.json \
  --canonical-filter true

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v6_inline_labels_depth_visible_random100_20260513 \
  --run-id v6_inline_labels_depth_visible_random100_20260513 \
  --branch feat/nr3d-v4-agent-guards-fair-views \
  --commit 41253ad \
  --backend pack_v1 \
  --judge-model none \
  --notes "v6 random100 no-NMS rerun; preserved v4 random100 sample ids and keyframe frame_ids via rerender pack; depth-aware visibility/rerendered inline #proposal_id category marked boxes; TADG + no-match + evidence-frame guards; workers=100; 100/100 checkpoints; 0 failed sentinels" \
  --leaderboard-metrics tmp/nr3d_eval_v6_inline_labels_depth_visible_random100_20260513/leaderboard_metrics.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

## Metrics

### Leaderboard-Track Classification

| Metric | v4 invalidated random100 | v6 no-NMS rerender | Delta |
|---|---:|---:|---:|
| Overall | 71.00 | 71.00 | +0.00 pp |
| Easy | 82.93 | 78.05 | -4.88 pp |
| Hard | 62.71 | 66.10 | +3.39 pp |
| View-Dep | 67.65 | 70.59 | +2.94 pp |
| View-Indep | 72.73 | 71.21 | -1.52 pp |

### IoU Proxy on GT Pool

| Metric | Value |
|---|---:|
| n | 100 |
| mean IoU | 0.7131 |
| Acc@0.25 | 0.7100 |
| Acc@0.50 | 0.7100 |
| failed sentinels | 0 |
| uncaught sample errors | 0 |

### Per-Sample Movement vs v4

| Check | Count |
|---|---:|
| Samples with changed selected proposal id | 20 |
| Wrong -> correct | 7 |
| Correct -> wrong | 7 |
| Net overall movement | 0 |

The unchanged overall score hides offsetting sample-level changes. The rerender
and inline labels helped several cases but also regressed several others on the
same fold.

## Runtime Notes

- Prep completed 58 scenes and 100 samples.
- Prep RSS stayed around 2.6GB in spot checks.
- Stage 2 completed 100/100 checkpoints with `workers=100`.
- The log contains 357 retryable `429` lines and 86 retry attempts at attempt
  2 or above, but all samples completed and no failed sentinels were written.

## Caveats

- This is a 100-sample internal ablation, not a full NR3D row.
- It preserves v4 keyframe frame ids rather than rerunning the selector after
  the depth-aware visibility rebuild. This is intentional for isolating the
  marked-frame/prompt change before the planned NMS ablation.
- The v4 comparison row is invalidated by projection-only visibility and is
  used here only as an internal same-fold reference.
