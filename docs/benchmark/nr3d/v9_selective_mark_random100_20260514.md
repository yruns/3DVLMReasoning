# v9 Selective Mark Random100 - 2026-05-14

NR3D random100 rerun on the same depth-aware v4/v6/v7/v8 fold after adding a
selective marked-image tool flow. Initial keyframes remain clean RGB images with
text-only proposal inventory, but the agent can now inspect a single frame's
proposal list through `list_frame_proposals(frame_id)` and then request a
filtered marked image through `view_keyframe_marked(frame_id, categories=[...],
proposal_ids=[...])`.

This is a partial internal pilot, not a public leaderboard row. It is the best
current depth-aware random100 result on this fixed fold.

## Run Identity

- Branch: `feat/nr3d-v4-agent-guards-fair-views`
- Tip commit at run time: `4a1fba1-dirty-selective-mark`
  - The run used the current uncommitted selective-mark working tree on top of
    commit `4a1fba1`.
- Internal version: `v9_selective_mark_random100`
- Run ID: `v9_selective_mark_random100_20260514`
- Stage 2 backend: `gpt-5.4-2026-03-05` via internal ModelHub
- Judge: none; scoring is programmatic target-instance classification
- SQLite row:
  `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v9_selective_mark_random100_20260514'`

## What Changed vs v8

- Added `list_frame_proposals(frame_id)` as a text-only tool that returns one
  frame's visible proposal ids, categories, `left_to_right`, and `boxes_2d`
  without injecting another image.
- Extended `view_keyframe_marked` with optional `categories` and
  `proposal_ids` filters. The filters use union semantics and dynamically
  render a marked image from the raw RGB frame containing only the selected
  proposals.
- Filtered marked images use thicker boxes and higher-contrast label text
  (`#proposal_id category`) so crowded frames are less likely to hide the
  target pixels.
- Updated `vg-grounding-playbook` to prefer:
  `list_frame_proposals(frame_id)` -> filtered `view_keyframe_marked(...)`
  for crowded frames or known candidate categories/ids.
- Bumped `chassis_tools_version` to `19` so prompt-cache/session ids do not
  reuse the older tool surface.
- Stage 2 guard flags remain enabled:
  - `--use-tool-answer-disagreement-gate`
  - `--use-no-match-candidate-guard`
  - `--use-evidence-frame-guard`

## Fold

- Split: NR3D `test`
- Fold size: 100 samples
- Selection file:
  `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json`
- Selection mechanism: same deterministic v4/v6/v7/v8 random100 fold.
- Canonical filter after restriction: `n_full=100`, `n_filtered=100`

## Raw Artifacts

- Prepared pack:
  `data/nr3d/scannet/<scene>/pack_nr3d_v8_clean_initial_marked_on_demand/`
- Eval output:
  `tmp/nr3d_eval_v9_selective_mark_random100_20260514/`
- Per-sample checkpoints:
  `tmp/nr3d_eval_v9_selective_mark_random100_20260514/per_sample/pack_nr3d_v8_clean_initial_marked_on_demand/*.json`
- Side-by-side output:
  `tmp/nr3d_eval_v9_selective_mark_random100_20260514/side_by_side.json`
- Leaderboard metrics:
  `tmp/nr3d_eval_v9_selective_mark_random100_20260514/leaderboard_metrics.json`
- Log:
  `tmp/nr3d_eval_v9_selective_mark_random100_20260514_w100.log`
- Static 2T/2F trace viewer:
  `docs/benchmark/nr3d/v9_selective_mark_trace_2t2f_20260514.html`
- Trace assets:
  `docs/benchmark/nr3d/assets/v9_selective_mark_trace_2t2f_20260514/`

## Commands

Run Stage 2 checkpoints:

```bash
tmux new-session -d -s nr3d_v9_selective_mark_random100 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v8_clean_initial_marked_on_demand --output-dir tmp/nr3d_eval_v9_selective_mark_random100_20260514 --workers 100 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v9_selective_mark_random100_20260514_w100.log'
```

Assemble without re-running samples:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v8_clean_initial_marked_on_demand \
  --output-dir tmp/nr3d_eval_v9_selective_mark_random100_20260514 \
  --workers 100 \
  --max-new-samples 0 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard
```

Compute metrics, ingest, and generate trace HTML:

```bash
PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v9_selective_mark_random100_20260514/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --output tmp/nr3d_eval_v9_selective_mark_random100_20260514/leaderboard_metrics.json \
  --canonical-filter true

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v9_selective_mark_random100_20260514 \
  --run-id v9_selective_mark_random100_20260514 \
  --branch feat/nr3d-v4-agent-guards-fair-views \
  --commit 4a1fba1-dirty-selective-mark \
  --backend pack_v1 \
  --judge-model none \
  --notes "v9 random100 on same v4/v6/v7/v8 fold; current dirty selective-mark tool surface; clean initial RGB with text inventory; list_frame_proposals text inventory plus filtered view_keyframe_marked categories/proposal_ids rendering; TADG + no-match + evidence-frame guards; workers=100 under 15GB RSS guard; 100/100 completed checkpoints; 0 failed sentinels" \
  --leaderboard-metrics tmp/nr3d_eval_v9_selective_mark_random100_20260514/leaderboard_metrics.json \
  --db docs/benchmark/nr3d/runs.sqlite

PYTHONPATH=src .venv/bin/python scripts/generate_nr3d_case_study_html.py \
  --leaderboard-metrics tmp/nr3d_eval_v9_selective_mark_random100_20260514/leaderboard_metrics.json \
  --per-sample-dir tmp/nr3d_eval_v9_selective_mark_random100_20260514/per_sample/pack_nr3d_v8_clean_initial_marked_on_demand \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v8_clean_initial_marked_on_demand \
  --output docs/benchmark/nr3d/v9_selective_mark_trace_2t2f_20260514.html \
  --assets-dir docs/benchmark/nr3d/assets/v9_selective_mark_trace_2t2f_20260514 \
  --image-mode recorded
```

## Metrics

### Leaderboard-Track Classification

| Metric | v7 callbacks no-CLIP | v8 clean initial | v9 selective mark | Delta vs v8 | Delta vs v7 |
|---|---:|---:|---:|---:|---:|
| Overall | 73.00 | 67.00 | **74.00** | +7.00 pp | +1.00 pp |
| Easy | 82.93 | 85.37 | **90.24** | +4.88 pp | +7.32 pp |
| Hard | **66.10** | 54.24 | 62.71 | +8.47 pp | -3.39 pp |
| View-Dep | **70.59** | 58.82 | 64.71 | +5.88 pp | -5.88 pp |
| View-Indep | 74.24 | 71.21 | **78.79** | +7.58 pp | +4.55 pp |

Raw metrics:

```text
n_full=100 n_filtered=100
classification_acc_full     = 0.7400
classification_acc_filtered = 0.7400
Easy   (n=41): 0.9024
Hard   (n=59): 0.6271
V-Dep  (n=34): 0.6471
V-Ind  (n=66): 0.7879
```

## Runtime / Tool Usage Notes

- Completed checkpoints: `100/100`
- Failed sentinels: `0`
- Per-sample statuses: `completed=100`
- ModelHub retryable `429` lines: `212`
- RSS guard: no `RSS exceeded` event in the log
- Tool calls across 100 samples:
  - `list_frame_proposals`: `74`
  - `view_keyframe_marked`: `310`
  - filtered marked responses: `380`
  - `request_more_views`: `47`
  - `request_crops`: `17`
- Samples with at least one `list_frame_proposals`: `38/100`
- Samples with at least one filtered marked response: `100/100`

The run finished the checkpoint phase quickly at `workers=100`; practical
throughput was still limited by ModelHub 429 backoff rather than local memory.

## 2T/2F Trace Viewer

HTML:
`docs/benchmark/nr3d/v9_selective_mark_trace_2t2f_20260514.html`

Cases:

| Case | Result | Query | Selected / GT | Why included |
|---|---|---|---|---|
| `scannet/scene0435_00::38::20355` | correct | `The lamp between the two beds.` | `38 / 38` | Uses `list_frame_proposals` 4 times and filtered marks to narrow bed/lamp/nightstand candidates. |
| `scannet/scene0025_00::30::25001` | correct | `The black keyboard.` | `30 / 30` | Demonstrates category-filtered keyboard marks after reading crowded frame inventories. |
| `scannet/scene0643_00::22::25584` | failed | `Facing the side with 2 books, the book on the right.` | `34 / 22` | Uses text inventories and filtered marks but still fails a right-side relation. |
| `scannet/scene0651_00::8::37347` | failed | `This chair has it's back facing the sink.` | `7 / 8` | Multiple chair/sink filtered marks; failure is still view-orientation reasoning, not unreadable marks. |

## SQLite Reproduction Query

```sql
SELECT run_id, n, n_filtered,
       printf('%.4f', classification_acc_filtered) AS overall,
       printf('%.4f', acc_easy) AS easy,
       printf('%.4f', acc_hard) AS hard,
       printf('%.4f', acc_view_dep) AS view_dep,
       printf('%.4f', acc_view_indep) AS view_indep
FROM runs
WHERE run_id = 'v9_selective_mark_random100_20260514';
```

Verified output:

```text
v9_selective_mark_random100_20260514|100|100|0.7400|0.9024|0.6271|0.6471|0.7879
```

## Caveats

- This is a 100-sample internal pilot, not a public full NR3D leaderboard row.
- The run used a dirty working tree (`4a1fba1-dirty-selective-mark`) because
  the selective-mark implementation had not yet been committed at run time.
- v9 improves the fixed random100 fold overall and recovers the v8
  clean-initial regression, but it still trails v7 on Hard and View-Dep. The
  remaining failures are dominated by relational/viewpoint reasoning rather
  than unreadable bbox overlays.
