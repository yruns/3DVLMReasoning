# NR3D v11 proposal enrichment strat600

This run evaluates the NR3D 600-case canonical stratified fold after adding
offline object enrichment to the proposal context:

- compact enriched notes are shown in the initial `Proposal notes:` block;
- `inspect_proposal(id)` returns the full enriched object description when
  available;
- `pack_nr3d_v9_catalog_first` was updated in place for the 119 strat600
  scenes, with 4 074 enriched proposals in both `proposals.jsonl` and
  `scene_catalog.json`.

The result is **71.33 % Overall** on strat600: below the best observed fair
v10 row at 72.00 %, but above the two recent reproductions of that family
(70.00 % and 69.17 %).

## Pre-run Checklist

| Item | Value |
|---|---|
| Pending tracked changes committed before launch | yes |
| Branch | `best/v10-multi-anchor-tadg-72-a3ff7f1` |
| Head commit at launch | `793cb57` |
| Run-time code commit | `793cb57` — no worktree drift |
| Commit subject | `Add NR3D object enrichment to proposal context` |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` |
| Fold design | `docs/benchmark/nr3d/v9_3_strat600_subset_design_20260517.md` |
| Pack name | `pack_nr3d_v9_catalog_first` |
| Output dir | `tmp/nr3d_eval_v11_enriched_strat600_20260521_793cb57/` |
| Run log | `tmp/nr3d_eval_v11_enriched_strat600_20260521_793cb57/run.log` |
| Launch info | `tmp/nr3d_eval_v11_enriched_strat600_20260521_793cb57/launch_info.log` |
| Side-by-side JSON | `tmp/nr3d_eval_v11_enriched_strat600_20260521_793cb57/side_by_side.json` |
| Leaderboard metrics | `tmp/nr3d_eval_v11_enriched_strat600_20260521_793cb57/leaderboard_metrics.json` |
| SQLite run id | `v11_enriched_strat600_20260521` |
| Started | `2026-05-21T17:30:58+08:00` |
| Observed complete | `2026-05-21T18:20:06+08:00` |
| Final status | 600 checkpoints, `side_by_side.json` present |

## Exact Commands

Initial launch used `workers=20`:

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --output-dir tmp/nr3d_eval_v11_enriched_strat600_20260521_793cb57 \
  --workers 20 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard
```

At 232 checkpoints, the run was interrupted to increase throughput. A
`workers=40` relaunch immediately hit ModelHub 429 bursts, so it was stopped
and resumed from the same checkpoint directory with `workers=30`:

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --output-dir tmp/nr3d_eval_v11_enriched_strat600_20260521_793cb57 \
  --workers 30 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard
```

Leaderboard aggregation:

```bash
PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v11_enriched_strat600_20260521_793cb57/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --output tmp/nr3d_eval_v11_enriched_strat600_20260521_793cb57/leaderboard_metrics.json
```

SQLite ingestion:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v11_enriched_strat600_20260521_793cb57 \
  --run-id v11_enriched_strat600_20260521 \
  --branch best/v10-multi-anchor-tadg-72-a3ff7f1 \
  --commit 793cb57 \
  --db docs/benchmark/nr3d/runs.sqlite \
  --leaderboard-metrics tmp/nr3d_eval_v11_enriched_strat600_20260521_793cb57/leaderboard_metrics.json \
  --notes "NR3D strat600 after proposal enrichment in initial Proposal notes and inspect_proposal full enrichment; workers 20 then resumed at 30 after workers=40 hit 429."
```

## Headline Metrics

| Metric | v11 enriched | v10 best observed (`a3ff7f1`) | v10 w30 repro | v10 w20 repro | v9.3 text-first |
|---|---:|---:|---:|---:|---:|
| Overall | **71.33** | 72.00 | 70.00 | 69.17 | 66.67 |
| Easy | 79.66 | 82.41 | 78.62 | 79.31 | 73.45 |
| Hard | **63.55** | 62.26 | 61.94 | 59.68 | 60.32 |
| V-Dep | 61.61 | 62.56 | 61.61 | 62.56 | 54.98 |
| V-Indep | 76.61 | 77.12 | 74.55 | 72.75 | 73.01 |

Counts:

| Slice | n | correct |
|---|---:|---:|
| Full | 600 | 428 |
| Easy | 290 | 231 |
| Hard | 310 | 197 |
| V-Dep | 211 | 130 |
| V-Indep | 389 | 298 |

Pack-v1 side-by-side metrics:

| Metric | Value |
|---|---:|
| `n` | 600 |
| `status=completed` | 600 |
| missing `selected_object_id` | 0 |
| mean IoU | 0.7163 |
| Acc@0.25 | 0.7133 |
| Acc@0.50 | 0.7133 |

## Runtime Notes

- `workers=40` was too aggressive for the active ModelHub quota and produced a
  burst of retryable 429s before useful checkpoint progress. The final useful
  resume used `workers=30`.
- The merged log contains 272 retryable 429 lines, 44 retryable/rotated 403
  lines, and one `All 4 keys exhausted` parser event near the worker-change
  window. The final assembled output still has 600 / 600 completed samples and
  no missing selected object ids.
- `select_by_text` logs confirm the selector was loading per-scene enrichment,
  e.g. `Loaded enrichment for ... objects from enriched_objects.json`.

## SQLite Reproduction Query

```sql
SELECT
  run_id,
  n,
  ROUND(classification_acc_full * 100, 2) AS overall,
  ROUND(acc_easy * 100, 2) AS easy,
  ROUND(acc_hard * 100, 2) AS hard,
  ROUND(acc_view_dep * 100, 2) AS view_dep,
  ROUND(acc_view_indep * 100, 2) AS view_indep
FROM runs
WHERE run_id = 'v11_enriched_strat600_20260521';
-- v11_enriched_strat600_20260521 | 600 | 71.33 | 79.66 | 63.55 | 61.61 | 76.61
```

## Interpretation

This is a positive stabilization result rather than a new best. Enrichment is
worth keeping: it restores most of the gap between the lower 2026-05-20 /
2026-05-21 reproductions and the original 72.00 best observed row, and it
slightly improves Hard over that row. The remaining gap is within the strat600
comparison band and mixed across tiers, so the next useful check is a matched
failure/success diff against `a3ff7f1` and the two recent reproductions rather
than claiming a deterministic regression or win.
