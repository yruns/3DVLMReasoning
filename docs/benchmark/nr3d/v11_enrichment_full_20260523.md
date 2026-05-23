# NR3D v11 proposal enrichment FULL

This run evaluates the full NR3D test fold after the v11 proposal-enrichment
changes:

- compact enriched notes are included in the initial `Proposal notes:` block;
- `inspect_proposal(id)` returns the full enriched object description;
- `data/nr3d/scannet/*/conceptgraph/enriched_objects.json` exists for all
  141 prepared NR3D scenes, and all 130 full-test scenes had complete pack
  enrichment before launch.

The final filtered benchmark result is **72.26 % Overall** on the canonical
7805-query NR3D classification slice.

## Pre-run Checklist

| Item | Value |
|---|---|
| Pending tracked changes committed before launch | yes |
| Branch | `best/v10-multi-anchor-tadg-72-a3ff7f1` |
| Head commit at launch | `63dc417` |
| Run-time code commit | `63dc417` - no worktree drift |
| Commit subject | `Record NR3D v11 enrichment strat600 run` |
| Fold | `tmp/nr3d_artifacts/full_test_sample_ids.json` |
| Fold size | 8584 prepared utterances; canonical filtered leaderboard n=7805 |
| Pack name | `pack_nr3d_v9_catalog_first` |
| Output dir | `tmp/nr3d_eval_v11_enriched_full_20260522_63dc417/` |
| Run log | `tmp/nr3d_eval_v11_enriched_full_20260522_63dc417/run.log` |
| Launch info | `tmp/nr3d_eval_v11_enriched_full_20260522_63dc417/launch_info.log` |
| Side-by-side JSON | `tmp/nr3d_eval_v11_enriched_full_20260522_63dc417/side_by_side.json` |
| Leaderboard metrics | `tmp/nr3d_eval_v11_enriched_full_20260522_63dc417/leaderboard_metrics.json` |
| SQLite run id | `v11_enriched_full_20260522_63dc417` |
| Started | `2026-05-22T00:08:26+08:00` |
| Final metrics regenerated | `2026-05-23T01:35:02+08:00` |
| Final status | 8584 checkpoints, 0 checkpoint `error` fields |
| Judge model | none - deterministic NR3D classification metric |

## Exact Commands

Initial checkpoint pass:

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 MODELHUB_AK_WEIGHTS=1,1,1 \
scripts/run_with_rss_guard.sh --rss-limit-mb 26000 --check-interval-sec 60 -- \
  .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
    --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v9_catalog_first \
    --output-dir tmp/nr3d_eval_v11_enriched_full_20260522_63dc417 \
    --workers 20 \
    --sample-retries 2 \
    --use-tool-answer-disagreement-gate \
    --use-no-match-candidate-guard \
    --use-evidence-frame-guard \
    --checkpoint-only
```

The `workers=20` pass was interrupted after a ModelHub 403/timeout storm began
writing transient `PermissionDeniedError` sentinels. Those sentinel checkpoints
were deleted and the run was resumed from the same output directory at
`workers=8`, then `workers=12` after the backend stabilized:

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 MODELHUB_AK_WEIGHTS=1,1,1 \
.venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
  --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --output-dir tmp/nr3d_eval_v11_enriched_full_20260522_63dc417 \
  --workers 12 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard \
  --checkpoint-only
```

After the checkpoint pass, two `InternalServerError` sentinels were deleted and
rerun at `workers=2`. That final rerun regenerated both `side_by_side.json` and
`leaderboard_metrics.json`:

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 MODELHUB_AK_WEIGHTS=1,1,1 \
.venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
  --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --output-dir tmp/nr3d_eval_v11_enriched_full_20260522_63dc417 \
  --workers 2 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python -m evaluation.scripts.nr3d_leaderboard_metrics \
  --side-by-side tmp/nr3d_eval_v11_enriched_full_20260522_63dc417/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
  --output tmp/nr3d_eval_v11_enriched_full_20260522_63dc417/leaderboard_metrics.json \
  --canonical-filter true
```

SQLite ingestion:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v11_enriched_full_20260522_63dc417 \
  --run-id v11_enriched_full_20260522_63dc417 \
  --branch best/v10-multi-anchor-tadg-72-a3ff7f1 \
  --commit 63dc417 \
  --db docs/benchmark/nr3d/runs.sqlite \
  --leaderboard-metrics tmp/nr3d_eval_v11_enriched_full_20260522_63dc417/leaderboard_metrics.json \
  --notes "NR3D full 8584 after full-scene proposal enrichment; initial workers=20 hit 403/timeout storm, resumed w8 then w12; deleted/reran API-error sentinels before final metrics; final checkpoints have no error sentinels."
```

## Headline Metrics

| Metric | v11 enriched FULL | v11 enriched strat600 | v10 best observed strat600 |
|---|---:|---:|---:|
| Overall / filtered | **72.26** | 71.33 | 72.00 |
| Easy | 81.74 | 79.66 | **82.41** |
| Hard | **63.39** | 63.55 | 62.26 |
| V-Dep | 62.46 | 61.61 | **62.56** |
| V-Indep | **77.60** | 76.61 | 77.12 |

Counts:

| Slice | n | correct |
|---|---:|---:|
| Full prepared utterances | 8584 | 5897 |
| Canonical filtered leaderboard | 7805 | 5640 |
| Easy | 3773 | 3084 |
| Hard | 4032 | 2556 |
| V-Dep | 2752 | 1719 |
| V-Indep | 5053 | 3921 |

Pack-v1 side-by-side metrics:

| Metric | Value |
|---|---:|
| `n` | 8584 |
| `status=completed` | 8492 |
| `status=failed` | 92 |
| checkpoint `error` fields | 0 |
| missing `selected_object_id` for non-failed rows | 0 |
| mean IoU | 0.6905 |
| Acc@0.25 | 68.77 |
| Acc@0.50 | 68.70 |

## Runtime Notes

- `workers=20` was too aggressive during a backend 403/timeout storm. The run
  was stopped, transient `PermissionDeniedError` failed checkpoints were
  deleted, and checkpoint resume continued from the same output directory.
- `workers=8` cleaned up the 403 sentinels but was slow. `workers=12` was the
  stable final setting; it completed the remaining pass with no 403 sentinels.
- The final low-concurrency rerun deleted and regenerated the only remaining
  API-error checkpoints (`InternalServerError`), so the final metrics have 0
  `error` fields.
- Run log health counters over the whole append-only log: 25k+ retryable 429
  lines, 378 non-first-attempt retry lines, one QueryParser
  `All 4 keys exhausted` sample-level event, and no attempt-5 lines.

## SQLite Reproduction Query

```sql
SELECT
  run_id,
  n,
  ROUND(classification_acc_filtered * 100, 2) AS overall,
  ROUND(acc_easy * 100, 2) AS easy,
  ROUND(acc_hard * 100, 2) AS hard,
  ROUND(acc_view_dep * 100, 2) AS view_dep,
  ROUND(acc_view_indep * 100, 2) AS view_indep
FROM runs
WHERE run_id = 'v11_enriched_full_20260522_63dc417';
-- v11_enriched_full_20260522_63dc417 | 8584 | 72.26 | 81.74 | 63.39 | 62.46 | 77.60
```

## Interpretation

This is now the strongest valid NR3D row in the archive on the canonical
filtered full set. It also reconciles the strat600 picture: the 600-case v11
pilot was slightly below the best observed v10 pilot, but the full fold lands
at 72.26 % filtered Overall with strong Hard and View-Indep numbers. The
headline should be quoted as a full-set result, while keeping the runtime
caveat that high concurrency can turn backend transport failures into failed
sentinels unless they are explicitly cleaned and rerun.
