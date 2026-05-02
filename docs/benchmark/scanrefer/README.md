# ScanRefer VG Evaluation Results

This directory tracks ScanRefer visual-grounding evaluations for the
Stage-2 task-pack pipeline.

**Benchmark:** ScanRefer (Chen et al. ECCV 2020) — natural-language ScanNet references with detection-mode evaluation
**Metric:** axis-aligned 3D IoU; Acc@0.25 / Acc@0.50 × { Unique, Multiple, Overall }
**Judge:** none (programmatic IoU)

## Version Timeline

| Version | Date | Headline | Eval Scale | Key Change |
|---|---|---|---|---|
| [v1_mask3d_track](v1_mask3d_track_20260502.md) | 2026-05-02 | Acc@0.25 = 51.65 / Acc@0.50 = 16.22 | 9508 utts (full canonical val) | First ScanRefer eval. Mask3D ScanNet200 pool from ZSVG3D distribution; gpt-5.4-2026-03-05 backend; full 141 / 141 scene coverage (130 NR3D-overlap + 11 newly-built). |

## Current Interpretation

**v1_mask3d_track, 2026-05-02**: First ScanRefer detection-mode eval on
canonical 9508 utts. Headline **Acc@0.25 = 51.65** /
**Acc@0.50 = 16.22** with Unique/Multiple decomposition (Unique@0.25 =
62.97, Multiple@0.25 = 47.43; Unique@0.50 = 25.41, Multiple@0.50 =
12.79). Pool is Mask3D ScanNet200 from ZSVG3D's CUHK SharePoint
distribution (the de-facto shared detector for Camp-A zero-shot
methods); GT lookup is the Phase 8 GT-CG pkl (Vil3dRef-equivalent).
Acc@0.25 is competitive with Camp-A SOTA (mid-pack vs ZSVG3D 36.4 /
CSVG 49.6 / SeeGround 44.1 / VLM-Grounder 51.6 / Z3D 58.9). Acc@0.50 is
notably below Camp-A peers — see v1 doc for the localization-vs-fit
hypothesis. See `v1_mask3d_track_20260502.md` for SOTA comparison.

## Reproduction

See the `Reproduction Command` section in
[v1_mask3d_track_20260502.md](v1_mask3d_track_20260502.md).

## SQLite

Canonical DB: `docs/benchmark/scanrefer/runs.sqlite`

```sql
SELECT run_id, n_total,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50
FROM runs;
```

## Caveats

- ScanRefer test split is server-only; we report on val (9508 utts, 141 scenes).
- Mask3D-pool detection-mode is the canonical setup (ZSVG3D / SeeGround / CSVG / Z3D); GT-pool ablation is intentionally NOT pursued (would break benchmark).
- Wall/floor/ceiling Mask3D candidates dropped per Camp-A convention.
- Aggregator `compute_leaderboard_metrics` OOM'd on macOS due to full-val Phase 8 pkl load; v1 used a scene-by-scene memory-frugal driver (logically equivalent). v2 should patch the aggregator to accept a `sample_ids` filter.
