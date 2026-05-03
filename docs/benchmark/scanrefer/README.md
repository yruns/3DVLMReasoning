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

**v1_mask3d_track, 2026-05-02 (with 2026-05-03 oracle-analysis update)**:
First ScanRefer detection-mode eval on canonical 9508 utts. Headline
**Acc@0.25 = 51.65** / **Acc@0.50 = 16.22** with Unique/Multiple
decomposition (Unique@0.25 = 62.97, Multiple@0.25 = 47.43;
Unique@0.50 = 25.41, Multiple@0.50 = 12.79).

> **Important caveat:** v1's GT bbox is derived from Phase 8 GT-CG (a
> ConceptGraph reconstruction), which is systematically ~2× larger by
> volume than Mask3D bbox / ScanRefer official aggregation GT. The
> oracle ceiling on this fold is Acc@0.50 = 20.53 % — Z3D's published
> 52.7 % is unreachable here. Acc@0.50 = 16.22 is therefore **not
> directly comparable** to ZSVG3D 32.7 / SeeGround 39.4 / CSVG 39.8 /
> Z3D 52.7. **Acc@0.25 = 51.65 is approximately comparable** (at the
> looser threshold the bias mostly cancels). The agent's
> oracle-normalized picking quality is 73.97 % @0.25 / 79.00 % @0.50,
> consistent with NR3D v3 classification_acc = 80.79 %. v2 will
> re-aggregate against ScanNet aggregation-based GT to produce
> paper-comparable numbers.
>
> See [`v1_oracle_analysis_20260503.md`](v1_oracle_analysis_20260503.md)
> for the quantitative diagnosis, and
> `docs/superpowers/specs/2026-05-03-scanrefer-v2-aggregation-gt.md` for
> the v2 plan.

Pool is Mask3D ScanNet200 from ZSVG3D's CUHK SharePoint distribution
(the de-facto shared detector for Camp-A zero-shot methods); GT lookup
is the Phase 8 GT-CG pkl (Vil3dRef-equivalent). See
`v1_mask3d_track_20260502.md` for SOTA comparison.

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
