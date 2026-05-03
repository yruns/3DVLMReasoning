# ScanRefer VG Evaluation Results

This directory tracks ScanRefer visual-grounding evaluations for the
Stage-2 task-pack pipeline.

**Benchmark:** ScanRefer (Chen et al. ECCV 2020) — natural-language ScanNet references with detection-mode evaluation
**Metric:** axis-aligned 3D IoU; Acc@0.25 / Acc@0.50 × { Unique, Multiple, Overall }
**Judge:** none (programmatic IoU)

## Version Timeline

| Version | Date | Headline | Eval Scale | Key Change |
|---|---|---|---|---|
| [v1_mask3d_track](v1_mask3d_track_20260502.md) | 2026-05-02 | Acc@0.25 = 51.65 / Acc@0.50 = 16.22 | 9508 utts (full canonical val) | First ScanRefer eval. Mask3D ScanNet200 pool from ZSVG3D distribution; gpt-5.4-2026-03-05 backend; full 141 / 141 scene coverage. **Phase 8 GT-CG bbox — not paper-comparable** (see oracle analysis). |
| [**v2_aggregation_gt_track**](v2_aggregation_gt_track_20260503.md) | 2026-05-03 | **Acc@0.25 = 69.92 / Acc@0.50 = 62.79** | 9508 utts (same v1 run) | Same v1 agent decisions, GT bbox swapped to ScanNet aggregation-derived (mesh + segs + axis-align) — paper-comparable. **Camp-A SOTA across all 6 columns** (Z3D +10pp on @0.50). |

## Current Interpretation

**v2_aggregation_gt_track, 2026-05-03 (paper-comparable headline)**:
Re-aggregation of the v1 9508-utt agent run with ScanNet
aggregation-based GT bbox (the canonical Camp-A reference). Headline
**Acc@0.25 = 69.92 / Acc@0.50 = 62.79** with Unique/Multiple
decomposition (Unique@0.25 = 83.11, Multiple@0.25 = 65.00;
Unique@0.50 = 76.49, Multiple@0.50 = 57.68); mean IoU = 0.6022. Same
9508 agent decisions as v1 — only the GT side of IoU was swapped from
Phase 8 GT-CG to mesh-aggregation-derived AABB. Full diagnosis +
oracle/picking-quality sanity check in
[`v2_aggregation_gt_track_20260503.md`](v2_aggregation_gt_track_20260503.md).

**Camp-A SOTA on Mask3D-pool zero-shot across all 6 columns**:

| | Unique@0.25 | Unique@0.50 | Multiple@0.25 | Multiple@0.50 | Overall@0.25 | Overall@0.50 |
|---|---:|---:|---:|---:|---:|---:|
| Z3D (prior SOTA) | 82.3 | 74.8 | 51.5 | 45.7 | 58.9 | 52.7 |
| **Ours v2** | **83.11** | **76.49** | **65.00** | **57.68** | **69.92** | **62.79** |
| Δ | +0.81 | +1.69 | +13.5 | +11.98 | +11.02 | +10.09 |

The Multiple gap is the largest — language-disambiguation (the part
where RGB+VLM agents have a structural advantage over pure-3D models)
shows the strongest improvement.

**v1_mask3d_track, 2026-05-02 (historical record)**: First ScanRefer
detection-mode eval. Headline **Acc@0.25 = 51.65 / Acc@0.50 = 16.22**.
Used Phase 8 GT-CG bbox (ConceptGraph reconstruction), which is
systematically ~2× larger than mesh-aggregation GT. Oracle ceiling on
v1 was Acc@0.50 = 20.53 %, so the v1 number is **not directly
comparable** to published Camp-A baselines. v1 is preserved as the
audit trail showing why v2 was needed; see
[`v1_oracle_analysis_20260503.md`](v1_oracle_analysis_20260503.md) for
the quantitative diagnosis that motivated v2.

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
