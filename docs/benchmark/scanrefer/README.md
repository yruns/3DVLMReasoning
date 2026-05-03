# ScanRefer VG Evaluation Results

ScanRefer (Chen et al. ECCV 2020) — natural-language ScanNet references,
detection-mode evaluation. Metric: axis-aligned 3D IoU; Acc@0.25 /
Acc@0.50 × { Unique, Multiple, Overall }. Judge: none (programmatic IoU).

## Current headline (v2, paper-comparable, Camp-A zero-shot Mask3D pool)

| | Unique | Multiple | Overall |
|---|---:|---:|---:|
| Acc@0.25 | 83.11 | 65.00 | **69.92** |
| Acc@0.50 | 76.49 | 57.68 | **62.79** |

Camp-A SOTA across all 6 columns vs Z3D's prior best (Δ +0.81 / +1.69 /
+13.5 / +11.98 / +11.02 / +10.09). Full version doc:
[v2_aggregation_gt_track_20260503.md](v2_aggregation_gt_track_20260503.md).
External-method comparison table: [leaderboard.md](leaderboard.md).

## Version timeline

| Version | Date | Headline | Eval scale | Notes |
|---|---|---|---|---|
| v1_mask3d_track | 2026-05-02 | Acc@0.25 = 51.65 / Acc@0.50 = 16.22 | 9508 utts (full canonical val) | First eval. **Phase 8 GT-CG bbox — not paper-comparable** (oracle ceiling 70 / 21 due to ~2× volume inflation). Retained in `runs.sqlite` for audit. |
| **v2_aggregation_gt_track** | 2026-05-03 | **Acc@0.25 = 69.92 / Acc@0.50 = 62.79** | 9508 utts (same v1 run) | Same v1 agent decisions; GT bbox swapped to ScanNet aggregation-derived (mesh + segs + axis-align). Camp-A SOTA. See v2 doc § "Audit trail" for full v1 → v2 diagnosis. |

## SQLite

```sql
SELECT run_id, n_total,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', acc25_unique)   AS u25,
       printf('%.4f', acc50_unique)   AS u50,
       printf('%.4f', acc25_multiple) AS m25,
       printf('%.4f', acc50_multiple) AS m50
FROM runs ORDER BY ingested_at;
```

## Caveats

- ScanRefer test split is server-only; we report on val (9508 utts, 141 scenes).
- Mask3D-pool detection-mode is the canonical Camp-A setup (ZSVG3D /
  SeeGround / CSVG / Z3D); GT-pool ablation is intentionally NOT pursued
  (would break the benchmark's detection-mode definition).
- Wall / floor / ceiling Mask3D candidates dropped per Camp-A convention.
- Per-LLM-call durability gap carried over from NR3D v3 — `tool_calls`
  / `llm_calls` SQLite tables empty until callback-based instrumentation
  lands.

## Design specs (historical, design rationale)

- v1: `docs/superpowers/specs/2026-05-02-scanrefer-design.md`
- v2: `docs/superpowers/specs/2026-05-03-scanrefer-v2-aggregation-gt.md`
