# ScanRefer VG Evaluation Results

ScanRefer (Chen et al. ECCV 2020) — natural-language ScanNet references,
detection-mode evaluation. Metric: axis-aligned 3D IoU; Acc@0.25 /
Acc@0.50 × { Unique, Multiple, Overall }. Judge: none (programmatic IoU).

## Current headline (v2, GT-bbox paper-comparable, GT view oracle)

| | Unique | Multiple | Overall |
|---|---:|---:|---:|
| Acc@0.25 | 83.11 | 65.00 | **69.92** |
| Acc@0.50 | 76.49 | 57.68 | **62.79** |

Beats every Camp-A zero-shot baseline by per-column wins of Δ +0.81 /
+1.69 / +13.5 / +11.98 / +11.02 / +10.09 vs Z3D, **but the keyframe
selector uses a GT view oracle** (5 initial RGB frames are picked by
GT `target_id` visibility, not by query-driven Stage 1 retrieval),
so this is not a like-for-like zero-shot comparison. v3 query-driven
will be the apples-to-apples headline. Full version doc:
[v2_aggregation_gt_track_20260503.md](v2_aggregation_gt_track_20260503.md).
External-method comparison table: [leaderboard.md](leaderboard.md).

## Version timeline

| Version | Date | Headline | Eval scale | Notes |
|---|---|---|---|---|
| v1_mask3d_track | 2026-05-02 | Acc@0.25 = 51.65 / Acc@0.50 = 16.22 | 9508 utts (full canonical val) | First eval. **Phase 8 GT-CG bbox — not paper-comparable** (oracle ceiling 70 / 21 due to ~2× volume inflation). Also carries the GT view oracle (see v2 row). Retained in `runs.sqlite` for audit. |
| **v2_aggregation_gt_track** | 2026-05-03 | **Acc@0.25 = 69.92 / Acc@0.50 = 62.79** | 9508 utts (same v1 run) | Same v1 agent decisions; GT bbox swapped to ScanNet aggregation-derived (mesh + segs + axis-align). GT-bbox paper-comparable, but **keyframe selector uses GT view oracle**: 5 RGB are picked by Phase 8 visibility of GT `target_id`, not query-driven Stage 1. Numbers are a controlled upper bound, not zero-shot SOTA. See v2 doc § Caveats and § Audit trail. |
| v3_query_driven *(planned)* | TBD | TBD | 9508 utts | Same agent + Mask3D pool as v2; replaces `select_keyframes_from_phase8_target(target_id)` with `query_scene.keyframe_selector.select_keyframes_v2(query, ...)` (the OpenEQA entry, hypothesis-driven). First apples-to-apples zero-shot Camp-A run. |

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

- **GT view oracle in v1 / v2 keyframe selection.** The 5 initial RGB
  keyframes per query are picked by Phase 8 visibility of the GT
  `target_id`, not via query-driven Stage 1 retrieval. This guarantees
  the target appears in the initial visual evidence (the `proposal_id`
  itself is still picked from the Mask3D pool — no pool leakage). The
  v3 query-driven track will remove this oracle. See v2 doc § Caveats
  and the [‡] footnote on `leaderboard.md`.
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
