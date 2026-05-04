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

Current honest query-driven development result: v3.3 on the frozen random100
fold is Acc@0.25 **48.0** / Acc@0.50 **42.0** after aggregation-GT rescoring;
see [v3p3_vertical_spatial_20260504.md](v3p3_vertical_spatial_20260504.md).

## Version timeline

| Version | Date | Headline | Eval scale | Notes |
|---|---|---|---|---|
| v1_mask3d_track | 2026-05-02 | Acc@0.25 = 51.65 / Acc@0.50 = 16.22 | 9508 utts (full canonical val) | First eval. **Phase 8 GT-CG bbox — not paper-comparable** (oracle ceiling 70 / 21 due to ~2× volume inflation). Also carries the GT view oracle (see v2 row). Retained in `runs.sqlite` for audit. |
| **v2_aggregation_gt_track** | 2026-05-03 | **Acc@0.25 = 69.92 / Acc@0.50 = 62.79** | 9508 utts (same v1 run) | Same v1 agent decisions; GT bbox swapped to ScanNet aggregation-derived (mesh + segs + axis-align). GT-bbox paper-comparable, but **keyframe selector uses GT view oracle**: 5 RGB are picked by Phase 8 visibility of GT `target_id`, not query-driven Stage 1. Numbers are a controlled upper bound, not zero-shot SOTA. See v2 doc § Caveats and § Audit trail. |
| v3_query_driven (smoke) | 2026-05-03 | Phase8-GT quote: 39.0 / 14.0; **aggregation-GT corrected (re-run with v3.2 runtime fixes): 47.0 / 40.0** | 100-utt random fold (frozen seed=20260503) | First zero-shot Camp-A run; replaces `select_keyframes_from_phase8_target(target_id)` with `KeyframeSelector.select_keyframes_v2(query, k=3)` (Phase 8 hypothesis-parser, OpenEQA-style). Pack fallback rate **38%**; Stage 2 ↔ Stage 1 callbacks all wired. Re-run on 2026-05-04 with the v3.2 runtime fixes (callback-image durability, stride=1 selector, PNG paths, structured payload extractor, tool-trace persistence) and rescored against ScanNet aggregation GT for apples-to-apples vs v3.1. SQLite run: `v3_aggcorrected_random100_20260504`. **The X1 hypothesis (Mask3D-CG visibility KFs help) is falsified under this comparison: v3 (Phase 8 KFs) at 47/40 ≥ v3.1 (Mask3D-CG KFs) at 45/39.** |
| v3.1_mask3d_query_driven (smoke) | 2026-05-03 | Phase8-GT quote: 39.0 / 15.0; **aggregation-GT corrected: 45.0 / 39.0** | same 100-utt random fold | X1 patch: KFs now scored against the Mask3D-CG visibility used to render annotated PNGs. Pack fallback **0%**. The original doc's 39/15 headline used the wrong GT bbox source; the corrected aggregation-GT baseline is `v3p_agg_gt_random100_20260503` in SQLite. |
| **v3.2_callbacks_durable** | 2026-05-04 | **Acc@0.25 = 46.0 / Acc@0.50 = 40.0** | same 100-utt random fold | Runtime/evaluator correctness pass: callback images must be injected before same-turn finalization, ScanRefer selectors use stride=1, raw PNG frames resolve, structured `proposal_id` payloads extract correctly, and `tool_calls` are durable in SQLite (1899 rows). Full doc: [v3p2_callbacks_durable_20260504.md](v3p2_callbacks_durable_20260504.md). |
| **v3.3_vertical_spatial** | 2026-05-04 | **Acc@0.25 = 48.0 / Acc@0.50 = 42.0** | same 100-utt random fold | Adds `above` / `below` to `compare_proposals_spatial` and updates VG spatial skills. Net +2pp/+2pp vs v3.2 with 1891 durable `tool_calls`. Full doc: [v3p3_vertical_spatial_20260504.md](v3p3_vertical_spatial_20260504.md). |
| v3.4_select_among_proposals (NEGATIVE) | 2026-05-04 | Acc@0.25 = 44.0 / Acc@0.50 = 38.0 | same 100-utt random fold | Adds `select_among_proposals` (forced VLM 1-of-K choice over same-category candidates). Adoption was high (81/100 samples invoked the tool, 98 total calls), but Multiple@0.50 dropped 7pp (29.17 → 22.22) — the judge sees only the candidate's most-visible frame, not the spatial anchor that the description references. v3.3 stays as the headline. Documents the structural fix (co-visible-anchor frame selection) for the next iteration. Full doc: [v3p4_select_among_proposals_20260504.md](v3p4_select_among_proposals_20260504.md). |
| v3.5_cvra (NEGATIVE, smoke only) | 2026-05-05 | Acc@0.25 = 48.0 / Acc@0.50 = 42.0 (unchanged from v3.3 — no full random100 run) | 6-utt addressable smoke fold | CLIP-Visible Retrieval Augmentation (`open_clip ViT-H-14`/`ViT-B-32` reranks visible non-label-hit Mask3D proposals; K_AUG=8/10 + label-mismatch overflow tier; gated `--use-clip-visible-aug`). Smoke6: 6/6 retrieval recall (target pid in `label_hits ∪ clip_visible_aug`), 0/6 F5a flips, 5/6 identical agent picks vs v3.3 baseline. The bbox-IoU "addressability" definition (footprint overlap) overstates CVRA's reachable surface — `compare_proposals_spatial` correctly rejects audit-tagged candidates whose 3D position contradicts the description's spatial referent (e.g. on `scene0011_00::20::2`, CVRA-aug candidate is at refrigerator level while query says "above the refrigerator"). Infrastructure shipped behind `use_clip_visible_aug=False`, no SQLite ingest. Cross-validation methodology: 4 PUSHBACK rounds during dev (spec ×2, schema, peer review) + bridge-bug catch at M3a smoke saved a 3-4 h misdiagnosed full-random100 run. Full doc: [v3p5_cvra_negative_20260505.md](v3p5_cvra_negative_20260505.md). |

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
  itself is still picked from the Mask3D pool — no pool leakage). v3
  / v3.1 remove this oracle. See v2 doc § Caveats and the [‡]
  footnote on `leaderboard.md`.
- **v3.x is currently a 48% / 42% zero-shot baseline on the 100-utt
  random fold after aggregation-GT rescoring**, ~19-17pp behind the v2
  GT-view-oracle upper bound on the same fold. The original 39/15 v3.1
  quote used Phase8 GT-CG bboxes and is retained only as audit trail.
  The remaining driver is multi-distractor picking error in
  `is_unique=False` cases, not initial-keyframe coverage.
- **X1 hypothesis falsified under apples-to-apples conditions.** A
  controlled re-run of v3 (Phase 8 hypothesis-parser KFs) under the
  v3.2 runtime fixes and aggregation-GT evaluator (run id
  `v3_aggcorrected_random100_20260504`) lands at 47/40 — within noise
  of v3.1's 45/39 (Mask3D-CG visibility KFs, the X1 patch) and slightly
  ahead on Acc@0.25. Mask3D-CG candidate-aligned initial keyframes
  do not help once the runtime correctness bugs are fixed; the
  picking-stage bottleneck is unaffected by the source of the initial
  keyframes. Future iterations should target multi-distractor ranking
  rather than keyframe coverage.
- **CVRA (v3.5) negative on a different axis from v3.4.** v3.4 falsified
  judge-stage forced 1-of-K choice (`select_among_proposals`); v3.5
  falsifies retrieval-stage label augmentation (`find_proposals_by_category`
  with CLIP-text rerank over visible non-label-hit Mask3D proposals).
  Both negative results converge: the agent's pre-existing reasoning
  trajectory dominates fold-level outcomes, so improvements need to
  operate inside that trajectory rather than widening or filtering the
  candidate pool. The CVRA infrastructure stays in tree gated off and is
  reusable for any future visual-similarity index (Linux + ViT-H-14
  retry, alternative reranker, parser-fidelity probe). v3.5 also
  exposes a measurement-side defect: bbox-IoU "addressability" double-counts
  samples that the agent's spatial reasoning can correctly reject;
  future audits should add a spatial-referent verifier.
- ScanRefer test split is server-only; we report on val (9508 utts, 141 scenes).
- Mask3D-pool detection-mode is the canonical Camp-A setup (ZSVG3D /
  SeeGround / CSVG / Z3D); GT-pool ablation is intentionally NOT pursued
  (would break the benchmark's detection-mode definition).
- Wall / floor / ceiling Mask3D candidates dropped per Camp-A convention.
- Tool-call durability landed in v3.2 (`tool_calls` has 1899 rows for the
  random100 run). `llm_calls` is still empty for ScanRefer until callback-based
  LLM instrumentation lands.

## Design specs (historical, design rationale)

- v1: `docs/superpowers/specs/2026-05-02-scanrefer-design.md`
- v2: `docs/superpowers/specs/2026-05-03-scanrefer-v2-aggregation-gt.md`

## Investigation Summaries

- [final_investigation_summary_20260504.md](final_investigation_summary_20260504.md)
