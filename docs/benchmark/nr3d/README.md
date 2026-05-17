# NR3D VG Benchmark Archive

This directory is the permanent process archive for NR3D visual grounding
evaluations in this repo. Keep the version docs immutable; use this README as
the current human-facing index.

## Entry Points

| File | Purpose |
|---|---|
| [leaderboard.md](leaderboard.md) | Public SOTA context plus our invalidated v5.1 audit row and historical rows. |
| [protocol.md](protocol.md) | Consolidated protocol notes: metric family, fold/filter rules, candidate-pool equivalence, fairness boundary. |
| [depth_visibility_rebuild_20260513.md](depth_visibility_rebuild_20260513.md) | Root-cause record and rebuild summary for depth-aware NR3D visibility indices. |
| [depth_visibility_spotcheck_20260513.html](depth_visibility_spotcheck_20260513.html) | Visual spotcheck frames rendered from rebuilt depth-aware `view_to_objects`. |
| [v9_2_full_select_by_text_ab_20260516.md](v9_2_full_select_by_text_ab_20260516.md) | **Full 8584-utterance A/B** of the `enable_stage1_text_retrieval` toggle. Filtered (n=7805): text-first **65.20 %** vs catalog-first **64.65 %** (Δ **+0.55 pp** overall). Per-tier: text-first wins on **Hard +1.14** and **V-Indep +1.56**; catalog-first wins on **V-Dep −1.31**; Easy is tied. **0** Python errors across 17 168 sample runs; 46 graceful no-match `-1` failures. Random100's 5 pp gap shrinks to 0.55 pp at full scale — the audit hypothesis is now *partially* validated (it holds on view-dep, not on hard / view-indep). **Decision: keep `select_by_text` on by default; v9.3 should route per query type.** |
| [v9_2_full_text_first_trace_20260516.html](v9_2_full_text_first_trace_20260516.html) | VG agent trace viewer (2T+4F) for the **text-first** full run. Cases picked to show variant-specific wins/losses: 2 correct (where text-first was uniquely right), 2 losses where no-text was correct, 2 losses both shared. Covers attribute / superlative / spatial-anchor / view-dep query types. |
| [v9_2_full_no_text_trace_20260516.html](v9_2_full_no_text_trace_20260516.html) | VG agent trace viewer (2T+4F) for the **catalog-first** full run. Mirrors the text-first selection: 2 correct (where catalog-first uniquely won), 2 losses where text-first was correct, 2 losses both shared. Direct side-by-side reading of the same cases (`scannet/scene0249_00::32::36779` appears in both as opposite outcomes). |
| [v9_3_strat10_v93ui_smoke_trace_20260517.html](v9_3_strat10_v93ui_smoke_trace_20260517.html) | **10-case smoke validating v9.3 UI overhaul** (BEV on-demand labels + mark_frame_with_bbox top-left colour-matched labels). Overall **60.0 %** (6/10), Easy **83.3 %** (5/6), Hard **25.0 %** (1/4), V-Dep **40.0 %** (2/5), V-Indep **80.0 %** (4/5). Tool usage: `select_by_text` 3× (0 errors, 6 frames), `view_bev` 9× including **3× with new `categories=` argument**, `mark_frame_with_bbox` 23×. **0 None-selector errors** + **0 stale cache hits** (new `_v93` cache tag invalidates the May-16 OLD-style overlays). Also includes a 4-pane visual at `assets/v9_3_strat10_v93ui_smoke_trace_20260517/_summary_4pane.png` showing (A) clean default BEV, (B) view_bev(highlight) with focused labels, (C–D) mark_frame with colour-matched top-left labels. |
| [assets/v9_3_mark_label_ux_20260517/comparison_old_vs_new.png](assets/v9_3_mark_label_ux_20260517/comparison_old_vs_new.png) | **v9.3 `mark_frame_with_bbox` label overhaul** (commit pending). Label is now (a) ~20 % larger (SIMPLEX→DUPLEX, scale floor 0.7→0.84, +20 % at the 1500 px ref width); (b) always anchored at the bbox top-left INSIDE the bbox (with auto fallback to "just above the bbox top" when the bbox is too small); (c) background filled with the bbox stroke colour, text colour auto-picked from black/white by ITU-R BT.601 luminance (>140 → black, else white). Before/after composite on scene0608_00 frame 7 with 4 overlapping bboxes (plant / coffee table / beanbag chair / recliner chair). 7 new tests, 16/16 mark_frame_with_bbox tests green. |
| [assets/v9_3_bev_ux_20260517/bev_4way_comparison.png](assets/v9_3_bev_ux_20260517/bev_4way_comparison.png) | **v9.3 BEV UX overhaul** (commit pending). 4-way visual: (1) OLD = label-every-proposal (cluttered, 1500×1500 with ~30 % wasted whitespace, labels illegible at thumbnail scale); (2) NEW default = mesh + trajectory + small dots, **no text labels** (918×1162 after tighter crop, 53 % area reduction); (3) NEW `view_bev(highlight=[6, 20])` = label only those two proposals on bright-yellow background with bold black text; (4) NEW `view_bev(categories=["door"])` = label every proposal whose category matches. Mirrors `mark_frame_with_bbox` UX. 8 new tests; default-no-labels contract pinned. |
| [v9_3_strat10_smoke_trace_20260517.html](v9_3_strat10_smoke_trace_20260517.html) | **10-case smoke on the strat600 fold** validating the v9.3 "runtime.keyframe_selector is None" root-cause fix (commit `32bfd0e`). Default config (text retrieval ON) + all 3 guards (TADG / no-match / evidence-frame) + `pack_nr3d_v9_catalog_first`. Overall **70.0 %** (7/10), Easy **83.3 %** (5/6), Hard **50.0 %** (2/4), V-Dep **60.0 %** (3/5), V-Indep **80.0 %** (4/5). `select_by_text` invoked **3×** across the 10 cases with **0 ERROR responses** (vs 100/100 in the historical v9.1_fix wrapper-bypass bug); **6 valid frames** returned from real Stage-1; 1 case hit the audit's "empty-pred" wall (spatial-anchor phrasing on a chalkboard query). 145 tool calls rendered, 115 image assets. |
| [v9_3_strat600_subset_design_20260517.md](v9_3_strat600_subset_design_20260517.md) | **600-case stratified NR3D fold** intended as a faster proxy for the canonical filtered 7805. Stratified on `(is_easy, is_view_dep)` with salt search across 4096 candidates to pin the salt-locked metrics to within ±0.19 pp of v9.1_fix FULL on every leaderboard column (Overall 0.8300 vs 0.8295, Easy −0.09, Hard +0.19, V-Dep −0.03, V-Indep +0.09). 1000-trial Monte-Carlo bootstrap confirms the design is unbiased (|bias| < 0.0015 / metric) with ~±2.3 pp 90 % band on Overall. 119 / 130 scenes (91.5 %), 65 / 72 categories (90.3 %). Fold ID: `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json`; salt: `nr3d_v9_3_strat600_v291`. |
| [v9_1_fix_FULL_REPRO_20260516.md](v9_1_fix_FULL_REPRO_20260516.md) | 🥇 **v9.1_fix run on FULL 8584 test fold** at the original commit `d5f40ba`. **classification_acc_filtered = 82.95 %** — the highest depth-aware NR3D test result in the project (beats the invalidated v3 80.79 %). +17.75 pp over v9.2 text-first / +18.30 pp over v9.2 no-text on the same 7805-Q fold. Hard +22.80, V-Dep +21.40 — confirms the catalog-only fallback policy dominates "Stage-1 first move" by a large margin, not just at random100 scale. **v9.3 should default to catalog-only and surface Stage-1 only conditionally.** |
| [v9_1_fix_FULL_trace_20260516.html](v9_1_fix_FULL_trace_20260516.html) | VG agent trace viewer (2T+4F) for the v9.1_fix FULL REPRO run. 2 correct cases where v9.1_fix uniquely won vs v9.2 text-first (view-dep "standing in the middle facing the carts", attribute "tall skinny white picture") — show the catalog-enumerate-after-Stage-1-ERROR pattern. 4 wrong cases (2 where v9.2 won, 2 both-wrong) for failure-mode contrast. |
| [v9_1_fix_reproduction_20260516.md](v9_1_fix_reproduction_20260516.md) | **v9.1_fix random100 reproduction** at the original commit `d5f40ba` (worktree). 86 → **84** within ±2 pp single-seed noise band. Confirms the 86 baseline is **not** a measurement artifact, and the v9.1_fix → v9.2 16-18 pp regression is real (8-9× the noise band). Adds the `Pre-run checklist (MANDATORY)` rule to `CLAUDE.md` so every iteration commits + records both head + run-time SHAs. |
| [v9_2_select_by_text_ab_20260516.md](v9_2_select_by_text_ab_20260516.md) | **random100 A/B** (n=100, ~13 min): text-first **68.0 %** vs catalog-first **63.0 %** (Δ +5 pp). Direction same as full set, magnitude inflated by small-fold noise. Useful as a quick smoke before launching the 15h full-set run. |
| [v9_1_select_by_text_audit_20260516.md](v9_1_select_by_text_audit_20260516.md) | Diagnostic — `select_by_text` reliability vs GT frame visibility on the same random100 fold. Headline: **hit@3 = 55 %** (47 % at K=1, plateaus at K=10), driven by a **32 % empty-prediction wall** (parser/executor returns `no_evidence`). When non-empty, hit@3 = **81 %** and `mean_first_hit_rank` ≈ 1.16 — Stage-1 is either great or hopeless. Strongest on hard / rare-target queries (+31 pp vs random), hurts on easy ones. Quantitatively reconciles the v9.1_fix → v9.1_real regression and gives the v9.1 → v9.2 playbook trade. |
| [v9_1_real_stage1_actually_works_20260516.md](v9_1_real_stage1_actually_works_20260516.md) | v9.1_real — Stage-1 wrapper-fix (`de8225f`); `select_by_text` truly reaches Stage-1 in 100/100 samples (Overall **69.00**, Hard **57.63**, View-dep **58.82**). **−17 pp regression** reveals the v9.1 "first move = `select_by_text`" prior is wrong for NR3D. Bug fix is correct; playbook reorder is the next move. |
| [v9_1_real_trace_20260516.html](v9_1_real_trace_20260516.html) | VG agent trace viewer (2T+4F) on the v9.1_real run — failures show how Stage-1 returns geometrically misleading frames for view-dep / attribute queries. |
| [v9_1_fix_keyframe_selector_wiring_20260515.md](v9_1_fix_keyframe_selector_wiring_20260515.md) | v9.1_fix — INCOMPLETE wiring fix; `Stage2DeepResearchAgent.build_agent` wrapper still bypassed the selector. Run scored Overall **86.00** but with `select_by_text` silently erroring in 100/100 samples — score reflects catalog-only fallback, not v9.1 design. Superseded by `v9_1_real_*`. |
| [v9_1_fix_trace_20260515.html](v9_1_fix_trace_20260515.html) | VG agent trace viewer (2T+2F batch 1) on the v9.1_fix run — same 4 sample IDs as the v9.1 broken trace for side-by-side comparison. |
| [v9_1_fix_trace_batch2_20260515.html](v9_1_fix_trace_batch2_20260515.html) | VG agent trace viewer (2T+2F batch 2) on the v9.1_fix run — 4 fresh cases including hard / view-dep wins (scene0164, scene0131) and instructive failures (scene0697, scene0149). |
| [v9_1_selectors_return_images_20260515.md](v9_1_selectors_return_images_20260515.md) | v9.1 initial — selectors-return-RGB + mark_frame_with_bbox + BEV fixes (Overall **81.00**, Hard **79.66**). −5 pp regression caused by a Stage-1 wiring bug; see v9.1_fix for the resolution. |
| [v9_1_selectors_return_images_trace_20260515.html](v9_1_selectors_return_images_trace_20260515.html) | VG agent trace viewer (4 cases) from the v9.1 initial run; mostly useful as a "before-fix" reference. |
| [v9_catalog_first_20260515.md](v9_catalog_first_20260515.md) | v9 catalog-first random100 pilot, leak-fixed (Overall **86.00**, Hard **81.36**). |
| [v9_selective_mark_random100](v9_selective_mark_random100_20260514.md) | Latest clean-initial + selective marked-image random100 pilot. |
| [v9_selective_mark_trace_2t2f_20260514.html](v9_selective_mark_trace_2t2f_20260514.html) | Static VG agent trace viewer for 2 correct and 2 failed v9 random100 cases. |
| [v8_clean_initial_marked_on_demand_random100](v8_clean_initial_marked_on_demand_random100_20260514.md) | Clean-initial/marked-on-demand negative random100 ablation. |
| [v8_clean_initial_trace_2t2f_20260514.html](v8_clean_initial_trace_2t2f_20260514.html) | Static VG agent trace viewer for 2 correct and 2 failed v8 random100 cases. |
| [v7_stage1_callbacks_noclip_random100](v7_stage1_callbacks_noclip_random100_20260513.md) | Best current depth-aware random100 callback-wired pilot. |
| [v6_inline_labels_depth_visible_random100](v6_inline_labels_depth_visible_random100_20260513.md) | Depth-aware random100 no-NMS rerender pilot before NR3D callbacks were wired. |
| [v6_random100_case_studies_20260513.html](v6_random100_case_studies_20260513.html) | Full Stage1+Stage2 visual walkthrough for 2 correct and 2 failed v6 random100 cases. |
| [v5p1_failed_rerun_full_20260513.md](v5p1_failed_rerun_full_20260513.md) | Latest full-test audit record; invalidated pending depth-aware rerun. |
| [v5p1_case_studies_20260513.html](v5p1_case_studies_20260513.html) | Visual stage1+stage2 reasoning walkthrough for selected correct and failed cases. |
| [runs.sqlite](runs.sqlite) | Queryable per-run and per-sample metrics. |

## Current Result Status

**Headline (depth-aware, full 8584 / 7805 filtered)**:
**v9.1_fix FULL REPRO 82.95 %** — see
[v9_1_fix_FULL_REPRO_20260516.md](v9_1_fix_FULL_REPRO_20260516.md).
Run-time code commit `d5f40ba` (worktree, intentional drift to reproduce
the broken-Stage-1 wrapper bypass), head commit at launch `e750d8b`,
classification accuracy with NR3D canonical filter, single-side
workers=40, ~8h32m wall, 0 Python errors.

| Metric | **v9.1_fix FULL REPRO** | v3 (invalidated) | v5.1 (invalidated) | v9.2 text-first | v9.2 no-text |
|---|---:|---:|---:|---:|---:|
| Overall   | **82.95** | 80.79 | 68.48 | 65.20 | 64.65 |
| Easy      | 88.36 | 86.06 | 78.43 | 76.01 | 76.09 |
| Hard      | **77.88** | 75.87 | 59.18 | 55.08 | 53.94 |
| V-Dep     | **78.23** | 72.46 | 57.38 | 56.83 | 58.14 |
| V-Indep   | 85.51 | 85.34 | 74.53 | 69.76 | 68.20 |

The catalog-only fallback policy (broken Stage-1) wins by 17-23 pp vs
v9.2's "Stage-1 first-move" prior, confirming the random100 finding at
full scale. v3's 80.79 % is invalidated due to projection-only
visibility; v9.1_fix uses the canonical depth-aware visibility, so the
2.16 pp lead over v3 is the first **valid** depth-aware result above the
v3 number.

The v5/v5.1 records (older "fair-view" full runs) are still
**invalidated pending rerun**. Their packs were built from NR3D
`visibility_index.pkl` files whose metadata records `use_depth=False`, so
`view_to_objects` / `object_to_views` encoded projection/frustum
candidates rather than depth-occlusion visibility. Those indices are not
valid as agent-visible evidence.

Latest recorded, invalidated fair-view full-test row:

- Version: `v5p1_failed_rerun_full_20260513`
- Branch: `feat/nr3d-v4-agent-guards-fair-views`
- Run-time code commit: `c404536`
- Documentation commit introducing the v5.1 record: `6000e2b`
- Fold: 8584 NR3D test utterances; n_filtered=7805 after canonical
  `mentions_target_class=True`
- Evidence selection: query-driven fair keyframes; target id / GT bbox used
  only for scoring
- Stage 2 backend: `gpt-5.4-2026-03-05`
- Raw merged artifacts: `tmp/nr3d_eval_v5_failed_rerun_merged_20260513/`
- SQLite run id: `v5p1_failed_rerun_full_20260513`
- Case-study HTML:
  [v5p1_case_studies_20260513.html](v5p1_case_studies_20260513.html)

| Metric | v5.1 recorded, invalidated | v5 before failed-rerun | v3 recorded, invalidated | UniVLG public SOTA |
|---|---:|---:|---:|---:|
| Overall | **68.48** | 66.53 | 80.79 | 65.20 |
| Easy | **78.43** | 76.09 | 86.06 | 73.30 |
| Hard | **59.18** | 57.59 | 75.87 | 57.00 |
| View-Dep | **57.38** | 55.74 | 72.46 | 55.10 |
| View-Indep | **74.53** | 72.41 | 85.34 | 69.90 |

The v5.1 run reran all 241 failed sentinels from v5. Of those, 240 recovered
to completed outputs and 1 persisted as no-match. That persistent row is
filtered out of the 7805Q headline because `mentions_target_class=false`.
The rerun adds 152 correct samples inside the filtered fold and moves overall
from 66.53 to 68.48.

Interpretation:

- v5.1 is no longer a valid fair-view NR3D result because the underlying
  object-frame visibility index did not use depth occlusion.
- Do not compare v5.1 against UniVLG or any other SOTA row until the depth-aware
  visibility indices are rebuilt, packs regenerated, and the full run rerun.
- v3 is also invalidated as a benchmark claim because its GT-target-visible
  shortcut used the same projection-only visibility source.

Latest depth-aware partial pilot (catalog-first, leak-fixed):

- Version: `v9_catalog_first` (clean re-run)
- Branch / commit: `feat/v9-catalog-first-scene-exploration` / leak-fix commit
  (see version doc)
- Scope: same v4/v6/v7/v8/v9_selective_mark random100 fold; **BEV image + Cat-B
  text inventory only** as initial evidence (no first-person seed keyframes;
  `Stage2RuntimeState.initial_keyframe_paths` now blocks the pack-prep seed
  drain); 6 catalog-first selectors + unified `view_keyframe(mode='auto')` +
  `view_bev`
- Result: Overall **86.00**, Easy **92.68**, Hard **81.36**, View-Dep **82.35**,
  View-Indep **87.88**
- Raw artifacts: `tmp/nr3d_eval_v9_full_clean_20260515_1442/`
- Note: First *clean* catalog-first random100 result. **+12.00 overall** and
  **+18.65 Hard** vs v9_selective_mark — confirms that a global BEV primer
  plus selector-driven evidence wins decisively over first-person seed
  keyframes on this fold. Earlier `v9_full` (leaky, `tmp/nr3d_eval_v9_full_20260515_1401/`)
  scored 81.00 / 85.37 / 77.97 / 73.53 / 84.85; the +5 / +7.3 / +3.4 / +8.8 /
  +3.0 lift is the post-fix evidence that the leak was *biasing* the agent
  toward the GT-target-visible seeds rather than helping it.

Previous depth-aware partial pilot:

- Version: `v9_selective_mark_random100`
- Branch / commit: `feat/nr3d-v4-agent-guards-fair-views` /
  `4a1fba1-dirty-selective-mark`
- Scope: same v4/v6/v7/v8 random100 fold; clean initial RGB keyframes with
  text-only inventories; `list_frame_proposals` plus filtered
  `view_keyframe_marked(categories/proposal_ids)` for crowded frames
- Result: Overall 74.00, Easy 90.24, Hard 62.71, View-Dep 64.71,
  View-Indep 78.79
- Raw artifacts:
  `tmp/nr3d_eval_v9_selective_mark_random100_20260514/`
- Case-study HTML:
  [v9_selective_mark_trace_2t2f_20260514.html](v9_selective_mark_trace_2t2f_20260514.html)
- Note: Selective marked-image rendering, but kept first-person seed
  keyframes; superseded by v9_catalog_first.

Earlier depth-aware partial pilot:

- Version: `v8_clean_initial_marked_on_demand_random100`
- Branch / commit: `feat/nr3d-v4-agent-guards-fair-views` / `f86c161`
- Scope: same v4/v6/v7 random100 fold; clean initial RGB keyframes with
  text-only left-to-right proposal inventories; marked images only through
  `view_keyframe_marked`
- Result: Overall 67.00, Easy 85.37, Hard 54.24, View-Dep 58.82,
  View-Indep 71.21
- Raw artifacts:
  `tmp/nr3d_eval_v8_clean_initial_marked_on_demand_random100_20260514/`
- Note: Negative ablation vs v7. The change improves Easy but significantly
  hurts Hard and View-Dep; keep v7 as the stronger random100 baseline.

Previous best depth-aware random100 pilot:

- Version: `v7_stage1_callbacks_noclip_random100`
- Branch / commit: `feat/nr3d-v4-agent-guards-fair-views` / `690cbf6`
- Scope: same v4 random100 fold; preserved v4 keyframe frame ids; no selector
  NMS
- Result: Overall 73.00, Easy 82.93, Hard 66.10, View-Dep 70.59,
  View-Indep 74.24
- Raw artifacts:
  `tmp/nr3d_eval_v7_callbacks_noclip_random100_20260513/`
- Note: NR3D Stage1 callbacks are wired; `request_more_views` disables
  per-selector CLIP object-term fallback to stay within the 15GB RSS budget at
  `workers=100`.

## Version Timeline

| Version | Date | Branch / commit | Headline | Scope | Status |
|---|---|---|---:|---|---|
| [v1_phase8_smoke20_mac](v1_phase8_smoke20_mac_20260430.md) | 2026-04-30 | `feat/nr3d-vg-benchmark` / `9115fd7` | Acc@0.50=65.00 | 20Q smoke | Invalidated: projection-only visibility |
| [v2_phase8_full](v2_phase8_full_20260501.md) | 2026-05-01 | `feat/nr3d-vg-benchmark` / `ad8d439` plus in-flight fixes | Acc@0.50=77.62 | 8584Q full | Invalidated: projection-only visibility |
| [v3_referit3d_track](v3_referit3d_track_20260501.md) | 2026-05-01 | `feat/nr3d-vg-benchmark` / `b6f211a` | Overall=80.79 | 8584Q / 7805Q filtered | Invalidated: projection-only GT-visible source |
| [v4_agent_guards_fair_views](v4_agent_guards_fair_views_20260512.md) | 2026-05-12 | `feat/nr3d-v4-agent-guards-fair-views` / `3f1c6d8` | Overall=71.00 | 100Q pilot | Invalidated: projection-only visibility |
| [v5_agent_guards_fair_views_full](v5_agent_guards_fair_views_full_20260513.md) | 2026-05-13 | `feat/nr3d-v4-agent-guards-fair-views` / `7c996ad` | Overall=66.53 | 8584Q / 7805Q filtered | Invalidated: projection-only visibility |
| [v5p1_failed_rerun_full](v5p1_failed_rerun_full_20260513.md) | 2026-05-13 | `feat/nr3d-v4-agent-guards-fair-views` / `c404536` | Overall=68.48 | 8584Q / 7805Q filtered | Invalidated: projection-only visibility |
| [v6_inline_labels_depth_visible_random100](v6_inline_labels_depth_visible_random100_20260513.md) | 2026-05-13 | `feat/nr3d-v4-agent-guards-fair-views` / `41253ad` | Overall=71.00 | 100Q pilot | Depth-aware partial, no NMS |
| [v7_stage1_callbacks_noclip_random100](v7_stage1_callbacks_noclip_random100_20260513.md) | 2026-05-13 | `feat/nr3d-v4-agent-guards-fair-views` / `690cbf6` | Overall=73.00 | 100Q pilot | Depth-aware partial, callbacks wired, no NMS |
| [v8_clean_initial_marked_on_demand_random100](v8_clean_initial_marked_on_demand_random100_20260514.md) | 2026-05-14 | `feat/nr3d-v4-agent-guards-fair-views` / `f86c161` | Overall=67.00 | 100Q pilot | Depth-aware partial, negative clean-initial ablation |
| [v9_selective_mark_random100](v9_selective_mark_random100_20260514.md) | 2026-05-14 | `feat/nr3d-v4-agent-guards-fair-views` / `4a1fba1-dirty-selective-mark` | Overall=74.00 | 100Q pilot | Depth-aware partial, selective marked-image rendering |
| [v9_catalog_first_20260515](v9_catalog_first_20260515.md) | 2026-05-15 | `feat/v9-catalog-first-scene-exploration` / leak-fix | Overall=**86.00** (clean) / 81.00 (leaky) | 100Q pilot | Depth-aware partial, BEV-first + 6 selectors, no first-person seed; Stage-1 seed-keyframe drain leak documented + fixed |
| [v9_2_select_by_text_ab_20260516](v9_2_select_by_text_ab_20260516.md) | 2026-05-16 | `feat/v9-1-selectors-return-images` / `c2c52d0` | text-first **68.00** / catalog-first **63.00** | 100Q A/B | Depth-aware partial; `enable_stage1_text_retrieval` toggle; same code/prompts/pack — only the toggle differs. Audit's catalog-first hypothesis falsified at random100 (−5 pp). |
| [v9_2_full_select_by_text_ab_20260516](v9_2_full_select_by_text_ab_20260516.md) | 2026-05-16 | `feat/v9-1-selectors-return-images` / `c2c52d0` | text-first **65.20** / catalog-first **64.65** | **8584Q / 7805Q filtered A/B** | First full-set v9.x evaluation. 15h28m parallel on Mac, workers=20 each, 0 Python errors. Gap shrinks from random100's 5pp to 0.55pp at full scale; per-tier text-first wins Hard / V-Indep, catalog-first wins V-Dep. Default decision: keep `select_by_text` on. |
| [v9_1_fix_FULL_REPRO_20260516](v9_1_fix_FULL_REPRO_20260516.md) | 2026-05-16 | `feat/v9-1-selectors-return-images` / head `e750d8b`, runtime `d5f40ba` (worktree) | **82.95 / 88.36 / 77.88 / 78.23 / 85.51** | 8584Q / 7805Q filtered | **🥇 New depth-aware NR3D high.** v9.1_fix wrapper-bypass bug reproduces broken Stage-1 catalog fallback on full set. +17.75 / +22.80 / +21.40 pp on Overall / Hard / V-Dep vs v9.2 text-first. workers=40 single side, ~8h32m wall, 0 Python errors. |
| [v9_3_strat600_subset_design_20260517](v9_3_strat600_subset_design_20260517.md) | 2026-05-17 | `feat/v9-1-selectors-return-images` / `4728205` | _no run; fold design_ | 600Q stratified ⊂ 7805Q filtered | Salt-locked 600-case fold pinned to within ±0.19 pp of v9.1_fix FULL on every leaderboard metric (Overall **0.8300** vs ref 0.8295). 1000-trial bootstrap shows design unbiased, 90 % band ±2.3 pp on Overall. Use as a fast proxy for full-set runs (estimated 1h vs 8h on workers=20+). |

## Protocol Summary

- Public NR3D leaderboard headline is classification accuracy:
  `selected_object_id == target_id`.
- Canonical local full fold: 8584 utterances, 130 scenes.
- Canonical filtered fold: 7805 utterances after `mentions_target_class=True`.
- Easy/Hard split: `n_objects <= 2`.
- View-Dep split: canonical 10-token literal set from ReferIt3D.
- Candidate pool: full-scene GT segmented / object proposals; the old
  "target-type-only public pool" assumption is retracted.
- Fairness boundary: v3 uses GT-target-visible keyframes; v5/v5.1 use
  query-driven keyframes but are invalidated because their object-frame
  visibility was projection-only. v6/v7/v8/v9 are depth-aware but partial and
  preserve v4 keyframe choices to isolate Stage2/runtime changes before NMS.

See [protocol.md](protocol.md) for the consolidated evidence and caveats.

## Reproduction

### Canonical pilot fold (use this for all small-batch validation)

**`tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json`** (600 samples,
stratified on `(is_easy × is_view_dep)`, salt-locked to within ±0.19 pp
of the FULL filtered 7805 on the v9.1_fix reference; bootstrap 90 % band
±2.3 pp on Overall). Use this fold for every iteration / A/B / smoke run
that doesn't need full-7805 statistical power.

```bash
# 1. (one-time) regenerate the fold from data/nr3d/ — byte-stable; MD5 = 12a69d8d14a81519024bbe00d6334434
PYTHONPATH=src python scripts/build_nr3d_strat600_fold.py

# 2. (one-time) prep the pack on the 600 ids (~10 min on Mac)
PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
    --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v9_catalog_first \
    --split test \
    --keyframe-mode query_driven --ensure-lightweight-cache

# 3. run the agent on the 600 ids
PYTHONPATH=src python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
    --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v9_catalog_first \
    --output-dir tmp/nr3d_eval_<run_id>/ \
    --workers 20

# 4. aggregate the canonical 5-column leaderboard metrics on the 600 ids
PYTHONPATH=src python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
    --side-by-side tmp/nr3d_eval_<run_id>/side_by_side.json \
    --nr3d-data-root data/nr3d \
    --phase8-data-root data/nr3d/scannet \
    --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
    --output docs/benchmark/nr3d/assets/<run_id>_strat600_leaderboard.json
```

Design + bootstrap validation: see
[v9_3_strat600_subset_design_20260517.md](v9_3_strat600_subset_design_20260517.md).

**Variance budget when comparing two strat600 runs**: Overall ±2.3 pp 90 %,
Easy / V-Indep ±2.7–2.9 pp, Hard ±3.5 pp, V-Dep ±4.5 pp (n=119). Any
delta inside its band should be confirmed on the FULL 7805 before being
claimed as a real change.

The older `v4_agent_guards_fair_views_random100_sample_ids.json` fold is
**deprecated for new comparisons** — preserved only for reproducing the
historical v4–v9.2 random100 runs. Per-tier ±5–10 pp noise on random100
was the source of the v9.1 → v9.2 misread before this fold existed.

### Other reproduction recipes

Download the raw NR3D annotation files:

```bash
bash scripts/download_nr3d.sh data/nr3d
```

Load test samples with Phase 8 GT-CG boxes:

```bash
PYTHONPATH=src python -c "
from benchmarks.nr3d_loader import Nr3dDataset
ds = Nr3dDataset.from_path(
    data_root='data/nr3d',
    split='test',
    bbox_source='phase8_gt_cg',
    phase8_data_root='data/nr3d/scannet',
    max_samples=200,
)
print(f'loaded={len(ds)} stats={ds.stats}')
"
```

Ingest a future side-by-side output:

```bash
PYTHONPATH=src python scripts/ingest_nr3d_run.py \
    --output-dir tmp/nr3d_eval_<run> \
    --run-id <run> \
    --branch feat/nr3d-vg-benchmark \
    --commit <short_sha> \
    --backend pack_v1 \
    --judge-model none \
    --notes "NR3D pack_v1 run"
```

Prepare a Phase 8 pack:

```bash
PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
    --sample-ids tmp/nr3d_artifacts/v1_smoke20_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v1 \
    --split test
```

For high-concurrency query-driven prep, build lightweight ConceptGraph object
caches once, then enable the ensure flag during prep. If a cache is still
missing, it is built under a cross-process lock and the sample continues:

```bash
PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
    --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
    --data-root data/nr3d/scannet \
    --build-lightweight-cache-only

PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
    --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v4_agent_guards_fair_views \
    --split test \
    --keyframe-mode query_driven \
    --ensure-lightweight-cache
```

Run the pack-v1 Stage 2 runner:

```bash
PYTHONPATH=src python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
    --sample-ids tmp/nr3d_artifacts/v1_smoke20_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v1 \
    --output-dir tmp/nr3d_eval_v1_smoke20 \
    --workers 1
```

Run the v5.1 reproduction path from the version doc:

- Failed rerun subset:
  `tmp/nr3d_artifacts/v5_failed241_sample_ids_20260513.json`
- Base full output:
  `tmp/nr3d_eval_v4_agent_guards_fair_views_full/`
- Failed rerun output:
  `tmp/nr3d_eval_v5_failed_rerun_20260513/`
- Merged output:
  `tmp/nr3d_eval_v5_failed_rerun_merged_20260513/`
- Detailed CLI and ingest commands:
  [v5p1_failed_rerun_full_20260513.md](v5p1_failed_rerun_full_20260513.md)

## SQLite

Canonical DB: `docs/benchmark/nr3d/runs.sqlite`

```sql
SELECT run_id, n,
       printf('%.4f', classification_acc_filtered) AS acc_filtered,
       n_filtered,
       printf('%.4f', mean_iou) AS mean_iou,
       printf('%.4f', acc25) AS acc25,
       printf('%.4f', acc50) AS acc50
FROM runs;
```

Current headline query:

```sql
SELECT run_id, n, n_filtered,
       printf('%.4f', classification_acc_filtered) AS overall,
       printf('%.4f', acc_easy) AS easy,
       printf('%.4f', acc_hard) AS hard,
       printf('%.4f', acc_view_dep) AS vdep,
       printf('%.4f', acc_view_indep) AS vind
FROM runs
WHERE run_id='v5p1_failed_rerun_full_20260513';
-- v5p1_failed_rerun_full_20260513|8584|7805|0.6848|0.7843|0.5918|0.5738|0.7453
```

## Retained / Deleted Docs

Retained:

- README, leaderboard, protocol, runs.sqlite
- one version doc per evaluated run: v1, v2, v3, v4, v5, v5.1

Deleted during consolidation:

- `protocol_audit_20260501.md`
- `paper_crosscheck_20260501.md`
- `pool_equivalence_log_20260501.md`

Their durable conclusions were merged into [protocol.md](protocol.md),
[leaderboard.md](leaderboard.md), and the current-result section above.
