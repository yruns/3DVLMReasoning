# ScanRefer VG Evaluation Results

ScanRefer (Chen et al. ECCV 2020) — natural-language ScanNet references,
detection-mode evaluation. Metric: axis-aligned 3D IoU; Acc@0.25 /
Acc@0.50 × { Unique, Multiple, Overall }. Judge: none (programmatic IoU).

## Current Results

Use these rows depending on the comparison protocol:

| Protocol | Current row | Acc@0.25 | Acc@0.50 | Scope | Caveat |
|---|---|---:|---:|---|---|
| Query-driven full val | [v3.20 consensus3](v3p20_consensus3_full_20260511.md) | **55.36** | **49.57** | `9508` val utterances | Three independent no-GT source trajectories plus proposal-id consensus; not a single-run result. |
| Query-driven development fold | [v3.19 consensus3](v3p19_consensus3_20260508.md) | **61.0** | **54.0** | frozen random100 | Post-hoc consensus over v3.10/v3.11/v3.13 on the same development fold. |
| Query-driven single run | [v3.11 anchor-exclusion](v3p11_anchor_exclusion_20260508.md) | **54.0** | **50.0** | frozen random100 | Best single agent trajectory on the frozen fold. |
| GT-view-oracle upper bound | [v2 aggregation-GT track](v2_aggregation_gt_track_20260503.md) | **69.92** | **62.79** | `9508` val utterances | Paper-comparable GT bbox source, but the initial keyframes use GT target visibility. |

The v3.20 full-val row is the result to quote for the current no-GT
query-driven pipeline, with the consensus caveat. It uses
`mask3d_query_driven` keyframes, no GT at inference, aggregation-GT scoring,
and three fresh source trajectories. The three single-run source metrics are
53.18-54.00 Acc@0.25 and 47.60-48.36 Acc@0.50; consensus lifts the average
single-run result by +1.80 / +1.61 pp. Full details and source split:
[v3p20_consensus3_full_20260511.md](v3p20_consensus3_full_20260511.md).

Current query-driven full-random100 development headline is v3.19 three-run
no-GT consensus on the frozen random100 fold: Acc@0.25 **61.0** / Acc@0.50
**54.0** after aggregation-GT scoring (`100/100` completed, no GT used by the
selector). It applies proposal-id plurality plus confidence tie-break over the
already harvested v3.10, v3.11, and v3.13 no-GT runs. This clears the immediate
random100 target but remains a development-fold consensus result, not a full-val
paper claim; see
[v3p19_consensus3_20260508.md](v3p19_consensus3_20260508.md). The best
single-run headline remains v3.11 at **54.0 / 50.0**; see
[v3p11_anchor_exclusion_20260508.md](v3p11_anchor_exclusion_20260508.md).
The v3.21 text-first policy gate is a negative single-run result at
**46.0 / 38.0** on the same random100 fold and is not promoted; see
[v3p21_textfirst_random100_20260514.md](v3p21_textfirst_random100_20260514.md).
The v3.12 vertical-relation guard and v3.13 late-override guard single-run gates
are negative results at 54.0 / 48.0 and are not promoted; see
[v3p12_vertical_relation_guard_20260508.md](v3p12_vertical_relation_guard_20260508.md)
and [v3p13_late_override_20260508.md](v3p13_late_override_20260508.md).
A one-sample v3.14 stale-left/right focus gate, a v3.15 cited-frame
inspected-visibility focus gate, a v3.16 category-supported TADG focus gate,
a v3.17 final-revision guard focus gate, and a v3.18 final-selection ledger
focus gate were also rejected and are
documented separately, but are not fold-level leaderboard results; see
[v3p14_stale_left_right_focus_20260508.md](v3p14_stale_left_right_focus_20260508.md),
[v3p15_cited_frame_guard_focus_20260508.md](v3p15_cited_frame_guard_focus_20260508.md),
[v3p16_category_supported_tadg_focus_20260508.md](v3p16_category_supported_tadg_focus_20260508.md),
[v3p17_final_revision_guard_focus_20260508.md](v3p17_final_revision_guard_focus_20260508.md),
and [v3p18_final_selection_ledger_focus_20260508.md](v3p18_final_selection_ledger_focus_20260508.md).
Canonical current goal/problem statement:
[scanrefer-date-goal.md](scanrefer-date-goal.md).
External-method comparison table: [leaderboard.md](leaderboard.md).

## Reading Map

- Start with this README for the version timeline and which row is quotable.
- Use [leaderboard.md](leaderboard.md) for external zero-shot and supervised
  comparisons.
- Use [scanrefer-date-goal.md](scanrefer-date-goal.md) for the consolidated
  problem statement, protocol boundaries, and current next-step diagnosis.
- Use the immutable per-version docs below for exact commands, raw artifacts,
  SQLite run ids, and caveats.

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
| v3.7_final_selection (MIXED) | 2026-05-08 | Acc@0.25 = 46.0 / Acc@0.50 = 42.0; `100/100` completed | same 100-utt random fold, cached 98+2 process | Adds no-match candidate guard, evidence-frame guard, ConceptGraph frame-view enrichment, marked-frame left/right geometry, and direct-response guard bypass handling. It eliminates no-prediction failures, but the two original failed hard cases remain IoU 0 in the cached random100 harvest and Acc@0.25 is -2pp vs v3.3. v3.3 stays as the headline. Full doc: [v3p7_final_selection_20260508.md](v3p7_final_selection_20260508.md). |
| v3.8_relation_ranking | 2026-05-08 | Acc@0.25 = 49.0 / Acc@0.50 = 43.0; `100/100` completed | same 100-utt random fold, fresh rerun plus two checkpoint resumes | Adds relation-aware `compare_proposals_spatial` support for `left_of` / `right_of` via co-viewed 2D marked-frame geometry and `near` / `next_to` via floor-plane distance, plus TADG aliases, tool-version bump, and a CLIP lazy-load lock found during the fresh gate. The cached 93+7 estimate was 51.0 / 45.0, but the fresh gate lands at 49.0 / 43.0: +1pp/+1pp vs v3.3, still far below Z3D. Full doc: [v3p8_relation_ranking_20260508.md](v3p8_relation_ranking_20260508.md). |
| **v3.9_anchor_guard** | 2026-05-08 | **Acc@0.25 = 53.0 / Acc@0.50 = 46.0**; `100/100` completed | same 100-utt random fold, checkpoint-resumed fresh gate | Tightens direct-final deferral, newly queued image injection, TADG lookback/anchor checks, EFG target filtering and spatial-anchor frame-citation guarding, pronoun/secondary-clue spatial prompt guidance, transient `invalid_prompt/-4321` retry classification, and a bounded selector cache after a 56/100 no-trace exit. Net +4pp/+3pp vs v3.8 fresh; still below Z3D. Full doc: [v3p9_anchor_guard_20260508.md](v3p9_anchor_guard_20260508.md). |
| **v3.10_terminal_latch** | 2026-05-08 | **Acc@0.25 = 54.0 / Acc@0.50 = 48.0**; `100/100` completed after one retry merge | same 100-utt random fold, fresh latched gate | Adds EFG proximity / anchor-relative wording fixes, TADG inverse-relation and candidate-coverage checks, and a `submit_final` terminal latch so same-turn accepted finals cannot be overwritten before the run loop sees them. Net +1pp/+2pp vs v3.9; still below Z3D. Full doc: [v3p10_terminal_latch_20260508.md](v3p10_terminal_latch_20260508.md). |
| **v3.11_anchor_exclusion** | 2026-05-08 | **Acc@0.25 = 54.0 / Acc@0.50 = 50.0**; `100/100` completed | same 100-utt random fold, fresh gate | TADG candidate coverage ignores the current anchor when same-category compare candidate_ids intentionally exclude it. The preceding focus13 ablation fixed scene0660 and rejected an EFG 2D-mark experiment; the fresh random100 gate ties v3.10 on Acc@0.25 and improves Acc@0.50 by +2pp, with mean IoU 0.4615. Still below Z3D. Full doc: [v3p11_anchor_exclusion_20260508.md](v3p11_anchor_exclusion_20260508.md); focus doc: [v3p11_anchor_exclusion_focus13_20260508.md](v3p11_anchor_exclusion_focus13_20260508.md). |
| v3.12_vertical_relation_guard (NEGATIVE) | 2026-05-08 | Acc@0.25 = 54.0 / Acc@0.50 = 48.0; `100/100` completed | same 100-utt random fold, fresh gate | Adds a TADG guard that blocks target `above` / `below` submissions until a matching vertical spatial comparison has been run. It ties v3.11 on Acc@0.25 but lowers Acc@0.50 by -2pp and mean IoU by -0.0056, so v3.11 remains the best single-run headline. Full doc: [v3p12_vertical_relation_guard_20260508.md](v3p12_vertical_relation_guard_20260508.md). |
| v3.13_late_override_guard (NEGATIVE) | 2026-05-08 | Acc@0.25 = 54.0 / Acc@0.50 = 48.0; `100/100` completed | same 100-utt random fold, fresh gate | Attempts to reject late left/right overrides after a relation-ranked proposal was already accepted. The new branch triggered 0 times in random100; the gate ties v3.11 on Acc@0.25 but lowers Acc@0.50 by -2pp and mean IoU by -0.0014. Full doc: [v3p13_late_override_20260508.md](v3p13_late_override_20260508.md). |
| v3.14_stale_left_right_focus (REJECTED) | 2026-05-08 | Acc@0.25 = 0.0 / Acc@0.50 = 0.0 on `n=1` | one-sample focus gate (`scene0552_00::14::0`) | Tested an EFG branch for stale zero-support left/right ranks. The fresh trace did not satisfy the narrow "submitted absent, alternative present" condition, still selected proposal 24 with IoU 0.0, and the candidate was reverted. Not a leaderboard row. Full doc: [v3p14_stale_left_right_focus_20260508.md](v3p14_stale_left_right_focus_20260508.md). |
| v3.15_cited_frame_guard_focus (REJECTED) | 2026-05-08 | focus1 100.0 / 100.0; focus10 20.0 / 20.0 | one-sample and 10-sample focus gates | Tested an EFG branch that cross-checks cited final frames against `inspect_proposal.frames_appeared`. It recovered `scene0203_00::14::3`, but the 10-case control landed at 20 / 20 and the candidate was reverted. Not a leaderboard row. Full doc: [v3p15_cited_frame_guard_focus_20260508.md](v3p15_cited_frame_guard_focus_20260508.md). |
| v3.16_category_supported_tadg_focus (REJECTED) | 2026-05-08 | Acc@0.25 = 50.0 / Acc@0.50 = 50.0 on `n=6` | six-sample focus gate | Tested a TADG branch that rejects `not_in_candidates` overrides when the submitted proposal never appeared in any non-empty category lookup but the relation-ranked proposal did. The new branch did not fire on the motivating fresh trace; the slice tied v3.11 on threshold accuracy but lowered mean IoU, and the candidate was reverted. Not a leaderboard row. Full doc: [v3p16_category_supported_tadg_focus_20260508.md](v3p16_category_supported_tadg_focus_20260508.md). |
| v3.17_final_revision_guard_focus (REJECTED) | 2026-05-08 | Acc@0.25 = 41.67 / Acc@0.50 = 25.00 on `n=12` | twelve-sample focus gate | Tested a runtime guard that blocks changing an already accepted final proposal without `tool_override_reason`. It fired on 6 samples but the agent resubmitted changed proposals and the slice trailed both v3.11 and v3.13, so the candidate was reverted. Not a leaderboard row. Full doc: [v3p17_final_revision_guard_focus_20260508.md](v3p17_final_revision_guard_focus_20260508.md). |
| v3.18_final_selection_ledger_focus (REJECTED) | 2026-05-08 | Acc@0.25 = 16.67 / Acc@0.50 = 16.67 on `n=12` | twelve-sample focus gate | Tested a pre-submit `record_final_selection` ledger. The agent called the ledger on all 12 samples, but the slice fell far below v3.11/v3.13/v3.17, so the candidate was reverted. Not a leaderboard row. Full doc: [v3p18_final_selection_ledger_focus_20260508.md](v3p18_final_selection_ledger_focus_20260508.md). |
| **v3.19_consensus3** | 2026-05-08 | **Acc@0.25 = 61.0 / Acc@0.50 = 54.0**; `100/100` completed | same 100-utt random fold, no new LLM inference | Deterministic no-GT plurality consensus over v3.10/v3.11/v3.13 `selected_object_id` votes, tie-broken by confidence. It beats the Z3D random100 reference on this development fold, with paired delta vs v3.11 of Acc@0.25 `9/2`, Acc@0.50 `6/2`, mean IoU `+0.0465`. Caveat: source set was selected after auditing this fold; v3.20 is the fresh full-val follow-up, and v3.19 remains the development-fold precursor. Full doc: [v3p19_consensus3_20260508.md](v3p19_consensus3_20260508.md). |
| **v3.20_consensus3_full** | 2026-05-11 | **Acc@0.25 = 55.36 / Acc@0.50 = 49.57** | 9508 utts (full canonical val) | Full-val no-GT consensus over three fresh source trajectories using `mask3d_query_driven` keyframes; aggregation-GT rescored before consensus. Source runs land at 53.18-54.00 / 47.60-48.36; consensus improves over the source mean by +1.80 / +1.61 pp. Full doc: [v3p20_consensus3_full_20260511.md](v3p20_consensus3_full_20260511.md). |
| v3.21_textfirst_random100 (NEGATIVE) | 2026-05-14 | Acc@0.25 = 46.0 / Acc@0.50 = 38.0; `100/100` completed | same 100-utt random fold, fresh single run | Applies the NR3D v11 structured/text-first policy to ScanRefer detector-pool VG. It regresses sharply vs v3.11 single-run (Acc@0.50 recoveries/regressions `1/13`) and is not promoted. Full doc: [v3p21_textfirst_random100_20260514.md](v3p21_textfirst_random100_20260514.md). |

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
- **v3.19 is currently a 61% / 54% no-GT consensus development-fold headline
  after aggregation-GT rescoring**, but it is not a single-run agent path and
  the consensus source set was selected on the same frozen random100 fold. The
  original 39/15 v3.1 quote used Phase8 GT-CG bboxes and is retained only as
  audit trail. The remaining driver is multi-distractor picking error in
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
- **v3.7 is a durability win, not a metric win.** It removes no-prediction
  failures on the cached random100 fold (`100/100` completed, `tool_calls`
  durable in SQLite), but Acc@0.25 is 46.0 vs v3.3's 48.0. Focused two-case
  runs showed the hard targets are reachable, but the cached random100 harvest
  did not reproduce those correct final selections.
- **v3.8 is a small fresh metric win, not a solution.** The cached 51.0 / 45.0
  estimate was optimistic; the fresh random100 gate lands at 49.0 / 43.0, with
  all 100 samples completed and no-prediction failures still eliminated. It is
  +1pp/+1pp vs v3.3, but still well below Z3D's 58.9 / 52.7.
- **v3.10 was superseded by v3.11, but remains an important control.** It
  landed at 54.0 / 48.0 on the frozen development fold, with all 100 samples
  completed after one no-GT retry merge and 2584 durable `tool_calls`.
- **v3.11 is the best single-run random100 query-driven headline.** It lands
  at 54.0 / 50.0 on the frozen development fold, with all 100 samples completed
  and 2474 durable `tool_calls`. It ties v3.10 on Acc@0.25 and improves
  Acc@0.50 by +2pp, but remains below Z3D and still has unresolved stochastic
  same-category / spatial-anchor failures.
- **v3.12 is a negative guard experiment.** It completes random100 and is
  ingested as `v3p12_vertical_relation_guard_random100_20260508`, but lands at
  54.0 / 48.0 with mean IoU 0.4559. It demonstrates that forcing a missing
  `above` / `below` compare does not solve the downstream proposal-ranking
  failure; v3.11 remains the best single-run headline.
- **v3.13 is also a negative guard experiment.** It completes random100 and is
  ingested as `v3p13_late_left_right_override_random100_20260508`, but lands at
  54.0 / 48.0 with mean IoU 0.4601. The newly added late-left/right
  "already accepted" branch triggered 0 times, so the patch is not promoted and
  v3.11 remains the best single-run headline.
- **v3.14 is a rejected focus gate, not a fold-level result.** It is ingested
  as `v3p14_stale_left_right_focus1_20260508` because every benchmark attempt
  needs a durable process record, but it only covers one sample and is excluded
  from leaderboard claims. The attempted EFG stale-left/right branch did not
  fire on the fresh trace and was reverted.
- **v3.15 is another rejected focus gate, not a fold-level result.** It is
  ingested as `v3p15_cited_frame_guard_focus1_20260508` and
  `v3p15_cited_frame_guard_focus10_20260508`. The inspected-visibility
  cited-frame branch recovered the motivating pillow sample, but the 10-case
  control landed at 20 / 20 and the candidate was reverted.
- **v3.16 is a rejected focus gate, not a fold-level result.** It is ingested
  as `v3p16_category_supported_tadg_focus6_20260508`. The category-supported
  `not_in_candidates` TADG branch did not fire on the motivating fresh trace;
  focus6 tied v3.11 at 50 / 50 but lowered mean IoU, so the candidate was
  reverted without random100.
- **v3.17 is a rejected focus gate, not a fold-level result.** It is ingested
  as `v3p17_final_revision_guard_focus12_20260508`. The final-revision guard
  fired on 6 of 12 focus samples, but the final slice landed at 41.67 / 25.00,
  below v3.11 and v3.13 on the same sample ids, so the candidate was reverted
  without random100.
- **v3.18 is a rejected focus gate, not a fold-level result.** It is ingested
  as `v3p18_final_selection_ledger_focus12_20260508`. The final-selection
  ledger was used on all 12 focus samples, but the slice landed at 16.67 /
  16.67, so the candidate was reverted without random100.
- **v3.19 is a consensus result, not a fresh single-run architecture claim.**
  It is ingested as `v3p19_consensus3_random100_20260508` and reaches 61.0 /
  54.0 by voting over v3.10/v3.11/v3.13 no-GT predictions. The selector does
  not read GT or IoU, but the source-set choice was informed by existing
  development-fold artifacts; validate on a held-out fold or full val before
  treating it as a paper result.
- **v3.20 is the full-val query-driven headline, but still consensus3.** It is
  ingested as `v3p20_consensus3_full_20260511` and reaches 55.36 / 49.57 on all
  9508 val utterances. The three single source trajectories score 54.00 / 48.36,
  53.52 / 47.93, and 53.18 / 47.60, so any cost-normalized comparison should
  cite both the source metrics and the consensus lift.
- **v3.21 text-first is a negative ScanRefer transfer.** It is ingested as
  `v3p21_textfirst_random100_20260514` and reaches only 46.00 / 38.00 on the
  frozen random100 fold. The same structured-first policy that modestly helped
  NR3D random100 appears harmful for ScanRefer's noisy Mask3D detector pool,
  where labels and metadata are weak priors and visual confirmation remains
  critical.
- **SQLite size control.** To keep `runs.sqlite` pushable as a normal git file,
  very long `tool_calls.response_text` values for intermediate v3.7-v3.18 and
  v3.21 development runs are compacted after ingest. Run rows, sample rows,
  tool-call row counts, tool names, and tool inputs remain in SQLite; full raw
  JSON/log artifact paths are recorded in each per-version doc. v3.19/v3.20
  consensus traces are compact by construction.
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

- [scanrefer-date-goal.md](scanrefer-date-goal.md) — current canonical
  goal/problem statement; supersedes the old final investigation, keyframe
  funnel, picking-error process notes, and the two old ScanRefer handoff docs
  that were removed after their valid conclusions were consolidated.
