# ScanRefer Public Leaderboard (reference)

Source: https://kaldir.vc.in.tum.de/scanrefer_benchmark/benchmark_localization
(test-server) and SeeGround Table 1 (val numbers from supervised SOTA).

## Test-server top-5 (official, Acc@0.5IoU Overall)

| Rank | Method | Unique@0.25 | Unique@0.50 | Multiple@0.25 | Multiple@0.50 | Overall@0.25 | Overall@0.50 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | UniVLG | 88.95 | 82.36 | 59.21 | 50.30 | 65.88 | 57.49 |
| 2 | Chat-Scene | 88.87 | 80.05 | 54.21 | 48.61 | 61.98 | 55.66 |
| 3 | ConcreteNet | 86.07 | 79.23 | 47.46 | 40.91 | 56.12 | 49.50 |
| 4 | cus3d | 83.84 | 70.73 | 49.08 | 40.00 | 56.88 | 46.89 |
| 5 | D-LISA | 81.95 | 69.00 | 49.75 | 39.67 | 56.97 | 46.25 |

## Validation top-5 (supervised, SeeGround Table 1)

| Rank | Method | Venue | Unique@0.25 | Unique@0.50 | Multiple@0.25 | Multiple@0.50 | Overall@0.25 | Overall@0.50 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | ConcreteNet | ECCV 2024 | 86.4 | 82.1 | 42.4 | 38.4 | 50.6 | 46.5 |
| 2 | 3D-VisTA | ICCV 2023 | 81.6 | 75.1 | 43.7 | 39.1 | 50.6 | 45.8 |
| 3 | MCLN | ECCV 2024 | 86.9 | 72.7 | 52.0 | 40.8 | 57.2 | 45.7 |
| 4 | G3-LQ | CVPR 2024 | 88.6 | 73.3 | 50.2 | 39.7 | 56.0 | 44.7 |
| 5 | EDA | CVPR 2023 | 85.8 | 68.6 | 49.1 | 37.6 | 54.6 | 42.3 |

## Zero-shot (Mask3D-pool) reference

| Method | Year | Unique@0.25 | Unique@0.50 | Multiple@0.25 | Multiple@0.50 | Overall@0.25 | Overall@0.50 |
|---|---|---:|---:|---:|---:|---:|---:|
| LLM-Grounder (998 sub-sample) | ICRA 2024 | - | - | - | - | 17.1 | 5.3 |
| ZSVG3D / GPT-4-Turbo | CVPR 2024 | 63.8 | 58.4 | 27.7 | 24.6 | 36.4 | 32.7 |
| CSVG (Mask3D) | BMVC 2025 | 68.8 | 61.2 | 38.4 | 27.3 | 49.6 | 39.8 |
| SeeGround / Qwen2-VL-72B | CVPR 2025 | 75.7 | 68.9 | 34.0 | 30.0 | 44.1 | 39.4 |
| VLM-Grounder (250 sub-sample) | CoRL 2024 | 66.0 | 29.8 | 48.3 | 33.5 | 51.6 | 32.8 |
| Z3D (Mask3D row) | 2026 arXiv | 82.3 | 74.8 | 51.5 | 45.7 | 58.9 | 52.7 |
| **Ours v2 (gpt-5.4) [‡]** | this work | **83.11** | **76.49** | **65.00** | **57.68** | **69.92** | **62.79** |
| Ours v1 (gpt-5.4) [†][‡] | this work | 62.97 | 25.41 | 47.43 | 12.79 | 51.65 | 16.22 |
| Ours v3 corrected (gpt-5.4) [§] | this work | 71.43 | 71.43 | 37.50 | 27.78 | 47.00 | 40.00 |
| Ours v3.1 corrected (gpt-5.4) [§] | this work | 67.86 | 67.86 | 36.11 | 27.78 | 45.00 | 39.00 |
| Ours v3.2 callback-durable (gpt-5.4) [§] | this work | 71.43 | 71.43 | 36.11 | 27.78 | 46.00 | 40.00 |
| Ours v3.3 vertical-spatial (gpt-5.4) [§] | this work | 75.00 | 75.00 | 37.50 | 29.17 | 48.00 | 42.00 |
| Ours v3.4 select-among-proposals (gpt-5.4) [§][¶] | this work | 78.57 | 78.57 | 30.56 | 22.22 | 44.00 | 38.00 |
| Ours v3.5 CVRA (gpt-5.4) [§][♦] | this work | 75.00 | 75.00 | 37.50 | 29.17 | 48.00 | 42.00 |
| Ours v3.7 final-selection guards (gpt-5.4) [§][♠] | this work | 75.00 | 75.00 | 34.72 | 29.17 | 46.00 | 42.00 |
| Ours v3.8 relation-ranking fresh (gpt-5.4) [§][♣] | this work | 75.00 | 75.00 | 38.89 | 30.56 | 49.00 | 43.00 |
| Ours v3.9 anchor/evidence guards (gpt-5.4) [§][#] | this work | 78.57 | 78.57 | 43.06 | 33.33 | 53.00 | 46.00 |
| Ours v3.10 terminal latch (gpt-5.4) [§][※] | this work | 82.14 | 82.14 | 43.06 | 34.72 | 54.00 | 48.00 |
| Ours v3.11 anchor-exclusion (gpt-5.4) [§][¤] | this work | 82.14 | 82.14 | 43.06 | 37.50 | 54.00 | 50.00 |
| Ours v3.12 vertical-relation guard (gpt-5.4) [§][!] | this work | 71.43 | 71.43 | 47.22 | 38.89 | 54.00 | 48.00 |
| Ours v3.13 late-override guard (gpt-5.4) [§][!!] | this work | 75.00 | 75.00 | 45.83 | 37.50 | 54.00 | 48.00 |
| **Ours v3.19 consensus3 (gpt-5.4) [§][+]** | this work | **85.71** | **85.71** | **51.39** | **41.67** | **61.00** | **54.00** |
| Ours v3.20 full-val consensus3 (gpt-5.4) [◇] | this work | 78.58 | 72.31 | 46.71 | 41.09 | 55.36 | 49.57 |

**v2's GT bbox source is paper-comparable** — same v1 agent decisions
re-aggregated against ScanNet aggregation-derived GT (the bbox source
that Mask3D was trained against and that all other rows in this table
use). However, the keyframe selector uses a GT view oracle (see [‡]
below), so v2's wins over zero-shot Camp-A baselines are **not**
directly comparable. v3 (Phase 8 hypothesis-parser KFs), v3.1
(Mask3D-CG visibility KFs), v3.2 (runtime/evaluator durability), and
v3.3 (vertical `above`/`below` spatial relations) are all zero-shot
Camp-A on the 100-utt random fold. v3 originally quoted 39/14
under the wrong Phase8-GT bbox evaluator; the row above is the
v3 KF source re-run on the v3.2 runtime fixes and rescored against
ScanNet aggregation GT. The X1 (Mask3D-CG visibility) hypothesis is
falsified under like-for-like conditions: v3 and v3.1 land within
2pp on Overall@0.25 and tie on Overall@0.50 (47/40 vs 45/39). Docs:
[`v3p1_mask3d_query_driven_20260503.md`](v3p1_mask3d_query_driven_20260503.md),
[`v3p2_callbacks_durable_20260504.md`](v3p2_callbacks_durable_20260504.md),
[`v3p3_vertical_spatial_20260504.md`](v3p3_vertical_spatial_20260504.md).
v3.19 is the current query-driven random100 development-fold headline. It uses
no-GT plurality consensus over v3.10/v3.11/v3.13 predictions and reaches
61.00 / 54.00 on the frozen 100-utt fold. This clears the immediate random100
target but should be read as the development-fold precursor to the fresh
full-val v3.20 run, not as a paper row by itself. v3.11 remains the best
single-run random100 headline at 54.00 / 50.00. v3.12 and v3.13 are included as
negative single-run audit results and are not promoted. v3.14, v3.15, v3.16,
v3.17, and v3.18 are rejected focus gates, so they are documented for audit but
intentionally omitted from the leaderboard table. v3.20 is the current full-val
query-driven headline, but it is also a three-run consensus result; the
individual source trajectories are 54.00 / 48.36, 53.52 / 47.93, and
53.18 / 47.60.

[†] **v1 GT bbox not paper-comparable.** v1 uses Phase 8 GT-CG bbox
(~2× larger by volume than the ScanNet aggregation-based GT bbox the
other rows use). Oracle-picker ceiling on v1 is Acc@0.50 = 20.53 %;
v1 is preserved as audit trail in `runs.sqlite`. Full diagnosis in
[v2 doc § Audit trail](v2_aggregation_gt_track_20260503.md#audit-trail--why-v2-exists).

[‡] **GT view oracle in keyframe selection.** Both v1 and v2 select
the 5 initial RGB keyframes via Phase 8 visibility of the GT
`target_id` (`select_keyframes_from_phase8_target`), not via
query-driven Stage 1 retrieval. This guarantees the target object
appears in the initial RGB evidence and is a form of GT-assisted
evidence selection (a view oracle, not full label leakage — `proposal_id`
is still picked from the Mask3D pool). Z3D / SeeGround / ZSVG3D / CSVG
do not use a GT view oracle, so per-column wins above are not
apples-to-apples as a zero-shot Camp-A comparison. v3 / v3.1 swap
this oracle for query-driven Stage 1 (`select_keyframes_v2(query)`
and `mask3d_query_driven` respectively). Detailed discussion in
[v2 doc § Caveats](v2_aggregation_gt_track_20260503.md#caveats).

[¶] **v3.4 is a NEGATIVE result, not the new headline.** The forced
1-of-K VLM choice tool (`select_among_proposals`) was adopted by the
agent on 81/100 samples (98 total invocations) but moved Multiple@0.50
DOWN by 7pp (29.17 → 22.22) and Overall@0.50 DOWN by 4pp (42 → 38).
The Unique split improved (+3.57pp) but the multi-distractor bottleneck
the tool was designed to fix actually got worse — most likely because
the judge sees the most-visible frame per candidate without the
spatial anchor co-visible. v3.3 remains the development-fold
headline. v3.4 stays in the table as audit trail and to motivate
the next iteration (co-visible-anchor frame selection). See
[`v3p4_select_among_proposals_20260504.md`](v3p4_select_among_proposals_20260504.md).

[♦] **v3.5 CVRA is a NEGATIVE result, smoke only — quoted numbers are
v3.3's, not a v3.5 full-fold run.** CVRA (CLIP-Visible Retrieval
Augmentation) extends `find_proposals_by_category` with a CLIP-text
rerank over Mask3D proposals visible in the agent's cumulative seen
frames; the augmented set is appended (not replacing) label-based
retrieval, with K_AUG=8/10 and a label-mismatch overflow tier. The
6-utt addressable smoke (`tmp/scanrefer_artifacts/cvra_addressable6_sample_ids.json`,
selected from the bbox-IoU v2 audit) gave **6/6 retrieval recall but
0/6 F5a flips** — 5 of 6 samples produce identical agent picks vs the
v3.3 baseline, so the projected random100 lift is ~0pp. The full
random100 run was deliberately skipped to avoid spending 3-4 h
producing a 0pp number. CVRA infrastructure (proposal_pool bridge fix,
`BatchedClipProvider`, pack-prep `frame_views`, runner
`--use-clip-visible-aug`) ships behind `use_clip_visible_aug=False`
default; the row above is reproduced from v3.3 for table continuity,
not a fresh run. The bottleneck identified by the smoke is
agent-side spatial reasoning correctly rejecting audit-tagged
candidates whose 3D position contradicts the description's spatial
referent (e.g. `scene0011_00::20::2`: CVRA-aug pid is at refrigerator
level, query says "above the refrigerator"). The bbox-IoU v2
"addressability" definition therefore over-counts retrieval-side
reachable surface; future audits need a spatial-referent verifier.
See [`v3p5_cvra_negative_20260505.md`](v3p5_cvra_negative_20260505.md).

[♠] **v3.7 final-selection guards are MIXED, not the new headline.** v3.7
adds no-match candidate guarding, evidence-frame guarding, ConceptGraph
frame-view enrichment, and marked-frame left/right geometry. It eliminates
no-prediction failures on the cached random100 process (`100/100` completed),
but Acc@0.25 is 46.0 vs v3.3's 48.0. The two original failed hard cases
complete but remain IoU 0 after aggregation-GT rescoring. See
[`v3p7_final_selection_20260508.md`](v3p7_final_selection_20260508.md).

[♣] **v3.8 relation-ranking was superseded by v3.9.** It adds
`left_of` / `right_of` ranking from co-viewed 2D marked-frame geometry, `near`
/ `next_to` ranking from floor-plane distance, TADG aliases, a tool-version
bump, and a CLIP lazy-load lock found during the fresh gate. A cached 93+7
estimate reached 51.00 / 45.00, but the fresh rerun supersedes it at
49.00 / 43.00. Multiple improves to 38.89 / 30.56 vs v3.7's 34.72 / 29.17,
but the result remains far below Z3D's 58.9 / 52.7. See
[`v3p8_relation_ranking_20260508.md`](v3p8_relation_ranking_20260508.md).

[#] **v3.9 anchor/evidence guards are superseded by v3.10.** v3.9 tightens direct-final deferral, newly queued image injection,
TADG lookback and anchor checks, EFG target filtering and spatial-anchor
frame-citation guarding, VG spatial prompt guidance, transient
`invalid_prompt/-4321` retry classification, and bounded selector caching during
long resumes. It reaches 53.00 / 46.00 on the frozen random100 fold, +4pp/+3pp
over the v3.8 fresh gate, with `100/100` completed and no GT at inference. The
result still trails Z3D's 58.9 / 52.7 and remains a development-fold result.
See [`v3p9_anchor_guard_20260508.md`](v3p9_anchor_guard_20260508.md).

[※] **v3.10 terminal latch is superseded by v3.11.**
v3.10 adds EFG proximity / anchor-relative wording fixes, TADG inverse-relation
and candidate-coverage checks, and a `submit_final` terminal latch that prevents
same-turn accepted finals from being overwritten before the run loop observes
the terminal signal. It reaches 54.00 / 48.00 on the frozen random100 fold,
+1pp/+2pp over v3.9, with all 100 samples completed after one no-GT retry merge.
It still trails Z3D's 58.9 / 52.7 and remains a development-fold result. See
[`v3p10_terminal_latch_20260508.md`](v3p10_terminal_latch_20260508.md).

[¤] **v3.11 anchor-exclusion is the best single-run query-driven random100
headline.**
TADG candidate coverage now ignores the current spatial anchor when
same-category compare candidate_ids intentionally exclude it. It reaches
54.00 / 50.00 on the frozen random100 fold, tying v3.10 on Acc@0.25 and
improving Acc@0.50 by +2pp. The preceding focus13 ablation and rejected EFG
2D-mark experiment are recorded in
[`v3p11_anchor_exclusion_focus13_20260508.md`](v3p11_anchor_exclusion_focus13_20260508.md).
The full random100 doc is
[`v3p11_anchor_exclusion_20260508.md`](v3p11_anchor_exclusion_20260508.md).

[!] **v3.12 vertical-relation guard is NEGATIVE.** v3.12 adds a TADG guard that
requires target `above` / `below` descriptions to run a matching vertical
`compare_proposals_spatial` before final submission. The full random100 gate
lands at 54.00 / 48.00, tying v3.11 on Acc@0.25 but lowering Acc@0.50 by -2pp
and mean IoU by -0.0056. It is retained as process evidence, not a new
headline. See
[`v3p12_vertical_relation_guard_20260508.md`](v3p12_vertical_relation_guard_20260508.md).

[!!] **v3.13 late-override guard is NEGATIVE.** v3.13 attempts to reject late
left/right overrides after a relation-ranked proposal was already accepted by
`submit_final`. The new branch triggered 0 times in the full random100 gate,
and the result lands at 54.00 / 48.00 with mean IoU 0.4601, below v3.11 on
Acc@0.50 and mean IoU. It is retained as process evidence, not a new headline.
See [`v3p13_late_override_20260508.md`](v3p13_late_override_20260508.md).

[+] **v3.19 consensus3 is a no-GT development-fold consensus result.** It
votes over the already harvested v3.10, v3.11, and v3.13 random100 predictions
by `selected_object_id` plurality and confidence tie-break. The selector does
not read GT/IoU, but the source set was chosen after auditing the same frozen
random100 fold. It is the current random100 headline, not yet a full-val paper
claim. See [`v3p19_consensus3_20260508.md`](v3p19_consensus3_20260508.md).

v3.14 stale-left/right was a rejected one-sample focus gate on
`scene0552_00::14::0`. It is ingested as
`v3p14_stale_left_right_focus1_20260508`, but excluded from the table because
`n=1` is not a random100 or full-val leaderboard result. See
[`v3p14_stale_left_right_focus_20260508.md`](v3p14_stale_left_right_focus_20260508.md).

v3.15 cited-frame inspected-visibility was a rejected focus gate. It recovered
`scene0203_00::14::3` in focus1, but the 10-case control landed at 20.00 /
20.00 and tied v3.11 on that slice while trailing the existing v3.13 trace. It
is ingested as `v3p15_cited_frame_guard_focus1_20260508` and
`v3p15_cited_frame_guard_focus10_20260508`, but excluded from the table because
it is not a random100 or full-val leaderboard result. See
[`v3p15_cited_frame_guard_focus_20260508.md`](v3p15_cited_frame_guard_focus_20260508.md).

v3.16 category-supported TADG was a rejected focus gate. It tested a
`not_in_candidates` override guard based on the agent's own category lookup
trace, but the new branch did not fire on the motivating fresh trace. The
focus6 slice tied v3.11 on threshold accuracy at 50.00 / 50.00 while lowering
mean IoU, so no random100 gate was run. It is ingested as
`v3p16_category_supported_tadg_focus6_20260508`, but excluded from the table
because it is not a random100 or full-val leaderboard result. See
[`v3p16_category_supported_tadg_focus_20260508.md`](v3p16_category_supported_tadg_focus_20260508.md).

v3.17 final-revision guard was a rejected focus gate. It tested a runtime guard
that soft-blocks changing an already accepted final proposal unless the agent
provides `tool_override_reason`. The branch fired on 6/12 focus samples, but
the slice landed at 41.67 / 25.00, below both v3.11 and v3.13 on the same
sample ids, so no random100 gate was run. It is ingested as
`v3p17_final_revision_guard_focus12_20260508`, but excluded from the table
because it is not a random100 or full-val leaderboard result. See
[`v3p17_final_revision_guard_focus_20260508.md`](v3p17_final_revision_guard_focus_20260508.md).

v3.18 final-selection ledger was a rejected focus gate. It tested a pre-submit
`record_final_selection` contract binding the winner, candidate set, evidence
frames, and reject reasons. The agent called the ledger on all 12 samples, but
the slice landed at 16.67 / 16.67, far below v3.11/v3.13/v3.17, so no
random100 gate was run. It is ingested as
`v3p18_final_selection_ledger_focus12_20260508`, but excluded from the table
because it is not a random100 or full-val leaderboard result. See
[`v3p18_final_selection_ledger_focus_20260508.md`](v3p18_final_selection_ledger_focus_20260508.md).

[§] **100-utt random fold, not full 9508 val.** v3 / v3.1 / v3.2 /
v3.3 / v3.4 / v3.5 / v3.7 / v3.8 / v3.9 / v3.10 / v3.11 / v3.12 / v3.13 numbers above are reported on a
frozen 100-utt random sub-fold
(seed=20260503, file `tmp/scanrefer_artifacts/random100_sample_ids.json`)
per the project's iteration-on-100-first methodology. v2 on the same
fold is 67.0 / 59.0 (vs 69.92 / 62.79 on full val), so fold
representativeness is consistent within ~3pp. The original v3 39/14
and v3.1 39/15 quotes were Phase8-GT smoke numbers; under the
corrected aggregation-GT evaluator and v3.2 runtime fixes, v3
(Phase 8 hypothesis-parser KFs) is 47/40 and v3.1 (Mask3D-CG visibility
KFs, the X1 patch) is 45/39 — i.e. the X1 hypothesis is falsified
under like-for-like conditions. v3.2 (46/40) layered runtime
correctness on top, and v3.3 (48/42) adds vertical relations. The first
full-val v3.x run is v3.20 consensus3. v3.7 (46/42) is a cached final-selection
durability harvest, v3.8 (49/43) is the fresh relation-ranking gate after the
earlier cached 51/45 estimate proved optimistic, v3.9 (53/46) is the
anchor/evidence-guard gate, v3.10 (54/48) is the terminal-latch /
guard-cleanup gate, v3.11 (54/50) is the best single-run TADG
anchor-exclusion gate, v3.12 (54/48) is a negative vertical-relation guard
gate, v3.13 (54/48) is a negative late-override guard gate, v3.14-v3.18 are
rejected focus gates excluded from the table, and v3.19 (61/54) is the current
no-GT consensus development-fold headline.
See
[`v3p3_vertical_spatial_20260504.md`](v3p3_vertical_spatial_20260504.md)
and the v3-rerun row `v3_aggcorrected_random100_20260504` in
`runs.sqlite`.

[◇] **v3.20 full-val consensus3.** Full canonical ScanRefer val run (`n=9508`)
using `mask3d_query_driven` keyframes and no-GT three-source consensus after
aggregation-GT rescoring. It is a full-val result, but still a consensus result
rather than a single-run architecture. Source A/B/C are 54.00 / 48.36,
53.52 / 47.93, and 53.18 / 47.60; the single-run average is 53.57 / 47.96, so
consensus adds +1.80 / +1.61 pp over the source mean. Consensus group sizes are
5293 unanimous, 3068 majority, and 1147 all-different fallback selections. See
[`v3p20_consensus3_full_20260511.md`](v3p20_consensus3_full_20260511.md).
