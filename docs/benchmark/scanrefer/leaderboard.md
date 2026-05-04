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

[§] **100-utt random fold, not full 9508 val.** v3 / v3.1 / v3.2 / v3.3
numbers above are reported on a frozen 100-utt random sub-fold
(seed=20260503, file `tmp/scanrefer_artifacts/random100_sample_ids.json`)
per the project's iteration-on-100-first methodology. v2 on the same
fold is 67.0 / 59.0 (vs 69.92 / 62.79 on full val), so fold
representativeness is consistent within ~3pp. The original v3 39/14
and v3.1 39/15 quotes were Phase8-GT smoke numbers; under the
corrected aggregation-GT evaluator and v3.2 runtime fixes, v3
(Phase 8 hypothesis-parser KFs) is 47/40 and v3.1 (Mask3D-CG visibility
KFs, the X1 patch) is 45/39 — i.e. the X1 hypothesis is falsified
under like-for-like conditions. v3.2 (46/40) layered runtime
correctness on top, and v3.3 (48/42) adds vertical relations. Full-val
v3.x runs are deferred until the multi-distractor ranking gap moves
on the development fold. See
[`v3p3_vertical_spatial_20260504.md`](v3p3_vertical_spatial_20260504.md)
and the v3-rerun row `v3_aggcorrected_random100_20260504` in
`runs.sqlite`.
