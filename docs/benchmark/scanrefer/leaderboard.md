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
| Ours v3 (gpt-5.4) [§] | this work | — | — | — | — | 39.0 | 14.0 |
| Ours v3.1 corrected (gpt-5.4) [§] | this work | 67.86 | 67.86 | 36.11 | 27.78 | 45.00 | 39.00 |
| Ours v3.2 callback-durable (gpt-5.4) [§] | this work | 71.43 | 71.43 | 36.11 | 27.78 | 46.00 | 40.00 |
| Ours v3.3 vertical-spatial (gpt-5.4) [§] | this work | 75.00 | 75.00 | 37.50 | 29.17 | 48.00 | 42.00 |

**v2's GT bbox source is paper-comparable** — same v1 agent decisions
re-aggregated against ScanNet aggregation-derived GT (the bbox source
that Mask3D was trained against and that all other rows in this table
use). However, the keyframe selector uses a GT view oracle (see [‡]
below), so v2's wins over zero-shot Camp-A baselines are **not**
directly comparable. v3 (Phase 8 hypothesis-parser KFs) and v3.1
(Mask3D-CG visibility KFs) are zero-shot Camp-A but currently
evaluated only on a 100-utt random fold. v3.1's originally quoted
39/15 number used the old Phase8 GT-CG bbox evaluator; the corrected
row above uses ScanNet aggregation GT. v3.2 adds callback-image
durability and tool-call persistence. v3.3 adds vertical `above` /
`below` spatial relations. Docs:
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

[§] **100-utt random fold, not full 9508 val.** v3 / v3.1 / v3.2 numbers
above are reported on a frozen 100-utt random sub-fold (seed=20260503,
file `tmp/scanrefer_artifacts/random100_sample_ids.json`) per the
project's iteration-on-100-first methodology. v2 on the same fold is
67.0 / 59.0 (vs 69.92 / 62.79 on full val), so fold representativeness
is consistent within ~3pp. v3's 39/14 and v3.1's original 39/15 were
Phase8-GT smoke numbers; aggregation-GT corrected v3.1 is 45/39,
v3.2 is 46/40, and v3.3 is 48/42. Full-val v3.x runs are deferred until the multi-distractor
ranking gap moves on the development fold. See
[`v3p3_vertical_spatial_20260504.md`](v3p3_vertical_spatial_20260504.md).
