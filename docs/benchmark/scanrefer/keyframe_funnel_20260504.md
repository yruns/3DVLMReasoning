# ScanRefer Keyframe Pipeline Funnel — 2026-05-04

> Calibration audit per request: stop guessing fixes; first quantify
> per-stage correctness on the random100 fold so we know **which stage
> is actually leaking samples** before proposing the next change.

## What this measures

For each sample in the frozen `random100_sample_ids.json` fold, the
script `scripts/evaluate_keyframe_funnel.py` walks five gated checks
end-to-end against on-disk artifacts only (no LLM calls):

```
100 samples
└── F1: Phase 8 visibility index has the GT target_id
    └── F2: pack-prep initial KFs cover at least one Phase-8-visible
            frame for the GT target
        └── F3: initial KFs carry at least one Mask3D candidate whose
                label fuzzy-matches the GT category
            └── F4: cumulative KFs (initial + view_keyframe_marked +
                    callback-added) cover Phase 8 GT visibility
                └── F4b: cumulative KFs carry a Mask3D same-category
                         candidate
                    └── F5a: agent submitted a proposal with IoU >= 0.25
                        └── F5b: agent IoU >= 0.50
```

`(ref) mask3d_pool_has_category_match` is the ceiling — if the Mask3D
pool has no proposal of the GT category, no picker can succeed.

## Results on random100 (Phase 8 GT bbox eval — same as the per-sample IoU)

| Stage | v3 (Phase 8 KFs) | v3.1+ (Mask3D KFs) v3.3 | v3.4 (D5 reverted) |
|---|---:|---:|---:|
| F1: Phase 8 has visible frames for target_id | 100 | 100 | 100 |
| F2: initial KFs cover Phase 8 target | **67** | 65 | 65 |
| F3: initial KFs carry same-category Mask3D | 86 | 83 | 83 |
| F4: cumulative KFs cover Phase 8 target | **89** | 86 | 84 |
| F4b: cumulative same-category Mask3D | 93 | 94 | 94 |
| F5a: agent IoU >= 0.25 | 39 | 42 | 39 |
| F5b: agent IoU >= 0.50 | 14 | 14 | 16 |
| (ref) Mask3D pool ceiling | 94 | 94 | 94 |

(All numbers from `scripts/evaluate_keyframe_funnel.py` against
`tmp/v3{,p3,p4}*_eval/per_sample/<pack>/*.json`. Reports persisted at
`tmp/v3_funnel_report.json`, `tmp/v3p3_funnel_report.json`, and
`tmp/v3p4_funnel_report.json`.)

## What we learn

### 1. The X1 motivation (Mask3D-CG visibility KFs help) is empirically false at calibration.

v3 (Phase 8 hypothesis-parser KFs) and v3.1+ (Mask3D-CG candidate
visibility KFs) land within 2pp on every keyframe-coverage metric,
and v3 is in fact slightly **better** on F2 (67 vs 65) and F4
(89 vs 86). The X1 patch was supposed to guarantee same-category
Mask3D coverage in initial KFs, but F3 actually drops from 86 to 83
under Mask3D-driven selection. The Mask3D-CG visibility ranking
optimizes for "frame has many same-category candidates" rather than
"frame has the right same-category candidate," and the GT target
sometimes ends up in a frame without other same-category clutter
that the Mask3D scorer prefers.

### 2. The Multi-distractor picking gap is real and is the dominant loss bucket.

Out of v3.3's 100 samples (best healthy run):

| Bucket | Count | % |
|---|---:|---:|
| F1 fail (no Phase 8 target visibility anywhere) | 0 | 0 |
| F2 fail, F4 fail (callbacks didn't recover) | 14 | 14 |
| F2 fail, F4 recovered (callbacks helped) | 21 | 21 |
| **F4 yes, picker wrong (F5a fail)** | **44** | **44** |
| F4 yes, picker right (F5a pass) | 42 | 42 |
| Mask3D pool has no same-category candidate | 6 | 6 |

86% of samples have the GT target visible in **some** frame the
agent saw. The agent picks correctly on only 42% of those — i.e.
**when the right answer is on screen, the agent gets it wrong
51% of the time**. That is the open question.

### 3. Stage 2 callbacks are doing useful work, but more than half of F2 failures are unrecoverable.

Of the 35 samples where initial KFs missed the GT target:
- 21 (60%) recovered through Stage 2 callbacks (the iterative
  Stage 2 -> Stage 1 loop is doing what it was designed for).
- 14 (40%) never recovered — the agent's `view_keyframe_marked`,
  `request_more_views`, `request_crops`, and
  `switch_or_expand_hypothesis` calls did not surface the right
  frame.

This means callback-side improvements (e.g., better `request_more_views`
ranking, smarter `switch_or_expand_hypothesis` reformulation) can
recover at most 14 samples — a 14pp ceiling on F4 improvements.
**Picking-error fixes have a 44pp ceiling**, three times larger.

### 4. D5 (`select_among_proposals`) made the funnel slightly worse, not better.

Under the same pack, v3.4's F4 dropped from 86 to 84 (the model
trusted the new tool's verdict and stopped exploring as much), and
F5a dropped from 42 to 39. F5b actually nudged up by 2 (14 -> 16),
but the agg-GT-rescored headline of 38/22.22 (Multiple@0.50) is
strictly worse than v3.3's 42/29.17. The forced 1-of-K choice
without anchor co-visibility is empirically a regression, not a fix.

(D5 has been reverted in the working tree as of 228044b. The v3p4
artifacts are kept in SQLite + leaderboard with [¶] footnote as
audit trail.)

### 5. F2 -> F3 reversal (67 vs 86) is a useful cross-check.

- 67% of v3's initial KFs cover the GT target's visibility.
- 86% of v3's initial KFs carry **some** same-category Mask3D candidate.
- Therefore in 19% of samples, initial KFs show the **wrong**
  same-category candidate while the actual GT target sits in a
  different frame. This is the multi-distractor problem at the
  selection stage already, before the agent even reads the frame.

The number does not change much across packs — it's a structural
property of how visibility scoring + multi-instance scenes interact.

## Where to look next

The funnel rules out three otherwise-plausible explanations and
points the next iteration:

1. **Initial-keyframe selection is NOT the lever** (X1 already
   tested this directly; the funnel re-confirms it via F2/F3).
2. **Callback-side recovery has a 14pp ceiling** — worth fixing
   eventually but not the first lever.
3. **The Mask3D pool itself is fine** — F4b ~ 94% means same-category
   candidates are reachable.
4. **Picking is the dominant loss bucket — 44pp out of the 100→42
   funnel is "saw the right frame, picked the wrong proposal".**

Concrete next-step proposals (each must be validated against the
funnel before being claimed as a fix):

1. **Per-sample bad-case audit on the 44 picking-error samples.**
   For each sample, look at the agent's tool_trace to see what
   spatial reasoning it attempted, what categories it was choosing
   between, and whether the description's spatial cue could have
   been applied with the available tools. Group by failure mode
   (left/right confusion, attribute mismatch, spurious anchor, etc.)
   to identify patterns.

2. **Parser correctness audit on the 14 F4-fail samples.** Are the
   parsed target categories actually the right category for those
   queries? If the parser mis-categorized "kitchen cabinet" as
   "cabinet" (or vice versa), the same-category Mask3D filter
   missed the right pool. Run parser fresh on those samples and
   compare to GT category.

3. **Add a parser-fidelity check to the funnel script.** Currently
   the funnel uses `target_category` from the pack-prep input
   (which equals ScanRefer's `object_name`, i.e. weakly leaks GT).
   Adding "did the LLM parser of the description recover the same
   category?" as F0 makes the parsing stage measurable.

## Reproducing the funnel

```bash
# v3 (Phase 8 hypothesis-parser KFs)
PYTHONPATH=src .venv/bin/python scripts/evaluate_keyframe_funnel.py \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --pack-name pack_scanrefer_v3_iterative \
  --data-root data/scanrefer/scannet \
  --phase8-data-root data/nr3d/scannet \
  --per-sample-dir tmp/v3_aggcorrected_random100_eval/per_sample/pack_scanrefer_v3_iterative \
  --output tmp/v3_funnel_report.json

# v3.3 (Mask3D-CG visibility KFs)
PYTHONPATH=src .venv/bin/python scripts/evaluate_keyframe_funnel.py \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --pack-name pack_scanrefer_v3p_iterative \
  --data-root data/scanrefer/scannet \
  --phase8-data-root data/nr3d/scannet \
  --per-sample-dir tmp/v3p3_vertical_spatial_random100_eval/per_sample/pack_scanrefer_v3p_iterative \
  --output tmp/v3p3_funnel_report.json
```

Each invocation walks 100 samples in ~30 s (no LLM calls). The
output JSON includes a per-sample row with all five flags and a
`summary` with the count at each stage.

## Files

- Code: `scripts/evaluate_keyframe_funnel.py`
- v3 funnel report: `tmp/v3_funnel_report.json`
- v3.3 funnel report: `tmp/v3p3_funnel_report.json`
- v3.4 funnel report: `tmp/v3p4_funnel_report.json` (audit)

## Caveats

- F2/F4 use Phase 8 visibility for the **GT target_id**. This is
  an oracle: it answers "did the agent see the right *target*",
  not "did the agent see the right *category*". F3/F4b answer
  the latter weaker question via Mask3D label fuzzy match.
- F5a / F5b in this funnel use the per-sample IoU from the run,
  which used Phase 8 GT-CG bboxes. Agg-GT-rescored Acc@0.25 numbers
  are 47/45/46/48/44 for v3 / v3.1 / v3.2 / v3.3 / v3.4 — the
  ordering across packs is preserved on both evaluators.
- The `view_keyframe_marked` tool_trace records `frame_id` directly;
  callback-added frames are parsed from the response text via the
  regex `New view IDs: \[...\]`. `request_crops` does not emit new
  frame_ids in its response (only counts) so its contribution to
  cumulative coverage is not measured here.
- The Mask3D label match is the same fuzzy token-overlap matcher
  used in pack-prep (`_matching_proposal_ids`), so noise in
  ScanNet200 vocab vs ScanRefer `object_name` is shared with the
  pipeline under test.
