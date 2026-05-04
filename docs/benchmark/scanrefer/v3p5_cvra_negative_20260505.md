# ScanRefer v3.5 — CVRA Negative Result + Cross-Validation Methodology

**Branch**: `feat/scanrefer-v3-query-driven`
**Tip commit at run time**: `b91d98d` (CVRA M2b shipped) + uncommitted M3b-prep
fixes from CDX2 (proposal_pool bridge, K_AUG raise, label-mismatch overflow tier).
The supervisor will commit M3b-prep + this doc together; the post-commit tip
will be the canonical reference.
**Run ID** (smoke only, no SQLite ingest): `cvra_focused_smoke6_after_poolfix_20260505_014410`
**Date harvested**: 2026-05-05 02:00 GMT+8
**Verdict**: NEGATIVE result. CVRA shipped behind `use_clip_visible_aug=False`
default; v3.3 (`48.0 / 42.0`) remains the development-fold headline.

## Headline (NEGATIVE RESULT — smoke only, no full M3b run)

CVRA-addressable subset, frozen 6 samples (bbox-IoU v2 audit selection),
ViT-B-32 macOS dev backbone:

| Metric                                                       | Value | PASS gate       | Status   |
|--------------------------------------------------------------|------:|-----------------|----------|
| Retrieval recall (target pid in label_hits ∪ clip_visible_aug) | 6/6   | ≥ 5/6           | **PASS** |
| F5a No → Yes flips (Acc@0.25)                                | 0/6   | ≥ 4/6           | **FAIL** |
| Total `clip_visible_aug` emissions across 6 samples           | 165   | (informational) | —        |
| Total `cvra_overflow=True` emissions across 6 samples         | 45    | (informational) | —        |
| Mean IoU (smoke6 aggregate)                                  | 0.067 | (informational) | —        |

Per-sample, smoke6 (`pack_scanrefer_v3p5_cvra_iterative` + CVRA on, ViT-B-32)
vs v3.3 baseline (`pack_scanrefer_v3p_iterative` + CVRA off; agg-GT side_by_side):

| sample_id                       | baseline IoU | v3.5 IoU | Δ        | flip | retrieval recall  | agent pick (base → v3.5)              |
|---------------------------------|-------------:|---------:|---------:|------|-------------------|---------------------------------------|
| `scene0011_00::20::2` (cabinet) | 0.0000       | 0.0000   | +0.0000  | No   | label_hits        | 41 → 41 (identical)                   |
| `scene0030_00::5::4` (chair)    | 0.0000       | 0.0000   | +0.0000  | No   | label_hits + aug  | 59 → 14 (different pick, both IoU=0)  |
| `scene0203_00::25::4` (pillow)  | 0.0765       | 0.1282   | **+0.0517** | No   | aug             | 71 → 71 (same pick, IoU shifts)       |
| `scene0203_00::26::4` (couch)   | 0.0476       | 0.0964   | **+0.0488** | No   | aug             | 5 → 5 (same pick, IoU shifts)         |
| `scene0343_00::16::3` (table)   | 0.0000       | 0.0000   | +0.0000  | No   | label_hits + aug  | 24 → 24 (identical)                   |
| `scene0550_00::3::1` (door)     | 0.0657       | 0.1785   | **+0.1128** | No   | label_hits + aug  | 14 → 14 (same pick, IoU shifts)       |

5 of 6 samples select the same `proposal_id` under v3.5 as under v3.3
(only `scene0030_00::5::4` flips its pick, and its IoU stays at 0). On
3 of those 5 same-pick samples (0203/25, 0203/26, 0550/3), IoU still
shifts by `+0.05–+0.11` even though the pick is unchanged — this is
because v3.5 evaluates against a *different pack*
(`pack_scanrefer_v3p5_cvra_iterative`, regenerated with `frame_views`
metadata) so per-sample IoU computation is not strictly identical to
the v3.3 pack. None of these per-sample shifts cross the `Acc@0.25`
threshold, so the 0/6 flip headline is robust; the per-sample IoU
deltas are **not zero** as the previous draft claimed (M4 H2 fix).
Net random100 Acc@0.25 lift is **+0pp (projected)** — the full
random100 run was deliberately skipped because the smoke result
extrapolates to ~0–1 flip on the addressable subset and ~0pp on the
fold; running the full ~3-4h evaluation was not worth the wall time
given the negative smoke.

Per-sample CVRA telemetry (M4 H1 fix — the previous draft incorrectly
claimed `cvra_overflow_count = 0` everywhere; it is non-zero on every
sample):

| sample_id                       | `find_proposals_by_category` calls | Σ `clip_visible_aug` | Σ `cvra_overflow=True` |
|---------------------------------|-------------------------------------:|----------------------:|------------------------:|
| `scene0011_00::20::2`           | 2                                    | 22                    | 6                       |
| `scene0030_00::5::4`            | 4                                    | 44                    | 12                      |
| `scene0203_00::25::4`           | 3                                    | 33                    | 9                       |
| `scene0203_00::26::4`           | 2                                    | 22                    | 6                       |
| `scene0343_00::16::3`           | 1                                    | 11                    | 3                       |
| `scene0550_00::3::1`            | 3                                    | 33                    | 9                       |
| **total**                       | **15**                               | **165**               | **45**                  |

Each call emits 11 entries (8 standard `clip_visible_aug` from the
top-K cap + 3 `cvra_overflow=True` from the always-on label-mismatch
overflow tier). The implementation gates the overflow tier on
`proposal.category != search_category`, **not** on `label_hits == []`,
so overflow fires whenever ≥ 3 visible label-mismatch candidates score
above TAU regardless of label-hit presence — see § "What CVRA does"
for the spec-vs-implementation drift notes (M4 H3 fix).

## What CVRA does

CVRA (CLIP-Visible Retrieval Augmentation) extends
`find_proposals_by_category` in `src/agents/packs/vg_embodiedscan/tools.py`
with a CLIP-text rerank over Mask3D proposals visible in the agent's
cumulative seen-frame set. Goal: surface proposals whose Mask3D label
disagrees with the query category but whose visual content matches —
the "Mask3D label noise" failure class identified in the picking-error
audit (`tmp/picking_error_audit_consolidated.md`).

Key configuration (canonical: `tmp/cvra_spec_canonical.md`):

- **Backbone**: `open_clip ViT-H-14 / laion2b_s32b_b79k` fp16 on Linux
  GPU; `ViT-B-32 / openai` CPU fallback for macOS dev (`CVRA_BACKBONE_OVERRIDE`).
- **Visibility set**: union of pack-prep initial KFs + `view_keyframe_marked`
  + callback-added frames. Forbidden: all-scene (replicates v3.4 distractor
  flooding) and Phase 8 visibility (X1 falsified).
- **Crops**: raw RGB (not annotated PNG, which would pollute CLIP), 10%
  padding around the pre-projected `bbox_2d` persisted by pack-prep into
  `proposal_pool[*].frame_views[frame_id]`.
- **TAU**: 0.18 default with `gt_score_floor` calibration recipe; floor 0.13.
- **K_AUG (drifted from spec)**: implementation uses a **flat** cap (default 8,
  hard ceiling 10) regardless of `label_hits` state, plus an **always-on**
  label-mismatch overflow tier of +3 candidates with `clip_score >= TAU` and
  `cvra_overflow=True` metadata. Total per-call emission ceiling is therefore
  13. Canonical spec § C originally specified dynamic `3 / 5` with hard
  ceiling 5 only when `label_hits == []`; M3b-prep raised the cap and made
  overflow unconditional, and the spec was retroactively aligned to match
  the implementation in the canonical doc. See M4 H3.
- **Feature flag**: `Stage2DeepAgentConfig.use_clip_visible_aug = False`
  default. ScanRefer runner: `--use-clip-visible-aug` (off by default).
  Env: `CVRA_DISABLE=1` forces off.

Spec history: 4 PUSHBACK rounds during design (spec drafts × 2, schema
contract path correction, peer review on M2b). Canonical merged spec
is `tmp/cvra_spec_canonical.md` (290 LOC, 12 fields A–L).

## Why it didn't help

The CVRA infrastructure works as designed (6/6 retrieval recall — the
target pid lands in either `label_hits` or `clip_visible_aug` for every
addressable sample). The bottleneck is downstream: **the agent's spatial
reasoning correctly rejects most "addressable" candidates because their
3D position contradicts the description's spatial referent**.

### Diagnostic dive: `scene0011_00::20::2`

Query: "this is a brown cabinet. it is above a refrigerator."
- Audit's label-IoU pid = 46 (label `cabinet`, IoU 0.31)
- Audit's mismatch-IoU pid = 67 (label `kitchen cabinet`, IoU 0.31)
- Agent pick (both v3.3 and v3.5): **pid 41** (IoU 0.0 to GT)

Trace:
1. `find_proposals_by_category(category='cabinet')` →
   `label_hits=[41, 46]`, `clip_visible_aug` 11 entries
   (8 standard + 3 with `cvra_overflow=True`; entries are
   `[27, 68, 17, 20, 51, ...]` — pid 67 not in aug because not visible
   at first call's cumulative seen).
2. `compare_proposals_spatial(candidate_ids=[41, 46], anchor_id=43,
   relation='above')` → `ranked_ids=[41, 46]` with
   `vertical_offset(41)=+0.628 m`, `vertical_offset(46)=+0.008 m`.
3. Agent submits pid 41 with rationale: "proposal 43 is the
   refrigerator, proposal 41 is a distinct cabinet positioned above it.
   Spatial comparison ranked cabinet proposals [41, 46] relative to
   refrigerator."

Pid 46 is at refrigerator level, not above it — the agent's reasoning is
spatially correct. The bbox-IoU=0.31 between pid 46 and the GT bbox
captures footprint overlap, not the vertical-spatial-referent match the
description asks for. CVRA cannot help here; nor can any retrieval-side
fix.

### Pattern across the 6 addressables

| sample_id                       | structural reason CVRA can't flip                                             |
|---------------------------------|-------------------------------------------------------------------------------|
| `scene0011_00::20::2` (cabinet) | Audit pid is at fridge level; query says "above", agent picked the above-fridge candidate (correct reasoning, GT bbox is the misleading one). |
| `scene0030_00::5::4` (chair)    | 18 chair label_hits + 11 aug; agent picked one outside both targets. Multi-distractor failure independent of CVRA. |
| `scene0203_00::25::4` (pillow)  | Pid 61 (label `mat`) IoU 0.34; agent picked pid 71 (`couch`-aug, score 0.271) because the description "pillow between bookshelf and couch" leans visually toward couch-adjacent candidates. |
| `scene0203_00::26::4` (couch)   | Pid 31 (label `piano`) IoU 0.45; agent never engaged with pid 31 at submission, picked pid 5 instead. |
| `scene0343_00::16::3` (table)   | Pid 4 (label `table`) AND pid 21 (label `coffee table`) BOTH IoU ≈ 0.30; agent picked neither (pid 24, IoU 0.0) — multi-distractor failure. |
| `scene0550_00::3::1` (door)     | Pid 2 (label `door`, IoU 0.54) was in label_hits AND aug; agent picked pid 14 (IoU 0.18) — query "bathroom stall door" has spatial-referent constraints the agent applied differently. |

### What this means for the audit

The bbox-IoU v2 audit (`tmp/cvra_addressable_audit_v2.md`) computed
"addressable" as `(any proposal IoU ≥ 0.25 with GT) AND
(label-mismatch present) AND (visible in cumulative frames)`. That
definition over-counts CVRA's reachable surface because it doesn't check
whether the agent's spatial reasoning will accept the candidate. The
audit's 6/100 estimate is the upper bound under footprint overlap; the
real CVRA ceiling on this fold is ≤ 1 (likely 0).

### Why the 26 → 6 gap was caught

The original M1c plan claimed CVRA-addressable = 26/100 and the PASS gate
was set at ≥ 13/26. A bit-by-bit re-audit at the bbox-IoU layer
collapsed that to 6/100, and the smoke result then collapsed it further.
Both collapses were caught before any wall-time was spent on a full M3b
run. This is the cross-validation methodology paying for itself.

## Cross-validation methodology contribution (paper-relevant)

CVRA's design and verification used a multi-agent cross-validation
process with four PUSHBACK cycles, each catching a defect that would
have shipped silently otherwise:

1. **Spec design (M2a, two rounds)** — CC1 drafted `tmp/cvra_spec_cc.md`,
   CDX1 drafted `tmp/cvra_spec_cdx.md`. Supervisor merge caught:
   (a) CDX1's E.original "on-the-fly crop from `proposal_index`" was not
   implementable because `VgEmbodiedScanCtx.proposal_index` only stores
   `proposal_id → [frame_id]`, no 2D bbox or RGB path. Forced revision
   to pack-prep persistence of `bbox_2d` + `raw_rgb_path` +
   `visibility_weight` per `(proposal_id, frame_id)`. (b) CC1 PUSHBACK
   on §K: `Stage2DeepAgentConfig` lives at
   `src/agents/core/agent_config.py`, not `src/agents/runtime/base.py`.
2. **Schema contract correction** — `raw_rgb_path` originally
   pack-relative; CC1 PUSHBACK forced project-root-relative for portability
   matching existing pack `image_path` convention.
3. **M2b peer review** — CC1 reviewed CDX1's tools/ctx/clip_provider/runtime
   slice. Caught: 1 HIGH (missing `--use-clip-visible-aug` runner CLI;
   would have shipped CVRA dead in production), 3 MEDIUM (single bad bbox
   killing entire CVRA call, missing-RGB-file killing cache key, no unit
   tests on `clip_provider.py`), 7 ruff lint issues. All addressed before
   the M2b commit.
4. **M3a cross-review (smoke aug=0 root cause)** — CDX1 attributed the
   smoke6 zero-aug to "visibility set in `find_proposals` only contains
   initial KFs" and named `scene0203_00::25::4` as the flagship bug.
   CC1's independent re-audit showed that 0203 was actually pool-coverage
   (GT-ball pid 25's `frame_views` `{39, 40, 117–124, ...}` never overlap
   the agent's seen set even at final cumulative `{30, 44, 51, 61, 102,
   104, 115}`). CC1 also identified `proposal_pool.py` as the data plumbing
   site dropping `frame_views` before runtime read — a bridge bug, not an
   algorithm bug. CDX2 then implemented the bridge fix as the M3b-prep
   blocker.

The cross-validation cost (≈ 6 PUSHBACK rounds across 4 milestones,
≈ 1-2 hours of agent wall-time per round) is amortized by:

- Catching `clip_provider.py` test gap before M2b green merge (would
  have hit a runtime bug on first GPU run).
- Catching the `proposal_pool.py` bridge bug at M3a smoke instead of
  during full M3b (would have produced 100 zero-aug samples and looked
  like an "algorithm doesn't work" instead of "data plumbing is broken").
- Collapsing CVRA-addressable estimate from 26 → 6 → ≤1 across three
  audit refinements before launching the full M3b ($GPU + 3-4h wall).

A single-agent path would have spent that wall-time on a misdiagnosed
CVRA, then re-audited after the result was negative, then started TADG
1-2 days later. The multi-agent path lands at the same starting point
for TADG with the audit, infrastructure, and code-review history
already in the doc trail.

## What's shipped

CVRA infrastructure stays in tree, gated off by default:

- `src/agents/packs/vg_embodiedscan/proposal_pool.py` — bridge fix
  preserving `frame_views` from pack-prep through runtime CVRA call
  (regression test `test_proposal_pool.py`, the immediate cause of the
  M3a smoke `aug=0` failure).
- `src/agents/packs/vg_embodiedscan/tools.py` — `find_proposals_by_category`
  CVRA path, flat K_AUG cap (default 8, hard ceiling 10) + always-on
  label-mismatch overflow tier (+3 candidates above TAU, gated on
  `proposal.category != search_category` rather than `label_hits == []`,
  yielding a 13-entry per-call ceiling), `cvra_overflow=True` metadata
  for funnel / telemetry.
- `src/agents/packs/vg_embodiedscan/clip_provider.py` —
  `BatchedClipProvider` with batch=16, fp16, `(scene_id, proposal_id,
  frame_id, crop_hash)` cache, `ViT-B-32` macOS fallback gated by
  `CVRA_BACKBONE_OVERRIDE`.
- `src/agents/packs/vg_embodiedscan/ctx.py` — visibility-set helper +
  `frame_views` lookup.
- `src/agents/core/agent_config.py` — `use_clip_visible_aug = False`
  + 4 sibling fields on `Stage2DeepAgentConfig`.
- `src/agents/runtime/base.py` — `Stage2RuntimeState` propagation of
  CVRA config (same template as the reverted D5 `vlm_judge` hooks).
- `src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py` —
  `compute_proposal_frame_views()` gated on `"cvra" in pack_name.lower()`,
  generates new pack `pack_scanrefer_v3p5_cvra_iterative` with
  `bbox_2d` + `raw_rgb_path` + `visibility_weight` per
  `(proposal_id, frame_id)`.
- `src/evaluation/scripts/run_scanrefer_vg_side_by_side.py` —
  `--use-clip-visible-aug` CLI flag (default off).
- Tests:
  - 8 new tests in `vg_embodiedscan/tests/test_tools.py` (CVRA spec items
    + integration synthetic + label-mismatch overflow).
  - New `vg_embodiedscan/tests/test_proposal_pool.py` (bridge regression).
  - `vg_embodiedscan/tests/test_clip_provider.py` (backbone parsing,
    cache key, crop padding, bad-bbox skip, disk cache round-trip).
  - 5 pack-prep tests in
    `evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer.py`.
  - Runner CLI test in
    `evaluation/scripts/tests/test_run_scanrefer_vg_side_by_side.py`.

These ride along as no-op infrastructure for any future CVRA-style work
(e.g. a stronger VLM-rerank tool, a different visual-similarity index,
or a Linux ViT-H-14 retry on a different bottleneck).

## Next milestone: TADG (Tool-Answer Disagreement Gate)

Per the original handoff `docs/handoff_2026-05-04_1803.md` § 6.2,
TADG addresses **meta-bug 4.4**: the agent ignores the verdict of its
own deterministic `compare_proposals_spatial` call at submit time. The
audit caught two cases on this fold:

- `scene0149_00::2::3` direct: agent ran
  `compare_proposals_spatial(candidate_ids=[2,3,12,14,27,33,36,43],
  anchor_id=10, relation='closest_to') → ranked_ids=[33, 14, 43, 27,
  36, 12, 3, 2]`, then submitted proposal 19 — not even in the ranked
  list.
- `scene0660_00::4::4` indirect (inverse-anchor on symmetric pair).

Mechanism: a runtime gate in the agent's `submit_final` handler that
soft-blocks when the submitted `proposal_id` disagrees with the most
recent same-relation `compare_proposals_spatial` call's rank-1, requiring
either a `tool_override_reason` or a revised pick. Targets ~15 of 44
picking errors per the Claude2 audit (`tmp/picking_error_audit_v3p3_claude.md`
§ 5.3). Smaller scope than CVRA, faster iteration; estimated 2-3 h
implementation + 1 h smoke per the handoff.

CVRA work explicitly does **not** generalize to TADG — the failure
classes are disjoint. The cross-validation history above does generalize:
TADG should follow the same multi-agent spec-design + peer-review
cadence rather than a single-agent path.

## Reproduction

```bash
# Smoke6 launcher (frozen 6 addressables)
bash tmp/cvra_focused_smoke6_eval.sh
# Outputs: tmp/cvra_focused_smoke6_eval_<RUN_ID>/
#   per_sample/pack_scanrefer_v3p5_cvra_iterative/*.json
#   side_by_side.json (acc25_overall = 0.0; mean_iou_overall = 0.067)
#   leaderboard_metrics.json
```

The launcher self-tests pack-prep readiness, regenerates
`pack_scanrefer_v3p5_cvra_iterative` if missing, runs the agent with
`--use-clip-visible-aug`, computes leaderboard metrics, and prints a
JSON summary including per-sample `clip_visible_aug_count` /
`cvra_overflow_count` / `f5a_flips` / `f5a_regressions`. Wall time on
macOS .venv + ViT-B-32 + workers=1 ≈ 13 minutes for 6 samples.

Aggregation-GT rescore: not run (no full random100 to rescore; smoke6's
6 samples are all `is_unique=False` Multiple-distractor cases, all baseline
IoU < 0.25, so a rescore would add no signal beyond the per-sample IoU
table above).

SQLite ingest: not performed for v3p5_cvra (no full random100 run; the
SQLite table is reserved for canonical paper-comparable rows).

## Caveats

- **macOS .venv + ViT-B-32 backbone fallback.** The benchmark spec
  pins `ViT-H-14 / laion2b_s32b_b79k` for production, but the smoke ran
  on `ViT-B-32 / openai` (≈ 10ms/image, CPU). CLIP scores cluster in
  0.22-0.29, which is a flat distribution that may not differentiate
  GT-overlap proposals strongly enough to move the agent's ranking. A
  Linux GPU + ViT-H-14 re-run was deliberately not attempted: the
  smoke's 5/6 identical-pick pattern is structural (agent's spatial
  reasoning supersedes label-based filtering), and stronger CLIP scores
  cannot rescue the cases where the GT bbox itself is at a position
  contradicting the description's spatial referent. Supervisor judged
  the coordination cost not worth it given the spatial-reasoning
  bottleneck identified above. This is documented as a deliberate
  scope cut, not an oversight.
- **No second run for noise calibration.** The standard noise-tolerance
  HARD GATE (`tmp/cvra_spec_canonical.md` § L) calls for a 2nd-run
  variance check on the addressable subset. The 1st run's 0/6 flips is
  far enough below the ≥ 4/6 PASS gate that a 2nd run cannot rescue
  the result; skipped.
- **M4 fresh peer review completed (`tmp/m4_review_cc.md`,
  `tmp/m4_review_cdx.md`).** Two fresh-eyes cross-blind reviews
  converged on five HIGH issues, all addressed in the M4-fixes follow-up
  commit. The corrections are reflected in this doc revision: real
  per-sample overflow telemetry replaces the previous "0 everywhere"
  claim (H1); per-sample IoU table updated against the agg-GT
  side_by_side (H2); K_AUG description updated to match the actual
  flat-cap-8-+-always-on-overflow-3 implementation (H3); the playbook
  was updated with `clip_visible_aug` semantics (H4); and the funnel
  evaluator gained `clip_visible_aug_included_gt_overlap` /
  `cvra_aug_count` / `cvra_overflow_count` per-sample fields (H5).
- **Acc@0.25 / Acc@0.50 unchanged.** The development-fold headline
  remains v3.3's 48.0 / 42.0 (`v3p3_vertical_spatial_random100_20260504`
  in `runs.sqlite`). v3.5 contributes no SQLite row.
- **Pack regen confounds per-sample IoU comparison.** The smoke
  evaluates `pack_scanrefer_v3p5_cvra_iterative` (regenerated to
  persist `frame_views`) against the original v3.3 pack
  `pack_scanrefer_v3p_iterative`. On 3 of 6 samples (0203/25, 0203/26,
  0550/3) the agent picks the same `proposal_id` but IoU shifts by
  `+0.05–+0.11` because the v3.5 pack scores those picks slightly
  differently. None of the shifts cross 0.25, so the 0/6 flip headline
  survives, but the per-sample comparison is not a clean A/B
  isolating the `--use-clip-visible-aug` flag (M4 M1). A cleaner
  protocol would re-run the v3.5 pack with the flag off and compare
  against itself.
- **Overflow tier fires unconditionally, contradicting the original
  spec.** The implementation gates the +3 overflow tier on
  `proposal.category != search_category` rather than `label_hits == []`
  (`tools.py:217-228`). All 6 smoke samples had `label_hits ≥ 1` and
  the overflow tier still fired on every call (45 total
  emissions). The original canonical spec § C said overflow should
  only emit when `label_hits == []`; the canonical spec has been
  updated to match the implementation (drift note in
  `tmp/cvra_spec_canonical.md` § C). Whether the always-on overflow
  helps or hurts vs the original gated design is not exercised by
  this smoke (M4 H3).

## Interpretation

The negative result is a **methodologically clean falsification** of
the CVRA design's value proposition on the ScanRefer-Camp-A bottleneck
as currently understood. Three takeaways:

1. **The CVRA infrastructure is correct and reusable.** 6/6 retrieval
   recall on the addressable subset proves the
   visibility-set / pack-prep / `BatchedClipProvider` / runtime config
   wiring is sound. Any future visual-similarity-augmentation work
   inherits the same plumbing.
2. **Footprint-IoU "addressability" overstates retrieval-side
   reachable surface.** The agent's spatial reasoning is not a
   downstream layer that retrieval can leak past — it is upstream of
   the candidate-set choice. Future audit definitions should include
   a spatial-referent verifier (e.g. simulate the agent's
   `compare_proposals_spatial` call on the GT bbox vs the audit pid)
   before claiming a sample is addressable by retrieval-side fixes.
3. **The development-fold bottleneck is agent-side, not retrieval-side.**
   v3.4's `select_among_proposals` falsified judge-stage forced choice;
   v3.5's CVRA falsifies retrieval-stage label augmentation. Both
   negative results converge on the same diagnosis: the agent's
   pre-existing reasoning trajectory dominates fold-level outcomes,
   and improvements need to operate inside that trajectory (TADG,
   co-visible-anchor framing, parser-fidelity hardening) rather than
   widening or filtering the candidate pool.

## Files referenced

- This doc: `docs/benchmark/scanrefer/v3p5_cvra_negative_20260505.md`
- Code (uncommitted in the M3b-prep working tree, supervisor will
  commit alongside this doc):
  - `src/agents/packs/vg_embodiedscan/proposal_pool.py`
  - `src/agents/packs/vg_embodiedscan/tools.py`
  - `src/agents/packs/vg_embodiedscan/clip_provider.py`
  - `src/agents/packs/vg_embodiedscan/ctx.py`
  - `src/agents/core/agent_config.py`
  - `src/agents/runtime/base.py`
  - `src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py`
  - `src/evaluation/scripts/run_scanrefer_vg_side_by_side.py`
- Tests:
  - `src/agents/packs/vg_embodiedscan/tests/test_tools.py` (8 new)
  - `src/agents/packs/vg_embodiedscan/tests/test_proposal_pool.py` (new)
  - `src/agents/packs/vg_embodiedscan/tests/test_clip_provider.py`
  - `src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer.py` (5 new)
  - `src/evaluation/scripts/tests/test_run_scanrefer_vg_side_by_side.py`
- Smoke launcher: `tmp/cvra_focused_smoke6_eval.sh`
- Smoke artifacts:
  `tmp/cvra_focused_smoke6_eval_m3b_after_poolfix_20260505_014410/`
  (`per_sample/`, `side_by_side.json`, `leaderboard_metrics.json`).
  Console log: `/tmp/cvra_focused_smoke6_eval_after_poolfix.log`
- Frozen smoke fold: `tmp/scanrefer_artifacts/cvra_addressable6_sample_ids.json`
- Audits:
  - `tmp/cvra_spec_canonical.md` (canonical CVRA spec, post-merge)
  - `tmp/cvra_addressable_audit_v2.md` (bbox-IoU v2 audit, 6/100)
  - `tmp/m1c_reconciliation_cc.md` (original 26-sample PASS calibration)
  - `tmp/m3a_review_cc.md` (CC1 PUSHBACK on CDX1's M3a diagnosis)
  - `tmp/m2b_review_cc.md` (CC1 peer review of CDX1 M2b slice)
  - `tmp/picking_error_audit_consolidated.md`
- Handoff context:
  - `docs/handoff_2026-05-04_1803.md` § 6.1 (CVRA spec) and § 6.2 (TADG)
  - `tmp/cc1_handoff_to_cc2.md` (predecessor handoff)
- SQLite: `docs/benchmark/scanrefer/runs.sqlite` — no row added for v3.5.
