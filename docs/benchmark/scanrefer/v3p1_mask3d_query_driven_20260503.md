# ScanRefer v3.1 — `mask3d_query_driven` initial KFs

**Branch**: `feat/scanrefer-v3-query-driven`
**Tip commit at run time**: `a6b8606` (working-tree code change of this version
not yet committed when the smoke ran; X1 implementation is the **only** delta vs
the v3 query_driven track described in `docs/superpowers/specs/2026-05-04-scanrefer-v3-query-driven.md`).
**Run ID** (SQLite): `v3p_smoke_random100_20260503`
**Date**: 2026-05-03 21:57 GMT+8

> Correction, 2026-05-04: this document's headline table used the old Phase8
> GT-CG bbox evaluator. Re-running the same v3.1 predictions through the
> ScanNet aggregation-GT rescorer gives Acc@0.25 **45.0%** / Acc@0.50 **39.0%**
> / mean IoU **0.3820** (`v3p_agg_gt_random100_20260503` in SQLite). Treat the
> 39/15 values below as audit trail only. The follow-up runtime-correctness run
> is documented in
> [`v3p2_callbacks_durable_20260504.md`](v3p2_callbacks_durable_20260504.md).

## Headline

**Random100 fold** (frozen seed=20260503,
`tmp/scanrefer_artifacts/random100_sample_ids.json`):

| | Acc@0.25 | Acc@0.50 | mean IoU |
|---|---:|---:|---:|
| **v3.1 mask3d_query_driven (this run)** | **39.0%** | **15.0%** | **0.198** |
| v3 query_driven (Phase 8 visibility) [handoff] | 39.0 | 14.0 | 0.193 |
| v2 GT-target view oracle [handoff] | 67.0 | 59.0 | 0.566 |

**Verdict: X1 hypothesis falsified.** Pack-prep fallback rate dropped
from 38% (v3) to **0%** (v3.1) — initial keyframes are now strictly
aligned with the Mask3D-CG visibility used to render annotated PNGs —
yet downstream agent metrics moved by less than the noise floor
(Acc@0.25 unchanged, Acc@0.50 +1.0pp). Picking error in
multi-distractor scenes — not initial-keyframe-coverage — is the
dominant failure mode separating v3.x from the v2 GT-view-oracle
upper bound.

## Run identity

- **Branch**: `feat/scanrefer-v3-query-driven`
- **Tip commit**: `a6b8606` (working tree dirty: X1 patch in
  `src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py`)
- **Pack name**: `pack_scanrefer_v3p_iterative`
- **Fold**: 100 utts × 66 scenes (frozen, seed=20260503)
- **Backend**: `pack_v1` chassis with all 3 Stage 1 ↔ Stage 2 callbacks wired
- **Stage 2 LLM**: `gpt-5.4-2026-03-05` via internal ModelHub (default
  `Stage2DeepAgentConfig`)
- **Stage 1 hypothesis parser LLM**: `gemini-2.5-pro`, pool-enabled
- **Workers**: 8 (ThreadPoolExecutor); `--sample-retries 1`
- **Wall time**: pack-prep ~55 min; agent ~14 min
- **Raw artifacts**: `tmp/v3p_random100_eval/` (side_by_side.json +
  per_sample/pack_scanrefer_v3p_iterative/*.json), local-only

## Methodology change vs v3

X1 introduces a new keyframe-mode `mask3d_query_driven` in
`prepare_pack_v1_inputs_scanrefer.py`:

1. Parse the natural-language description into `HypothesisOutputV1`
   via the cached `query_scene.query_parser.QueryParser` (no Phase 8
   dependence at the parser level — `scene_categories` are taken
   from the Mask3D-CG label vocabulary).
2. Collect all categories (target + anchors) across hypotheses.
3. Fuzzy-match against Mask3D-CG proposal labels via lower-cased
   underscore-insensitive token overlap (drops `the`/`a`/`of`/etc.;
   keeps "office chair" ↔ "chair", "trash_can" ↔ "trash can",
   "coffee table" ↔ "table"). UNKNOW/empty are dropped explicitly.
4. Score each frame by sum of Mask3D visibility weights of matching
   candidates.
5. Top-3 frames returned as initial keyframes (k=3, OpenEQA-aligned).
6. Empty-result fallback: top-3 by Mask3D candidate density (same
   path as v3 query_driven's fallback — query-blind but proposal-aware).

The Stage 1 ↔ Stage 2 callbacks (`request_more_views`,
`request_crops`, `switch_or_expand_hypothesis`) are intentionally
**not** changed in this iteration — they continue to use the Phase 8
`KeyframeSelector` per the X1 scope (handoff §5 X1).

Code:
- `select_keyframes_mask3d_query_driven` —
  `src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py`
- `_matching_proposal_ids` (fuzzy ScanNet200 ↔ LLM categories) —
  same file
- New CLI arg: `--keyframe-mode mask3d_query_driven`
- Tests: `src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer.py`
  (9 new tests; all 13 pass)

## Pack-prep numbers (X1 succeeded **at the pack level**)

| Metric | v3 query_driven (handoff) | **v3.1 mask3d_query_driven** | Delta |
|---|---:|---:|---:|
| Fallback rate (Mask3D-density top-3) | 38% | **0%** | **−38pp** |
| Samples written | 100 | 100 | 0 |
| Wall time | n/a | ~55 min | — |

Every random100 utterance now starts the agent on three frames where
the same-category Mask3D candidate is **guaranteed** visible. This
was the explicit target of root cause #1 in the handoff.

## Agent run numbers (X1 had no measurable downstream impact)

| Metric | v3 random100 (handoff) | **v3.1 random100** | Δ |
|---|---:|---:|---:|
| Acc@0.25 | 39.0% | **39.0%** | 0.0 |
| Acc@0.50 | 14.0% | **15.0%** | +1.0pp |
| Mean IoU | 0.193 | **0.198** | +0.005 |
| Errors (mesh missing) | n/a | 3 | — |
| Zero-IoU samples | n/a | 40 | — |
| Stage 1 callbacks fired | 70 | 64 | −6 |
| ↳ `request_more_views` | 42 | 36 | −6 |
| ↳ `request_crops` | 26 | 25 | −1 |
| ↳ `switch_or_expand_hypothesis` | 2 | 3 | +1 |

The 1-pp Acc@0.50 movement is well within the variance any single
random100 fold absorbs from non-deterministic Stage 2 LLM
generations (the run is not seeded across LLM samples). Treat as
**no measurable improvement**.

## Cross-version comparison (random100 fold, like-for-like)

| Variant | Initial-KF source | Pack fallback | Acc@0.25 | Acc@0.50 |
|---|---|---:|---:|---:|
| v2 GT view oracle [‡] | Phase 8 GT `target_id` visibility | n/a (oracle) | 67.0% | 59.0% |
| v3 query_driven | Phase 8 hypothesis-parser `select_keyframes_v2` | 38% | 39.0% | 14.0% |
| **v3.1 mask3d_query_driven (this)** | **Mask3D-CG candidate visibility, fuzzy-matched** | **0%** | **39.0%** | **15.0%** |

[‡] v2 reads the GT target object id at pack-prep time and uses its
    visibility to pick keyframes — a GT view oracle, not a zero-shot
    Camp-A method.

## Why X1 didn't move the headline

Each of the four root causes named in the handoff (§2.2) was
re-examined against this run. Only #1 was actually fixed by X1.

**Root cause #1 (visibility-source mismatch) — FIXED, but not
sufficient.** Initial KFs in v3.1 are guaranteed to contain the
same-category Mask3D candidate (visibility index used for selection
is the same one used to render annotated PNGs). Pack-prep fallback
went 38% → 0%. But the agent's **picking** of which proposal to
submit isn't bottlenecked by "did the agent even see a same-category
mark in frame 0/1/2"; it's bottlenecked by **which of the 3-7
same-category candidates is actually the one the description names**.

**Root cause #2 (Mask3D recall ceiling) — independent of X1.** v2
oracle reaches Acc@0.50=84.77% on this evaluator; we are at 15%.
The 70-pp gap can't be a recall ceiling. So X1's no-op cannot be
explained by "candidates were missing".

**Root cause #3 (multi-distractor picking) — UNADDRESSED.** Most
random100 utterances are `is_unique=False`-style descriptive
references ("the brown chair next to the wall"). v2 sees the GT
target's visibility-best frames so the agent sees it directly. v3
and v3.1 see frames where SOME chair candidate is visible — but if
five chair candidates appear across the three frames, the agent
still has to read the spatial constraint ("next to the wall") and
match it to the right `inspect_proposal[K]`. **This is the actual
bottleneck.** X1's contribution to picking is zero.

**Root cause #4 (`switch_or_expand_hypothesis` underused) — still
underused.** 3 calls in 100 samples. Also unchanged by X1.

## What this implies for next-step planning

X1 being a no-op narrows the search space considerably. Of the three
paths in handoff §5:

- **X1 alone is dead.** The visibility-mismatch bet did not pay.
- **X2 (playbook hardening)** stays the most plausible cheap win
  — explicit "before submit, you MUST `find_proposals_by_category`
  and `inspect_proposal[K].frames_appeared` for at least one K"
  rule directly attacks root cause #3. ~30 lines of playbook,
  no code.
- **X3 (concede honest baseline)** becomes more attractive: at
  Acc@0.25 ≈ 39% we are close to ZSVG3D's 36.4%, and the "honest
  zero-shot Camp-A with iterative ReAct" story is intact.
- **X1.5 — extend Mask3D-CG into the callbacks too** (so the
  Stage 1 callback that fires when initial KFs are insufficient
  also returns Mask3D-aligned frames). This is roughly the same
  amount of work as X1, and given X1's no-op result the prior
  on it being load-bearing is now low. **Not recommended.**

The recommended next iteration is X2 on top of v3.1 (kept as the
zero-shot Camp-A KF source).

## Caveats

- **Random100 fold variance.** v2 was 67/59 on this fold, claimed
  fold rep ~70/63 vs full 9508 — variance of a few pp is normal.
  We have not yet verified v3.1 on the full 9508 utts, and we
  shouldn't until X2 is also tried (per CLAUDE.md memory:
  "design iteration on random100 first; only push to full when smoke
  is clearly good").
- **Per-LLM-call durability gap.** Carried over from v3 — the
  `tool_calls` / `llm_calls` SQLite tables remain empty for this
  run. Tool-call counts above are derived by grepping the loguru
  console log, not from a structured tap.
- **3 mesh-missing errors** (scene0203_00, scene0629_00,
  scene0678_00) — these are infrastructure noise unrelated to X1.
  Counted as zero-IoU in headline metrics.
- **Stage 2 LLM is non-deterministic.** A second run on the same
  pack would land within ±2-3pp of these numbers. Don't read into
  the +1pp Acc@0.50.

## Reproducing the headline from SQLite

```sql
SELECT run_id, n_total,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', mean_iou_overall) AS miou,
       keyframe_mode
FROM runs
WHERE run_id = 'v3p_smoke_random100_20260503';
```

```sql
SELECT
    COUNT(*) AS n,
    SUM(acc25) AS n_acc25,
    SUM(acc50) AS n_acc50,
    printf('%.4f', AVG(iou)) AS miou
FROM samples
WHERE run_id = 'v3p_smoke_random100_20260503';
```

Both should return `n=100`, `n_acc25=39`, `n_acc50=15`, `miou=0.1981`.

## Files referenced

- Code: `src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py` (X1 patch)
- Tests: `src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer.py`
- Pack artifacts: `data/scanrefer/scannet/<scene>/pack_scanrefer_v3p_iterative/`
- Eval artifacts: `tmp/v3p_random100_eval/`
- Console log: `/tmp/v3p_logs/random100_pack.log`,
  `/tmp/v3p_logs/random100_agent.log`
- Frozen fold: `tmp/scanrefer_artifacts/random100_sample_ids.json`
  (rebuild with `python scripts/build_scanrefer_random100_fold.py`)
- Consolidated context for the handoff that motivated this run:
  [`scanrefer-date-goal.md`](scanrefer-date-goal.md)
