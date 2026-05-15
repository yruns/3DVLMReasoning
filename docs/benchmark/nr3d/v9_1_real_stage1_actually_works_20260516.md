# NR3D v9.1_real — Stage-1 Text Retrieval Genuinely Wired (and Hurts)

**Date:** 2026-05-16 00:35
**Branch:** `feat/v9-1-selectors-return-images`
**Tip commit at run time:** `de8225f`
**Run output:** `tmp/nr3d_eval_v9_1_real_20260516_0035/`
**Eval log:** `tmp/nr3d_v9_catalog_first_eval_v9_1_real_20260516_0035.log`
**Workers:** 32 (Mac local, BEV cache warm — no pack-prep this run)

## Headline — bug is fixed, accuracy regresses

| Metric | v9 clean | v9.1_fix (Stage-1 silently broken) | **v9.1_real (Stage-1 actually working)** | Δ |
| --- | ---: | ---: | ---: | ---: |
| Overall (n=100) | 86.0% | 86.0% | **69.0%** | **−17.0 pp** |
| Easy (n=41) | — | 87.80% | 85.37% | −2.4 |
| **Hard (n=59)** | — | 84.75% | **57.63%** | **−27.1** |
| **View-dep (n=34)** | — | 82.35% | **58.82%** | **−23.5** |
| View-indep (n=66) | — | 87.88% | 74.24% | −13.6 |

The previous v9.1_fix headline (86.0 %) was **misleading**: in that run every
single one of 100 samples logged
`ERROR: runtime.keyframe_selector is None; cannot run Stage-1 text retrieval`
in its tool trace, so the agent dispatched all subsequent reasoning over the
four catalog-only selectors (`select_by_proposal`, `select_by_frame_neighbor`,
`select_by_region`, `select_by_coverage`). The 86 % score was therefore the
**no-Stage-1 fallback**, not the v9.1 design.

v9.1_real is the first run where `select_by_text` actually executes Stage-1.
Two facts in the per-sample tool traces confirm it:

- `rg -Fc "runtime.keyframe_selector is None" tmp/nr3d_eval_v9_1_real_20260516_0035/per_sample/` → 0 hits across 100 sample artefacts (was 100/100 in v9.1_fix).
- 100/100 samples invoke `select_by_text` at least once, and the response payloads contain real `{frame_id, visible_proposal_ids, ...}` envelopes instead of the error stub.

## The two-step bug

Both steps had to go right for `select_by_text` to reach Stage-1.

1. **Selector cached on the agent** — `Stage2DeepResearchAgent.__init__` must accept `keyframe_selector` and forward it. Without this, the agent has nothing to plumb. Fixed in `d5f40ba`.
2. **Selector copied into the *correct* `Stage2RuntimeState`** — `Stage2DeepResearchAgent.build_agent` is a compatibility wrapper that **constructs its own runtime state** (line 147) instead of delegating to `DeepAgentsStage2Runtime.build_agent`. Production calls the wrapper (`Stage2DeepResearchAgent.run()` → `self.build_agent(...)`), so the assignment must happen *there* too. Without this, the wrapper's freshly built `Stage2RuntimeState` has `keyframe_selector = None`, and every `select_by_text` invocation reads that and bails out. Fixed in `de8225f`.

The first fix (d5f40ba) was a no-op in production because it only patched the runtime impl's `build_agent` — a function production never called. The wrapper version of `build_agent` lives in `stage2_deep_agent.py` and is the one production actually invokes; that's where the wiring has to be.

## What the data tells us about the v9.1 hypothesis

The v9.1 design wager was:

> "`select_by_text` is the keystone first-move tool. The agent should call it
> first; it returns ≤3 candidate frames mapped from natural language via
> Stage-1; everything else (proposal / region / coverage selectors,
> mark_frame_with_bbox) is refinement."

With Stage-1 actually online, that wager **loses 17 pp on the same fold,
same prompt, same code**. The drop is concentrated in:

- **Hard-tier samples (−27.1 pp)** — these need precise disambiguation, and Stage-1's 3 candidate frames are often the wrong instances.
- **View-dependent samples (−23.5 pp)** — these need a frame that shows the referent from the implied vantage point. Stage-1's frame ranking does not understand "facing", "across from", "to the right of".

Easy / view-indep samples lose only −2 to −14 pp. Those cases were already
discoverable through the catalog tools, so when `select_by_text` returned
something usable the agent did not need it; when it returned a misleading
frame, the catalog signal still anchored the answer.

## Root-cause hypotheses for the regression

(Not yet validated — see "Next steps".)

1. **Stage-1 retrieves "target-visible frame", not "disambiguating frame"** — `select_keyframes_v2` was trained / tuned for OpenEQA-style "where does this object appear" workloads. NR3D referring expressions need the frame from which the spatial referent is unambiguous (e.g. "the chair on the right" — Stage-1 may pick any chair-containing frame, agent then misreads "right" relative to that arbitrary camera).
2. **`mark_frame_with_bbox` over-trusts the first returned frame** — once Stage-1 hands the agent a frame, the playbook says "mark the candidate proposal there". If the frame is geometrically misleading, the mark only confirms the wrong proposal more vividly.
3. **Multi-candidate playbook bias** — for hard samples there are often 4-6 catalog candidates of the queried category. `select_by_text` collapses them into a single "best match" before the agent has reasoned about the spatial relation. The catalog-first path (broken v9.1_fix) keeps all candidates visible until the agent explicitly resolves the relation.

## What this means in practice

- The v9.1 tool surface (selectors return RGB, `mark_frame_with_bbox` replaces `view_keyframe(mode='marked')`, BEV quality fixes) **stands**. None of those are responsible for the regression — they were live in both v9.1_fix and v9.1_real and v9.1_fix scored 86 % without Stage-1.
- The v9.1 **prompt prior** ("first move = `select_by_text`") **is wrong for NR3D**. We need to either reorder the playbook or condition on task / query type.
- The `de8225f` fix is correct as a wiring fix — Stage-1 must reach the agent, otherwise we have no chance to use it where it does help (OpenEQA QA, etc.). What we choose to do with Stage-1 in the playbook is a separate question.

## Reproduce

```bash
git checkout feat/v9-1-selectors-return-images
git log --oneline de8225f -1  # confirm the wrapper-fix commit is present
WORKERS=32 bash scripts/run_v9_full_nr3d_random100.sh v9_1_real
```

BEV cache is warm — pack-prep takes <1 min; eval takes ~10-12 min.

## Raw artifacts

- Per-sample dir: `tmp/nr3d_eval_v9_1_real_20260516_0035/per_sample/`
- Side-by-side JSON: `tmp/nr3d_eval_v9_1_real_20260516_0035/side_by_side.json`
- Leaderboard JSON: `tmp/nr3d_eval_v9_1_real_20260516_0035/leaderboard_metrics.json`
- Fold: same `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json` as v9 clean / v9.1_fix.

## Next steps (in order of expected impact)

1. **Revert the "first move = `select_by_text`" wording** in `scene_exploration_playbook.md`, `vg_grounding_playbook.md`, `vg_spatial_disambiguation.md`. Restore the v9-clean ordering: catalog selectors (`select_by_proposal` / `select_by_region`) first, fall back to `select_by_text` only when no catalog id matches the query. Rerun random100 — if it returns to 86 %+ we ship the catalog-first playbook with the wired Stage-1 still available as a tool but not a default.
2. **Per-task playbook**: NR3D / VG → catalog-first; OpenEQA / QA → keep `select_by_text` as default (Stage-1 is genuinely useful for "where is the kitchen counter" kind of queries).
3. **Stage-1 query parsing tweak**: have `select_keyframes_v2` ignore directional / orientation tokens ("on the right", "facing", "behind") when scoring frames; NR3D query parsing already extracts these into the hypothesis tree, so emit them as a spatial filter the agent applies AFTER Stage-1, not as a retrieval signal.
4. **Mark-first confidence**: change `evidence_frame_guard` so a mark on a Stage-1-suggested frame is treated with lower confidence than a mark on a catalog-derived frame — encourages the agent to verify Stage-1's frame against the catalog candidate set.

## Caveats

- Single seed, single fold. Variance ±2 pp on this fold ≪ the 17 pp drop, so the regression is real.
- No LLM judge required — programmatic NR3D classifier compares `selected_object_id == target_id`.
- Eval used 32 workers; same ModelHub 429 / key-rotation retries as v9.1_fix.

## SQLite ingestion

```bash
python scripts/ingest_openeqa_run.py \
    --output-dir tmp/nr3d_eval_v9_1_real_20260516_0035/ \
    --run-id v9_1_real_stage1_actually_works \
    --branch feat/v9-1-selectors-return-images \
    --commit de8225f \
    --judge-model nr3d-classifier \
    --notes "v9.1_real: select_by_text genuinely calls Stage-1; -17pp vs v9.1_fix BROKEN; v9.1 first-move prior validated as wrong for NR3D" \
    --db docs/benchmark/nr3d/runs.sqlite
```
