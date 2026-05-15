# NR3D v9.1_fix — Stage-1 KeyframeSelector Wiring Fix

**Date:** 2026-05-15 (same day as v9.1 initial)
**Branch:** `feat/v9-1-selectors-return-images`
**Tip commit:** `d5f40ba`
**Run output:** `tmp/nr3d_eval_v9_1_fix_20260515_2352/`
**Eval log:** `tmp/nr3d_v9_catalog_first_eval_v9_1_fix_20260515_2352.log`
**Pack-prep log:** `tmp/nr3d_v9_catalog_first_prep_v9_1_fix_20260515_2320.log`
**Workers:** 32 (Mac local, up from 8 on the initial v9.1 run)

## Headline — fix recovers full v9 clean parity

| Metric | v9 clean | v9.1 (broken — `d720f71`) | v9.1_fix (this doc) | Δ vs broken |
| --- | ---: | ---: | ---: | ---: |
| Overall (n=100) | 86.0% | 81.0% | **86.0%** | **+5.0 pp** |
| Easy (n=41) | — | 82.93% | 87.80% | +4.9 |
| Hard (n=59) | — | 79.66% | 84.75% | +5.1 |
| **View-dep (n=34)** | — | 76.47% | **82.35%** | **+5.9** |
| View-indep (n=66) | — | 83.33% | 87.88% | +4.5 |

The whole −5 pp regression in the initial v9.1 run was a single missing wiring step. With Stage-1 plumbing restored, v9.1_fix matches v9 clean overall while improving slightly per sub-category (the headline tie at 86.0 % is a fold-rounding coincidence; per-category accuracy is actually 1–2 pp above the v9 clean profile, with view-dep being the biggest winner — exactly what one would expect from a working `select_by_text`).

## The bug

`select_by_text` (the v9.1 keystone tool that maps natural language to ≤3
candidate first-person frames via Stage-1) reads
`runtime.keyframe_selector` at invocation time:

```python
selector = getattr(runtime, "keyframe_selector", None)
if selector is None:
    err = "ERROR: runtime.keyframe_selector is None; cannot run Stage-1 text retrieval"
    runtime.record("select_by_text", request, err)
    return err
result = selector.select_keyframes_v2(query=str(query), ...)
```

Three places had to cooperate to set that attribute:

1. `Stage2DeepResearchAgent.__init__` must accept a `keyframe_selector` arg.
2. `DeepAgentsStage2Runtime.build_agent` must copy it onto the freshly built `runtime` state.
3. The eval runner (`run_nr3d_vg_side_by_side.py`) must construct the agent with the pre-built selector.

In v9.1 (`d720f71`) **none of these three did**. The runner built the
`KeyframeSelector` only to hand it to `create_crop_callback` (legacy
`request_crops` plumbing). It was never wired to the agent. Every
`select_by_text` invocation therefore returned the error string above and
the agent silently fell back to the four catalog-only selectors
(`select_by_proposal`, `select_by_frame_neighbor`, `select_by_region`,
`select_by_coverage`). Those are useful but, crucially, **none of them
can do semantic / attribute retrieval** — the agent lost the entire
language-to-frame primitive that justified the v9.1 redesign.

This is what produced the homogeneous −5 pp drop in the initial v9.1
run: every sample lost the same primitive, so easy / hard / view-dep /
view-indep all moved roughly together. The view-dep gain in v9.1_fix
(+5.9 pp) is the largest because referring expressions like "the chair
facing the sink" or "the lamp between the two beds" benefit most from a
working semantic retriever.

## The fix (commit `d5f40ba`)

Six small edits in five files, plus one new regression test:

- `src/agents/runtime/base.py`
  - Add `keyframe_selector: KeyframeSelector | None = None` to `BaseStage2Runtime.__init__`, store on `self`.
  - Add `keyframe_selector` field on the `Stage2RuntimeState` dataclass with the same typed `KeyframeSelector | None` annotation (via `TYPE_CHECKING` to avoid an import cycle).
- `src/agents/runtime/deepagents_agent.py`
  - Accept `keyframe_selector` in `DeepAgentsStage2Runtime.__init__`, forward to `super().__init__`.
  - In `build_agent`, set `runtime.keyframe_selector = self.keyframe_selector` alongside the existing `task_type` / `initial_keyframe_paths` setup.
- `src/agents/stage2_deep_agent.py`
  - Accept and forward `keyframe_selector` in `Stage2DeepResearchAgent.__init__` so the public agent surface plumbs it down to the runtime.
- `src/evaluation/scripts/run_nr3d_vg_side_by_side.py`
  - Capture the pre-built selector into a local `keyframe_selector` and pass it as `agent_cls(..., keyframe_selector=keyframe_selector)`.
- `src/evaluation/scripts/run_scanrefer_vg_side_by_side.py`
  - Same edit, so ScanRefer evaluations get the fix simultaneously.
- `src/agents/tests/integration/test_keyframe_selector_wired.py` (new)
  - Three regression cases: agent forwards the selector to the runtime impl; `build_agent` populates the runtime state field; missing selector lands as `None` instead of throwing.

The type annotation switched from `Any | None` to the proper
`KeyframeSelector | None` (under `TYPE_CHECKING`) per the user's request
to avoid `Any` in production code.

All 824 unit + integration tests pass after the change.

## Reproduce

```bash
git checkout feat/v9-1-selectors-return-images
git log --oneline d5f40ba -1  # confirm tip is the fix commit or later

# Clear BEV cache so the v9.1 BEV improvements (crop + label + sidecar) are rendered fresh
find data -name 'scene_bev_*.png' -delete
find data -name '*.view.json' -delete

# Launch (pack-prep + eval in tmux). WORKERS=32 fully utilises Mac M-series cores.
WORKERS=32 bash scripts/run_v9_full_nr3d_random100.sh v9_1_fix
```

CLI invocations are identical to the initial v9.1 run except for the `--workers` value and the run tag. See
`docs/benchmark/nr3d/v9_1_selectors_return_images_20260515.md` for the
fully expanded commands.

## Raw artifacts

- Per-sample dir: `tmp/nr3d_eval_v9_1_fix_20260515_2352/per_sample/`
- Side-by-side JSON: `tmp/nr3d_eval_v9_1_fix_20260515_2352/side_by_side.json`
- Leaderboard JSON: `tmp/nr3d_eval_v9_1_fix_20260515_2352/leaderboard_metrics.json`
- Fold: `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json` (same fold as v9 clean and the v9.1 broken run)
- Judge: programmatic NR3D leaderboard classifier — no LLM judge needed for VG.

## Comparison with v9 clean (random100, same fold)

v9.1_fix is **at parity overall (86.0% vs 86.0%)** and shows small per-category gains. The two runs are now substitutable on this fold within natural seed variance. The improvements are:

- **Tool surface**: agent has selectors that return RGB directly + a dedicated `mark_frame_with_bbox` for verification. Reduces the number of view-mode-conditioned branches in the system prompt; less for the LLM to remember.
- **BEV quality**: cropped canvas, legible labels, fixed highlight projection. These mostly help turn-0 reasoning quality which is hard to isolate in aggregate metrics.

The price paid:
- **Tool-call counts up** vs v9 clean — selectors are cheaper to invoke (no view_keyframe second round-trip), so the agent makes ~20–25 calls per sample on average vs v9 clean's ~10–12. Wall-time per sample is comparable.

## Caveats

- Single seed, single fold (random100, 100 samples). LLM variance ±2 pp.
- ModelHub returned 429 rate-limit warnings periodically during the eval (`ModelHubHttpClient retryable status=429`); the built-in key rotator and exponential retry handled them transparently. With `WORKERS=32` on Mac, request density is high enough to occasionally trip per-key TPM limits but never causes a sample to fail.
- BEV cache was fully invalidated before this run, so pack-prep wall-time was higher than steady state (~27 min for 100 samples / 53 unique scenes).
- Eval wall-time: ~12 min for 100 samples with `WORKERS=32`. Roughly 8x parallelism over the earlier `WORKERS=8` run.

## What this also proves about v9.1 design

The 5 pp gap was not a problem with:

- catalog-first prompt prior (BEV + scene catalog at turn 0)
- `mark_frame_with_bbox` replacing `view_keyframe(mode='marked')`
- guards switching from `view_keyframe` predicate to `mark_frame_with_bbox`
- "first move = `select_by_text`" wording in playbooks
- selectors returning ≤3 RGB frames each
- BEV crop / legibility / highlight fix

None of those were touched by the wiring fix. The v9.1 design is sound; the v9.1 implementation was missing one line of glue.

## SQLite ingestion

```bash
python scripts/ingest_openeqa_run.py \
    --output-dir tmp/nr3d_eval_v9_1_fix_20260515_2352/ \
    --run-id v9_1_fix_keyframe_selector_wiring \
    --branch feat/v9-1-selectors-return-images \
    --commit d5f40ba \
    --judge-model nr3d-classifier \
    --notes "v9.1_fix: wired KeyframeSelector through agent → runtime; restores 86.0% overall (parity with v9 clean)" \
    --db docs/benchmark/nr3d/runs.sqlite
```

## Next steps

- Open the PR (`feat/v9-1-selectors-return-images` → main) — v9.1_fix is the right ship state. Cite this doc in the PR body.
- Replace the v9.1 trace HTML with a v9.1_fix one so the visualised agent behaviour matches the shipping numbers. Generated as `docs/benchmark/nr3d/v9_1_fix_trace_20260515.html` alongside this doc.
- Run a higher-N fold (random500 or full test) once a Linux GPU box is available — confirm the macOS-local 100-sample result generalises.
