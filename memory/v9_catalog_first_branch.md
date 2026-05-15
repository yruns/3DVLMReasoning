# v9 Catalog-First Branch — Status Memory

Date: 2026-05-15

## TL;DR

`feat/v9-catalog-first-scene-exploration` is the canonical branch for the
v9 catalog-first NR3D pilot. It now exists on `origin` (pushed 2026-05-15,
HEAD = `8ebf701`). The active worktree on this machine is at
`.worktrees/v9-catalog-first/` and tracks `origin/feat/v9-catalog-first-scene-exploration`.

The Stage-1 seed-keyframe drain leak in
`agents.runtime.deepagents_agent.build_evidence_update_message` was fixed
on 2026-05-15. **Clean v9 random100 = 86.00 overall, 92.68 Easy, 81.36 Hard,
82.35 V-Dep, 87.88 V-Indep** (vs. leaky 81 / 85.4 / 78.0 / 73.5 / 84.9).

## Branch ancestry

```
master  ──> feat/nr3d-v4-agent-guards-fair-views  ──> feat/v9-catalog-first-scene-exploration
                                                       (45 commits ahead of master via this branch's full history)
```

Main repo checkout sits on `feat/nr3d-v4-agent-guards-fair-views`. The
worktree at `.worktrees/v9-catalog-first/` is the canonical place to
edit v9 code. `data/` and `tmp/` inside the worktree are symlinks back to
the main checkout (both are .gitignored).

## Where to find what

- **Canonical results doc**: `docs/benchmark/nr3d/v9_catalog_first_20260515.md`
- **Trace viewer (4 cases)**: `docs/benchmark/nr3d/v9_catalog_first_langsmith_trace_20260515.html`
  - Per-case assets under `docs/benchmark/nr3d/assets/v9_catalog_first_langsmith_trace_20260515/`
- **README timeline + leaderboard entry**: `docs/benchmark/nr3d/README.md`
  (latest-pilot section now shows the clean 86.00; the timeline row notes
  both clean and leaky numbers + a short leak summary)
- **Random100 fold**: `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json`
- **Pack name**: `pack_nr3d_v9_catalog_first`
- **Raw artifacts**:
  - Clean run (the canonical one): `tmp/nr3d_eval_v9_full_clean_20260515_1442/`
  - Leaky run (retained for delta analysis): `tmp/nr3d_eval_v9_full_20260515_1401/`
- **Launcher**: `scripts/run_v9_full_nr3d_random100.sh`
- **Trace HTML generator**: `scripts/generate_nr3d_langsmith_trace_html.py`
  (rewritten for v9 — Turn-0 panel = BEV + SceneCatalog table; tool cards
  colored by family setup/catalog/selector/view/reason/final)

## How to reproduce (Mac, ~30 min total)

```bash
# Activate worktree env
cd /Users/bytedance/project/3DVLMReasoning/.worktrees/v9-catalog-first
source .venv/bin/activate
export PYTHONPATH=src PYTHONUNBUFFERED=1

# Pack-prep (BEV is cached -> ~5 min on repeat, ~25 min cold)
python -m evaluation.scripts.prepare_pack_v1_inputs_nr3d \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --nr3d-root data/nr3d \
  --pack-name pack_nr3d_v9_catalog_first \
  --split test \
  --keyframe-mode gt_target \
  --ensure-lightweight-cache

# Eval (workers=50, ~5 min)
./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- \
python -m evaluation.scripts.run_nr3d_vg_side_by_side \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --output-dir tmp/nr3d_eval_v9_full_clean_<DATE>/ \
  --workers 50 --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

# Metrics
python -m evaluation.scripts.nr3d_leaderboard_metrics \
  --side-by-side tmp/nr3d_eval_v9_full_clean_<DATE>/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --output tmp/nr3d_eval_v9_full_clean_<DATE>/leaderboard_metrics.json \
  --canonical-filter true
```

No GPU required. ModelHub AKs default to the fallback list in
`src/agents/core/agent_config.py`; set `MODELHUB_AKS` env var to override.

## Stage-1 seed-keyframe leak (root cause)

`build_user_message` is correctly catalog-first (only BEV + SceneCatalog
text). But `build_evidence_update_message` ran after every agent turn and
unconditionally drained every keyframe in `bundle.keyframes` — which
includes the 5 GT-target-visible Stage-1 seeds pack-prep writes for the
runner's "non-empty keyframes" assertion. So on the agent's first
premature `submit_final` attempt (which `evidence_frame_guard` defers),
the 5 seeds quietly entered the context.

Fix (in `8ebf701`):

- `Stage2RuntimeState.initial_keyframe_paths` snapshots seed paths at
  runtime construction.
- `DeepAgentsStage2Runtime.build_agent` populates the snapshot from the
  copied bundle's keyframes.
- `build_evidence_update_message` skips any keyframe whose path is in
  `initial_keyframe_paths`. Tool-produced keyframes (e.g. `request_crops`
  appending new entries) are not in the snapshot and are still drained,
  so the legacy crop callback keeps working.

Score impact: **every split improved**. The leak was biasing the agent
toward GT-target-visible seeds (which only show ONE of N candidates),
starving the selector chain of opportunities to disambiguate
same-category clutter. Removing the bias let the agent's catalog-first
flow actually run.

## Known small followups (v10 cleanup)

1. `request_crops` in v9 is a no-op placeholder (returns text only). The
   agent wasted 21 calls on it in the clean random100 run. Either remove
   it from the v9 VG tool list or wire a real backend that crops from
   *viewed* frames (not from `bundle.keyframes` indices).
2. Pack-prep still writes the 5 GT-target-visible `keyframes` list to
   each sample artifact for runner backward-compat. Drop the runner's
   "keyframes must be non-empty" assertion and stop emitting the list.
3. `view_keyframe(mode='auto')` always resolves to `'marked'` for VG in
   this run — promote `'marked'` to the default and drop the indirection.
4. Per-LLM-call durability for `llm_calls.question_id` is still
   process-global. See CLAUDE.md "Per-LLM-call durability (going
   forward)" for the planned loguru contextvars approach.

## Tool-usage profile (clean random100)

Total tool calls across 100 samples (median ~25 per sample):

  view_keyframe              882    (8.8 / sample)
  inspect_proposal           429    (4.3 / sample)
  submit_final               228    (2.3 / sample — most include a deferred retry)
  load_skill                 203    (2.0 / sample — both playbooks)
  select_by_proposal         161
  view_bev                   111
  compare_proposals_spatial   58
  list_scene_proposals        46
  select_by_frame_neighbor    38
  request_crops               21    (NO-OP — see followup 1)
  retrieve_object_context      6
  list_frame_proposals         6
  select_by_text               3
  select_by_region             2
  select_by_coverage           2
  list_skills                  1

## Open ingest task

Per CLAUDE.md "Mandatory: SQLite ingestion of per-run logs", run:

```bash
python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v9_full_clean_20260515_1442/ \
  --run-id v9_full_clean_20260515_1442 \
  --branch feat/v9-catalog-first-scene-exploration \
  --commit 8ebf701 \
  --judge-model none \
  --notes "v9 catalog-first leak-fixed; +5.0 overall vs leaky v9_full; Easy +7.3 Hard +3.4 VDep +8.8" \
  --db docs/benchmark/nr3d/runs.sqlite
```

Has NOT been run yet (operator step). After ingesting, the row should
appear in `docs/benchmark/nr3d/runs.sqlite::runs`.
