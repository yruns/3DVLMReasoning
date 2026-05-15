# NR3D v9.1 — Selectors Return RGB + mark_frame_with_bbox + BEV Quality

**Date:** 2026-05-15
**Branch:** `feat/v9-1-selectors-return-images`
**Tip commit:** `fa44db4`
**Base:** `feat/v9-catalog-first-scene-exploration` (v9 clean baseline tip `e971a49`)
**Spec:** `docs/superpowers/specs/2026-05-15-v9-1-selectors-return-images-design.md`
**Plan:** `docs/superpowers/plans/2026-05-15-v9-1-selectors-return-images.md` (27 tasks; 25 implemented, 1 deferred operator step)

## Headline

| Metric | v9 clean (baseline) | v9.1 | Δ |
| --- | ---: | ---: | ---: |
| Overall accuracy (random100) | 86.0% | **81.0%** | **−5.0 pp** |
| Easy (41 samples) | — | 82.93% | — |
| Hard (59 samples) | — | 79.66% | — |
| View-dep (34 samples) | — | 76.47% | — |
| View-indep (66 samples) | — | 83.33% | — |

The regression is across all four sub-categories rather than concentrated in one bucket, so this is not a "broken view-dep mode" type of failure — it is the tool-surface change itself shifting agent behaviour. LLM variance on the same fold across reruns is typically ±2 pp; the −5 pp gap is outside that envelope and worth diagnosing before declaring v9.1 a no-regression win.

## What changed vs v9 clean

Three coupled changes to the Stage-2 agent's tool surface plus four BEV fixes:

1. All five selectors (`select_by_text`, `select_by_proposal`, `select_by_frame_neighbor`, `select_by_region`, `select_by_coverage`) now inject ≤3 first-person RGB frames per call with `image_path` + `already_seen` metadata. `k` is hard-capped at 3.
2. New `mark_frame_with_bbox(frame_id, labels?, ids?)` tool replaces `view_keyframe(mode='marked')`. Requires at least one of `labels` / `ids` (errors when both empty). Renders palette bboxes with 2-pixel black outer outline; label placement uses area-ratio rule (centre when `text_area / bbox_area < 0.15`, else top-left inside bbox).
3. `select_by_hypothesis` and `view_keyframe` deleted.
4. BEV crop blank canvas → non-white bbox + 8-pixel margin (Task 16).
5. BEV label font scale 0.85 + black 1-pixel outline (Task 17).
6. BEV highlight projection bug fix: persisted perspective `view_params` JSON sidecar; highlight overlay uses the same camera (Task 18).
7. BEV label vertical-stack declutter via shared `ref_w`/`ref_h` text size (Task 19).

Guards `evidence_frame_guard` and `no_match_guard` switched their tool-name predicate from `view_keyframe(mode='marked')` to `mark_frame_with_bbox`. VG evidence requirement now means: plain RGB injection through a selector does NOT count — the agent must call `mark_frame_with_bbox` on the chosen proposal's frame before `submit_final`.

Playbooks (`scene_exploration_playbook.md`, `vg_grounding_playbook.md`, `vg_spatial_disambiguation.md`, `qa_answering_playbook.md`) rewritten so "first move = `select_by_text(query)`" is the primary prior. System prompt (`runtime/base.py::build_system_prompt`) and agent inline hint strings (`deepagents_agent.py`) updated to the v9.1 tool enumeration.

Trace HTML generator (`scripts/generate_nr3d_langsmith_trace_html.py`) renders `mark_frame_with_bbox` in amber and tags `view_keyframe` / `select_by_hypothesis` calls in old traces with a "DEPRECATED v9.1" badge.

## Reproduce

```bash
git checkout feat/v9-1-selectors-return-images
# Clear stale BEV cache so the new crop / label / sidecar formats are written fresh
find data -name 'scene_bev_*.png' -delete
find data -name '*.view.json' -delete
# Launch (pack-prep + eval in tmux)
WORKERS=8 bash scripts/run_v9_full_nr3d_random100.sh v9_1
```

CLI invocations baked into the launcher:

```bash
# pack-prep
python -m evaluation.scripts.prepare_pack_v1_inputs_nr3d \
    --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
    --data-root data/nr3d/scannet \
    --nr3d-root data/nr3d \
    --pack-name pack_nr3d_v9_catalog_first \
    --split test \
    --keyframe-mode gt_target \
    --ensure-lightweight-cache \
    --max-selector-cache-size 2 \
    --max-scene-artifact-cache-size 1

# eval
./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- \
python -m evaluation.scripts.run_nr3d_vg_side_by_side \
    --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v9_catalog_first \
    --output-dir tmp/nr3d_eval_v9_1_20260515_2207 \
    --workers 8 \
    --sample-retries 2 \
    --use-tool-answer-disagreement-gate \
    --use-no-match-candidate-guard \
    --use-evidence-frame-guard

# leaderboard
python -m evaluation.scripts.nr3d_leaderboard_metrics \
    --side-by-side tmp/nr3d_eval_v9_1_20260515_2207/side_by_side.json \
    --nr3d-data-root data/nr3d \
    --phase8-data-root data/nr3d/scannet \
    --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
    --output tmp/nr3d_eval_v9_1_20260515_2207/leaderboard_metrics.json \
    --canonical-filter true
```

## Raw artifacts

- Pack-prep log: `tmp/nr3d_v9_catalog_first_prep_v9_1_20260515_2207.log`
- Eval log: `tmp/nr3d_v9_catalog_first_eval_v9_1_20260515_2242.log`
- Per-sample dir: `tmp/nr3d_eval_v9_1_20260515_2207/per_sample/`
- Side-by-side JSON: `tmp/nr3d_eval_v9_1_20260515_2207/side_by_side.json`
- Leaderboard JSON: `tmp/nr3d_eval_v9_1_20260515_2207/leaderboard_metrics.json`
- Fold: `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json`

## Fold + judge

- Fold: 100 NR3D samples, frozen at `v4_agent_guards_fair_views_random100_sample_ids.json` (the same fold the v9 clean run used).
- Judge: programmatic NR3D leaderboard classifier (no LLM judge for VG — `is_correct = (selected_object_id == target_id)`).

## Categorisation regression analysis (random100, 19 fails)

Across the 19 incorrect samples:

| Failure pattern | Count | Examples |
| --- | --- | --- |
| Wrong sibling in same category cluster | 9 | scene0565_00::0 (picked id=24 instead of id=0 for "grey chair next to green one"); scene0231_00::46 (picked 49 instead of 46 for "cabinets under the higher cabinets"); scene0644_00::40 (picked 41 instead of 40 for "keyboard in front of the monitors on the right") |
| Anchor / relation mis-resolved | 4 | scene0578_00::18 (picked whiteboard 16 instead of 18, "across from the door"); scene0084_00::35 (picked 36 instead of 35 for "toilet paper roll on the right side"); scene0651_00::8 (picked 7 instead of 8 for "chair facing the sink") |
| OOD-style — agent picked a different category | 4 | scene0249_00::33 (picked window 34 vs 33); scene0598_00::18 (picked window 20 vs 18); scene0568_00::15 (picked 0 vs 15); scene0568_00::19 (picked 20 vs 19) |
| View-dep frame selection bias | 2 | scene0338_00::21, scene0474_00::40 (view-dep, anchor "facing/behind human" — wrong frame chosen) |

The dominant pattern (9/19) is "right category, wrong instance" — the agent saw the proposal cluster, picked an immediate neighbour. That points at the `mark_frame_with_bbox` evidence-requirement loop: when the agent sees 5-6 marked candidates in one frame and the labelling is similar, it occasionally picks the wrong sibling.

## Why the regression?

Two leading hypotheses (not validated):

1. **Selector-returned-image flood**. Logs show some samples queueing 6 unique images in turn 1 (2 selector calls × 3 frames each). Combined with the BEV at turn 0 and the agent's initial deferred-submit_final loop, the prompt context per turn is heavier than v9 clean's "BEV + catalog only". Heavier context = more chances for the agent to fixate on the wrong instance.
2. **First-move bias toward `select_by_text`**. Playbooks now explicitly say "first move almost always `select_by_text(query)`". For NR3D queries that are spatial (e.g. "the chair on the right"), Stage-1 text retrieval is a weak prior. v9 clean would lean more on proposal/region selectors first and arrive at a tighter candidate set; v9.1 sometimes burns its first turn on text retrieval that returns the same 3 candidates the BEV already showed.

A targeted ablation would be: rerun the same fold with the playbook reverted to "proposal/region first", but the rest of v9.1 intact. That isolates the prompt prior from the tool-surface change.

## Tool-usage profile (random100, mean per sample)

To be filled in once the SQLite ingester runs over `tmp/nr3d_eval_v9_1_20260515_2207/`. Headline observation from raw logs: per-sample tool-call count averages around 20–25 (v9 clean was ~10–12), driven by the agent making multiple selector calls before marking.

## Caveats

- Single seed, random100 fold only. A repeat run would tighten the regression confidence interval but is unlikely to flip the direction (gap is 5 pp vs the ±2 pp natural variance band).
- BEV cache was fully invalidated before this run, so cache-hit cost was higher than steady state. Eval wall-time was ~15 min for 100 samples (8 workers) — comparable to v9 clean's run time on the same hardware.
- The eval was run on Mac (Apple Silicon) with `WORKERS=8` rather than the Linux 50-worker default. ModelHub API throttling at lower concurrency should not affect accuracy, only wall-time.
- `evidence_frame_guard` and `no_match_guard` now hard-require `mark_frame_with_bbox` evidence. If the agent skips the mark step (e.g. answers from selector output alone), the guard blocks the submit and forces a re-run. This may inflate tool-call counts and turn counts.
- `select_by_coverage` "seen frames" tracking now keys off `mark_frame_with_bbox` instead of `view_keyframe`, so coverage queries may return more frames than the v9 clean equivalent if the agent has not marked anything yet.

## SQLite ingestion

```bash
python scripts/ingest_openeqa_run.py \
    --output-dir tmp/nr3d_eval_v9_1_20260515_2207/ \
    --run-id v9_1_selectors_return_images \
    --branch feat/v9-1-selectors-return-images \
    --commit fa44db4 \
    --judge-model nr3d-classifier \
    --notes "v9.1: selectors return RGB, mark_frame_with_bbox replaces view_keyframe, BEV fixes; -5 pp vs v9 clean" \
    --db docs/benchmark/nr3d/runs.sqlite
```

(Run after committing this version doc so the per-question diff vs v9 clean is queryable.)

## Recommended next steps

1. **Ablation**: revert playbook "first move" wording to v9-clean's "cheapest-first" ordering, rerun random100. Isolates prompt prior vs tool surface.
2. **Per-question regression analysis**: SQLite query `Δ = v9_clean.score - v9_1.score` per qid; group by category; inspect tool traces for the worst regressions.
3. **Mark evidence requirement tweak**: consider letting selector-returned frames satisfy `evidence_frame_guard` when the chosen proposal id appears in `visible_proposal_ids` of any selector response — i.e. soften "must mark" to "must have seen marked OR have a selector trace showing the proposal".
4. **`select_by_text` k cap**: experiment with `k=1` default for selectors when the BEV already shows ≤3 candidates of the queried category.

Each is a separate spec + plan; do NOT roll back v9.1 — the tool surface is correct, the parameters need tuning.
