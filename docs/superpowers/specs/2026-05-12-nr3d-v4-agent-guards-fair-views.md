# NR3D v4 Agent Guards With Fair Views - Design

**Date:** 2026-05-12
**Branch:** `feat/nr3d-v4-agent-guards-fair-views`
**Tip commit at design time:** `e5b37bf`
**Approved approach:** A1 - fixed subset plus new-scheme single run, reusing the existing full-run baseline

## Goal

Run a first NR3D partial evaluation with the ScanRefer v3.20 Stage-2 agent
framework while removing the known GT-target-visible keyframe assist from the
old NR3D pack preparation.

The first deliverable is a 100-sample result that answers one narrow question:
on a fixed subset where the old v2/v3 NR3D run already has predictions, does
the optimized agent framework plus non-GT keyframe selection improve
classification accuracy?

This is not a new full NR3D leaderboard claim yet. It is a controlled smoke /
pilot that decides whether a larger 500-sample or full run is worth the cost.

## Background

The current documented NR3D headline is
`v3_referit3d_track_20260501`: Overall 80.79, Easy 86.06, Hard 75.87,
View-Dep 72.46, View-Indep 85.34 on the canonical filtered fold. That row was
post-aggregated from the v2 full run.

The old NR3D pack path uses the full ScanNet annotated instance pool, which is
valid for the NR3D GT-track classification setting. The problem is upstream
evidence selection: `prepare_pack_v1_inputs_nr3d.py::select_keyframes_for_sample`
currently reads `visibility.object_to_views[target_id]` and picks the first
target-visible views. That means the agent is not told the target id, but its
initial images are chosen with GT target visibility.

ScanRefer v3.20 solved the analogous issue by moving to query-driven /
proposal-visible evidence and by adding Stage-2 final-decision guards. This
design ports the same principles back to NR3D.

## Locked Decisions

1. Use a new branch: `feat/nr3d-v4-agent-guards-fair-views`.
2. Use approach A1: fixed 100-sample subset plus one new run.
3. Reuse the existing full NR3D baseline from
   `tmp/nr3d_eval_v1_full/side_by_side.json`; do not rerun the old baseline in
   the first pass.
4. Keep the NR3D proposal pool as ScanNet annotated bboxes. This is the
   canonical GT-track candidate pool and is not considered leakage by itself.
5. Do not use `target_id`, `gt_bbox_3d_9dof`, `object_to_views[target_id]`,
   target visibility, IoU, or old correctness labels when choosing keyframes or
   during agent inference.
6. Use `target_id` only after inference for offline scoring and documentation.
7. First run size is 100 canonical-filtered samples, selected deterministically
   from the old full run. If the new run has low failure rate and positive or
   neutral signal, a later version can scale to 500 or full.

## Evaluation Design

### Baseline

The baseline is the existing v2/v3 NR3D full output:

- Raw predictions: `tmp/nr3d_eval_v1_full/side_by_side.json`
- Leaderboard aggregation: `tmp/nr3d_eval_v1_full/leaderboard_metrics.json`
- Documented row: `docs/benchmark/nr3d/v3_referit3d_track_20260501.md`

For the v4 pilot, the baseline metric is recomputed only on the selected
100-sample subset. Baseline failures remain failures; they are not filtered out.

### New Run

The new run uses the same 100 sample ids and the same proposal pool, but it
changes two runtime surfaces:

- NR3D pack-prep keyframes use a non-GT keyframe mode.
- NR3D runner uses the current Stage-2 VG final guards from ScanRefer v3.20.

The primary metric is classification accuracy:

```text
selected_object_id == target_id
```

The pilot also reports Easy / Hard and View-Dep / View-Indep, using the existing
NR3D leaderboard metric implementation.

## Sample Selection

Create a durable sample-id file:

```text
tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json
```

Selection rules:

1. Load the canonical filtered NR3D fold through the existing leaderboard
   metadata path (`mentions_target_class_only=True`).
2. Inner-join with `tmp/nr3d_eval_v1_full/side_by_side.json`.
3. Select 100 sample ids by a deterministic SHA1 ordering over
   `sample_id + "nr3d_v4_agent_guards_fair_views_20260512"`.
4. Do not condition on correctness, old IoU, failure status, scene, difficulty,
   or view-dependence.
5. Persist the selected ids and the selection summary under `tmp/nr3d_artifacts/`
   so results are reproducible.

This gives a representative first read without cherry-picking hard failures or
easy successes.

## Keyframe Selection

Add a keyframe-mode switch to NR3D pack prep:

```text
--keyframe-mode gt_target | query_driven
```

`gt_target` remains the backward-compatible default for old docs and tests.
The v4 run uses `query_driven`.

`query_driven` behavior:

1. Build or reuse a `KeyframeSelector` for the scene's Phase-8 ConceptGraph
   package.
2. Call `select_keyframes_v2(query, k=3, use_visual_context=False)`.
3. Convert returned frame ids into the same sample-artifact keyframe schema used
   by the existing runner.
4. If Stage 1 returns no usable frame, fall back to scene-level frames with high
   proposal density. The fallback must not read `target_id`.

The mode may use the natural-language query and scene/proposal metadata. It must
not use the annotation target id, the target bbox, or any per-sample GT
visibility map entry.

Each prepared sample records:

```json
{
  "keyframe_mode": "query_driven",
  "keyframe_selection_uses_gt_target": false
}
```

This makes later audits cheap.

## Agent Configuration

Port the ScanRefer v3.20 guard switches to the NR3D runner:

- `--use-tool-answer-disagreement-gate`
- `--use-no-match-candidate-guard`
- `--use-evidence-frame-guard`

The pilot enables all three. The VG pack remains the same pack-v1 interface:
`list_keyframes_with_proposals`, `view_keyframe_marked`, `inspect_proposal`,
`find_proposals_by_category`, `compare_proposals_spatial`, and `submit_final`.

The guards use only the agent's own tool trace and final rationale. They do not
read GT target id, GT bbox, target visibility, or metrics.

## Runner Engineering

The first 100-sample run should reuse ScanRefer v3.20's stable execution
patterns where practical:

- Per-sample checkpoints under
  `tmp/nr3d_eval_v4_agent_guards_fair_views_random100/per_sample/`.
- Optional `--checkpoint-only` and `--max-new-samples` support if the NR3D
  runner needs resumability parity.
- Worker count starts conservatively at 8-15. The run can use 15 if RSS stays
  comfortably under 15 GB and LLM rate-limit pressure is low.
- Any long-running command runs inside tmux.
- Existing completed per-sample checkpoints are reused on resume.

For this pilot, it is acceptable to assemble `side_by_side.json` only after all
100 checkpoints complete.

## Documentation And Ingestion

After the pilot completes, write a durable benchmark record:

```text
docs/benchmark/nr3d/v4_agent_guards_fair_views_20260512.md
```

The doc must include:

- Branch and tip commit.
- Exact sample-id file and selection mechanism.
- Exact prepare and run commands.
- Baseline subset metrics from the existing v2/v3 full output.
- New subset metrics.
- Delta table: Overall, Easy, Hard, View-Dep, View-Indep, failure count.
- Caveat that this is a 100-sample pilot, not a full leaderboard row.
- Explicit no-GT-inference checklist.

Ingest the run into `docs/benchmark/nr3d/runs.sqlite` using
`scripts/ingest_nr3d_run.py`. If the ingester needs fields for keyframe mode or
subset notes, add backward-compatible nullable columns rather than replacing old
rows.

Update:

- `docs/benchmark/nr3d/README.md`
- `docs/benchmark/nr3d/leaderboard.md` only if the doc clearly labels the row as
  partial / non-leaderboard.
- `docs/benchmark/README.md` if it has an active-result table.

## Tests

Add focused tests before running the pilot:

1. NR3D pack-prep preserves legacy `gt_target` behavior by default.
2. NR3D `query_driven` sample artifacts set
   `keyframe_selection_uses_gt_target=false`.
3. The `query_driven` path does not require target-visible frames and can fall
   back from an empty Stage-1 result without reading `object_to_views[target_id]`.
4. NR3D runner propagates the three guard flags into `Stage2DeepAgentConfig`.
5. Subset metric computation scores only the fixed 100 ids and keeps failures in
   the denominator.

Also run the existing targeted tests for NR3D runner/metrics and the Stage-2 VG
guard tests.

## Success Criteria

The pilot is successful if all are true:

- 100 selected samples are fixed and reproducible from the selection artifact.
- New prepared samples do not use GT-target-visible keyframes.
- The new run completes with no unhandled batch crash.
- Results are ingested and documented under `docs/benchmark/nr3d/`.
- The doc reports both baseline subset and new subset metrics, even if the new
  result is negative.

Metric interpretation:

- `delta >= +1 pp` on Overall classification accuracy: promising; expand to
  500 samples.
- `-1 pp < delta < +1 pp`: neutral; inspect case-level failures before scaling.
- `delta <= -1 pp`: likely evidence-selection regression; audit whether
  query-driven keyframes are missing target evidence.

## Non-Goals

- Full NR3D rerun in the first implementation pass.
- 3-run voting / consensus for NR3D v4.
- Detector-pool NR3D; this remains the GT-track/classification setup.
- Parser redesign.
- Changing the public NR3D metric definition.
- Using GT target visibility as an inference-time rescue path.

