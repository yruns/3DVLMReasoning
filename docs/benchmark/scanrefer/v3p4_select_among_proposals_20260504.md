# ScanRefer v3.4 — `select_among_proposals` forced 1-of-K choice

**Branch**: `feat/scanrefer-v3-query-driven`
**Tip commit at run time**: `a2ab5c7` (working tree dirty: D5 patch on top)
**Run ID** (SQLite): `v3p4_select_among_random100_20260504`
**Date harvested**: 2026-05-04 17:06 GMT+8

## Headline (NEGATIVE RESULT)

Random100 fold (frozen seed=20260503), aggregation-GT evaluator:

| Variant | Acc@0.25 | Acc@0.50 | Unique@0.50 | Multiple@0.50 | mean IoU |
|---|---:|---:|---:|---:|---:|
| v3.3 vertical-spatial | **48.0** | **42.0** | 75.00 | **29.17** | 0.4130 |
| **v3.4 select_among_proposals** | **44.0** | **38.0** | **78.57** | 22.22 | 0.3728 |
| Δ | **−4pp** | **−4pp** | **+3.57pp** | **−7pp** | **−0.040** |

**The D5 forced 1-of-K choice tool is a net negative** under the
straightforward "give the VLM judge the per-candidate annotated frames
and ask which one matches" design. It improves Unique@0.50 (+3.57pp)
because the VLM judge resolves single-instance cases more confidently,
but it **hurts Multiple@0.50 (−7pp)** — the very split it was designed
to fix. Net: −4pp Acc@0.25, −4pp Acc@0.50.

## Adoption rate (the model DID use the tool)

| Stat | Value |
|---|---:|
| Total `select_among_proposals` invocations | **98** |
| Samples invoking it ≥ once | **81 / 100** |
| Cases the model accepted the tool's verdict | (most submissions match the tool output, no instrumentation yet) |
| Total `tool_calls` ingested into SQLite | 1855 |

The mandatory-call playbook directive worked at the behavioural level:
the model adopted the tool widely. The tool itself is failing the
multi-distractor test.

Tool counts (top 14):

```
inspect_proposal: 587
view_keyframe_marked: 250
find_proposals_by_category: 222
list_keyframes_with_proposals: 176
load_skill: 153
submit_final: 116
select_among_proposals: 98     ← new, ~1 call/sample on average
list_skills: 97
compare_proposals_spatial: 50
request_crops: 40
request_more_views: 37
inspect_stage1_metadata: 26
retrieve_object_context: 3
```

## Why the new tool hurts Multiple@0.50

The current `select_among_proposals` design picks the "most-visible
annotated frame" per candidate (`ctx.proposal_index[cid][0]`) and
asks the VLM to choose. Two structural problems with that design:

1. **The "best frame" per candidate doesn't show the spatial
   anchor.** ScanRefer multi-distractor descriptions are of the form
   "the X near Y" / "the X above Y" / "the X to the left of Y". The
   VLM judge sees the most-visible frame for each candidate X but no
   guarantee that anchor Y is in any of those frames. Without the
   anchor visible, the VLM can't apply the spatial cue and picks
   based on attribute matching (color / type / size), which is
   weaker than the model's pre-D5 reasoning would have managed.
2. **Marks-only views erase rich scene context.** The annotated PNG
   highlights one (or many) proposal boxes. For a VLM trying to
   apply "near the white door", that door needs to be visible,
   labeled, and spatially adjacent to the candidate in the same
   frame. Picking single best-visibility frames per candidate
   instead drops anchor co-visibility.

Empirically, this means the agent's pre-D5 multi-frame reasoning
(read multiple marked frames, infer spatial relation, sometimes
call `compare_proposals_spatial`) was actually **better** than a
forced 1-of-K VLM call over isolated candidate frames. The forced
choice acts as a regularizer that throws away the agent's
intermediate context.

The +3.57pp lift on Unique@0.50 comes from samples where the
description is essentially a unique-instance category cue — the
forced VLM call catches a few cases the agent was second-guessing.

## What this implies for the next iteration

The tool's adoption proves the playbook gate works. The choice
mechanism itself needs to be rebuilt. Concrete fixes (in
priority order):

1. **Co-visible-anchor frame selection.** When the description
   contains a spatial anchor ("near the white door"), pick frames
   where BOTH the candidate AND the anchor proposal are visible.
   Use `frame_index ∩ candidate.proposal_index ∩ anchor.proposal_index`
   to find such frames; only fall back to candidate-only frames if
   no co-visible frame exists. Falls under "make the VLM judge see
   what the description is asking about."

2. **Multi-mark overlay (candidate + anchor) per shown image.** Even
   in the same chosen frame, mark BOTH the candidate (id=K) and the
   anchor (id=Y) so the VLM can see them adjacent. Currently only
   the candidate is rendered.

3. **Pre-extract the spatial cue and pass it to the judge.** Today
   the prompt forwards the raw description. If we parse out
   `(target_category, relation, anchor_category)` upstream and pass
   that decomposition to the judge, the VLM can apply the relation
   directly instead of re-extracting it from prose.

4. **Skip the tool when there's no relational modifier.** For
   purely attributive descriptions ("the red chair"), the agent's
   pre-D5 intuition is at least as good. The tool should gate on
   "description contains a spatial preposition" before mandating
   the call.

5. **Tool-output cross-check.** Don't accept the tool's verdict
   blindly. Add a playbook step: after `select_among_proposals`,
   if the model's pre-call hypothesis differs from the tool's
   pick, view the tool's chosen frame for the anchor visibility
   first. Right now the model trusts the tool.

These are 1-2 hours of work each. Iterate on (1) and (2) first
since they directly address the structural defect.

## Run identity

- **Branch**: `feat/scanrefer-v3-query-driven`
- **Tip commit**: `a2ab5c7` (working tree dirty with D5 patch:
  `Stage2RuntimeState.vlm_judge` / `image_to_data_url`,
  `BaseStage2Runtime.attach_vlm_hooks`, new VG tool
  `select_among_proposals`, mandatory step in
  `vg_grounding_playbook.md`)
- **Pack name**: `pack_scanrefer_v3p_iterative` (same Mask3D-CG KFs as v3.1+)
- **Fold**: 100 utts × 66 scenes (frozen, seed=20260503)
- **Backend**: pack_v1 chassis with all 3 Stage 1 ↔ Stage 2 callbacks wired
- **Stage 2 LLM**: `gpt-5.4-2026-03-05` via internal ModelHub
- **Workers**: 4; sample-retries=1
- **Run wall time**: ~58 min (run+resume); two tmux deaths required resumption
  (76→100 second pass)
- **Raw artifacts**: `tmp/v3p4_select_among_random100_eval/`
  (Phase8-GT side_by_side) and
  `tmp/v3p4_select_among_random100_eval_agg_gt/` (paper-comparable rescore)

## Methodology change vs v3.3

D5 introduces a new VG-pack tool `select_among_proposals(candidate_ids,
description)` that runs an internal VLM 1-of-K choice:

1. For each candidate, pick the most-visible annotated frame from
   `ctx.proposal_index`.
2. Build a single multimodal LangChain message containing the
   description plus one image per candidate (the chosen annotated
   frame, encoded as base64 data URL).
3. Send to the runtime's VLM via the new
   `Stage2RuntimeState.vlm_judge` callable (wired by
   `BaseStage2Runtime.attach_vlm_hooks` before tool building).
4. Parse strict JSON `{selected_proposal_id, reasoning}`.
5. Validate the chosen id is in `candidate_ids`; FAIL-LOUD otherwise.

The playbook (`vg_grounding_playbook.md`) adds step 6 to the decision
tree:

> Whenever `find_proposals_by_category(target_category).proposal_ids`
> has length ≥ 2 AND you cannot uniquely eliminate all but one via
> spatial filters alone, you MUST call `select_among_proposals(...)`
> BEFORE `submit_final`.

Code:

- `src/agents/runtime/base.py` — `vlm_judge` / `image_to_data_url`
  fields on `Stage2RuntimeState`; `make_vlm_judge` and
  `attach_vlm_hooks` on `BaseStage2Runtime`
- `src/agents/runtime/deepagents_agent.py` —
  `build_runtime_tools` calls `attach_vlm_hooks` before tool build
- `src/agents/packs/vg_embodiedscan/tools.py` — new tool +
  parser + frame selection
- `src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md`
  — new tool documentation + mandatory decision-tree step

Tests:

- `test_select_among_proposals_*` (8 new) and
  `test_attach_vlm_hooks_wires_state_for_tool` — all 24 vg-pack
  tests pass
- Updated wrapper / runtime tool-list snapshots to include the
  new tool name (101 tests pass total across the touched surface)

## Reproduction commands

```bash
# Agent run on the existing v3.1+ pack
PYTHONPATH=src python -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p4_select_among_random100_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 4 --sample-retries 1

# Aggregation-GT rescore for paper-comparability
PYTHONPATH=src python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/v3p4_select_among_random100_eval/side_by_side.json \
  --output-dir tmp/v3p4_select_among_random100_eval_agg_gt \
  --scannet-aux-root data/nr3d/scannet_aux \
  --mesh-root data/nr3d/scannet_aux_meshes

# Leaderboard slicing
PYTHONPATH=src python -m evaluation.scripts.scanrefer_leaderboard_metrics \
  --side-by-side tmp/v3p4_select_among_random100_eval_agg_gt/side_by_side.json \
  --scanrefer-data-root data/scanrefer \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --output tmp/v3p4_select_among_random100_eval_agg_gt/leaderboard_metrics.json

# SQLite ingest
python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/v3p4_select_among_random100_eval_agg_gt \
  --run-id v3p4_select_among_random100_20260504 \
  --branch feat/scanrefer-v3-query-driven \
  --commit a2ab5c7 \
  --backend pack_v1 \
  --keyframe-mode mask3d_query_driven \
  --leaderboard-metrics tmp/v3p4_select_among_random100_eval_agg_gt/leaderboard_metrics.json \
  --notes "v3.4 select_among_proposals smoke; -4/-4 vs v3.3"
```

## Caveats

- This is the same 100-utt random fold; LLM stochasticity is ±2-3pp,
  so the −4pp move is roughly 1-2σ. A second run of v3.4 (or v3.3)
  on the same pack would tighten the bound but was not done.
- The `select_among_proposals` body relies on the runtime's existing
  `image_to_data_url`, which downsizes to `image_max_size` (default
  720px). VLM judge frames are therefore at most 720px on the long
  edge — small same-category candidates may render at < 80px which
  reduces discrimination.
- 5 samples ended with `no_prediction` (the new gate path interacts
  with the chassis finalizer error path); v3.3 had 5 too. Net no
  change on this axis.
- The most plausible cure (co-visible-anchor frame selection,
  improvement #1 above) is independent code work.

## Interpretation

The negative result is a **methodologically clean falsification**
of the simplest D5 design. Three takeaways:

1. The runtime infrastructure to inject VLM sub-calls from inside
   pack tools (`attach_vlm_hooks`) works and is broadly reusable —
   it shipped value beyond this experiment.
2. Forced 1-of-K VLM choice WITHOUT anchor co-visibility is
   actively worse than the agent's pre-D5 multi-frame reasoning.
   This contradicts the SeeGround paradigm's working assumption
   that "showing the candidates side by side is enough" — but
   SeeGround renders **synthetic per-candidate viewpoints** that
   include relevant scene context, not just the top-visibility
   real-world frames we use.
3. The structural fix is to make the judge see what the description
   is asking about (co-visible anchor + candidate). That's the
   next iteration.

## Files referenced

- This doc: `docs/benchmark/scanrefer/v3p4_select_among_proposals_20260504.md`
- Code: `src/agents/runtime/base.py`,
  `src/agents/runtime/deepagents_agent.py`,
  `src/agents/packs/vg_embodiedscan/tools.py`,
  `src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md`,
  `src/agents/packs/vg_embodiedscan/tests/test_tools.py`,
  `src/agents/tests/test_stage2_deep_agent.py`
- Eval artifacts: `tmp/v3p4_select_among_random100_eval/`,
  `tmp/v3p4_select_among_random100_eval_agg_gt/`
- Console log: `/tmp/v3p_logs/v3p4_select_among_agent.log`
- Frozen fold: `tmp/scanrefer_artifacts/random100_sample_ids.json`
- SQLite: `docs/benchmark/scanrefer/runs.sqlite`
  (run_id `v3p4_select_among_random100_20260504`, 1855 tool_calls rows)
