# Remove Initial Keyframes From Stage-2 Runtime

## Summary

Hard-delete the initial-keyframe path from the Stage-2 runtime and all
benchmark adapters. The agent must never receive first-person RGB frames from
pack preparation, sample JSON, or bundle construction. The only runtime path
for first-person evidence is active tool use: `select_by_*`,
`mark_frame_with_bbox`, and `request_crops` queue images that are then injected
on the next evidence-update turn. BEV remains allowed as non-first-person
scene context through `bundle.bev_image_path` or `view_bev`.

This implements the v9 catalog-first contract that was already specified in
`docs/superpowers/specs/2026-05-14-v9-catalog-first-scene-exploration-design.md`:
pack prep writes scene-level catalog/BEV artifacts, not per-sample keyframes.

## Motivation

The current implementation still carries compatibility state from the old
Stage-1 -> Stage-2 handoff:

- `Stage2EvidenceBundle.keyframes`
- `KeyframeEvidence`
- `Stage2RuntimeState.initial_keyframe_paths`
- `Stage2DeepAgentConfig.restore_stage1_seed_keyframe_drain`
- Stage-2-facing names such as `runtime.keyframe_selector`
- runner checks that sample JSON has non-empty `keyframes`
- pack-prep code that writes `keyframes`
- runtime evidence-update logic that can drain `bundle.keyframes`
- tests, trace pages, examples, and prompt copy that still talk about initial
  keyframes

Those compatibility seams enabled the v9.4-D leak reproduction: GT-target-
visible pack-prep frames could be auto-injected into agent context. The right
fix is not another filter. The right fix is removing the data path entirely.

## Scope

This is a whole-repo API cleanup, not only an NR3D runner cleanup.

In scope:

- Core models and runtime under `src/agents/`.
- Stage-2 runtime compatibility wrapper.
- VG pack context/tools/finalizers where they read bundle keyframes.
- NR3D, ScanRefer, EmbodiedScan, detector, OpenEQA, SQA3D, Space3D adapters
  and scripts that construct `Stage2EvidenceBundle`.
- Trace server and trace HTML fields that display initial/final keyframes.
- Tests and examples that instantiate `KeyframeEvidence` or inspect
  `bundle.keyframes`.
- Documentation and prompt copy for current v9 agent behavior, including
  current architecture docs that still claim Stage 2 receives initial
  keyframes.

Out of scope:

- The lower-level Stage-1 `KeyframeSelector.select_keyframes_v2` algorithm and
  its tests. That function is still valid as the implementation behind
  `select_by_text`. It may be imported by the selector tool adapter, but the
  Stage-2 public/runtime surface must not expose a `keyframe_selector` field or
  any initial-keyframe concept.
- Historical markdown benchmark records. They may describe old runs with the
  word "keyframe"; do not rewrite old evidence unless a current command or doc
  is misleading users about the new runtime.
- Dataset files already present under `data/` and old `tmp/` outputs. New prep
  code must not write `keyframes`, but existing artifacts do not need in-place
  migration unless a test fixture depends on them.

## Pre-Implementation Inventory

A broad preflight scan on this branch currently finds keyframe-related matches
in 101 files under `src/agents`, `src/evaluation`, `scripts`, and `tests`.
Implementation must not treat the first clean pass through `src/agents` as
complete; the following groups are known touch points:

- Core schema/runtime: `src/agents/core/task_types.py`,
  `src/agents/core/agent_config.py`, `src/agents/runtime/base.py`,
  `src/agents/runtime/deepagents_agent.py`, `src/agents/stage2_deep_agent.py`,
  `src/agents/models.py`, and package exports.
- Legacy callback/crop path: `src/agents/stage1_callbacks.py`,
  `src/agents/tools/request_crops.py`, and their tests.
- Selector-tool naming: `src/agents/tools/selectors.py` and tests currently
  use `runtime.keyframe_selector`; Stage-2-facing names must become neutral.
- VG pack context/tools/playbooks: `src/agents/packs/vg_embodiedscan/*`,
  `src/agents/packs/qa_default/*`, and skill consistency tests.
- Benchmark adapters and examples: `src/agents/benchmark_adapters.py`,
  `src/agents/stage1_adapters.py`, `src/agents/space3d_bench_adapter.py`,
  `src/agents/adapters*/`, and `src/agents/examples/*`.
- Pack prep and runners: `src/evaluation/scripts/prepare_*pack*`,
  `src/evaluation/scripts/run_*oneshot.py`,
  `src/evaluation/scripts/run_*stage2_full.py`, and
  `src/evaluation/scripts/run_*vg_side_by_side.py`.
- Trace/report/ingest support: `src/agents/trace.py`,
  `src/agents/trace_server.py`, `src/evaluation/trace_html.py`,
  `src/evaluation/trace_integration.py`, `scripts/generate_*trace*.py`,
  `scripts/generate_eval_report.py`, and benchmark ingesters that persist
  initial/final keyframe counts.
- Legacy analysis scripts: `scripts/evaluate_keyframe_funnel.py`,
  `scripts/run_migration_scorecard.py`,
  `scripts/prepare_nr3d_rerender_pack_from_existing.py`, and old launchers
  that advertise keyframe-mode runs.
- Active docs: current architecture/agent docs under `docs/` must be updated
  when they describe present behavior. Historical benchmark logs,
  superseded superpowers specs/plans, and migration evidence may remain as
  historical records.

## Target Contract

### Data Model

`Stage2EvidenceBundle` becomes scene-context only:

```python
class Stage2EvidenceBundle(BaseModel):
    scene_id: str = ""
    stage1_query: str = ""
    bev_image_path: str | None = None
    scene_summary: str = ""
    object_context: dict[str, str] = Field(default_factory=dict)
    hypothesis: Stage1HypothesisSummary | None = None
    extra_metadata: dict[str, Any] = Field(default_factory=dict)
```

`KeyframeEvidence` is removed from the public agent model exports. New visual
evidence is represented as pending image paths plus frame metadata in
`extra_metadata`, not as typed keyframes in the bundle.

Stage-2 constructor/runtime naming must also move away from keyframe concepts:

- Replace `keyframe_selector` arguments and runtime fields with a neutral name
  such as `frame_selector` or `text_frame_selector`.
- Keep `query_scene.keyframe_selector.KeyframeSelector` only as a private
  implementation detail behind `select_by_text`.
- Do not expose `initial_keyframe`, `seed_keyframe`, or `keyframe_selector`
  names in agent config, runtime state, runner APIs, prompt text, or trace
  schemas.

Required runtime image queues:

- `extra_metadata["vg_pending_images"]`: paths queued by v9 tools.
- `extra_metadata["vg_pending_image_metadata"]`: optional parallel metadata
  records keyed by image path, frame id, source tool, and selected reason.

The metadata key is additive and should be best-effort. Absence of metadata
must not block image injection.

### Runtime

Initial agent message:

- May include the BEV image from `bundle.bev_image_path`.
- Must not include first-person RGB frames from sample JSON or bundle fields.

Evidence update:

- Drains only explicit pending queues (`vg_pending_images`) and optional BEV
  re-injection requests.
- Does not iterate over `runtime.bundle.keyframes`.
- Does not need `initial_keyframe_paths`, because there are no initial
  keyframes to filter.
- Removes `restore_stage1_seed_keyframe_drain`; the leak reproduction flag is
  deleted, not kept hidden.

Text/prompt copy:

- Replace user-facing "keyframes" wording with "first-person frames",
  "selected frames", or "visual evidence".
- `select_by_text` can still mention Stage-1 language-to-frame retrieval,
  because that describes the internal selector, not an initial bundle field.

### Pack Prep And Runner

Prepared sample JSON for current pack-v1 flows must not write:

- `keyframes`
- `keyframe_mode`
- `keyframe_selection_uses_gt_target`
- `keyframe_selection_used_fallback`

Prepared sample JSON should write:

- `sample_id`
- `scene_id`
- target/scoring fields where needed
- `query`
- `scene_artifacts_dir`
- `scene_catalog_path`
- `bev_image_path`
- `camera_trajectory_path`

Runners must not require a non-empty keyframe list. Bundle construction should
pass scene catalog, BEV, visibility, proposal pool, and trajectory only.

### Selectors And Tool-Driven Evidence

The only first-person frame acquisition interface exposed to the agent is the
tool set:

- `select_by_text`
- `select_by_proposal`
- `select_by_frame_neighbor`
- `select_by_region`
- `select_by_coverage`
- `mark_frame_with_bbox`
- `request_crops`

Implementation detail:

- Selectors may continue to return text payloads that include `frame_id`,
  `image_path`, visible proposal ids, camera pose, and reason.
- If a selector should cause the frame image to be shown, it must queue the
  path in `vg_pending_images`; it must not append to `bundle.keyframes`.
- `request_crops` must return/update pending-image queues, not append
  `KeyframeEvidence`.

## Migration Plan

### Phase 1: Core Schema And Runtime Cut

1. Delete `KeyframeEvidence` from `src/agents/core/task_types.py`.
2. Remove `keyframes` from `Stage2EvidenceBundle`.
3. Remove `KeyframeEvidence` exports from `src/agents/models.py` and
   `src/agents/__init__.py`.
4. Remove `initial_keyframe_paths` and `restore_stage1_seed_keyframe_drain`
   from runtime state/config.
5. Rename Stage-2 `keyframe_selector` constructor/runtime plumbing to
   `frame_selector` or `text_frame_selector`, preserving only the private
   low-level import of `query_scene.KeyframeSelector`.
6. Rewrite `build_evidence_update_message()` so it only drains pending-image
   queues.
7. Update runtime logs and prompts to avoid initial-keyframe terminology.

### Phase 2: VG Pack And Selector Queue Contract

1. Update `build_ctx_from_bundle()` to derive visible/viewed frames from
   `seen_image_paths`, `vg_pending_image_metadata`, and scene catalog data,
   not `bundle.keyframes`.
2. Update selector tools to queue metadata consistently.
3. Update `request_crops` to return pending-image updates without
   `KeyframeEvidence`.
4. Delete or rewrite helper functions whose only job is formatting keyframe
   inventories.

### Phase 3: Pack Prep And Runners

1. NR3D prep: stop selecting/writing sample-level keyframes.
2. ScanRefer/EmbodiedScan/detector prep: same contract.
3. NR3D/ScanRefer/EmbodiedScan runners: remove `sample["keyframes"]` parsing
   and non-empty checks.
4. `build_pack_v1_bundle()` and tests: remove `keyframes` parameter.
5. Existing `data/` artifacts may still contain old keys, but new code must
   ignore them.

### Phase 4: Legacy Scripts, Trace, Tests

1. Rewrite one-shot and stage1-only scripts that evaluate "VLM over retrieved
   keyframes" into selector-driven smoke tools, or mark them unsupported if
   their purpose is exclusively the removed baseline.
2. Update trace server/schema to display initial BEV and tool-acquired images,
   not initial/final keyframe counts.
3. Update tests to instantiate bundles without `keyframes`.
4. Delete `test_evidence_update_seed_drain.py`; replace it with a regression
   test proving that no runtime/config field can re-enable seed drain.

## Strict Omission Checks

Implementation is not complete until all checks below pass.

### Forbidden Runtime Symbols

These must have zero matches in current Stage-2 runtime, evaluation, support
scripts, and tests:

```bash
rg -n "initial_keyframe_paths|restore_stage1_seed_keyframe_drain|KeyframeEvidence|bundle\\.keyframes|\\.keyframes|sample\\[\"keyframes\"\\]|sample\\.get\\(\"keyframes\"|keyframe_mode|keyframe_selection_|select_keyframes_for_sample|select_keyframes_from_phase8_target|normalize_prepared_keyframes" src/agents src/evaluation scripts tests
```

Stage-2-facing keyframe terminology must also be gone:

```bash
rg -n "keyframe_selector|initial keyframe|initial_keyframe|seed keyframe|seed_keyframe|Current keyframes|Newly added keyframes|viewed 0 keyframes|list_keyframes_with_proposals|view_keyframe" src/agents src/evaluation scripts tests
```

Current user-facing docs must not describe initial Stage-2 keyframes as present
behavior:

```bash
rg -n "Stage 2.*keyframes|initial keyframes|bundle\\.keyframes|KeyframeEvidence|view_keyframe|list_keyframes_with_proposals" docs/00_research_manifest.md docs/01_overview.md docs/02_architecture.md docs/04_stage2_agent.md docs/05_evaluation.md docs/09_gotchas.md
```

Allowed exceptions:

- Historical markdown under `docs/benchmark/**`, superseded
  `docs/superpowers/specs/**`, and `docs/superpowers/plans/**`.
- Stage-1 selector internals or tests that use "keyframe" to name the
  retrieval algorithm, not bundle-provided initial evidence.
- Stage-1-only examples under `src/query_scene/**` that demonstrate the
  retrieval algorithm without constructing a Stage-2 bundle.
- A private import path or type alias that references
  `query_scene.keyframe_selector.KeyframeSelector` inside the `select_by_text`
  adapter is allowed only if no public argument, config field, runtime field,
  prompt text, or trace schema uses the keyframe name.
- Archived/unsupported scripts if they are explicitly marked unsupported and
  are excluded from current CLI/test paths.

### Forbidden New Prepared Sample Keys

Current prep tests must assert new sample JSON does not contain:

```text
keyframes
keyframe_mode
keyframe_selection_uses_gt_target
keyframe_selection_used_fallback
```

### Positive Invariant Tests

Add or update tests proving:

- Constructing `Stage2EvidenceBundle(...)` has no `keyframes` attribute.
- Initial user message includes BEV only.
- Evidence update injects images only from `vg_pending_images`.
- NR3D runner accepts sample JSON with no `keyframes`.
- NR3D prep writes sample JSON with no `keyframes`.
- No config flag can restore seed-keyframe drain.
- `request_crops` and each selector queue images without mutating a bundle
  keyframe list.

### Verification Commands

Run at minimum:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  src/agents/runtime/tests \
  src/agents/tools/tests \
  src/agents/packs/vg_embodiedscan/tests \
  src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d.py \
  src/evaluation/scripts/tests/test_run_nr3d_vg_side_by_side.py -q

PYTHONPATH=src .venv/bin/python -m pytest src/agents/tests src/evaluation/scripts/tests -q

PYTHONPATH=src .venv/bin/python -m pytest src/ -q

ruff check src/
```

If full `src/` tests expose legacy scripts whose only purpose is the removed
one-shot keyframe baseline, either rewrite the scripts to the new selector
contract or mark them unsupported with tests that assert a clear error.

## Risks And Decisions

- This is intentionally breaking. Compatibility with old sample JSON `keyframes`
  is not preserved.
- Old benchmark outputs remain auditable, but rerunning old one-shot baselines
  may require checking out the historical commit.
- The term "keyframe" may remain inside the Stage-1 retrieval implementation
  (`KeyframeSelector`) because the algorithm still selects frames. It must not
  appear as an initial Stage-2 evidence field.
- `select_by_text` remains valid. Removing initial keyframes does not mean
  removing query-driven frame retrieval; it means making retrieval agent-
  initiated and tool-mediated.

## Acceptance Criteria

The feature is done when:

1. `Stage2EvidenceBundle` has no keyframe field and no public
   `KeyframeEvidence` model exists.
2. No Stage-2 runtime state can snapshot, filter, drain, or restore initial
   keyframes.
3. Current pack-prep code writes no sample-level keyframe fields.
4. Current VG runners run from sample JSON that has no keyframes.
5. Agent-visible first-person frames enter only through explicit tool queues.
6. Strict omission scans are documented in the implementation final response
   with any remaining matches explained as allowed exceptions.
7. Focused and broad test commands above pass, or every remaining failure is
   tied to an intentionally unsupported legacy baseline with a clear migration
   decision.
