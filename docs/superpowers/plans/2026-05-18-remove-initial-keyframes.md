# Remove Initial Keyframes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove every Stage-2 initial-keyframe data path so first-person frames enter the agent only through selector/mark/crop pending-image queues.

**Architecture:** Delete `KeyframeEvidence` and `Stage2EvidenceBundle.keyframes` from the public model, then repair runtime/tools/runners around `extra_metadata["vg_pending_images"]` and frame metadata. Keep low-level `query_scene.keyframe_selector.KeyframeSelector` as the private implementation behind `select_by_text`, but expose it to Stage 2 as `text_frame_selector`.

**Tech Stack:** Python 3.12, Pydantic, DeepAgents/LangChain tools, pytest, ruff, `rg` omission scans.

---

### Task 1: Core Model And Runtime Contract

**Files:**
- Modify: `src/agents/core/task_types.py`
- Modify: `src/agents/core/__init__.py`
- Modify: `src/agents/models.py`
- Modify: `src/agents/__init__.py`
- Modify: `src/agents/core/agent_config.py`
- Modify: `src/agents/runtime/base.py`
- Modify: `src/agents/runtime/deepagents_agent.py`
- Modify: `src/agents/stage2_deep_agent.py`
- Test: `src/agents/runtime/tests/test_no_initial_keyframes.py`

- [ ] **Step 1: Write failing tests**

Create `src/agents/runtime/tests/test_no_initial_keyframes.py` with:

```python
from pathlib import Path

import pytest

from agents.core.agent_config import Stage2DeepAgentConfig, Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle, Stage2TaskSpec
from agents.runtime.base import Stage2RuntimeState
from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime


def test_stage2_evidence_bundle_has_no_keyframes_field() -> None:
    bundle = Stage2EvidenceBundle(scene_id="scene")

    assert "keyframes" not in Stage2EvidenceBundle.model_fields
    assert not hasattr(bundle, "keyframes")


def test_config_has_no_seed_keyframe_restore_flag() -> None:
    assert "restore_stage1_seed_keyframe_drain" not in Stage2DeepAgentConfig.model_fields


def test_runtime_state_has_no_initial_keyframe_snapshot() -> None:
    runtime = Stage2RuntimeState(bundle=Stage2EvidenceBundle(scene_id="scene"))

    assert not hasattr(runtime, "initial_keyframe_paths")


def test_initial_message_attaches_only_bev(tmp_path: Path) -> None:
    from PIL import Image

    bev = tmp_path / "bev.png"
    rgb = tmp_path / "rgb.png"
    Image.new("RGB", (8, 8), "white").save(bev)
    Image.new("RGB", (8, 8), "red").save(rgb)
    bundle = Stage2EvidenceBundle(
        scene_id="scene",
        bev_image_path=str(bev),
        extra_metadata={
            "scene_catalog": {
                "scene_id": "scene",
                "scene_category": "room",
                "total_frames": 1,
                "frame_id_range": [0, 0],
                "valid_frame_ids": [0],
                "bev_image_path": str(bev),
                "proposals": [],
            },
            "vg_pending_images": [str(rgb)],
        },
    )
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        user_query="chair",
    )
    rt = DeepAgentsStage2Runtime(
        config=Stage2DeepAgentConfig(enable_stage1_text_retrieval=False)
    )
    state = Stage2RuntimeState(bundle=bundle)

    message = rt.build_user_message(task, state)

    image_parts = [p for p in message.content if p.get("type") == "image_url"]
    assert len(image_parts) == 1
    assert str(bev) in state.seen_image_paths
    assert str(rgb) not in state.seen_image_paths


def test_evidence_update_drains_pending_images_only(tmp_path: Path) -> None:
    from PIL import Image

    pending = tmp_path / "pending.png"
    Image.new("RGB", (8, 8), "blue").save(pending)
    bundle = Stage2EvidenceBundle(
        scene_id="scene",
        extra_metadata={"vg_pending_images": [str(pending)]},
    )
    rt = DeepAgentsStage2Runtime(
        config=Stage2DeepAgentConfig(enable_stage1_text_retrieval=False)
    )
    state = Stage2RuntimeState(bundle=bundle)

    message = rt.build_evidence_update_message(state)

    assert message is not None
    assert str(pending) in state.seen_image_paths
    assert state.bundle.extra_metadata["vg_pending_images"] == []
```

- [ ] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/runtime/tests/test_no_initial_keyframes.py -q
```

Expected: failure because `keyframes`, `KeyframeEvidence`, `initial_keyframe_paths`, and the restore flag still exist.

- [ ] **Step 3: Implement model/runtime cut**

Delete `KeyframeEvidence`, remove `Stage2EvidenceBundle.keyframes`, remove public exports, remove restore flag, rename Stage-2 `keyframe_selector` fields/args to `text_frame_selector`, and rewrite `build_evidence_update_message()` to drain only `vg_pending_images`.

- [ ] **Step 4: Verify green**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/runtime/tests/test_no_initial_keyframes.py -q
```

Expected: pass.

### Task 2: Selector Queue Metadata And Crop Tool

**Files:**
- Modify: `src/agents/runtime/scene_runtime.py`
- Modify: `src/agents/tools/selectors.py`
- Modify: `src/agents/tools/mark_frame_with_bbox.py`
- Modify: `src/agents/tools/request_crops.py`
- Test: `src/agents/tools/tests/test_no_keyframe_mutation.py`

- [ ] **Step 1: Write failing tests**

Create tests that assert selector/mark helpers add `vg_pending_image_metadata` and that `CropBackend.process_requests()` returns crops in `vg_pending_images` without mutating `bundle.keyframes`.

- [ ] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_no_keyframe_mutation.py -q
```

Expected: failure while crop code still reads `bundle.keyframes`.

- [ ] **Step 3: Implement queue contract**

Add a metadata-aware queue helper, update selectors and mark tool to pass source/frame metadata, and rewrite crop backend to resolve frame image paths from `vg_pending_image_metadata`, `seen_image_paths`, or catalog frame views instead of `bundle.keyframes`.

- [ ] **Step 4: Verify green**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_no_keyframe_mutation.py src/agents/tools/tests/test_selectors_text.py -q
```

Expected: pass.

### Task 3: VG Pack Context And Tool Inventories

**Files:**
- Modify: `src/agents/packs/vg_embodiedscan/ctx.py`
- Modify: `src/agents/packs/vg_embodiedscan/tools.py`
- Modify: `src/agents/packs/vg_embodiedscan/registration.py`
- Modify: `src/agents/packs/*/skills/**`
- Test: `src/agents/packs/vg_embodiedscan/tests/test_tools.py`
- Test: `src/agents/packs/*/skills/tests/*`

- [ ] **Step 1: Update tests first**

Replace bundle-keyframe expectations with selected-frame / marked-frame expectations.

- [ ] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/packs/vg_embodiedscan/tests src/agents/packs/vg_embodiedscan/skills/tests src/agents/packs/qa_default/skills/tests -q
```

Expected: failures from old keyframe inventory APIs.

- [ ] **Step 3: Implement pack cleanup**

Make `cumulative_seen_frame_ids()` count only successful selector and mark tool traces; remove `format_keyframe_proposal_inventory`; replace user-facing keyframe wording with frame/evidence wording.

- [ ] **Step 4: Verify green**

Run the same pytest command. Expected: pass.

### Task 4: Pack Prep And Runners

**Files:**
- Modify: `src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py`
- Modify: `src/evaluation/scripts/prepare_pack_v1_inputs.py`
- Modify: `src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py`
- Modify: `src/evaluation/scripts/prepare_detector_pack_inputs.py`
- Modify: `src/evaluation/scripts/run_nr3d_vg_side_by_side.py`
- Modify: `src/evaluation/scripts/run_scanrefer_vg_side_by_side.py`
- Modify: `src/evaluation/scripts/run_embodiedscan_vg_side_by_side.py`
- Test: `src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d.py`
- Test: `src/evaluation/scripts/tests/test_run_nr3d_vg_side_by_side.py`

- [ ] **Step 1: Update tests first**

Assert prepared sample JSON lacks `keyframes`, `keyframe_mode`, `keyframe_selection_uses_gt_target`, and `keyframe_selection_used_fallback`; assert NR3D runner accepts a sample without `keyframes`.

- [ ] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d.py src/evaluation/scripts/tests/test_run_nr3d_vg_side_by_side.py -q
```

Expected: failure while prep still writes and runner still requires keyframes.

- [ ] **Step 3: Implement runner/prep cut**

Stop writing sample keyframe fields, remove keyframe-mode CLI args where current runners expose them, and call `build_pack_v1_bundle()` without a `keyframes` argument.

- [ ] **Step 4: Verify green**

Run the same pytest command. Expected: pass.

### Task 5: Adapters, Trace, Reports, Current Docs

**Files:**
- Modify: `src/agents/stage1_adapters.py`
- Modify: `src/agents/benchmark_adapters.py`
- Modify: `src/agents/space3d_bench_adapter.py`
- Modify: `src/agents/adapters*/`
- Modify: `src/agents/examples/*`
- Modify: `src/agents/trace.py`
- Modify: `src/agents/trace_server.py`
- Modify: `src/evaluation/trace_html.py`
- Modify: `src/evaluation/trace_integration.py`
- Modify: `scripts/generate_eval_report.py`
- Modify: `scripts/ingest_*run.py`
- Modify: `docs/00_research_manifest.md`, `docs/01_overview.md`, `docs/02_architecture.md`, `docs/04_stage2_agent.md`, `docs/05_evaluation.md`, `docs/09_gotchas.md`

- [ ] **Step 1: Update tests first**

Adjust adapter/trace tests so they no longer instantiate `KeyframeEvidence` or assert initial/final keyframe counts.

- [ ] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tests src/evaluation/tests src/evaluation/scripts/tests -q
```

Expected: failures from old bundle shape and old trace/report fields.

- [ ] **Step 3: Implement compatibility cleanup**

Rewrite adapters to place selected-frame metadata into `extra_metadata`; rewrite trace/report/ingest fields to show initial BEV and tool-acquired images; mark old one-shot keyframe baselines unsupported where preserving them would reintroduce initial frame injection.

- [ ] **Step 4: Verify green**

Run the same pytest command. Expected: pass or only explicitly unsupported legacy baseline tests remain updated to assert the clear error.

### Task 6: Omission Scans And Broad Verification

**Files:**
- Modify any remaining files reported by scans.

- [ ] **Step 1: Run forbidden-symbol scans**

```bash
rg -n "initial_keyframe_paths|restore_stage1_seed_keyframe_drain|KeyframeEvidence|bundle\\.keyframes|\\.keyframes|sample\\[\"keyframes\"\\]|sample\\.get\\(\"keyframes\"|keyframe_mode|keyframe_selection_|select_keyframes_for_sample|select_keyframes_from_phase8_target|normalize_prepared_keyframes" src/agents src/evaluation scripts tests
rg -n "keyframe_selector|initial keyframe|initial_keyframe|seed keyframe|seed_keyframe|Current keyframes|Newly added keyframes|viewed 0 keyframes|list_keyframes_with_proposals|view_keyframe" src/agents src/evaluation scripts tests
rg -n "Stage 2.*keyframes|initial keyframes|bundle\\.keyframes|KeyframeEvidence|view_keyframe|list_keyframes_with_proposals" docs/00_research_manifest.md docs/01_overview.md docs/02_architecture.md docs/04_stage2_agent.md docs/05_evaluation.md docs/09_gotchas.md
```

Expected: zero matches except explicitly allowed Stage-1 internals outside these current Stage-2/evaluation paths.

- [ ] **Step 2: Run focused tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  src/agents/runtime/tests \
  src/agents/tools/tests \
  src/agents/packs/vg_embodiedscan/tests \
  src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d.py \
  src/evaluation/scripts/tests/test_run_nr3d_vg_side_by_side.py -q
```

Expected: pass.

- [ ] **Step 3: Run broad tests and lint**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/ -q
ruff check src/
```

Expected: pass, or any remaining failures are documented as intentionally unsupported legacy baselines with tests asserting clear errors.
