# VG Clean Initial Marked-On-Demand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make pack-v1 VG initial evidence use clean RGB images plus a text-only left-to-right `#id category` proposal inventory, while keeping marked images available only through `view_keyframe_marked`.

**Architecture:** Keep proposal generation, visibility, and Stage-1 selection unchanged. Change sample keyframe normalization to preserve raw RGB paths, add a runtime VG initial inventory formatter from `vg_proposal_pool`, and keep marked-frame tool injection through `vg_pending_images`.

**Tech Stack:** Python, Pydantic Stage-2 bundle models, LangChain/DeepAgents runtime prompt assembly, pytest.

---

### Task 1: Preserve Raw Initial Keyframe Paths

**Files:**
- Modify: `src/evaluation/scripts/prepare_pack_v1_inputs.py`
- Modify: `src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py`
- Modify: `src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py`
- Modify: `src/evaluation/scripts/prepare_detector_pack_inputs.py`
- Test: `src/evaluation/scripts/tests/test_prepare_pack_v1_inputs.py`
- Test: `src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d.py`
- Test: `src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer.py`
- Test: `src/evaluation/scripts/tests/test_prepare_detector_pack_inputs.py`

- [ ] **Step 1: Write failing tests**

Add tests that create an annotated `frame_<id>.png` next to a raw RGB path and assert normalized keyframes still point to raw RGB paths.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest \
  src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d.py \
  src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer.py \
  -q
```

Expected before implementation: at least one assertion fails because `image_path` still points to `annotated/frame_<id>.png`.

- [ ] **Step 3: Implement minimal normalization change**

Change `normalize_prepared_keyframes(...)` so `image_path` is always the raw incoming `keyframe["image_path"]`. Keep `frame_id` and `keyframe_idx` unchanged. Leave annotated rendering and `vg_proposal_pool.annotated_image_dir` untouched.

- [ ] **Step 4: Rerun focused tests**

Run the same pytest command. Expected: tests pass.

### Task 2: Add Text-Only Initial VG Proposal Inventory

**Files:**
- Modify: `src/agents/runtime/base.py`
- Modify: `src/agents/runtime/deepagents_agent.py`
- Modify: `src/agents/packs/vg_embodiedscan/tools.py`
- Test: `src/agents/tests/test_stage2_deep_agent.py`
- Test: `src/agents/packs/vg_embodiedscan/tests/test_tools.py`

- [ ] **Step 1: Write failing runtime prompt test**

Add a test that builds a VG bundle with clean keyframe paths and a `vg_proposal_pool` containing proposals and frame views. Assert the initial prompt contains `Initial frame proposal inventory`, `frame_id=<id>`, and left-to-right `#<id> <category>` entries.

- [ ] **Step 2: Assert forbidden initial text is absent**

In the same test, assert the initial prompt does not include `bbox_2d`, `visibility score`, or `annotated_image`.

- [ ] **Step 3: Implement inventory formatter**

Add a helper that reads `extra_metadata["vg_proposal_pool"]["frame_index"]` and `["proposals"]`, orders visible proposals by frame-specific 2D center x when available, and returns text-only lines:

```text
## Initial frame proposal inventory
frame_id=69 left_to_right: #12 monitor, #31 keyboard, #4 chair
```

- [ ] **Step 4: Inject inventory into VG initial prompt**

Call the helper in `DeepAgentsStage2Runtime.build_task_message(...)` only for `Stage2TaskType.VISUAL_GROUNDING`.

- [ ] **Step 5: Make list tool match clean inventory**

Update `list_keyframes_with_proposals()` so it no longer returns `annotated_image` and returns `left_to_right` entries with `#id category`.

- [ ] **Step 6: Rerun focused tests**

Run:

```bash
.venv/bin/python -m pytest \
  src/agents/tests/test_stage2_deep_agent.py \
  src/agents/packs/vg_embodiedscan/tests/test_tools.py \
  -q
```

Expected: tests pass.

### Task 3: Verify Marked-On-Demand Still Works

**Files:**
- Test: `src/agents/packs/vg_embodiedscan/tests/test_tools.py`
- Test: `src/agents/tests/test_stage2_deep_agent.py`

- [ ] **Step 1: Confirm marked tool queue behavior**

Keep or update tests asserting `view_keyframe_marked(frame_id)` appends `annotated/frame_<frame_id>.png` to `vg_pending_images`.

- [ ] **Step 2: Confirm follow-up image injection**

Keep or update tests asserting pending marked images are injected in the next follow-up message and then drained.

- [ ] **Step 3: Rerun combined focused suite**

Run:

```bash
.venv/bin/python -m pytest \
  src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d.py \
  src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer.py \
  src/agents/packs/vg_embodiedscan/tests/test_tools.py \
  src/agents/tests/test_stage2_deep_agent.py \
  -q
```

Expected: tests pass.

### Task 4: Final Checks

**Files:**
- Modify if needed: `docs/superpowers/specs/2026-05-14-vg-clean-initial-marked-on-demand-design.md`

- [ ] **Step 1: Static diff check**

Run:

```bash
git diff --check HEAD
```

Expected: no output.

- [ ] **Step 2: Review final diff**

Run:

```bash
git diff --stat HEAD
git diff HEAD -- src/evaluation/scripts src/agents docs/superpowers
```

Expected: changes are limited to the plan, normalization, VG prompt/tool text, and focused tests.

- [ ] **Step 3: Commit implementation**

Run:

```bash
git add docs/superpowers/plans/2026-05-14-vg-clean-initial-marked-on-demand.md src/evaluation/scripts src/agents
git commit -m "Implement VG clean initial evidence flow"
```

Expected: one implementation commit on top of the Chinese spec commit.
