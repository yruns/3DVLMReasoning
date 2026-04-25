# Plan C — Legacy `vg_*` 字段与 legacy VG 分支清理

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** spec 第 10 节 step 9。删除 `Stage2RuntimeState` 的
`vg_scene_objects` / `vg_axis_align_matrix` /
`vg_selected_object_id` / `vg_selected_bbox_3d` / `vg_selection_rationale`
字段,迁移 `select_object.py` / `spatial_compare.py` /
`stage2_deep_agent.py` 中的消费方,并删除
`deepagents_agent.py:266-308` 的 legacy VG 分支。`vg_backend` 默认值
从 `'legacy'` 切换到 `'pack_v1'`。

**Architecture:** 纯删除 + 配置默认翻转。所有写入新字段的代码已经在
Plan A 落地。本 plan 收敛掉读取旧字段的 5 个调用点。

**Tech Stack:** 同 Plan A/B。

**触发条件:** Plan B step 8 OpenEQA 5-sample smoke 通过 + Plan A
side-by-side 在更大规模(≥ 100 sample)的复测里 pack-v1 仍 ≥ legacy。

**Source spec:** `docs/superpowers/specs/2026-04-25-stage2-multi-task-agent-design.md`

**粒度说明:** 中粒度。删除性 PR 风险点是"是否真的没人在读了"。每个
task 把 grep 命令明确写出,执行者跑完才能放心删。

---

## File Structure

修改:

- `src/agents/runtime/base.py:43-49` 删 5 个 `vg_*` 字段
- `src/agents/runtime/deepagents_agent.py:266-308` 删 legacy VG 分支
- `src/agents/runtime/deepagents_agent.py:570-588` `vg_backend != "pack_v1"` 分支删
- `src/agents/tools/select_object.py:75,91` 改读 `runtime.task_ctx`
- `src/agents/tools/spatial_compare.py:58-62` 改读 `runtime.task_ctx`
- `src/agents/stage2_deep_agent.py:153-170` 删 legacy 镜像逻辑
- `src/agents/core/agent_config.py` `vg_backend` 默认值 `'pack_v1'`
- `src/agents/examples/embodiedscan_vg_pilot.py` 删除或改为指向 pack-v1 pilot

测试:

- `src/agents/tests/test_stage2_deep_agent.py` 删 `test_legacy_vg_tool_list_snapshot`
- `src/agents/tools/tests/test_select_object.py` `FakeRuntimeState`
  改成新字段;旧 `vg_scene_objects` 字段消除
- 同样处理 `test_spatial_compare.py`

---

## Section 1 — `select_object.py` / `spatial_compare.py` 切换到 `task_ctx`

### Task 1 — `select_object.py` 改读 `runtime.task_ctx`

**Files:**
- Modify: `src/agents/tools/select_object.py:65-110`
- Modify: `src/agents/tools/tests/test_select_object.py`

行为约束:

- `handle_select_object` 不再读 `runtime_state.vg_scene_objects` 或
  `vg_axis_align_matrix`。改读 `runtime_state.task_ctx`,把
  `task_ctx` 视为 `VgEmbodiedScanCtx`,从 `task_ctx.proposals` 找
  matching `obj_id`。
- 既然 pack-v1 不再用 `select_object`(终止动作走 `submit_final`),
  本 task **可以选**:
  (a) 把 `select_object` 整个工具删掉(更彻底),或
  (b) 保留 `select_object` 给某些 ad-hoc 调用,但内部走新 ctx。
- **推荐 (a)**:删除 `src/agents/tools/select_object.py` 与
  `tests/test_select_object.py`;在 Plan B Task 8 验证 OpenEQA 5
  sample 不再触发 `select_object`(grep `tool_trace` 验证)。

**TDD 步骤:**

1. 删除 `src/agents/tools/select_object.py` 与 `tests/test_select_object.py`。
2. `pytest src/agents/ -v` — 任何 import 该 module 的代码都会失败,
   按报错顺次清理。
3. commit: `refactor(agents/tools): remove select_object (replaced by submit_final)`

---

### Task 2 — `spatial_compare.py` 同样处理

**Files:**
- Modify or delete: `src/agents/tools/spatial_compare.py`
- Modify or delete: `src/agents/tools/tests/test_spatial_compare.py`

VG pack 的 `compare_proposals_spatial` 已经覆盖空间比较语义,旧
`spatial_compare`(基于 category + scene_objects)应当删除。

**TDD 步骤:** 类比 Task 1。

commit: `refactor(agents/tools): remove spatial_compare (replaced by compare_proposals_spatial)`

---

### Task 3 — `stage2_deep_agent.py:153-170` 镜像逻辑清理

**Files:**
- Modify: `src/agents/stage2_deep_agent.py`

`stage2_deep_agent.py:140-182` 那段 `WARNING: ... must stay in sync with
runtime copy` 在 deepagents 切换之后已经长期裸奔。本 task 把它改成只
做必要的 task_ctx 注入。

**TDD 步骤:**

1. 跑既有测试(应该全绿)。
2. 砍 `vg_*` 镜像;只剩 `runtime.task_type = task.task_type` 与
   `runtime.task_ctx = build_ctx_from_bundle(...)` 两行。
3. 跑测试。
4. commit: `refactor(agents/stage2_deep_agent): drop vg_* state mirroring; rely on task_ctx`

---

## Section 2 — `Stage2RuntimeState` 字段删除

### Task 4 — 删 5 个 `vg_*` 字段

**Files:**
- Modify: `src/agents/runtime/base.py:43-49`

行为约束:

- 在 Tasks 1-3 完成后,**全仓 grep** 确认无任何 `vg_scene_objects` /
  `vg_axis_align_matrix` / `vg_selected_object_id` /
  `vg_selected_bbox_3d` / `vg_selection_rationale` 残留:

```bash
grep -rn "vg_scene_objects\|vg_axis_align_matrix\|vg_selected_object_id\|vg_selected_bbox_3d\|vg_selection_rationale" src/
```

期望: 除 `runtime/base.py` 自身的 dataclass field 之外,**零**匹配。

- 删 5 个字段。
- 跑全测试套件。

commit: `refactor(agents/runtime): drop legacy vg_* fields from Stage2RuntimeState`

---

## Section 3 — `vg_backend='legacy'` 分支整体删除

### Task 5 — 删 `deepagents_agent.py` legacy VG 分支

**Files:**
- Modify: `src/agents/runtime/deepagents_agent.py:266-310`(VG-tool 分支
  里的 `elif runtime.vg_scene_objects is not None` 整段)
- Modify: `src/agents/runtime/deepagents_agent.py:570-588`
  (`if self.config.vg_backend == "pack_v1":` 现在是唯一分支)

行为约束:

- 只剩一个分支:VG → 跑 pack。
- `Stage2DeepAgentConfig.vg_backend` 默认值改 `'pack_v1'`。
- `vg_backend='legacy'` 直接 raise(明确删除):

```python
if self.config.vg_backend != "pack_v1":
    raise ValueError(
        f"vg_backend={self.config.vg_backend!r} no longer supported; "
        "legacy branch removed in Plan C. Set vg_backend='pack_v1'."
    )
```

- 删 `test_legacy_vg_tool_list_snapshot`(已无意义)。
- 把 `test_pack_v1_vg_tool_list_snapshot` 改为默认 config 也通过(因为
  默认值已切换)。

commit: `refactor(agents/runtime): remove legacy VG branch; default vg_backend=pack_v1`

---

### Task 6 — 删 `embodiedscan_vg_pilot.py`

**Files:**
- Delete: `src/agents/examples/embodiedscan_vg_pilot.py`

行为约束:

- 在 Plan A Task 20 已经新建 `embodiedscan_vg_pack_v1_pilot.py`。
- 旧 pilot 不再需要。
- 在 `embodiedscan_vg_pack_v1_pilot.py` 的 docstring 末尾加一段
  "(replaces legacy embodiedscan_vg_pilot.py removed in Plan C)" 备注。

commit: `chore(agents/examples): remove legacy embodiedscan_vg_pilot (use pack_v1 pilot)`

---

### Task 7 — 全仓回归

**Files:** 无(只跑测试)

```bash
pytest src/ -v
python src/agents/examples/openeqa_official_question_pilot.py --max-samples 5 --output-root outputs/qa_post_plan_c
python src/agents/examples/embodiedscan_vg_pack_v1_pilot.py ... # 跟 Plan A Task 20 同样的入口
```

期望:全部 GREEN,1 sample VG 跑通,5 sample QA 跑通。

commit: `test: full suite green after legacy cleanup (Plan C complete)`

---

## Self-Review checklist (Plan C)

- [ ] `grep -rn "vg_scene_objects\|vg_axis_align_matrix\|vg_selected_object_id\|vg_selected_bbox_3d\|vg_selection_rationale" src/` 返回 0 行
- [ ] `grep -rn "select_object\|spatial_compare" src/` 返回 0 行(除 history-only 文档)
- [ ] `Stage2DeepAgentConfig.vg_backend` 默认 `'pack_v1'`
- [ ] `tests/test_stage2_deep_agent.py` 不再含 `test_legacy_vg_tool_list_snapshot`
- [ ] `embodiedscan_vg_pilot.py` 不存在
- [ ] full pytest 套件全绿

---

## 触发 Plan D 的条件

Plan D (Nav Plan pack) 是**完全独立** 的子项目,不依赖 Plan C
完成 — 可以并行启动。本 plan 完成只是把 chassis 收敛到"零 legacy 包袱"
的状态,让 Nav Plan 实施时不被遗留代码干扰。
