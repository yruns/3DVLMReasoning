# Plan B — QA 迁默认 ToolPack 与 subagents 降级 skill

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 spec 第 10 节 step 7 (QA 进默认 ToolPack 并启用
chassis trio) 与 step 8 (subagents `evidence_scout` + `task_head`
降级为 skill body)。

**Architecture:** 在 Plan A 落地的注册中心 + chassis 之上,新增
`src/agents/packs/qa_default/`(空 ToolPack + 一个或两个 skill 复用
现有 `evidence-scouting`),再写一个独立 PR 把 DeepAgents 的
`subagents=[]` 移除并把对应内容放进 skill markdown。

**Tech Stack:** 同 Plan A。

**触发条件:** Plan A step 23 验收通过(`pack_v1.Acc@0.25 ≥ legacy − 1pp`)。

**Source spec:** `docs/superpowers/specs/2026-04-25-stage2-multi-task-agent-design.md`

**粒度说明:** 本 plan 是**中粒度** — 每个 task 给出 file/test/key
behavior,但不把每个 step 都展开 5 行 TDD 节奏。Plan A 的执行体感
告诉我们某些细节(尤其 QA 的 finalizer payload)在那时会有更准确的
信息;本 plan 在 Plan A 完成后会被 re-review 一次再开工。

**Re-review 触发点:** Plan A 全部 task 完成、Plan A self-review
也跑过之后,**重新读一次 spec 第 9 节 OpenEQA compatibility**,
把任何在 Plan A 期间发现的契约偏差写进本 plan 的 task,然后再开工。

---

## File Structure

新增:

- `src/agents/packs/qa_default/__init__.py`
- `src/agents/packs/qa_default/registration.py`
- `src/agents/packs/qa_default/finalizer.py`
- `src/agents/packs/qa_default/tools.py`(可能为空,复用现有 5 个共享 tool)
- `src/agents/packs/qa_default/skills/qa_answering_playbook.md`
- `src/agents/packs/qa_default/skills/evidence_scouting.md`(若与 VG 共享则 symlink 或复制)

修改:

- `src/agents/runtime/deepagents_agent.py:312-342` 删 `build_subagents` 返回
  非空列表的逻辑,改为始终返回 `[]`(与 step 8 同步)
- `src/agents/runtime/base.py:412` 把 `"Subagents may be used in FULL mode"`
  prompt 文案改为指向 `evidence-scouting` skill
- `src/agents/tests/test_stage2_deep_agent.py:235-268` 重写
  `test_build_agent_uses_deepagents_response_format_and_full_mode_subagents`
  ,改为断言 `subagents=[]` 与 system prompt 含 skill catalog
- `src/agents/runtime/deepagents_agent.py:570-588` QA 路径在
  `enable_chassis_tools=True OR pack_v1` 时挂 chassis trio;否则保持现状

---

## Section 1 — QA 默认 ToolPack (spec step 7)

### Task 1 — `qa_default` pack 骨架与注册

**Files:**
- Create: `src/agents/packs/qa_default/__init__.py`
- Create: `src/agents/packs/qa_default/registration.py`
- Create: `src/agents/packs/qa_default/finalizer.py`
- Create: `src/agents/packs/qa_default/tools.py`
- Modify: `src/agents/packs/__init__.py` 增加一行 import 触发 qa_default
- Test:   `src/agents/packs/qa_default/tests/test_registration.py`

行为约束:

- `tool_builder` 返回 `[]`,因为 QA 复用 chassis 上已注册的 5 个共享 tool
  (`inspect_stage1_metadata`、`retrieve_object_context`、
  `request_more_views`、`request_crops`、`switch_or_expand_hypothesis`)。
- `required_extra_metadata = []`
- `required_primary_skill = "qa-answering-playbook"`
- `ctx_factory = lambda bundle: bundle`(QA 不需要 typed ctx;直接挂 bundle)

**TDD 步骤:** 类比 Plan A Task 18(VG 注册测试),改成 QA。

测试断言:

```python
assert Stage2TaskType.QA in PACKS
pack = PACKS[Stage2TaskType.QA]
assert sorted(s.name for s in pack.skills) == ["evidence-scouting", "qa-answering-playbook"]
assert pack.required_primary_skill == "qa-answering-playbook"
assert pack.required_extra_metadata == []
```

**commit message:** `feat(agents/packs/qa_default): register QA pack with answering-playbook + evidence-scouting`

---

### Task 2 — `QaPayload` 与 QA `FinalizerSpec`

**Files:**
- Modify: `src/agents/packs/qa_default/finalizer.py`
- Test:   `src/agents/packs/qa_default/tests/test_finalizer.py`

行为约束:

- `QaPayload(answer: str, supporting_claims: list[str])`,严格 schema(注:
  保持与现有 OpenEQA `default_payload_schema(QA)` 一致,见
  `src/agents/runtime/base.py:96`)
- `validator(payload, runtime)`: trim、验证 `answer` 非空 string、
  `supporting_claims` 是 list[str]
- `adapter(payload, runtime) -> dict`: 平铺成
  `{"status": "completed", "answer": ..., "supporting_claims": [...]}`,
  与现有 `Stage2StructuredResponse` 字段一致

**TDD 步骤:**

1. 写测试:`test_qa_validator_rejects_empty_answer`、
   `test_qa_adapter_emits_supporting_claims`、
   `test_qa_validator_strips_whitespace`
2. 实现 `finalizer.py`
3. commit: `feat(agents/packs/qa_default): QA FinalizerSpec`

---

### Task 3 — 接通 chassis trio for QA(`enable_chassis_tools=True`)

**Files:**
- Modify: `src/agents/runtime/deepagents_agent.py:266-310`(QA 分支)
- Modify: `src/agents/tests/test_stage2_deep_agent.py`

行为约束:

- `build_runtime_tools` 在 `runtime.task_type == QA` 时,如果
  `self.config.enable_chassis_tools=True` **OR** `Stage2TaskType.QA in PACKS`,挂 chassis trio。
- 默认 `enable_chassis_tools=False`,但只要 QA pack 已注册(本 Plan task 1
  落地后即 True),自动启用 chassis trio。
- `derive_eval_session_id` 因 chassis 表面变化,把 `chassis_tools_version`
  从 `1` 提到 `2`(在 Plan A task 4 已经支持 hash 化)。

**TDD 步骤:**

1. 改 Plan A 引入的 `test_qa_tool_list_snapshot`:把期望列表改为
   原 5 个 + chassis trio (`list_skills`、`load_skill`、`submit_final`)。
   先修改测试,确认它**红**。
2. 改 `build_runtime_tools` QA 分支。
3. 跑测试 PASS。
4. 把 `chassis_tools_version` 默认值从 1 提到 2(在
   `Stage2DeepAgentConfig`),并提供 changelog 注释解释何时再 bump。
5. commit: `feat(agents/runtime): enable chassis tools for QA when qa_default pack registered`

---

### Task 4 — `qa-answering-playbook` skill body

**Files:**
- Create: `src/agents/packs/qa_default/skills/qa_answering_playbook.md`

至少包含:

- 何时使用(覆盖 OpenEQA 类型问题:目标存在 / 颜色 / 数量 / 状态 /
  位置)
- 决策树:先看 keyframe 概览 → 用 `request_more_views` 或
  `request_crops` 收齐细节 → 用 `inspect_stage1_metadata` 检查
  Stage 1 假设 → `submit_final({answer, supporting_claims}, rationale)`
- 5 个共享 tool 的详细签名 + 何时调
- chassis trio 用法
- 反例 5 条(不要不看图就答、不要在 evidence_scout-like 取证之前 submit、
  不要把 supporting_claims 写成完整段落)

**TDD 不要求**(纯 markdown);手工 verify ≥ 80 行。

commit: `docs(agents/packs/qa_default): qa-answering-playbook skill`

---

### Task 5 — QA `evidence-scouting` skill 共享(symlink 还是复制?)

**Files:**
- Create: `src/agents/packs/qa_default/skills/evidence_scouting.md`(直接复制 Plan A 同名文件,**不做 symlink**)

理由: filesystem symlink 在 Windows / 某些 CI 不友好;直接复制并在
两份文件首行加注释 `<!-- shared with packs/vg_embodiedscan/skills/ -->`。
后续如果有 drift,可在 CI 加一个 diff 检查。

或者:把 `evidence_scouting.md` 提到 `src/agents/skills/shared_skills/`
集中存放,两个 pack 都 reference 这个集中目录的 path。**Plan A 完成
之后再决定**。

**Action:** 在 Plan A self-review 之后,把这个决定写进本 task,然后
执行。

commit: `docs(agents/packs/qa_default): include evidence-scouting skill (shared with VG)`

---

## Section 2 — Subagents 降级 skill (spec step 8,独立 PR)

### Task 6 — 写 `evidence-scouting` 与 `task-head-output` 的 skill body

**Files:**
- Modify or extract: `src/agents/runtime/deepagents_agent.py:312-342`
  把 `evidence_scout` 与 `task_head` 的 system_prompt 文本搬到:
  - `src/agents/packs/qa_default/skills/evidence_scouting.md` 已经有一份(
    Plan A + Plan B Task 5),把 `evidence_scout` 的 system prompt 增添到
    "Subagent legacy guidance" 一节
  - 新建 `src/agents/skills/shared_skills/task_head_output.md`(集中放在
    chassis 共享目录)

(如果决定 Task 5 提到 `shared_skills/`,所有 shared skill 都迁过去。)

**commit message:** `docs(agents/skills): demote evidence_scout + task_head subagent prompts into skill bodies`

---

### Task 7 — 删除 `build_subagents` 返回非空逻辑

**Files:**
- Modify: `src/agents/runtime/deepagents_agent.py:312-342`
- Modify: `src/agents/runtime/deepagents_agent.py:595` (`subagents=self.build_subagents(task)` → `subagents=[]`)
- Modify: `src/agents/runtime/base.py:412` ("Subagents may be used in FULL mode" 文案改为指向 skill catalog)
- Modify: `src/agents/core/agent_config.py:60` 删 `enable_subagents` 字段(或保留 deprecated 注释);为安全起见**保留字段**但不再被读

**TDD 步骤:**

1. 改 `test_build_agent_uses_deepagents_response_format_and_full_mode_subagents`
   (Plan A 期间已经红了一段时间因为 subagents 仍返回 2;Plan B 完成
   后这个测试要恰好 GREEN,断言 `kwargs["subagents"] == []`)。
2. 跑测试,先红;改实现;跑测试 PASS。
3. system prompt 文案 hash 会变,确认 `chassis_tools_version` 提到 `3`,
   `derive_eval_session_id` 重新分桶。
4. commit: `refactor(agents/runtime): drop subagents=[evidence_scout, task_head]; replace with skill bodies`

---

### Task 8 — 跑 OpenEQA 1-sample 回归 smoke

**Files:** 无新文件;运行既有 OpenEQA pilot

```bash
# QA path,vg_backend 默认 legacy(本 plan 不切换 VG)
python src/agents/examples/openeqa_official_question_pilot.py \
  --max-samples 1 \
  --output-root outputs/qa_smoke_after_plan_b
```

行为约束:

- 不 crash
- `Stage2StructuredResponse.answer` 非空
- 不会因为 chassis trio 加入而 hit context length 上限

**commit message:** `test(openeqa): smoke verifies QA pack-v1 path on 1 sample`

---

## Self-Review checklist (Plan B)

- [ ] spec 第 10 节 step 7 全部 task 覆盖(qa_default pack + chassis
  trio for QA + qa-answering-playbook skill)
- [ ] spec 第 10 节 step 8 全部 task 覆盖(subagents 降级 + system
  prompt 文案 + 测试重写)
- [ ] 没有遗留 `enable_subagents` 的"幽灵开关"
- [ ] QA snapshot test 与 chassis_tools_version 都更新到正确值
- [ ] OpenEQA 1-sample smoke 通过

---

## 触发 Plan C 的条件

完成本 plan 后,运行:

```bash
pytest src/agents/ src/evaluation/ -v
python src/agents/examples/openeqa_official_question_pilot.py --max-samples 5 --output-root outputs/qa_post_plan_b
```

通过即可启动 Plan C(legacy `vg_*` 字段与 legacy VG 分支删除)。
