# Plan D — Nav Plan pack(占位)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:writing-plans to expand this placeholder into a full plan when triggered. Steps use checkbox (`- [ ]`) syntax for tracking.

**状态:** 占位。此 plan **不应被执行**,直到下面的"触发条件"全部满足。

**Goal (occurrence trigger):** 在 spec 第 11 节 "Future Pack: Nav
Plan" 附录的契约上,实施 Nav Plan pack —— 用同一个 Stage 2 chassis
解决 navigation planning。

**Source spec:** `docs/superpowers/specs/2026-04-25-stage2-multi-task-agent-design.md`,§"Future Pack: Nav Plan"。

---

## 何时把本 plan 展开为详细 task

满足全部条件再展开:

- [ ] **Plan A 完成:** chassis + VG pack-v1 已上线、`pack_v1.Acc@0.25 ≥ legacy − 1pp`
  在更大规模(≥ 100 sample)的复测中保持。
- [ ] **Plan C 完成:** legacy `vg_*` 字段与 legacy VG 分支已删除。
  这是**关键**条件 —— Nav Plan 的 chassis 改造(尤其
  `Stage2EvidenceBundle.extra_metadata.nav_context` 的字段验证)
  在 legacy 字段尚存时容易引入新的隐蔽 fallback。
- [ ] **Stage 1 给出了可用的 Nav Plan 输入:** 至少在一个数据集上有
  `start_pose / coordinate_frame / action_space / navigation_graph |
  navmesh | occupancy_grid / candidate_goals` 可读。当前的
  `data/embodiedscan/` + `data/OpenEQA/scannet/` 两个数据集是否够
  Nav Plan,需独立调研。如果都不够,先完成数据准备再展开本 plan。
- [ ] **新 spec:** 写一份 `docs/superpowers/specs/YYYY-MM-DD-stage2-nav-plan-design.md`,沿用 Stage-2 多任务 spec
  附录里的 8-tool / 3-skill / `nav_context` 字段 + smallest-breaker test,但补全实施细节(payload schema、validator 行为、数据来源、metric)。
- [ ] **Pilot dataset 选定:** 在 ScanNet / Habitat / RoboTHOR / SQA3D 中选 1
  个作为 Nav Plan 的 ScanNet-equivalent。spec 必须明确写哪个,以及
  对应数据准备的脚本。

---

## 占位 Outline (展开时按这个骨架填)

类比 Plan A 的章节切分:

### Section 1 — Foundation (与 Plan A 重叠的部分已完成,本 plan 不重做)

- chassis 基元(已完成 in Plan A Task 1-8)
- `validate_packs` (已完成)
- chassis tools (已完成)

### Section 2 — Nav Plan pack v1

参考 spec 第 11 节附录。占位 task 列表:

- Task — `NavPlanningCtx` 数据类与 `build_ctx_from_bundle`
- Task — Nav `FinalizerSpec` (`NavPayload`,plan_type 三选一:
  waypoints / actions / subgoals)
- Task — 8 个 Nav 工具(每个一个 task,带 FAIL-LOUD gate)
- Task — 3 个 skill markdown body
- Task — pack 注册(类比 Plan A Task 18)

### Section 3 — 数据准备 + Pilot

- Task — 离线生成 `nav_context`(start_pose / map / candidate_goals)的脚本
- Task — Nav Plan pack-v1 pilot 脚本
- Task — Nav 验收 metric 与对比基线(spec 附录里的 smallest-breaker
  test 决定:metric 必须在 metric waypoints 上验证,而非高级 subgoal
  文本)

### Section 4 — 端到端 smoke + spec Implementation Status

- Task — 5 sample smoke
- Task — 把 metric 写回 spec 末尾

---

## 不在 Plan D 范围内

- chassis 任何改造 — 都应该已经在 Plan A 落地。如果 Nav Plan 暴露
  chassis 缺陷,**回到 Plan A 的 chassis 章节** 再开一个修订计划,
  不要在 Plan D 里偷偷扩 chassis。
- Manipulation pack —— spec 里 `Stage2TaskType.MANIPULATION` 也存在,但
  目前没有 owner 没有数据,留给更晚的 Plan E。

---

## 待 owner

本 plan 没有指定 owner;由完成 Plan C 的工程师在 trigger condition 满
足时**重新跑 `superpowers:writing-plans` skill** 把本 plan 展开为完
整 task 表。
