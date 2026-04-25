# Stage-2 多任务 Agent 架构设计

日期: 2026-04-25  
分支: `feat/explore_3dbbox`  
配套评审产物: `~/.super-orchestrator/agent-arch/`

## 目标

一个 Stage-2 agent 主循环,处理 `Stage2TaskType` 中的所有下游任务:

- **QA** — 已在 OpenEQA 上线,不能回归。
- **VG** — 当前在 EmbodiedScan VG val (ScanNet 子集) 上构建。
- **Nav Plan** — 未来任务,本 spec 不实现,但设计不能偏向 VG 语义。

每个任务的差异性完全收敛在插拔式 **pack** 里(一个 `ToolPack`、一个
`SkillPack`、一个 `FinalizerSpec`)。chassis 保持任务无关。

## 锁定原则

一个 DeepAgents 主循环。一套 trace 契约。一个终止动作。绝不允许任何任务
分叉成并行管线。

## 架构总览

```
Chassis (任务无关) ──────────────────────────────────────────
  · DeepAgents 主循环 (通过 create_deep_agent 构建)
  · LLM session (单 key AzureChatOpenAI, session_id 缓存前缀)
  · Stage2RuntimeState (扩展 task_ctx 与 skills_loaded)
  · 多模态 user-message 构造器、trace 记录器
  · 始终启用的 chassis tool:
      list_skills()                    -> [{name, description}]
      load_skill(name)                 -> markdown body
      submit_final(payload, rationale, evidence_refs=[]) -> 终止

ToolPack (每个 task_type 注册一个) ──────────────────────────
  · @tool 函数 (LangChain BaseTool),docstring 极简
  · 真正的用法说明放在 skill 里,不在这里
  · 每次 run() 调用时构建,不在进程启动时构建

SkillPack (每个 task_type 注册一个) ─────────────────────────
  · SkillSpec 条目 (name、≤1 行 description、body_path)
  · System prompt 仅展示 catalog (name + description)
  · body 是 markdown,运行期通过 load_skill 拉取

FinalizerSpec (每个 task_type 注册一个) ─────────────────────
  · payload schema (Pydantic 模型)
  · 必要前置条件(哪些 ctx 字段必须存在)
  · validator/resolver 回调
  · 把 payload 转换成 Stage2StructuredResponse 字段的 adapter
```

运行时在 `build_agent()` 阶段把当前任务的三个组件组装好,通过
`validate_packs()` 在第一次 LLM call 之前完成校验。

## 注册中心

`src/agents/skills/registry.py`:

```python
@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str            # ≤1 行,出现在 catalog 中
    body_path: Path             # markdown,按需加载
    task_types: set[Stage2TaskType]

@dataclass(frozen=True)
class TaskPack:
    task_type: Stage2TaskType
    tool_builder: Callable[[Stage2RuntimeState], list[BaseTool]]
    skills: list[SkillSpec]
    finalizer: FinalizerSpec
    required_primary_skill: str             # 必须先加载的 skill 名
    required_extra_metadata: list[str]      # bundle.extra_metadata 必备 key
    ctx_factory: Callable[[Stage2EvidenceBundle], Any]   # 构造 task_ctx

PACKS: dict[Stage2TaskType, TaskPack] = {}

def register_pack(pack: TaskPack) -> None:
    if pack.task_type in PACKS:
        raise RuntimeError(f"duplicate pack: {pack.task_type}")
    PACKS[pack.task_type] = pack
```

每个 pack 模块在 import 时自注册:

```python
# src/agents/packs/vg_embodiedscan/__init__.py
from .registration import VG_PACK
register_pack(VG_PACK)
```

运行时通过 import `src/agents/packs/__init__.py` 触发所有 pack 的确定性
导入。

## Chassis tool (始终启用)

三个工具,均带完整 docstring:

```python
@tool
def list_skills() -> str:
    """List skills available for the current task. Returns name + short description."""

@tool
def load_skill(skill_name: str) -> str:
    """Fetch the full instructions for a skill. Returns markdown body. Records load."""

@tool
def submit_final(payload: dict, rationale: str, evidence_refs: list[dict] = []) -> str:
    """Submit the final task answer. Payload must match this task's FinalizerSpec.schema.
    The chassis validates payload + preconditions; on success, terminates the run."""
```

行为约定:

- `list_skills()` 读取 `PACKS[task_type].skills`,返回 JSON 数组。
- `load_skill(name)` 校验 `name` 在当前 pack 的 skill 列表中,读取
  `body_path`,通过 `runtime.record('load_skill', ...)` 记录,把 `name`
  加入 `runtime.skills_loaded`,返回 markdown body。未知名字 → 返回
  ERROR 字符串("skill X 未在 task_type Y 注册;available: [...]");
  **FAIL-LOUD**:同一个 run 内出现第二次未知 skill 名时 chassis 直接抛
  异常,避免无限循环。
- `submit_final(payload, rationale, evidence_refs)` 调用
  `pack.finalizer.validator(payload, runtime)`。校验通过则把解析结果写
  入 runtime,通过既有的 `Stage2StructuredResponse` 通道发出终止信号
  (StructuredResponse 由 FinalizerSpec adapter 构造,不依赖 DeepAgents
  的 `response_format`)。校验失败返回 ERROR 字符串,agent 必须修正后
  重试。

## EmbodiedScan VG pack

`src/agents/packs/vg_embodiedscan/`:

```
__init__.py            # imports + register_pack
registration.py        # 构造 TaskPack
tools.py               # 5 个新 @tool 函数 (docstring 极简)
ctx.py                 # VgEmbodiedScanCtx dataclass
finalizer.py           # FinalizerSpec for VG
skills/
  vg_grounding_playbook.md
  vg_spatial_disambiguation.md
  evidence_scouting.md          # 后续与 QA 共享
```

### ToolPack (5 个新工具 + 1 个复用,每个新工具 docstring ≤1 行)

| Tool | 用途 |
|---|---|
| `list_keyframes_with_proposals` | (新) 概览 keyframes 及每帧可见的 proposal id |
| `view_keyframe_marked` | (新) 取一帧的 set-of-marks 标注图 |
| `inspect_proposal` | (新) proposal 元信息 + 近距离 crop + 出现过的所有帧 |
| `find_proposals_by_category` | (新) 按类别名查 proposal 池 |
| `compare_proposals_spatial` | (新) 给定一组候选 + anchor + relation,排序 |
| `request_more_views` | (复用现有共享工具) 让 Stage 1 二次响应,详细语义在 `evidence-scouting` skill |

VG 通过 chassis 的 `submit_final` 终止,**不存在**任务专属
`submit_grounding_answer`。

每个新 VG-pack 工具的 docstring 都是:`"VG tool. Detailed usage in skill 'vg-grounding-playbook'."` 复用的共享工具保留现有 docstring。

**FAIL-LOUD 门控。** 每个工具体内首先检查
`'vg-grounding-playbook' in runtime.skills_loaded`,不满足则返回:

```
ERROR: load_skill('vg-grounding-playbook') before calling this tool.
```

### SkillPack (3 个 skill)

| Skill | catalog desc | body 内容 |
|---|---|---|
| `vg-grounding-playbook` | EmbodiedScan VG 主流程:看标注 keyframe、选 proposal、提交。 | 决策树;6 个 VG-pack 工具的完整用法;VG 的 `submit_final` payload schema;反例;OOD 情况 = `submit_final({"proposal_id": -1, "confidence": 0.0}, rationale="GT not in proposal pool")` —— 显式标记为 failed sample,纳入失败统计,**绝不**静默给一个默认 bbox |
| `vg-spatial-disambiguation` | query 含 `next to` / `closest to` 等空间关系时使用。 | 拆 target/relation/anchor;`find_proposals_by_category`、`compare_proposals_spatial`;anchor 自身的递归消歧;示例 |
| `evidence-scouting` | 何时请求更多 keyframe 或 crop,以及怎么写 request 内容。 | `request_more_views` 三种 mode (`targeted` / `explore` / `temporal_fan`);`request_crops`;sufficiency 判据;反例。后续与 QA 共享。 |

### `VgEmbodiedScanCtx` (挂在 `Stage2RuntimeState.task_ctx`)

```python
@dataclass
class VgEmbodiedScanCtx:
    proposal_pool_source: Literal["vdetr", "conceptgraph"]
    proposals: list[Proposal]                     # id, bbox_3d_9dof, category, score
    frame_index: dict[int, list[int]]             # frame_id -> [proposal_id]
    proposal_index: dict[int, list[int]]          # proposal_id -> [frame_id]
    annotated_image_dir: Path
    axis_align_matrix: np.ndarray | None
```

### VG 的 `FinalizerSpec`

```python
class VgPayload(BaseModel):
    proposal_id: int            # -1 = "GT 不在池里,显式标记失败"
    confidence: float = Field(ge=0.0, le=1.0)

def vg_validator(payload: VgPayload, runtime) -> Stage2StructuredResponse:
    ctx: VgEmbodiedScanCtx = runtime.task_ctx
    if payload.proposal_id == -1:
        # 显式 fail-loud opt-out (failed sample,纳入指标)
        return _failed_response(...)
    proposal = next((p for p in ctx.proposals if p.id == payload.proposal_id), None)
    if proposal is None:
        raise ValueError(f"proposal_id {payload.proposal_id} not in pool")
    return _success_response(bbox_3d_9dof=proposal.bbox_3d_9dof, ...)
```

## 数据契约 — `Stage2EvidenceBundle.extra_metadata`

每个任务的字段槽位,在 `build_agent()` 阶段强校验:

| Task | 必备 key | 校验 |
|---|---|---|
| QA | (无) | no-op |
| VG | `vg_proposal_pool` | schema 校验;proposal 列表非空;`annotated_image_dir` 可读;axis-align matrix 形状 `(4,4)` 或 `None` |
| Nav Plan (未来) | `nav_context` | 见附录 |

`pack.ctx_factory(bundle)` 构造典型 ctx。提取或校验失败 **在
`build_agent()` 抛异常**,不在 tool 调用时报错。

## FAIL-LOUD 策略 (不可妥协)

1. **`build_agent()` 时执行 `validate_packs()`** 并断言:
   - `task.task_type` 已注册 pack
   - `pack.skills[*].body_path` 文件全部存在且可读
   - 跨 pack 的 tool name + skill name 全局唯一
   - `pack.required_extra_metadata` 所有 key 在 bundle 中存在
   - `pack.ctx_factory(bundle)` 返回非 None 的 typed ctx
   任何失败均在第一次 LLM call 之前抛出。

2. **任务专属 tool 必须先 load skill。** 每个 task-specific tool 检查
   `runtime.skills_loaded`,若 gating skill 缺失,返回 ERROR 字符串。

3. **路由仅由 `task_type` 决定。** **绝不**根据
   `extra_metadata.vg_proposal_pool` 是否存在选择代码路径。legacy VG
   分支与新 pack 之间的切换由显式 config
   `vg_backend: Literal['legacy', 'pack_v1']` 控制。

4. **修掉 `select_object.compute_bbox_3d` 的 silent fallback。**
   `select_object.py:50-56` 当前对没有点云的对象用 `[0.3, 0.3, 0.3]`
   兜底。改成 `raise ValueError(...)`。chassis 已经能处理 tool ERROR
   字符串,这只是把错误回报给 agent。**这是"不修改既有 tool"原则的
   一个明确例外。**

5. **pilot 不允许 silent 路由 fallback。** legacy 的
   `embodiedscan_vg_pilot.py` 在异常或 CG 缺 proposal 时返回
   `bbox_3d: None`。新的 pack-v1 pilot 必须显式标记样本失败,**绝不**
   静默给空答案。

## OpenEQA 兼容性

QA 不能回归。具体保护措施:

- **chassis tool 注册规则:** 三件套 (`list_skills` / `load_skill` /
  `submit_final`) 仅当当前 `task_type` 已注册 `TaskPack` 时才注册。QA
  在 step 7 之前**没有**注册 pack,因此 QA 完全拿不到这三个 chassis
  tool —— tool 列表与今天完全一致。**override flag:**
  `Stage2DeepAgentConfig.enable_chassis_tools: bool = False` 允许调用方
  在没有 pack 的情况下提前给 QA 装上 chassis 三件套(用于 ad-hoc 评测);
  默认 OFF,因此当前 QA 行为字节级稳定。
- **`derive_eval_session_id`**(`openeqa_official_question_pilot.py:280-292`)
  哈希新增 `chassis_tools_version: int` 与当前 `vg_backend` 字段,确保
  chassis 表面变化时 prompt cache 正确失效。
- **snapshot test** 加在 `tests/test_stage2_deep_agent.py`,断言每个
  `task_type` 的 tool name 列表精确值。迁移前:
  - QA = `[inspect_stage1_metadata, retrieve_object_context,
          request_more_views, request_crops,
          switch_or_expand_hypothesis]`
  - VG = QA + `[select_object, spatial_compare]`
  step 5 之后:VG (在 `vg_backend='pack_v1'` 下) = 上面 5 个共享 tool
  + 5 个 VG-pack 新 tool (`list_keyframes_with_proposals` /
  `view_keyframe_marked` / `inspect_proposal` /
  `find_proposals_by_category` / `compare_proposals_spatial`) + chassis
  三件套。`request_more_views` 是复用,不是新增。QA 列表在 step 7 之前
  保持不变。
- **subagents 在 step 7 之前完全不动。**
  `subagents=[evidence_scout, task_head]` 在每条 QA 路径上原样保留。这
  两个 subagent 受 `enable_subagents=True AND plan_mode==FULL` 双门门
  控;OpenEQA 全部 preset 都不走 FULL,因此它们在 QA 链路上是死代码,
  零行为变化。step 8 才把它们降级为 skill (见迁移顺序)。
- **VG legacy 分支 (`vg_backend='legacy'`) 是默认值,直到 pack-v1
  指标对齐。** pack-v1 上线后,OpenEQA 现存 VG 评测路径仍走 legacy 代
  码,无变化。

## 迁移顺序 (10 步,OpenEQA-safe)

1. **chassis 基元 + flag。** 在 `Stage2DeepAgentConfig` 加
   `enable_chassis_tools` 与 `vg_backend`。在 `Stage2RuntimeState`
   加 `task_ctx: Any | None = None` 与
   `skills_loaded: set[str] = field(default_factory=set)`。两个默认值
   都保持现有行为。落地一个空壳 `validate_packs()`:当前任务无 pack
   注册时直接 no-op。把 `chassis_tools_version` 字段加到
   `derive_eval_session_id`。
2. **`task_ctx` additive。** 不动消费方;`vg_*` 字段保留在
   runtime state。新代码读 `task_ctx`,旧代码读 `vg_*` 字段。
3. **修 `select_object.compute_bbox_3d` 的 no-fallback 违例。** 把
   `[0.3,0.3,0.3]` extent 替换为 `raise ValueError`。加回归测试。
4. **chassis 终止动作 `submit_final` + 仅 VG 的 FinalizerSpec。** QA
   保留现有 `response_format`-based 退出方式,不动。snapshot 测试锁定
   两条退出表面。
5. **EmbodiedScan VG pack (pack_v1)。** 落地
   `src/agents/packs/vg_embodiedscan/` (tools / skills / ctx /
   finalizer / registration)。在 `build_runtime_tools()` 与 system
   prompt 内通过 `vg_backend='pack_v1'` 切入。skill catalog 注入位置
   在 `base.py:312-426`。
6. **side-by-side 对齐验证。** 在同一份 EmbodiedScan ScanNet val 子集
   上同时跑 pack-v1 VG 与 legacy VG。验收门槛:30 个样本的 sweep 上
   pack-v1 Acc@0.25 ≥ legacy − 1pp,且无 proposal-pool 元数据路由
   回归。
7. **QA 进入默认 ToolPack。** 把现有 QA tool 移入
   `src/agents/packs/qa_default/`。把 QA 的 `enable_chassis_tools`
   翻成 `True`。snapshot test 确认 QA tool name 列表恰好新增 chassis
   三件套 (`list_skills` / `load_skill` / `submit_final`)。QA 的
   `submit_final` finalizer 适配现有 `response_format` schema。
8. **subagents → skill。** 把 `evidence_scout` 与 `task_head` 降级为
   skill body。修改 `base.py:412` 的 system prompt 文案。重写
   `tests/test_stage2_deep_agent.py:235-268`。从 `build_agent()` 拿掉
   `subagents=[]`。**独立 PR。**
9. **删除 legacy `vg_*` 字段 + legacy VG 分支。** 迁移
   `select_object.py:75,91`、`spatial_compare.py:58-62` 与
   `stage2_deep_agent.py:153-170` 的消费方。删除
   `deepagents_agent.py:266-308` 的 legacy VG 代码路径。
   `vg_backend` 默认改为 `'pack_v1'`。
10. **Nav Plan pack。** 在已稳定的契约上,按附录实现。

每步都伴随测试。step 1–4 在保持现状的默认值下落地。step 5–6 在
opt-in flag 下启用新 VG 路径。step 7–9 完成 QA 迁移并删除死代码路径。

## 未来 pack:Nav Plan (附录,本 spec 不实现)

记录在此,用于约束 chassis 在 VG 实现期间保持 task-typed。

### 工具 (8 个)

| Tool | 用途 |
|---|---|
| `inspect_navigation_context` | 起始位姿、坐标系、动作空间、缺失输入 |
| `list_candidate_goal_regions` | 把语言落到对象/房间/affordance 区域/可观测位姿 |
| `render_pose_or_waypoint_view` | 在候选位姿附近取 RGB/depth/语义 evidence |
| `query_traversability` | 自由空间、障碍、未知 cell、楼层切换 |
| `plan_path_between_poses` | 在 nav graph / navmesh / occupancy 上跑路径规划,返回 waypoint 路径 |
| `validate_navigation_plan` | 碰撞、可达性、坐标系一致性、动作空间可执行性 |
| `request_route_evidence` | 请求 Stage 1 在候选区域附近补 view / BEV crop / 地图 evidence |
| (终止) | 通过 `submit_final` 提交 `NavPayload` |

### Skill (3 个)

| Skill | catalog desc |
|---|---|
| `nav-planning-playbook` | 从起始位姿出发,选目标、规划路径、校验、提交。 |
| `nav-map-and-reachability` | 当任务依赖可通行性、坐标系、楼层、障碍、动作空间约束时使用。 |
| `nav-evidence-scouting` | 判断路径/目标/地标 evidence 是否充足,以及该补什么 view 或地图证据。 |

### `nav_context` 必备字段 (上线时强校验)

`start_pose`、`coordinate_frame`、`action_space`、`navigation_graph` |
`navmesh` | `occupancy_grid`、`candidate_goals`、
`route_evidence_index`、`blocked_or_unknown_regions`、
`validation_policy`。

### smallest-breaker test

如果 Nav Plan 被要求返回"在场景坐标系下经过碰撞校验的米制 waypoint"
(而不是文字 subgoal),chassis 不需要变化:只有 Nav ToolPack 与
`FinalizerSpec` 扩展。如果 chassis 必须改,本设计就失败了。

## 不在本 spec 范围内

- Nav Plan pack 的实现(仅留附录)。
- 新的 3D detector 训练 / proposal pool 生成算法(在
  `docs/superpowers/specs/2026-04-24-embodiedscan-3d-bbox-feasibility-design.md`
  的可行性研究里覆盖)。
- 除"tool name 列表 snapshot test"以外的测试基础设施改造。
- 除"把 skill catalog 放进 cached system prompt 前缀"以外的性能 / 成本
  优化。

## 配套产物

- 被评审的初版架构提案:
  `~/.super-orchestrator/agent-arch/architecture-proposal.md`
- 三份 reviewer 报告 (claude `runtime-critic`、codex
  `alt-design-critic`、codex `nav-plan-stress-tester`):
  `~/.super-orchestrator/agent-arch/findings/`
- 多 agent 评审综合:
  `~/.super-orchestrator/agent-arch/final-synthesis.md`
- 锁定原则与所有 P0/P1 收敛点都在综合文件里。
