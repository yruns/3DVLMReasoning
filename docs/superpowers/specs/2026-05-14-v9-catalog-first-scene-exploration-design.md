# v9 Catalog-First Scene Exploration Design

日期：2026-05-14  
分支：`feat/nr3d-v4-agent-guards-fair-views`  
范围：Stage 1 / Stage 2 跨层重构 — 把 KeyframeSelector 从 pack-prep 阶段的前置流水线重定位为 Stage 2 agent 的 first-class 工具集；引入统一的 SceneCatalog 抽象，VG 和 QA 共用同一套场景探索接口。

## 问题

当前 v8 架构把 Stage 1（query 解析 + keyframe selection）放在 pack prep 阶段一次性跑完，结果以"clean RGB seed keyframes + 文本 inventory"的形式注入 Stage 2 agent。Stage 1 与 Stage 2 之间通过 3 个 callback 工具（`request_more_views`、`request_crops`、`switch_or_expand_hypothesis`）做有限的双向交互，但：

1. Stage 1 在 agent 心智模型里不是 first-class 能力，而是隐藏在 callback 后面的黑盒。
2. Stage 1 的 query 接口只有自然语言文本，agent 不能用 frame anchor / proposal id / 3D region / coverage 等 modality 驱动 frame 选取。
3. v8 random100（73.00 → 67.00 overall）表明：单靠 clean initial RGB 不足以治疗 anchoring 病；agent 没有"主动切换观察视角"的工具。
4. QA pack 仍走旧 callback + `retrieve_object_context` 路径，没有跨 task 统一的 scene 探索接口。

## 目标行为

Agent 通过 **catalog-first 初始状态**（BEV 图 + SceneCatalog 文本）形成对场景的全局认知，**没有任何 first-person seed keyframe**。它通过 6 个 modality 各异的 `select_by_*` 工具按需 fetch 帧，通过 `view_keyframe(mode)` 注入图像，通过 `view_bev(highlight=...)` 重渲焦点 BEV。VG 和 QA 共用同一套 selector / catalog / BEV / view 工具，仅 finalizer / guards / playbook 在 task 维度分化。

## 决策摘要

| 决策项 | 决定 |
|---|---|
| Selector modality 集合 | A 文本 + B 结构化 hypothesis + C frame-neighbor + D by-proposal + E by-region + F coverage（几何实现） |
| Agent 拓扑 | 无 subagent，所有工具挂主 agent |
| 工具粒度 | 拆分 6 个独立 selector 工具（不合并 super-tool） |
| 初始状态 | catalog-first：BEV image + SceneCatalog 文本，无 first-person seed |
| BEV 内容 | mesh + 相机轨迹 + 每个 proposal 在 3D 中心点叠 `#id label`（不画框） |
| BEV builder | 每 benchmark 一个独立类（Nr3d / ScanRefer / OpenEqa / Sqa3d），共享 `ScanNetSceneBEVBuilderBase` |
| 文本 catalog 形式 | Cat-B：`category → [#id, ...]` 紧凑表 + scene metadata，无 per-proposal 行 |
| Selector 返回 shape | Path A：text only；agent 单独调 `view_keyframe(N, mode)` 看图 |
| 旧 3 callback 处理 | M1 硬切换：删 `request_more_views` + `switch_or_expand_hypothesis`，保留 `request_crops` |
| 任务范围 | D2：VG（NR3D + ScanRefer）+ QA（OpenEQA + SQA3D）；Nav / Manipulation 不在 scope |
| QA catalog 抽象 | QA-B：统一 SceneCatalog，VG 由 proposal pool 适配，QA 由 conceptgraph 适配 |
| 工具命名 | 沿用 "proposal" 术语：`select_by_proposal` / `inspect_proposal` / `list_scene_proposals` / `compare_proposals_spatial` |
| `view_keyframe` 默认 mode | 按 `runtime.task_type` 自动：VG → `marked`，QA → `rgb` |
| Hypothesis 状态 | 不维护 `runtime.hypothesis_history`；删除 `list_active_hypotheses` 工具 |

---

## Section A — 整体架构 & Mental Model

### 工具拓扑（all 主 agent, no subagent）

```
┌─ 场景认知层 ──────────────────────────────────────┐
│ • 初始 HumanMessage: BEV 图 + SceneCatalog(Cat-B) 文本
│ • view_bev(highlight=[ids]?)         重渲聚焦 BEV
│ • list_scene_proposals(category, region_bev, limit)
│ • inspect_proposal(proposal_id)
└──────────────────────────────────────────────────┘
┌─ Selector 层 (A-F) ──────────────────────────────┐
│ A • select_by_text(query, k, hidden_categories)
│ B • select_by_hypothesis(hypothesis_json, k, hidden_categories)
│ C • select_by_frame_neighbor(anchor_frame_id, mode, k)
│ D • select_by_proposal(proposal_ids, require_all, k)
│ E • select_by_region(region, region_type, k)
│ F • select_by_coverage(method, k, seen_frame_ids?)
└──────────────────────────────────────────────────┘
┌─ 视觉证据层 ─────────────────────────────────────┐
│ • view_keyframe(frame_id, mode=auto|"rgb"|"marked")
│   ↳ runtime.task_type → default(VG="marked", QA="rgb")
│ • request_crops(request_text, object_terms)
└──────────────────────────────────────────────────┘
┌─ 空间推理 ─────────────────────────────────────┐
│ • compare_proposals_spatial(candidate_ids, anchor_id, relation)
└──────────────────────────────────────────────────┘
┌─ 终结层 ────────────────────────────────────────┐
│ • submit_final(payload, rationale, evidence_refs, tool_override_reason)
│   ↳ TADG → no_match_guard → evidence_frame_guard → finalizer schema
│   ↳ VG payload : {proposal_id, confidence}
│   ↳ QA payload : {answer, supporting_claims}
└──────────────────────────────────────────────────┘
┌─ 元控制 ───────────────────────────────────────┐
│ • load_skill / list_skills
└──────────────────────────────────────────────────┘
```

工具总数：15。

### 数据流（per sample）

1. **Pack prep**（offline）：渲染 BEV（mesh + traj + `#id label`）+ 构造 SceneCatalog（proposals + frame_views）+ 写入 bundle。
2. **Agent 第一条 HumanMessage**：task / query / BEV image / Cat-B 文本 / scene metadata，**无 first-person 帧**。
3. **ReAct loop**：load_skill → (view_bev / list_scene_proposals 任意) → select_by_* → view_keyframe → (空间推理 / compare 任意) → submit_final。
4. **新图注入**：`view_keyframe` 把 image path 入 `vg_pending_images` 队列，下一轮 HumanMessage 注入；selector 不直接注入图。

### Mental model

1. Agent 从一开始就**知道**：scene 有 N 帧 / M 个 proposal / category 分布，**自己已看 0 帧**。
2. BEV 是空间认知锚点；`#id label` 让"看 BEV 上 #23 → `select_by_proposal([23])` → `view_keyframe(N, marked)`"成为自然反射弧。
3. Stage 1 callback 在 agent 视角下完全消失，被 select_by_text / by_hypothesis / by_coverage 取代——agent 只看到一套统一接口。
4. VG / QA 在工具列表上几乎相同，差别只在 finalizer / guards / playbook 内容。
5. v8 的"clean RGB seed + inventory + marked-on-demand"是 v9 的特例（v9 把 inventory 升级成完整 catalog + BEV，把 RGB seed 拿掉）。

---

## Section B — SceneCatalog 数据 Schema + Pack-v1 Prep 改动

### Pydantic 模型（新模块 `src/agents/catalog/`）

```python
# src/agents/catalog/models.py
from pydantic import BaseModel
from typing import Literal


class FrameView(BaseModel):
    """一个 proposal 在某一帧里的可见信息（per-(proposal, frame) 投影）。"""
    frame_id: int
    bbox_2d: tuple[int, int, int, int]   # [x1, y1, x2, y2] 像素
    raw_rgb_path: str
    visibility_weight: float | None = None  # 0..1，来自 view_to_objects 等


class SceneProposal(BaseModel):
    """统一的 scene 实体单元（VG 的 Mask3D/V-DETR/GT proposal、QA 的 conceptgraph object）。"""
    proposal_id: int
    category: str
    position_3d: tuple[float, float, float]            # 中心点（BEV 用前两维）
    bbox_3d_9dof: tuple[float, ...] | None = None      # [cx, cy, cz, dx, dy, dz, rx, ry, rz]
    frame_views: dict[int, FrameView] = {}
    source: Literal["mask3d", "vdetr", "gt", "conceptgraph"]


class SceneCatalog(BaseModel):
    """Stage 2 看到的"场景全景"。一次构造，整段 session 共享。"""
    scene_id: str
    scene_category: str | None = None        # bedroom / kitchen / lounge ...
    proposals: list[SceneProposal]
    total_frames: int
    frame_id_range: tuple[int, int]
    valid_frame_ids: list[int]
    bev_image_path: str
    axis_align_matrix: list[list[float]] | None = None

    def proposals_by_category(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for p in self.proposals:
            out.setdefault(p.category, []).append(p.proposal_id)
        return out
```

**SceneProposal 不存 detector confidence / score**（避免 agent 把它当 prior 滥用）。

### Bundle 改动

```python
# src/agents/models.py
class Stage2EvidenceBundle(BaseModel):
    ...                                            # 现有字段
    bev_image_path: str | None = None              # v9 必填（之前 VG 不填）
    keyframes: list[KeyframeEvidence] = []         # v9 默认 [] (catalog-first)
    extra_metadata: dict = {}

    # 新规则：v9 在 extra_metadata 里强制带 scene_catalog
    # extra_metadata["scene_catalog"] : SceneCatalog.model_dump()
    # extra_metadata["vg_proposal_pool"] 字段 v9 不再写入（被 scene_catalog 取代）
```

不把 `scene_catalog` 提升为顶层字段，原因是老 pipeline（OpenEQA `llm_evaluator_v2` 等）仍用 `Stage2EvidenceBundle` 但不需要 catalog；放 `extra_metadata` 是兼容入口。

### 三个 Adapter（pack prep 端用）

```python
# src/agents/catalog/adapters.py

def from_vg_proposal_pool(
    pool: dict,
    scene_id: str,
    bev_image_path: str,
    scene_category: str | None,
    axis_align_matrix: list | None,
    valid_frame_ids: list[int],
) -> SceneCatalog:
    """NR3D / ScanRefer / EmbodiedScan: proposals.jsonl + frame_views → SceneCatalog."""

def from_conceptgraph_objects(
    pcd_saves_dir: Path,
    detections_dir: Path,
    view_to_objects: dict,
    scene_id: str,
    bev_image_path: str,
    scene_category: str | None,
    valid_frame_ids: list[int],
) -> SceneCatalog:
    """OpenEQA / SQA3D: ConceptGraph object segmentation → SceneCatalog."""

def from_gt_embodiedscan(
    es_annotations: dict,
    scene_id: str,
    bev_image_path: str,
    valid_frame_ids: list[int],
) -> SceneCatalog:
    """EmbodiedScan GT → SceneCatalog (oracle pack 用)."""
```

### Pack-v1 Prep 改动

**NR3D / ScanRefer / EmbodiedScan VG**：现有 `prepare_pack_v1_inputs*.py` 已经写 `proposals.jsonl` + `proposal.frame_views` + `annotated/`，v9 增量：

1. 调 `Nr3dScanNetBEVBuilder.build_from_scene(...)` → `<scene>/bev/scene_bev_<config_hash>.png`
2. `SceneCatalog = from_vg_proposal_pool(...)`，写到 `<scene>/scene_catalog.json`
3. Sample JSON：不再写 `keyframes`（catalog-first）；写 `scene_catalog_path` 和 `bev_image_path` 指向 scene 级 cache
4. `normalize_prepared_keyframes` 调用点删除

**OpenEQA / SQA3D QA**（新写 `prepare_pack_qa_inputs.py`）：

1. 调 `OpenEqaScanNetBEVBuilder.build_from_scene(...)` → BEV png
2. 从 `<scene>/conceptgraph/pcd_saves/` + `gsa_detections_ram_withbg_allclasses/` 抽 per-object 3D bbox + category
3. 从 `view_to_objects`（conceptgraph 已有）取每个 object 在哪些 frame 可见
4. 渲染 `annotated/frame_<id>.png`（QA 默认 RGB，但 marked 备用）
5. `SceneCatalog = from_conceptgraph_objects(...)`，写到 `<scene>/scene_catalog.json`
6. Sample JSON：scene_catalog_path + bev_image_path，无 first-person seed

### BEV Builder（每 benchmark 独立类）

```python
# src/query_scene/bev_builder.py
class ScanNetSceneBEVBuilderBase(ABC):
    """共享 mesh+traj 渲染逻辑（frustum、ceiling 去除、perspective、轨迹叠加、label overlay）。"""

    @abstractmethod
    def resolve_paths(self, scene_id: str, data_root: Path) -> tuple[Path, Path, Path]:
        """子类负责返回 (mesh_path, traj_path, intrinsic_path)，benchmark-specific 发现逻辑。"""

    def build_with_labels(
        self,
        scene_id: str,
        data_root: Path,
        proposals: list[SceneProposal],
        output_path: Path,
        highlight_ids: list[int] | None = None,  # None = 全标
    ) -> Path:
        mesh_path, traj_path, intr_path = self.resolve_paths(scene_id, data_root)
        img, _ = self._render_mesh_with_traj(mesh_path, traj_path, intr_path)
        img = self._overlay_proposal_labels(img, proposals, highlight_ids)
        cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return output_path


class Nr3dScanNetBEVBuilder(ScanNetSceneBEVBuilderBase): ...
class ScanReferScanNetBEVBuilder(ScanNetSceneBEVBuilderBase): ...
class OpenEqaScanNetBEVBuilder(ScanNetSceneBEVBuilderBase): ...  # 从老 OpenEQAScanNetBEVBuilder 改写
class Sqa3dScanNetBEVBuilder(ScanNetSceneBEVBuilderBase): ...
```

### 缓存路径

- BEV 图：`<scene_dir>/bev/scene_bev_<benchmark>_<config_hash>.png`（每 benchmark 独立 cache）
- SceneCatalog：`<scene_dir>/scene_catalog.json`（scene 级，多 sample 同 scene 共享）
- Sample JSON：仅引用 path，不 inline scene-level 资产

---

## Section C — 工具完整清单 + 行为合约

### 工具一览

| 分类 | 工具 | 状态 |
|---|---|---|
| 元控制 (2) | `load_skill`, `list_skills` | 保留 |
| 场景认知 (3) | `view_bev`, `list_scene_proposals`, `inspect_proposal` | view_bev/list 新；inspect 原 `inspect_proposal` 改读 SceneCatalog |
| 视觉证据 (2) | `view_keyframe`, `request_crops` | view_keyframe 合并 view_keyframe_marked；crops 保留 |
| Selectors A-F (6) | `select_by_text`, `select_by_hypothesis`, `select_by_frame_neighbor`, `select_by_proposal`, `select_by_region`, `select_by_coverage` | 全新 |
| 空间推理 (1) | `compare_proposals_spatial` | 保留，内部改读 SceneCatalog |
| 终结 (1) | `submit_final` | 保留，三 guard 顺序不变 |

**M1 硬切换删除**：`request_more_views`、`switch_or_expand_hypothesis`、`view_keyframe_marked`、`list_keyframes_with_proposals`、`find_proposals_by_category`、`inspect_stage1_metadata`。`retrieve_object_context` 保留（仅 QA 使用）。

### Selector 签名 + 返回 shape（Path A）

所有 selector 返回 text only，per-frame 字段：`frame_id / visible_proposal_ids / bev_xy / camera_yaw / selected_because`。

```python
def select_by_text(query: str, k: int = 3, hidden_categories: list[str] = []) -> str:
    """A. 文本 query → Stage1 parser → 执行 hypothesis → 选 frame.
    Returns JSON: {hypothesis_summary, frames: [...]}"""

def select_by_hypothesis(hypothesis: dict, k: int = 3, hidden_categories: list[str] = []) -> str:
    """B. agent 直接提交 HypothesisOutputV1（不过 parser），直接执行."""

def select_by_frame_neighbor(
    anchor_frame_id: int,
    mode: Literal["temporal", "viewpoint_diverse"] = "temporal",
    k: int = 3,
) -> str:
    """C. temporal=轨迹时间邻近;  viewpoint_diverse=位置相近但 yaw 不同."""

def select_by_proposal(proposal_ids: list[int], require_all: bool = False, k: int = 3) -> str:
    """D. 找包含指定 proposal 的 frame. require_all=False 取 union; True 取 intersection."""

def select_by_region(
    region: list[float],
    region_type: Literal["bbox_3d", "bev_2d"],
    k: int = 3,
) -> str:
    """E. bbox_3d=[xmin,ymin,zmin,xmax,ymax,zmax]; bev_2d=[xmin,ymin,xmax,ymax].
    选 camera frustum 与 region 相交的 frame."""

def select_by_coverage(
    method: Literal["obj_iou", "pose_depth"] = "obj_iou",
    k: int = 3,
    seen_frame_ids: list[int] | None = None,
) -> str:
    """F. 找和 seen_frame_ids 最不同的 k 帧.
    obj_iou: visible_proposal 集合的 Jaccard distance.
    pose_depth: camera xyz + yaw 在已见集合中的最远.
    seen_frame_ids=None → 用 runtime.seen_image_paths 反推."""
```

### 场景认知工具

```python
def view_bev(highlight: list[int] | None = None) -> str:
    """重渲 BEV. highlight=None → 标全部 proposal; 给 ids → 只标这些.
    Queues bev_image_path to vg_pending_images."""

def list_scene_proposals(
    category: str | None = None,
    region_bev: list[float] | None = None,
    limit: int | None = None,
) -> str:
    """过滤 SceneCatalog. category 精确匹配; region_bev=[xmin,ymin,xmax,ymax] 过滤 position_3d[:2]."""

def inspect_proposal(proposal_id: int) -> str:
    """Returns: {proposal_id, category, position_3d, bbox_3d_9dof, frames_appeared, source}."""
```

### 视觉证据工具

```python
def view_keyframe(
    frame_id: int,
    mode: Literal["rgb", "marked", "auto"] = "auto",
) -> str:
    """mode='auto' → runtime.task_type (VG='marked', QA='rgb').
    Queues image path to vg_pending_images."""

def request_crops(request_text: str, object_terms: list[str]) -> str:
    """保留，现状行为不变（CLIP-fallback zoom crop）."""
```

### 空间推理

```python
def compare_proposals_spatial(
    candidate_ids: list[int],
    anchor_id: int,
    relation: Literal["closest_to", "near", "next_to", "farthest_from",
                       "above", "below", "left_of", "right_of"],
) -> str:
    """现状字段保留 (ranked_ids, distances, ..., supporting_frame_counts, contradicting_frame_counts).
    left_of/right_of 仍用 co-viewed 2D frame 投票."""
```

### 终结

```python
def submit_final(
    payload: dict,             # VG: {proposal_id, confidence}; QA: {answer, supporting_claims}
    rationale: str,
    evidence_refs: list[dict] | None = None,
    tool_override_reason: str | None = None,
) -> str:
    """VG: TADG → no_match_guard → evidence_frame_guard → VgFinalizerSpec (proposal_id ∈ catalog).
    QA: TADG → evidence_frame_guard → QaFinalizerSpec (answer 非空).
    QA 跳过 no_match_guard (没有 OOD 概念)."""
```

### 关键行为合约

**图像注入路径**：只有 3 个工具会把图加入 `vg_pending_images` 队列：`view_keyframe`、`view_bev`、`request_crops`。所有 `select_by_*` 工具不直接注入图（Path A 决策）。

**Skill gate**：所有 selector + view_keyframe + view_bev + list_scene_proposals + compare_proposals_spatial + inspect_proposal 被 `_gate(runtime)` 保护，必须先 `load_skill("scene-exploration-playbook")`。

**Guard 适配**：
- TADG：现状逻辑保持，仅术语换为 proposal_id
- no_match_guard：仅 VG（QA 无 OOD 概念）
- evidence_frame_guard：VG / QA 都用；rationale 引用 frame N 但 submit 不在 N 的 visible_proposal_ids 时阻断

---

## Section D — 初始 HumanMessage 模板 + Playbook 策略

### 初始 HumanMessage 的 content 结构

```python
content = [
    {"type": "text", "text": <task_prompt>},
    {"type": "image_url", "image_url": {"url": data_url(bev_image_path)}},
]
```

**没有 first-person 图**。BEV 是唯一的初始图。

### `<task_prompt>` 模板（VG 示例）

```text
## Task
Task type: visual_grounding
Plan mode: brief
User query: "this is a brown wooden chair next to the kitchen counter"
Output instruction: Submit the proposal_id of the matched object via submit_final.

Expected payload schema:
{"proposal_id": int, "confidence": float}

## Scene
Scene id: scannet/scene0123_45
Scene category: kitchen-living-room
Total frames: 187 (frame_id range: [0, 1860], stride 10)
Proposal pool: 47 items, source=mask3d

Proposals by category:
  chair: [#4, #5, #17, #23, #28, #31, #35, #38, #41, #44, #46, #47]
  table: [#1, #8, #19]
  sofa:  [#2, #9]
  lamp:  [#11, #20, #36, #45]
  refrigerator: [#13]
  ...

## BEV image (attached above)
- mesh-based top-down render with camera trajectory (blue line)
- each proposal labeled "#id category" at its 3D center
- use view_bev(highlight=[ids]) to declutter

## Available tools
You have 15 tools. Always load_skill("scene-exploration-playbook") first,
then load_skill("vg-grounding-playbook") for VG-specific guidance.

The BEV labels are not first-person evidence.
You currently have viewed 0 keyframes out of 187.
Use selectors + view_keyframe to fetch first-person frames.
```

QA 任务的 task_prompt 结构一致，仅 task section（query / instruction / payload schema）换成 QA 的；BEV note 改成 "use view_keyframe(mode='rgb') for first-person scene observation"。

### Playbook 重组

```
src/agents/skills/shared_skills/
  └─ scene_exploration_playbook.md      [新]  ← VG + QA 共用 prerequisite

src/agents/packs/vg_embodiedscan/skills/
  ├─ vg_grounding_playbook.md           [重写] ← VG-specific 流程 + guards
  └─ vg_spatial_disambiguation.md       [小改] ← 工具名 proposal* 不变

src/agents/packs/qa_default/skills/
  ├─ qa_answering_playbook.md           [重写] ← QA-specific 流程 + view_keyframe(mode='rgb')
  └─ evidence_scouting.md               [删除]
```

**`scene-exploration-playbook` 内容大纲**（VG/QA 共用 prerequisite）：
- 心智模型：BEV + catalog 不是答案；first-person 是；必须 fetch
- 6 个 selector 的何时用 / cost / 例子（cheapest-first 排序）
- view_keyframe / view_bev / inspect_proposal / list_scene_proposals 的何时用
- 通用 anti-pattern

**`vg_grounding_playbook` 内容大纲**（VG 专属）：
- VG 流程：BEV 看 #id → select_by_proposal → view_keyframe(marked) → 验证
- TADG / no_match_guard / evidence_frame_guard 触发与处理
- OOD（proposal_id = -1）

**`qa_answering_playbook` 内容大纲**（QA 专属）：
- QA 流程：BEV → select_by_text 找 task-relevant frames → view_keyframe(rgb) → 描述
- supporting_claims 怎么 cite proposal / frame
- counting / color / state 等各类问题的策略

### Skill load 顺序

Agent 必须先 `load_skill("scene-exploration-playbook")` 解所有 selector / view tool 的 gate，再 `load_skill("vg-grounding-playbook")` 或 `load_skill("qa-answering-playbook")` 拿任务专属指南。

### Prompt cache 友好性

`<task_prompt>` 里 "Scene" / "BEV image" / "Available tools" / skill 说明四部分是 **scene-level + system-level 信息**，每个 sample 在同一 scene 里完全相同——可被 prompt cache 复用。task-specific 部分（query / instruction / payload schema）约 150-300 token，cache miss 成本低。

预估每个 sample 第一轮 prompt：scene-level header ~1500-2500 token（含 BEV image）+ task-specific ~150-300 token。

---

## Section E — 迁移计划 + 测试 / Ablation 策略

### 文件级 Migration 清单

**删除（M1 硬切换）**：
```
src/agents/stage1_callbacks.py
  ├─ create_more_views_callback        [删]
  ├─ create_hypothesis_callback        [删]
  └─ create_crop_callback              [保留]

src/agents/runtime/deepagents_agent.py
  ├─ _request_more_views_impl + tool wrapper [删 ~80 lines]
  ├─ switch_or_expand_hypothesis wrapper      [删 ~40 lines]
  └─ inspect_stage1_metadata                  [删]

src/agents/packs/vg_embodiedscan/tools.py
  ├─ list_keyframes_with_proposals     [删]
  ├─ view_keyframe_marked              [删]
  ├─ find_proposals_by_category        [删]
  └─ (inspect_proposal / compare_proposals_spatial 保留，内部改读 SceneCatalog)

src/agents/packs/qa_default/skills/evidence_scouting.md  [删]

src/agents/core/agent_config.py
  ├─ enable_temporal_fan               [删]
  └─ enable_stage1_callback            [删]
```

**新增**：
```
src/agents/catalog/                          [新模块]
  ├─ models.py
  ├─ adapters.py
  └─ tests/

src/agents/tools/selectors.py                [新] 6 个 select_by_* + view_bev / list_scene_proposals
src/agents/tools/view_keyframe.py            [新] 合并 mode=auto/rgb/marked
src/agents/skills/shared_skills/scene_exploration_playbook.md   [新]

src/query_scene/bev_builder.py
  ├─ ScanNetSceneBEVBuilderBase
  ├─ Nr3dScanNetBEVBuilder
  ├─ ScanReferScanNetBEVBuilder
  ├─ OpenEqaScanNetBEVBuilder          [从 OpenEQAScanNetBEVBuilder 改写]
  └─ Sqa3dScanNetBEVBuilder

src/evaluation/scripts/prepare_pack_qa_inputs.py   [新] OpenEQA / SQA3D
```

**重写**：
```
src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py        [改]
src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py   [改]
src/agents/runtime/deepagents_agent.py                       [改 build_task_message]
src/agents/runtime/base.py                                   [改 collect_image_paths]
src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md   [重写]
src/agents/packs/qa_default/skills/qa_answering_playbook.md         [重写]
```

### Pack 版本命名

- `pack_nr3d_v9_catalog_first`
- `pack_scanrefer_v9_catalog_first`
- `pack_openeqa_v9_catalog_first`
- `pack_sqa3d_v9_catalog_first`

旧 v8 pack 在磁盘上保留，用于对照实验。

### 数据增量生成

可从 v8 pack 增量生成 v9 pack：v8 已有 proposals.jsonl + frame_views + annotated/，v9 增量只需要 BEV + scene_catalog.json。沿用 `scripts/prepare_nr3d_rerender_pack_from_existing.py` 模式。

### 测试矩阵

**单元测试（必须先通过）**：

| 模块 | 测试文件 |
|---|---|
| catalog 模型 | `tests/test_scene_catalog_models.py` |
| VG adapter | `tests/test_scene_catalog_adapter_vg.py` |
| QA adapter | `tests/test_scene_catalog_adapter_qa.py` |
| BEV builder (4 子类) | `tests/test_nr3d_scannet_bev_builder.py` 等 |
| 6 个 selector | `tests/test_selectors_*.py` |
| `view_keyframe` | `tests/test_view_keyframe_v9.py` |
| `view_bev` | `tests/test_view_bev.py` |
| 初始 prompt | `tests/test_build_task_message_v9.py` |
| Pack prep | `tests/test_prepare_*_v9.py` |

**端到端测试**：mock VLM 走一遍 ReAct loop，验证工具图谱完整可达，per benchmark 一个。

### Ablation Plan

#### Phase 1: VG headline (NR3D random100, 同 v7/v8 fold)

```
v9_full         所有 6 selector + view_keyframe + view_bev + request_crops
v9_no_view_bev  禁用 view_bev 重渲
v9_text_only    仅 select_by_text（砍 B/C/D/E/F）
v9_no_selector  禁用 6 selector（agent 退化到只能看初始 BEV）
```

对照基线：v7 callbacks no-CLIP (73.00) / v8 clean initial (67.00)。

主表（同 NR3D random100 fold）：

| Run | Overall | Easy | Hard | View-Dep | View-Indep |
|---|---|---|---|---|---|
| v7 callbacks no-CLIP | 73.00 | 82.93 | 66.10 | 70.59 | 74.24 |
| v8 clean initial | 67.00 | 85.37 | 54.24 | 58.82 | 71.21 |
| v9_full | TBD | | | | |
| v9_no_view_bev | TBD | | | | |
| v9_text_only | TBD | | | | |
| v9_no_selector | TBD | | | | |

#### Phase 2: VG ScanRefer random100

同 4 个 ablation 跑一遍。ScanRefer 没有现成 v7/v8 random100 baseline，v9 是 ScanRefer 的开局基线。

#### Phase 3: QA (OpenEQA + SQA3D)

独立 fold（建议 OpenEQA random100 + SQA3D random100）。对照基线：当前 chassis pipeline（v15 系列）。

```
v9_qa_full      所有工具
v9_qa_no_bev    禁用 view_bev
v9_qa_no_select 禁用 6 selector
```

Judge model：`gemini-2.5-pro`。

#### Phase 4: Per-selector ablation（关键验证）

leave-one-out：

| Run | 禁用 | 期望发现 |
|---|---|---|
| v9_drop_A | `select_by_text` | text query 是否仍是主路径 |
| v9_drop_B | `select_by_hypothesis` | agent 是否真用 (低使用率 → 删) |
| v9_drop_C | `select_by_frame_neighbor` | temporal/viewpoint 的贡献 |
| v9_drop_D | `select_by_proposal` | by-id fetch 是否核心 |
| v9_drop_E | `select_by_region` | region 使用率 |
| v9_drop_F | `select_by_coverage` | coverage 是否真有 marginal gain |

每个 leave-one-out 跑 NR3D random100，统计 selector 调用频率 + headline 退化。**调用频率 < 5% + metric 不退化 → 删掉该 selector**（YAGNI）。

### SQLite 记录约束（CLAUDE.md）

每个 ablation run 必须：
1. `scripts/ingest_nr3d_run.py` / `scripts/ingest_openeqa_run.py` 进 SQLite
2. 写独立 `<v9-tag>_<date>.md` 到 `docs/benchmark/<benchmark>/`
3. 更新 README + leaderboard

### 上线门槛

v9_full 必须满足以下才能 default-on：
- NR3D random100 overall ≥ v7 (73.00)
- ScanRefer random100 overall ≥ 现有 v3.19 baseline
- OpenEQA random100 MNAS ≥ 当前 chassis baseline
- 单 run 平均 turn 数 ≤ v7 的 1.5x

不满足 → 调 prompt / playbook 不修工具；3 轮没改善则 escalate 到 design 复盘。

---

## Section F — System Prompt & Skill 一致性 Audit

**Tool 的删除 / 重命名 / 新增不只是改代码**——agent 的 system prompt、task prompt、所有 skill markdown 都硬编码了旧工具名 / 旧 ReAct 模式 / 旧 seed 假设。**任何一个地方漏改，v9 的 agent 第一轮就会被错误的指令带跑**。这一节列出每个必须同步更新的位置和具体行为。

### F.1 System Prompt（`src/agents/runtime/base.py::build_system_prompt`）

**当前**（line 475-536）整段教 agent 用旧 callback：

```
Tool strategy (use in this order):
1. request_more_views(mode='targeted', object_terms=[...]) — ...
2. request_more_views(mode='explore') — ...
3. request_more_views(mode='temporal_fan', ...) — ...
4. request_crops(object_terms=[...]) — ...
5. switch_or_expand_hypothesis(new_query='...') — ...

MANDATORY tool-usage rules:
- If the target object is NOT visible in ANY keyframe, you MUST call request_more_views ...
- For ALL color/attribute questions, you MUST call request_crops on the target object BEFORE answering. ...
- For YES/NO state questions ..., you MUST request_crops or request_more_views ...
```

**v9 必须整段重写**，关键变化：
- 删除所有 `request_more_views` / `switch_or_expand_hypothesis` 引用
- 删除 "ALWAYS examine the provided keyframe images FIRST"（v9 没有 initial keyframes，agent 看到的是 BEV）
- 删除 `temporal_fan_line` 拼接逻辑（line 463-470）以及 `crops_index` / `hypothesis_index` 动态编号
- 新增 catalog-first 心智模型说明："You have NOT viewed any first-person frame yet. The BEV image + SceneCatalog text are your scene overview. Use selectors + view_keyframe to fetch first-person frames."
- 新增 6 个 selector 的 cheapest-first 提示
- 重申 `load_skill("scene-exploration-playbook")` 是任何 selector / view_keyframe / view_bev 调用的前置

**新结构建议**：
```
You are the Stage-2 scene reasoning agent.

Scene perception model:
- You start with a BEV overview image + SceneCatalog text (Cat-B by category).
- You have viewed 0 first-person frames out of {total_frames}.
- The BEV labels are a starting point, not first-person evidence.

Tool families (load_skill before use):
1. select_by_* (6 modalities) — find task-relevant frame_ids
2. view_keyframe(frame_id, mode='auto') — inject a first-person frame
3. view_bev(highlight=[ids]) — declutter the BEV by highlighting subset
4. list_scene_proposals / inspect_proposal — query catalog
5. compare_proposals_spatial — spatial reasoning (VG-heavy)
6. request_crops — zoom into small attribute / state
7. submit_final — terminate (guards run before adapter)

Cheapest-first principle:
- select_by_proposal / select_by_frame_neighbor / select_by_region: instant (catalog lookup)
- select_by_coverage: cheap (geometric)
- select_by_text: ~Stage 1 LLM parse cost (2-5s)
- select_by_hypothesis: instant (no parse)
- request_crops: moderate (CLIP fallback)
```

并删除 `enable_temporal_fan` / `crops_index` / `hypothesis_index` 等 config-driven 拼接，**改为静态文本**——v9 工具集合是稳定的，不需要 runtime flag。

### F.2 Task Prompt（`build_task_message`）

已在 Section D 详细描述。关键审计点：
- 删除 "Current keyframes:" 段（line 368）—— v9 没有 first-person seed
- 删除 v8 的 `vg_initial_inventory_section`（已被 Cat-B 取代）
- 新增 BEV image as image_url content block（不是 keyframe paths）
- 新增 "Proposals by category" 段（Cat-B 格式）
- 新增 "You have viewed 0 keyframes out of N. Use selectors + view_keyframe to fetch first-person frames."

### F.3 Skill Markdown 文件（全清单）

| 文件 | 当前引用的旧工具 | v9 动作 |
|---|---|---|
| `src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md` | `view_keyframe_marked`, `find_proposals_by_category`, `list_keyframes_with_proposals`, `inspect_proposal`, `compare_proposals_spatial`, `request_more_views`, `request_crops`, `switch_or_expand_hypothesis` | **整段重写**（保留 TADG / no_match_guard / evidence_frame_guard 流程描述）；改用新工具名 |
| `src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation.md` | `compare_proposals_spatial`, possibly others | **小改**（工具名不变；删去 view_keyframe_marked 引用，改 view_keyframe(mode='marked')） |
| `src/agents/packs/vg_embodiedscan/skills/evidence_scouting.md` | 旧 callback 流程 | **删除**（被 scene_exploration_playbook 取代） |
| `src/agents/packs/qa_default/skills/qa_answering_playbook.md` | `request_more_views`, `request_crops`, `switch_or_expand_hypothesis`, `retrieve_object_context`, `inspect_stage1_metadata` | **整段重写**；改用 selector + view_keyframe(mode='rgb') |
| `src/agents/packs/qa_default/skills/evidence_scouting.md` | 旧 callback 流程 | **删除**（被 scene_exploration_playbook 取代） |
| `src/agents/skills/shared_skills/task_head_output.md` | 可能引用旧工具 | **审查 + 按需小改** |
| `src/agents/skills/shared_skills/scene_exploration_playbook.md` | — | **新写**（v9 唯一的 gate-解锁 skill） |

### F.4 其它代码点（非 prompt / 非 skill）

| 文件 | 影响 | 动作 |
|---|---|---|
| `src/agents/packs/vg_embodiedscan/tools.py` | tool docstring 引用 `vg-grounding-playbook` skill | 改文案，gate 检查改为 `scene-exploration-playbook` |
| `src/agents/skills/tadg.py` | 引用 `compare_proposals_spatial` 工具名 | 验证（名字未变），但若内部用 `find_proposals_by_category` 要换 |
| `src/agents/skills/no_match_guard.py` | 引用 `find_proposals_by_category` / `view_keyframe_marked` 检查 trace | **改读新工具名**：`list_scene_proposals` / `view_keyframe`（mode='marked'） |
| `src/agents/skills/evidence_frame_guard.py` | 引用 `view_keyframe_marked` 的 evidence_refs | **改读 `view_keyframe`**（mode='marked' 才计入 marked 证据） |
| `src/evaluation/visualizations.py` | trace HTML 渲染按工具名分类 | 添加新工具名 |
| `src/evaluation/scripts/run_*_side_by_side.py` | 显式创建 `create_more_views_callback` / `create_hypothesis_callback` | 删除这两行；保留 `create_crop_callback` |
| `src/agents/runtime/deepagents_agent.py` | `tool_descriptions` 拼接 | 整体重写工具列表 |
| 所有现有测试（`test_tadg.py` / `test_evidence_frame_guard.py` / `test_no_match_guard.py` / `test_stage2_deep_agent.py` / `test_trace_html.py` / `test_batch_eval.py` 等） | mock 调用旧工具名 | 按需更新到新工具名 |

### F.5 Verification grep（实施完毕前必跑）

实施 v9 完毕后，**必须**跑以下 grep 验证无遗漏的旧工具引用：

```bash
# 在 src/ 和 docs/ 下都不应再有这些字符串
rg -l 'request_more_views|switch_or_expand_hypothesis|view_keyframe_marked' src/ docs/
rg -l 'find_proposals_by_category|list_keyframes_with_proposals|inspect_stage1_metadata' src/ docs/
rg -l 'enable_temporal_fan|enable_stage1_callback' src/ docs/

# 期望：全部返回 0 个文件（除了 v9 spec / migration doc / archived benchmark docs 本身）
```

任何残留必须 fix 或在 deprecation 注释里显式标注（如老的 benchmark doc 引用历史工具名是允许的，但活跃代码不允许）。

### F.6 黄金原则

**Tool 集合 = sys prompt + skill markdown + tool docstring + guard 内部引用 这四个层面必须严格一致**。任何一处提到 tool X，则 tool X 必须真实存在；删除 tool X 时，这四处全部要同步删除。Section E 的"删除清单 / 新增清单 / 重写清单"涵盖代码层面；Section F 涵盖文案 + 测试层面。两者合起来才是完整的 v9 迁移。

---

## 风险 / 注意事项

1. **catalog-first 第一轮全靠 BEV + 文本**：v8 random100 已经显示去掉 marked seed 会让 Hard/View-Dep 退步。v9 的赌注是：完整 catalog（含全部 #id label 在 BEV 上 + Cat-B 文本）比 v8 的局部 inventory 更强。如果 v9_full 在 NR3D random100 上仍然达不到 v7 水平，需要回炉考虑：(a) 在 BEV-only 初始之外加一张 BEV-traj-overhead 之类的辅助图；(b) 强制 playbook 在 view-dep 句子上必须先 view 一帧才能 submit。

2. **BEV 标号过密**：NR3D 场景常 40-60 proposal。若全标在 1500×1500 BEV 上挤到不可读，agent 必须靠 `view_bev(highlight=...)` 重渲。如果 agent 不学这个习惯，初始 BEV 信息过载 → 降级到 v9_no_view_bev 路径。playbook 必须明确教这一招。

3. **QA conceptgraph 数据可用性**：OpenEQA 89 个 prepared scene 都有 conceptgraph 输出，但 SQA3D 的 scene 集合不一定都跑过 conceptgraph。Phase 3 之前需要 audit 哪些 SQA3D scene 缺 conceptgraph，缺的要补跑或剔除。

4. **6 个 selector 的工具命名空间膨胀**：DeepAgents 默认能 work 20+ tool，但 prompt 里 tool 说明会变长。playbook 必须精炼，避免 docstring 过度展开。

5. **Hypothesis 状态删除的副作用**：删 `list_active_hypotheses` 后 agent 不能"反思之前用过哪些 query"。如果 trace 显示 agent 反复重复同样的 `select_by_text(query)` 查询（缓存命中），可以考虑 v9.1 加 dedup（同 query 第二次 → 直接报错"该 query 已查过"）。

6. **request_crops 的存续**：仍依赖 Stage 1 的 CLIP-fallback 路径。若未来 CLIP 模型迁移，要同步更新。

## 测试

目标测试覆盖：
- SceneCatalog Pydantic round-trip
- VG/QA adapter 字段保真
- 4 个 BEV builder 子类各自渲染输出 png + cache 命中
- 6 个 selector 输入合法返回 JSON shape；空结果 / 错误参数 fail-loud
- `view_keyframe(mode=auto)` 在 VG / QA 任务下的默认行为
- 初始 prompt 模板：无 first-person seed；BEV image 注入；Cat-B 文本格式正确
- Pack prep：scene_catalog.json + BEV png 写出；sample JSON 不含 `vg_proposal_pool` / `keyframes`
- 端到端 mock VLM：VG / QA 各跑一遍完整 ReAct loop

代码改完后，先跑 focused unit tests，再跑固定 NR3D random100 对比；在这些验证完成前，不基于此设计宣称 benchmark 指标提升。

## 后续 spec

本 spec 不在 scope 但已明确推迟：
- **v9.Nav**：选 Nav benchmark（候选：Habitat-VLN-CE / R2R-CE）+ 设计 Nav pack
- **v9.Manip**：选 Manipulation benchmark + 设计 Manip pack
- **v9.1**：基于 Phase 4 per-selector ablation 结果做 selector 精简；按需加 query dedup

这两个 spec 独立写，**不在 v9 主 spec 落地**。
