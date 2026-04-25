# Plan A — chassis 基础与 EmbodiedScan VG pack v1 落地

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 spec 的 step 1–6 落到代码:在 `Stage2DeepAgentConfig` /
`Stage2RuntimeState` / chassis 上加任务无关基元,实现一个完整的
`vg_embodiedscan` pack(5 个 tool + 3 个 skill + ctx + finalizer +
validate_packs),在 `vg_backend='pack_v1'` 下走得通,并在 EmbodiedScan
ScanNet val 30 样本子集上跟 legacy VG 对齐 metric。OpenEQA QA 全程零回归。

**Architecture:** 注册中心 (`src/agents/skills/registry.py`) + chassis
tool (`src/agents/skills/chassis_tools.py`) + 第一个 pack
(`src/agents/packs/vg_embodiedscan/`) + 现有 `DeepAgentsStage2Runtime` 的
最小切入点扩展。所有改动门控在 `vg_backend='pack_v1'` 与新的
`enable_chassis_tools` 后面;默认值保持现状,QA 字节级稳定。

**Tech Stack:** Python 3.11、Pydantic v2、LangChain v1、DeepAgents、
pytest、Pillow / numpy / open3d(VG-pack 依赖,均已存在)。

**Source spec:** `docs/superpowers/specs/2026-04-25-stage2-multi-task-agent-design.md`

**Companion review artifacts:** `~/.super-orchestrator/agent-arch/`

**Branch convention:** 每完成 1 个 task 就 commit。每完成 1 章
("Foundation" / "VG pack v1" / "Validation") 推一个 PR。

---

## File Structure

新增文件:

- `src/agents/skills/__init__.py` — 重导出 SkillSpec / TaskPack / register_pack / PACKS
- `src/agents/skills/registry.py` — 注册中心
- `src/agents/skills/chassis_tools.py` — list_skills / load_skill / submit_final
- `src/agents/skills/finalizer.py` — FinalizerSpec dataclass
- `src/agents/packs/__init__.py` — 触发所有 pack 自注册
- `src/agents/packs/vg_embodiedscan/__init__.py`
- `src/agents/packs/vg_embodiedscan/registration.py`
- `src/agents/packs/vg_embodiedscan/ctx.py`
- `src/agents/packs/vg_embodiedscan/finalizer.py`
- `src/agents/packs/vg_embodiedscan/tools.py`
- `src/agents/packs/vg_embodiedscan/proposal_pool.py`
- `src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md`
- `src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation.md`
- `src/agents/packs/vg_embodiedscan/skills/evidence_scouting.md`
- `src/agents/examples/embodiedscan_vg_pack_v1_pilot.py`
- `src/evaluation/scripts/run_embodiedscan_vg_side_by_side.py`
- 对应 `tests/` 文件

修改文件:

- `src/agents/core/agent_config.py` — 加 `enable_chassis_tools`,
  `vg_backend`, `chassis_tools_version`
- `src/agents/runtime/base.py` — `Stage2RuntimeState` 加
  `task_ctx`, `skills_loaded`;`build_system_prompt` 注入 skill catalog
- `src/agents/runtime/deepagents_agent.py` — `build_agent` 调用
  `validate_packs`,`build_runtime_tools` 在 `vg_backend='pack_v1'` 下挂
  pack tools + chassis trio
- `src/agents/tools/select_object.py:50-56` — 删 silent fallback
- `src/agents/examples/openeqa_official_question_pilot.py:280-292` —
  `derive_eval_session_id` 哈希新增 `chassis_tools_version` 与 `vg_backend`
- `src/agents/tests/test_stage2_deep_agent.py` — 加 snapshot test 锁工具列表

---

## Section 1 — Foundation (steps 1-4 of spec)

### Task 1 — 新增 `SkillSpec` / `TaskPack` 数据类与 `PACKS` 注册中心

**Files:**
- Create: `src/agents/skills/__init__.py`
- Create: `src/agents/skills/registry.py`
- Create: `src/agents/skills/finalizer.py`
- Test:   `src/agents/tests/test_skills_registry.py`

- [ ] **Step 1.1: 写失败测试**

`src/agents/tests/test_skills_registry.py`:

```python
"""Registry: SkillSpec/TaskPack/FinalizerSpec wiring."""
from __future__ import annotations

from pathlib import Path

import pytest

from agents.core.agent_config import Stage2TaskType
from agents.skills import (
    FinalizerSpec,
    PACKS,
    SkillSpec,
    TaskPack,
    register_pack,
    skills_for,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    PACKS.clear()
    yield
    PACKS.clear()


def _stub_finalizer() -> FinalizerSpec:
    return FinalizerSpec(
        payload_model=dict,
        validator=lambda payload, runtime: None,
        adapter=lambda payload, runtime: {},
    )


def test_register_pack_stores_pack_by_task_type(tmp_path: Path) -> None:
    body = tmp_path / "skill.md"
    body.write_text("# stub", encoding="utf-8")
    skill = SkillSpec(
        name="vg-grounding-playbook",
        description="stub",
        body_path=body,
        task_types={Stage2TaskType.VISUAL_GROUNDING},
    )
    pack = TaskPack(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        tool_builder=lambda runtime: [],
        skills=[skill],
        finalizer=_stub_finalizer(),
        required_primary_skill="vg-grounding-playbook",
        required_extra_metadata=["vg_proposal_pool"],
        ctx_factory=lambda bundle: {},
    )
    register_pack(pack)
    assert PACKS[Stage2TaskType.VISUAL_GROUNDING] is pack


def test_register_pack_rejects_duplicate(tmp_path: Path) -> None:
    body = tmp_path / "skill.md"
    body.write_text("# stub", encoding="utf-8")
    pack = TaskPack(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        tool_builder=lambda runtime: [],
        skills=[],
        finalizer=_stub_finalizer(),
        required_primary_skill="x",
        required_extra_metadata=[],
        ctx_factory=lambda bundle: {},
    )
    register_pack(pack)
    with pytest.raises(RuntimeError, match="duplicate pack"):
        register_pack(pack)


def test_skills_for_filters_by_task_type(tmp_path: Path) -> None:
    body = tmp_path / "skill.md"
    body.write_text("# stub", encoding="utf-8")
    vg_skill = SkillSpec(
        name="vg-grounding-playbook",
        description="vg",
        body_path=body,
        task_types={Stage2TaskType.VISUAL_GROUNDING},
    )
    qa_skill = SkillSpec(
        name="qa-answering-playbook",
        description="qa",
        body_path=body,
        task_types={Stage2TaskType.QA},
    )
    register_pack(
        TaskPack(
            task_type=Stage2TaskType.VISUAL_GROUNDING,
            tool_builder=lambda r: [],
            skills=[vg_skill, qa_skill],
            finalizer=_stub_finalizer(),
            required_primary_skill="vg-grounding-playbook",
            required_extra_metadata=[],
            ctx_factory=lambda b: {},
        )
    )
    names = [s.name for s in skills_for(Stage2TaskType.VISUAL_GROUNDING)]
    assert names == ["vg-grounding-playbook"]
```

- [ ] **Step 1.2: 跑测试确认 ImportError**

```bash
pytest src/agents/tests/test_skills_registry.py -v
```

期望: 因 `agents.skills` 不存在而失败。

- [ ] **Step 1.3: 实现 `src/agents/skills/finalizer.py`**

```python
"""Per-task finalization contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Type


@dataclass(frozen=True)
class FinalizerSpec:
    """How a pack validates + adapts its `submit_final` payload."""

    payload_model: Type[Any]
    validator: Callable[[Any, Any], Any]   # (payload, runtime) -> resolved
    adapter: Callable[[Any, Any], dict]    # (payload, runtime) -> dict for Stage2StructuredResponse


__all__ = ["FinalizerSpec"]
```

- [ ] **Step 1.4: 实现 `src/agents/skills/registry.py`**

```python
"""SkillSpec / TaskPack registry shared across Stage-2 packs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import BaseTool

from agents.core.agent_config import Stage2TaskType
from agents.skills.finalizer import FinalizerSpec


@dataclass(frozen=True)
class SkillSpec:
    """A loadable skill: catalog entry + lazy markdown body."""

    name: str
    description: str
    body_path: Path
    task_types: frozenset[Stage2TaskType] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # Allow set inputs by normalizing to frozenset
        object.__setattr__(self, "task_types", frozenset(self.task_types))


@dataclass(frozen=True)
class TaskPack:
    """Per-task plug-in: tools, skills, finalizer, ctx factory."""

    task_type: Stage2TaskType
    tool_builder: Callable[[Any], list[BaseTool]]
    skills: list[SkillSpec]
    finalizer: FinalizerSpec
    required_primary_skill: str
    required_extra_metadata: list[str]
    ctx_factory: Callable[[Any], Any]


PACKS: dict[Stage2TaskType, TaskPack] = {}


def register_pack(pack: TaskPack) -> None:
    if pack.task_type in PACKS:
        raise RuntimeError(f"duplicate pack: {pack.task_type}")
    PACKS[pack.task_type] = pack


def skills_for(task_type: Stage2TaskType) -> list[SkillSpec]:
    pack = PACKS.get(task_type)
    if pack is None:
        return []
    return [s for s in pack.skills if task_type in s.task_types]


__all__ = [
    "SkillSpec",
    "TaskPack",
    "FinalizerSpec",
    "PACKS",
    "register_pack",
    "skills_for",
]
```

- [ ] **Step 1.5: 实现 `src/agents/skills/__init__.py`**

```python
"""Re-export the registry surface."""
from agents.skills.finalizer import FinalizerSpec
from agents.skills.registry import (
    PACKS,
    SkillSpec,
    TaskPack,
    register_pack,
    skills_for,
)

__all__ = [
    "FinalizerSpec",
    "PACKS",
    "SkillSpec",
    "TaskPack",
    "register_pack",
    "skills_for",
]
```

- [ ] **Step 1.6: 跑测试确认 PASS**

```bash
pytest src/agents/tests/test_skills_registry.py -v
```

- [ ] **Step 1.7: commit**

```bash
git add src/agents/skills/ src/agents/tests/test_skills_registry.py
git commit -m "feat(agents/skills): introduce TaskPack/SkillSpec/FinalizerSpec registry"
```

---

### Task 2 — `Stage2RuntimeState` 加 `task_ctx` 与 `skills_loaded` (additive)

**Files:**
- Modify: `src/agents/runtime/base.py:32-71`
- Test:   `src/agents/tests/test_runtime_state_additive.py`

- [ ] **Step 2.1: 写失败测试**

`src/agents/tests/test_runtime_state_additive.py`:

```python
"""Additive task_ctx + skills_loaded fields on Stage2RuntimeState."""
from __future__ import annotations

from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState


def test_task_ctx_defaults_to_none() -> None:
    rs = Stage2RuntimeState(bundle=Stage2EvidenceBundle())
    assert rs.task_ctx is None


def test_skills_loaded_defaults_empty_set() -> None:
    rs = Stage2RuntimeState(bundle=Stage2EvidenceBundle())
    assert rs.skills_loaded == set()


def test_existing_vg_fields_unchanged() -> None:
    rs = Stage2RuntimeState(bundle=Stage2EvidenceBundle())
    assert rs.vg_scene_objects is None
    assert rs.vg_axis_align_matrix is None
    assert rs.vg_selected_object_id is None
    assert rs.vg_selected_bbox_3d is None
    assert rs.vg_selection_rationale == ""


def test_skills_loaded_is_per_instance() -> None:
    a = Stage2RuntimeState(bundle=Stage2EvidenceBundle())
    b = Stage2RuntimeState(bundle=Stage2EvidenceBundle())
    a.skills_loaded.add("vg-grounding-playbook")
    assert "vg-grounding-playbook" not in b.skills_loaded
```

- [ ] **Step 2.2: 跑测试确认失败**

```bash
pytest src/agents/tests/test_runtime_state_additive.py -v
```

期望: `AttributeError` on `rs.task_ctx`.

- [ ] **Step 2.3: 修改 `Stage2RuntimeState`**

`src/agents/runtime/base.py`,在第 49 行 `vg_selection_rationale: str = ""` 之后追加:

```python
    # Pack-v1 additive fields (kept alongside legacy vg_* until step 9)
    task_ctx: Any | None = None
    skills_loaded: set[str] = field(default_factory=set)
```

- [ ] **Step 2.4: 跑测试确认 PASS**

```bash
pytest src/agents/tests/test_runtime_state_additive.py -v
pytest src/agents/tests/test_stage2_deep_agent.py -v   # 既有测试不能挂
```

- [ ] **Step 2.5: commit**

```bash
git add src/agents/runtime/base.py src/agents/tests/test_runtime_state_additive.py
git commit -m "feat(agents/runtime): add task_ctx + skills_loaded to Stage2RuntimeState (additive)"
```

---

### Task 3 — `Stage2DeepAgentConfig` 加 `enable_chassis_tools`、`vg_backend`、`chassis_tools_version`

**Files:**
- Modify: `src/agents/core/agent_config.py:40-77`
- Test:   `src/agents/tests/test_agent_config_flags.py`

- [ ] **Step 3.1: 写失败测试**

`src/agents/tests/test_agent_config_flags.py`:

```python
"""New chassis-related flags on Stage2DeepAgentConfig."""
from __future__ import annotations

import pytest

from agents.core.agent_config import Stage2DeepAgentConfig


def test_enable_chassis_tools_defaults_off() -> None:
    cfg = Stage2DeepAgentConfig()
    assert cfg.enable_chassis_tools is False


def test_vg_backend_defaults_legacy() -> None:
    cfg = Stage2DeepAgentConfig()
    assert cfg.vg_backend == "legacy"


def test_chassis_tools_version_default_is_int() -> None:
    cfg = Stage2DeepAgentConfig()
    assert isinstance(cfg.chassis_tools_version, int)


def test_vg_backend_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        Stage2DeepAgentConfig(vg_backend="bogus")
```

- [ ] **Step 3.2: 跑测试确认失败**

```bash
pytest src/agents/tests/test_agent_config_flags.py -v
```

期望: 全部 fail with `ValidationError` or `AttributeError`.

- [ ] **Step 3.3: 添加字段**

`src/agents/core/agent_config.py`,在第 77 行 `enable_temporal_fan` 字段之后添加:

```python
    enable_chassis_tools: bool = Field(
        default=False,
        description="Register chassis trio (list_skills, load_skill, submit_final) "
        "for tasks without a registered TaskPack. Default OFF preserves QA byte-stable.",
    )
    vg_backend: Literal["legacy", "pack_v1"] = Field(
        default="legacy",
        description="Which VG code path to run: legacy if/else branch or new TaskPack.",
    )
    chassis_tools_version: int = Field(
        default=1,
        ge=1,
        description="Bump when chassis tool surface changes; folded into "
        "derive_eval_session_id so prompt-cache invalidates correctly.",
    )
```

并在文件顶部 `from typing import Any` 后追加:

```python
from typing import Any, Literal
```

- [ ] **Step 3.4: 跑测试确认 PASS**

```bash
pytest src/agents/tests/test_agent_config_flags.py -v
```

- [ ] **Step 3.5: commit**

```bash
git add src/agents/core/agent_config.py src/agents/tests/test_agent_config_flags.py
git commit -m "feat(agents/config): add enable_chassis_tools, vg_backend, chassis_tools_version"
```

---

### Task 4 — `derive_eval_session_id` 哈希纳入 `chassis_tools_version` 与 `vg_backend`

**Files:**
- Modify: `src/agents/examples/openeqa_official_question_pilot.py:280-292`
- Test:   `src/evaluation/scripts/tests/test_run_openeqa_stage2_full.py`(扩展)+ 新建 `src/agents/tests/test_derive_eval_session_id.py`

- [ ] **Step 4.1: 写失败测试**

`src/agents/tests/test_derive_eval_session_id.py`:

```python
"""derive_eval_session_id includes chassis_tools_version + vg_backend."""
from __future__ import annotations

from pathlib import Path

from agents.examples.openeqa_official_question_pilot import derive_eval_session_id


def test_session_id_changes_when_chassis_tools_version_changes(tmp_path: Path) -> None:
    a = derive_eval_session_id(
        output_root=tmp_path,
        enable_temporal_fan=False,
        chassis_tools_version=1,
        vg_backend="legacy",
    )
    b = derive_eval_session_id(
        output_root=tmp_path,
        enable_temporal_fan=False,
        chassis_tools_version=2,
        vg_backend="legacy",
    )
    assert a != b


def test_session_id_changes_when_vg_backend_changes(tmp_path: Path) -> None:
    a = derive_eval_session_id(
        output_root=tmp_path,
        enable_temporal_fan=False,
        chassis_tools_version=1,
        vg_backend="legacy",
    )
    b = derive_eval_session_id(
        output_root=tmp_path,
        enable_temporal_fan=False,
        chassis_tools_version=1,
        vg_backend="pack_v1",
    )
    assert a != b


def test_explicit_session_id_overrides(tmp_path: Path) -> None:
    sid = derive_eval_session_id(
        output_root=tmp_path,
        enable_temporal_fan=False,
        chassis_tools_version=1,
        vg_backend="legacy",
        explicit_session_id="custom",
    )
    assert sid == "custom"
```

- [ ] **Step 4.2: 跑测试确认失败**

```bash
pytest src/agents/tests/test_derive_eval_session_id.py -v
```

期望: `TypeError: derive_eval_session_id() got an unexpected keyword argument 'chassis_tools_version'`.

- [ ] **Step 4.3: 修改函数签名**

`src/agents/examples/openeqa_official_question_pilot.py:280-292` 替换为:

```python
def derive_eval_session_id(
    *,
    output_root: Path,
    enable_temporal_fan: bool,
    chassis_tools_version: int = 1,
    vg_backend: str = "legacy",
    explicit_session_id: str | None = None,
) -> str:
    if explicit_session_id:
        return explicit_session_id

    digest = hashlib.sha256(
        (
            f"{output_root.resolve()}|"
            f"temporal_fan={enable_temporal_fan}|"
            f"chassis_tools_version={chassis_tools_version}|"
            f"vg_backend={vg_backend}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"v15_{digest}"
```

并搜索同文件内对 `derive_eval_session_id(...)` 的现有 caller,补上
默认的 `chassis_tools_version=1, vg_backend="legacy"`(默认值不影响
当前 OpenEQA hash,因为 `vg_backend` 与 `version` 在新签名下默认值都
是稳定的"添加新字段"扩展)。

注意: 即使是默认值,改动会**改变现有 session_id 哈希**。这是**预期**
变化(spec 第 8 节明确要求),让 prompt cache 在 chassis 改造前后正确
分桶。如果有持续运行的 OpenEQA 任务,**与 OpenEQA 维护者沟通后再合**。

- [ ] **Step 4.4: 跑测试确认 PASS**

```bash
pytest src/agents/tests/test_derive_eval_session_id.py -v
pytest src/evaluation/scripts/tests/test_run_openeqa_stage2_full.py -v
```

- [ ] **Step 4.5: commit**

```bash
git add src/agents/examples/openeqa_official_question_pilot.py src/agents/tests/test_derive_eval_session_id.py
git commit -m "feat(agents/openeqa): fold chassis_tools_version + vg_backend into derive_eval_session_id"
```

---

### Task 5 — Snapshot test 锁住 QA 与 legacy VG 当前工具列表

**Files:**
- Modify: `src/agents/tests/test_stage2_deep_agent.py`

- [ ] **Step 5.1: 写新测试 + 跑确认通过**(snapshot test 是"锁现状",
  需要先通过来固化当前行为)

在 `src/agents/tests/test_stage2_deep_agent.py` 末尾追加:

```python
def test_qa_tool_list_snapshot() -> None:
    """Lock QA tool name list to catch silent chassis additions on QA."""
    from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime
    from agents.core.task_types import (
        Stage2EvidenceBundle, Stage2TaskSpec, Stage2TaskType,
    )

    runtime = DeepAgentsStage2Runtime()
    bundle = Stage2EvidenceBundle()
    task = Stage2TaskSpec(task_type=Stage2TaskType.QA, user_query="?")

    state = runtime._make_runtime_state(task, bundle) if hasattr(runtime, "_make_runtime_state") else None
    if state is None:
        # Fallback for current API: build_runtime_tools takes a runtime obj
        from agents.runtime.base import Stage2RuntimeState
        state = Stage2RuntimeState(bundle=bundle)
        state.task_type = Stage2TaskType.QA

    tool_names = sorted(t.name for t in runtime.build_runtime_tools(state))
    assert tool_names == sorted([
        "inspect_stage1_metadata",
        "retrieve_object_context",
        "request_more_views",
        "request_crops",
        "switch_or_expand_hypothesis",
    ]), f"QA tool list drifted: {tool_names}"


def test_legacy_vg_tool_list_snapshot() -> None:
    """Lock VG tool name list under vg_backend='legacy'."""
    from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime
    from agents.core.agent_config import Stage2DeepAgentConfig
    from agents.core.task_types import (
        Stage2EvidenceBundle, Stage2TaskType,
    )
    from agents.runtime.base import Stage2RuntimeState

    runtime = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig(vg_backend="legacy"))
    bundle = Stage2EvidenceBundle()
    state = Stage2RuntimeState(bundle=bundle)
    state.task_type = Stage2TaskType.VISUAL_GROUNDING
    state.vg_scene_objects = []   # non-None to enable VG branch

    tool_names = sorted(t.name for t in runtime.build_runtime_tools(state))
    assert tool_names == sorted([
        "inspect_stage1_metadata",
        "retrieve_object_context",
        "request_more_views",
        "request_crops",
        "switch_or_expand_hypothesis",
        "select_object",
        "spatial_compare",
    ]), f"Legacy VG tool list drifted: {tool_names}"
```

- [ ] **Step 5.2: 跑测试确认 PASS**(锁定现状)

```bash
pytest src/agents/tests/test_stage2_deep_agent.py::test_qa_tool_list_snapshot -v
pytest src/agents/tests/test_stage2_deep_agent.py::test_legacy_vg_tool_list_snapshot -v
```

- [ ] **Step 5.3: 如果第 5.2 步失败**(签名不一致),按报错消息精确调整
  fixture 中 `Stage2RuntimeState` 的字段;不要为通过测试而修改
  `build_runtime_tools` 的代码 — 这两条测试的目的就是**锁现状**。

- [ ] **Step 5.4: commit**

```bash
git add src/agents/tests/test_stage2_deep_agent.py
git commit -m "test(agents): snapshot QA + legacy VG tool name lists to guard chassis migration"
```

---

### Task 6 — 修掉 `select_object.compute_bbox_3d` 的 silent fallback

**Files:**
- Modify: `src/agents/tools/select_object.py:50-56`
- Modify: `src/agents/tools/tests/test_select_object.py`

- [ ] **Step 6.1: 写失败测试**

在 `src/agents/tools/tests/test_select_object.py` 末尾追加:

```python
def test_compute_bbox_3d_raises_when_no_pcd() -> None:
    """No-fallback rule: object without point cloud must raise, not default."""
    from agents.tools.select_object import compute_bbox_3d

    obj = FakeSceneObject(obj_id=42, category="lamp", pcd_np=None)
    with pytest.raises(ValueError, match="object 42 has no pcd"):
        compute_bbox_3d(obj)


def test_compute_bbox_3d_raises_on_empty_pcd() -> None:
    from agents.tools.select_object import compute_bbox_3d

    obj = FakeSceneObject(obj_id=7, category="cup", pcd_np=np.zeros((0, 3)))
    with pytest.raises(ValueError, match="object 7 has no pcd"):
        compute_bbox_3d(obj)


def test_handle_select_object_surfaces_no_pcd_error() -> None:
    """Error must propagate to the agent as a tool ERROR string."""
    from agents.tools.select_object import handle_select_object

    obj = FakeSceneObject(obj_id=3, category="picture", pcd_np=None)
    runtime = FakeRuntimeState(vg_scene_objects=[obj])
    response = handle_select_object(runtime, object_id=3, rationale="test")
    assert response.startswith("ERROR:")
    assert "no pcd" in response
```

- [ ] **Step 6.2: 跑测试确认失败**

```bash
pytest src/agents/tools/tests/test_select_object.py::test_compute_bbox_3d_raises_when_no_pcd -v
```

期望: 当前 `[0.3,0.3,0.3]` fallback 路径让函数返回 list,无异常。

- [ ] **Step 6.3: 改实现**

`src/agents/tools/select_object.py` 第 26-62 行替换为:

```python
def compute_bbox_3d(
    obj: Any,
    axis_align_matrix: np.ndarray | None = None,
) -> list[float]:
    """Compute precise 9-DOF bbox from an object's point cloud.

    Transforms all points to the aligned frame, then computes
    centroid and axis-aligned extent.

    FAIL-LOUD: raises ValueError if the object has no point cloud.
    Per the project's strict no-fallback rule, we do not substitute a
    default extent when geometry is missing.

    Returns:
        [cx, cy, cz, dx, dy, dz, 0, 0, 0]

    Raises:
        ValueError: if obj.pcd_np is None or empty.
    """
    pcd = getattr(obj, "pcd_np", None)
    if pcd is None or len(pcd) == 0:
        obj_id = getattr(obj, "obj_id", "?")
        raise ValueError(
            f"object {obj_id} has no pcd; cannot compute 9-DOF bbox"
        )

    pts = np.array(pcd, dtype=np.float64)
    if axis_align_matrix is not None:
        ones = np.ones((len(pts), 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])
        pts = (axis_align_matrix @ pts_h.T).T[:, :3]
    centroid = pts.mean(axis=0)
    extent = pts.max(axis=0) - pts.min(axis=0)

    return [
        float(centroid[0]), float(centroid[1]), float(centroid[2]),
        float(extent[0]), float(extent[1]), float(extent[2]),
        0.0, 0.0, 0.0,
    ]
```

并在 `handle_select_object` 中包裹 `compute_bbox_3d` 调用:

`src/agents/tools/select_object.py:91` 附近:

```python
    try:
        bbox_3d = compute_bbox_3d(obj, runtime_state.vg_axis_align_matrix)
    except ValueError as exc:
        return f"ERROR: {exc}"
```

- [ ] **Step 6.4: 跑测试确认 PASS**

```bash
pytest src/agents/tools/tests/test_select_object.py -v
```

(既有测试也必须全绿:既有测试用了 `FakeSceneObject` 都带 pcd,不会
触发新 raise 路径。)

- [ ] **Step 6.5: commit**

```bash
git add src/agents/tools/select_object.py src/agents/tools/tests/test_select_object.py
git commit -m "fix(agents/tools/select_object): raise on missing pcd instead of [0.3,0.3,0.3] fallback"
```

---

### Task 7 — chassis tool: `list_skills` / `load_skill` / `submit_final`

**Files:**
- Create: `src/agents/skills/chassis_tools.py`
- Test:   `src/agents/tests/test_chassis_tools.py`

- [ ] **Step 7.1: 写失败测试**

`src/agents/tests/test_chassis_tools.py`:

```python
"""Chassis tools: list_skills, load_skill, submit_final."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.skills import (
    FinalizerSpec,
    PACKS,
    SkillSpec,
    TaskPack,
    register_pack,
)
from agents.skills.chassis_tools import build_chassis_tools


@pytest.fixture(autouse=True)
def _reset_registry():
    PACKS.clear()
    yield
    PACKS.clear()


def _runtime(task_type: Stage2TaskType) -> Stage2RuntimeState:
    rs = Stage2RuntimeState(bundle=Stage2EvidenceBundle())
    rs.task_type = task_type
    return rs


def _register_vg_pack(tmp_path: Path) -> None:
    body = tmp_path / "vg_grounding_playbook.md"
    body.write_text("# VG Grounding Playbook\n...details...", encoding="utf-8")
    register_pack(
        TaskPack(
            task_type=Stage2TaskType.VISUAL_GROUNDING,
            tool_builder=lambda r: [],
            skills=[
                SkillSpec(
                    name="vg-grounding-playbook",
                    description="VG main loop.",
                    body_path=body,
                    task_types={Stage2TaskType.VISUAL_GROUNDING},
                ),
            ],
            finalizer=FinalizerSpec(
                payload_model=dict,
                validator=lambda payload, runtime: payload,
                adapter=lambda payload, runtime: {"answer": payload},
            ),
            required_primary_skill="vg-grounding-playbook",
            required_extra_metadata=[],
            ctx_factory=lambda b: object(),
        )
    )


def test_list_skills_returns_catalog(tmp_path: Path) -> None:
    _register_vg_pack(tmp_path)
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    list_skills, _, _ = build_chassis_tools(rs)
    payload = json.loads(list_skills.invoke({}))
    assert payload == [{"name": "vg-grounding-playbook", "description": "VG main loop."}]


def test_load_skill_returns_body_and_records(tmp_path: Path) -> None:
    _register_vg_pack(tmp_path)
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    _, load_skill, _ = build_chassis_tools(rs)
    body = load_skill.invoke({"skill_name": "vg-grounding-playbook"})
    assert "VG Grounding Playbook" in body
    assert "vg-grounding-playbook" in rs.skills_loaded
    assert any(t.tool_name == "load_skill" for t in rs.tool_trace)


def test_load_skill_unknown_returns_error_first(tmp_path: Path) -> None:
    _register_vg_pack(tmp_path)
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    _, load_skill, _ = build_chassis_tools(rs)
    response = load_skill.invoke({"skill_name": "no-such-skill"})
    assert response.startswith("ERROR:")


def test_load_skill_unknown_twice_raises(tmp_path: Path) -> None:
    _register_vg_pack(tmp_path)
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    _, load_skill, _ = build_chassis_tools(rs)
    load_skill.invoke({"skill_name": "no-such-skill"})
    with pytest.raises(RuntimeError, match="repeated unknown skill"):
        load_skill.invoke({"skill_name": "no-such-skill"})


def test_submit_final_calls_validator_and_adapter(tmp_path: Path) -> None:
    _register_vg_pack(tmp_path)
    rs = _runtime(Stage2TaskType.VISUAL_GROUNDING)
    _, _, submit_final = build_chassis_tools(rs)
    response = submit_final.invoke(
        {"payload": {"value": 42}, "rationale": "ok", "evidence_refs": []}
    )
    # Validator returns payload; adapter wraps as {"answer": payload}
    # The chassis stores resolved payload + signals termination
    assert "submitted" in response.lower()
    assert rs.skills_loaded.intersection({"vg-grounding-playbook"}) == set()  # no auto-load
```

- [ ] **Step 7.2: 跑测试确认失败 (ImportError)**

```bash
pytest src/agents/tests/test_chassis_tools.py -v
```

- [ ] **Step 7.3: 实现 `src/agents/skills/chassis_tools.py`**

```python
"""Chassis tools: list_skills, load_skill, submit_final.

These are always-on tools registered when the active task has a TaskPack
or when Stage2DeepAgentConfig.enable_chassis_tools=True.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool

from agents.skills.registry import PACKS, skills_for


def build_chassis_tools(runtime: Any) -> tuple[BaseTool, BaseTool, BaseTool]:
    """Construct the three chassis tools bound to one runtime state."""
    unknown_skill_loads: dict[str, int] = {}

    @tool
    def list_skills() -> str:
        """List skills available for the current task. Returns JSON array of {name, description}."""
        catalog = [
            {"name": s.name, "description": s.description}
            for s in skills_for(runtime.task_type)
        ]
        text = json.dumps(catalog, ensure_ascii=False)
        runtime.record("list_skills", {}, text)
        return text

    @tool
    def load_skill(skill_name: str) -> str:
        """Fetch the full instructions for a skill. Returns markdown body. Records load."""
        catalog = {s.name: s for s in skills_for(runtime.task_type)}
        if skill_name not in catalog:
            unknown_skill_loads[skill_name] = unknown_skill_loads.get(skill_name, 0) + 1
            available = sorted(catalog.keys())
            err = (
                f"ERROR: skill {skill_name!r} not registered for task_type "
                f"{runtime.task_type}; available: {available}"
            )
            runtime.record("load_skill", {"skill_name": skill_name}, err)
            if unknown_skill_loads[skill_name] >= 2:
                raise RuntimeError(
                    f"repeated unknown skill load: {skill_name!r}; available: {available}"
                )
            return err

        spec = catalog[skill_name]
        body = spec.body_path.read_text(encoding="utf-8")
        runtime.skills_loaded.add(skill_name)
        runtime.record("load_skill", {"skill_name": skill_name}, body)
        return body

    @tool
    def submit_final(
        payload: dict,
        rationale: str,
        evidence_refs: list[dict] | None = None,
    ) -> str:
        """Submit the final task answer. Payload must match this task's FinalizerSpec.schema.
        The chassis validates payload + preconditions; on success, terminates the run."""
        pack = PACKS.get(runtime.task_type)
        if pack is None:
            err = f"ERROR: no pack registered for {runtime.task_type}; cannot submit_final"
            runtime.record("submit_final", {"payload": payload}, err)
            return err
        try:
            validated = pack.finalizer.validator(payload, runtime)
            adapted = pack.finalizer.adapter(validated, runtime)
        except Exception as exc:
            err = f"ERROR: submit_final validation failed: {exc}"
            runtime.record("submit_final", {"payload": payload}, err)
            return err

        # Stash the resolved payload onto the runtime so build_agent's
        # downstream normalization can pick it up.
        runtime.bundle = runtime.bundle.model_copy(
            update={"extra_metadata": {**(runtime.bundle.extra_metadata or {}),
                                       "stage2_submission": adapted}}
        )
        msg = (
            f"submitted; rationale={rationale!r}; "
            f"evidence_refs={len(evidence_refs or [])}"
        )
        runtime.record(
            "submit_final",
            {"payload": payload, "rationale": rationale, "evidence_refs": evidence_refs or []},
            msg,
        )
        return msg

    return list_skills, load_skill, submit_final


__all__ = ["build_chassis_tools"]
```

- [ ] **Step 7.4: 跑测试确认 PASS**

```bash
pytest src/agents/tests/test_chassis_tools.py -v
```

- [ ] **Step 7.5: commit**

```bash
git add src/agents/skills/chassis_tools.py src/agents/tests/test_chassis_tools.py
git commit -m "feat(agents/skills): chassis tools list_skills/load_skill/submit_final with FAIL-LOUD"
```

---

### Task 8 — `validate_packs()` + 在 `build_agent` 接入

**Files:**
- Create: `src/agents/skills/validate.py`
- Modify: `src/agents/runtime/deepagents_agent.py:558-599`
- Test:   `src/agents/tests/test_validate_packs.py`

- [ ] **Step 8.1: 写失败测试**

`src/agents/tests/test_validate_packs.py`:

```python
"""validate_packs runs at build_agent and FAILS LOUD on contract violations."""
from __future__ import annotations

from pathlib import Path

import pytest

from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.skills import (
    FinalizerSpec,
    PACKS,
    SkillSpec,
    TaskPack,
    register_pack,
)
from agents.skills.validate import validate_packs


@pytest.fixture(autouse=True)
def _reset_registry():
    PACKS.clear()
    yield
    PACKS.clear()


def _make_pack(tmp_path: Path, missing_body: bool = False) -> TaskPack:
    body = tmp_path / "playbook.md"
    if not missing_body:
        body.write_text("# stub", encoding="utf-8")
    return TaskPack(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        tool_builder=lambda r: [],
        skills=[
            SkillSpec(
                name="vg-grounding-playbook",
                description="VG.",
                body_path=body,
                task_types={Stage2TaskType.VISUAL_GROUNDING},
            )
        ],
        finalizer=FinalizerSpec(
            payload_model=dict,
            validator=lambda p, r: p,
            adapter=lambda p, r: {},
        ),
        required_primary_skill="vg-grounding-playbook",
        required_extra_metadata=["vg_proposal_pool"],
        ctx_factory=lambda b: object(),
    )


def test_validate_packs_passes_on_well_formed_pack(tmp_path: Path) -> None:
    register_pack(_make_pack(tmp_path))
    bundle = Stage2EvidenceBundle(
        extra_metadata={"vg_proposal_pool": {"proposals": [{}]}}
    )
    validate_packs(Stage2TaskType.VISUAL_GROUNDING, bundle)


def test_validate_packs_raises_when_no_pack(tmp_path: Path) -> None:
    bundle = Stage2EvidenceBundle()
    with pytest.raises(RuntimeError, match="no pack"):
        validate_packs(Stage2TaskType.NAV_PLAN, bundle, require_pack=True)


def test_validate_packs_raises_when_skill_body_missing(tmp_path: Path) -> None:
    register_pack(_make_pack(tmp_path, missing_body=True))
    bundle = Stage2EvidenceBundle(
        extra_metadata={"vg_proposal_pool": {"proposals": [{}]}}
    )
    with pytest.raises(RuntimeError, match="skill body not readable"):
        validate_packs(Stage2TaskType.VISUAL_GROUNDING, bundle)


def test_validate_packs_raises_on_missing_extra_metadata(tmp_path: Path) -> None:
    register_pack(_make_pack(tmp_path))
    bundle = Stage2EvidenceBundle(extra_metadata={})
    with pytest.raises(RuntimeError, match="missing required extra_metadata"):
        validate_packs(Stage2TaskType.VISUAL_GROUNDING, bundle)


def test_validate_packs_no_op_when_pack_absent_and_not_required(tmp_path: Path) -> None:
    bundle = Stage2EvidenceBundle()
    # QA without registered pack: ok (legacy QA path)
    validate_packs(Stage2TaskType.QA, bundle, require_pack=False)
```

- [ ] **Step 8.2: 跑测试确认失败 (ImportError)**

```bash
pytest src/agents/tests/test_validate_packs.py -v
```

- [ ] **Step 8.3: 实现 `src/agents/skills/validate.py`**

```python
"""Pack-level static validation, run before the first LLM call."""
from __future__ import annotations

from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.skills.registry import PACKS


def validate_packs(
    task_type: Stage2TaskType,
    bundle: Stage2EvidenceBundle,
    *,
    require_pack: bool = False,
) -> None:
    """Validate that the active task's pack is well-formed and the bundle
    carries everything the pack needs. Raises RuntimeError on the first
    contract violation. No-ops when no pack is registered AND
    require_pack is False (the legacy QA path)."""
    pack = PACKS.get(task_type)
    if pack is None:
        if require_pack:
            raise RuntimeError(f"no pack registered for task_type={task_type}")
        return

    # 1. all skill body files exist + readable
    for skill in pack.skills:
        if not skill.body_path.exists() or not skill.body_path.is_file():
            raise RuntimeError(
                f"skill body not readable: {skill.body_path} (skill={skill.name})"
            )

    # 2. unique tool + skill names within pack
    skill_names = [s.name for s in pack.skills]
    if len(set(skill_names)) != len(skill_names):
        raise RuntimeError(f"duplicate skill name in pack {task_type}: {skill_names}")

    # 3. required_primary_skill is in pack
    if pack.required_primary_skill not in skill_names:
        raise RuntimeError(
            f"required_primary_skill {pack.required_primary_skill!r} not in "
            f"pack {task_type} skills {skill_names}"
        )

    # 4. bundle.extra_metadata carries every required key
    extra = bundle.extra_metadata or {}
    missing = [k for k in pack.required_extra_metadata if k not in extra]
    if missing:
        raise RuntimeError(
            f"bundle missing required extra_metadata for {task_type}: {missing}"
        )

    # 5. ctx_factory returns non-None
    ctx = pack.ctx_factory(bundle)
    if ctx is None:
        raise RuntimeError(
            f"ctx_factory for {task_type} returned None"
        )


__all__ = ["validate_packs"]
```

并在 `src/agents/skills/__init__.py` 添加导出:

```python
from agents.skills.validate import validate_packs
```

(`__all__` 也要追加 `"validate_packs"`)

- [ ] **Step 8.4: 在 `build_agent` 接入 `validate_packs`**

`src/agents/runtime/deepagents_agent.py:558-599` 在 `runtime = Stage2RuntimeState(...)` 之后、`tools = self.build_runtime_tools(runtime)` 之前插入:

```python
        from agents.skills.validate import validate_packs
        validate_packs(task.task_type, bundle)
```

- [ ] **Step 8.5: 跑测试确认 PASS**

```bash
pytest src/agents/tests/test_validate_packs.py -v
pytest src/agents/tests/test_stage2_deep_agent.py -v   # 既有不能挂
```

- [ ] **Step 8.6: commit**

```bash
git add src/agents/skills/validate.py src/agents/skills/__init__.py src/agents/runtime/deepagents_agent.py src/agents/tests/test_validate_packs.py
git commit -m "feat(agents/skills): validate_packs at build_agent (FAIL-LOUD on contract violation)"
```

---

## Section 2 — VG pack v1 (step 5 of spec)

### Task 9 — `VgEmbodiedScanCtx` 数据类与 `proposal_pool` 适配器

**Files:**
- Create: `src/agents/packs/vg_embodiedscan/__init__.py` (空)
- Create: `src/agents/packs/vg_embodiedscan/ctx.py`
- Create: `src/agents/packs/vg_embodiedscan/proposal_pool.py`
- Test:   `src/agents/packs/vg_embodiedscan/tests/__init__.py` (空)
- Test:   `src/agents/packs/vg_embodiedscan/tests/test_ctx.py`

- [ ] **Step 9.1: 写失败测试**

`src/agents/packs/vg_embodiedscan/tests/test_ctx.py`:

```python
"""VgEmbodiedScanCtx + proposal_pool adapter."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agents.core.task_types import Stage2EvidenceBundle
from agents.packs.vg_embodiedscan.ctx import (
    Proposal,
    VgEmbodiedScanCtx,
    build_ctx_from_bundle,
)


def test_proposal_round_trip() -> None:
    p = Proposal(
        id=3,
        bbox_3d_9dof=[0, 0, 0, 1, 1, 1, 0, 0, 0],
        category="chair",
        score=0.9,
    )
    assert p.id == 3 and p.category == "chair"


def test_build_ctx_from_bundle_minimal(tmp_path: Path) -> None:
    annotated = tmp_path / "ann"
    annotated.mkdir()
    bundle = Stage2EvidenceBundle(
        extra_metadata={
            "vg_proposal_pool": {
                "source": "vdetr",
                "proposals": [
                    {"id": 1, "bbox_3d_9dof": [0]*9, "category": "chair", "score": 0.5},
                    {"id": 2, "bbox_3d_9dof": [1]*9, "category": "desk", "score": 0.8},
                ],
                "frame_index": {10: [1, 2], 11: [2]},
                "proposal_index": {1: [10], 2: [10, 11]},
                "annotated_image_dir": str(annotated),
                "axis_align_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
            }
        }
    )
    ctx = build_ctx_from_bundle(bundle)
    assert isinstance(ctx, VgEmbodiedScanCtx)
    assert ctx.proposal_pool_source == "vdetr"
    assert {p.id for p in ctx.proposals} == {1, 2}
    assert ctx.frame_index[10] == [1, 2]
    assert ctx.proposal_index[2] == [10, 11]
    assert ctx.annotated_image_dir == annotated
    assert ctx.axis_align_matrix.shape == (4, 4)


def test_build_ctx_rejects_unknown_source(tmp_path: Path) -> None:
    bundle = Stage2EvidenceBundle(
        extra_metadata={
            "vg_proposal_pool": {
                "source": "foo",
                "proposals": [],
                "frame_index": {},
                "proposal_index": {},
                "annotated_image_dir": str(tmp_path),
            }
        }
    )
    with pytest.raises(ValueError, match="proposal_pool_source"):
        build_ctx_from_bundle(bundle)


def test_build_ctx_rejects_unreadable_annotated_dir(tmp_path: Path) -> None:
    bundle = Stage2EvidenceBundle(
        extra_metadata={
            "vg_proposal_pool": {
                "source": "vdetr",
                "proposals": [],
                "frame_index": {},
                "proposal_index": {},
                "annotated_image_dir": str(tmp_path / "nope"),
            }
        }
    )
    with pytest.raises(ValueError, match="annotated_image_dir"):
        build_ctx_from_bundle(bundle)
```

- [ ] **Step 9.2: 跑测试确认失败 (ImportError)**

```bash
pytest src/agents/packs/vg_embodiedscan/tests/test_ctx.py -v
```

- [ ] **Step 9.3: 实现 `src/agents/packs/vg_embodiedscan/ctx.py`**

```python
"""Typed runtime ctx for the EmbodiedScan VG pack."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from agents.core.task_types import Stage2EvidenceBundle


@dataclass(frozen=True)
class Proposal:
    id: int
    bbox_3d_9dof: list[float]
    category: str
    score: float


@dataclass
class VgEmbodiedScanCtx:
    proposal_pool_source: Literal["vdetr", "conceptgraph"]
    proposals: list[Proposal]
    frame_index: dict[int, list[int]]      # frame_id -> [proposal_id]
    proposal_index: dict[int, list[int]]   # proposal_id -> [frame_id]
    annotated_image_dir: Path
    axis_align_matrix: np.ndarray | None = None


def build_ctx_from_bundle(bundle: Stage2EvidenceBundle) -> VgEmbodiedScanCtx:
    extra = bundle.extra_metadata or {}
    pool = extra.get("vg_proposal_pool")
    if pool is None:
        raise ValueError("bundle.extra_metadata.vg_proposal_pool is missing")

    source = pool.get("source")
    if source not in ("vdetr", "conceptgraph"):
        raise ValueError(
            f"proposal_pool_source must be 'vdetr' or 'conceptgraph', got {source!r}"
        )

    raw_proposals = pool.get("proposals") or []
    proposals = [
        Proposal(
            id=int(p["id"]),
            bbox_3d_9dof=[float(x) for x in p["bbox_3d_9dof"]],
            category=str(p.get("category", "")),
            score=float(p.get("score", 0.0)),
        )
        for p in raw_proposals
    ]

    frame_index = {
        int(k): [int(x) for x in v] for k, v in (pool.get("frame_index") or {}).items()
    }
    proposal_index = {
        int(k): [int(x) for x in v] for k, v in (pool.get("proposal_index") or {}).items()
    }

    annotated_dir = Path(pool.get("annotated_image_dir") or "")
    if not annotated_dir.exists() or not annotated_dir.is_dir():
        raise ValueError(
            f"annotated_image_dir must exist and be a directory: {annotated_dir}"
        )

    axis_align_matrix: np.ndarray | None = None
    matrix = pool.get("axis_align_matrix")
    if matrix is not None:
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.shape != (4, 4):
            raise ValueError(
                f"axis_align_matrix must be 4x4, got shape {arr.shape}"
            )
        axis_align_matrix = arr

    return VgEmbodiedScanCtx(
        proposal_pool_source=source,
        proposals=proposals,
        frame_index=frame_index,
        proposal_index=proposal_index,
        annotated_image_dir=annotated_dir,
        axis_align_matrix=axis_align_matrix,
    )


__all__ = ["Proposal", "VgEmbodiedScanCtx", "build_ctx_from_bundle"]
```

并创建空 `src/agents/packs/vg_embodiedscan/__init__.py`(后续 task
填充)与 `src/agents/packs/vg_embodiedscan/tests/__init__.py` 空文件。

- [ ] **Step 9.4: 跑测试确认 PASS**

```bash
pytest src/agents/packs/vg_embodiedscan/tests/test_ctx.py -v
```

- [ ] **Step 9.5: commit**

```bash
git add src/agents/packs/vg_embodiedscan/
git commit -m "feat(agents/packs/vg_embodiedscan): VgEmbodiedScanCtx + build_ctx_from_bundle"
```

---

### Task 10 — `proposal_pool.py`:从 feasibility 模块产物构造 vg_proposal_pool

**Files:**
- Create: `src/agents/packs/vg_embodiedscan/proposal_pool.py`
- Test:   `src/agents/packs/vg_embodiedscan/tests/test_proposal_pool.py`

**目的:** 把 `src/benchmarks/embodiedscan_bbox_feasibility/` 下落地的
scene-level proposal (V-DETR 或 2D-CG) + per-frame visibility 投影成
`vg_proposal_pool` dict,便于 pilot/eval 把它塞进
`bundle.extra_metadata`。

- [ ] **Step 10.1: 写失败测试**

`src/agents/packs/vg_embodiedscan/tests/test_proposal_pool.py`:

```python
"""Adapter: feasibility-module artifacts -> vg_proposal_pool dict."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agents.packs.vg_embodiedscan.proposal_pool import (
    build_vg_proposal_pool,
)


def _write_proposals_json(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"proposals": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_build_vg_proposal_pool_with_visibility(tmp_path: Path) -> None:
    proposals_path = tmp_path / "props.json"
    _write_proposals_json(
        proposals_path,
        [
            {"bbox_3d": [0,0,0,1,1,1,0,0,0], "score": 0.9, "label": "chair", "metadata": {"class_id": 2}},
            {"bbox_3d": [3,0,0,1,1,1,0,0,0], "score": 0.7, "label": "desk", "metadata": {"class_id": 10}},
        ],
    )
    annotated = tmp_path / "ann"
    annotated.mkdir()

    # frustum/depth visibility precomputed externally — pass as dict
    visibility = {
        # frame_id -> [proposal_idx]
        100: [0, 1],
        101: [1],
    }

    pool = build_vg_proposal_pool(
        proposals_jsonl=proposals_path,
        source="vdetr",
        annotated_image_dir=annotated,
        frame_visibility=visibility,
        axis_align_matrix=np.eye(4),
    )
    assert pool["source"] == "vdetr"
    assert pool["annotated_image_dir"] == str(annotated)
    assert {p["id"] for p in pool["proposals"]} == {0, 1}
    assert pool["frame_index"] == {100: [0, 1], 101: [1]}
    assert pool["proposal_index"] == {0: [100], 1: [100, 101]}
    assert len(pool["axis_align_matrix"]) == 4
```

- [ ] **Step 10.2: 跑测试确认失败 (ImportError)**

```bash
pytest src/agents/packs/vg_embodiedscan/tests/test_proposal_pool.py -v
```

- [ ] **Step 10.3: 实现 `src/agents/packs/vg_embodiedscan/proposal_pool.py`**

```python
"""Build the vg_proposal_pool dict that the runtime adapter expects.

Inputs:
- proposals_jsonl: a JSON file from the feasibility module containing
  a `{"proposals": [{"bbox_3d": [...], "score": ..., "label": ..., "metadata": {...}}, ...]}`
- source: 'vdetr' or 'conceptgraph'
- annotated_image_dir: directory of pre-rendered set-of-marks frames
- frame_visibility: precomputed mapping frame_id -> list of proposal indices
  visible in that frame (computed offline by Stage 1's visibility builder)
- axis_align_matrix: 4x4 numpy array or None
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np


def build_vg_proposal_pool(
    *,
    proposals_jsonl: Path,
    source: Literal["vdetr", "conceptgraph"],
    annotated_image_dir: Path,
    frame_visibility: dict[int, list[int]],
    axis_align_matrix: np.ndarray | None,
) -> dict:
    """Read proposals + visibility and produce the `vg_proposal_pool`
    dict expected at `bundle.extra_metadata.vg_proposal_pool`."""
    raw = json.loads(proposals_jsonl.read_text(encoding="utf-8"))
    proposals_in = raw.get("proposals") or []

    proposals_out = []
    for idx, p in enumerate(proposals_in):
        proposals_out.append(
            {
                "id": idx,
                "bbox_3d_9dof": [float(x) for x in p["bbox_3d"]],
                "category": str(p.get("label") or ""),
                "score": float(p.get("score", 0.0)),
            }
        )

    proposal_index: dict[int, list[int]] = defaultdict(list)
    frame_index: dict[int, list[int]] = {}
    for frame_id, prop_ids in frame_visibility.items():
        ids = [int(i) for i in prop_ids]
        frame_index[int(frame_id)] = ids
        for pid in ids:
            proposal_index[pid].append(int(frame_id))

    pool: dict = {
        "source": source,
        "proposals": proposals_out,
        "frame_index": frame_index,
        "proposal_index": dict(proposal_index),
        "annotated_image_dir": str(annotated_image_dir),
    }
    if axis_align_matrix is not None:
        pool["axis_align_matrix"] = axis_align_matrix.tolist()
    return pool


__all__ = ["build_vg_proposal_pool"]
```

- [ ] **Step 10.4: 跑测试确认 PASS**

```bash
pytest src/agents/packs/vg_embodiedscan/tests/test_proposal_pool.py -v
```

- [ ] **Step 10.5: commit**

```bash
git add src/agents/packs/vg_embodiedscan/proposal_pool.py src/agents/packs/vg_embodiedscan/tests/test_proposal_pool.py
git commit -m "feat(agents/packs/vg_embodiedscan): build_vg_proposal_pool adapter"
```

---

### Task 11 — VG `FinalizerSpec`

**Files:**
- Create: `src/agents/packs/vg_embodiedscan/finalizer.py`
- Test:   `src/agents/packs/vg_embodiedscan/tests/test_finalizer.py`

- [ ] **Step 11.1: 写失败测试**

`src/agents/packs/vg_embodiedscan/tests/test_finalizer.py`:

```python
"""VG FinalizerSpec validates payload + adapts to Stage2StructuredResponse fields."""
from __future__ import annotations

from pathlib import Path

import pytest

from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.packs.vg_embodiedscan.ctx import (
    Proposal,
    VgEmbodiedScanCtx,
)
from agents.packs.vg_embodiedscan.finalizer import (
    VG_FINALIZER,
    VgPayload,
)


def _runtime_with_ctx(tmp_path: Path) -> Stage2RuntimeState:
    rs = Stage2RuntimeState(bundle=Stage2EvidenceBundle())
    rs.task_ctx = VgEmbodiedScanCtx(
        proposal_pool_source="vdetr",
        proposals=[
            Proposal(id=0, bbox_3d_9dof=[0]*9, category="chair", score=0.9),
            Proposal(id=1, bbox_3d_9dof=[1]*9, category="desk", score=0.8),
        ],
        frame_index={10: [0, 1]},
        proposal_index={0: [10], 1: [10]},
        annotated_image_dir=tmp_path,
    )
    return rs


def test_validator_resolves_known_proposal(tmp_path: Path) -> None:
    rs = _runtime_with_ctx(tmp_path)
    payload = VgPayload(proposal_id=1, confidence=0.9)
    result = VG_FINALIZER.validator(payload, rs)
    assert result.proposal_id == 1


def test_validator_raises_unknown_proposal_id(tmp_path: Path) -> None:
    rs = _runtime_with_ctx(tmp_path)
    payload = VgPayload(proposal_id=99, confidence=0.5)
    with pytest.raises(ValueError, match="proposal_id 99 not in pool"):
        VG_FINALIZER.validator(payload, rs)


def test_validator_accepts_minus_one_as_failed_marker(tmp_path: Path) -> None:
    rs = _runtime_with_ctx(tmp_path)
    payload = VgPayload(proposal_id=-1, confidence=0.0)
    result = VG_FINALIZER.validator(payload, rs)
    assert result.proposal_id == -1


def test_adapter_emits_bbox_3d_for_known_proposal(tmp_path: Path) -> None:
    rs = _runtime_with_ctx(tmp_path)
    payload = VgPayload(proposal_id=1, confidence=0.9)
    validated = VG_FINALIZER.validator(payload, rs)
    out = VG_FINALIZER.adapter(validated, rs)
    assert out["selected_object_id"] == 1
    assert out["bbox_3d"] == [1.0]*9
    assert out["status"] == "completed"


def test_adapter_emits_failed_status_for_minus_one(tmp_path: Path) -> None:
    rs = _runtime_with_ctx(tmp_path)
    payload = VgPayload(proposal_id=-1, confidence=0.0)
    validated = VG_FINALIZER.validator(payload, rs)
    out = VG_FINALIZER.adapter(validated, rs)
    assert out["status"] == "failed"
    assert out["selected_object_id"] is None
```

- [ ] **Step 11.2: 跑测试确认失败**

```bash
pytest src/agents/packs/vg_embodiedscan/tests/test_finalizer.py -v
```

- [ ] **Step 11.3: 实现 `src/agents/packs/vg_embodiedscan/finalizer.py`**

```python
"""VG FinalizerSpec: validate proposal_id, resolve to bbox_3d."""
from __future__ import annotations

from pydantic import BaseModel, Field

from agents.skills.finalizer import FinalizerSpec
from agents.packs.vg_embodiedscan.ctx import VgEmbodiedScanCtx


class VgPayload(BaseModel):
    """submit_final payload schema for VG."""

    proposal_id: int = Field(
        description="Selected proposal id from the pool. Use -1 to mark this sample as failed (GT not in pool)."
    )
    confidence: float = Field(ge=0.0, le=1.0)


def vg_validator(payload: VgPayload, runtime) -> VgPayload:
    if payload.proposal_id == -1:
        return payload
    ctx: VgEmbodiedScanCtx = runtime.task_ctx
    ids = {p.id for p in ctx.proposals}
    if payload.proposal_id not in ids:
        raise ValueError(
            f"proposal_id {payload.proposal_id} not in pool (have {sorted(ids)})"
        )
    return payload


def vg_adapter(payload: VgPayload, runtime) -> dict:
    """Convert validated payload to fields suitable for Stage2StructuredResponse."""
    if payload.proposal_id == -1:
        return {
            "status": "failed",
            "selected_object_id": None,
            "bbox_3d": None,
            "confidence": payload.confidence,
            "rationale_marker": "GT not in proposal pool",
        }
    ctx: VgEmbodiedScanCtx = runtime.task_ctx
    proposal = next(p for p in ctx.proposals if p.id == payload.proposal_id)
    return {
        "status": "completed",
        "selected_object_id": proposal.id,
        "bbox_3d": list(proposal.bbox_3d_9dof),
        "category": proposal.category,
        "confidence": payload.confidence,
    }


VG_FINALIZER = FinalizerSpec(
    payload_model=VgPayload,
    validator=vg_validator,
    adapter=vg_adapter,
)


__all__ = ["VgPayload", "VG_FINALIZER", "vg_validator", "vg_adapter"]
```

- [ ] **Step 11.4: 跑测试确认 PASS**

```bash
pytest src/agents/packs/vg_embodiedscan/tests/test_finalizer.py -v
```

- [ ] **Step 11.5: commit**

```bash
git add src/agents/packs/vg_embodiedscan/finalizer.py src/agents/packs/vg_embodiedscan/tests/test_finalizer.py
git commit -m "feat(agents/packs/vg_embodiedscan): VG FinalizerSpec with -1 failed-sample marker"
```

---

### Task 12 — VG tool 1: `list_keyframes_with_proposals`

**Files:**
- Create: `src/agents/packs/vg_embodiedscan/tools.py` (新建,后续 task 累加)
- Test:   `src/agents/packs/vg_embodiedscan/tests/test_tools.py`

- [ ] **Step 12.1: 写失败测试**

`src/agents/packs/vg_embodiedscan/tests/test_tools.py`:

```python
"""VG pack tools: per-tool tests + FAIL-LOUD gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import KeyframeEvidence, Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.packs.vg_embodiedscan.ctx import (
    Proposal,
    VgEmbodiedScanCtx,
)
from agents.packs.vg_embodiedscan.tools import build_vg_tools


def _runtime(tmp_path: Path) -> Stage2RuntimeState:
    annotated = tmp_path / "ann"
    annotated.mkdir()
    bundle = Stage2EvidenceBundle(
        keyframes=[
            KeyframeEvidence(keyframe_idx=0, image_path="a.png", frame_id=10),
            KeyframeEvidence(keyframe_idx=1, image_path="b.png", frame_id=11),
        ]
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.task_ctx = VgEmbodiedScanCtx(
        proposal_pool_source="vdetr",
        proposals=[
            Proposal(id=0, bbox_3d_9dof=[0]*9, category="chair", score=0.9),
            Proposal(id=1, bbox_3d_9dof=[1]*9, category="desk", score=0.8),
            Proposal(id=2, bbox_3d_9dof=[2]*9, category="chair", score=0.7),
        ],
        frame_index={10: [0, 1], 11: [1, 2]},
        proposal_index={0: [10], 1: [10, 11], 2: [11]},
        annotated_image_dir=annotated,
    )
    return rs


def test_list_keyframes_with_proposals_gates_on_skill(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    tool = next(t for t in build_vg_tools(rs) if t.name == "list_keyframes_with_proposals")
    response = tool.invoke({})
    assert response.startswith("ERROR")
    assert "vg-grounding-playbook" in response


def test_list_keyframes_with_proposals_returns_structured(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "list_keyframes_with_proposals")
    payload = json.loads(tool.invoke({}))
    assert len(payload) == 2
    assert payload[0]["frame_id"] == 10
    assert payload[0]["visible_proposal_ids"] == [0, 1]
    assert payload[0]["annotated_image"].endswith("/ann/frame_10.png")
```

- [ ] **Step 12.2: 跑测试确认失败 (ImportError)**

```bash
pytest src/agents/packs/vg_embodiedscan/tests/test_tools.py::test_list_keyframes_with_proposals_gates_on_skill -v
```

- [ ] **Step 12.3: 实现 `src/agents/packs/vg_embodiedscan/tools.py`**

```python
"""EmbodiedScan VG tools. All bodies FAIL-LOUD on missing primary skill."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool

PRIMARY_SKILL = "vg-grounding-playbook"


def _gate(runtime: Any) -> str | None:
    """Return ERROR string if skill not loaded, else None."""
    if PRIMARY_SKILL not in runtime.skills_loaded:
        return f"ERROR: load_skill({PRIMARY_SKILL!r}) before calling this tool."
    return None


def build_vg_tools(runtime: Any) -> list[BaseTool]:
    ctx = runtime.task_ctx  # VgEmbodiedScanCtx

    @tool
    def list_keyframes_with_proposals() -> str:
        """VG tool. Detailed usage in skill 'vg-grounding-playbook'."""
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("list_keyframes_with_proposals", {}, gate)
            return gate
        items = []
        for kf in runtime.bundle.keyframes:
            fid = kf.frame_id
            visible = ctx.frame_index.get(fid, []) if fid is not None else []
            items.append(
                {
                    "keyframe_idx": kf.keyframe_idx,
                    "frame_id": fid,
                    "visible_proposal_ids": visible,
                    "n_proposals": len(visible),
                    "annotated_image": str(ctx.annotated_image_dir / f"frame_{fid}.png"),
                }
            )
        text = json.dumps(items, ensure_ascii=False)
        runtime.record("list_keyframes_with_proposals", {}, text)
        return text

    return [list_keyframes_with_proposals]


__all__ = ["build_vg_tools", "PRIMARY_SKILL"]
```

- [ ] **Step 12.4: 跑测试确认 PASS**

```bash
pytest src/agents/packs/vg_embodiedscan/tests/test_tools.py -v
```

- [ ] **Step 12.5: commit**

```bash
git add src/agents/packs/vg_embodiedscan/tools.py src/agents/packs/vg_embodiedscan/tests/test_tools.py
git commit -m "feat(agents/packs/vg_embodiedscan): list_keyframes_with_proposals + FAIL-LOUD gate"
```

---

### Task 13 — VG tool 2: `view_keyframe_marked`

**Files:**
- Modify: `src/agents/packs/vg_embodiedscan/tools.py`
- Modify: `src/agents/packs/vg_embodiedscan/tests/test_tools.py`

- [ ] **Step 13.1: 写失败测试**

在 `tests/test_tools.py` 末尾追加:

```python
def test_view_keyframe_marked_returns_image_content(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    # create a fake marked image
    marked = rs.task_ctx.annotated_image_dir / "frame_10.png"
    marked.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header

    tool = next(t for t in build_vg_tools(rs) if t.name == "view_keyframe_marked")
    response = tool.invoke({"frame_id": 10})
    assert "frame_10.png" in response
    assert "visible_proposals" in response
    assert "[0, 1]" in response or "0, 1" in response


def test_view_keyframe_marked_unknown_frame_errors(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "view_keyframe_marked")
    response = tool.invoke({"frame_id": 999})
    assert response.startswith("ERROR")
```

- [ ] **Step 13.2: 跑测试确认失败**

```bash
pytest src/agents/packs/vg_embodiedscan/tests/test_tools.py::test_view_keyframe_marked_returns_image_content -v
```

- [ ] **Step 13.3: 在 `tools.py` 的 `build_vg_tools` 内追加**

```python
    @tool
    def view_keyframe_marked(frame_id: int) -> str:
        """VG tool. Detailed usage in skill 'vg-grounding-playbook'."""
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("view_keyframe_marked", {"frame_id": frame_id}, gate)
            return gate
        if frame_id not in ctx.frame_index:
            err = (
                f"ERROR: frame_id={frame_id} not in proposal index; "
                f"available: {sorted(ctx.frame_index.keys())[:20]}"
            )
            runtime.record("view_keyframe_marked", {"frame_id": frame_id}, err)
            return err
        marked_path = ctx.annotated_image_dir / f"frame_{frame_id}.png"
        if not marked_path.exists():
            err = f"ERROR: annotated image not found: {marked_path}"
            runtime.record("view_keyframe_marked", {"frame_id": frame_id}, err)
            return err
        visible = ctx.frame_index[frame_id]
        # Mark the path as a fresh image to inject into the next user message
        runtime.bundle.extra_metadata = dict(runtime.bundle.extra_metadata or {})
        runtime.bundle.extra_metadata.setdefault("vg_pending_images", []).append(str(marked_path))
        runtime.mark_evidence_updated()
        body = (
            f"frame_id={frame_id} marked image at {marked_path}; "
            f"visible_proposals={visible}; "
            f"categories={[next((p.category for p in ctx.proposals if p.id == pid), '?') for pid in visible]}"
        )
        runtime.record("view_keyframe_marked", {"frame_id": frame_id}, body)
        return body
```

并把这个新工具加进 return 列表:

```python
    return [list_keyframes_with_proposals, view_keyframe_marked]
```

- [ ] **Step 13.4: 跑测试确认 PASS**

```bash
pytest src/agents/packs/vg_embodiedscan/tests/test_tools.py -v
```

- [ ] **Step 13.5: commit**

```bash
git add src/agents/packs/vg_embodiedscan/tools.py src/agents/packs/vg_embodiedscan/tests/test_tools.py
git commit -m "feat(agents/packs/vg_embodiedscan): view_keyframe_marked tool"
```

---

### Task 14 — VG tool 3: `inspect_proposal`

**Files:**
- Modify: `src/agents/packs/vg_embodiedscan/tools.py`
- Modify: `src/agents/packs/vg_embodiedscan/tests/test_tools.py`

- [ ] **Step 14.1: 写失败测试**

在 `tests/test_tools.py` 末尾追加:

```python
def test_inspect_proposal_returns_metadata_and_frames(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "inspect_proposal")
    payload = json.loads(tool.invoke({"proposal_id": 1}))
    assert payload["proposal_id"] == 1
    assert payload["category"] == "desk"
    assert payload["score"] == 0.8
    assert payload["frames_appeared"] == [10, 11]
    assert payload["bbox_3d_9dof"] == [1]*9


def test_inspect_proposal_unknown_id_errors(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "inspect_proposal")
    response = tool.invoke({"proposal_id": 99})
    assert response.startswith("ERROR")
```

- [ ] **Step 14.2: 跑测试确认失败**

```bash
pytest src/agents/packs/vg_embodiedscan/tests/test_tools.py::test_inspect_proposal_returns_metadata_and_frames -v
```

- [ ] **Step 14.3: 在 `tools.py` 的 `build_vg_tools` 内追加**

```python
    @tool
    def inspect_proposal(proposal_id: int) -> str:
        """VG tool. Detailed usage in skill 'vg-grounding-playbook'."""
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("inspect_proposal", {"proposal_id": proposal_id}, gate)
            return gate
        proposal = next((p for p in ctx.proposals if p.id == proposal_id), None)
        if proposal is None:
            err = (
                f"ERROR: proposal_id={proposal_id} not in pool; "
                f"available count={len(ctx.proposals)}"
            )
            runtime.record("inspect_proposal", {"proposal_id": proposal_id}, err)
            return err
        payload = {
            "proposal_id": proposal.id,
            "category": proposal.category,
            "score": proposal.score,
            "bbox_3d_9dof": list(proposal.bbox_3d_9dof),
            "frames_appeared": ctx.proposal_index.get(proposal_id, []),
            "source": ctx.proposal_pool_source,
        }
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("inspect_proposal", {"proposal_id": proposal_id}, text)
        return text
```

并把它加进 return 列表。

- [ ] **Step 14.4: 跑测试确认 PASS**

```bash
pytest src/agents/packs/vg_embodiedscan/tests/test_tools.py -v
```

- [ ] **Step 14.5: commit**

```bash
git add src/agents/packs/vg_embodiedscan/tools.py src/agents/packs/vg_embodiedscan/tests/test_tools.py
git commit -m "feat(agents/packs/vg_embodiedscan): inspect_proposal tool"
```

---

### Task 15 — VG tool 4: `find_proposals_by_category`

**Files:** 同上

- [ ] **Step 15.1: 写失败测试**

```python
def test_find_proposals_by_category_lists_ids(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "find_proposals_by_category")
    payload = json.loads(tool.invoke({"category": "chair"}))
    assert payload["proposal_ids"] == [0, 2]


def test_find_proposals_by_category_unknown_returns_empty(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "find_proposals_by_category")
    payload = json.loads(tool.invoke({"category": "spaceship"}))
    assert payload["proposal_ids"] == []
    assert "available_categories" in payload
```

- [ ] **Step 15.2: 跑测试确认失败**

```bash
pytest src/agents/packs/vg_embodiedscan/tests/test_tools.py::test_find_proposals_by_category_lists_ids -v
```

- [ ] **Step 15.3: 在 `tools.py` 内追加**

```python
    @tool
    def find_proposals_by_category(category: str) -> str:
        """VG tool. Detailed usage in skill 'vg-grounding-playbook'."""
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("find_proposals_by_category", {"category": category}, gate)
            return gate
        ids = [p.id for p in ctx.proposals if p.category.strip().lower() == category.strip().lower()]
        payload = {
            "category": category,
            "proposal_ids": ids,
            "available_categories": sorted({p.category for p in ctx.proposals if p.category}),
        }
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("find_proposals_by_category", {"category": category}, text)
        return text
```

加进 return 列表。

- [ ] **Step 15.4: 跑测试确认 PASS**

```bash
pytest src/agents/packs/vg_embodiedscan/tests/test_tools.py -v
```

- [ ] **Step 15.5: commit**

```bash
git add src/agents/packs/vg_embodiedscan/tools.py src/agents/packs/vg_embodiedscan/tests/test_tools.py
git commit -m "feat(agents/packs/vg_embodiedscan): find_proposals_by_category tool"
```

---

### Task 16 — VG tool 5: `compare_proposals_spatial`

**Files:** 同上

- [ ] **Step 16.1: 写失败测试**

```python
def test_compare_proposals_spatial_closest_to(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    # set proposal centers far apart so order is deterministic
    rs.task_ctx.proposals = [
        Proposal(id=0, bbox_3d_9dof=[0,0,0,1,1,1,0,0,0], category="chair", score=0.9),
        Proposal(id=1, bbox_3d_9dof=[5,5,5,1,1,1,0,0,0], category="chair", score=0.7),
        Proposal(id=2, bbox_3d_9dof=[10,10,10,1,1,1,0,0,0], category="desk", score=0.8),
    ]
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")
    payload = json.loads(tool.invoke({
        "candidate_ids": [0, 1],
        "anchor_id": 2,
        "relation": "closest_to",
    }))
    assert payload["ranked_ids"] == [1, 0]


def test_compare_proposals_spatial_farthest_from(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    rs.task_ctx.proposals = [
        Proposal(id=0, bbox_3d_9dof=[0,0,0,1,1,1,0,0,0], category="chair", score=0.9),
        Proposal(id=1, bbox_3d_9dof=[5,5,5,1,1,1,0,0,0], category="chair", score=0.7),
        Proposal(id=2, bbox_3d_9dof=[10,10,10,1,1,1,0,0,0], category="desk", score=0.8),
    ]
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")
    payload = json.loads(tool.invoke({
        "candidate_ids": [0, 1],
        "anchor_id": 2,
        "relation": "farthest_from",
    }))
    assert payload["ranked_ids"] == [0, 1]


def test_compare_proposals_spatial_unknown_relation_errors(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")
    response = tool.invoke({
        "candidate_ids": [0, 1],
        "anchor_id": 2,
        "relation": "left_of",
    })
    assert response.startswith("ERROR")
```

- [ ] **Step 16.2: 跑测试确认失败**

```bash
pytest src/agents/packs/vg_embodiedscan/tests/test_tools.py::test_compare_proposals_spatial_closest_to -v
```

- [ ] **Step 16.3: 在 `tools.py` 内追加**

```python
    @tool
    def compare_proposals_spatial(
        candidate_ids: list[int],
        anchor_id: int,
        relation: str,
    ) -> str:
        """VG tool. Detailed usage in skill 'vg-grounding-playbook'."""
        gate = _gate(runtime)
        request = {"candidate_ids": candidate_ids, "anchor_id": anchor_id, "relation": relation}
        if gate is not None:
            runtime.record("compare_proposals_spatial", request, gate)
            return gate
        if relation not in ("closest_to", "farthest_from"):
            err = f"ERROR: unsupported relation {relation!r}; allowed: closest_to | farthest_from"
            runtime.record("compare_proposals_spatial", request, err)
            return err

        import numpy as np
        anchor = next((p for p in ctx.proposals if p.id == anchor_id), None)
        if anchor is None:
            err = f"ERROR: anchor_id={anchor_id} not in pool"
            runtime.record("compare_proposals_spatial", request, err)
            return err
        candidates = [p for p in ctx.proposals if p.id in set(candidate_ids)]
        missing = sorted(set(candidate_ids) - {p.id for p in candidates})
        if missing:
            err = f"ERROR: candidate ids not in pool: {missing}"
            runtime.record("compare_proposals_spatial", request, err)
            return err

        anchor_center = np.array(anchor.bbox_3d_9dof[:3])
        scored = [
            (p.id, float(np.linalg.norm(np.array(p.bbox_3d_9dof[:3]) - anchor_center)))
            for p in candidates
        ]
        reverse = (relation == "farthest_from")
        scored.sort(key=lambda x: x[1], reverse=reverse)
        payload = {
            "anchor_id": anchor_id,
            "relation": relation,
            "ranked_ids": [pid for pid, _ in scored],
            "distances": [d for _, d in scored],
        }
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("compare_proposals_spatial", request, text)
        return text
```

加进 return 列表。

- [ ] **Step 16.4: 跑测试确认 PASS**

```bash
pytest src/agents/packs/vg_embodiedscan/tests/test_tools.py -v
```

- [ ] **Step 16.5: commit**

```bash
git add src/agents/packs/vg_embodiedscan/tools.py src/agents/packs/vg_embodiedscan/tests/test_tools.py
git commit -m "feat(agents/packs/vg_embodiedscan): compare_proposals_spatial tool"
```

---

### Task 17 — 写 3 个 skill markdown body

**Files:**
- Create: `src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md`
- Create: `src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation.md`
- Create: `src/agents/packs/vg_embodiedscan/skills/evidence_scouting.md`

每个 skill body 必须包含 spec 第 5 节 SkillPack 表里列的字段:决策树、
工具用法 (含参数语义)、反例、示例。每个 ≥ 80 行 markdown。

- [ ] **Step 17.1: 写 `vg_grounding_playbook.md`** — 至少包含:
  - 何时使用 (catalog desc 的扩写)
  - 决策树:`list_keyframes_with_proposals` → 选 1-3 个 keyframe →
    `view_keyframe_marked` → 若 ≥2 候选则 `inspect_proposal` →
    `submit_final({proposal_id, confidence}, rationale)`
  - 6 个工具的详细 input/output schema 与典型用法
  - VG payload schema 完整说明
  - OOD 处理:`submit_final({"proposal_id": -1, "confidence": 0.0}, rationale="GT not in proposal pool")`
  - 反例 5 条 (e.g. "不要在没看任何标注图前 submit"、"不要重复看同一帧")

- [ ] **Step 17.2: 写 `vg_spatial_disambiguation.md`** — 至少包含:
  - 何时使用 (query 出现 "next to" / "closest to" / "between" 等)
  - 三段拆解:target / relation / anchor
  - 工具序列:`find_proposals_by_category(anchor)` →
    `compare_proposals_spatial(candidates, anchor_id, relation)`
  - anchor 自身多义时的递归处理
  - 4 个示例 (covers closest_to / farthest_from / between / above)

- [ ] **Step 17.3: 写 `evidence_scouting.md`** — 至少包含:
  - 何时使用 (sufficiency 不够)
  - `request_more_views` 三种 mode (`targeted` / `explore` /
    `temporal_fan`) 的选择规则与典型 prompt 写法
  - `request_crops` 用法
  - sufficiency 判据 (target 可见?anchor 可见?几何关系可见?)
  - 反例 (e.g. "连续要 3 次同样视角")

- [ ] **Step 17.4: 跑一个简单的"文件可读"测试** — 已经被
  `test_validate_packs.py` 覆盖,无需新测试。手工 verify:

```bash
ls -la src/agents/packs/vg_embodiedscan/skills/
wc -l src/agents/packs/vg_embodiedscan/skills/*.md
```

每个文件 ≥ 80 行。

- [ ] **Step 17.5: commit**

```bash
git add src/agents/packs/vg_embodiedscan/skills/
git commit -m "docs(agents/packs/vg_embodiedscan): add 3 skill bodies (playbook + spatial + scouting)"
```

---

### Task 18 — Pack 注册:`registration.py` + `__init__.py` + `packs/__init__.py`

**Files:**
- Create: `src/agents/packs/vg_embodiedscan/registration.py`
- Modify: `src/agents/packs/vg_embodiedscan/__init__.py`
- Create: `src/agents/packs/__init__.py`
- Test:   `src/agents/packs/vg_embodiedscan/tests/test_registration.py`

- [ ] **Step 18.1: 写失败测试**

`src/agents/packs/vg_embodiedscan/tests/test_registration.py`:

```python
"""VG pack registration smoke test."""
from __future__ import annotations

from agents.core.agent_config import Stage2TaskType
from agents.skills import PACKS


def test_vg_pack_registers_on_import() -> None:
    PACKS.clear()
    import agents.packs  # triggers all pack imports
    assert Stage2TaskType.VISUAL_GROUNDING in PACKS
    pack = PACKS[Stage2TaskType.VISUAL_GROUNDING]
    skill_names = sorted(s.name for s in pack.skills)
    assert skill_names == [
        "evidence-scouting",
        "vg-grounding-playbook",
        "vg-spatial-disambiguation",
    ]
    assert pack.required_primary_skill == "vg-grounding-playbook"
    assert pack.required_extra_metadata == ["vg_proposal_pool"]
```

- [ ] **Step 18.2: 跑测试确认失败**

```bash
pytest src/agents/packs/vg_embodiedscan/tests/test_registration.py -v
```

- [ ] **Step 18.3: 实现 `registration.py`**

`src/agents/packs/vg_embodiedscan/registration.py`:

```python
"""VG pack registration: assemble TaskPack and register it."""
from __future__ import annotations

from pathlib import Path

from agents.core.agent_config import Stage2TaskType
from agents.skills import SkillSpec, TaskPack, register_pack
from agents.packs.vg_embodiedscan.ctx import build_ctx_from_bundle
from agents.packs.vg_embodiedscan.finalizer import VG_FINALIZER
from agents.packs.vg_embodiedscan.tools import build_vg_tools

_PACK_DIR = Path(__file__).resolve().parent
_SKILLS_DIR = _PACK_DIR / "skills"

VG_PACK = TaskPack(
    task_type=Stage2TaskType.VISUAL_GROUNDING,
    tool_builder=build_vg_tools,
    skills=[
        SkillSpec(
            name="vg-grounding-playbook",
            description="EmbodiedScan VG main loop: read marked keyframes, pick proposal, submit.",
            body_path=_SKILLS_DIR / "vg_grounding_playbook.md",
            task_types={Stage2TaskType.VISUAL_GROUNDING},
        ),
        SkillSpec(
            name="vg-spatial-disambiguation",
            description="Use when the query contains spatial relations like 'next to' or 'closest to'.",
            body_path=_SKILLS_DIR / "vg_spatial_disambiguation.md",
            task_types={Stage2TaskType.VISUAL_GROUNDING},
        ),
        SkillSpec(
            name="evidence-scouting",
            description="Decide when to request more keyframes or crops, and how to phrase the request.",
            body_path=_SKILLS_DIR / "evidence_scouting.md",
            task_types={Stage2TaskType.VISUAL_GROUNDING, Stage2TaskType.QA},
        ),
    ],
    finalizer=VG_FINALIZER,
    required_primary_skill="vg-grounding-playbook",
    required_extra_metadata=["vg_proposal_pool"],
    ctx_factory=build_ctx_from_bundle,
)


def register() -> None:
    register_pack(VG_PACK)


__all__ = ["VG_PACK", "register"]
```

- [ ] **Step 18.4: 实现 `vg_embodiedscan/__init__.py`**

```python
"""EmbodiedScan VG pack — auto-registers on import."""
from agents.packs.vg_embodiedscan.registration import VG_PACK, register

register()

__all__ = ["VG_PACK"]
```

- [ ] **Step 18.5: 实现 `src/agents/packs/__init__.py`**

```python
"""Stage-2 task packs. Importing this module triggers deterministic
pack imports so they self-register into agents.skills.PACKS."""
from agents.packs import vg_embodiedscan as _vg_embodiedscan  # noqa: F401

__all__ = []
```

- [ ] **Step 18.6: 跑测试确认 PASS**

```bash
pytest src/agents/packs/vg_embodiedscan/tests/test_registration.py -v
pytest src/agents/tests/ -v       # 既有不能挂
pytest src/agents/skills/ src/agents/packs/ -v   # 全 pack 测试
```

- [ ] **Step 18.7: commit**

```bash
git add src/agents/packs/
git commit -m "feat(agents/packs): register VG_PACK on import via packs/__init__.py"
```

---

### Task 19 — `build_runtime_tools` 接入 VG pack(`vg_backend='pack_v1'`)

**Files:**
- Modify: `src/agents/runtime/deepagents_agent.py:266-310` 与 `:570-588`
- Modify: `src/agents/runtime/base.py:312-426` (注入 skill catalog)
- Test:   `src/agents/tests/test_stage2_deep_agent.py` (新 snapshot)

- [ ] **Step 19.1: 写失败测试** (新增 pack_v1 snapshot test)

在 `test_stage2_deep_agent.py` 末尾追加:

```python
def test_pack_v1_vg_tool_list_snapshot(tmp_path) -> None:
    """Lock VG tool name list under vg_backend='pack_v1'."""
    import agents.packs  # triggers VG_PACK registration  # noqa: F401
    from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime
    from agents.core.agent_config import Stage2DeepAgentConfig
    from agents.core.task_types import (
        KeyframeEvidence, Stage2EvidenceBundle, Stage2TaskType,
    )
    from agents.runtime.base import Stage2RuntimeState
    from agents.packs.vg_embodiedscan.ctx import (
        Proposal, VgEmbodiedScanCtx,
    )

    annotated = tmp_path / "ann"
    annotated.mkdir()
    runtime = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig(vg_backend="pack_v1"))
    bundle = Stage2EvidenceBundle(
        keyframes=[KeyframeEvidence(keyframe_idx=0, image_path="a.png", frame_id=10)],
        extra_metadata={"vg_proposal_pool": {
            "source": "vdetr",
            "proposals": [],
            "frame_index": {},
            "proposal_index": {},
            "annotated_image_dir": str(annotated),
        }},
    )
    state = Stage2RuntimeState(bundle=bundle)
    state.task_type = Stage2TaskType.VISUAL_GROUNDING
    state.task_ctx = VgEmbodiedScanCtx(
        proposal_pool_source="vdetr",
        proposals=[],
        frame_index={},
        proposal_index={},
        annotated_image_dir=annotated,
    )

    tool_names = sorted(t.name for t in runtime.build_runtime_tools(state))
    assert tool_names == sorted([
        # shared
        "inspect_stage1_metadata", "retrieve_object_context",
        "request_more_views", "request_crops",
        "switch_or_expand_hypothesis",
        # VG-pack new
        "list_keyframes_with_proposals", "view_keyframe_marked",
        "inspect_proposal", "find_proposals_by_category",
        "compare_proposals_spatial",
        # chassis
        "list_skills", "load_skill", "submit_final",
    ]), f"pack-v1 VG tool list drifted: {tool_names}"
```

- [ ] **Step 19.2: 跑测试确认失败**

```bash
pytest src/agents/tests/test_stage2_deep_agent.py::test_pack_v1_vg_tool_list_snapshot -v
```

- [ ] **Step 19.3: 修改 `build_runtime_tools` (`deepagents_agent.py:266-310`)**

把现有 VG 分支:

```python
        if (
            runtime.task_type == Stage2TaskType.VISUAL_GROUNDING
            and runtime.vg_scene_objects is not None
        ):
            from ..tools.select_object import handle_select_object
            ...
            tools.extend([select_object, spatial_compare])
```

替换为:

```python
        if runtime.task_type == Stage2TaskType.VISUAL_GROUNDING:
            backend = self.config.vg_backend
            if backend == "pack_v1":
                from agents.skills import PACKS
                from agents.skills.chassis_tools import build_chassis_tools
                pack = PACKS.get(Stage2TaskType.VISUAL_GROUNDING)
                if pack is None:
                    raise RuntimeError(
                        "vg_backend='pack_v1' but VG pack not registered; "
                        "import agents.packs to trigger registration"
                    )
                tools.extend(pack.tool_builder(runtime))
                tools.extend(build_chassis_tools(runtime))
            elif runtime.vg_scene_objects is not None:
                # legacy branch (vg_backend='legacy', kept until step 9)
                from ..tools.select_object import handle_select_object
                from ..tools.spatial_compare import handle_spatial_compare

                @tool
                def select_object(object_id: int, rationale: str) -> str:
                    """[unchanged docstring]"""
                    response = handle_select_object(runtime, object_id, rationale)
                    runtime.record("select_object", {"object_id": object_id, "rationale": rationale}, response)
                    return response

                @tool
                def spatial_compare(target_category: str, relation: str, anchor_category: str) -> str:
                    """[unchanged docstring]"""
                    response = handle_spatial_compare(runtime, target_category, relation, anchor_category)
                    runtime.record("spatial_compare", {"target_category": target_category, "relation": relation, "anchor_category": anchor_category}, response)
                    return response

                tools.extend([select_object, spatial_compare])
```

(把 docstring 留作"[unchanged docstring]"占位实际上要保留**与原始
完全一致**的文本 — 直接 git show 把 26x-30x 的原 docstring 粘回去。)

- [ ] **Step 19.4: 在 `_make_runtime_state` 等价路径(`deepagents_agent.py:570-588`)中,基于 `vg_backend='pack_v1'` 时调用 `build_ctx_from_bundle` 把 ctx 挂到 `runtime.task_ctx`**

```python
        # Populate VG runtime state from bundle extra_metadata
        if task.task_type == Stage2TaskType.VISUAL_GROUNDING:
            if self.config.vg_backend == "pack_v1":
                from agents.packs.vg_embodiedscan.ctx import build_ctx_from_bundle
                runtime.task_ctx = build_ctx_from_bundle(runtime.bundle)
            else:
                # legacy
                extra = runtime.bundle.extra_metadata or {}
                runtime.vg_scene_objects = extra.get("scene_objects")
                mat = extra.get("axis_align_matrix")
                if mat is not None:
                    runtime.vg_axis_align_matrix = np.array(mat, dtype=np.float64)
                cleaned = {
                    k: v for k, v in extra.items()
                    if k not in ("scene_objects", "axis_align_matrix")
                }
                runtime.bundle = runtime.bundle.model_copy(
                    update={"extra_metadata": cleaned}
                )
```

- [ ] **Step 19.5: 在 `build_system_prompt` 注入 skill catalog**

`src/agents/runtime/base.py:312-426` 的 `build_system_prompt` 中,
在 `f"{self._format_scene_inventory(object_context)}"` 之前插入:

```python
        skill_catalog = self._format_skill_catalog(task.task_type)
```

并在 prompt 拼接里加入 `f"{skill_catalog}"`。

实现 helper:

```python
    def _format_skill_catalog(self, task_type: Stage2TaskType) -> str:
        from agents.skills.registry import skills_for
        skills = skills_for(task_type)
        if not skills:
            return ""
        lines = ["Available skills (use list_skills() / load_skill(name) for details):"]
        for s in skills:
            lines.append(f"- {s.name}: {s.description}")
        return "\n".join(lines) + "\n\n"
```

- [ ] **Step 19.6: 跑测试确认 PASS**

```bash
pytest src/agents/tests/test_stage2_deep_agent.py -v
pytest src/agents/ -v   # 全部
```

- [ ] **Step 19.7: commit**

```bash
git add src/agents/runtime/deepagents_agent.py src/agents/runtime/base.py src/agents/tests/test_stage2_deep_agent.py
git commit -m "feat(agents/runtime): wire vg_backend='pack_v1' through build_runtime_tools + skill catalog"
```

---

## Section 3 — Pilot + side-by-side validation (step 6 of spec)

### Task 20 — 新 EmbodiedScan VG pack-v1 pilot 脚本

**Files:**
- Create: `src/agents/examples/embodiedscan_vg_pack_v1_pilot.py`

仿照现有 `embodiedscan_vg_pilot.py` 结构,但:
- 设 `Stage2DeepAgentConfig(vg_backend='pack_v1', enable_chassis_tools=False)`
- 在 `build_stage2_evidence_bundle(...)` 之后,把 V-DETR 或 ConceptGraph
  的 proposal 文件 + 离线 visibility index 通过
  `build_vg_proposal_pool(...)` 塞进 `bundle.extra_metadata.vg_proposal_pool`
- 不再注入 `scene_objects` / `axis_align_matrix` (legacy 字段)

(本 task 由于 pilot 脚本与外部数据耦合较多,采用**少量手工 smoke
测试 + 一份单 sample 集成测试**而非纯 TDD。)

- [ ] **Step 20.1: 写 smoke 集成测试**

`src/agents/examples/tests/test_embodiedscan_vg_pack_v1_pilot.py`:

```python
"""Smoke: pack-v1 pilot can build a bundle + run validate_packs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.integration
def test_pilot_build_bundle_passes_validate_packs(tmp_path: Path) -> None:
    """Verify the pilot's bundle builder produces a bundle that
    passes validate_packs without exception. Mocks proposal file."""
    import agents.packs  # noqa: F401

    from agents.examples.embodiedscan_vg_pack_v1_pilot import (
        build_pack_v1_bundle,
    )

    # mock proposal file
    proposals_path = tmp_path / "props.json"
    proposals_path.write_text(json.dumps({"proposals": [
        {"bbox_3d": [0]*9, "score": 0.9, "label": "chair"}
    ]}), encoding="utf-8")
    annotated = tmp_path / "ann"
    annotated.mkdir()
    (annotated / "frame_10.png").write_bytes(b"\x89PNG")

    bundle = build_pack_v1_bundle(
        proposals_jsonl=proposals_path,
        source="vdetr",
        annotated_image_dir=annotated,
        frame_visibility={10: [0]},
        keyframes=[(0, "/tmp/a.png", 10)],
        scene_id="scene0415_00",
    )
    assert "vg_proposal_pool" in bundle.extra_metadata

    from agents.skills.validate import validate_packs
    from agents.core.agent_config import Stage2TaskType
    validate_packs(Stage2TaskType.VISUAL_GROUNDING, bundle)  # no raise
```

- [ ] **Step 20.2: 实现 pilot 骨架**

`src/agents/examples/embodiedscan_vg_pack_v1_pilot.py`:

```python
#!/usr/bin/env python
"""EmbodiedScan VG pilot using pack-v1 backend.

Differences from legacy embodiedscan_vg_pilot.py:
- Stage2DeepAgentConfig(vg_backend="pack_v1")
- bundle.extra_metadata.vg_proposal_pool populated by build_vg_proposal_pool
- agent terminates via chassis submit_final, not select_object side effect
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import agents.packs  # noqa: F401  (triggers VG_PACK registration)
from agents.core.agent_config import Stage2DeepAgentConfig, Stage2TaskType
from agents.core.task_types import (
    KeyframeEvidence,
    Stage2EvidenceBundle,
    Stage2TaskSpec,
)
from agents.packs.vg_embodiedscan.proposal_pool import build_vg_proposal_pool
from agents.stage2_deep_agent import Stage2DeepResearchAgent


def build_pack_v1_bundle(
    *,
    proposals_jsonl: Path,
    source: str,
    annotated_image_dir: Path,
    frame_visibility: dict[int, list[int]],
    keyframes: Sequence[tuple[int, str, int]],
    scene_id: str,
    axis_align_matrix: np.ndarray | None = None,
) -> Stage2EvidenceBundle:
    pool = build_vg_proposal_pool(
        proposals_jsonl=proposals_jsonl,
        source=source,
        annotated_image_dir=annotated_image_dir,
        frame_visibility=frame_visibility,
        axis_align_matrix=axis_align_matrix,
    )
    return Stage2EvidenceBundle(
        scene_id=scene_id,
        keyframes=[
            KeyframeEvidence(keyframe_idx=idx, image_path=path, frame_id=fid)
            for idx, path, fid in keyframes
        ],
        extra_metadata={"vg_proposal_pool": pool},
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scene-id", required=True)
    p.add_argument("--proposals-jsonl", type=Path, required=True)
    p.add_argument("--source", choices=["vdetr", "conceptgraph"], required=True)
    p.add_argument("--annotated-image-dir", type=Path, required=True)
    p.add_argument("--visibility-json", type=Path, required=True)
    p.add_argument("--keyframes-json", type=Path, required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    visibility = {int(k): [int(x) for x in v]
                  for k, v in json.loads(args.visibility_json.read_text()).items()}
    keyframes = json.loads(args.keyframes_json.read_text())  # [[idx, path, fid], ...]

    bundle = build_pack_v1_bundle(
        proposals_jsonl=args.proposals_jsonl,
        source=args.source,
        annotated_image_dir=args.annotated_image_dir,
        frame_visibility=visibility,
        keyframes=keyframes,
        scene_id=args.scene_id,
    )

    cfg = Stage2DeepAgentConfig(vg_backend="pack_v1")
    agent = Stage2DeepResearchAgent(config=cfg)
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        user_query=args.query,
    )
    result = agent.run(task=task, bundle=bundle)
    args.output.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("pack-v1 pilot done; result -> {}", args.output)


if __name__ == "__main__":
    main()
```

(注意:`Stage2DeepResearchAgent.run` 现有接口可能需要适配
`build_pack_v1_bundle` 的输出。如果 `run()` 方法签名跟此不一致,优先
**调整 pilot** 来对齐既有 `run()`,**不要**改 `run()`。)

- [ ] **Step 20.3: 跑 smoke 测试确认 PASS**

```bash
pytest src/agents/examples/tests/test_embodiedscan_vg_pack_v1_pilot.py -v
```

- [ ] **Step 20.4: 手工 smoke**(在已有 EmbodiedScan val 数据上跑 1 个
  sample)

```bash
# (具体 visibility/keyframes 由 step 1 数据准备,但若已存在历史产物可直接用)
python src/agents/examples/embodiedscan_vg_pack_v1_pilot.py \
  --scene-id scene0415_00 \
  --proposals-jsonl path/to/proposals.json \
  --source vdetr \
  --annotated-image-dir path/to/ann \
  --visibility-json path/to/visibility.json \
  --keyframes-json path/to/keyframes.json \
  --query "the chair next to the desk" \
  --output /tmp/pack_v1_smoke.json
cat /tmp/pack_v1_smoke.json | head -20
```

期望:JSON 输出含 `selected_object_id` (或 `proposal_id == -1`),
`bbox_3d` 9 个 float,无 exception。

- [ ] **Step 20.5: commit**

```bash
git add src/agents/examples/embodiedscan_vg_pack_v1_pilot.py src/agents/examples/tests/
git commit -m "feat(agents/examples): EmbodiedScan VG pack-v1 pilot script"
```

---

### Task 21 — Visibility index 离线构造工具(若现有数据缺失)

**Files:**
- Create: `src/benchmarks/embodiedscan_bbox_feasibility/visibility_index.py`(或
  扩展现有 module)
- Test:   `src/benchmarks/tests/test_visibility_index.py`

**目的:** 给定 scene 的 proposals + 相机内外参 + depth,产出
`frame_visibility: dict[frame_id, list[proposal_idx]]`。这是
`build_vg_proposal_pool` 的关键输入。

如果 `data/embodiedscan/` 已经有现成 visibility index(在
`scene_info.json` 或独立 pickle 里),则**跳过本 task**,直接用解析
脚本读取既有数据。

- [ ] **Step 21.1: 先检查既有数据是否够**

```bash
find data/embodiedscan -name "scene_info.json" | head -3 | xargs -I {} python -c "
import json
d = json.load(open('{}'))
print('keys:', list(d.keys())[:10])
print('first image keys:', list(d.get('images',[{}])[0].keys())[:10])
"
```

如果 `scene_info.json` 的 `images[*].visible_instance_ids` 已经按
target_id 给出,且我们的 proposal pool 是 V-DETR 或 CG 输出(其
`bbox` 与 GT instance 不一致),需要 frustum-based 投影。

- [ ] **Step 21.2: 写失败测试**(若需要新写 visibility index)

`src/benchmarks/tests/test_visibility_index.py`:

```python
"""Visibility: project 3D bbox into camera frustum + depth check."""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks.embodiedscan_bbox_feasibility.visibility_index import (
    bbox_visible_in_frustum,
    build_frame_visibility,
)


def test_bbox_visible_when_in_frustum() -> None:
    # camera at origin, looking down +Z, fov ~60deg
    intrinsic = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
    extrinsic = np.eye(4)  # camera at origin, no rotation
    bbox_9dof = [0, 0, 5, 1, 1, 1, 0, 0, 0]   # 5m in front
    assert bbox_visible_in_frustum(
        bbox_9dof, intrinsic, extrinsic, image_size=(640, 480), depth_max=10.0
    )


def test_bbox_invisible_behind_camera() -> None:
    intrinsic = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
    extrinsic = np.eye(4)
    bbox_9dof = [0, 0, -5, 1, 1, 1, 0, 0, 0]   # 5m behind
    assert not bbox_visible_in_frustum(
        bbox_9dof, intrinsic, extrinsic, image_size=(640, 480), depth_max=10.0
    )


def test_build_frame_visibility_dispatches_per_frame() -> None:
    intrinsic = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
    extrinsic = np.eye(4)
    frames = {10: extrinsic, 11: extrinsic}
    proposals = [
        {"bbox_3d_9dof": [0,0,5,1,1,1,0,0,0]},
        {"bbox_3d_9dof": [0,0,-5,1,1,1,0,0,0]},
    ]
    visibility = build_frame_visibility(
        proposals=proposals,
        intrinsic=intrinsic,
        extrinsics_per_frame=frames,
        image_size=(640, 480),
        depth_max=10.0,
    )
    assert visibility[10] == [0]
    assert visibility[11] == [0]
```

- [ ] **Step 21.3: 实现** (基于 frustum 投影 + depth_max 检查)

`src/benchmarks/embodiedscan_bbox_feasibility/visibility_index.py`:

```python
"""Per-frame visibility for 3D bbox proposals."""
from __future__ import annotations

import numpy as np


def _bbox_corners(bbox_9dof: list[float]) -> np.ndarray:
    cx, cy, cz, dx, dy, dz, *_ = bbox_9dof
    half = np.array([dx, dy, dz]) / 2.0
    base = np.array([cx, cy, cz])
    signs = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
    ])
    return base + signs * half  # shape (8, 3)


def bbox_visible_in_frustum(
    bbox_9dof: list[float],
    intrinsic: np.ndarray,
    extrinsic_world_to_cam: np.ndarray,
    image_size: tuple[int, int],
    depth_max: float = 10.0,
) -> bool:
    """Return True if any corner of the bbox falls inside the image
    frustum at positive depth and within depth_max."""
    corners_world = _bbox_corners(bbox_9dof)
    # world -> camera
    corners_h = np.hstack([corners_world, np.ones((8, 1))])
    cam = (extrinsic_world_to_cam @ corners_h.T).T[:, :3]
    if (cam[:, 2] <= 0).all() or (cam[:, 2] >= depth_max).all():
        return False
    # project to pixel
    valid = cam[(cam[:, 2] > 0) & (cam[:, 2] < depth_max)]
    if len(valid) == 0:
        return False
    px = (intrinsic @ valid.T).T
    px = px[:, :2] / px[:, 2:3]
    w, h = image_size
    in_image = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    return bool(in_image.any())


def build_frame_visibility(
    *,
    proposals: list[dict],
    intrinsic: np.ndarray,
    extrinsics_per_frame: dict[int, np.ndarray],
    image_size: tuple[int, int],
    depth_max: float = 10.0,
) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for fid, extr in extrinsics_per_frame.items():
        visible = []
        for idx, p in enumerate(proposals):
            bbox = p.get("bbox_3d_9dof") or p.get("bbox_3d")
            if bbox and bbox_visible_in_frustum(bbox, intrinsic, extr, image_size, depth_max):
                visible.append(idx)
        out[int(fid)] = visible
    return out


__all__ = ["bbox_visible_in_frustum", "build_frame_visibility"]
```

- [ ] **Step 21.4: 跑测试 PASS**

```bash
pytest src/benchmarks/tests/test_visibility_index.py -v
```

- [ ] **Step 21.5: commit**

```bash
git add src/benchmarks/embodiedscan_bbox_feasibility/visibility_index.py src/benchmarks/tests/test_visibility_index.py
git commit -m "feat(benchmarks/embodiedscan): per-frame visibility index for 3D bbox proposals"
```

---

### Task 22 — Annotated set-of-marks 渲染器

**Files:**
- Create: `src/benchmarks/embodiedscan_bbox_feasibility/render_marks.py`
- Test:   `src/benchmarks/tests/test_render_marks.py`

**目的:** 把每个 keyframe 的 `(RGB image, 可见 proposals 的 2D bbox 投影,
proposal_id, label)` 渲染成一张 set-of-marks 标注 PNG,产出到
`annotated_image_dir/frame_<fid>.png`。

- [ ] **Step 22.1: 写失败测试** (PIL-based, 输出尺寸 + 最少 1 个标注)

```python
"""Set-of-marks rendering for VG keyframes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from benchmarks.embodiedscan_bbox_feasibility.render_marks import (
    render_marked_keyframe,
)


def test_render_marked_keyframe_writes_png(tmp_path: Path) -> None:
    img = Image.new("RGB", (320, 240), color="white")
    img_path = tmp_path / "rgb.png"
    img.save(img_path)
    out_path = tmp_path / "ann" / "frame_10.png"

    render_marked_keyframe(
        rgb_path=img_path,
        out_path=out_path,
        marks=[
            {"proposal_id": 0, "label": "chair", "bbox_2d": (50, 60, 150, 180)},
            {"proposal_id": 1, "label": "desk",  "bbox_2d": (200, 50, 310, 230)},
        ],
    )
    assert out_path.exists()
    out = Image.open(out_path)
    assert out.size == (320, 240)
```

- [ ] **Step 22.2: 实现**

```python
"""Render set-of-marks annotated keyframes."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_COLORS = [
    (220, 50, 50), (50, 200, 80), (50, 100, 220), (220, 180, 30),
    (180, 50, 200), (50, 200, 200), (240, 130, 30),
]


def render_marked_keyframe(
    *,
    rgb_path: Path,
    out_path: Path,
    marks: list[dict],
    font_size: int = 18,
) -> None:
    """marks: list of {proposal_id, label, bbox_2d=(x1,y1,x2,y2)}."""
    img = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    for i, m in enumerate(marks):
        color = _COLORS[i % len(_COLORS)]
        x1, y1, x2, y2 = m["bbox_2d"]
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        tag = f"{m['proposal_id']}: {m['label']}"
        tw, th = draw.textbbox((0, 0), tag, font=font)[2:]
        draw.rectangle([(x1, y1 - th - 4), (x1 + tw + 6, y1)], fill=color)
        draw.text((x1 + 3, y1 - th - 3), tag, fill="white", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


__all__ = ["render_marked_keyframe"]
```

- [ ] **Step 22.3: 跑测试 PASS**

```bash
pytest src/benchmarks/tests/test_render_marks.py -v
```

- [ ] **Step 22.4: commit**

```bash
git add src/benchmarks/embodiedscan_bbox_feasibility/render_marks.py src/benchmarks/tests/test_render_marks.py
git commit -m "feat(benchmarks/embodiedscan): set-of-marks keyframe renderer for VG pack-v1"
```

---

### Task 23 — Side-by-side validation 脚本

**Files:**
- Create: `src/evaluation/scripts/run_embodiedscan_vg_side_by_side.py`
- Test:   `src/evaluation/scripts/tests/test_run_embodiedscan_vg_side_by_side.py`

**目的:** 对同一份 EmbodiedScan ScanNet val 30-sample 子集,分别用
`vg_backend='legacy'` 与 `vg_backend='pack_v1'` 跑一遍,对比
`Acc@0.25 / Acc@0.5 / mean IoU`。验收门槛: pack-v1 的 Acc@0.25
不能比 legacy 低 1 个百分点以上。

- [ ] **Step 23.1: 写最小集成测试**(mock 一个 1 sample,确认两路径都
  能跑出 metric dict)

```python
"""Side-by-side runs both backends and produces a comparison table."""
from __future__ import annotations

import pytest


@pytest.mark.integration
def test_side_by_side_emits_comparison_dict(monkeypatch, tmp_path) -> None:
    from evaluation.scripts.run_embodiedscan_vg_side_by_side import (
        compare_backends,
    )

    def fake_run_one(sample_id, backend):
        return {"sample_id": sample_id, "iou": 0.5 if backend == "pack_v1" else 0.4}

    monkeypatch.setattr(
        "evaluation.scripts.run_embodiedscan_vg_side_by_side.run_one_sample",
        fake_run_one,
    )
    out = compare_backends(sample_ids=["s1", "s2"], output_dir=tmp_path)
    assert "legacy" in out and "pack_v1" in out
    assert out["pack_v1"]["mean_iou"] >= out["legacy"]["mean_iou"]
```

- [ ] **Step 23.2: 实现脚本骨架**

```python
"""Run EmbodiedScan VG legacy + pack-v1 backends side-by-side and report."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Sequence

from loguru import logger


def run_one_sample(sample_id: str, backend: str) -> dict:
    """Adapter: run one sample through the chosen backend; return
    {sample_id, iou, status}. Wires into your existing runner; for the
    test, this is monkeypatched."""
    raise NotImplementedError("wire into your existing pilot runner")


def compare_backends(
    *,
    sample_ids: Sequence[str],
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for backend in ("legacy", "pack_v1"):
        per_sample = [run_one_sample(s, backend) for s in sample_ids]
        ious = [r["iou"] for r in per_sample if r.get("iou") is not None]
        acc25 = sum(1 for v in ious if v >= 0.25) / max(len(ious), 1)
        acc50 = sum(1 for v in ious if v >= 0.50) / max(len(ious), 1)
        results[backend] = {
            "n": len(per_sample),
            "mean_iou": statistics.mean(ious) if ious else 0.0,
            "Acc@0.25": acc25,
            "Acc@0.50": acc50,
            "per_sample": per_sample,
        }
    (output_dir / "side_by_side.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "legacy: Acc@0.25={:.3f}  pack_v1: Acc@0.25={:.3f}",
        results["legacy"]["Acc@0.25"], results["pack_v1"]["Acc@0.25"],
    )
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-ids", required=True, type=Path,
                   help="JSON file with [sample_id, ...]")
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sample_ids = json.loads(args.sample_ids.read_text())
    compare_backends(sample_ids=sample_ids, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 23.3: 跑测试 PASS**

```bash
pytest src/evaluation/scripts/tests/test_run_embodiedscan_vg_side_by_side.py -v
```

- [ ] **Step 23.4: 在 30-sample 子集上跑(手工验收)**

```bash
# 选 30 个 sample(来自 batch30 同样的子集即可保证类别平衡)
python src/evaluation/scripts/run_embodiedscan_vg_side_by_side.py \
  --sample-ids docs/10_experiment_log/embodiedscan_3d_bbox_feasibility_report/resources/data/batch30_sample_ids.json \
  --output-dir outputs/vg_side_by_side
cat outputs/vg_side_by_side/side_by_side.json | jq '.legacy.Acc@0.25, .pack_v1.Acc@0.25'
```

**验收门槛:** `pack_v1.Acc@0.25 >= legacy.Acc@0.25 - 0.01`。如果不
满足,**不要** 推进到 Plan B/C;先回退、定位差距。

- [ ] **Step 23.5: commit**

```bash
git add src/evaluation/scripts/run_embodiedscan_vg_side_by_side.py src/evaluation/scripts/tests/
git commit -m "feat(evaluation): side-by-side EmbodiedScan VG legacy vs pack_v1 with acceptance gate"
```

---

### Task 24 — 触发 Plan A 验收 + 把结果写进 spec 末尾的 Implementation Status

**Files:**
- Modify: `docs/superpowers/specs/2026-04-25-stage2-multi-task-agent-design.md`(末尾追加)

- [ ] **Step 24.1: 把 step 23 的 metric 表追加到 spec**

在 spec 末尾追加 `## Plan A Implementation Status (YYYY-MM-DD)` 一节
(写入 30 样本上的实际 Acc@0.25 / Acc@0.5 / mean IoU 对比、PR
链接、未发现的回归)。

- [ ] **Step 24.2: commit**

```bash
git add docs/superpowers/specs/2026-04-25-stage2-multi-task-agent-design.md
git commit -m "docs(stage2): record Plan A implementation status with side-by-side metrics"
```

---

## Self-Review checklist (Plan A)

完成所有 task 后,本人对照 spec 自检:

- [ ] spec 第 1 节 Goal — Plan A 是否覆盖? **是** (steps 1–6)
- [ ] spec 第 2 节 Locked principle — chassis 是否仍然 task-agnostic?
  **是**(VG-specific 内容全在 `packs/vg_embodiedscan/` 内)
- [ ] spec 第 3 节 Architecture — 三层契约都落地? **是**
- [ ] spec 第 4 节 Registry + Task 1, 18 — **是**
- [ ] spec 第 5 节 Chassis tools + Task 7 — **是**
- [ ] spec 第 6 节 EmbodiedScan VG pack + Tasks 9–18 — **是**
- [ ] spec 第 7 节 Data contract + Task 9, 8 — **是**
- [ ] spec 第 8 节 FAIL-LOUD policies + Tasks 6, 7, 8, 12-16 — **是**
- [ ] spec 第 9 节 OpenEQA compatibility + Tasks 3, 4, 5 — **是**
- [ ] spec 第 10 节 Migration order — Plan A 覆盖 step 1-6 — **是**
- [ ] spec 第 11 节 Future Pack: Nav Plan — Plan D 处理(本 plan 不
  涉及) — **是**

placeholder scan: 全文搜索 "TBD"、"TODO"、"implement later"、
"add appropriate":

```bash
grep -n -E "TBD|TODO|implement later|add appropriate|fill in details" \
  docs/superpowers/plans/2026-04-25-stage2-multi-task-agent-plan-a-foundation-and-vg-pack-v1.md \
  || echo OK
```

期望: 仅匹配本 self-review 自身或注释里的"don't write TBD"。

类型/方法名一致性:在所有 task 里 `VgEmbodiedScanCtx`、`Proposal`、
`VgPayload`、`build_vg_proposal_pool`、`build_vg_tools` 命名一致。
**已确认**。

---

## 触发 Plan B 的条件

完成本 plan 全部 task,且 `outputs/vg_side_by_side/side_by_side.json`
满足:

```
pack_v1.Acc@0.25 >= legacy.Acc@0.25 - 0.01
```

时,可启动 Plan B (steps 7–8)。

如果不满足,**不要** 进 Plan B;在本仓建立 follow-up issue 定位
差距,优先级:
1. proposal_pool 是否带正确的 axis_align_matrix?
2. visibility_index 是否漏掉应可见 proposal?
3. set-of-marks 是否标错 proposal_id?
4. submit_final adapter 的 bbox_3d 是否被 axis-align 处理两次?
