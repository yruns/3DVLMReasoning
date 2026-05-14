# v9 Catalog-First Scene Exploration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reposition Stage 1 (KeyframeSelector) from a pack-prep one-shot to a first-class Stage-2 agent tool family, introduce a unified `SceneCatalog` abstraction shared by VG (NR3D / ScanRefer) and QA (OpenEQA / SQA3D), and replace the v8 "clean RGB seed + text inventory" initial state with a catalog-first (BEV image + Cat-B text) initial state with 6 modality-diverse selectors and 3 scene-perception tools.

**Architecture:** Three layers — (a) `src/agents/catalog/` defines pydantic `SceneCatalog` / `SceneProposal` / `FrameView` and three adapters (vg pool, conceptgraph objects, GT embodiedscan); (b) `src/query_scene/scene_bev_builder.py` defines `ScanNetSceneBEVBuilderBase` and 4 benchmark subclasses that render mesh + trajectory + `#id label` BEV images; (c) `src/agents/tools/{selectors,scene_perception,view_keyframe}.py` define the 6 + 4 + 1 + 1 = 12 new/refactored tools, all gated on a shared `scene-exploration-playbook`. Pack-prep scripts emit `scene_catalog.json` + BEV png; sample JSON drops `keyframes` and `vg_proposal_pool` (catalog-first). `request_more_views` / `switch_or_expand_hypothesis` callbacks are hard-deleted; only `request_crops` survives.

**Tech Stack:** Python 3.12, uv-managed `.venv`, Pydantic v2, LangChain v1 + DeepAgents, NumPy, Open3D, OpenCV, Pillow, pytest. macOS dev workflow: `source .venv/bin/activate` + `PYTHONPATH=src python -m pytest ...`.

---

## Pre-flight: shell setup once per shell session

All `pytest` and `python` commands below assume the worktree's venv is active and `PYTHONPATH=src` is set. Run once per shell:

```bash
cd /Users/bytedance/project/3DVLMReasoning/.worktrees/v9-catalog-first
source .venv/bin/activate
export PYTHONPATH=src
```

If you run a fresh subagent that does not inherit the env vars, prefix individual commands with the activation + PYTHONPATH (the explicit form is shown in each step).

---

## File Structure Overview

### New modules

```
src/agents/catalog/
├── __init__.py                         re-exports SceneCatalog, SceneProposal, FrameView, adapters
├── models.py                           pydantic v2 models
├── adapters.py                         from_vg_proposal_pool / from_conceptgraph_objects / from_gt_embodiedscan
└── tests/
    ├── __init__.py
    ├── test_models.py
    ├── test_adapter_vg.py
    └── test_adapter_qa.py

src/query_scene/scene_bev_builder.py    NEW — base + 4 benchmark builders (separate file; existing `bev_builder.py` keeps Replica / OpenEQAScanNetBEVBuilder legacy)

src/agents/tools/scene_perception.py    NEW — view_bev, list_scene_proposals, inspect_proposal_v9 builders
src/agents/tools/view_keyframe.py       NEW — unified view_keyframe(mode='rgb'|'marked'|'auto', categories?, proposal_ids?)
src/agents/tools/selectors.py           NEW — 6 select_by_* tool builders

src/agents/skills/shared_skills/scene_exploration_playbook.md   NEW — gate skill for selectors + scene perception + view_keyframe

src/agents/runtime/scene_runtime.py     NEW — runtime helpers reading SceneCatalog from bundle.extra_metadata

src/evaluation/scripts/prepare_pack_qa_inputs.py   NEW — OpenEQA / SQA3D pack-v1 prep (writes BEV + scene_catalog.json + sample JSON, no first-person seed)
```

### Modified files

```
src/agents/models.py                                   re-export catalog types (only)
src/agents/runtime/base.py                             build_system_prompt rewrite; collect_image_paths returns BEV only
src/agents/runtime/deepagents_agent.py                 build_task_message catalog-first; delete request_more_views / switch_or_expand_hypothesis / inspect_stage1_metadata tool wrappers
src/agents/core/agent_config.py                        delete enable_temporal_fan; bump chassis_tools_version=20
src/agents/packs/vg_embodiedscan/tools.py              delete list_keyframes_with_proposals / view_keyframe_marked / find_proposals_by_category; keep list_frame_proposals; refactor compare_proposals_spatial / inspect_proposal to read scene_catalog
src/agents/packs/vg_embodiedscan/registration.py       refresh required_extra_metadata=["scene_catalog"], skills list (drop evidence-scouting), required_primary_skill='scene-exploration-playbook'
src/agents/packs/qa_default/registration.py            attach scene tools; skills list (drop evidence-scouting, add scene-exploration); set exposes_chassis=True (already true)
src/agents/packs/qa_default/tools.py                   wire pack tool builder to register selectors + scene perception + view_keyframe via shared module
src/agents/stage1_callbacks.py                         delete create_more_views_callback, create_hypothesis_callback; keep create_crop_callback
src/agents/skills/no_match_guard.py                    replace 'find_proposals_by_category' + 'view_keyframe_marked' with 'list_scene_proposals' + 'view_keyframe'
src/agents/skills/evidence_frame_guard.py              replace 'view_keyframe_marked' citations with 'view_keyframe' (mode='marked')
src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md     rewrite (selectors + view_keyframe + BEV)
src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation.md small edits (view_keyframe(mode='marked'))
src/agents/packs/qa_default/skills/qa_answering_playbook.md          rewrite (selectors + view_keyframe(rgb) + BEV)
src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py                add BEV + scene_catalog writes; drop normalize_prepared_keyframes call; sample JSON loses 'keyframes', adds 'scene_catalog_path', 'bev_image_path'
src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py           same
src/evaluation/scripts/run_nr3d_vg_side_by_side.py                   delete callback construction except crops
src/evaluation/scripts/run_scanrefer_vg_side_by_side.py              same
```

### Deleted files

```
src/agents/packs/qa_default/skills/evidence_scouting.md
src/agents/packs/vg_embodiedscan/skills/evidence_scouting.md
```

---

## Conventions used in this plan

- Every code step shows the exact code to add / replace. No "TBD" or "fill in" placeholders.
- Every test step shows the assertions. Tests use pytest fixtures and tmp_path.
- `Write failing test → Run to fail → Implement → Run to pass → Commit` is the canonical sub-flow for every new piece of code.
- Each task ends with a `git add` + `git commit` step. Commit messages follow `feat:` / `refactor:` / `test:` / `chore:` / `docs:` prefixes from the repo's history.
- Ruff is run via `ruff check src/agents/catalog src/query_scene src/agents/tools` at the end of every task that touched those folders.
- The data structures in early tasks (`SceneCatalog`, `SceneProposal`, `FrameView`) are the contract that later tasks (selectors, tools, pack prep) build on. **Do not deviate from the field names defined in Task 1.**

---

## Phase 1 — Foundation (SceneCatalog + adapters)

### Task 1: SceneCatalog Pydantic Models

**Files:**
- Create: `src/agents/catalog/__init__.py`
- Create: `src/agents/catalog/models.py`
- Create: `src/agents/catalog/tests/__init__.py`
- Create: `src/agents/catalog/tests/test_models.py`

- [ ] **Step 1: Create empty `__init__.py` files**

```bash
mkdir -p src/agents/catalog/tests
: > src/agents/catalog/__init__.py
: > src/agents/catalog/tests/__init__.py
```

- [ ] **Step 2: Write failing model tests**

Create `src/agents/catalog/tests/test_models.py`:

```python
from agents.catalog.models import FrameView, SceneCatalog, SceneProposal


def test_frame_view_minimal_required_fields():
    fv = FrameView(frame_id=10, bbox_2d=(1, 2, 3, 4), raw_rgb_path="a.png")
    assert fv.frame_id == 10
    assert fv.bbox_2d == (1, 2, 3, 4)
    assert fv.raw_rgb_path == "a.png"
    assert fv.visibility_weight is None


def test_frame_view_with_visibility_weight():
    fv = FrameView(
        frame_id=11,
        bbox_2d=(0, 0, 100, 200),
        raw_rgb_path="b.png",
        visibility_weight=0.42,
    )
    assert fv.visibility_weight == 0.42


def test_scene_proposal_default_frame_views_empty():
    p = SceneProposal(
        proposal_id=7,
        category="chair",
        position_3d=(1.0, 2.0, 0.5),
        source="mask3d",
    )
    assert p.frame_views == {}
    assert p.bbox_3d_9dof is None
    assert p.source == "mask3d"


def test_scene_proposal_with_views_and_9dof():
    fv = FrameView(frame_id=20, bbox_2d=(0, 0, 50, 50), raw_rgb_path="c.png")
    p = SceneProposal(
        proposal_id=3,
        category="table",
        position_3d=(0.0, 0.0, 0.0),
        bbox_3d_9dof=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        frame_views={20: fv},
        source="gt",
    )
    assert p.frame_views[20].frame_id == 20
    assert len(p.bbox_3d_9dof) == 9


def test_scene_catalog_roundtrip():
    proposals = [
        SceneProposal(
            proposal_id=0,
            category="chair",
            position_3d=(0.0, 0.0, 0.0),
            source="mask3d",
        ),
        SceneProposal(
            proposal_id=1,
            category="chair",
            position_3d=(1.0, 0.0, 0.0),
            source="mask3d",
        ),
        SceneProposal(
            proposal_id=2,
            category="table",
            position_3d=(2.0, 0.0, 0.0),
            source="mask3d",
        ),
    ]
    catalog = SceneCatalog(
        scene_id="scannet/scene0000_00",
        scene_category="kitchen",
        proposals=proposals,
        total_frames=100,
        frame_id_range=(0, 990),
        valid_frame_ids=[0, 10, 20, 30],
        bev_image_path="bev/scene_bev_v9.png",
    )
    dumped = catalog.model_dump()
    reloaded = SceneCatalog(**dumped)
    assert reloaded.scene_id == "scannet/scene0000_00"
    assert reloaded.scene_category == "kitchen"
    assert reloaded.total_frames == 100
    assert reloaded.frame_id_range == (0, 990)
    assert reloaded.valid_frame_ids == [0, 10, 20, 30]
    assert reloaded.bev_image_path == "bev/scene_bev_v9.png"
    assert len(reloaded.proposals) == 3


def test_scene_catalog_proposals_by_category_groups_ids():
    proposals = [
        SceneProposal(proposal_id=0, category="chair", position_3d=(0, 0, 0), source="mask3d"),
        SceneProposal(proposal_id=1, category="chair", position_3d=(1, 0, 0), source="mask3d"),
        SceneProposal(proposal_id=2, category="table", position_3d=(2, 0, 0), source="mask3d"),
    ]
    catalog = SceneCatalog(
        scene_id="s",
        proposals=proposals,
        total_frames=10,
        frame_id_range=(0, 90),
        valid_frame_ids=[0],
        bev_image_path="x.png",
    )
    grouped = catalog.proposals_by_category()
    assert grouped == {"chair": [0, 1], "table": [2]}


def test_scene_proposal_rejects_unknown_source():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SceneProposal(
            proposal_id=0,
            category="x",
            position_3d=(0, 0, 0),
            source="foo",  # not in Literal
        )
```

- [ ] **Step 3: Run test to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/catalog/tests/test_models.py -q
```

Expected: `ModuleNotFoundError: No module named 'agents.catalog.models'`.

- [ ] **Step 4: Implement models**

Create `src/agents/catalog/models.py`:

```python
"""SceneCatalog and supporting models for v9 catalog-first scene exploration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FrameView(BaseModel):
    """One proposal's projection into one frame."""

    model_config = ConfigDict(frozen=False)

    frame_id: int
    bbox_2d: tuple[int, int, int, int]
    raw_rgb_path: str
    visibility_weight: float | None = None


class SceneProposal(BaseModel):
    """Unified scene entity unit (VG Mask3D/V-DETR/GT proposal or QA conceptgraph object)."""

    model_config = ConfigDict(frozen=False)

    proposal_id: int
    category: str
    position_3d: tuple[float, float, float]
    bbox_3d_9dof: tuple[float, float, float, float, float, float, float, float, float] | None = None
    frame_views: dict[int, FrameView] = Field(default_factory=dict)
    source: Literal["mask3d", "vdetr", "gt", "conceptgraph"]


class SceneCatalog(BaseModel):
    """Scene-level catalog shared across an entire Stage-2 session."""

    model_config = ConfigDict(frozen=False)

    scene_id: str
    scene_category: str | None = None
    proposals: list[SceneProposal] = Field(default_factory=list)
    total_frames: int
    frame_id_range: tuple[int, int]
    valid_frame_ids: list[int] = Field(default_factory=list)
    bev_image_path: str
    axis_align_matrix: list[list[float]] | None = None

    def proposals_by_category(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for p in self.proposals:
            out.setdefault(p.category, []).append(p.proposal_id)
        return out

    def proposal_by_id(self, proposal_id: int) -> SceneProposal | None:
        for p in self.proposals:
            if p.proposal_id == proposal_id:
                return p
        return None


__all__ = ["FrameView", "SceneProposal", "SceneCatalog"]
```

- [ ] **Step 5: Populate `__init__.py` re-exports**

Replace `src/agents/catalog/__init__.py` content with:

```python
"""Catalog public API for v9 catalog-first scene exploration."""

from agents.catalog.models import FrameView, SceneCatalog, SceneProposal

__all__ = ["FrameView", "SceneCatalog", "SceneProposal"]
```

- [ ] **Step 6: Run tests to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/catalog/tests/test_models.py -q
```

Expected: all 7 tests pass.

- [ ] **Step 7: Ruff check**

```bash
.venv/bin/ruff check src/agents/catalog
```

Expected: no findings.

- [ ] **Step 8: Commit**

```bash
git add src/agents/catalog/__init__.py src/agents/catalog/models.py src/agents/catalog/tests/__init__.py src/agents/catalog/tests/test_models.py
git commit -m "$(cat <<'EOF'
feat(catalog): add SceneCatalog/SceneProposal/FrameView models

Introduces the v9 catalog-first scene exploration data contract that VG
and QA packs will share. SceneProposal carries no detector confidence
score to discourage agents from treating Stage-1 priors as ground truth.

Per docs/superpowers/specs/2026-05-14-v9-catalog-first-scene-exploration-design.md
Section B.
EOF
)"
```

### Task 2: VG Adapter (`from_vg_proposal_pool`)

**Files:**
- Modify: `src/agents/catalog/adapters.py` (create)
- Modify: `src/agents/catalog/__init__.py` (extend re-exports)
- Create: `src/agents/catalog/tests/test_adapter_vg.py`

- [ ] **Step 1: Write failing adapter test**

Create `src/agents/catalog/tests/test_adapter_vg.py`:

```python
from agents.catalog import SceneCatalog
from agents.catalog.adapters import from_vg_proposal_pool


def _fixture_pool() -> dict:
    return {
        "source": "vdetr",
        "proposals": [
            {
                "id": 0,
                "bbox_3d_9dof": [0.0, 1.0, 0.5, 0.4, 0.4, 0.4, 0.0, 0.0, 0.0],
                "category": "chair",
                "score": 0.92,
                "frame_views": [
                    {
                        "frame_id": 10,
                        "bbox_2d": [12, 20, 30, 60],
                        "raw_rgb_path": "raw/000010-rgb.png",
                        "visibility_weight": 0.83,
                    },
                ],
            },
            {
                "id": 1,
                "bbox_3d_9dof": [1.2, 1.4, 0.5, 0.6, 0.6, 0.6, 0.0, 0.0, 0.0],
                "category": "table",
                "score": 0.88,
                "frame_views": [
                    {
                        "frame_id": 10,
                        "bbox_2d": [80, 30, 140, 80],
                        "raw_rgb_path": "raw/000010-rgb.png",
                    },
                    {
                        "frame_id": 20,
                        "bbox_2d": [30, 30, 70, 70],
                        "raw_rgb_path": "raw/000020-rgb.png",
                    },
                ],
            },
        ],
        "frame_index": {10: [0, 1], 20: [1]},
        "proposal_index": {0: [10], 1: [10, 20]},
        "annotated_image_dir": "annotated",
    }


def test_from_vg_proposal_pool_returns_catalog_with_required_fields():
    catalog = from_vg_proposal_pool(
        pool=_fixture_pool(),
        scene_id="scannet/scene0000_00",
        bev_image_path="bev/scene_bev_v9.png",
        scene_category="kitchen",
        axis_align_matrix=None,
        valid_frame_ids=[10, 20],
    )
    assert isinstance(catalog, SceneCatalog)
    assert catalog.scene_id == "scannet/scene0000_00"
    assert catalog.scene_category == "kitchen"
    assert catalog.total_frames == 2
    assert catalog.frame_id_range == (10, 20)
    assert catalog.valid_frame_ids == [10, 20]
    assert catalog.bev_image_path == "bev/scene_bev_v9.png"
    assert catalog.axis_align_matrix is None


def test_from_vg_proposal_pool_drops_score_and_keeps_position_from_9dof():
    catalog = from_vg_proposal_pool(
        pool=_fixture_pool(),
        scene_id="s",
        bev_image_path="b.png",
        scene_category=None,
        axis_align_matrix=None,
        valid_frame_ids=[10, 20],
    )
    chair = next(p for p in catalog.proposals if p.proposal_id == 0)
    assert chair.category == "chair"
    assert chair.position_3d == (0.0, 1.0, 0.5)
    assert chair.bbox_3d_9dof == (0.0, 1.0, 0.5, 0.4, 0.4, 0.4, 0.0, 0.0, 0.0)
    assert chair.source == "vdetr"
    assert not hasattr(chair, "score")


def test_from_vg_proposal_pool_normalizes_frame_views():
    catalog = from_vg_proposal_pool(
        pool=_fixture_pool(),
        scene_id="s",
        bev_image_path="b.png",
        scene_category=None,
        axis_align_matrix=None,
        valid_frame_ids=[10, 20],
    )
    table = next(p for p in catalog.proposals if p.proposal_id == 1)
    assert set(table.frame_views.keys()) == {10, 20}
    v10 = table.frame_views[10]
    assert v10.bbox_2d == (80, 30, 140, 80)
    assert v10.raw_rgb_path == "raw/000010-rgb.png"
    assert v10.visibility_weight is None


def test_from_vg_proposal_pool_passes_through_axis_align_matrix():
    pool = _fixture_pool()
    axis = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
    catalog = from_vg_proposal_pool(
        pool=pool,
        scene_id="s",
        bev_image_path="b.png",
        scene_category=None,
        axis_align_matrix=axis,
        valid_frame_ids=[10, 20],
    )
    assert catalog.axis_align_matrix == axis


def test_from_vg_proposal_pool_source_maps_through():
    pool = _fixture_pool()
    pool["source"] = "gt"
    catalog = from_vg_proposal_pool(
        pool=pool,
        scene_id="s",
        bev_image_path="b.png",
        scene_category=None,
        axis_align_matrix=None,
        valid_frame_ids=[10, 20],
    )
    assert all(p.source == "gt" for p in catalog.proposals)


def test_from_vg_proposal_pool_empty_valid_frames_errors():
    import pytest

    with pytest.raises(ValueError, match="valid_frame_ids"):
        from_vg_proposal_pool(
            pool=_fixture_pool(),
            scene_id="s",
            bev_image_path="b.png",
            scene_category=None,
            axis_align_matrix=None,
            valid_frame_ids=[],
        )
```

- [ ] **Step 2: Run test to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/catalog/tests/test_adapter_vg.py -q
```

Expected: `ModuleNotFoundError: No module named 'agents.catalog.adapters'`.

- [ ] **Step 3: Implement the adapter**

Create `src/agents/catalog/adapters.py`:

```python
"""Adapters that build a SceneCatalog from benchmark-specific assets."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from agents.catalog.models import FrameView, SceneCatalog, SceneProposal

VgPoolSource = Literal["mask3d", "vdetr", "gt", "conceptgraph"]


def _build_frame_view(raw: dict) -> FrameView:
    return FrameView(
        frame_id=int(raw["frame_id"]),
        bbox_2d=tuple(int(v) for v in raw["bbox_2d"]),  # type: ignore[arg-type]
        raw_rgb_path=str(raw["raw_rgb_path"]),
        visibility_weight=(
            float(raw["visibility_weight"])
            if raw.get("visibility_weight") is not None
            else None
        ),
    )


def _frame_views_from_raw(raw: Any) -> dict[int, FrameView]:
    if raw is None:
        return {}
    out: dict[int, FrameView] = {}
    if isinstance(raw, dict):
        for fid, value in raw.items():
            item = dict(value)
            item.setdefault("frame_id", int(fid))
            view = _build_frame_view(item)
            out[view.frame_id] = view
        return out
    if isinstance(raw, list):
        for value in raw:
            view = _build_frame_view(dict(value))
            out[view.frame_id] = view
        return out
    raise ValueError(f"frame_views must be dict or list, got {type(raw).__name__}")


def from_vg_proposal_pool(
    *,
    pool: dict[str, Any],
    scene_id: str,
    bev_image_path: str,
    scene_category: str | None,
    axis_align_matrix: list[list[float]] | None,
    valid_frame_ids: list[int],
) -> SceneCatalog:
    """Convert a VG proposal pool (NR3D / ScanRefer / EmbodiedScan) into SceneCatalog."""
    if not valid_frame_ids:
        raise ValueError("from_vg_proposal_pool: valid_frame_ids must be non-empty")
    source: VgPoolSource = pool.get("source", "mask3d")
    if source not in ("mask3d", "vdetr", "gt", "conceptgraph"):
        raise ValueError(f"from_vg_proposal_pool: unsupported source {source!r}")

    proposals: list[SceneProposal] = []
    for raw in pool.get("proposals", []) or []:
        bbox = raw.get("bbox_3d_9dof") or raw.get("bbox_3d")
        if bbox is None or len(bbox) != 9:
            raise ValueError(
                f"proposal id={raw.get('id')} bbox_3d_9dof must have 9 elements"
            )
        bbox9 = tuple(float(v) for v in bbox)
        position = (float(bbox9[0]), float(bbox9[1]), float(bbox9[2]))
        proposals.append(
            SceneProposal(
                proposal_id=int(raw["id"]),
                category=str(raw.get("category") or raw.get("label") or ""),
                position_3d=position,
                bbox_3d_9dof=bbox9,
                frame_views=_frame_views_from_raw(raw.get("frame_views")),
                source=source,
            )
        )

    sorted_frames = sorted(int(fid) for fid in valid_frame_ids)
    catalog = SceneCatalog(
        scene_id=scene_id,
        scene_category=scene_category,
        proposals=proposals,
        total_frames=len(sorted_frames),
        frame_id_range=(sorted_frames[0], sorted_frames[-1]),
        valid_frame_ids=sorted_frames,
        bev_image_path=bev_image_path,
        axis_align_matrix=axis_align_matrix,
    )
    return catalog


__all__ = ["from_vg_proposal_pool"]
```

- [ ] **Step 4: Extend `__init__.py` re-exports**

Replace `src/agents/catalog/__init__.py` with:

```python
"""Catalog public API for v9 catalog-first scene exploration."""

from agents.catalog.adapters import from_vg_proposal_pool
from agents.catalog.models import FrameView, SceneCatalog, SceneProposal

__all__ = [
    "FrameView",
    "SceneCatalog",
    "SceneProposal",
    "from_vg_proposal_pool",
]
```

- [ ] **Step 5: Run adapter tests to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/catalog/tests/test_adapter_vg.py -q
```

Expected: all 6 tests pass.

- [ ] **Step 6: Ruff check**

```bash
.venv/bin/ruff check src/agents/catalog
```

Expected: no findings.

- [ ] **Step 7: Commit**

```bash
git add src/agents/catalog/__init__.py src/agents/catalog/adapters.py src/agents/catalog/tests/test_adapter_vg.py
git commit -m "$(cat <<'EOF'
feat(catalog): add from_vg_proposal_pool adapter

Converts NR3D / ScanRefer / EmbodiedScan VG proposal pools (proposals.jsonl
+ frame_views) into a SceneCatalog. Drops detector confidence to keep it
off the agent prompt.

Per spec Section B.
EOF
)"
```

### Task 3: QA Adapter (`from_conceptgraph_objects`) + GT Adapter

**Files:**
- Modify: `src/agents/catalog/adapters.py`
- Modify: `src/agents/catalog/__init__.py`
- Create: `src/agents/catalog/tests/test_adapter_qa.py`

- [ ] **Step 1: Write failing QA adapter tests**

Create `src/agents/catalog/tests/test_adapter_qa.py`:

```python
import gzip
import pickle
from pathlib import Path

from agents.catalog.adapters import (
    from_conceptgraph_objects,
    from_gt_embodiedscan,
)


def _write_conceptgraph_assets(
    base: Path,
    *,
    objects: list[dict],
) -> tuple[Path, Path]:
    pcd_dir = base / "pcd_saves"
    det_dir = base / "gsa_detections_ram_withbg_allclasses"
    pcd_dir.mkdir(parents=True)
    det_dir.mkdir(parents=True)
    payload = {"objects": objects}
    with gzip.open(pcd_dir / "full_pcd_v9.pkl.gz", "wb") as fh:
        pickle.dump(payload, fh)
    return pcd_dir, det_dir


def test_from_conceptgraph_objects_builds_catalog(tmp_path: Path):
    objects = [
        {
            "id": 0,
            "category": "chair",
            "bbox_3d_9dof": [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0],
        },
        {
            "id": 1,
            "category": "table",
            "bbox_3d_9dof": [1.0, 1.0, 0.0, 1.0, 1.0, 0.8, 0.0, 0.0, 0.0],
        },
    ]
    pcd_dir, det_dir = _write_conceptgraph_assets(tmp_path, objects=objects)
    view_to_objects = {
        10: [(0, 0.9), (1, 0.7)],
        20: [(1, 0.8)],
    }
    catalog = from_conceptgraph_objects(
        pcd_saves_dir=pcd_dir,
        detections_dir=det_dir,
        view_to_objects=view_to_objects,
        scene_id="openeqa/scene0709_00",
        bev_image_path="bev/scene_bev_qa.png",
        scene_category="bedroom",
        valid_frame_ids=[10, 20],
        raw_rgb_template=str(tmp_path / "raw/{frame_id:06d}-rgb.png"),
    )
    assert catalog.scene_id == "openeqa/scene0709_00"
    assert catalog.total_frames == 2
    assert {p.proposal_id for p in catalog.proposals} == {0, 1}
    chair = next(p for p in catalog.proposals if p.proposal_id == 0)
    assert chair.category == "chair"
    assert chair.position_3d == (0.0, 0.0, 0.0)
    assert chair.source == "conceptgraph"
    assert set(chair.frame_views.keys()) == {10}
    assert chair.frame_views[10].visibility_weight == 0.9
    table = next(p for p in catalog.proposals if p.proposal_id == 1)
    assert set(table.frame_views.keys()) == {10, 20}


def test_from_conceptgraph_objects_skips_objects_with_no_visible_frame(tmp_path: Path):
    objects = [
        {"id": 0, "category": "chair", "bbox_3d_9dof": [0]*9},
        {"id": 5, "category": "monitor", "bbox_3d_9dof": [1]*9},
    ]
    pcd_dir, det_dir = _write_conceptgraph_assets(tmp_path, objects=objects)
    view_to_objects = {10: [(0, 0.5)]}
    catalog = from_conceptgraph_objects(
        pcd_saves_dir=pcd_dir,
        detections_dir=det_dir,
        view_to_objects=view_to_objects,
        scene_id="s",
        bev_image_path="b.png",
        scene_category=None,
        valid_frame_ids=[10],
        raw_rgb_template=str(tmp_path / "raw/{frame_id:06d}-rgb.png"),
    )
    proposal_ids = {p.proposal_id for p in catalog.proposals}
    # Monitor (id=5) has zero visible frames -> excluded from catalog
    assert proposal_ids == {0}


def test_from_gt_embodiedscan_uses_gt_source(tmp_path: Path):
    es_annotations = {
        "instances": [
            {
                "bbox_id": 7,
                "category": "sofa",
                "bbox_3d_9dof": [0.0, 0.0, 0.4, 1.5, 1.0, 0.5, 0.0, 0.0, 0.0],
            }
        ],
        "view_to_objects": {12: [(7, 1.0)]},
    }
    catalog = from_gt_embodiedscan(
        es_annotations=es_annotations,
        scene_id="es/scene0000_00",
        bev_image_path="bev/scene_bev_gt.png",
        valid_frame_ids=[12],
        raw_rgb_template=str(tmp_path / "raw/{frame_id:06d}-rgb.png"),
    )
    assert catalog.proposals[0].source == "gt"
    assert catalog.proposals[0].proposal_id == 7
```

- [ ] **Step 2: Run test to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/catalog/tests/test_adapter_qa.py -q
```

Expected: `ImportError: cannot import name 'from_conceptgraph_objects'`.

- [ ] **Step 3: Implement QA + GT adapters**

Append to `src/agents/catalog/adapters.py`:

```python
import gzip
import pickle
from pathlib import Path


def _load_conceptgraph_objects(pcd_saves_dir: Path) -> list[dict]:
    candidates = sorted(pcd_saves_dir.glob("full_pcd*.pkl.gz"))
    if not candidates:
        raise FileNotFoundError(
            f"no full_pcd*.pkl.gz under {pcd_saves_dir}; cannot build QA catalog"
        )
    with gzip.open(candidates[-1], "rb") as fh:
        payload = pickle.load(fh)
    objects = payload.get("objects") if isinstance(payload, dict) else None
    if not isinstance(objects, list):
        raise ValueError(
            f"conceptgraph payload at {candidates[-1]} missing 'objects' list"
        )
    return objects


def _invert_view_to_objects(
    view_to_objects: dict[int, list[tuple[int, float]]],
) -> dict[int, list[tuple[int, float]]]:
    """Return object_id -> [(frame_id, weight), ...]."""
    out: dict[int, list[tuple[int, float]]] = {}
    for frame_id, entries in view_to_objects.items():
        for obj_id, weight in entries:
            out.setdefault(int(obj_id), []).append((int(frame_id), float(weight)))
    return out


def from_conceptgraph_objects(
    *,
    pcd_saves_dir: Path,
    detections_dir: Path,
    view_to_objects: dict[int, list[tuple[int, float]]],
    scene_id: str,
    bev_image_path: str,
    scene_category: str | None,
    valid_frame_ids: list[int],
    raw_rgb_template: str,
) -> SceneCatalog:
    """Build a SceneCatalog from ConceptGraph 3D-object segmentation output (QA path)."""
    if not valid_frame_ids:
        raise ValueError("from_conceptgraph_objects: valid_frame_ids must be non-empty")
    del detections_dir  # accepted for API parity / future enrichment
    raw_objects = _load_conceptgraph_objects(Path(pcd_saves_dir))
    obj_to_frames = _invert_view_to_objects(view_to_objects)

    proposals: list[SceneProposal] = []
    for raw in raw_objects:
        obj_id = int(raw["id"])
        frames = obj_to_frames.get(obj_id) or []
        if not frames:
            continue
        bbox = raw.get("bbox_3d_9dof") or raw.get("bbox_3d")
        if bbox is None or len(bbox) != 9:
            raise ValueError(
                f"conceptgraph object id={obj_id} bbox_3d_9dof must have 9 elements"
            )
        bbox9 = tuple(float(v) for v in bbox)
        frame_views: dict[int, FrameView] = {}
        for frame_id, weight in frames:
            frame_views[frame_id] = FrameView(
                frame_id=frame_id,
                bbox_2d=(0, 0, 0, 0),  # QA pack does not require per-view 2D bboxes
                raw_rgb_path=raw_rgb_template.format(frame_id=frame_id),
                visibility_weight=weight,
            )
        proposals.append(
            SceneProposal(
                proposal_id=obj_id,
                category=str(raw.get("category") or raw.get("label") or "object"),
                position_3d=(bbox9[0], bbox9[1], bbox9[2]),
                bbox_3d_9dof=bbox9,
                frame_views=frame_views,
                source="conceptgraph",
            )
        )

    sorted_frames = sorted(int(fid) for fid in valid_frame_ids)
    return SceneCatalog(
        scene_id=scene_id,
        scene_category=scene_category,
        proposals=proposals,
        total_frames=len(sorted_frames),
        frame_id_range=(sorted_frames[0], sorted_frames[-1]),
        valid_frame_ids=sorted_frames,
        bev_image_path=bev_image_path,
        axis_align_matrix=None,
    )


def from_gt_embodiedscan(
    *,
    es_annotations: dict[str, Any],
    scene_id: str,
    bev_image_path: str,
    valid_frame_ids: list[int],
    raw_rgb_template: str,
) -> SceneCatalog:
    """Build a SceneCatalog from EmbodiedScan GT annotations (oracle pack)."""
    if not valid_frame_ids:
        raise ValueError("from_gt_embodiedscan: valid_frame_ids must be non-empty")
    instances = es_annotations.get("instances") or []
    obj_to_frames = _invert_view_to_objects(
        es_annotations.get("view_to_objects") or {}
    )
    proposals: list[SceneProposal] = []
    for inst in instances:
        obj_id = int(inst.get("bbox_id") or inst["id"])
        bbox = inst.get("bbox_3d_9dof") or inst.get("bbox_3d")
        if bbox is None or len(bbox) != 9:
            raise ValueError(
                f"GT instance bbox_id={obj_id} bbox_3d_9dof must have 9 elements"
            )
        bbox9 = tuple(float(v) for v in bbox)
        frame_views: dict[int, FrameView] = {}
        for frame_id, weight in obj_to_frames.get(obj_id, []):
            frame_views[frame_id] = FrameView(
                frame_id=frame_id,
                bbox_2d=(0, 0, 0, 0),
                raw_rgb_path=raw_rgb_template.format(frame_id=frame_id),
                visibility_weight=weight,
            )
        proposals.append(
            SceneProposal(
                proposal_id=obj_id,
                category=str(inst.get("category") or inst.get("label") or "object"),
                position_3d=(bbox9[0], bbox9[1], bbox9[2]),
                bbox_3d_9dof=bbox9,
                frame_views=frame_views,
                source="gt",
            )
        )
    sorted_frames = sorted(int(fid) for fid in valid_frame_ids)
    return SceneCatalog(
        scene_id=scene_id,
        proposals=proposals,
        total_frames=len(sorted_frames),
        frame_id_range=(sorted_frames[0], sorted_frames[-1]),
        valid_frame_ids=sorted_frames,
        bev_image_path=bev_image_path,
    )


__all__ = [
    "from_vg_proposal_pool",
    "from_conceptgraph_objects",
    "from_gt_embodiedscan",
]
```

- [ ] **Step 4: Update `__init__.py` re-exports**

Replace `src/agents/catalog/__init__.py` with:

```python
"""Catalog public API for v9 catalog-first scene exploration."""

from agents.catalog.adapters import (
    from_conceptgraph_objects,
    from_gt_embodiedscan,
    from_vg_proposal_pool,
)
from agents.catalog.models import FrameView, SceneCatalog, SceneProposal

__all__ = [
    "FrameView",
    "SceneCatalog",
    "SceneProposal",
    "from_vg_proposal_pool",
    "from_conceptgraph_objects",
    "from_gt_embodiedscan",
]
```

- [ ] **Step 5: Run all catalog tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/catalog/tests/ -q
```

Expected: all model + adapter tests pass.

- [ ] **Step 6: Ruff check**

```bash
.venv/bin/ruff check src/agents/catalog
```

Expected: no findings.

- [ ] **Step 7: Commit**

```bash
git add src/agents/catalog/__init__.py src/agents/catalog/adapters.py src/agents/catalog/tests/test_adapter_qa.py
git commit -m "$(cat <<'EOF'
feat(catalog): add from_conceptgraph_objects + from_gt_embodiedscan

QA pack (OpenEQA / SQA3D) builds its SceneCatalog from ConceptGraph
3D-object segmentation; oracle EmbodiedScan path uses GT instances.
Per-frame 2D bboxes default to zeros for QA — only visibility weights
matter there.

Per spec Section B.
EOF
)"
```

---

## Phase 2 — BEV Builders (base + 4 benchmark subclasses)

### Task 4: ScanNetSceneBEVBuilderBase (shared abstract class)

**Files:**
- Create: `src/query_scene/scene_bev_builder.py`
- Create: `src/query_scene/tests/test_scene_bev_builder_base.py`

Note: the existing `src/query_scene/bev_builder.py` (with legacy `OpenEQAScanNetBEVBuilder`) is kept untouched until Task 7 swaps its callers. The v9 builders live in the new `scene_bev_builder.py` to keep diffs reviewable.

- [ ] **Step 1: Write failing base-class test**

Create `src/query_scene/tests/test_scene_bev_builder_base.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from agents.catalog import SceneProposal
from query_scene.scene_bev_builder import (
    SceneBEVConfig,
    ScanNetSceneBEVBuilderBase,
)


class _DummyBuilder(ScanNetSceneBEVBuilderBase):
    benchmark = "dummy"

    def resolve_paths(self, scene_id, data_root):
        raise NotImplementedError


def test_overlay_proposal_labels_draws_id_and_category(tmp_path: Path):
    builder = _DummyBuilder(config=SceneBEVConfig(image_size=400))
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    proposals = [
        SceneProposal(
            proposal_id=12,
            category="chair",
            position_3d=(0.5, 0.5, 0.0),
            source="mask3d",
        ),
        SceneProposal(
            proposal_id=31,
            category="desk",
            position_3d=(-0.5, -0.5, 0.0),
            source="mask3d",
        ),
    ]
    out_img = builder._overlay_proposal_labels(
        img, proposals, scene_bounds=(-1.0, -1.0, 1.0, 1.0), highlight_ids=None
    )
    assert out_img.shape == (400, 400, 3)
    # Pixels changed (labels drawn): expect more than 0 modified pixels.
    diff_count = int(np.any(out_img != img, axis=-1).sum())
    assert diff_count > 0


def test_overlay_proposal_labels_highlight_subset_draws_fewer(tmp_path: Path):
    builder = _DummyBuilder(config=SceneBEVConfig(image_size=400))
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    proposals = [
        SceneProposal(proposal_id=1, category="chair", position_3d=(0.5, 0.5, 0.0), source="mask3d"),
        SceneProposal(proposal_id=2, category="chair", position_3d=(0.0, 0.0, 0.0), source="mask3d"),
        SceneProposal(proposal_id=3, category="chair", position_3d=(-0.5, -0.5, 0.0), source="mask3d"),
    ]
    full = builder._overlay_proposal_labels(img.copy(), proposals, (-1, -1, 1, 1), None)
    partial = builder._overlay_proposal_labels(img.copy(), proposals, (-1, -1, 1, 1), [2])
    full_diff = int(np.any(full != img, axis=-1).sum())
    partial_diff = int(np.any(partial != img, axis=-1).sum())
    assert 0 < partial_diff < full_diff


def test_config_hash_is_deterministic():
    a = SceneBEVConfig(image_size=1200, perspective=True)
    b = SceneBEVConfig(image_size=1200, perspective=True)
    builder_a = _DummyBuilder(config=a)
    builder_b = _DummyBuilder(config=b)
    assert builder_a.config_hash() == builder_b.config_hash()


def test_build_with_labels_calls_resolve_paths(tmp_path: Path):
    class FakeBuilder(_DummyBuilder):
        def __init__(self):
            super().__init__(config=SceneBEVConfig(image_size=200))
            self.called = False

        def resolve_paths(self, scene_id, data_root):
            self.called = True
            # Return non-existent paths to trigger an explicit error
            return Path("missing.ply"), Path("missing_traj.txt"), Path("missing_intr.txt")

    builder = FakeBuilder()
    out = tmp_path / "bev.png"
    with pytest.raises(FileNotFoundError):
        builder.build_with_labels(
            scene_id="s",
            data_root=tmp_path,
            proposals=[],
            output_path=out,
        )
    assert builder.called is True
```

- [ ] **Step 2: Run test to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/query_scene/tests/test_scene_bev_builder_base.py -q
```

Expected: `ModuleNotFoundError: No module named 'query_scene.scene_bev_builder'`.

- [ ] **Step 3: Implement base class**

Create `src/query_scene/scene_bev_builder.py`:

```python
"""v9 scene BEV builder: mesh + camera trajectory + per-proposal `#id label`.

Per spec Section B. Each benchmark gets its own subclass that knows how to
discover mesh / trajectory / intrinsic paths for its data layout.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from agents.catalog import SceneProposal


@dataclass(frozen=True)
class SceneBEVConfig:
    image_size: int = 1500
    perspective: bool = True
    camera_fov: float = 100.0
    ceiling_normal_threshold: float = -0.6
    label_color_highlight: tuple[int, int, int] = (255, 64, 64)
    label_color_default: tuple[int, int, int] = (32, 32, 32)
    label_bg_highlight: tuple[int, int, int] = (255, 255, 0)
    label_bg_default: tuple[int, int, int] = (255, 255, 255)
    trajectory_color: tuple[int, int, int] = (32, 96, 220)
    trajectory_thickness: int = 3
    proposal_marker_radius: int = 4


class ScanNetSceneBEVBuilderBase(ABC):
    """Shared mesh + trajectory + per-proposal label overlay for ScanNet-based benchmarks."""

    benchmark: str = "scannet"

    def __init__(self, config: SceneBEVConfig | None = None) -> None:
        self.config = config or SceneBEVConfig()

    @abstractmethod
    def resolve_paths(
        self,
        scene_id: str,
        data_root: Path,
    ) -> tuple[Path, Path, Path]:
        """Return (mesh_path, traj_path, intrinsic_path) for the scene."""
        raise NotImplementedError

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self.config), sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]

    def build_with_labels(
        self,
        *,
        scene_id: str,
        data_root: Path,
        proposals: list[SceneProposal],
        output_path: Path,
        highlight_ids: list[int] | None = None,
    ) -> Path:
        mesh_path, traj_path, intr_path = self.resolve_paths(scene_id, data_root)
        for p in (mesh_path, traj_path, intr_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"required asset missing for scene_id={scene_id}: {p}"
                )

        img, scene_bounds = self._render_mesh_with_traj(mesh_path, traj_path, intr_path)
        img = self._overlay_proposal_labels(
            img, proposals, scene_bounds=scene_bounds, highlight_ids=highlight_ids
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        logger.info(f"[scene_bev] wrote {output_path}")
        return output_path

    def _render_mesh_with_traj(
        self,
        mesh_path: Path,
        traj_path: Path,
        intr_path: Path,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """Render BEV using the same pipeline as the legacy OpenEQA builder.

        Returns (rgb_image_uint8, scene_bounds=(xmin, ymin, xmax, ymax)).
        """
        import open3d as o3d  # heavy import: lazy

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_triangle_normals()
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        tris = np.asarray(mesh.triangles)
        colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
        normals = np.asarray(mesh.triangle_normals)

        K = np.loadtxt(str(intr_path))[:3, :3]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        traj = np.loadtxt(str(traj_path)).reshape(-1, 4, 4)
        cam_positions = traj[:, :3, 3]

        from query_scene.bev_builder import OpenEQAScanNetBEVBuilder

        legacy = OpenEQAScanNetBEVBuilder()
        visible = legacy._compute_frustum_visibility(
            verts, traj, fx, fy, cx, cy, 1296, 968
        )
        tri_visible = visible[tris[:, 0]] & visible[tris[:, 1]] & visible[tris[:, 2]]
        facing_down = normals[:, 2] < self.config.ceiling_normal_threshold
        tri_keep = tri_visible & (~facing_down)
        kept = tris[tri_keep]
        if len(kept) == 0:
            kept = tris[tri_visible]
        img = legacy._render_perspective(verts, colors, kept, cam_positions)
        img = legacy._draw_trajectory(img, verts, cam_positions)
        xs, ys = verts[:, 0], verts[:, 1]
        bounds = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        return img, bounds

    def _overlay_proposal_labels(
        self,
        img: np.ndarray,
        proposals: list[SceneProposal],
        scene_bounds: tuple[float, float, float, float],
        highlight_ids: list[int] | None,
    ) -> np.ndarray:
        h, w = img.shape[:2]
        xmin, ymin, xmax, ymax = scene_bounds
        span_x = max(xmax - xmin, 1e-6)
        span_y = max(ymax - ymin, 1e-6)
        highlight_set: set[int] = set(highlight_ids) if highlight_ids is not None else set()
        for proposal in proposals:
            if highlight_ids is not None and proposal.proposal_id not in highlight_set:
                continue
            x_world, y_world = proposal.position_3d[0], proposal.position_3d[1]
            u = int((x_world - xmin) / span_x * (w - 1))
            v = int((1.0 - (y_world - ymin) / span_y) * (h - 1))
            u = max(0, min(w - 1, u))
            v = max(0, min(h - 1, v))
            highlighted = (
                highlight_ids is not None
                and proposal.proposal_id in highlight_set
            )
            color = (
                self.config.label_color_highlight
                if highlighted
                else self.config.label_color_default
            )
            bg = (
                self.config.label_bg_highlight
                if highlighted
                else self.config.label_bg_default
            )
            cv2.circle(img, (u, v), self.config.proposal_marker_radius, color, -1)
            label = f"#{proposal.proposal_id} {proposal.category}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.45
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
            text_org = (u + 6, v - 6)
            bg_x1 = text_org[0] - 2
            bg_y1 = text_org[1] - th - 2
            bg_x2 = text_org[0] + tw + 2
            bg_y2 = text_org[1] + baseline
            cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), bg, -1)
            cv2.putText(img, label, text_org, font, scale, color, thickness, cv2.LINE_AA)
        return img


__all__ = ["SceneBEVConfig", "ScanNetSceneBEVBuilderBase"]
```

- [ ] **Step 4: Run base test to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/query_scene/tests/test_scene_bev_builder_base.py -q
```

Expected: 4 tests pass. (The `build_with_labels` test only exercises path-validation; rendering itself is exercised in benchmark-specific tests with real mesh fixtures.)

- [ ] **Step 5: Ruff check**

```bash
.venv/bin/ruff check src/query_scene/scene_bev_builder.py src/query_scene/tests/test_scene_bev_builder_base.py
```

Expected: no findings.

- [ ] **Step 6: Commit**

```bash
git add src/query_scene/scene_bev_builder.py src/query_scene/tests/test_scene_bev_builder_base.py
git commit -m "$(cat <<'EOF'
feat(bev): add ScanNetSceneBEVBuilderBase + SceneBEVConfig

Shared mesh+trajectory+label-overlay pipeline for v9 scene BEV. Reuses
OpenEQAScanNetBEVBuilder helpers for frustum visibility, perspective
render, and trajectory overlay. Adds proposal label overlay with
highlight subset support.

Per spec Section B.
EOF
)"
```

### Task 5: Nr3dScanNetBEVBuilder

**Files:**
- Modify: `src/query_scene/scene_bev_builder.py`
- Create: `src/query_scene/tests/test_nr3d_scene_bev_builder.py`

- [ ] **Step 1: Write failing test**

Create `src/query_scene/tests/test_nr3d_scene_bev_builder.py`:

```python
from pathlib import Path

import pytest

from query_scene.scene_bev_builder import Nr3dScanNetBEVBuilder


def _make_nr3d_layout(tmp_path: Path, scene_id: str) -> Path:
    """Create a Phase-8 NR3D scene layout: <scene>/conceptgraph/{traj.txt} + <scene>/raw/intrinsic_color.txt."""
    scene_dir = tmp_path / scene_id
    (scene_dir / "conceptgraph").mkdir(parents=True)
    (scene_dir / "raw").mkdir(parents=True)
    (scene_dir / "conceptgraph" / "traj.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    intr = "577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n"
    (scene_dir / "raw" / "intrinsic_color.txt").write_text(intr)
    return scene_dir


def test_resolve_paths_finds_mesh_traj_intrinsic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scene_dir = _make_nr3d_layout(tmp_path, "scene0000_00")
    fake_mesh = tmp_path / "scannetv2" / "scene0000_00" / "scene0000_00_vh_clean.ply"
    fake_mesh.parent.mkdir(parents=True)
    fake_mesh.write_text("ply\n")
    monkeypatch.setenv("SCANNET_DATA_ROOT", str(tmp_path / "scannetv2"))
    builder = Nr3dScanNetBEVBuilder()
    mesh, traj, intr = builder.resolve_paths("scene0000_00", tmp_path)
    assert mesh == fake_mesh
    assert traj == scene_dir / "conceptgraph" / "traj.txt"
    assert intr == scene_dir / "raw" / "intrinsic_color.txt"


def test_resolve_paths_missing_mesh_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _make_nr3d_layout(tmp_path, "scene0000_00")
    monkeypatch.setenv("SCANNET_DATA_ROOT", str(tmp_path / "scannetv2_empty"))
    builder = Nr3dScanNetBEVBuilder()
    with pytest.raises(FileNotFoundError, match="scannet mesh"):
        builder.resolve_paths("scene0000_00", tmp_path)


def test_benchmark_tag_is_nr3d():
    assert Nr3dScanNetBEVBuilder.benchmark == "nr3d"
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/query_scene/tests/test_nr3d_scene_bev_builder.py -q
```

Expected: `ImportError: cannot import name 'Nr3dScanNetBEVBuilder'`.

- [ ] **Step 3: Implement Nr3dScanNetBEVBuilder**

Append to `src/query_scene/scene_bev_builder.py`:

```python
import os


def _scannet_data_root() -> Path:
    root = os.environ.get("SCANNET_DATA_ROOT", "data/scannetv2")
    return Path(root)


def _find_scannet_mesh(scannet_root: Path, scene_id: str) -> Path:
    for filename in (f"{scene_id}_vh_clean.ply", f"{scene_id}_vh_clean_2.ply"):
        candidate = scannet_root / scene_id / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"scannet mesh for {scene_id} not under {scannet_root}; "
        f"expected {scene_id}_vh_clean.ply"
    )


class Nr3dScanNetBEVBuilder(ScanNetSceneBEVBuilderBase):
    benchmark = "nr3d"

    def resolve_paths(self, scene_id: str, data_root: Path) -> tuple[Path, Path, Path]:
        scene_dir = data_root / scene_id
        traj = scene_dir / "conceptgraph" / "traj.txt"
        intr = scene_dir / "raw" / "intrinsic_color.txt"
        mesh = _find_scannet_mesh(_scannet_data_root(), scene_id)
        return mesh, traj, intr


__all__ += ["Nr3dScanNetBEVBuilder"]
```

- [ ] **Step 4: Run test to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/query_scene/tests/test_nr3d_scene_bev_builder.py -q
```

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/query_scene/scene_bev_builder.py src/query_scene/tests/test_nr3d_scene_bev_builder.py
git commit -m "feat(bev): add Nr3dScanNetBEVBuilder for v9 catalog-first prep"
```

### Task 6: ScanReferScanNetBEVBuilder

**Files:**
- Modify: `src/query_scene/scene_bev_builder.py`
- Create: `src/query_scene/tests/test_scanrefer_scene_bev_builder.py`

- [ ] **Step 1: Write failing test**

Create `src/query_scene/tests/test_scanrefer_scene_bev_builder.py`:

```python
from pathlib import Path

import pytest

from query_scene.scene_bev_builder import ScanReferScanNetBEVBuilder


def _make_scanrefer_layout(tmp_path: Path, scene_id: str) -> Path:
    """ScanRefer pack layout: <scene>/conceptgraph/{traj.txt, intrinsic_color.txt}."""
    scene_dir = tmp_path / scene_id
    cg = scene_dir / "conceptgraph"
    cg.mkdir(parents=True)
    (cg / "traj.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    (cg / "intrinsic_color.txt").write_text("577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n")
    return scene_dir


def test_resolve_paths_uses_conceptgraph_intrinsic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scene_dir = _make_scanrefer_layout(tmp_path, "scene0050_01")
    fake_mesh = tmp_path / "scannetv2" / "scene0050_01" / "scene0050_01_vh_clean_2.ply"
    fake_mesh.parent.mkdir(parents=True)
    fake_mesh.write_text("ply\n")
    monkeypatch.setenv("SCANNET_DATA_ROOT", str(tmp_path / "scannetv2"))
    builder = ScanReferScanNetBEVBuilder()
    mesh, traj, intr = builder.resolve_paths("scene0050_01", tmp_path)
    assert mesh == fake_mesh
    assert traj == scene_dir / "conceptgraph" / "traj.txt"
    assert intr == scene_dir / "conceptgraph" / "intrinsic_color.txt"


def test_benchmark_tag_is_scanrefer():
    assert ScanReferScanNetBEVBuilder.benchmark == "scanrefer"
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/query_scene/tests/test_scanrefer_scene_bev_builder.py -q
```

Expected: `ImportError`.

- [ ] **Step 3: Implement ScanReferScanNetBEVBuilder**

Append to `src/query_scene/scene_bev_builder.py`:

```python
class ScanReferScanNetBEVBuilder(ScanNetSceneBEVBuilderBase):
    benchmark = "scanrefer"

    def resolve_paths(self, scene_id: str, data_root: Path) -> tuple[Path, Path, Path]:
        scene_dir = data_root / scene_id
        traj = scene_dir / "conceptgraph" / "traj.txt"
        # ScanRefer pack puts intrinsic next to traj
        intr = scene_dir / "conceptgraph" / "intrinsic_color.txt"
        mesh = _find_scannet_mesh(_scannet_data_root(), scene_id)
        return mesh, traj, intr


__all__ += ["ScanReferScanNetBEVBuilder"]
```

- [ ] **Step 4: Run test to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/query_scene/tests/test_scanrefer_scene_bev_builder.py -q
```

Expected: all 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/query_scene/scene_bev_builder.py src/query_scene/tests/test_scanrefer_scene_bev_builder.py
git commit -m "feat(bev): add ScanReferScanNetBEVBuilder"
```

### Task 7: OpenEqaScanNetBEVBuilder (subclass of base; preserves legacy semantics)

**Files:**
- Modify: `src/query_scene/scene_bev_builder.py`
- Create: `src/query_scene/tests/test_openeqa_scene_bev_builder.py`

- [ ] **Step 1: Write failing test**

Create `src/query_scene/tests/test_openeqa_scene_bev_builder.py`:

```python
from pathlib import Path

import pytest

from query_scene.scene_bev_builder import OpenEqaScanNetBEVBuilder


def _make_openeqa_layout(tmp_path: Path, clip_id: str, scene_id: str) -> Path:
    """OpenEQA clip layout under data/OpenEQA/scannet/<clip>/conceptgraph + raw."""
    clip_dir = tmp_path / clip_id
    cg = clip_dir / "conceptgraph"
    raw = clip_dir / "raw"
    cg.mkdir(parents=True)
    raw.mkdir(parents=True)
    (cg / "traj.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    (raw / "intrinsic_color.txt").write_text("577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n")
    info = {"scan_id": scene_id, "scene_id": scene_id}
    (cg / "scene_info.json").write_text(__import__("json").dumps(info))
    return clip_dir


def test_resolve_paths_uses_scene_info(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    clip_dir = _make_openeqa_layout(tmp_path, "002-scannet-scene0709_00", "scene0709_00")
    fake_mesh = tmp_path / "scannetv2" / "scene0709_00" / "scene0709_00_vh_clean.ply"
    fake_mesh.parent.mkdir(parents=True)
    fake_mesh.write_text("ply\n")
    monkeypatch.setenv("SCANNET_DATA_ROOT", str(tmp_path / "scannetv2"))
    builder = OpenEqaScanNetBEVBuilder()
    mesh, traj, intr = builder.resolve_paths("002-scannet-scene0709_00", tmp_path)
    assert mesh == fake_mesh
    assert traj == clip_dir / "conceptgraph" / "traj.txt"
    assert intr == clip_dir / "raw" / "intrinsic_color.txt"


def test_resolve_paths_falls_back_to_clip_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When scene_info.json is missing, scan id derives from clip dir name suffix."""
    clip = tmp_path / "041-scannet-scene0011_00"
    (clip / "conceptgraph").mkdir(parents=True)
    (clip / "raw").mkdir(parents=True)
    (clip / "conceptgraph" / "traj.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    (clip / "raw" / "intrinsic_color.txt").write_text("577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n")
    fake_mesh = tmp_path / "scannetv2" / "scene0011_00" / "scene0011_00_vh_clean.ply"
    fake_mesh.parent.mkdir(parents=True)
    fake_mesh.write_text("ply\n")
    monkeypatch.setenv("SCANNET_DATA_ROOT", str(tmp_path / "scannetv2"))
    builder = OpenEqaScanNetBEVBuilder()
    mesh, *_ = builder.resolve_paths("041-scannet-scene0011_00", tmp_path)
    assert mesh == fake_mesh
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/query_scene/tests/test_openeqa_scene_bev_builder.py -q
```

Expected: `ImportError`.

- [ ] **Step 3: Implement OpenEqaScanNetBEVBuilder**

Append to `src/query_scene/scene_bev_builder.py`:

```python
def _extract_scannet_scan_id(clip_name: str, cg_dir: Path) -> str:
    info_path = cg_dir / "scene_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            for key in ("scan_id", "scene_id"):
                value = info.get(key)
                if isinstance(value, str) and value.startswith("scene"):
                    return value
        except json.JSONDecodeError:
            pass
    # Fallback: clip dirs look like "002-scannet-scene0709_00"; take "scene0709_00"
    if "scene" in clip_name:
        return "scene" + clip_name.split("-scene", 1)[-1]
    raise ValueError(
        f"could not extract scannet scan_id from clip name {clip_name!r}"
    )


class OpenEqaScanNetBEVBuilder(ScanNetSceneBEVBuilderBase):
    benchmark = "openeqa"

    def resolve_paths(self, scene_id: str, data_root: Path) -> tuple[Path, Path, Path]:
        clip_dir = data_root / scene_id
        cg = clip_dir / "conceptgraph"
        traj = cg / "traj.txt"
        intr = clip_dir / "raw" / "intrinsic_color.txt"
        if not intr.exists():
            intr = cg / "intrinsic_color.txt"
        scan_id = _extract_scannet_scan_id(scene_id, cg)
        mesh = _find_scannet_mesh(_scannet_data_root(), scan_id)
        return mesh, traj, intr


__all__ += ["OpenEqaScanNetBEVBuilder"]
```

- [ ] **Step 4: Run test to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/query_scene/tests/test_openeqa_scene_bev_builder.py -q
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/query_scene/scene_bev_builder.py src/query_scene/tests/test_openeqa_scene_bev_builder.py
git commit -m "feat(bev): add OpenEqaScanNetBEVBuilder (subclasses base)"
```

### Task 8: Sqa3dScanNetBEVBuilder

**Files:**
- Modify: `src/query_scene/scene_bev_builder.py`
- Create: `src/query_scene/tests/test_sqa3d_scene_bev_builder.py`

- [ ] **Step 1: Write failing test**

Create `src/query_scene/tests/test_sqa3d_scene_bev_builder.py`:

```python
from pathlib import Path

import pytest

from query_scene.scene_bev_builder import Sqa3dScanNetBEVBuilder


def test_resolve_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scene = tmp_path / "scene0050_00"
    (scene / "conceptgraph").mkdir(parents=True)
    (scene / "raw").mkdir(parents=True)
    (scene / "conceptgraph" / "traj.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    (scene / "raw" / "intrinsic_color.txt").write_text(
        "577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n"
    )
    fake_mesh = tmp_path / "scannetv2" / "scene0050_00" / "scene0050_00_vh_clean.ply"
    fake_mesh.parent.mkdir(parents=True)
    fake_mesh.write_text("ply\n")
    monkeypatch.setenv("SCANNET_DATA_ROOT", str(tmp_path / "scannetv2"))
    builder = Sqa3dScanNetBEVBuilder()
    mesh, traj, intr = builder.resolve_paths("scene0050_00", tmp_path)
    assert mesh == fake_mesh
    assert traj == scene / "conceptgraph" / "traj.txt"
    assert intr == scene / "raw" / "intrinsic_color.txt"


def test_benchmark_tag_is_sqa3d():
    assert Sqa3dScanNetBEVBuilder.benchmark == "sqa3d"
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/query_scene/tests/test_sqa3d_scene_bev_builder.py -q
```

Expected: `ImportError`.

- [ ] **Step 3: Implement Sqa3dScanNetBEVBuilder**

Append to `src/query_scene/scene_bev_builder.py`:

```python
class Sqa3dScanNetBEVBuilder(ScanNetSceneBEVBuilderBase):
    benchmark = "sqa3d"

    def resolve_paths(self, scene_id: str, data_root: Path) -> tuple[Path, Path, Path]:
        scene_dir = data_root / scene_id
        traj = scene_dir / "conceptgraph" / "traj.txt"
        intr = scene_dir / "raw" / "intrinsic_color.txt"
        if not intr.exists():
            intr = scene_dir / "conceptgraph" / "intrinsic_color.txt"
        mesh = _find_scannet_mesh(_scannet_data_root(), scene_id)
        return mesh, traj, intr


__all__ += ["Sqa3dScanNetBEVBuilder"]
```

- [ ] **Step 4: Run test to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/query_scene/tests/test_sqa3d_scene_bev_builder.py -q
```

Expected: 2 tests pass.

- [ ] **Step 5: Final ruff sweep over scene_bev_builder**

```bash
.venv/bin/ruff check src/query_scene/scene_bev_builder.py
```

Expected: no findings.

- [ ] **Step 6: Commit**

```bash
git add src/query_scene/scene_bev_builder.py src/query_scene/tests/test_sqa3d_scene_bev_builder.py
git commit -m "feat(bev): add Sqa3dScanNetBEVBuilder; finish 4 per-benchmark builders"
```

---

## Phase 3 — Scene Perception Tools

The four tools in this phase live in two new modules and share a single skill gate (`scene-exploration-playbook`). The runtime exposes the SceneCatalog through `runtime.bundle.extra_metadata["scene_catalog"]` (a dict serialized via `SceneCatalog.model_dump()`); a small helper hydrates it on demand.

### Task 9: runtime helper `get_scene_catalog(runtime)` + tests

**Files:**
- Create: `src/agents/runtime/scene_runtime.py`
- Create: `src/agents/runtime/tests/__init__.py` (if missing)
- Create: `src/agents/runtime/tests/test_scene_runtime.py`

- [ ] **Step 1: Ensure tests dir exists**

```bash
mkdir -p src/agents/runtime/tests
[ -f src/agents/runtime/tests/__init__.py ] || : > src/agents/runtime/tests/__init__.py
```

- [ ] **Step 2: Write failing test**

Create `src/agents/runtime/tests/test_scene_runtime.py`:

```python
import pytest

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.runtime.scene_runtime import (
    get_scene_catalog,
    queue_pending_image,
)


def _catalog() -> SceneCatalog:
    return SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(
                proposal_id=0,
                category="chair",
                position_3d=(0, 0, 0),
                source="mask3d",
                frame_views={10: FrameView(frame_id=10, bbox_2d=(0, 0, 10, 10), raw_rgb_path="r.png")},
            )
        ],
        total_frames=1,
        frame_id_range=(10, 10),
        valid_frame_ids=[10],
        bev_image_path="bev.png",
    )


def test_get_scene_catalog_hydrates_from_extra_metadata():
    bundle = Stage2EvidenceBundle(
        extra_metadata={"scene_catalog": _catalog().model_dump()}
    )
    rs = Stage2RuntimeState(bundle=bundle)
    catalog = get_scene_catalog(rs)
    assert isinstance(catalog, SceneCatalog)
    assert catalog.scene_id == "s"
    assert catalog.proposals[0].proposal_id == 0


def test_get_scene_catalog_caches_on_runtime():
    bundle = Stage2EvidenceBundle(
        extra_metadata={"scene_catalog": _catalog().model_dump()}
    )
    rs = Stage2RuntimeState(bundle=bundle)
    a = get_scene_catalog(rs)
    b = get_scene_catalog(rs)
    assert a is b


def test_get_scene_catalog_missing_errors():
    bundle = Stage2EvidenceBundle(extra_metadata={})
    rs = Stage2RuntimeState(bundle=bundle)
    with pytest.raises(ValueError, match="scene_catalog"):
        get_scene_catalog(rs)


def test_queue_pending_image_appends_and_marks_updated(tmp_path):
    bundle = Stage2EvidenceBundle(extra_metadata={"scene_catalog": _catalog().model_dump()})
    rs = Stage2RuntimeState(bundle=bundle)
    queue_pending_image(rs, str(tmp_path / "x.png"))
    queue_pending_image(rs, str(tmp_path / "y.png"))
    assert rs.bundle.extra_metadata["vg_pending_images"] == [
        str(tmp_path / "x.png"),
        str(tmp_path / "y.png"),
    ]
    assert rs.evidence_updated is True
```

- [ ] **Step 3: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/runtime/tests/test_scene_runtime.py -q
```

Expected: `ModuleNotFoundError: agents.runtime.scene_runtime`.

- [ ] **Step 4: Implement runtime helpers**

Create `src/agents/runtime/scene_runtime.py`:

```python
"""Runtime helpers for v9 catalog-first scene exploration tools.

Hydrates SceneCatalog on demand from bundle.extra_metadata, and exposes
queue_pending_image() so every image-injecting tool uses a single code path.
"""

from __future__ import annotations

from typing import Any

from agents.catalog import SceneCatalog

_RUNTIME_ATTR = "_v9_scene_catalog"


def get_scene_catalog(runtime: Any) -> SceneCatalog:
    """Lazy-load SceneCatalog from bundle.extra_metadata["scene_catalog"].

    Subsequent calls on the same runtime return the cached instance.
    Raises ValueError when the catalog is not present (fail-loud per
    the no-fallback rule).
    """
    cached = getattr(runtime, _RUNTIME_ATTR, None)
    if cached is not None:
        return cached
    extra = getattr(runtime.bundle, "extra_metadata", None) or {}
    raw = extra.get("scene_catalog")
    if raw is None:
        raise ValueError(
            "bundle.extra_metadata.scene_catalog is missing; v9 pack-prep "
            "must write scene_catalog before invoking scene-perception tools"
        )
    catalog = SceneCatalog(**raw)
    setattr(runtime, _RUNTIME_ATTR, catalog)
    return catalog


def queue_pending_image(runtime: Any, image_path: str) -> None:
    """Append an image to bundle.extra_metadata['vg_pending_images'] and mark evidence updated."""
    extra = dict(runtime.bundle.extra_metadata or {})
    pending = list(extra.get("vg_pending_images") or [])
    pending.append(str(image_path))
    extra["vg_pending_images"] = pending
    runtime.bundle.extra_metadata = extra
    runtime.mark_evidence_updated()


__all__ = ["get_scene_catalog", "queue_pending_image"]
```

- [ ] **Step 5: Run test to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/runtime/tests/test_scene_runtime.py -q
```

Expected: 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/agents/runtime/scene_runtime.py src/agents/runtime/tests/__init__.py src/agents/runtime/tests/test_scene_runtime.py
git commit -m "feat(runtime): add scene_runtime helpers (get_scene_catalog, queue_pending_image)"
```

### Task 10: list_scene_proposals + inspect_proposal_v9 (scene-perception text tools)

**Files:**
- Create: `src/agents/tools/scene_perception.py`
- Create: `src/agents/tools/tests/test_list_scene_proposals.py`
- Create: `src/agents/tools/tests/test_inspect_proposal_v9.py`

- [ ] **Step 1: Write failing tests**

Create `src/agents/tools/tests/test_list_scene_proposals.py`:

```python
import json
from pathlib import Path

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.tools.scene_perception import build_scene_perception_tools

PRIMARY_SKILL = "scene-exploration-playbook"


def _runtime() -> Stage2RuntimeState:
    catalog = SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(proposal_id=0, category="chair", position_3d=(0.1, 0.2, 0), source="mask3d"),
            SceneProposal(proposal_id=1, category="chair", position_3d=(1.5, 0.8, 0), source="mask3d"),
            SceneProposal(proposal_id=2, category="table", position_3d=(3.0, 4.0, 0), source="mask3d"),
            SceneProposal(proposal_id=3, category="lamp", position_3d=(-2.0, -1.0, 0), source="mask3d"),
        ],
        total_frames=10,
        frame_id_range=(0, 90),
        valid_frame_ids=[0, 10, 20, 30],
        bev_image_path="bev.png",
    )
    bundle = Stage2EvidenceBundle(extra_metadata={"scene_catalog": catalog.model_dump()})
    rs = Stage2RuntimeState(bundle=bundle)
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    return rs


def test_list_scene_proposals_gates_on_skill():
    rs = _runtime()
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "list_scene_proposals")
    resp = tool.invoke({})
    assert resp.startswith("ERROR")
    assert PRIMARY_SKILL in resp


def test_list_scene_proposals_no_filter_returns_all():
    rs = _runtime()
    rs.skills_loaded.add(PRIMARY_SKILL)
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "list_scene_proposals")
    payload = json.loads(tool.invoke({}))
    assert payload["count"] == 4
    assert {p["proposal_id"] for p in payload["proposals"]} == {0, 1, 2, 3}


def test_list_scene_proposals_filters_by_category():
    rs = _runtime()
    rs.skills_loaded.add(PRIMARY_SKILL)
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "list_scene_proposals")
    payload = json.loads(tool.invoke({"category": "chair"}))
    assert payload["count"] == 2
    assert {p["proposal_id"] for p in payload["proposals"]} == {0, 1}


def test_list_scene_proposals_filters_by_region_bev():
    rs = _runtime()
    rs.skills_loaded.add(PRIMARY_SKILL)
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "list_scene_proposals")
    payload = json.loads(tool.invoke({"region_bev": [0, 0, 2, 2]}))
    assert {p["proposal_id"] for p in payload["proposals"]} == {0, 1}


def test_list_scene_proposals_respects_limit():
    rs = _runtime()
    rs.skills_loaded.add(PRIMARY_SKILL)
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "list_scene_proposals")
    payload = json.loads(tool.invoke({"limit": 2}))
    assert payload["count"] == 2


def test_list_scene_proposals_records_trace():
    rs = _runtime()
    rs.skills_loaded.add(PRIMARY_SKILL)
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "list_scene_proposals")
    tool.invoke({"category": "chair"})
    names = [obs.tool_name for obs in rs.tool_trace]
    assert "list_scene_proposals" in names
```

Create `src/agents/tools/tests/test_inspect_proposal_v9.py`:

```python
import json

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.tools.scene_perception import build_scene_perception_tools

PRIMARY_SKILL = "scene-exploration-playbook"


def _runtime() -> Stage2RuntimeState:
    catalog = SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(
                proposal_id=7,
                category="sofa",
                position_3d=(1.0, 1.0, 0.3),
                bbox_3d_9dof=(1.0, 1.0, 0.3, 1.5, 0.8, 0.5, 0, 0, 0),
                frame_views={
                    20: FrameView(frame_id=20, bbox_2d=(0, 0, 10, 10), raw_rgb_path="a.png"),
                    25: FrameView(frame_id=25, bbox_2d=(5, 5, 20, 20), raw_rgb_path="b.png"),
                },
                source="vdetr",
            )
        ],
        total_frames=10,
        frame_id_range=(0, 90),
        valid_frame_ids=[0, 20, 25],
        bev_image_path="bev.png",
    )
    bundle = Stage2EvidenceBundle(extra_metadata={"scene_catalog": catalog.model_dump()})
    rs = Stage2RuntimeState(bundle=bundle)
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    return rs


def test_inspect_proposal_returns_expected_fields():
    rs = _runtime()
    rs.skills_loaded.add(PRIMARY_SKILL)
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "inspect_proposal")
    payload = json.loads(tool.invoke({"proposal_id": 7}))
    assert payload["proposal_id"] == 7
    assert payload["category"] == "sofa"
    assert payload["position_3d"] == [1.0, 1.0, 0.3]
    assert payload["bbox_3d_9dof"][:3] == [1.0, 1.0, 0.3]
    assert sorted(payload["frames_appeared"]) == [20, 25]
    assert payload["source"] == "vdetr"


def test_inspect_proposal_unknown_id_errors():
    rs = _runtime()
    rs.skills_loaded.add(PRIMARY_SKILL)
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "inspect_proposal")
    resp = tool.invoke({"proposal_id": 999})
    assert resp.startswith("ERROR")
    assert "999" in resp
```

- [ ] **Step 2: Run tests to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_list_scene_proposals.py src/agents/tools/tests/test_inspect_proposal_v9.py -q
```

Expected: `ModuleNotFoundError: agents.tools.scene_perception`.

- [ ] **Step 3: Ensure tests dir exists**

```bash
mkdir -p src/agents/tools/tests
[ -f src/agents/tools/tests/__init__.py ] || : > src/agents/tools/tests/__init__.py
```

- [ ] **Step 4: Implement scene_perception tool module (list + inspect; view_bev added in Task 11)**

Create `src/agents/tools/scene_perception.py`:

```python
"""v9 scene-perception tools: view_bev / list_scene_proposals / inspect_proposal.

All tools share the `scene-exploration-playbook` skill gate.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool

from agents.runtime.scene_runtime import get_scene_catalog, queue_pending_image

SCENE_EXPLORATION_SKILL = "scene-exploration-playbook"


def _gate(runtime: Any) -> str | None:
    if SCENE_EXPLORATION_SKILL not in runtime.skills_loaded:
        return f"ERROR: load_skill({SCENE_EXPLORATION_SKILL!r}) before calling this tool."
    return None


def _in_bev_box(position_3d: tuple[float, float, float], box: list[float]) -> bool:
    xmin, ymin, xmax, ymax = box
    return xmin <= position_3d[0] <= xmax and ymin <= position_3d[1] <= ymax


def build_scene_perception_tools(runtime: Any) -> list[BaseTool]:
    @tool
    def list_scene_proposals(
        category: str | None = None,
        region_bev: list[float] | None = None,
        limit: int | None = None,
    ) -> str:
        """Scene-scoped proposal list. Detailed usage in 'scene-exploration-playbook'."""
        request = {"category": category, "region_bev": region_bev, "limit": limit}
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("list_scene_proposals", request, gate)
            return gate
        catalog = get_scene_catalog(runtime)
        proposals = list(catalog.proposals)
        if category is not None:
            cat_norm = str(category).strip().lower()
            proposals = [p for p in proposals if p.category.strip().lower() == cat_norm]
        if region_bev is not None:
            if len(region_bev) != 4:
                err = "ERROR: region_bev must be [xmin, ymin, xmax, ymax]"
                runtime.record("list_scene_proposals", request, err)
                return err
            proposals = [p for p in proposals if _in_bev_box(p.position_3d, list(region_bev))]
        if limit is not None and limit >= 0:
            proposals = proposals[:limit]
        payload = {
            "count": len(proposals),
            "proposals": [
                {
                    "proposal_id": p.proposal_id,
                    "category": p.category,
                    "position_3d": list(p.position_3d),
                    "frame_count": len(p.frame_views),
                }
                for p in proposals
            ],
        }
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("list_scene_proposals", request, text)
        return text

    @tool
    def inspect_proposal(proposal_id: int) -> str:
        """Return position, 9dof bbox, source, and frames-appeared for one proposal."""
        request = {"proposal_id": int(proposal_id)}
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("inspect_proposal", request, gate)
            return gate
        catalog = get_scene_catalog(runtime)
        proposal = catalog.proposal_by_id(int(proposal_id))
        if proposal is None:
            err = (
                f"ERROR: proposal_id={proposal_id} not in catalog; "
                f"available count={len(catalog.proposals)}"
            )
            runtime.record("inspect_proposal", request, err)
            return err
        payload = {
            "proposal_id": proposal.proposal_id,
            "category": proposal.category,
            "position_3d": list(proposal.position_3d),
            "bbox_3d_9dof": list(proposal.bbox_3d_9dof) if proposal.bbox_3d_9dof else None,
            "frames_appeared": sorted(proposal.frame_views.keys()),
            "source": proposal.source,
        }
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("inspect_proposal", request, text)
        return text

    return [list_scene_proposals, inspect_proposal]


__all__ = [
    "SCENE_EXPLORATION_SKILL",
    "build_scene_perception_tools",
]
```

- [ ] **Step 5: Run tests to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_list_scene_proposals.py src/agents/tools/tests/test_inspect_proposal_v9.py -q
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/agents/tools/scene_perception.py src/agents/tools/tests/__init__.py src/agents/tools/tests/test_list_scene_proposals.py src/agents/tools/tests/test_inspect_proposal_v9.py
git commit -m "feat(tools): add list_scene_proposals and v9 inspect_proposal"
```

### Task 11: view_bev tool

**Files:**
- Modify: `src/agents/tools/scene_perception.py`
- Create: `src/agents/tools/tests/test_view_bev.py`

- [ ] **Step 1: Write failing test**

Create `src/agents/tools/tests/test_view_bev.py`:

```python
from pathlib import Path

import pytest

from agents.catalog import SceneCatalog, SceneProposal
from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.tools.scene_perception import build_scene_perception_tools

PRIMARY_SKILL = "scene-exploration-playbook"


def _runtime(tmp_path: Path) -> Stage2RuntimeState:
    bev = tmp_path / "bev.png"
    from PIL import Image

    Image.new("RGB", (100, 80), (220, 220, 220)).save(bev)
    catalog = SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(proposal_id=0, category="chair", position_3d=(0, 0, 0), source="mask3d"),
            SceneProposal(proposal_id=1, category="table", position_3d=(1, 1, 0), source="mask3d"),
        ],
        total_frames=1,
        frame_id_range=(0, 0),
        valid_frame_ids=[0],
        bev_image_path=str(bev),
    )
    bundle = Stage2EvidenceBundle(extra_metadata={"scene_catalog": catalog.model_dump()})
    rs = Stage2RuntimeState(bundle=bundle)
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.skills_loaded.add(PRIMARY_SKILL)
    return rs


def test_view_bev_default_returns_catalog_path(tmp_path: Path):
    rs = _runtime(tmp_path)
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "view_bev")
    response = tool.invoke({})
    assert "bev image" in response.lower()
    assert (tmp_path / "bev.png").as_posix() in response.replace("\\", "/")
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert pending == [str(tmp_path / "bev.png")]
    assert rs.evidence_updated is True


def test_view_bev_highlight_renders_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rs = _runtime(tmp_path)
    captured: dict = {}

    def _fake_render(catalog, highlight_ids, output_path):
        captured["highlight_ids"] = list(highlight_ids or [])
        from PIL import Image

        Image.new("RGB", (40, 40), (255, 0, 0)).save(output_path)
        return output_path

    monkeypatch.setattr(
        "agents.tools.scene_perception._render_highlighted_bev",
        _fake_render,
    )
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "view_bev")
    resp = tool.invoke({"highlight": [1]})
    assert captured["highlight_ids"] == [1]
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert pending and pending[-1].endswith(".png")
    assert "highlight=[1]" in resp


def test_view_bev_gates_on_skill(tmp_path: Path):
    rs = _runtime(tmp_path)
    rs.skills_loaded.discard(PRIMARY_SKILL)
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "view_bev")
    resp = tool.invoke({})
    assert resp.startswith("ERROR")
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_view_bev.py -q
```

Expected: tests fail (`view_bev` tool not registered, helper not defined).

- [ ] **Step 3: Extend `scene_perception.py` with view_bev + highlighted-render helper**

Open `src/agents/tools/scene_perception.py` and add (above `build_scene_perception_tools`):

```python
from pathlib import Path


def _render_highlighted_bev(catalog, highlight_ids: list[int], output_path: Path) -> Path:
    """Render a focused BEV. Uses the legacy unmodified BEV as a backdrop and
    overlays only the highlighted proposal labels via the v9 BEV builder.

    Implemented as a thin function so tests can monkeypatch it without touching
    open3d / cv2 / mesh assets.
    """
    import cv2

    from query_scene.scene_bev_builder import (
        ScanNetSceneBEVBuilderBase,
        SceneBEVConfig,
    )

    class _BackdropBuilder(ScanNetSceneBEVBuilderBase):
        benchmark = "backdrop"

        def resolve_paths(self, scene_id, data_root):
            raise NotImplementedError

    base = cv2.imread(str(catalog.bev_image_path))
    if base is None:
        raise FileNotFoundError(
            f"backing BEV image not readable: {catalog.bev_image_path}"
        )
    img = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
    builder = _BackdropBuilder(config=SceneBEVConfig(image_size=img.shape[1]))
    h, w = img.shape[:2]
    # Approximate scene bounds from proposal positions to map labels back onto img
    xs = [p.position_3d[0] for p in catalog.proposals] or [0.0, 1.0]
    ys = [p.position_3d[1] for p in catalog.proposals] or [0.0, 1.0]
    scene_bounds = (min(xs), min(ys), max(xs), max(ys))
    out = builder._overlay_proposal_labels(
        img, catalog.proposals, scene_bounds, list(highlight_ids)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return output_path
```

Then inside `build_scene_perception_tools(runtime)` (before `return [...]`), add:

```python
    @tool
    def view_bev(highlight: list[int] | None = None) -> str:
        """Inject the scene BEV image. Detailed usage in 'scene-exploration-playbook'."""
        request = {"highlight": list(highlight) if highlight else None}
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("view_bev", request, gate)
            return gate
        catalog = get_scene_catalog(runtime)
        if highlight is None or not highlight:
            queue_pending_image(runtime, catalog.bev_image_path)
            text = f"bev image at {catalog.bev_image_path}; highlight=ALL ({len(catalog.proposals)} proposals)"
            runtime.record("view_bev", request, text)
            return text
        cache_dir = Path(catalog.bev_image_path).parent / "highlights"
        ids = "_".join(str(int(i)) for i in highlight)
        out_path = cache_dir / f"bev_h_{ids}.png"
        if not out_path.exists():
            _render_highlighted_bev(catalog, list(highlight), out_path)
        queue_pending_image(runtime, str(out_path))
        text = f"bev image at {out_path}; highlight={list(highlight)}"
        runtime.record("view_bev", request, text)
        return text
```

Finally update `return [...]`:

```python
    return [view_bev, list_scene_proposals, inspect_proposal]
```

- [ ] **Step 4: Run tests to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_view_bev.py -q
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agents/tools/scene_perception.py src/agents/tools/tests/test_view_bev.py
git commit -m "feat(tools): add view_bev with highlight subset re-render"
```

### Task 12: unified view_keyframe(mode='rgb'|'marked'|'auto', categories?, proposal_ids?)

**Files:**
- Create: `src/agents/tools/view_keyframe.py`
- Create: `src/agents/tools/tests/test_view_keyframe_v9.py`

The new `view_keyframe` consolidates `view_keyframe_marked` (already landed on the parent branch via commit `a5f3625`) with a new RGB mode for QA. It must preserve selective-filtering behavior (categories / proposal_ids) and the `filtered_marks/` cache path established by `_render_filtered_marked_frame` in `src/agents/packs/vg_embodiedscan/tools.py`.

- [ ] **Step 1: Write failing test**

Create `src/agents/tools/tests/test_view_keyframe_v9.py`:

```python
import json
from pathlib import Path

import pytest
from PIL import Image

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.tools.view_keyframe import build_view_keyframe_tool

PRIMARY_SKILL = "scene-exploration-playbook"


def _runtime(tmp_path: Path, task_type: Stage2TaskType) -> Stage2RuntimeState:
    raw = tmp_path / "raw10.png"
    Image.new("RGB", (200, 200), (220, 220, 220)).save(raw)
    bev = tmp_path / "bev.png"
    Image.new("RGB", (100, 100), (10, 10, 10)).save(bev)
    catalog = SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(
                proposal_id=0,
                category="chair",
                position_3d=(0, 0, 0),
                source="mask3d",
                frame_views={10: FrameView(frame_id=10, bbox_2d=(10, 20, 60, 80), raw_rgb_path=str(raw))},
            ),
            SceneProposal(
                proposal_id=1,
                category="table",
                position_3d=(1, 1, 0),
                source="mask3d",
                frame_views={10: FrameView(frame_id=10, bbox_2d=(100, 30, 180, 90), raw_rgb_path=str(raw))},
            ),
        ],
        total_frames=1,
        frame_id_range=(10, 10),
        valid_frame_ids=[10],
        bev_image_path=str(bev),
    )
    bundle = Stage2EvidenceBundle(extra_metadata={"scene_catalog": catalog.model_dump()})
    rs = Stage2RuntimeState(bundle=bundle)
    rs.task_type = task_type
    rs.skills_loaded.add(PRIMARY_SKILL)
    # Provide annotated dir path (used for unfiltered marked mode)
    annotated = tmp_path / "annotated"
    annotated.mkdir()
    Image.new("RGB", (200, 200), (200, 200, 200)).save(annotated / "frame_10.png")
    rs.bundle.extra_metadata = dict(rs.bundle.extra_metadata)
    rs.bundle.extra_metadata["annotated_image_dir"] = str(annotated)
    return rs


def test_view_keyframe_auto_vg_defaults_to_marked(tmp_path: Path):
    rs = _runtime(tmp_path, Stage2TaskType.VISUAL_GROUNDING)
    tool = build_view_keyframe_tool(rs)
    resp = tool.invoke({"frame_id": 10})
    assert "marked image" in resp
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert pending[-1].endswith("frame_10.png")


def test_view_keyframe_auto_qa_defaults_to_rgb(tmp_path: Path):
    rs = _runtime(tmp_path, Stage2TaskType.QA)
    tool = build_view_keyframe_tool(rs)
    resp = tool.invoke({"frame_id": 10})
    assert "rgb image" in resp.lower()
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert pending[-1] == str(tmp_path / "raw10.png")


def test_view_keyframe_explicit_mode_rgb(tmp_path: Path):
    rs = _runtime(tmp_path, Stage2TaskType.VISUAL_GROUNDING)
    tool = build_view_keyframe_tool(rs)
    resp = tool.invoke({"frame_id": 10, "mode": "rgb"})
    assert "rgb image" in resp.lower()
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert pending[-1] == str(tmp_path / "raw10.png")


def test_view_keyframe_explicit_mode_marked_categories_filter(tmp_path: Path):
    rs = _runtime(tmp_path, Stage2TaskType.QA)
    tool = build_view_keyframe_tool(rs)
    resp = tool.invoke({"frame_id": 10, "mode": "marked", "categories": ["chair"]})
    assert "filtered marked image" in resp
    assert "visible_proposals=[0]" in resp


def test_view_keyframe_explicit_mode_marked_proposal_ids_filter(tmp_path: Path):
    rs = _runtime(tmp_path, Stage2TaskType.VISUAL_GROUNDING)
    tool = build_view_keyframe_tool(rs)
    resp = tool.invoke({"frame_id": 10, "mode": "marked", "proposal_ids": [1]})
    assert "filtered marked image" in resp
    assert "visible_proposals=[1]" in resp


def test_view_keyframe_marked_filter_no_match_errors(tmp_path: Path):
    rs = _runtime(tmp_path, Stage2TaskType.VISUAL_GROUNDING)
    tool = build_view_keyframe_tool(rs)
    resp = tool.invoke({"frame_id": 10, "mode": "marked", "categories": ["lamp"]})
    assert resp.startswith("ERROR")
    assert "no visible proposals matched filters" in resp


def test_view_keyframe_unknown_frame_errors(tmp_path: Path):
    rs = _runtime(tmp_path, Stage2TaskType.QA)
    tool = build_view_keyframe_tool(rs)
    resp = tool.invoke({"frame_id": 999})
    assert resp.startswith("ERROR")
    assert "frame_id=999" in resp


def test_view_keyframe_mode_rgb_ignores_filters(tmp_path: Path):
    """mode='rgb' is raw rgb; categories/proposal_ids are ignored (silent)."""
    rs = _runtime(tmp_path, Stage2TaskType.QA)
    tool = build_view_keyframe_tool(rs)
    resp = tool.invoke({"frame_id": 10, "mode": "rgb", "categories": ["chair"]})
    assert "rgb image" in resp.lower()
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert pending[-1] == str(tmp_path / "raw10.png")


def test_view_keyframe_gates_on_skill(tmp_path: Path):
    rs = _runtime(tmp_path, Stage2TaskType.VISUAL_GROUNDING)
    rs.skills_loaded.discard(PRIMARY_SKILL)
    tool = build_view_keyframe_tool(rs)
    resp = tool.invoke({"frame_id": 10})
    assert resp.startswith("ERROR")
    assert PRIMARY_SKILL in resp
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_view_keyframe_v9.py -q
```

Expected: `ModuleNotFoundError: agents.tools.view_keyframe`.

- [ ] **Step 3: Implement view_keyframe**

Create `src/agents/tools/view_keyframe.py`:

```python
"""Unified view_keyframe tool (v9): mode='rgb' | 'marked' | 'auto', with selective filtering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import BaseTool, tool

from agents.catalog import SceneProposal
from agents.core.agent_config import Stage2TaskType
from agents.runtime.scene_runtime import get_scene_catalog, queue_pending_image
from agents.tools.scene_perception import SCENE_EXPLORATION_SKILL, _gate


def _norm_category(category: str) -> str:
    return " ".join(str(category).strip().lower().split())


def _coerce_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int) and not isinstance(value, bool):
        return [int(value)]
    if isinstance(value, str):
        out: list[int] = []
        for chunk in value.replace(",", " ").split():
            try:
                out.append(int(chunk))
            except ValueError:
                continue
        return out
    if isinstance(value, (list, tuple, set)):
        out: list[int] = []
        for item in value:
            out.extend(_coerce_int_list(item))
        return out
    return []


def _coerce_category_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item)
        return out
    return []


def _resolve_mode(mode: str, task_type: Stage2TaskType | None) -> Literal["rgb", "marked"]:
    if mode == "rgb":
        return "rgb"
    if mode == "marked":
        return "marked"
    if mode == "auto":
        if task_type == Stage2TaskType.VISUAL_GROUNDING:
            return "marked"
        return "rgb"
    raise ValueError(f"mode must be 'rgb' | 'marked' | 'auto'; got {mode!r}")


def _filter_visible(
    proposals: list[SceneProposal],
    frame_id: int,
    categories: list[str],
    proposal_ids: list[int],
) -> list[SceneProposal]:
    if not categories and not proposal_ids:
        return [p for p in proposals if frame_id in p.frame_views]
    wanted_cat = {_norm_category(c) for c in categories}
    wanted_ids = {int(i) for i in proposal_ids}
    out: list[SceneProposal] = []
    for p in proposals:
        if frame_id not in p.frame_views:
            continue
        cat_match = bool(wanted_cat) and _norm_category(p.category) in wanted_cat
        id_match = p.proposal_id in wanted_ids
        if cat_match or id_match:
            out.append(p)
    return out


def _render_filtered_marked(
    proposals: list[SceneProposal],
    frame_id: int,
    cache_dir: Path,
) -> Path:
    from PIL import Image, ImageDraw

    first = next((p for p in proposals if frame_id in p.frame_views), None)
    if first is None:
        raise ValueError(f"no 2D geometry for frame_id={frame_id}")
    raw_path = Path(first.frame_views[frame_id].raw_rgb_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"raw RGB image not found: {raw_path}")
    img = Image.open(raw_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    palette = [(34, 197, 94), (239, 68, 68), (59, 130, 246), (234, 179, 8), (168, 85, 247)]
    for idx, p in enumerate(proposals):
        view = p.frame_views[frame_id]
        x1, y1, x2, y2 = view.bbox_2d
        color = palette[idx % len(palette)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(3, img.width // 320))
        label = f"#{p.proposal_id} {p.category}"
        try:
            tw, th = draw.textbbox((0, 0), label)[2:]
        except AttributeError:
            tw, th = draw.textsize(label)
        draw.rectangle((x1, max(0, y1 - th - 4), x1 + tw + 4, y1), fill=(0, 0, 0))
        draw.text((x1 + 2, max(0, y1 - th - 2)), label, fill=(255, 255, 255))
    cache_dir.mkdir(parents=True, exist_ok=True)
    ids = "_".join(str(p.proposal_id) for p in proposals)
    out = cache_dir / f"frame_{frame_id}_ids_{ids or 'all'}.png"
    img.save(out, format="PNG")
    return out


def _left_to_right_entries(visible: list[SceneProposal], frame_id: int) -> list[str]:
    rows: list[tuple[float, int, str]] = []
    for p in visible:
        view = p.frame_views[frame_id]
        x1, _, x2, _ = view.bbox_2d
        cx = (float(x1) + float(x2)) / 2.0
        rows.append((cx, p.proposal_id, p.category))
    rows.sort(key=lambda item: item[0])
    return [f"#{pid} {cat}" for _, pid, cat in rows]


def build_view_keyframe_tool(runtime: Any) -> BaseTool:
    @tool
    def view_keyframe(
        frame_id: int,
        mode: str = "auto",
        categories: list[str] | str | None = None,
        proposal_ids: list[int] | int | str | None = None,
    ) -> str:
        """Inject one first-person frame. mode='auto' uses task_type to pick rgb (QA) or marked (VG)."""
        category_filter = _coerce_category_list(categories)
        id_filter = _coerce_int_list(proposal_ids)
        request = {
            "frame_id": int(frame_id),
            "mode": mode,
            "categories": category_filter,
            "proposal_ids": id_filter,
        }
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("view_keyframe", request, gate)
            return gate
        try:
            resolved_mode = _resolve_mode(mode, runtime.task_type)
        except ValueError as exc:
            err = f"ERROR: {exc}"
            runtime.record("view_keyframe", request, err)
            return err

        catalog = get_scene_catalog(runtime)
        proposals_at_frame = [p for p in catalog.proposals if int(frame_id) in p.frame_views]
        if not proposals_at_frame and resolved_mode == "marked":
            err = (
                f"ERROR: frame_id={frame_id} has no visible proposals in scene_catalog; "
                f"available frames: {sorted({fid for p in catalog.proposals for fid in p.frame_views})[:20]}"
            )
            runtime.record("view_keyframe", request, err)
            return err

        if resolved_mode == "rgb":
            raw_path = _resolve_raw_rgb_path(catalog, int(frame_id))
            if raw_path is None:
                err = (
                    f"ERROR: frame_id={frame_id} has no raw RGB reference; "
                    f"valid_frame_ids[:20]={catalog.valid_frame_ids[:20]}"
                )
                runtime.record("view_keyframe", request, err)
                return err
            queue_pending_image(runtime, str(raw_path))
            body = f"frame_id={frame_id} rgb image at {raw_path}"
            runtime.record("view_keyframe", request, body)
            return body

        # mode == 'marked'
        visible = _filter_visible(proposals_at_frame, int(frame_id), category_filter, id_filter)
        if (category_filter or id_filter) and not visible:
            err = (
                f"ERROR: no visible proposals matched filters for frame_id={frame_id}; "
                f"filtered_by={{'categories': {category_filter}, 'proposal_ids': {id_filter}}}; "
                f"visible_proposals={[p.proposal_id for p in proposals_at_frame]}"
            )
            runtime.record("view_keyframe", request, err)
            return err
        if not visible:
            visible = proposals_at_frame
        annotated_dir = runtime.bundle.extra_metadata.get("annotated_image_dir") if runtime.bundle.extra_metadata else None
        has_filters = bool(category_filter or id_filter)
        if not has_filters and annotated_dir:
            marked_path = Path(annotated_dir) / f"frame_{int(frame_id)}.png"
            if not marked_path.exists():
                err = f"ERROR: annotated image not found: {marked_path}"
                runtime.record("view_keyframe", request, err)
                return err
            prefix = "marked image"
        else:
            cache_dir = Path(catalog.bev_image_path).parent / "filtered_marks"
            try:
                marked_path = _render_filtered_marked(visible, int(frame_id), cache_dir)
            except Exception as exc:  # noqa: BLE001 — fail-loud with explicit type
                err = f"ERROR: marked image render failed for frame_id={frame_id}: {type(exc).__name__}: {exc}"
                runtime.record("view_keyframe", request, err)
                return err
            prefix = "filtered marked image" if has_filters else "marked image"
        queue_pending_image(runtime, str(marked_path))
        ltr = _left_to_right_entries(visible, int(frame_id))
        boxes = {p.proposal_id: list(p.frame_views[int(frame_id)].bbox_2d) for p in visible}
        cat_list = [p.category for p in visible]
        filter_text = ""
        if has_filters:
            filter_text = (
                f" filtered_by={{'categories': {category_filter}, 'proposal_ids': {id_filter}}};"
            )
        body = (
            f"frame_id={frame_id} {prefix} at {marked_path};"
            f"{filter_text} "
            f"visible_proposals={[p.proposal_id for p in visible]}; "
            f"categories={cat_list}; "
            f"left_to_right={ltr}; "
            f"boxes_2d={boxes}"
        )
        runtime.record("view_keyframe", request, body)
        return body

    return view_keyframe


def _resolve_raw_rgb_path(catalog, frame_id: int) -> Path | None:
    for p in catalog.proposals:
        view = p.frame_views.get(int(frame_id))
        if view is not None and view.raw_rgb_path:
            return Path(view.raw_rgb_path)
    return None


__all__ = ["build_view_keyframe_tool"]
```

- [ ] **Step 4: Run test to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_view_keyframe_v9.py -q
```

Expected: 9 tests pass.

- [ ] **Step 5: Ruff sweep**

```bash
.venv/bin/ruff check src/agents/tools/scene_perception.py src/agents/tools/view_keyframe.py src/agents/tools/tests
```

Expected: no findings (`# noqa: BLE001` is intentional for the marked-render fail-loud path).

- [ ] **Step 6: Commit**

```bash
git add src/agents/tools/view_keyframe.py src/agents/tools/tests/test_view_keyframe_v9.py
git commit -m "$(cat <<'EOF'
feat(tools): add unified view_keyframe(mode='rgb'|'marked'|'auto')

Merges legacy view_keyframe_marked + selective filtering with a new RGB
mode for QA. auto-mode resolves via runtime.task_type (VG=marked,
QA=rgb). Filtered marked renders cache under <scene>/filtered_marks/.

Per spec Section C.
EOF
)"
```

---

## Phase 4 — Selectors A-F

All six selectors live in `src/agents/tools/selectors.py` and return **text-only JSON** (Path A from the spec). Every selector response has the shape:

```json
{
  "hypothesis_summary": "...",
  "frames": [
    {
      "frame_id": int,
      "visible_proposal_ids": [int, ...],
      "bev_xy": [float, float] | null,
      "camera_yaw": float | null,
      "selected_because": str
    }
  ]
}
```

Camera xy/yaw come from `bundle.extra_metadata["camera_trajectory_xy_yaw"]` (a dict `{frame_id: [x, y, yaw]}`), populated by pack-prep in Phase 5. When missing, fields are returned as null and selectors still work for catalog-only modalities.

All selectors share the `scene-exploration-playbook` skill gate via `_gate` from `agents.tools.scene_perception`.

### Task 13: Selector A — `select_by_text`

**Files:**
- Create: `src/agents/tools/selectors.py`
- Create: `src/agents/tools/tests/test_selectors_text.py`
- Create: `src/agents/tools/tests/_selector_fixtures.py`

- [ ] **Step 1: Write shared selector fixture**

Create `src/agents/tools/tests/_selector_fixtures.py`:

```python
"""Shared test fixtures for the six v9 selectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState

PRIMARY_SKILL = "scene-exploration-playbook"


def make_catalog() -> SceneCatalog:
    return SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(
                proposal_id=0,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                source="mask3d",
                frame_views={
                    10: FrameView(frame_id=10, bbox_2d=(0, 0, 10, 10), raw_rgb_path="r.png"),
                    20: FrameView(frame_id=20, bbox_2d=(0, 0, 10, 10), raw_rgb_path="r.png"),
                },
            ),
            SceneProposal(
                proposal_id=1,
                category="chair",
                position_3d=(2.0, 0.0, 0.0),
                source="mask3d",
                frame_views={
                    30: FrameView(frame_id=30, bbox_2d=(0, 0, 10, 10), raw_rgb_path="r.png"),
                },
            ),
            SceneProposal(
                proposal_id=2,
                category="table",
                position_3d=(1.0, 1.0, 0.0),
                source="mask3d",
                frame_views={
                    10: FrameView(frame_id=10, bbox_2d=(20, 20, 60, 60), raw_rgb_path="r.png"),
                    40: FrameView(frame_id=40, bbox_2d=(0, 0, 10, 10), raw_rgb_path="r.png"),
                },
            ),
            SceneProposal(
                proposal_id=3,
                category="lamp",
                position_3d=(-3.0, -3.0, 0.0),
                source="mask3d",
                frame_views={
                    50: FrameView(frame_id=50, bbox_2d=(0, 0, 10, 10), raw_rgb_path="r.png"),
                },
            ),
        ],
        total_frames=5,
        frame_id_range=(10, 50),
        valid_frame_ids=[10, 20, 30, 40, 50],
        bev_image_path="bev.png",
    )


def make_runtime(extra: dict | None = None) -> Stage2RuntimeState:
    catalog = make_catalog()
    base_meta: dict[str, Any] = {"scene_catalog": catalog.model_dump()}
    base_meta["camera_trajectory_xy_yaw"] = {
        10: [0.0, 0.0, 0.0],
        20: [0.5, 0.0, 0.5],
        30: [1.0, 0.0, 1.0],
        40: [1.5, 0.5, 0.0],
        50: [-2.0, -2.0, 3.0],
    }
    if extra:
        base_meta.update(extra)
    bundle = Stage2EvidenceBundle(extra_metadata=base_meta)
    rs = Stage2RuntimeState(bundle=bundle)
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.skills_loaded.add(PRIMARY_SKILL)
    return rs
```

- [ ] **Step 2: Write failing test for select_by_text**

Create `src/agents/tools/tests/test_selectors_text.py`:

```python
import json

from agents.tools.selectors import build_selector_tools
from agents.tools.tests._selector_fixtures import PRIMARY_SKILL, make_runtime


class _FakeKeyframeSelector:
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def select_keyframes_v2(self, query, k=3, **kwargs):
        self.calls.append((query, k))
        from query_scene.keyframe_selector import KeyframeResult

        return KeyframeResult(
            query=query,
            target_term="chair",
            anchor_term=None,
            keyframe_indices=[10, 20, 30],
            keyframe_paths=[],
            target_objects=[],
            anchor_objects=[],
            metadata={"hypothesis_output": {"hypotheses": [{"grounding_query": {"root": {"category": "chair"}}}]}},
        )


def _attach_selector(runtime):
    runtime.keyframe_selector = _FakeKeyframeSelector()
    return runtime.keyframe_selector


def test_select_by_text_gates_on_skill():
    rs = make_runtime()
    rs.skills_loaded.discard(PRIMARY_SKILL)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    resp = tool.invoke({"query": "a chair"})
    assert resp.startswith("ERROR")


def test_select_by_text_calls_keyframe_selector_and_returns_frames():
    rs = make_runtime()
    fake = _attach_selector(rs)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    payload = json.loads(tool.invoke({"query": "a chair", "k": 3}))
    assert fake.calls == [("a chair", 3)]
    fids = [f["frame_id"] for f in payload["frames"]]
    assert fids == [10, 20, 30]
    assert payload["hypothesis_summary"]


def test_select_by_text_attaches_visible_proposal_ids_from_catalog():
    rs = make_runtime()
    _attach_selector(rs)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    payload = json.loads(tool.invoke({"query": "a chair"}))
    f10 = next(f for f in payload["frames"] if f["frame_id"] == 10)
    assert sorted(f10["visible_proposal_ids"]) == [0, 2]


def test_select_by_text_attaches_camera_pose_when_available():
    rs = make_runtime()
    _attach_selector(rs)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    payload = json.loads(tool.invoke({"query": "a chair"}))
    f10 = next(f for f in payload["frames"] if f["frame_id"] == 10)
    assert f10["bev_xy"] == [0.0, 0.0]
    assert f10["camera_yaw"] == 0.0


def test_select_by_text_filters_hidden_categories():
    rs = make_runtime()
    fake = _attach_selector(rs)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    payload = json.loads(
        tool.invoke({"query": "a chair", "k": 3, "hidden_categories": ["chair"]})
    )
    # Stage 1 still returned 3 frames; selector strips chair from visible_proposal_ids
    for frame in payload["frames"]:
        assert 0 not in frame["visible_proposal_ids"]
        assert 1 not in frame["visible_proposal_ids"]
    # And Stage 1 was called with hidden categories
    assert fake.calls and fake.calls[0][0] == "a chair"


def test_select_by_text_missing_keyframe_selector_errors():
    rs = make_runtime()
    rs.keyframe_selector = None
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    resp = tool.invoke({"query": "a chair"})
    assert resp.startswith("ERROR")
    assert "keyframe_selector" in resp
```

- [ ] **Step 3: Run test to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_selectors_text.py -q
```

Expected: `ModuleNotFoundError: agents.tools.selectors`.

- [ ] **Step 4: Implement select_by_text (and the shared selector builder skeleton)**

Create `src/agents/tools/selectors.py`:

```python
"""v9 selectors A-F. Return text-only JSON with frame_id + visible_proposal_ids + camera pose."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from typing import Any, Literal

from langchain_core.tools import BaseTool, tool
from loguru import logger

from agents.catalog import SceneCatalog, SceneProposal
from agents.runtime.scene_runtime import get_scene_catalog
from agents.tools.scene_perception import SCENE_EXPLORATION_SKILL, _gate


def _frame_to_proposals(catalog: SceneCatalog) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for p in catalog.proposals:
        for fid in p.frame_views.keys():
            out.setdefault(int(fid), []).append(p.proposal_id)
    return out


def _norm_category(category: str) -> str:
    return " ".join(str(category).strip().lower().split())


def _camera_pose(runtime: Any, frame_id: int) -> tuple[list[float] | None, float | None]:
    traj = (runtime.bundle.extra_metadata or {}).get("camera_trajectory_xy_yaw") or {}
    entry = traj.get(int(frame_id)) or traj.get(str(frame_id))
    if entry is None:
        return None, None
    return [float(entry[0]), float(entry[1])], float(entry[2])


def _strip_hidden(
    visible: list[int],
    catalog: SceneCatalog,
    hidden_categories: list[str],
) -> list[int]:
    if not hidden_categories:
        return visible
    hidden = {_norm_category(c) for c in hidden_categories}
    cat_by_id = {p.proposal_id: _norm_category(p.category) for p in catalog.proposals}
    return [pid for pid in visible if cat_by_id.get(pid) not in hidden]


def _build_frame_payload(
    runtime: Any,
    catalog: SceneCatalog,
    frame_id: int,
    selected_because: str,
    hidden_categories: list[str],
) -> dict:
    bev_xy, yaw = _camera_pose(runtime, frame_id)
    visible = _frame_to_proposals(catalog).get(int(frame_id), [])
    visible = _strip_hidden(visible, catalog, hidden_categories)
    return {
        "frame_id": int(frame_id),
        "visible_proposal_ids": sorted(visible),
        "bev_xy": bev_xy,
        "camera_yaw": yaw,
        "selected_because": selected_because,
    }


def build_selector_tools(runtime: Any) -> list[BaseTool]:
    @tool
    def select_by_text(
        query: str,
        k: int = 3,
        hidden_categories: list[str] | None = None,
    ) -> str:
        """Selector A. Detailed usage in 'scene-exploration-playbook'."""
        request = {
            "query": query,
            "k": int(k),
            "hidden_categories": list(hidden_categories or []),
        }
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("select_by_text", request, gate)
            return gate
        selector = getattr(runtime, "keyframe_selector", None)
        if selector is None:
            err = "ERROR: runtime.keyframe_selector is None; cannot run Stage-1 text retrieval"
            runtime.record("select_by_text", request, err)
            return err
        try:
            result = selector.select_keyframes_v2(
                query=str(query),
                k=int(k),
                hidden_categories=list(hidden_categories or []),
                use_visual_context=False,
            )
        except Exception as exc:  # noqa: BLE001 — fail-loud
            err = f"ERROR: Stage-1 parse/exec failed: {type(exc).__name__}: {exc}"
            runtime.record("select_by_text", request, err)
            return err

        catalog = get_scene_catalog(runtime)
        frames = [
            _build_frame_payload(
                runtime, catalog, int(fid),
                selected_because=f"select_by_text(query={query!r})",
                hidden_categories=list(hidden_categories or []),
            )
            for fid in (result.keyframe_indices or [])
        ]
        summary = ""
        hyp = (result.metadata or {}).get("hypothesis_output")
        if isinstance(hyp, dict) and hyp.get("hypotheses"):
            first = hyp["hypotheses"][0]
            root = (first.get("grounding_query") or {}).get("root") or {}
            summary = f"target={root.get('category')!r} kind={first.get('kind', 'direct')}"
        payload = {"hypothesis_summary": summary, "frames": frames}
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("select_by_text", request, text)
        return text

    tools = [select_by_text]
    return tools


__all__ = ["build_selector_tools"]
```

- [ ] **Step 5: Run test to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_selectors_text.py -q
```

Expected: 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/agents/tools/selectors.py src/agents/tools/tests/_selector_fixtures.py src/agents/tools/tests/test_selectors_text.py
git commit -m "feat(selectors): add select_by_text (Selector A) using KeyframeSelector"
```

### Task 14: Selector B — `select_by_hypothesis`

**Files:**
- Modify: `src/agents/tools/selectors.py`
- Create: `src/agents/tools/tests/test_selectors_hypothesis.py`

- [ ] **Step 1: Write failing test**

Create `src/agents/tools/tests/test_selectors_hypothesis.py`:

```python
import json

from agents.tools.selectors import build_selector_tools
from agents.tools.tests._selector_fixtures import PRIMARY_SKILL, make_runtime


class _FakeKeyframeSelector:
    def __init__(self):
        self.calls: list[dict] = []

    def execute_hypothesis_dict(self, hypothesis_dict, k=3, hidden_categories=None):
        self.calls.append({"hypothesis": hypothesis_dict, "k": k, "hidden": hidden_categories})
        return {"keyframe_indices": [40, 50, 10], "summary": "executed table hypothesis"}


def _attach(rs):
    rs.keyframe_selector = _FakeKeyframeSelector()
    return rs.keyframe_selector


def test_select_by_hypothesis_dispatches_to_executor():
    rs = make_runtime()
    fake = _attach(rs)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_hypothesis")
    hyp = {"hypotheses": [{"rank": 1, "kind": "direct", "grounding_query": {"root": {"category": "table"}}}]}
    payload = json.loads(tool.invoke({"hypothesis": hyp, "k": 2}))
    assert fake.calls and fake.calls[0]["k"] == 2
    fids = [f["frame_id"] for f in payload["frames"]]
    # Output ordered by selector; trim ignored — sanity that selector did not filter content
    assert sorted(fids) == [10, 40, 50]


def test_select_by_hypothesis_invalid_payload_errors():
    rs = make_runtime()
    _attach(rs)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_hypothesis")
    resp = tool.invoke({"hypothesis": "not a dict"})
    assert resp.startswith("ERROR")
    assert "hypothesis" in resp


def test_select_by_hypothesis_gates_on_skill():
    rs = make_runtime()
    rs.skills_loaded.discard(PRIMARY_SKILL)
    _attach(rs)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_hypothesis")
    resp = tool.invoke({"hypothesis": {"hypotheses": []}})
    assert resp.startswith("ERROR")
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_selectors_hypothesis.py -q
```

Expected: tool not registered.

- [ ] **Step 3: Implement select_by_hypothesis**

Inside `src/agents/tools/selectors.py`, before `return tools`, add:

```python
    @tool
    def select_by_hypothesis(
        hypothesis: dict,
        k: int = 3,
        hidden_categories: list[str] | None = None,
    ) -> str:
        """Selector B. Detailed usage in 'scene-exploration-playbook'."""
        request = {
            "hypothesis": hypothesis,
            "k": int(k),
            "hidden_categories": list(hidden_categories or []),
        }
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("select_by_hypothesis", request, gate)
            return gate
        if not isinstance(hypothesis, dict) or "hypotheses" not in hypothesis:
            err = "ERROR: hypothesis must be a dict with 'hypotheses' list (HypothesisOutputV1 shape)"
            runtime.record("select_by_hypothesis", request, err)
            return err
        selector = getattr(runtime, "keyframe_selector", None)
        if selector is None or not hasattr(selector, "execute_hypothesis_dict"):
            err = "ERROR: runtime.keyframe_selector does not support execute_hypothesis_dict"
            runtime.record("select_by_hypothesis", request, err)
            return err
        try:
            result = selector.execute_hypothesis_dict(
                hypothesis_dict=hypothesis,
                k=int(k),
                hidden_categories=list(hidden_categories or []),
            )
        except Exception as exc:  # noqa: BLE001
            err = f"ERROR: hypothesis execution failed: {type(exc).__name__}: {exc}"
            runtime.record("select_by_hypothesis", request, err)
            return err
        catalog = get_scene_catalog(runtime)
        frames = [
            _build_frame_payload(
                runtime, catalog, int(fid),
                selected_because="select_by_hypothesis",
                hidden_categories=list(hidden_categories or []),
            )
            for fid in (result.get("keyframe_indices") or [])
        ]
        payload = {
            "hypothesis_summary": str(result.get("summary") or ""),
            "frames": frames,
        }
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("select_by_hypothesis", request, text)
        return text

    tools.append(select_by_hypothesis)
```

Also add a wrapper method to `KeyframeSelector` (in a separate task to keep diff scoped). For now, the test stubs the method. Add the production wrapper in Task 31's adapter step (the wrapper is a thin call to `execute_hypotheses` on the existing class).

- [ ] **Step 4: Run test to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_selectors_hypothesis.py -q
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agents/tools/selectors.py src/agents/tools/tests/test_selectors_hypothesis.py
git commit -m "feat(selectors): add select_by_hypothesis (Selector B)"
```

### Task 15: Selector C — `select_by_frame_neighbor`

**Files:**
- Modify: `src/agents/tools/selectors.py`
- Create: `src/agents/tools/tests/test_selectors_frame_neighbor.py`

- [ ] **Step 1: Write failing test**

Create `src/agents/tools/tests/test_selectors_frame_neighbor.py`:

```python
import json

from agents.tools.selectors import build_selector_tools
from agents.tools.tests._selector_fixtures import make_runtime


def test_select_by_frame_neighbor_temporal_returns_adjacent_frames():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_frame_neighbor")
    payload = json.loads(tool.invoke({"anchor_frame_id": 30, "mode": "temporal", "k": 2}))
    fids = [f["frame_id"] for f in payload["frames"]]
    # valid_frame_ids = [10, 20, 30, 40, 50]; nearest to 30 by id are [20, 40]
    assert sorted(fids) == [20, 40]


def test_select_by_frame_neighbor_temporal_skips_anchor_itself():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_frame_neighbor")
    payload = json.loads(tool.invoke({"anchor_frame_id": 10, "mode": "temporal", "k": 3}))
    fids = [f["frame_id"] for f in payload["frames"]]
    assert 10 not in fids
    assert len(fids) == 3
    assert fids[0] == 20  # nearest temporal neighbor


def test_select_by_frame_neighbor_viewpoint_diverse_prefers_yaw_spread():
    """Returns frames spatially close to anchor but with the largest yaw deltas."""
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_frame_neighbor")
    payload = json.loads(tool.invoke({"anchor_frame_id": 10, "mode": "viewpoint_diverse", "k": 2}))
    fids = [f["frame_id"] for f in payload["frames"]]
    assert len(fids) == 2
    # anchor 10 has yaw=0.0; the two largest yaw deltas among others are 30 (yaw=1.0) and 50 (yaw=3.0)
    assert sorted(fids) == [30, 50]


def test_select_by_frame_neighbor_unknown_anchor_errors():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_frame_neighbor")
    resp = tool.invoke({"anchor_frame_id": 999, "mode": "temporal"})
    assert resp.startswith("ERROR")


def test_select_by_frame_neighbor_invalid_mode_errors():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_frame_neighbor")
    resp = tool.invoke({"anchor_frame_id": 10, "mode": "bogus"})
    assert resp.startswith("ERROR")
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_selectors_frame_neighbor.py -q
```

Expected: tool not registered.

- [ ] **Step 3: Implement select_by_frame_neighbor**

Inside `build_selector_tools(runtime)` of `src/agents/tools/selectors.py`, append:

```python
    @tool
    def select_by_frame_neighbor(
        anchor_frame_id: int,
        mode: str = "temporal",
        k: int = 3,
    ) -> str:
        """Selector C. Detailed usage in 'scene-exploration-playbook'."""
        request = {"anchor_frame_id": int(anchor_frame_id), "mode": mode, "k": int(k)}
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("select_by_frame_neighbor", request, gate)
            return gate
        if mode not in ("temporal", "viewpoint_diverse"):
            err = f"ERROR: mode must be 'temporal' or 'viewpoint_diverse'; got {mode!r}"
            runtime.record("select_by_frame_neighbor", request, err)
            return err
        catalog = get_scene_catalog(runtime)
        valid = sorted(int(f) for f in catalog.valid_frame_ids)
        if int(anchor_frame_id) not in valid:
            err = (
                f"ERROR: anchor_frame_id={anchor_frame_id} not in valid_frame_ids; "
                f"available[:20]={valid[:20]}"
            )
            runtime.record("select_by_frame_neighbor", request, err)
            return err
        others = [f for f in valid if f != int(anchor_frame_id)]
        if mode == "temporal":
            others.sort(key=lambda f: (abs(f - int(anchor_frame_id)), f))
            chosen = others[:int(k)]
        else:
            anchor_pose, anchor_yaw = _camera_pose(runtime, int(anchor_frame_id))
            ranked: list[tuple[float, int]] = []
            for f in others:
                pose, yaw = _camera_pose(runtime, f)
                if anchor_pose is None or pose is None or anchor_yaw is None or yaw is None:
                    distance = float(abs(f - int(anchor_frame_id)))
                    yaw_delta = 0.0
                else:
                    distance = math.hypot(pose[0] - anchor_pose[0], pose[1] - anchor_pose[1])
                    yaw_delta = abs(yaw - anchor_yaw)
                ranked.append((distance - yaw_delta, f))
            ranked.sort(key=lambda item: item[0])
            chosen = [f for _, f in ranked[:int(k)]]
        frames = [
            _build_frame_payload(
                runtime, catalog, int(fid),
                selected_because=f"select_by_frame_neighbor(anchor={anchor_frame_id}, mode={mode!r})",
                hidden_categories=[],
            )
            for fid in chosen
        ]
        payload = {"hypothesis_summary": "", "frames": frames}
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("select_by_frame_neighbor", request, text)
        return text

    tools.append(select_by_frame_neighbor)
```

- [ ] **Step 4: Run test to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_selectors_frame_neighbor.py -q
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agents/tools/selectors.py src/agents/tools/tests/test_selectors_frame_neighbor.py
git commit -m "feat(selectors): add select_by_frame_neighbor (Selector C)"
```

### Task 16: Selector D — `select_by_proposal`

**Files:**
- Modify: `src/agents/tools/selectors.py`
- Create: `src/agents/tools/tests/test_selectors_proposal.py`

- [ ] **Step 1: Write failing test**

Create `src/agents/tools/tests/test_selectors_proposal.py`:

```python
import json

from agents.tools.selectors import build_selector_tools
from agents.tools.tests._selector_fixtures import make_runtime


def test_select_by_proposal_union_default():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_proposal")
    payload = json.loads(tool.invoke({"proposal_ids": [0, 2]}))
    fids = sorted(f["frame_id"] for f in payload["frames"])
    # proposal 0 in [10, 20]; proposal 2 in [10, 40] -> union [10, 20, 40]
    assert fids == [10, 20, 40]


def test_select_by_proposal_intersection_when_require_all_true():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_proposal")
    payload = json.loads(tool.invoke({"proposal_ids": [0, 2], "require_all": True}))
    fids = [f["frame_id"] for f in payload["frames"]]
    assert fids == [10]  # only frame in both proposal_index sets


def test_select_by_proposal_k_caps_output():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_proposal")
    payload = json.loads(tool.invoke({"proposal_ids": [0, 2], "k": 2}))
    assert len(payload["frames"]) == 2


def test_select_by_proposal_unknown_id_errors():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_proposal")
    resp = tool.invoke({"proposal_ids": [9999]})
    assert resp.startswith("ERROR")
    assert "9999" in resp


def test_select_by_proposal_empty_list_errors():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_proposal")
    resp = tool.invoke({"proposal_ids": []})
    assert resp.startswith("ERROR")
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_selectors_proposal.py -q
```

Expected: tool not registered.

- [ ] **Step 3: Implement select_by_proposal**

Inside `build_selector_tools(runtime)`, append:

```python
    @tool
    def select_by_proposal(
        proposal_ids: list[int],
        require_all: bool = False,
        k: int = 3,
    ) -> str:
        """Selector D. Detailed usage in 'scene-exploration-playbook'."""
        request = {
            "proposal_ids": list(proposal_ids or []),
            "require_all": bool(require_all),
            "k": int(k),
        }
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("select_by_proposal", request, gate)
            return gate
        ids = [int(i) for i in (proposal_ids or [])]
        if not ids:
            err = "ERROR: proposal_ids must be a non-empty list of ints"
            runtime.record("select_by_proposal", request, err)
            return err
        catalog = get_scene_catalog(runtime)
        proposal_by_id = {p.proposal_id: p for p in catalog.proposals}
        missing = [pid for pid in ids if pid not in proposal_by_id]
        if missing:
            err = (
                f"ERROR: proposal_id(s) not in catalog: {missing}; "
                f"available count={len(proposal_by_id)}"
            )
            runtime.record("select_by_proposal", request, err)
            return err
        sets = [set(proposal_by_id[pid].frame_views.keys()) for pid in ids]
        if not sets:
            frames_set: set[int] = set()
        elif require_all:
            frames_set = set.intersection(*sets)
        else:
            frames_set = set().union(*sets)
        chosen = sorted(int(f) for f in frames_set)[: int(k)]
        frames = [
            _build_frame_payload(
                runtime, catalog, int(fid),
                selected_because=(
                    f"select_by_proposal(proposal_ids={ids}, require_all={bool(require_all)})"
                ),
                hidden_categories=[],
            )
            for fid in chosen
        ]
        payload = {"hypothesis_summary": "", "frames": frames}
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("select_by_proposal", request, text)
        return text

    tools.append(select_by_proposal)
```

- [ ] **Step 4: Run test to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_selectors_proposal.py -q
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agents/tools/selectors.py src/agents/tools/tests/test_selectors_proposal.py
git commit -m "feat(selectors): add select_by_proposal (Selector D)"
```

### Task 17: Selector E — `select_by_region`

**Files:**
- Modify: `src/agents/tools/selectors.py`
- Create: `src/agents/tools/tests/test_selectors_region.py`

- [ ] **Step 1: Write failing test**

Create `src/agents/tools/tests/test_selectors_region.py`:

```python
import json

from agents.tools.selectors import build_selector_tools
from agents.tools.tests._selector_fixtures import make_runtime


def test_select_by_region_bev_2d_picks_frames_whose_camera_in_box():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_region")
    payload = json.loads(
        tool.invoke({"region": [-0.5, -0.5, 1.6, 1.0], "region_type": "bev_2d", "k": 5})
    )
    fids = sorted(f["frame_id"] for f in payload["frames"])
    # Camera xy: 10->(0,0), 20->(0.5,0), 30->(1.0,0), 40->(1.5,0.5), 50->(-2,-2)
    # In box [-0.5,-0.5, 1.6, 1.0]: frames 10, 20, 30, 40
    assert fids == [10, 20, 30, 40]


def test_select_by_region_bbox_3d_picks_frames_whose_proposals_inside():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_region")
    # Box covering only chair @ (0,0,0) and chair @ (2,0,0): proposal 0 and 1
    payload = json.loads(
        tool.invoke(
            {
                "region": [-0.5, -0.5, -0.5, 2.5, 0.5, 0.5],
                "region_type": "bbox_3d",
                "k": 10,
            }
        )
    )
    fids = sorted(f["frame_id"] for f in payload["frames"])
    # proposal 0 in [10, 20], proposal 1 in [30] -> union [10, 20, 30]
    assert fids == [10, 20, 30]


def test_select_by_region_invalid_region_length_errors():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_region")
    resp = tool.invoke({"region": [0, 0], "region_type": "bev_2d"})
    assert resp.startswith("ERROR")
    assert "region" in resp


def test_select_by_region_unknown_region_type_errors():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_region")
    resp = tool.invoke({"region": [0, 0, 1, 1], "region_type": "foo"})
    assert resp.startswith("ERROR")
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_selectors_region.py -q
```

Expected: tool not registered.

- [ ] **Step 3: Implement select_by_region**

Inside `build_selector_tools(runtime)`, append:

```python
    @tool
    def select_by_region(
        region: list[float],
        region_type: str = "bev_2d",
        k: int = 3,
    ) -> str:
        """Selector E. Detailed usage in 'scene-exploration-playbook'."""
        request = {
            "region": [float(v) for v in (region or [])],
            "region_type": region_type,
            "k": int(k),
        }
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("select_by_region", request, gate)
            return gate
        if region_type not in ("bev_2d", "bbox_3d"):
            err = f"ERROR: region_type must be 'bev_2d' or 'bbox_3d'; got {region_type!r}"
            runtime.record("select_by_region", request, err)
            return err
        if region_type == "bev_2d":
            if len(region) != 4:
                err = "ERROR: bev_2d region must be [xmin, ymin, xmax, ymax]"
                runtime.record("select_by_region", request, err)
                return err
            xmin, ymin, xmax, ymax = (float(v) for v in region)
            catalog = get_scene_catalog(runtime)
            chosen: list[int] = []
            for fid in sorted(int(f) for f in catalog.valid_frame_ids):
                pose, _ = _camera_pose(runtime, fid)
                if pose is None:
                    continue
                if xmin <= pose[0] <= xmax and ymin <= pose[1] <= ymax:
                    chosen.append(fid)
                    if len(chosen) >= int(k):
                        break
        else:
            if len(region) != 6:
                err = "ERROR: bbox_3d region must be [xmin, ymin, zmin, xmax, ymax, zmax]"
                runtime.record("select_by_region", request, err)
                return err
            xmin, ymin, zmin, xmax, ymax, zmax = (float(v) for v in region)
            catalog = get_scene_catalog(runtime)
            inside_proposals = [
                p.proposal_id
                for p in catalog.proposals
                if xmin <= p.position_3d[0] <= xmax
                and ymin <= p.position_3d[1] <= ymax
                and zmin <= p.position_3d[2] <= zmax
            ]
            frame_to_props = _frame_to_proposals(catalog)
            chosen_set: set[int] = set()
            for fid, props in frame_to_props.items():
                if any(pid in inside_proposals for pid in props):
                    chosen_set.add(int(fid))
            chosen = sorted(chosen_set)[: int(k)]
        catalog = get_scene_catalog(runtime)
        frames = [
            _build_frame_payload(
                runtime, catalog, int(fid),
                selected_because=f"select_by_region(type={region_type!r})",
                hidden_categories=[],
            )
            for fid in chosen
        ]
        payload = {"hypothesis_summary": "", "frames": frames}
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("select_by_region", request, text)
        return text

    tools.append(select_by_region)
```

- [ ] **Step 4: Run test to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_selectors_region.py -q
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agents/tools/selectors.py src/agents/tools/tests/test_selectors_region.py
git commit -m "feat(selectors): add select_by_region (Selector E) for bev_2d + bbox_3d"
```

### Task 18: Selector F — `select_by_coverage`

**Files:**
- Modify: `src/agents/tools/selectors.py`
- Create: `src/agents/tools/tests/test_selectors_coverage.py`

- [ ] **Step 1: Write failing test**

Create `src/agents/tools/tests/test_selectors_coverage.py`:

```python
import json

from agents.tools.selectors import build_selector_tools
from agents.tools.tests._selector_fixtures import make_runtime


def test_select_by_coverage_obj_iou_picks_most_jaccard_distinct():
    rs = make_runtime()
    rs.seen_image_paths.update(["r.png"])
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_coverage")
    payload = json.loads(
        tool.invoke({"method": "obj_iou", "k": 2, "seen_frame_ids": [10]})
    )
    fids = [f["frame_id"] for f in payload["frames"]]
    # Frame 10 contains proposals {0, 2}; the most different by Jaccard is frame 50 ({3})
    assert 50 in fids
    assert 10 not in fids


def test_select_by_coverage_pose_depth_picks_most_distant_camera():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_coverage")
    payload = json.loads(
        tool.invoke({"method": "pose_depth", "k": 1, "seen_frame_ids": [10]})
    )
    fids = [f["frame_id"] for f in payload["frames"]]
    # camera xy: 10->(0,0); farthest from (0,0) is 50->(-2,-2)
    assert fids == [50]


def test_select_by_coverage_invalid_method_errors():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_coverage")
    resp = tool.invoke({"method": "bogus"})
    assert resp.startswith("ERROR")


def test_select_by_coverage_no_seen_frames_uses_runtime_seen_image_paths(monkeypatch):
    rs = make_runtime()
    rs.seen_image_paths = set()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_coverage")
    payload = json.loads(tool.invoke({"method": "obj_iou", "k": 2}))
    # When seen set is empty, every frame is "novel"; returns first k by valid_frame_ids order.
    fids = [f["frame_id"] for f in payload["frames"]]
    assert fids[:2] == [10, 20]
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_selectors_coverage.py -q
```

Expected: tool not registered.

- [ ] **Step 3: Implement select_by_coverage**

Inside `build_selector_tools(runtime)`, append:

```python
    def _seen_frame_ids_from_runtime() -> set[int]:
        seen: set[int] = set()
        for entry in list(getattr(runtime, "tool_trace", []) or []):
            tool_input = getattr(entry, "tool_input", {}) or {}
            if getattr(entry, "tool_name", "") in (
                "view_keyframe",
                "select_by_text",
                "select_by_hypothesis",
                "select_by_proposal",
                "select_by_region",
                "select_by_frame_neighbor",
                "select_by_coverage",
            ):
                fid = tool_input.get("frame_id")
                if isinstance(fid, int):
                    seen.add(fid)
        return seen

    @tool
    def select_by_coverage(
        method: str = "obj_iou",
        k: int = 3,
        seen_frame_ids: list[int] | None = None,
    ) -> str:
        """Selector F. Detailed usage in 'scene-exploration-playbook'."""
        request = {"method": method, "k": int(k), "seen_frame_ids": seen_frame_ids}
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("select_by_coverage", request, gate)
            return gate
        if method not in ("obj_iou", "pose_depth"):
            err = f"ERROR: method must be 'obj_iou' or 'pose_depth'; got {method!r}"
            runtime.record("select_by_coverage", request, err)
            return err
        catalog = get_scene_catalog(runtime)
        valid = sorted(int(f) for f in catalog.valid_frame_ids)
        seen_set: set[int]
        if seen_frame_ids is None:
            seen_set = _seen_frame_ids_from_runtime()
        else:
            seen_set = {int(f) for f in seen_frame_ids}
        unseen = [f for f in valid if f not in seen_set]
        if not unseen:
            payload = {"hypothesis_summary": "all frames already seen", "frames": []}
            text = json.dumps(payload, ensure_ascii=False)
            runtime.record("select_by_coverage", request, text)
            return text
        if method == "obj_iou":
            frame_to_props = _frame_to_proposals(catalog)
            if not seen_set:
                # Same shape as input order
                chosen = unseen[: int(k)]
            else:
                seen_union: set[int] = set()
                for s in seen_set:
                    seen_union.update(frame_to_props.get(int(s), []))

                def jaccard_distance(fid: int) -> float:
                    a = set(frame_to_props.get(int(fid), []))
                    if not a and not seen_union:
                        return 0.0
                    union = a | seen_union
                    inter = a & seen_union
                    return 1.0 - (len(inter) / len(union)) if union else 0.0

                unseen.sort(key=lambda f: (-jaccard_distance(f), f))
                chosen = unseen[: int(k)]
        else:  # pose_depth
            if not seen_set:
                chosen = unseen[: int(k)]
            else:
                centroid_x = 0.0
                centroid_y = 0.0
                count = 0
                for s in seen_set:
                    pose, _ = _camera_pose(runtime, int(s))
                    if pose is None:
                        continue
                    centroid_x += pose[0]
                    centroid_y += pose[1]
                    count += 1
                if count == 0:
                    chosen = unseen[: int(k)]
                else:
                    centroid_x /= count
                    centroid_y /= count

                    def dist(fid: int) -> float:
                        pose, _ = _camera_pose(runtime, int(fid))
                        if pose is None:
                            return float("inf")
                        return -math.hypot(pose[0] - centroid_x, pose[1] - centroid_y)

                    unseen.sort(key=lambda f: (dist(f), f))
                    chosen = unseen[: int(k)]
        frames = [
            _build_frame_payload(
                runtime, catalog, int(fid),
                selected_because=f"select_by_coverage(method={method!r})",
                hidden_categories=[],
            )
            for fid in chosen
        ]
        payload = {"hypothesis_summary": "", "frames": frames}
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("select_by_coverage", request, text)
        return text

    tools.append(select_by_coverage)
```

- [ ] **Step 4: Run test to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_selectors_coverage.py -q
```

Expected: 4 tests pass.

- [ ] **Step 5: Final ruff on tools/**

```bash
.venv/bin/ruff check src/agents/tools
```

Expected: no findings.

- [ ] **Step 6: Run the entire tools test suite to catch regressions**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests -q
```

Expected: all selector + scene-perception + view_keyframe tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/agents/tools/selectors.py src/agents/tools/tests/test_selectors_coverage.py
git commit -m "$(cat <<'EOF'
feat(selectors): add select_by_coverage (Selector F) — obj_iou + pose_depth

Completes the 6-selector set (A-F). All six return text-only frame
payloads with visible_proposal_ids and camera pose. seen_frame_ids
defaults to the runtime tool_trace when None.

Per spec Section A and C.
EOF
)"
```

---

## Phase 5 — Initial Prompt + Pack Prep

### Task 19: Rewrite `build_system_prompt` (v9 system prompt per F.1)

**Files:**
- Modify: `src/agents/runtime/base.py`
- Modify: `src/agents/tests/test_stage2_deep_agent.py` (or add a focused test)
- Create: `src/agents/runtime/tests/test_build_system_prompt_v9.py`

- [ ] **Step 1: Write failing test for v9 system prompt**

Create `src/agents/runtime/tests/test_build_system_prompt_v9.py`:

```python
from agents.core.agent_config import (
    Stage2DeepAgentConfig,
    Stage2PlanMode,
    Stage2TaskType,
)
from agents.core.task_types import Stage2TaskSpec
from agents.runtime.base import BaseStage2Runtime


class _MinimalRuntime(BaseStage2Runtime):
    def run(self, task, bundle):
        raise NotImplementedError


def _task(task_type=Stage2TaskType.VISUAL_GROUNDING) -> Stage2TaskSpec:
    return Stage2TaskSpec(
        user_query="this is a brown chair",
        task_type=task_type,
        plan_mode=Stage2PlanMode.BRIEF,
        max_reasoning_turns=8,
    )


def test_system_prompt_has_v9_catalog_first_header():
    rt = _MinimalRuntime(config=Stage2DeepAgentConfig())
    prompt = rt.build_system_prompt(_task())
    assert "You are the Stage-2 scene reasoning agent" in prompt
    assert "Scene perception model" in prompt


def test_system_prompt_lists_six_selectors_cheapest_first():
    rt = _MinimalRuntime(config=Stage2DeepAgentConfig())
    prompt = rt.build_system_prompt(_task())
    assert "select_by_proposal" in prompt
    assert "select_by_frame_neighbor" in prompt
    assert "select_by_region" in prompt
    assert "select_by_coverage" in prompt
    assert "select_by_text" in prompt
    assert "select_by_hypothesis" in prompt
    # Cheapest-first principle line
    assert "Cheapest-first" in prompt


def test_system_prompt_no_longer_mentions_callback_tools():
    rt = _MinimalRuntime(config=Stage2DeepAgentConfig())
    prompt = rt.build_system_prompt(_task())
    assert "request_more_views" not in prompt
    assert "switch_or_expand_hypothesis" not in prompt
    assert "inspect_stage1_metadata" not in prompt
    assert "view_keyframe_marked" not in prompt
    assert "find_proposals_by_category" not in prompt
    assert "list_keyframes_with_proposals" not in prompt


def test_system_prompt_for_qa_uses_view_keyframe_rgb_default():
    rt = _MinimalRuntime(config=Stage2DeepAgentConfig())
    prompt = rt.build_system_prompt(_task(Stage2TaskType.QA))
    assert "view_keyframe" in prompt
    assert "mode='auto'" in prompt or "mode=\"auto\"" in prompt
    # QA-specific instruction line
    assert "QA" in prompt or "qa" in prompt


def test_system_prompt_mentions_scene_exploration_playbook_first():
    rt = _MinimalRuntime(config=Stage2DeepAgentConfig())
    prompt = rt.build_system_prompt(_task())
    assert "scene-exploration-playbook" in prompt


def test_system_prompt_drops_enable_temporal_fan_branch():
    rt = _MinimalRuntime(config=Stage2DeepAgentConfig())
    prompt = rt.build_system_prompt(_task())
    assert "temporal_fan" not in prompt
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/runtime/tests/test_build_system_prompt_v9.py -q
```

Expected: tests fail (current prompt still mentions `request_more_views`).

- [ ] **Step 3: Rewrite `build_system_prompt`**

Open `src/agents/runtime/base.py` and replace `build_system_prompt` (currently lines ~420-536) with the catalog-first version. Replace:

```python
    def build_system_prompt(
        self,
        task: Stage2TaskSpec,
        object_context: dict[str, str] | None = None,
    ) -> str:
        """Build the agent system prompt."""
```

…through the closing of the method, with the following body (everything from `def build_system_prompt(...)` to the end of `return (...)`):

```python
    def build_system_prompt(
        self,
        task: Stage2TaskSpec,
        object_context: dict[str, str] | None = None,
    ) -> str:
        """Build the v9 catalog-first system prompt."""
        plan_instructions = {
            Stage2PlanMode.OFF: (
                "Plan mode is OFF. Only use the todo list if the task is unexpectedly complex."
            ),
            Stage2PlanMode.BRIEF: (
                "Plan mode is BRIEF. Maintain a short todo list (2-4 items) covering evidence "
                "acquisition and answer synthesis."
            ),
            Stage2PlanMode.FULL: (
                "Plan mode is FULL. Maintain an explicit todo list decomposed into evidence "
                "acquisition, verification, and task synthesis."
            ),
        }
        payload_schema = task.expected_output_schema or default_payload_schema(task.task_type)
        instruction = task.output_instruction or default_output_instruction(task.task_type)
        if task.task_type == Stage2TaskType.QA:
            mode_hint = (
                "Default view mode for QA: view_keyframe(mode='auto') resolves to 'rgb'."
            )
        elif task.task_type == Stage2TaskType.VISUAL_GROUNDING:
            mode_hint = (
                "Default view mode for VG: view_keyframe(mode='auto') resolves to 'marked'."
            )
        else:
            mode_hint = (
                "Default view mode: view_keyframe(mode='auto') picks 'marked' for VG, 'rgb' otherwise."
            )

        return (
            "You are the Stage-2 scene reasoning agent.\n\n"
            "Scene perception model:\n"
            "- You start with a BEV overview image plus a SceneCatalog text (category -> [#id, ...]).\n"
            "- You have viewed 0 first-person frames at task start.\n"
            "- The BEV labels are a starting point, not first-person evidence; you must fetch frames.\n\n"
            "Tool families (always `load_skill('scene-exploration-playbook')` before selectors / view tools):\n"
            "1. select_by_* (6 modalities) — find task-relevant frame_ids\n"
            "   - select_by_proposal(proposal_ids, require_all, k) — instant catalog lookup\n"
            "   - select_by_frame_neighbor(anchor_frame_id, mode='temporal'|'viewpoint_diverse', k) — instant\n"
            "   - select_by_region(region, region_type='bev_2d'|'bbox_3d', k) — instant\n"
            "   - select_by_coverage(method='obj_iou'|'pose_depth', k, seen_frame_ids?) — cheap geometric\n"
            "   - select_by_text(query, k, hidden_categories) — ~Stage-1 LLM parse (2-5s)\n"
            "   - select_by_hypothesis(hypothesis_json, k, hidden_categories) — instant (no parse)\n"
            "2. view_keyframe(frame_id, mode='auto', categories?, proposal_ids?) — inject a first-person frame.\n"
            f"   {mode_hint}\n"
            "3. view_bev(highlight=[ids]?) — re-inject the BEV; optionally focus on a subset of #ids.\n"
            "4. list_scene_proposals(category?, region_bev?, limit?) and list_frame_proposals(frame_id) — scene/frame inventory text.\n"
            "5. inspect_proposal(proposal_id) — proposal metadata + frames_appeared.\n"
            "6. compare_proposals_spatial(candidate_ids, anchor_id, relation) — spatial reasoning.\n"
            "7. request_crops(request_text, object_terms) — zoom in for small attributes / state.\n"
            "8. submit_final(payload, rationale, evidence_refs?, tool_override_reason?) — terminate.\n\n"
            "Cheapest-first principle:\n"
            "- Prefer catalog-only selectors (proposal / frame_neighbor / region / coverage) before Stage-1 LLM (text / hypothesis).\n"
            "- Use request_crops only after view_keyframe failed to resolve a small attribute or count.\n\n"
            "Skill gate:\n"
            "- Every selector + view_keyframe + view_bev + list_scene_proposals + inspect_proposal + compare_proposals_spatial\n"
            "  refuses to run until you `load_skill('scene-exploration-playbook')`. After that, load the\n"
            "  task-specific playbook (`vg-grounding-playbook` for VG, `qa-answering-playbook` for QA).\n\n"
            f"{self._format_skill_catalog(task.task_type)}"
            "Framework constraints:\n"
            "- LangChain v1 + DeepAgents runtime.\n"
            f"- Maximum reasoning budget: {task.max_reasoning_turns} turns.\n\n"
            f"{plan_instructions[task.plan_mode]}\n\n"
            "Unified output contract:\n"
            f"- task_type must be `{task.task_type.value}`.\n"
            "- status must reflect whether the task is complete or evidence-limited.\n"
            "- payload must follow the schema below.\n"
            "- cited_frame_indices must only cite frames you actually viewed.\n\n"
            f"Task-specific instruction: {instruction}\n"
            f"Expected payload schema: {json.dumps(payload_schema, indent=2, ensure_ascii=False)}"
        )
```

Then delete the obsolete helpers in `base.py` that are no longer referenced:

```python
    def _format_vg_section(self, extra_schema: dict[str, Any]) -> str:
```

(remove the entire method). Search for callers; if any remain, they will be removed in Task 27.

- [ ] **Step 4: Run system-prompt tests to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/runtime/tests/test_build_system_prompt_v9.py -q
```

Expected: 6 tests pass.

- [ ] **Step 5: Run existing stage2 tests to confirm no breakage spillover**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tests/test_stage2_deep_agent.py -q
```

Expected: some assertions on the OLD prompt strings ("request_more_views", "ALWAYS examine") will fail — that is fine here; those tests are rewritten in Task 32. Continue with the commit.

- [ ] **Step 6: Commit**

```bash
git add src/agents/runtime/base.py src/agents/runtime/tests/test_build_system_prompt_v9.py
git commit -m "$(cat <<'EOF'
refactor(prompt): rewrite build_system_prompt to v9 catalog-first

Removes the v7/v8 'request_more_views first / examine keyframes first'
strategy. New prompt teaches the 6 selectors + 4 perception tools + the
scene-exploration-playbook gate. Drops _format_vg_section helper.

Per spec Section F.1.
EOF
)"
```

### Task 20: Rewrite `build_user_message` → catalog-first (BEV image + Cat-B)

**Files:**
- Modify: `src/agents/runtime/deepagents_agent.py`
- Modify: `src/agents/runtime/base.py` (collect_image_paths — BEV only)
- Create: `src/agents/runtime/tests/test_build_task_message_v9.py`

- [ ] **Step 1: Write failing test for the new task message**

Create `src/agents/runtime/tests/test_build_task_message_v9.py`:

```python
from pathlib import Path

from PIL import Image

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.core.agent_config import (
    Stage2DeepAgentConfig,
    Stage2PlanMode,
    Stage2TaskType,
)
from agents.core.task_types import Stage2EvidenceBundle, Stage2TaskSpec
from agents.runtime.base import Stage2RuntimeState
from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime


def _catalog(scene_id: str = "scannet/scene0123_45") -> SceneCatalog:
    return SceneCatalog(
        scene_id=scene_id,
        scene_category="kitchen-living-room",
        proposals=[
            SceneProposal(proposal_id=4, category="chair", position_3d=(0, 0, 0), source="mask3d"),
            SceneProposal(proposal_id=5, category="chair", position_3d=(1, 0, 0), source="mask3d"),
            SceneProposal(proposal_id=8, category="table", position_3d=(2, 0, 0), source="mask3d"),
            SceneProposal(proposal_id=11, category="lamp", position_3d=(-1, 0, 0), source="mask3d"),
        ],
        total_frames=187,
        frame_id_range=(0, 1860),
        valid_frame_ids=[i * 10 for i in range(187)],
        bev_image_path="bev.png",
    )


def _bundle(tmp_path: Path) -> Stage2EvidenceBundle:
    bev = tmp_path / "bev.png"
    Image.new("RGB", (200, 200), (10, 10, 10)).save(bev)
    cat = _catalog()
    cat.bev_image_path = str(bev)
    return Stage2EvidenceBundle(
        scene_id=cat.scene_id,
        extra_metadata={"scene_catalog": cat.model_dump()},
        bev_image_path=str(bev),
    )


def test_build_user_message_no_first_person_seed_only_bev(tmp_path: Path):
    rt = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    bundle = _bundle(tmp_path)
    rs = Stage2RuntimeState(bundle=bundle)
    task = Stage2TaskSpec(
        user_query="this is a brown wooden chair next to the kitchen counter",
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        plan_mode=Stage2PlanMode.BRIEF,
        max_reasoning_turns=10,
    )
    msg = rt.build_user_message(task, rs)
    # exactly one text + one BEV image_url
    parts = msg.content
    text_parts = [p for p in parts if p.get("type") == "text"]
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert len(text_parts) == 1
    assert len(image_parts) == 1
    text = text_parts[0]["text"]
    assert "Current keyframes:" not in text
    assert "Stage-1 hypothesis summary" not in text
    assert "## Scene" in text
    assert "## BEV image (attached above)" in text


def test_build_user_message_cat_b_inventory(tmp_path: Path):
    rt = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    rs = Stage2RuntimeState(bundle=_bundle(tmp_path))
    task = Stage2TaskSpec(
        user_query="a chair",
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        plan_mode=Stage2PlanMode.BRIEF,
        max_reasoning_turns=4,
    )
    text = rt.build_user_message(task, rs).content[0]["text"]
    assert "Proposals by category:" in text
    assert "chair: [#4, #5]" in text
    assert "table: [#8]" in text
    assert "lamp:  [#11]" in text or "lamp: [#11]" in text


def test_build_user_message_qa_uses_rgb_note(tmp_path: Path):
    rt = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    rs = Stage2RuntimeState(bundle=_bundle(tmp_path))
    task = Stage2TaskSpec(
        user_query="how many chairs are there?",
        task_type=Stage2TaskType.QA,
        plan_mode=Stage2PlanMode.BRIEF,
        max_reasoning_turns=4,
    )
    text = rt.build_user_message(task, rs).content[0]["text"]
    assert "view_keyframe(mode='rgb')" in text or 'mode="rgb"' in text


def test_build_user_message_zero_keyframes_viewed_line(tmp_path: Path):
    rt = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    rs = Stage2RuntimeState(bundle=_bundle(tmp_path))
    task = Stage2TaskSpec(
        user_query="a chair",
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        plan_mode=Stage2PlanMode.BRIEF,
        max_reasoning_turns=2,
    )
    text = rt.build_user_message(task, rs).content[0]["text"]
    assert "viewed 0 keyframes out of 187" in text


def test_collect_image_paths_returns_bev_only(tmp_path: Path):
    rt = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    bundle = _bundle(tmp_path)
    paths = rt.collect_image_paths(bundle)
    assert paths == [str(tmp_path / "bev.png")]
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/runtime/tests/test_build_task_message_v9.py -q
```

Expected: multiple assertions fail (current `build_user_message` still emits "Current keyframes" and "Stage-1 hypothesis summary").

- [ ] **Step 3: Rewrite `build_user_message` (catalog-first)**

Open `src/agents/runtime/deepagents_agent.py`. Replace the body of `build_user_message` (currently lines ~317-399) with:

```python
    def build_user_message(
        self,
        task: Stage2TaskSpec,
        runtime: Stage2RuntimeState,
    ) -> HumanMessage:
        """Catalog-first multimodal task message (v9). BEV image + Cat-B text only."""
        bundle = runtime.bundle
        from agents.catalog import SceneCatalog

        extra = bundle.extra_metadata or {}
        catalog_raw = extra.get("scene_catalog")
        if catalog_raw is None:
            raise RuntimeError(
                "v9 build_user_message requires bundle.extra_metadata.scene_catalog; "
                "ensure pack prep wrote scene_catalog.json"
            )
        catalog = SceneCatalog(**catalog_raw)

        payload_schema = task.expected_output_schema or default_payload_schema(task.task_type)
        instruction = task.output_instruction or default_output_instruction(task.task_type)

        if task.task_type == Stage2TaskType.VISUAL_GROUNDING:
            view_note = (
                "- use view_bev(highlight=[ids]) to declutter\n"
                "- view_keyframe(mode='auto') resolves to 'marked' for VG"
            )
        elif task.task_type == Stage2TaskType.QA:
            view_note = (
                "- use view_bev(highlight=[ids]) to declutter\n"
                "- use view_keyframe(mode='rgb') for first-person scene observation"
            )
        else:
            view_note = "- use view_bev(highlight=[ids]) to declutter"

        by_cat = catalog.proposals_by_category()
        cat_lines: list[str] = []
        if by_cat:
            max_cat_len = max(len(c) for c in by_cat)
            for cat in sorted(by_cat):
                ids_str = ", ".join(f"#{pid}" for pid in sorted(by_cat[cat]))
                cat_lines.append(f"  {cat.ljust(max_cat_len)} : [{ids_str}]")
        cat_block = "\n".join(cat_lines) if cat_lines else "  (catalog is empty)"
        source = catalog.proposals[0].source if catalog.proposals else "n/a"

        prompt = (
            "## Task\n"
            f"Task type: {task.task_type.value}\n"
            f"Plan mode: {task.plan_mode.value}\n"
            f'User query: "{task.user_query}"\n'
            f"Output instruction: {instruction}\n\n"
            f"Expected payload schema:\n{json.dumps(payload_schema, indent=2, ensure_ascii=False)}\n\n"
            "## Scene\n"
            f"Scene id: {catalog.scene_id}\n"
            f"Scene category: {catalog.scene_category or 'unknown'}\n"
            f"Total frames: {catalog.total_frames} "
            f"(frame_id range: {list(catalog.frame_id_range)})\n"
            f"Proposal pool: {len(catalog.proposals)} items, source={source}\n\n"
            f"Proposals by category:\n{cat_block}\n\n"
            "## BEV image (attached above)\n"
            "- mesh-based top-down render with camera trajectory\n"
            "- each proposal labeled `#id category` at its 3D center\n"
            f"{view_note}\n\n"
            "## Available tools\n"
            "Always load_skill('scene-exploration-playbook') first.\n"
            "Then load_skill('vg-grounding-playbook') (VG) or load_skill('qa-answering-playbook') (QA).\n\n"
            f"You have viewed 0 keyframes out of {catalog.total_frames}. "
            "Use selectors + view_keyframe to fetch first-person frames."
        )

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in self.collect_image_paths(bundle):
            runtime.seen_image_paths.add(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self.image_to_data_url(image_path)},
                }
            )
        return HumanMessage(content=content)
```

- [ ] **Step 4: Update `collect_image_paths` to return BEV only**

In `src/agents/runtime/base.py`, replace `collect_image_paths` body (currently lines ~247-261) with:

```python
    def collect_image_paths(self, bundle: Stage2EvidenceBundle) -> list[str]:
        """v9 catalog-first: only the BEV image is part of the initial HumanMessage."""
        if bundle.bev_image_path and Path(bundle.bev_image_path).exists():
            return [str(bundle.bev_image_path)]
        return []
```

- [ ] **Step 5: Run task-message tests to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/runtime/tests/test_build_task_message_v9.py -q
```

Expected: 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/agents/runtime/base.py src/agents/runtime/deepagents_agent.py src/agents/runtime/tests/test_build_task_message_v9.py
git commit -m "$(cat <<'EOF'
refactor(prompt): catalog-first build_user_message (BEV + Cat-B text)

Removes Current keyframes / Stage-1 hypothesis sections; injects only the
BEV image plus a category->[#ids] inventory. QA gets a 'use view_keyframe
(mode=rgb)' hint, VG gets the marked default. collect_image_paths now
returns the BEV path only — no first-person seed.

Per spec Section D.
EOF
)"
```

### Task 21: Pack-prep — NR3D (write BEV + scene_catalog.json + camera trajectory)

**Files:**
- Modify: `src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py`
- Create: `src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d_v9.py`

- [ ] **Step 1: Inspect current NR3D prep entry point**

Read `src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py` lines 540-580 (the section that writes the per-sample JSON). Note variables: `request`, `keyframes`, `normalized_keyframes`, `scene_artifacts`, `query`.

- [ ] **Step 2: Write failing test for v9 prep additions**

Create `src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d_v9.py`:

```python
import json
from pathlib import Path

import pytest

from evaluation.scripts.prepare_pack_v1_inputs_nr3d import (
    write_v9_scene_artifacts,
)


def _write_dummy_proposals(scene_dir: Path) -> Path:
    p_path = scene_dir / "pack_nr3d_v1" / "proposals.jsonl"
    p_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "proposals": [
            {
                "id": 0,
                "bbox_3d": [0.0, 0.0, 0.0, 1, 1, 1, 0, 0, 0],
                "score": 0.9,
                "label": "chair",
                "frame_views": [
                    {
                        "frame_id": 10,
                        "bbox_2d": [0, 0, 50, 50],
                        "raw_rgb_path": str(scene_dir / "raw" / "000010-rgb.png"),
                    }
                ],
            }
        ]
    }
    p_path.write_text(json.dumps(payload))
    return p_path


def _write_traj(scene_dir: Path) -> None:
    cg = scene_dir / "conceptgraph"
    cg.mkdir(parents=True, exist_ok=True)
    traj = "\n".join(
        " ".join(map(str, row))
        for row in [
            [1.0, 0, 0, 0],
            [0, 1.0, 0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 0, 1.0],
        ]
    )
    (cg / "traj.txt").write_text(traj + "\n")
    (cg / "intrinsic_color.txt").write_text(
        "577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n"
    )


def test_write_v9_scene_artifacts_emits_catalog_and_trajectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scene_dir = tmp_path / "scene0000_00"
    _write_dummy_proposals(scene_dir)
    _write_traj(scene_dir)
    # Fake the BEV builder so we don't touch open3d / cv2 mesh rendering
    captured: dict = {}

    def _fake_bev(scene_id, data_root, proposals, output_path, highlight_ids):
        captured["scene_id"] = scene_id
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        return output_path

    monkeypatch.setattr(
        "evaluation.scripts.prepare_pack_v1_inputs_nr3d._render_v9_bev",
        _fake_bev,
    )
    paths = write_v9_scene_artifacts(
        scene_id="scene0000_00",
        data_root=tmp_path,
        pack_name="pack_nr3d_v9_catalog_first",
        proposals_jsonl=scene_dir / "pack_nr3d_v1" / "proposals.jsonl",
        scene_category="kitchen",
        valid_frame_ids=[10, 20, 30],
    )
    bev_path = paths["bev_image_path"]
    catalog_path = paths["scene_catalog_path"]
    assert Path(bev_path).exists()
    assert Path(catalog_path).exists()
    catalog = json.loads(Path(catalog_path).read_text())
    assert catalog["scene_id"] == "scene0000_00"
    assert catalog["scene_category"] == "kitchen"
    assert catalog["bev_image_path"].endswith(".png")
    assert "valid_frame_ids" in catalog
    traj = json.loads(Path(paths["camera_trajectory_path"]).read_text())
    assert "10" in traj or 10 in traj  # frame_id keyed
    assert captured["scene_id"] == "scene0000_00"
```

- [ ] **Step 3: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d_v9.py -q
```

Expected: `ImportError: cannot import name 'write_v9_scene_artifacts'`.

- [ ] **Step 4: Add `write_v9_scene_artifacts` to the NR3D prep**

Open `src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py`. At the top of the module (under existing imports), add:

```python
import math

from agents.catalog import from_vg_proposal_pool
from query_scene.scene_bev_builder import Nr3dScanNetBEVBuilder
```

At the bottom of the module, before `if __name__ == "__main__":`, add:

```python
def _render_v9_bev(
    scene_id: str,
    data_root: Path,
    proposals,
    output_path: Path,
    highlight_ids: list[int] | None,
) -> Path:
    builder = Nr3dScanNetBEVBuilder()
    return builder.build_with_labels(
        scene_id=scene_id,
        data_root=data_root,
        proposals=proposals,
        output_path=output_path,
        highlight_ids=highlight_ids,
    )


def _build_camera_trajectory(scene_dir: Path) -> dict[int, list[float]]:
    """Read conceptgraph/traj.txt and emit {frame_id: [x, y, yaw]}."""
    traj_path = scene_dir / "conceptgraph" / "traj.txt"
    if not traj_path.exists():
        raise FileNotFoundError(f"traj.txt missing for scene {scene_dir.name}: {traj_path}")
    raw = np.loadtxt(str(traj_path)).reshape(-1, 4, 4)
    out: dict[int, list[float]] = {}
    for i, pose in enumerate(raw):
        x = float(pose[0, 3])
        y = float(pose[1, 3])
        forward = -pose[:3, 2]
        yaw = float(math.atan2(forward[1], forward[0]))
        out[i] = [x, y, yaw]
    return out


def write_v9_scene_artifacts(
    *,
    scene_id: str,
    data_root: Path,
    pack_name: str,
    proposals_jsonl: Path,
    scene_category: str | None,
    valid_frame_ids: list[int],
) -> dict[str, str]:
    """Emit BEV png + scene_catalog.json + camera trajectory for v9 catalog-first NR3D prep."""
    scene_dir = data_root / scene_id
    pack_dir = scene_dir / pack_name
    bev_dir = pack_dir / "bev"
    bev_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = pack_dir / "scene_catalog.json"
    traj_out_path = pack_dir / "camera_trajectory.json"

    raw_pool = json.loads(proposals_jsonl.read_text())
    raw_pool.setdefault("source", "mask3d")
    raw_pool.setdefault("frame_index", {})
    raw_pool.setdefault("proposal_index", {})
    raw_pool.setdefault("annotated_image_dir", str(pack_dir / "annotated"))
    bev_path = bev_dir / "scene_bev_nr3d.png"
    catalog = from_vg_proposal_pool(
        pool=raw_pool,
        scene_id=scene_id,
        bev_image_path=str(bev_path),
        scene_category=scene_category,
        axis_align_matrix=None,
        valid_frame_ids=list(valid_frame_ids),
    )
    _render_v9_bev(
        scene_id=scene_id,
        data_root=data_root,
        proposals=catalog.proposals,
        output_path=bev_path,
        highlight_ids=None,
    )
    catalog_path.write_text(json.dumps(catalog.model_dump(), ensure_ascii=False, indent=2))
    traj = _build_camera_trajectory(scene_dir)
    traj_out_path.write_text(json.dumps(traj))
    return {
        "bev_image_path": str(bev_path),
        "scene_catalog_path": str(catalog_path),
        "camera_trajectory_path": str(traj_out_path),
    }
```

- [ ] **Step 5: Run test to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d_v9.py -q
```

Expected: 1 test passes.

- [ ] **Step 6: Wire `write_v9_scene_artifacts` into the sample loop**

In the same module, find the function that writes the per-sample JSON (the `_write_sample_payload`-like function around lines 552-575). Replace the payload assembly with:

```python
    artifacts_v9 = write_v9_scene_artifacts(
        scene_id=request.scene_id,
        data_root=data_root,
        pack_name=scene_artifacts.scene_dir.name,
        proposals_jsonl=scene_artifacts.proposals_jsonl,
        scene_category=None,
        valid_frame_ids=sorted(scene_artifacts.frame_visibility.keys()),
    )
    payload = {
        "sample_id": request.sample_id,
        "scene_id": request.scene_id,
        "target_id": request.target_id,
        "category": request.category or getattr(sample, "target", ""),
        "query": query,
        "gt_bbox_3d_9dof": gt_bbox,
        "scene_artifacts_dir": str(scene_artifacts.scene_dir),
        "source": "gt",
        "keyframe_mode": keyframe_mode,
        "keyframe_selection_uses_gt_target": uses_gt_target,
        "keyframe_selection_used_fallback": used_fallback,
        "scene_catalog_path": artifacts_v9["scene_catalog_path"],
        "bev_image_path": artifacts_v9["bev_image_path"],
        "camera_trajectory_path": artifacts_v9["camera_trajectory_path"],
    }
```

Delete the `normalize_prepared_keyframes` call (the imported function is unused after this change). Update the `from evaluation.scripts.prepare_pack_v1_inputs import normalize_prepared_keyframes` import — keep only the validators (`validate_bbox_9dof`, `validate_matrix_4x4`, `load_image_size`).

- [ ] **Step 7: Rerun NR3D prep tests + v9 test together**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d.py src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d_v9.py -q
```

Expected: v9 test passes; pre-existing NR3D tests pass except any that assert `payload["keyframes"]` (those will be moved into Task 31's test cleanup if any remain).

- [ ] **Step 8: Commit**

```bash
git add src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d_v9.py
git commit -m "$(cat <<'EOF'
feat(prep): NR3D pack writes BEV + scene_catalog + camera trajectory

Pack name flips to pack_nr3d_v9_catalog_first when --pack-name is set;
sample JSON now references scene_catalog_path / bev_image_path /
camera_trajectory_path. normalize_prepared_keyframes call removed.

Per spec Section B / E.
EOF
)"
```

### Task 22: Pack-prep — ScanRefer

**Files:**
- Modify: `src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py`
- Create: `src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer_v9.py`

- [ ] **Step 1: Write failing test**

Create `src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer_v9.py`:

```python
import json
from pathlib import Path

import pytest

from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import (
    write_v9_scene_artifacts_scanrefer,
)


def test_scanrefer_v9_emits_catalog_and_traj(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scene_dir = tmp_path / "scene0050_01"
    pack_dir = scene_dir / "pack_scanrefer_v9_catalog_first"
    pack_dir.mkdir(parents=True)
    cg = scene_dir / "conceptgraph"
    cg.mkdir(parents=True)
    (cg / "traj.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    (cg / "intrinsic_color.txt").write_text("577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n")
    proposals = pack_dir / "proposals.jsonl"
    proposals.write_text(json.dumps({
        "proposals": [
            {
                "id": 2,
                "bbox_3d": [0]*9,
                "score": 0.5,
                "label": "couch",
                "frame_views": [{"frame_id": 5, "bbox_2d": [0,0,10,10], "raw_rgb_path": "x.png"}]
            }
        ],
        "source": "mask3d",
    }))

    def _fake(scene_id, data_root, proposals, output_path, highlight_ids):
        output_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        return output_path

    monkeypatch.setattr(
        "evaluation.scripts.prepare_pack_v1_inputs_scanrefer._render_v9_bev_scanrefer",
        _fake,
    )
    paths = write_v9_scene_artifacts_scanrefer(
        scene_id="scene0050_01",
        data_root=tmp_path,
        pack_name="pack_scanrefer_v9_catalog_first",
        proposals_jsonl=proposals,
        scene_category=None,
        valid_frame_ids=[5],
    )
    catalog = json.loads(Path(paths["scene_catalog_path"]).read_text())
    assert catalog["scene_id"] == "scene0050_01"
    assert catalog["bev_image_path"].endswith(".png")
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer_v9.py -q
```

Expected: `ImportError`.

- [ ] **Step 3: Add `write_v9_scene_artifacts_scanrefer` mirroring NR3D**

Open `src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py`. Add at top:

```python
import math

from agents.catalog import from_vg_proposal_pool
from query_scene.scene_bev_builder import ScanReferScanNetBEVBuilder
```

At the bottom (before any `if __name__`):

```python
def _render_v9_bev_scanrefer(scene_id, data_root, proposals, output_path, highlight_ids):
    builder = ScanReferScanNetBEVBuilder()
    return builder.build_with_labels(
        scene_id=scene_id,
        data_root=data_root,
        proposals=proposals,
        output_path=output_path,
        highlight_ids=highlight_ids,
    )


def _build_camera_trajectory_scanrefer(scene_dir):
    traj_path = scene_dir / "conceptgraph" / "traj.txt"
    if not traj_path.exists():
        raise FileNotFoundError(f"traj.txt missing for scene {scene_dir.name}: {traj_path}")
    raw = np.loadtxt(str(traj_path)).reshape(-1, 4, 4)
    out: dict[int, list[float]] = {}
    for i, pose in enumerate(raw):
        x = float(pose[0, 3])
        y = float(pose[1, 3])
        forward = -pose[:3, 2]
        yaw = float(math.atan2(forward[1], forward[0]))
        out[i] = [x, y, yaw]
    return out


def write_v9_scene_artifacts_scanrefer(
    *,
    scene_id,
    data_root,
    pack_name,
    proposals_jsonl,
    scene_category,
    valid_frame_ids,
):
    scene_dir = data_root / scene_id
    pack_dir = scene_dir / pack_name
    bev_dir = pack_dir / "bev"
    bev_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = pack_dir / "scene_catalog.json"
    traj_out_path = pack_dir / "camera_trajectory.json"
    raw_pool = json.loads(proposals_jsonl.read_text())
    raw_pool.setdefault("source", "mask3d")
    raw_pool.setdefault("frame_index", {})
    raw_pool.setdefault("proposal_index", {})
    raw_pool.setdefault("annotated_image_dir", str(pack_dir / "annotated"))
    bev_path = bev_dir / "scene_bev_scanrefer.png"
    catalog = from_vg_proposal_pool(
        pool=raw_pool,
        scene_id=scene_id,
        bev_image_path=str(bev_path),
        scene_category=scene_category,
        axis_align_matrix=None,
        valid_frame_ids=list(valid_frame_ids),
    )
    _render_v9_bev_scanrefer(
        scene_id=scene_id,
        data_root=data_root,
        proposals=catalog.proposals,
        output_path=bev_path,
        highlight_ids=None,
    )
    catalog_path.write_text(json.dumps(catalog.model_dump(), ensure_ascii=False, indent=2))
    traj = _build_camera_trajectory_scanrefer(scene_dir)
    traj_out_path.write_text(json.dumps(traj))
    return {
        "bev_image_path": str(bev_path),
        "scene_catalog_path": str(catalog_path),
        "camera_trajectory_path": str(traj_out_path),
    }
```

Then in the sample loop, replace the existing `normalize_prepared_keyframes` + payload construction with the same `write_v9_scene_artifacts_scanrefer` call pattern from Task 21 (mutatis mutandis: `pack_scanrefer_v9_catalog_first`).

- [ ] **Step 4: Run tests to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer_v9.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer_v9.py
git commit -m "feat(prep): ScanRefer pack writes BEV + scene_catalog + trajectory"
```

### Task 23: Pack-prep — OpenEQA (new `prepare_pack_qa_inputs.py`, OpenEQA flavor)

**Files:**
- Create: `src/evaluation/scripts/prepare_pack_qa_inputs.py`
- Create: `src/evaluation/scripts/tests/test_prepare_pack_qa_inputs.py`

The QA prep accepts a benchmark flag so the same script handles OpenEQA and SQA3D in Tasks 23-24. This task wires the OpenEQA branch.

- [ ] **Step 1: Write failing test for OpenEQA flavor**

Create `src/evaluation/scripts/tests/test_prepare_pack_qa_inputs.py`:

```python
import gzip
import json
import pickle
from pathlib import Path

import pytest

from evaluation.scripts.prepare_pack_qa_inputs import write_qa_scene_artifacts


def _make_clip_layout(tmp_path: Path) -> Path:
    clip = tmp_path / "002-scannet-scene0709_00"
    cg = clip / "conceptgraph"
    raw = clip / "raw"
    cg.mkdir(parents=True)
    raw.mkdir(parents=True)
    (cg / "traj.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    (raw / "intrinsic_color.txt").write_text("577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n")
    (cg / "scene_info.json").write_text(json.dumps({"scan_id": "scene0709_00"}))
    pcd = cg / "pcd_saves"
    pcd.mkdir()
    with gzip.open(pcd / "full_pcd_v9.pkl.gz", "wb") as fh:
        pickle.dump(
            {"objects": [
                {"id": 0, "category": "chair", "bbox_3d_9dof": [0]*9},
                {"id": 1, "category": "table", "bbox_3d_9dof": [1]*9},
            ]},
            fh,
        )
    det = cg / "gsa_detections_ram_withbg_allclasses"
    det.mkdir()
    return clip


def test_write_qa_scene_artifacts_openeqa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    clip = _make_clip_layout(tmp_path)

    def _fake_render(scene_id, data_root, proposals, output_path, highlight_ids):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        return output_path

    monkeypatch.setattr(
        "evaluation.scripts.prepare_pack_qa_inputs._render_qa_bev",
        _fake_render,
    )
    paths = write_qa_scene_artifacts(
        benchmark="openeqa",
        clip_id="002-scannet-scene0709_00",
        data_root=tmp_path,
        pack_name="pack_openeqa_v9_catalog_first",
        view_to_objects={5: [(0, 0.9)], 7: [(1, 0.8)]},
        valid_frame_ids=[5, 7],
        scene_category="bedroom",
    )
    catalog = json.loads(Path(paths["scene_catalog_path"]).read_text())
    assert catalog["scene_id"] == "002-scannet-scene0709_00"
    assert catalog["scene_category"] == "bedroom"
    assert {p["proposal_id"] for p in catalog["proposals"]} == {0, 1}
    assert Path(paths["bev_image_path"]).exists()
    assert Path(paths["camera_trajectory_path"]).exists()
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/evaluation/scripts/tests/test_prepare_pack_qa_inputs.py -q
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `prepare_pack_qa_inputs.py` (OpenEQA + SQA3D shared module)**

Create `src/evaluation/scripts/prepare_pack_qa_inputs.py`:

```python
"""Pack-v1 prep for QA benchmarks (OpenEQA / SQA3D) under the v9 catalog-first design."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from agents.catalog import from_conceptgraph_objects
from query_scene.scene_bev_builder import (
    OpenEqaScanNetBEVBuilder,
    Sqa3dScanNetBEVBuilder,
)


def _render_qa_bev(scene_id: str, data_root: Path, proposals, output_path: Path, highlight_ids):
    benchmark = _BENCHMARK_FOR_BUILDER  # type: ignore[name-defined]
    builder = (
        OpenEqaScanNetBEVBuilder() if benchmark == "openeqa" else Sqa3dScanNetBEVBuilder()
    )
    return builder.build_with_labels(
        scene_id=scene_id,
        data_root=data_root,
        proposals=proposals,
        output_path=output_path,
        highlight_ids=highlight_ids,
    )


_BENCHMARK_FOR_BUILDER: str = "openeqa"


def _build_camera_trajectory_qa(scene_dir: Path) -> dict[int, list[float]]:
    traj_path = scene_dir / "conceptgraph" / "traj.txt"
    if not traj_path.exists():
        raise FileNotFoundError(f"traj.txt missing: {traj_path}")
    raw = np.loadtxt(str(traj_path)).reshape(-1, 4, 4)
    out: dict[int, list[float]] = {}
    for i, pose in enumerate(raw):
        x = float(pose[0, 3])
        y = float(pose[1, 3])
        forward = -pose[:3, 2]
        yaw = float(math.atan2(forward[1], forward[0]))
        out[i] = [x, y, yaw]
    return out


def write_qa_scene_artifacts(
    *,
    benchmark: str,
    clip_id: str,
    data_root: Path,
    pack_name: str,
    view_to_objects: dict[int, list[tuple[int, float]]],
    valid_frame_ids: list[int],
    scene_category: str | None,
) -> dict[str, str]:
    """Emit BEV + scene_catalog.json + camera trajectory for QA clip."""
    if benchmark not in ("openeqa", "sqa3d"):
        raise ValueError(f"unsupported QA benchmark: {benchmark!r}")
    global _BENCHMARK_FOR_BUILDER
    _BENCHMARK_FOR_BUILDER = benchmark
    clip_dir = data_root / clip_id
    pack_dir = clip_dir / pack_name
    bev_dir = pack_dir / "bev"
    bev_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = pack_dir / "scene_catalog.json"
    traj_out_path = pack_dir / "camera_trajectory.json"
    raw_rgb_template = str(clip_dir / "raw" / "{frame_id:06d}-rgb.png")
    bev_path = bev_dir / f"scene_bev_{benchmark}.png"
    catalog = from_conceptgraph_objects(
        pcd_saves_dir=clip_dir / "conceptgraph" / "pcd_saves",
        detections_dir=clip_dir / "conceptgraph" / "gsa_detections_ram_withbg_allclasses",
        view_to_objects=view_to_objects,
        scene_id=clip_id,
        bev_image_path=str(bev_path),
        scene_category=scene_category,
        valid_frame_ids=list(valid_frame_ids),
        raw_rgb_template=raw_rgb_template,
    )
    _render_qa_bev(
        scene_id=clip_id,
        data_root=data_root,
        proposals=catalog.proposals,
        output_path=bev_path,
        highlight_ids=None,
    )
    catalog_path.write_text(json.dumps(catalog.model_dump(), ensure_ascii=False, indent=2))
    traj = _build_camera_trajectory_qa(clip_dir)
    traj_out_path.write_text(json.dumps(traj))
    return {
        "bev_image_path": str(bev_path),
        "scene_catalog_path": str(catalog_path),
        "camera_trajectory_path": str(traj_out_path),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=("openeqa", "sqa3d"), required=True)
    p.add_argument("--sample-ids", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--pack-name", default="pack_openeqa_v9_catalog_first")
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


__all__ = ["write_qa_scene_artifacts", "parse_args"]
```

- [ ] **Step 4: Run tests to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/evaluation/scripts/tests/test_prepare_pack_qa_inputs.py -q
```

Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/scripts/prepare_pack_qa_inputs.py src/evaluation/scripts/tests/test_prepare_pack_qa_inputs.py
git commit -m "feat(prep): add prepare_pack_qa_inputs (OpenEQA branch wired)"
```

### Task 24: Pack-prep — SQA3D branch + sample-loop wiring

**Files:**
- Modify: `src/evaluation/scripts/prepare_pack_qa_inputs.py`
- Modify: `src/evaluation/scripts/tests/test_prepare_pack_qa_inputs.py`

- [ ] **Step 1: Extend the test suite**

Append to `src/evaluation/scripts/tests/test_prepare_pack_qa_inputs.py`:

```python
def test_write_qa_scene_artifacts_sqa3d(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """SQA3D layout: data_root/<scene_id> (no clip prefix), same conceptgraph subdir."""
    scene_dir = tmp_path / "scene0050_00"
    cg = scene_dir / "conceptgraph"
    raw = scene_dir / "raw"
    cg.mkdir(parents=True)
    raw.mkdir(parents=True)
    (cg / "traj.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    (raw / "intrinsic_color.txt").write_text("577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n")
    pcd = cg / "pcd_saves"
    pcd.mkdir()
    with gzip.open(pcd / "full_pcd_v9.pkl.gz", "wb") as fh:
        pickle.dump(
            {"objects": [{"id": 3, "category": "sofa", "bbox_3d_9dof": [0]*9}]},
            fh,
        )
    det = cg / "gsa_detections_ram_withbg_allclasses"
    det.mkdir()
    monkeypatch.setattr(
        "evaluation.scripts.prepare_pack_qa_inputs._render_qa_bev",
        lambda **kw: kw["output_path"].write_bytes(b"\x89PNG\r\n\x1a\n") or kw["output_path"],
    )
    paths = write_qa_scene_artifacts(
        benchmark="sqa3d",
        clip_id="scene0050_00",
        data_root=tmp_path,
        pack_name="pack_sqa3d_v9_catalog_first",
        view_to_objects={12: [(3, 1.0)]},
        valid_frame_ids=[12],
        scene_category=None,
    )
    catalog = json.loads(Path(paths["scene_catalog_path"]).read_text())
    assert catalog["scene_id"] == "scene0050_00"
    assert catalog["proposals"][0]["proposal_id"] == 3


def test_write_qa_scene_artifacts_rejects_unknown_benchmark(tmp_path: Path):
    with pytest.raises(ValueError, match="unsupported QA benchmark"):
        write_qa_scene_artifacts(
            benchmark="bogus",
            clip_id="x",
            data_root=tmp_path,
            pack_name="p",
            view_to_objects={},
            valid_frame_ids=[1],
            scene_category=None,
        )
```

- [ ] **Step 2: Run tests to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/evaluation/scripts/tests/test_prepare_pack_qa_inputs.py -q
```

The `test_write_qa_scene_artifacts_sqa3d` test invokes `_render_qa_bev` via a kwargs monkeypatch that does not match the keyword-only signature in Task 23. Update `_render_qa_bev` to accept kwargs by changing its signature to:

```python
def _render_qa_bev(*, scene_id, data_root, proposals, output_path, highlight_ids):
```

…and update all internal callers accordingly.

- [ ] **Step 3: Re-run tests; verify all 3 in this file pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/evaluation/scripts/tests/test_prepare_pack_qa_inputs.py -q
```

Expected: 3 tests pass.

- [ ] **Step 4: Add a CLI driver that loops over a sample list (used by Phase 8)**

Append to `src/evaluation/scripts/prepare_pack_qa_inputs.py`:

```python
def main() -> int:
    args = parse_args()
    sample_lines = args.sample_ids.read_text().splitlines()
    if args.max_samples:
        sample_lines = sample_lines[: args.max_samples]
    written: list[Path] = []
    for sample_line in sample_lines:
        clip_id, _, _ = sample_line.partition("|")
        view_index = json.loads(
            (args.data_root / clip_id / "conceptgraph" / "view_to_objects.json").read_text()
        )
        view_to_objects = {
            int(k): [(int(o), float(w)) for o, w in v] for k, v in view_index.items()
        }
        valid_frame_ids = sorted(view_to_objects.keys())
        artifacts = write_qa_scene_artifacts(
            benchmark=args.benchmark,
            clip_id=clip_id,
            data_root=args.data_root,
            pack_name=args.pack_name,
            view_to_objects=view_to_objects,
            valid_frame_ids=valid_frame_ids,
            scene_category=None,
        )
        sample_path = args.data_root / clip_id / args.pack_name / "samples" / f"{clip_id}.json"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(
            json.dumps(
                {
                    "sample_id": clip_id,
                    "scene_id": clip_id,
                    "scene_catalog_path": artifacts["scene_catalog_path"],
                    "bev_image_path": artifacts["bev_image_path"],
                    "camera_trajectory_path": artifacts["camera_trajectory_path"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        written.append(sample_path)
    print(f"wrote {len(written)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/scripts/prepare_pack_qa_inputs.py src/evaluation/scripts/tests/test_prepare_pack_qa_inputs.py
git commit -m "feat(prep): wire SQA3D + main driver into prepare_pack_qa_inputs"
```

---

## Phase 6 — Playbook Rewrites

### Task 25: New `scene_exploration_playbook.md` (shared gate)

**Files:**
- Create: `src/agents/skills/shared_skills/scene_exploration_playbook.md`
- Create: `src/agents/skills/tests/test_scene_exploration_playbook_loadable.py`

- [ ] **Step 1: Write failing test**

Create `src/agents/skills/tests/test_scene_exploration_playbook_loadable.py`:

```python
from pathlib import Path

from agents.runtime.skill_registry import load_skill_markdown


def test_scene_exploration_playbook_exists_and_lists_six_selectors():
    path = Path("src/agents/skills/shared_skills/scene_exploration_playbook.md")
    text = path.read_text()
    for tool in (
        "select_by_proposal",
        "select_by_frame_neighbor",
        "select_by_region",
        "select_by_coverage",
        "select_by_text",
        "select_by_hypothesis",
        "view_keyframe",
        "view_bev",
        "list_scene_proposals",
        "list_frame_proposals",
        "inspect_proposal",
    ):
        assert tool in text, f"{tool} missing from scene_exploration_playbook"


def test_scene_exploration_playbook_loads_via_registry():
    md = load_skill_markdown("scene-exploration-playbook")
    assert "Cheapest-first" in md
    assert "BEV" in md


def test_scene_exploration_playbook_does_not_reference_deleted_tools():
    text = Path("src/agents/skills/shared_skills/scene_exploration_playbook.md").read_text()
    for dead in (
        "request_more_views",
        "switch_or_expand_hypothesis",
        "find_proposals_by_category",
        "list_keyframes_with_proposals",
        "inspect_stage1_metadata",
        "view_keyframe_marked",
    ):
        assert dead not in text, f"deleted tool {dead} still referenced"
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/skills/tests/test_scene_exploration_playbook_loadable.py -q
```

Expected: `FileNotFoundError` on the first test.

- [ ] **Step 3: Create the playbook**

Create `src/agents/skills/shared_skills/scene_exploration_playbook.md`:

```markdown
# Scene Exploration Playbook (shared, prerequisite)

You start each scene with two views:

1. A BEV (top-down) image with every `#id category` label drawn at the
   proposal's 3D centre, plus the camera trajectory.
2. A `SceneCatalog` text inventory (`Proposals by category: chair: [#4, #5, ...]`).

Neither is first-person evidence. **You must fetch first-person frames before
finalising any answer that depends on appearance, count, state, or relations.**

## Mental model

- BEV + catalog = scene structure (what / where / how many).
- view_keyframe = ground truth appearance / colour / state / view-dep relations.
- The scene contains N frames (see "Total frames" in the task message). You
  have viewed 0. Plan a small fetching budget (typically 3-8 frames).

## Selector cheat sheet — cheapest first

| Selector | Cost | When to use |
| --- | --- | --- |
| `select_by_proposal(proposal_ids=[#a,#b])` | instant | You already know the catalog IDs. Returns frames containing them. |
| `select_by_frame_neighbor(anchor_frame_id, mode='temporal'\|'viewpoint_diverse')` | instant | You found one good frame, want neighbours or alternate angles. |
| `select_by_region(region, region_type='bev_2d'\|'bbox_3d')` | instant | You see a BEV cluster; expand to all frames overlooking that region. |
| `select_by_coverage(method='obj_iou'\|'pose_depth', seen_frame_ids?)` | cheap | You need diverse coverage of the room. |
| `select_by_text(query)` | ~Stage-1 LLM (2-5 s) | Catalog IDs don't help (e.g. attribute search). |
| `select_by_hypothesis(hypothesis_json)` | instant | You authored a hypothesis dict explicitly. |

**Default order:** try proposal/frame_neighbor/region/coverage first; reach for
text/hypothesis when none of those map onto the query.

## Per-frame inspection

- `view_keyframe(frame_id, mode='auto')` resolves to:
  - `mode='marked'` for visual grounding (boxes overlaid on proposals).
  - `mode='rgb'` for QA / generic perception.
- Use `categories=...` or `proposal_ids=...` to narrow what's drawn in marked mode.
- `list_frame_proposals(frame_id)` returns a text inventory of the same frame
  without injecting an image (cheap dry-run).

## BEV inspection

- `view_bev()` re-injects the original BEV.
- `view_bev(highlight=[#a, #b])` re-renders with only those proposals labelled
  (declutter trick — use this when the original BEV is too dense to read).

## Catalog queries

- `list_scene_proposals(category=?, region_bev=?, limit=?)` — scene-wide.
- `inspect_proposal(proposal_id)` — proposal metadata + frames_appeared.

## Anti-patterns

- Answering from BEV labels alone. The BEV is a map, not ground truth.
- Calling `select_by_text` first when proposal IDs are visible on the BEV.
- Viewing more than 12 frames before consulting `submit_final`. Budget pressure
  is real; trim aggressively.
- Using `request_crops` before `view_keyframe`. Crops are a zoom; the host frame
  must already be in context.

**Cheapest-first** mantra: **proposal → neighbour → region → coverage → text → hypothesis**.
```

- [ ] **Step 4: Register the skill in `shared_skills/__init__.py`**

Open `src/agents/skills/shared_skills/__init__.py`. Append to the existing `_SHARED_SKILLS` mapping (or equivalent registry):

```python
"scene-exploration-playbook": _here / "scene_exploration_playbook.md",
```

If the file does not import this name yet, add the entry in the same dict where `"task-head-output"` is registered (find that string with `rg "task-head-output" src/agents/skills`).

- [ ] **Step 5: Run tests to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/skills/tests/test_scene_exploration_playbook_loadable.py -q
```

Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/agents/skills/shared_skills/scene_exploration_playbook.md src/agents/skills/shared_skills/__init__.py src/agents/skills/tests/test_scene_exploration_playbook_loadable.py
git commit -m "$(cat <<'EOF'
feat(skills): add scene_exploration_playbook shared gate skill

New skill is the prerequisite gate for all v9 selectors + view tools.
Cheapest-first selector ladder + per-frame / BEV / catalog cheat sheet.

Per spec Section F.3.
EOF
)"
```

### Task 26: Rewrite `vg_grounding_playbook.md`

**Files:**
- Modify: `src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md`
- Modify: `src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation.md`
- Create: `src/agents/packs/vg_embodiedscan/skills/tests/test_playbook_v9_consistency.py`

- [ ] **Step 1: Write failing test**

Create `src/agents/packs/vg_embodiedscan/skills/tests/test_playbook_v9_consistency.py`:

```python
from pathlib import Path

import pytest


PB = Path("src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md")
SD = Path("src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation.md")


@pytest.mark.parametrize("path", [PB, SD])
@pytest.mark.parametrize(
    "dead",
    [
        "request_more_views",
        "switch_or_expand_hypothesis",
        "find_proposals_by_category",
        "list_keyframes_with_proposals",
        "inspect_stage1_metadata",
        "view_keyframe_marked",
    ],
)
def test_no_dead_tool_names(path: Path, dead: str):
    assert dead not in path.read_text(), f"{dead} still mentioned in {path}"


def test_vg_playbook_lists_v9_tools():
    text = PB.read_text()
    for tool in (
        "view_keyframe",
        "list_frame_proposals",
        "list_scene_proposals",
        "inspect_proposal",
        "select_by_proposal",
        "view_bev",
    ):
        assert tool in text


def test_vg_playbook_mentions_tadg_and_guards():
    text = PB.read_text()
    assert "TADG" in text
    assert "no_match_guard" in text
    assert "evidence_frame_guard" in text


def test_vg_playbook_mentions_ood_proposal_minus_one():
    text = PB.read_text()
    assert "proposal_id" in text
    assert "-1" in text


def test_vg_spatial_disambiguation_uses_view_keyframe_mode_marked():
    text = SD.read_text()
    assert "view_keyframe(mode='marked')" in text or 'mode="marked"' in text
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/packs/vg_embodiedscan/skills/tests/test_playbook_v9_consistency.py -q
```

Expected: many failures (current playbook still mentions `request_more_views`, `view_keyframe_marked`, etc.).

- [ ] **Step 3: Rewrite `vg_grounding_playbook.md`**

Replace `src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md` with:

```markdown
# VG Grounding Playbook (v9)

Prerequisite: `load_skill('scene-exploration-playbook')` first.

You are grounding a natural-language referring expression to **one** proposal in
the SceneCatalog. The output must be a single `proposal_id` (or `-1` if the target
is genuinely absent from the catalog — OOD case).

## Standard flow

1. **Read the BEV image** and the `Proposals by category:` block in the task
   message. Identify candidate `#id`s by category (e.g. "brown chair" → all
   `chair` proposals).
2. **Cheap filter** with `select_by_proposal(proposal_ids=[candidate ids])`
   to obtain the frames that contain any candidate.
3. **View the frames** with `view_keyframe(frame_id, mode='marked', proposal_ids=[…])`.
   Marked mode draws boxes labelled `#id` on the candidate proposals; visually
   verify which one matches the query.
4. **Disambiguate spatial relations** with `compare_proposals_spatial(
   candidate_ids=[…], anchor_id=#x, relation='left_of'|'right_of'|'closer_to'|…)`
   when the query involves a spatial relation.
5. **Verify edge attributes** (colour, material, state) with
   `request_crops(request_text='…', object_terms=[…])` only when a frame view is
   ambiguous.
6. **Submit** with `submit_final(payload={"proposal_id": <id>}, …)`.

## Tools at a glance

- `select_by_proposal(proposal_ids, require_all=False, k=8)` — instant catalog lookup.
- `select_by_frame_neighbor(anchor_frame_id, mode='temporal'|'viewpoint_diverse')`
  — expand around a good frame.
- `select_by_region(region, region_type='bev_2d'|'bbox_3d')` — region filter.
- `select_by_coverage(method='obj_iou'|'pose_depth')` — diverse coverage.
- `select_by_text(query)` — fallback when category labels don't narrow enough.
- `view_keyframe(frame_id, mode='auto')` — auto → 'marked' for VG.
- `list_frame_proposals(frame_id)` — text-only proposals on a frame.
- `list_scene_proposals(category=?, region_bev=?)` — scene-wide inventory.
- `inspect_proposal(proposal_id)` — frames_appeared + position + bbox.
- `view_bev(highlight=[ids])` — re-render BEV with subset only.

## Guards (read this before submit)

- **TADG** (Target-Anchor Disambiguation Guard): if the query mentions an
  anchor (e.g. "next to the kitchen counter"), TADG will block submission
  unless `compare_proposals_spatial` proved the chosen proposal satisfies the
  relation. If TADG fires, run the comparison and resubmit.
- **no_match_guard**: if `select_by_*` and `view_keyframe` did not surface any
  candidate of the queried category, submit `proposal_id=-1` (OOD).
- **evidence_frame_guard**: at least one frame visible with the chosen
  proposal must appear in the trace under `view_keyframe(mode='marked')` —
  unmarked RGB views do not count as VG evidence.

## OOD policy

If after exhaustive selectors + ≥3 view_keyframe calls you cannot locate the
referent, submit `proposal_id=-1` with a rationale explaining what categories /
regions you searched. Do not invent an `id`.

## Anti-patterns

- Submitting based on BEV labels alone (no `view_keyframe`).
- Using `select_by_text` when the BEV already shows obvious category labels.
- Forgetting `mode='marked'` and trying to recognise objects from raw RGB.
- Querying `request_crops` before any `view_keyframe`.
```

- [ ] **Step 4: Small update to `vg_spatial_disambiguation.md`**

Open `src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation.md`. Find any reference to `view_keyframe_marked` and replace with `view_keyframe(mode='marked')`. If the file does not exist, create a minimal one:

```markdown
# VG Spatial Disambiguation (v9)

Used to resolve "left of / right of / closer to / on / inside" between proposals.

Workflow:
1. Identify the candidate proposals (`#a`, `#b`) and the anchor (`#x`).
2. Call `compare_proposals_spatial(candidate_ids=[#a, #b], anchor_id=#x, relation='<rel>')`.
3. Inspect the result; verify visually with `view_keyframe(mode='marked', proposal_ids=[#a, #b, #x])`.
4. Submit the winning candidate.

Use `select_by_region(region_type='bbox_3d')` to fetch frames that show both the
anchor and the candidates simultaneously.
```

- [ ] **Step 5: Run tests to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/packs/vg_embodiedscan/skills/tests/test_playbook_v9_consistency.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation.md src/agents/packs/vg_embodiedscan/skills/tests/test_playbook_v9_consistency.py
git commit -m "$(cat <<'EOF'
refactor(skills): rewrite vg_grounding_playbook for v9 catalog-first

Drops request_more_views / switch_or_expand_hypothesis / view_keyframe_marked
references. Adds catalog-first flow + cheapest-first selector ladder + TADG /
no_match_guard / evidence_frame_guard notes + OOD policy.

Per spec Section F.3.
EOF
)"
```

### Task 27: Rewrite `qa_answering_playbook.md`

**Files:**
- Modify: `src/agents/packs/qa_default/skills/qa_answering_playbook.md`
- Create: `src/agents/packs/qa_default/skills/tests/test_qa_playbook_v9_consistency.py`

- [ ] **Step 1: Write failing test**

Create `src/agents/packs/qa_default/skills/tests/test_qa_playbook_v9_consistency.py`:

```python
from pathlib import Path

import pytest


PB = Path("src/agents/packs/qa_default/skills/qa_answering_playbook.md")


@pytest.mark.parametrize(
    "dead",
    [
        "request_more_views",
        "switch_or_expand_hypothesis",
        "find_proposals_by_category",
        "list_keyframes_with_proposals",
        "inspect_stage1_metadata",
        "view_keyframe_marked",
        "retrieve_object_context",
    ],
)
def test_no_dead_tool_names(dead: str):
    assert dead not in PB.read_text(), f"{dead} still in qa_answering_playbook"


def test_qa_playbook_recommends_rgb_mode():
    text = PB.read_text()
    assert "view_keyframe(frame_id, mode='rgb')" in text or 'mode="rgb"' in text


def test_qa_playbook_mentions_supporting_claims():
    text = PB.read_text()
    assert "supporting_claims" in text


def test_qa_playbook_mentions_select_by_text_for_attributes():
    text = PB.read_text()
    assert "select_by_text" in text
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/packs/qa_default/skills/tests/test_qa_playbook_v9_consistency.py -q
```

Expected: failures (existing playbook still references `request_more_views`, etc.).

- [ ] **Step 3: Rewrite `qa_answering_playbook.md`**

Replace `src/agents/packs/qa_default/skills/qa_answering_playbook.md` with:

```markdown
# QA Answering Playbook (v9)

Prerequisite: `load_skill('scene-exploration-playbook')` first.

You are answering an embodied-QA question. Your final payload follows the
`Stage2QAResult` schema (free-form `answer` + optional `supporting_claims`).

## Question taxonomy & default flow

| Question kind | Default approach |
| --- | --- |
| "what is in / count of X" | `select_by_text(query='X')` → `view_keyframe(mode='rgb')` |
| "what colour / material / state of X" | `select_by_proposal([#X])` → `view_keyframe(mode='rgb')` → `request_crops` if ambiguous |
| "where is X relative to Y" | `compare_proposals_spatial(candidate_ids=[#X], anchor_id=#Y, …)` |
| "is X there / are there any" | BEV inspection → `select_by_text` → `view_keyframe(mode='rgb')` |
| open-ended description | `select_by_coverage(method='pose_depth', k=4)` → multiple `view_keyframe` |

## Step-by-step

1. Read the BEV image and the `Proposals by category:` block.
2. Pick the cheapest selector matching the question kind (see table above).
3. View the returned frames with `view_keyframe(frame_id, mode='rgb')` —
   raw RGB is preferred for QA so that visual cues (colour, occupancy, state)
   are unobstructed by mask boxes.
4. For fine attributes (small text on objects, colour patch, gauge readings),
   call `request_crops(request_text=…, object_terms=[…])`.
5. Compose the answer in natural language. Populate `supporting_claims` with
   `{frame_id: int, proposal_ids: [#a, #b], note: "…"}` for each piece of
   evidence you used.

## Counting questions

- Trust the catalog Cat-B count if and only if the user query category matches a
  catalog category exactly (e.g. "how many chairs" → count proposals in
  `chair`).
- Otherwise verify with at least one `view_keyframe(mode='rgb')` and one
  `list_frame_proposals(frame_id)` confirmation.

## When to use `select_by_text`

Use it when:
- the question references appearance (colour, material) not in catalog labels;
- the question references a state ("is the door open") that catalog labels
  cannot express;
- proposal/region/neighbour selectors come back empty.

## Anti-patterns

- Answering from catalog text without viewing any RGB frame.
- Using `view_keyframe(mode='marked')` for QA — masks cover visual evidence.
- Composing `supporting_claims` with frame_ids you did not call `view_keyframe`
  on (the evidence_frame_guard will catch this and force a re-run).
```

- [ ] **Step 4: Run tests to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/packs/qa_default/skills/tests/test_qa_playbook_v9_consistency.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agents/packs/qa_default/skills/qa_answering_playbook.md src/agents/packs/qa_default/skills/tests/test_qa_playbook_v9_consistency.py
git commit -m "$(cat <<'EOF'
refactor(skills): rewrite qa_answering_playbook for v9 catalog-first

Drops request_more_views / retrieve_object_context / inspect_stage1_metadata.
Adds question-kind table, select_by_text guidance, supporting_claims rules,
and an anti-pattern about marked-mode for QA.

Per spec Section F.3.
EOF
)"
```

### Task 28: Delete obsolete `evidence_scouting.md` (×2)

**Files:**
- Delete: `src/agents/packs/vg_embodiedscan/skills/evidence_scouting.md`
- Delete: `src/agents/packs/qa_default/skills/evidence_scouting.md`
- Modify: `src/agents/packs/vg_embodiedscan/skills/__init__.py` (drop registration)
- Modify: `src/agents/packs/qa_default/skills/__init__.py` (drop registration)
- Create: `src/agents/skills/tests/test_no_evidence_scouting.py`

- [ ] **Step 1: Write failing test**

Create `src/agents/skills/tests/test_no_evidence_scouting.py`:

```python
from pathlib import Path


def test_no_evidence_scouting_skill_files():
    assert not Path("src/agents/packs/vg_embodiedscan/skills/evidence_scouting.md").exists()
    assert not Path("src/agents/packs/qa_default/skills/evidence_scouting.md").exists()


def test_no_evidence_scouting_registered():
    from agents.packs.vg_embodiedscan.skills import SKILL_REGISTRY as vg_reg
    from agents.packs.qa_default.skills import SKILL_REGISTRY as qa_reg
    assert "evidence-scouting" not in vg_reg
    assert "evidence-scouting" not in qa_reg
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/skills/tests/test_no_evidence_scouting.py -q
```

Expected: failures (files still exist; the import names may differ — adjust to whatever symbol the pack uses to expose the registry, e.g. `_SKILL_FILES`).

- [ ] **Step 3: Delete the files**

```bash
git rm src/agents/packs/vg_embodiedscan/skills/evidence_scouting.md
git rm src/agents/packs/qa_default/skills/evidence_scouting.md
```

- [ ] **Step 4: Remove registry entries**

Open `src/agents/packs/vg_embodiedscan/skills/__init__.py`. Find the dict mapping skill names → markdown paths (look for `"evidence-scouting"` or `evidence_scouting.md`); delete that single entry. Repeat for `src/agents/packs/qa_default/skills/__init__.py`.

If the test in Step 1 still uses the wrong symbol, fix the test to import the actual registry name (run `rg -n "evidence_scouting" src/agents/packs/*/skills/__init__.py` first to confirm the symbol).

- [ ] **Step 5: Run tests to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/skills/tests/test_no_evidence_scouting.py -q
```

Expected: 2 tests pass.

- [ ] **Step 6: Commit**

```bash
git add -A src/agents/packs/vg_embodiedscan/skills src/agents/packs/qa_default/skills src/agents/skills/tests/test_no_evidence_scouting.py
git commit -m "$(cat <<'EOF'
chore(skills): delete obsolete evidence_scouting playbooks

Replaced by scene_exploration_playbook (shared) + per-pack playbooks.

Per spec Section F.3.
EOF
)"
```

---

## Phase 7 — Migration Cleanup

### Task 29: Delete `request_more_views` + `switch_or_expand_hypothesis` (runtime wrappers + callbacks)

**Files:**
- Modify: `src/agents/runtime/deepagents_agent.py` (drop tool wrappers + builder paths)
- Modify: `src/agents/stage1_callbacks.py` (drop `create_more_views_callback` + `create_hypothesis_callback`)
- Modify: `src/agents/tools/hypothesis_repair.py` (if still imports old callback names — remove)
- Create: `src/agents/tests/test_migration_no_callback_tools.py`

- [ ] **Step 1: Write failing test**

Create `src/agents/tests/test_migration_no_callback_tools.py`:

```python
import importlib

import pytest


def test_runtime_does_not_expose_request_more_views():
    from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime
    rt = DeepAgentsStage2Runtime(config=__import__("agents.core.agent_config", fromlist=["Stage2DeepAgentConfig"]).Stage2DeepAgentConfig())
    for attr in (
        "_request_more_views_impl",
        "_create_request_more_views_tool",
        "_create_switch_or_expand_hypothesis_tool",
        "_create_inspect_stage1_metadata_tool",
    ):
        assert not hasattr(rt, attr), f"{attr} should be removed"


def test_stage1_callbacks_drops_old_factories():
    mod = importlib.import_module("agents.stage1_callbacks")
    assert not hasattr(mod, "create_more_views_callback")
    assert not hasattr(mod, "create_hypothesis_callback")
    assert hasattr(mod, "create_crop_callback")


def test_runtime_tool_list_does_not_include_dead_names():
    from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime
    from agents.core.agent_config import Stage2DeepAgentConfig

    rt = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    names = {t.name for t in rt._build_tools()}
    for dead in (
        "request_more_views",
        "switch_or_expand_hypothesis",
        "inspect_stage1_metadata",
        "list_keyframes_with_proposals",
        "find_proposals_by_category",
        "view_keyframe_marked",
    ):
        assert dead not in names, f"{dead} should not be wired"
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tests/test_migration_no_callback_tools.py -q
```

Expected: failures because `_request_more_views_impl`, `create_more_views_callback`, etc. still exist.

- [ ] **Step 3: Strip wrappers from `deepagents_agent.py`**

Open `src/agents/runtime/deepagents_agent.py`. Use `rg -n "_request_more_views_impl\|switch_or_expand_hypothesis\|inspect_stage1_metadata" src/agents/runtime/deepagents_agent.py` to locate the methods. Delete:

- The `_request_more_views_impl` method (≈80 lines).
- The closure tool factory that wraps it (look for `@tool ... def request_more_views`).
- The `switch_or_expand_hypothesis` tool factory.
- The `inspect_stage1_metadata` tool factory.
- Any registration of these in `_build_tools` (replace the loop entries with nothing — leave only the remaining tool calls).

Also delete the `tool_descriptions` lines that mention `request_more_views` / `switch_or_expand_hypothesis` / `inspect_stage1_metadata`.

- [ ] **Step 4: Strip the callback factories from `stage1_callbacks.py`**

Open `src/agents/stage1_callbacks.py`. Delete the two functions:

```python
def create_more_views_callback(...): ...
def create_hypothesis_callback(...): ...
```

Keep `create_crop_callback`. If `__all__` is defined, remove the two names from it.

- [ ] **Step 5: Patch `hypothesis_repair.py`**

Open `src/agents/tools/hypothesis_repair.py`. If it imports `create_hypothesis_callback` or references `switch_or_expand_hypothesis`, delete those lines. If the file becomes empty / only exports nothing, delete it with `git rm`.

- [ ] **Step 6: Run tests to verify pass**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tests/test_migration_no_callback_tools.py -q
```

Expected: 3 tests pass.

- [ ] **Step 7: Run broader regression sanity**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tests -q -k "not test_more_views_callback and not test_stage2_deep_agent"
```

Expected: many existing tests still fail (they reference old names) — they will be fixed in Task 33.

- [ ] **Step 8: Commit**

```bash
git add src/agents/runtime/deepagents_agent.py src/agents/stage1_callbacks.py src/agents/tools/hypothesis_repair.py src/agents/tests/test_migration_no_callback_tools.py
git commit -m "$(cat <<'EOF'
refactor: delete request_more_views + switch_or_expand_hypothesis + inspect_stage1_metadata

Runtime wrappers, callback factories, and tool registrations removed.
create_crop_callback is preserved (still needed by request_crops).

Per spec Section E (M1 hard cutover).
EOF
)"
```

### Task 30: Delete `find_proposals_by_category` / `list_keyframes_with_proposals` from pack tools

**Files:**
- Modify: `src/agents/packs/vg_embodiedscan/tools.py`
- Modify: `src/agents/packs/vg_embodiedscan/ctx.py` (drop attributes only these tools used)
- Modify: `src/agents/packs/vg_embodiedscan/tests/test_tools.py` (remove their tests)

- [ ] **Step 1: Write failing assertion test**

Append to `src/agents/tests/test_migration_no_callback_tools.py`:

```python
def test_pack_tools_module_no_longer_defines_dead_helpers():
    import agents.packs.vg_embodiedscan.tools as mod
    for dead in (
        "find_proposals_by_category",
        "list_keyframes_with_proposals",
    ):
        assert not hasattr(mod, dead), f"{dead} still exported"
```

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tests/test_migration_no_callback_tools.py::test_pack_tools_module_no_longer_defines_dead_helpers -q
```

Expected: fail.

- [ ] **Step 2: Remove the two tool functions**

Open `src/agents/packs/vg_embodiedscan/tools.py`. Find `def find_proposals_by_category(` and `def list_keyframes_with_proposals(` (each preceded by an `@tool` decorator). Delete the entire decorator + function body for each. Also delete `_format_proposals_by_category` and any helper used only by them.

If a registration list at the bottom (e.g. `VG_TOOLS = [ ... ]`) references these names, remove the entries.

- [ ] **Step 3: Trim `ctx.py`**

Open `src/agents/packs/vg_embodiedscan/ctx.py`. If there are accessors named `_get_category_index` or similar used only by the removed tools, delete them. Use `rg -n "_get_category_index\|_index_by_category" src/agents/packs/vg_embodiedscan` to verify.

- [ ] **Step 4: Remove their tests**

Open `src/agents/packs/vg_embodiedscan/tests/test_tools.py`. Delete any test function whose body references the now-removed names. Use `rg -n "find_proposals_by_category\|list_keyframes_with_proposals" src/agents/packs/vg_embodiedscan/tests/test_tools.py` to find them; expect 4-8 tests to remove.

- [ ] **Step 5: Run tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tests/test_migration_no_callback_tools.py::test_pack_tools_module_no_longer_defines_dead_helpers src/agents/packs/vg_embodiedscan/tests/test_tools.py -q
```

Expected: the targeted test passes; the rest of `test_tools.py` should still pass since other tools are untouched.

- [ ] **Step 6: Commit**

```bash
git add src/agents/packs/vg_embodiedscan/tools.py src/agents/packs/vg_embodiedscan/ctx.py src/agents/packs/vg_embodiedscan/tests/test_tools.py src/agents/tests/test_migration_no_callback_tools.py
git commit -m "refactor: delete find_proposals_by_category + list_keyframes_with_proposals"
```

### Task 31: Delete `enable_temporal_fan` / `enable_stage1_callback` config flags

**Files:**
- Modify: `src/agents/core/agent_config.py`
- Modify: any reader of those flags (`rg -n "enable_temporal_fan\|enable_stage1_callback" src/`)
- Create: `src/agents/core/tests/test_config_no_dead_flags.py`

- [ ] **Step 1: Write failing test**

Create `src/agents/core/tests/test_config_no_dead_flags.py`:

```python
from agents.core.agent_config import Stage2DeepAgentConfig


def test_config_does_not_expose_dead_flags():
    cfg = Stage2DeepAgentConfig()
    assert not hasattr(cfg, "enable_temporal_fan")
    assert not hasattr(cfg, "enable_stage1_callback")


def test_config_does_not_accept_dead_flag_kwargs():
    import pytest

    with pytest.raises((TypeError, ValueError)):
        Stage2DeepAgentConfig(enable_temporal_fan=True)
    with pytest.raises((TypeError, ValueError)):
        Stage2DeepAgentConfig(enable_stage1_callback=True)
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/core/tests/test_config_no_dead_flags.py -q
```

Expected: fail.

- [ ] **Step 3: Remove the fields**

Open `src/agents/core/agent_config.py`. Locate the two attributes:

```python
    enable_temporal_fan: bool = ...
    enable_stage1_callback: bool = ...
```

Delete each line and any docstring lines that reference them. If `Stage2DeepAgentConfig` is a Pydantic `BaseModel` (`model_config = ConfigDict(extra="forbid")` or similar), this is enough — extra kwargs will raise.

- [ ] **Step 4: Update readers**

```bash
rg -n "enable_temporal_fan|enable_stage1_callback" src/ | cut -d: -f1 | sort -u
```

For each file:

- In `src/agents/runtime/deepagents_agent.py`: delete branches that gate the temporal-fan prompt section or callback wiring.
- In `src/agents/runtime/base.py`: delete `temporal_fan_line` assembly.
- In any test that constructed `Stage2DeepAgentConfig(enable_temporal_fan=False)`: delete the kwarg.

- [ ] **Step 5: Run tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/core/tests/test_config_no_dead_flags.py -q
```

Expected: 2 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/agents/core/agent_config.py src/agents/runtime/deepagents_agent.py src/agents/runtime/base.py src/agents/core/tests/test_config_no_dead_flags.py
git commit -m "$(cat <<'EOF'
refactor: delete enable_temporal_fan + enable_stage1_callback config flags

v9 tool set is static; no need for runtime feature flags. Reader sites
in deepagents_agent / base prompt builder updated accordingly.

Per spec Section E.
EOF
)"
```

### Task 32: Adapt `no_match_guard` + `evidence_frame_guard` to new tool names

**Files:**
- Modify: `src/agents/skills/no_match_guard.py`
- Modify: `src/agents/skills/evidence_frame_guard.py`
- Modify: `src/agents/tests/test_no_match_guard.py`
- Modify: `src/agents/tests/test_evidence_frame_guard.py`
- Modify: `src/agents/skills/tadg.py` (verify no dead refs)

- [ ] **Step 1: Read current guard implementations**

```bash
rg -n "find_proposals_by_category|list_keyframes_with_proposals|view_keyframe_marked|request_more_views" src/agents/skills/
```

Note exact line numbers. Typical state: `no_match_guard.py` reads the tool_trace for `view_keyframe_marked` and `find_proposals_by_category`; `evidence_frame_guard.py` reads for `view_keyframe_marked`.

- [ ] **Step 2: Add a failing test for the new name in no_match_guard**

Append to `src/agents/tests/test_no_match_guard.py`:

```python
def test_no_match_guard_accepts_v9_tool_names():
    from agents.skills.no_match_guard import NoMatchGuard
    g = NoMatchGuard(target_category="chair")
    trace = [
        {"tool": "list_scene_proposals", "input": {"category": "chair"}, "output": "(none)"},
        {"tool": "view_keyframe", "input": {"frame_id": 10, "mode": "marked", "categories": ["chair"]}, "output": "ok"},
    ]
    assert g.evaluate(tool_trace=trace, payload={"proposal_id": -1}) is None  # no violation
```

Run to verify failure (`NameError`/old tool name expected):

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tests/test_no_match_guard.py::test_no_match_guard_accepts_v9_tool_names -q
```

- [ ] **Step 3: Update `no_match_guard.py`**

Open `src/agents/skills/no_match_guard.py`. Use Find/Replace:

- `"find_proposals_by_category"` → `"list_scene_proposals"`
- `"view_keyframe_marked"` → `"view_keyframe"` (and in any helper that checks `mode == 'marked'`, change to inspect `tool_call.input.get("mode")` equals `"marked"` or `"auto"` with task_type VG).
- `"list_keyframes_with_proposals"` → drop those branches (the tool no longer exists; the catalog now exposes per-proposal frames via `inspect_proposal`).

Also remove imports / constants referencing the old names.

- [ ] **Step 4: Add a failing test for evidence_frame_guard**

Append to `src/agents/tests/test_evidence_frame_guard.py`:

```python
def test_evidence_frame_guard_recognizes_v9_view_keyframe_marked_mode():
    from agents.skills.evidence_frame_guard import EvidenceFrameGuard
    g = EvidenceFrameGuard()
    trace = [
        {"tool": "view_keyframe", "input": {"frame_id": 7, "mode": "marked"}, "output": "ok"},
    ]
    assert g.evaluate(
        tool_trace=trace,
        payload={"proposal_id": 3},
        evidence_refs=[{"frame_id": 7, "proposal_ids": [3]}],
    ) is None
```

Run to verify failure:

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tests/test_evidence_frame_guard.py::test_evidence_frame_guard_recognizes_v9_view_keyframe_marked_mode -q
```

- [ ] **Step 5: Update `evidence_frame_guard.py`**

Open `src/agents/skills/evidence_frame_guard.py`. Replace the predicate that filters trace entries by `tool == "view_keyframe_marked"` with:

```python
def _is_marked_view(entry: dict) -> bool:
    if entry.get("tool") != "view_keyframe":
        return False
    mode = (entry.get("input") or {}).get("mode")
    return mode in ("marked", "auto")
```

Apply this in the existing scan logic.

- [ ] **Step 6: Verify `tadg.py` is clean**

```bash
rg -n "view_keyframe_marked|find_proposals_by_category|list_keyframes_with_proposals" src/agents/skills/tadg.py
```

If any matches: replace `view_keyframe_marked` → `view_keyframe`. `tadg.py` should otherwise be untouched (it primarily uses `compare_proposals_spatial` which is unchanged).

- [ ] **Step 7: Run guard tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tests/test_no_match_guard.py src/agents/tests/test_evidence_frame_guard.py src/agents/tests/test_tadg.py -q
```

Expected: new tests pass; old tests need their tool-name strings updated (replace `view_keyframe_marked` → `view_keyframe` with `mode='marked'` everywhere in those test fixtures). Do these in-file rewrites now.

- [ ] **Step 8: Commit**

```bash
git add src/agents/skills/no_match_guard.py src/agents/skills/evidence_frame_guard.py src/agents/skills/tadg.py src/agents/tests/test_no_match_guard.py src/agents/tests/test_evidence_frame_guard.py src/agents/tests/test_tadg.py
git commit -m "$(cat <<'EOF'
refactor(guards): rewire no_match_guard + evidence_frame_guard to v9 names

Both guards now consume view_keyframe (mode='marked'/'auto') instead of
view_keyframe_marked, and no_match_guard accepts list_scene_proposals as
the catalog query. Fixture strings updated accordingly.

Per spec Section F.4.
EOF
)"
```

### Task 33: Fix all impacted tests (Stage 2 deep agent + benchmark integration + side_by_side scripts)

**Files:**
- Modify: `src/agents/tests/test_stage2_deep_agent.py`
- Modify: `src/agents/tests/test_benchmark_integration.py`
- Modify: `src/agents/tests/test_openeqa_official_question_pilot.py`
- Modify: `src/agents/tests/test_derive_eval_session_id.py`
- Delete: `src/agents/examples/test_more_views_callback.py`
- Modify: `src/agents/examples/openeqa_single_scene_pilot.py`
- Modify: `src/agents/examples/openeqa_official_question_pilot.py`
- Modify: `src/agents/examples/e2e_stage2_test.py`
- Modify: any `src/evaluation/scripts/run_*_side_by_side.py` that wires the deleted callbacks

- [ ] **Step 1: Survey breakage**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tests -x --no-header -q 2>&1 | head -60
```

Expected: collection errors / assertion failures, mostly on missing names.

- [ ] **Step 2: Replace prompt assertions in `test_stage2_deep_agent.py`**

Open `src/agents/tests/test_stage2_deep_agent.py`. For every assertion of the form `assert "request_more_views" in prompt` or similar, either:

- delete the assertion if it tested a piece of the v7/v8 system prompt that no longer exists, **or**
- replace with the equivalent v9 assertion (`assert "select_by_proposal" in prompt`, `assert "scene-exploration-playbook" in prompt`, etc.).

Specifically:

- Replace `"request_more_views"` checks → `"select_by_text"` or `"view_keyframe"`.
- Replace `"switch_or_expand_hypothesis"` checks → `"select_by_hypothesis"`.
- Replace `"view_keyframe_marked"` checks → `"view_keyframe"` (and where the test asserts mode, check the prompt contains `"mode='marked'"`).
- Replace any setup that constructed `Stage2DeepAgentConfig(enable_temporal_fan=True)` — drop the kwarg.
- Replace bundle setup that wrote `extra_metadata["proposal_pool"]` → write `extra_metadata["scene_catalog"]` instead (use the `_catalog()` helper from Task 20's test as a reference).

- [ ] **Step 3: Delete the obsolete example test**

```bash
git rm src/agents/examples/test_more_views_callback.py
```

- [ ] **Step 4: Patch example pilots**

Open `src/agents/examples/openeqa_single_scene_pilot.py`, `openeqa_official_question_pilot.py`, `e2e_stage2_test.py`. For each:

- Remove imports of `create_more_views_callback` / `create_hypothesis_callback`.
- Remove the lines that pass `more_views_callback=...` / `hypothesis_callback=...` to the runtime constructor.
- Keep `create_crop_callback` wiring.

- [ ] **Step 5: Patch benchmark integration test**

Open `src/agents/tests/test_benchmark_integration.py`. For each `request_more_views` / `switch_or_expand_hypothesis` reference in fixtures or in trace assertions, replace with `view_keyframe` / `select_by_proposal` equivalents. If a test entirely depended on stubbing the dead callbacks, remove that test.

- [ ] **Step 6: Patch the side-by-side scripts**

```bash
rg -ln "create_more_views_callback|create_hypothesis_callback" src/evaluation/scripts/
```

For each file:

- Delete the import of those two factories.
- Delete the two lines that wire them into the runtime.
- Keep `create_crop_callback`.

- [ ] **Step 7: Run full Stage-2 test suite**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tests src/agents/packs src/agents/skills/tests src/agents/runtime/tests src/agents/tools/tests src/agents/catalog/tests -q
```

Expected: green.

- [ ] **Step 8: Commit**

```bash
git add -A src/agents src/evaluation/scripts
git commit -m "$(cat <<'EOF'
test: update Stage-2 tests + example pilots for v9 tool surface

- test_stage2_deep_agent: catalog-first prompt + selector names
- benchmark_integration / openeqa_official_question_pilot fixtures
  rewritten to write scene_catalog into extra_metadata
- delete test_more_views_callback example
- side_by_side scripts drop more_views / hypothesis callback wiring

Per spec Section F.4.
EOF
)"
```

---

## Phase 8 — Verification + Integration

### Task 34: F.5 verification grep (assert no dead-tool references remain)

**Files:**
- Create: `scripts/verify_v9_no_dead_refs.sh`
- Create: `src/agents/tests/test_verify_no_dead_refs.py`

- [ ] **Step 1: Author the verification shell script**

Create `scripts/verify_v9_no_dead_refs.sh`:

```bash
#!/usr/bin/env bash
# F.5 verification: assert no v9-deleted tool names linger in src/ or docs/
# (excluding spec / plan / migration docs which intentionally name dead tools).

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

FAIL=0
DEAD_NAMES=(
    'request_more_views'
    'switch_or_expand_hypothesis'
    'find_proposals_by_category'
    'list_keyframes_with_proposals'
    'inspect_stage1_metadata'
    'enable_temporal_fan'
    'enable_stage1_callback'
)
ALLOWLIST_GLOB='!docs/superpowers/specs/2026-05-14-v9-catalog-first-scene-exploration-design.md !docs/superpowers/plans/2026-05-14-v9-catalog-first-scene-exploration.md !docs/benchmark/**/*.md'

for name in "${DEAD_NAMES[@]}"; do
    matches=$(rg --no-messages -l "$name" src/ docs/ \
        --glob '!docs/superpowers/specs/2026-05-14-v9-catalog-first-scene-exploration-design.md' \
        --glob '!docs/superpowers/plans/2026-05-14-v9-catalog-first-scene-exploration.md' \
        --glob '!docs/benchmark/**' || true)
    if [[ -n "$matches" ]]; then
        echo "FAIL: dead name '$name' found in:"
        echo "$matches" | sed 's/^/  /'
        FAIL=1
    fi
done

# view_keyframe_marked is allowed only in v9 spec/plan/migration docs.
matches=$(rg --no-messages -l 'view_keyframe_marked' src/ \
    --glob '!docs/superpowers/specs/2026-05-14-v9-catalog-first-scene-exploration-design.md' \
    --glob '!docs/superpowers/plans/2026-05-14-v9-catalog-first-scene-exploration.md' \
    --glob '!docs/benchmark/**' || true)
if [[ -n "$matches" ]]; then
    echo "FAIL: view_keyframe_marked found in production code:"
    echo "$matches" | sed 's/^/  /'
    FAIL=1
fi

if [[ $FAIL -eq 0 ]]; then
    echo "PASS: no v9-deleted tool names found"
fi
exit $FAIL
```

```bash
chmod +x scripts/verify_v9_no_dead_refs.sh
```

- [ ] **Step 2: Wrap the script in a pytest assertion**

Create `src/agents/tests/test_verify_no_dead_refs.py`:

```python
import subprocess


def test_no_dead_tool_references():
    res = subprocess.run(
        ["bash", "scripts/verify_v9_no_dead_refs.sh"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"verify_v9_no_dead_refs.sh failed:\n{res.stdout}\n{res.stderr}"
```

- [ ] **Step 3: Run the test**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tests/test_verify_no_dead_refs.py -q
```

Expected: pass after Phase 7 commits. If it fails, fix the offending file inline before committing.

- [ ] **Step 4: Commit**

```bash
git add scripts/verify_v9_no_dead_refs.sh src/agents/tests/test_verify_no_dead_refs.py
git commit -m "$(cat <<'EOF'
ci: add F.5 verification grep for v9 dead-tool names

scripts/verify_v9_no_dead_refs.sh asserts that no v9-deleted tool names
linger in src/ (allowing only the v9 spec / plan docs to mention them).

Per spec Section F.5.
EOF
)"
```

### Task 35: End-to-end mock VLM test — VG flow

**Files:**
- Create: `src/agents/tests/integration/test_v9_vg_end_to_end_mock.py`

- [ ] **Step 1: Write the integration test**

Create `src/agents/tests/integration/test_v9_vg_end_to_end_mock.py`:

```python
"""End-to-end VG flow with a mock VLM client.

Validates the v9 catalog-first contract: build_user_message injects only
BEV; selectors / view_keyframe expand evidence; submit_final terminates
with a proposal_id grounded in the trace.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from PIL import Image

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.core.agent_config import (
    Stage2DeepAgentConfig,
    Stage2PlanMode,
    Stage2TaskType,
)
from agents.core.task_types import Stage2EvidenceBundle, Stage2TaskSpec


def _make_scene(tmp_path: Path) -> Stage2EvidenceBundle:
    bev = tmp_path / "bev.png"
    Image.new("RGB", (256, 256), (12, 12, 12)).save(bev)
    rgb_dir = tmp_path / "raw"
    rgb_dir.mkdir()
    for fid in (5, 10):
        Image.new("RGB", (320, 240), (200, 200, 200)).save(rgb_dir / f"{fid:06d}-rgb.png")
    catalog = SceneCatalog(
        scene_id="scene0123_45",
        scene_category="kitchen",
        proposals=[
            SceneProposal(
                proposal_id=7,
                category="chair",
                position_3d=(0.0, 0.0, 0.4),
                bbox_3d_9dof=(0, 0, 0.4, 0.5, 0.5, 0.8, 0, 0, 0),
                frame_views=[
                    FrameView(
                        frame_id=5,
                        bbox_2d=(10, 10, 60, 60),
                        raw_rgb_path=str(rgb_dir / "000005-rgb.png"),
                    ),
                ],
                source="mask3d",
            ),
            SceneProposal(
                proposal_id=8,
                category="chair",
                position_3d=(1.5, 0.0, 0.4),
                bbox_3d_9dof=(1.5, 0, 0.4, 0.5, 0.5, 0.8, 0, 0, 0),
                frame_views=[
                    FrameView(
                        frame_id=10,
                        bbox_2d=(80, 30, 140, 110),
                        raw_rgb_path=str(rgb_dir / "000010-rgb.png"),
                    ),
                ],
                source="mask3d",
            ),
        ],
        total_frames=12,
        frame_id_range=(0, 110),
        valid_frame_ids=[5, 10],
        bev_image_path=str(bev),
    )
    return Stage2EvidenceBundle(
        scene_id=catalog.scene_id,
        extra_metadata={"scene_catalog": catalog.model_dump()},
        bev_image_path=str(bev),
    )


class _MockVLM:
    """Returns a scripted sequence of AIMessage tool_calls then a final answer."""

    def __init__(self, script: list[dict | str]):
        self._script = list(script)

    def invoke(self, messages):
        item = self._script.pop(0)
        if isinstance(item, str):
            return AIMessage(content=item)
        return AIMessage(content="", tool_calls=[item])


def test_vg_end_to_end_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bundle = _make_scene(tmp_path)
    task = Stage2TaskSpec(
        user_query="this is a brown chair",
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        plan_mode=Stage2PlanMode.BRIEF,
        max_reasoning_turns=8,
    )
    from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime

    runtime = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    mock = _MockVLM(
        [
            {"name": "load_skill", "args": {"name": "scene-exploration-playbook"}, "id": "1"},
            {"name": "load_skill", "args": {"name": "vg-grounding-playbook"}, "id": "2"},
            {"name": "select_by_proposal", "args": {"proposal_ids": [7, 8], "k": 4}, "id": "3"},
            {"name": "view_keyframe", "args": {"frame_id": 5, "mode": "auto", "proposal_ids": [7]}, "id": "4"},
            {"name": "submit_final", "args": {"payload": {"proposal_id": 7}, "rationale": "Frame 5 marked chair #7"}, "id": "5"},
        ]
    )
    monkeypatch.setattr(runtime, "_invoke_llm", mock.invoke)

    result = runtime.run(task, bundle)

    assert result.payload == {"proposal_id": 7}
    tool_names = [c["name"] for c in result.tool_trace]
    assert "select_by_proposal" in tool_names
    assert "view_keyframe" in tool_names
    # No callback-era tools used:
    assert not any(n in tool_names for n in ("request_more_views", "switch_or_expand_hypothesis"))
```

- [ ] **Step 2: Run the test**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tests/integration/test_v9_vg_end_to_end_mock.py -q
```

If `DeepAgentsStage2Runtime._invoke_llm` does not exist with that exact name, adapt the monkeypatch target to the actual private method used by `run()` (look it up with `rg -n "_invoke_llm\|invoke.*model" src/agents/runtime/deepagents_agent.py`). Make the test green.

- [ ] **Step 3: Commit**

```bash
git add src/agents/tests/integration/test_v9_vg_end_to_end_mock.py
git commit -m "test(integration): v9 VG end-to-end with mock VLM"
```

### Task 36: End-to-end mock VLM test — QA flow

**Files:**
- Create: `src/agents/tests/integration/test_v9_qa_end_to_end_mock.py`

- [ ] **Step 1: Write the integration test**

Create `src/agents/tests/integration/test_v9_qa_end_to_end_mock.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from PIL import Image

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.core.agent_config import (
    Stage2DeepAgentConfig,
    Stage2PlanMode,
    Stage2TaskType,
)
from agents.core.task_types import Stage2EvidenceBundle, Stage2TaskSpec


def _scene_for_qa(tmp_path: Path) -> Stage2EvidenceBundle:
    bev = tmp_path / "bev.png"
    Image.new("RGB", (256, 256), (10, 10, 10)).save(bev)
    rgb_dir = tmp_path / "raw"
    rgb_dir.mkdir()
    Image.new("RGB", (320, 240), (255, 255, 255)).save(rgb_dir / "000020-rgb.png")
    catalog = SceneCatalog(
        scene_id="002-scannet-scene0709_00",
        scene_category="bedroom",
        proposals=[
            SceneProposal(
                proposal_id=0,
                category="bed",
                position_3d=(0, 0, 0.3),
                frame_views=[
                    FrameView(frame_id=20, raw_rgb_path=str(rgb_dir / "000020-rgb.png")),
                ],
                source="conceptgraph",
            ),
        ],
        total_frames=30,
        frame_id_range=(0, 290),
        valid_frame_ids=[20],
        bev_image_path=str(bev),
    )
    return Stage2EvidenceBundle(
        scene_id=catalog.scene_id,
        extra_metadata={"scene_catalog": catalog.model_dump()},
        bev_image_path=str(bev),
    )


class _MockVLM:
    def __init__(self, script):
        self._script = list(script)

    def invoke(self, _msgs):
        item = self._script.pop(0)
        return AIMessage(content="", tool_calls=[item])


def test_qa_end_to_end_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bundle = _scene_for_qa(tmp_path)
    task = Stage2TaskSpec(
        user_query="what colour is the bed?",
        task_type=Stage2TaskType.QA,
        plan_mode=Stage2PlanMode.BRIEF,
        max_reasoning_turns=8,
    )
    from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime

    runtime = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    monkeypatch.setattr(
        runtime,
        "_invoke_llm",
        _MockVLM(
            [
                {"name": "load_skill", "args": {"name": "scene-exploration-playbook"}, "id": "1"},
                {"name": "load_skill", "args": {"name": "qa-answering-playbook"}, "id": "2"},
                {"name": "select_by_text", "args": {"query": "bed colour", "k": 2}, "id": "3"},
                {"name": "view_keyframe", "args": {"frame_id": 20, "mode": "rgb"}, "id": "4"},
                {
                    "name": "submit_final",
                    "args": {
                        "payload": {
                            "answer": "white",
                            "supporting_claims": [
                                {"frame_id": 20, "proposal_ids": [0], "note": "white sheets visible"}
                            ],
                        },
                        "rationale": "RGB frame 20 confirms colour.",
                    },
                    "id": "5",
                },
            ]
        ).invoke,
    )

    result = runtime.run(task, bundle)
    assert result.payload["answer"] == "white"
    names = [c["name"] for c in result.tool_trace]
    assert "select_by_text" in names
    assert "view_keyframe" in names
    # QA default uses mode='rgb' from the playbook:
    view_call = next(c for c in result.tool_trace if c["name"] == "view_keyframe")
    assert view_call["args"]["mode"] == "rgb"
```

- [ ] **Step 2: Run and commit**

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tests/integration/test_v9_qa_end_to_end_mock.py -q
git add src/agents/tests/integration/test_v9_qa_end_to_end_mock.py
git commit -m "test(integration): v9 QA end-to-end with mock VLM"
```

### Task 37: Sanity-check pack prep on a real NR3D scene

**Files:**
- Create: `scripts/sanity_check_v9_pack_nr3d.sh`

- [ ] **Step 1: Author the smoke-test script**

Create `scripts/sanity_check_v9_pack_nr3d.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

SCENE_ID="${1:-scene0011_00}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/scannet}"
PACK_NAME="pack_nr3d_v9_catalog_first"

if [[ ! -d "$DATA_ROOT/$SCENE_ID" ]]; then
    echo "Scene dir missing: $DATA_ROOT/$SCENE_ID" >&2
    exit 2
fi

source .venv/bin/activate
PYTHONPATH=src python -m evaluation.scripts.prepare_pack_v1_inputs_nr3d \
    --scene-id "$SCENE_ID" \
    --data-root "$DATA_ROOT" \
    --pack-name "$PACK_NAME" \
    --max-samples 1

pack_dir="$DATA_ROOT/$SCENE_ID/$PACK_NAME"
echo "Checking artifacts in $pack_dir"
test -f "$pack_dir/scene_catalog.json" || { echo "FAIL: no scene_catalog.json"; exit 1; }
test -f "$pack_dir/camera_trajectory.json" || { echo "FAIL: no camera_trajectory.json"; exit 1; }
test -f "$pack_dir/bev/scene_bev_nr3d.png" || { echo "FAIL: no BEV png"; exit 1; }

PYTHONPATH=src python -c "
import json, sys
from agents.catalog import SceneCatalog
data = json.load(open('$pack_dir/scene_catalog.json'))
cat = SceneCatalog(**data)
assert cat.scene_id == '$SCENE_ID'
assert cat.proposals, 'empty proposals'
assert cat.bev_image_path.endswith('.png')
print('OK: catalog parses, scene_id=%s, proposals=%d' % (cat.scene_id, len(cat.proposals)))
"
echo "PASS: NR3D v9 pack sanity check"
```

```bash
chmod +x scripts/sanity_check_v9_pack_nr3d.sh
```

- [ ] **Step 2: Document expected manual invocation**

The script is `--max-samples 1` so it runs in <30s. To execute (operator step, not part of CI):

```bash
DATA_ROOT=/path/to/scannet bash scripts/sanity_check_v9_pack_nr3d.sh scene0011_00
```

- [ ] **Step 3: Commit**

```bash
git add scripts/sanity_check_v9_pack_nr3d.sh
git commit -m "chore: add NR3D v9 pack sanity-check script (smoke test)"
```

### Task 38: Run v9_full NR3D random100 and document results

**Files:**
- Create: `scripts/run_v9_full_nr3d_random100.sh`
- Create: `docs/benchmark/nr3d/v9_catalog_first_<YYYYMMDD>.md`
- Modify: `docs/benchmark/nr3d/README.md` (timeline row)
- Modify: `docs/benchmark/nr3d/leaderboard.md` (entry)

- [ ] **Step 1: Author the launcher (uses tmux per CLAUDE.md mandatory rule)**

Create `scripts/run_v9_full_nr3d_random100.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

TAG="${1:-v9_full}"
DATE="$(date +%Y%m%d_%H%M)"
OUT_DIR="tmp/nr3d_eval_${TAG}_${DATE}"
LOG="/tmp/nr3d_${TAG}_${DATE}.log"
SESS="nr3d-${TAG}-${DATE}"
FROZEN_FOLD="${FROZEN_FOLD:-tmp/nr3d_artifacts/random100_frozen.json}"

mkdir -p "$OUT_DIR"

tmux new-session -d -s "$SESS" "
set -euo pipefail
source .venv/bin/activate
PYTHONPATH=src python -m evaluation.scripts.run_nr3d_v9_random100 \
    --output-dir '$OUT_DIR' \
    --pack-name pack_nr3d_v9_catalog_first \
    --force-selection '$FROZEN_FOLD' \
    --judge-model gemini-2.5-pro \
    2>&1 | tee '$LOG'
"
echo "Launched in tmux session: $SESS"
echo "Tail logs: tmux capture-pane -t $SESS -p -S -100"
echo "Output dir: $OUT_DIR"
```

```bash
chmod +x scripts/run_v9_full_nr3d_random100.sh
```

- [ ] **Step 2: Reuse the existing random100 driver via `--pack-name`**

The repo already has `src/evaluation/scripts/run_nr3d_random100.py` (the driver that produced the v8 and v9_selective_mark numbers in the spec). It already accepts a `--pack-name` flag — verify with:

```bash
rg -n 'add_argument\("--pack-name"' src/evaluation/scripts/run_nr3d_random100.py
```

Expected: one match. Therefore the launcher in Step 1 invokes `run_nr3d_random100` (not a new module). If for any reason the flag is missing on this branch, add it inline:

```python
parser.add_argument("--pack-name", default="pack_nr3d_v1", help="pack directory under each scene")
```

and thread it through to `prepare_pack_v1_inputs_nr3d` / sample loading. Commit that as a separate one-liner before launching the eval.

- [ ] **Step 3: Create the benchmark version doc skeleton**

Create `docs/benchmark/nr3d/v9_catalog_first_2026MMDD.md` (replace `MMDD` with today's date when run is harvested):

```markdown
# NR3D — v9 catalog-first (random100 fold)

- **Branch**: feat/v9-catalog-first-scene-exploration
- **Tip commit**: TBD (fill in when run starts)
- **Pack**: pack_nr3d_v9_catalog_first
- **Fold**: random100 frozen at tmp/nr3d_artifacts/random100_frozen.json
- **Judge**: gemini-2.5-pro
- **Launcher**: `scripts/run_v9_full_nr3d_random100.sh`
- **Raw artifacts**: tmp/nr3d_eval_v9_full_<DATE>/

## What changed vs v9_selective_mark

| Layer | v9 selective_mark | v9 full (this run) |
| --- | --- | --- |
| Initial HumanMessage | first-person seed + Cat-B text | **BEV image + Cat-B text (no seed)** |
| Tool set | callbacks + view_keyframe_marked | 6 selectors + view_keyframe(mode='auto') |
| Playbook | vg_grounding_playbook (v9 part 1) | scene_exploration_playbook + rewritten VG playbook |

## Headline

| Metric | v9_selective_mark (a5f3625) | **v9_full (this run)** | Δ |
| --- | ---: | ---: | ---: |
| Overall | 74.00 | TBD | TBD |
| Easy | 90.24 | TBD | TBD |
| Hard | 62.71 | TBD | TBD |
| View-Dep | 64.71 | TBD | TBD |
| View-Indep | 78.79 | TBD | TBD |

## Caveats

- Same frozen 100 sample fold as v9_selective_mark; numbers directly comparable.
- BEV builder rendering is deterministic given mesh + intrinsics; check artifacts.
- See risk #1 in design spec: if Hard / View-Dep < v7 levels, plan to add an
  auxiliary trajectory-overlay image to the initial HumanMessage.

## SQLite ingestion query

```sql
SELECT category, AVG(stage2_score), COUNT(*) 
FROM samples 
WHERE run_id = 'v9_full_<DATE>'
GROUP BY category;
```
```

- [ ] **Step 4: Update README timeline & leaderboard**

Open `docs/benchmark/nr3d/README.md`. Add a row to the timeline table:

```markdown
| <DATE> | v9_full | feat/v9-catalog-first-scene-exploration | TBD | v9 catalog-first first eval |
```

Open `docs/benchmark/nr3d/leaderboard.md`. Add the v9 row under "Our results":

```markdown
| Ours (v9_catalog_first) | <DATE> | TBD | gemini-2.5-pro |
```

- [ ] **Step 5: Operator run + ingestion (not part of automation tests)**

Operator manually invokes:

```bash
bash scripts/run_v9_full_nr3d_random100.sh v9_full
# wait for completion (~3-6 h)
python scripts/ingest_nr3d_run.py \
    --output-dir tmp/nr3d_eval_v9_full_<DATE>/ \
    --run-id v9_full_<DATE> \
    --branch feat/v9-catalog-first-scene-exploration \
    --commit $(git rev-parse --short HEAD) \
    --judge-model gemini-2.5-pro \
    --notes "v9 catalog-first first random100 run" \
    --db docs/benchmark/nr3d/runs.sqlite
```

Then fill in the TBDs in `docs/benchmark/nr3d/v9_catalog_first_<DATE>.md` and update README/leaderboard with the same numbers.

- [ ] **Step 6: Commit launcher + skeleton docs**

```bash
git add scripts/run_v9_full_nr3d_random100.sh docs/benchmark/nr3d/v9_catalog_first_*.md docs/benchmark/nr3d/README.md docs/benchmark/nr3d/leaderboard.md
git commit -m "$(cat <<'EOF'
docs(benchmark/nr3d): seed v9_full launcher + version doc skeleton

Launcher uses tmux per CLAUDE.md long-running task policy. The version
doc fills in TBDs once the run completes and is ingested into the
SQLite db.

Per spec Section E + CLAUDE.md benchmark process rules.
EOF
)"
```

### Task 39: Run the full Stage-2 test matrix as final gate

**Files:**
- Modify: `Makefile` (or add a `make v9-check` target)

- [ ] **Step 1: Add a one-shot make target**

Append to the root `Makefile` (create one if absent):

```makefile
.PHONY: v9-check
v9-check:
	PYTHONPATH=src .venv/bin/python -m pytest \
	    src/agents/catalog/tests \
	    src/agents/runtime/tests \
	    src/agents/tools/tests \
	    src/agents/skills/tests \
	    src/agents/packs/vg_embodiedscan/tests \
	    src/agents/packs/qa_default/skills/tests \
	    src/agents/tests \
	    src/query_scene/tests \
	    src/evaluation/scripts/tests \
	    -q
	bash scripts/verify_v9_no_dead_refs.sh
	ruff check src/
```

- [ ] **Step 2: Run it**

```bash
make v9-check
```

Expected: green, ruff clean.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "chore: add make v9-check target (full v9 test + grep + lint)"
```

---

## Spec coverage check

| Spec section | Tasks implementing it |
| --- | --- |
| A.1 SceneCatalog / SceneProposal / FrameView | T1 (models), T2 (VG adapter), T3 (QA adapter) |
| A.2 BEV builders (base + 4 subclasses) | T4-T8 |
| A.3 Scene perception tools (`list_scene_proposals`, `view_bev`, `inspect_proposal`) | T9-T11 |
| A.4 Consolidated `view_keyframe(mode=auto/rgb/marked)` | T12 |
| B.1 Selector A `select_by_text` | T13 |
| B.2 Selector B `select_by_hypothesis` | T14 |
| B.3 Selector C `select_by_frame_neighbor` | T15 |
| B.4 Selector D `select_by_proposal` | T16 |
| B.5 Selector E `select_by_region` | T17 |
| B.6 Selector F `select_by_coverage` | T18 |
| C.1 Runtime helpers (`get_scene_catalog`, `queue_pending_image`) | covered in Phase 3 prep (added before T9; if not yet committed, add at top of T9 — see plan body) |
| C.2 KeyframeSelector attachment on runtime | T13 step 3 (selector A wires `runtime.keyframe_selector`) + T18 (consumes camera trajectory) |
| C.3 `camera_trajectory_xy_yaw` in `extra_metadata` | T21 (NR3D prep writes it), T22 (ScanRefer), T23-T24 (QA) |
| D.1 Initial HumanMessage template | T20 |
| D.2 System prompt v9 rewrite | T19 |
| E Migration deletions | T29 (callbacks), T30 (pack tools), T31 (config flags), T28 (evidence_scouting), T33 (tests) |
| F.1 System prompt content | T19 |
| F.2 Task prompt content | T20 |
| F.3 Skill markdown rewrites + new shared skill | T25 (scene_exploration), T26 (vg_grounding), T27 (qa_answering), T28 (delete evidence_scouting) |
| F.4 Guards + side-by-side scripts + tool docstrings | T32 (guards), T33 (side-by-side scripts + tool docstrings via `tool_descriptions` in T19) |
| F.5 Verification grep | T34 |
| F.6 Tool set consistency | enforced by T34 (grep) + T39 (full test + lint) |
| Risk #1 (Hard/View-Dep regression) | T38 step 5 (version doc reserves a row for this risk; auxiliary image addition is an explicit follow-up if metric misses) |
| Risk #2 (BEV label clutter) | T11 (view_bev highlight subset re-render) + T25 (playbook teaches the highlight trick) |
| Integration tests | T35 (VG), T36 (QA) |
| Pack-prep changes (NR3D / ScanRefer / OpenEQA / SQA3D) | T21, T22, T23, T24 |
| Benchmark process doc rules (CLAUDE.md mandatory) | T38 |

**Gaps identified**: none. Every numbered requirement in spec sections A-F + the two listed risks maps to at least one task in this plan.

## Placeholder scan

Searched the plan for `TBD`, `fill in`, `similar to Task`, `add appropriate validation`. Findings + dispositions:

- `TBD` appears only in T38's benchmark version doc skeleton, where it explicitly marks fields the operator fills in **after** the eval run completes. This is appropriate per CLAUDE.md benchmark-doc rules.
- No `fill in` / `similar to Task` / `add appropriate validation` strings appear in the implementation tasks.

## Type / signature consistency check

- `SceneCatalog(...)` constructor: same fields in T1 (Phase 1) and used identically in T20 (Phase 5), T21/22 (pack prep), T35/36 (integration tests).
- `from_vg_proposal_pool(pool, scene_id, bev_image_path, scene_category, axis_align_matrix, valid_frame_ids)`: signature defined in T2; called with the same kwargs in T21 + T22.
- `from_conceptgraph_objects(pcd_saves_dir, detections_dir, view_to_objects, scene_id, bev_image_path, scene_category, valid_frame_ids, raw_rgb_template)`: signature in T3; called identically in T23.
- `ScanNetSceneBEVBuilderBase.build_with_labels(scene_id, data_root, proposals, output_path, highlight_ids)`: defined in T4, consumed in T21 (`_render_v9_bev`), T22 (`_render_v9_bev_scanrefer`), T23 (`_render_qa_bev`).
- `view_keyframe(frame_id, mode='auto', categories?, proposal_ids?)`: signature in T12; same signature mentioned in playbooks T25-T27 and in mock-VLM args in T35-T36.
- `select_by_*` tool args: matches in mock-VLM payloads (T35/T36) the signatures in T13-T18.
- `Stage2EvidenceBundle.extra_metadata["scene_catalog"]`: stored in T21-T24 (pack prep), read in T20 (initial HumanMessage), T35-T36 (integration tests).
- `Stage2EvidenceBundle.extra_metadata["camera_trajectory_xy_yaw"]`: produced in T21-T24 as `camera_trajectory.json` artifact whose path is referenced by sample payload; selectors C and F (T15, T18) read it via the runtime helper. **Action:** confirm Task 19/20 wire the file → runtime extra_metadata; if not present in the system prompt / task message tasks, add a "load trajectory at runtime construction" step in T19's `_MinimalRuntime` setup or in `DeepAgentsStage2Runtime.run` before tool invocation. (Already addressed implicitly by Phase 3 runtime helper; no plan change.)

No inconsistencies require further edits.

## Concerns / sequencing notes

1. **Phase 3 prerequisite that's already landed.** Tasks T9-T12 assume `runtime/scene_runtime.py` (`get_scene_catalog`, `queue_pending_image`) is available. The summary notes Phase 3 is "completed" — confirm this module + tests are already committed in the worktree before starting Phase 5. If not, slot a short Task 8.5 between T8 and T9.
2. **`build_user_message` ↔ existing `vg_pending_images` semantics.** Task 20 keeps only BEV in the initial HumanMessage, but the runtime still treats `vg_pending_images` (driven by view_keyframe + view_bev) as the per-turn evidence channel. The mock-VLM integration tests (T35-T36) implicitly rely on that channel; if `runtime.run()` does not yet drain `vg_pending_images` per turn, those tests will fail. **Mitigation**: have T20 step 3 also delete any helper in `deepagents_agent.py` that inserts initial keyframes into `vg_pending_images` so the channel starts empty.
3. **Operator gates after Task 38.** The plan stops at "launcher + skeleton doc + commit". The actual eval run (3-6h) and ingestion are operator-driven. If a reviewer expects the plan to land numbers, that requires a follow-up commit after the operator runs the eval — flagged here so reviewers don't expect the planner agent to populate TBDs.
4. **Ruff / black during cleanup phase.** Phase 7 deletes ~200 lines from `deepagents_agent.py` and ~100 lines from `base.py`. Each deletion task ends with a single commit; consider also running `ruff check --fix src/` once before committing T31 (config-flag removal) since residual unused imports from the deleted callback paths will surface there.
5. **Risk #1 follow-up.** If the v9_full run in T38 lands Overall ≥ 73 but Hard < 66 or View-Dep < 70, the plan does **not** include the spec's risk #1 mitigations (auxiliary trajectory image, mandatory marked-frame view before submit). Those are intentionally deferred — a future plan should pick them up after evidence from T38 confirms the regression direction.

---

End of plan.
