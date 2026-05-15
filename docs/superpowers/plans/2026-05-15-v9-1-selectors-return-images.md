# v9.1 Selectors-Return-Images + Mark-Frame Tool + BEV Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `select_by_text` the universal first-move tool that returns ≤3 RGB frames; promote all five selectors to image-injecting; replace `view_keyframe` with `mark_frame_with_bbox` for annotated single-frame inspection; delete `select_by_hypothesis`; fix BEV rendering (crop, legibility, highlight bug).

**Architecture:** Strangler-fig migration. Build mark tool + helper first, migrate selectors and guards over, then delete legacy `view_keyframe` and `select_by_hypothesis`. BEV fixes ride along in their own phase. Prompts/playbooks updated last so they describe shipped reality.

**Tech Stack:** Python 3.11, LangChain v1 + DeepAgents, OpenCV (cv2), open3d, Pillow (PIL), pytest, loguru.

**Spec:** `docs/superpowers/specs/2026-05-15-v9-1-selectors-return-images-design.md`

**Branch:** start a fresh branch from `feat/v9-catalog-first-scene-exploration`:
```bash
git checkout feat/v9-catalog-first-scene-exploration
git pull --ff-only
git checkout -b feat/v9-1-selectors-return-images
```

---

## Task 1: Add `queue_pending_image_if_new` runtime helper

**Files:**
- Modify: `src/agents/runtime/scene_runtime.py`
- Test: `src/agents/runtime/tests/test_scene_runtime_helpers.py`

- [ ] **Step 1: Write the failing test**

Create file `src/agents/runtime/tests/test_scene_runtime_helpers.py`:

```python
from pathlib import Path

import pytest

from agents.runtime.scene_runtime import (
    queue_pending_image,
    queue_pending_image_if_new,
)


class _FakeRuntime:
    def __init__(self) -> None:
        self.seen_image_paths: set[str] = set()

        class _Bundle:
            extra_metadata = {"vg_pending_images": []}

        self.bundle = _Bundle()


def test_queue_pending_image_if_new_queues_unseen():
    rs = _FakeRuntime()
    queue_pending_image_if_new(rs, "/tmp/frame_42.png")
    assert rs.bundle.extra_metadata["vg_pending_images"] == ["/tmp/frame_42.png"]


def test_queue_pending_image_if_new_skips_seen():
    rs = _FakeRuntime()
    rs.seen_image_paths.add("/tmp/frame_42.png")
    queue_pending_image_if_new(rs, "/tmp/frame_42.png")
    assert rs.bundle.extra_metadata["vg_pending_images"] == []


def test_queue_pending_image_if_new_handles_empty_path():
    rs = _FakeRuntime()
    queue_pending_image_if_new(rs, "")
    assert rs.bundle.extra_metadata["vg_pending_images"] == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/agents/runtime/tests/test_scene_runtime_helpers.py -v
```
Expected: `ImportError: cannot import name 'queue_pending_image_if_new' from 'agents.runtime.scene_runtime'`.

- [ ] **Step 3: Implement the helper**

Add to `src/agents/runtime/scene_runtime.py` (after `queue_pending_image`):

```python
def queue_pending_image_if_new(runtime: Any, path: str) -> bool:
    """Queue `path` for the next evidence update only if it has not already been seen.

    Returns True if the image was queued (caller should mark `already_seen=False`),
    False if it was a no-op (caller should mark `already_seen=True`). Empty paths
    are silently skipped (returns False).
    """
    if not path:
        return False
    if path in runtime.seen_image_paths:
        return False
    queue_pending_image(runtime, path)
    return True
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/agents/runtime/tests/test_scene_runtime_helpers.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agents/runtime/scene_runtime.py src/agents/runtime/tests/test_scene_runtime_helpers.py
git commit -m "feat(runtime): add queue_pending_image_if_new helper for selector dedup"
```

---

## Task 2: `mark_frame_with_bbox` skeleton — required-filter error path

**Files:**
- Create: `src/agents/tools/mark_frame_with_bbox.py`
- Test: `src/agents/tools/tests/test_mark_frame_with_bbox_errors.py`

- [ ] **Step 1: Write the failing tests**

Create `src/agents/tools/tests/test_mark_frame_with_bbox_errors.py`:

```python
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.catalog import SceneCatalog, SceneProposal, FrameView
from agents.runtime.base import Stage2RuntimeState
from agents.tools.mark_frame_with_bbox import build_mark_frame_with_bbox_tool


def _runtime(tmp_path: Path) -> Stage2RuntimeState:
    bev_path = tmp_path / "bev.png"
    bev_path.write_bytes(b"")
    catalog = SceneCatalog(
        scene_id="s_test",
        bev_image_path=str(bev_path),
        proposals=[
            SceneProposal(
                proposal_id=4,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                bbox_3d=(0, 0, 0, 1, 1, 1),
                frame_views={
                    42: FrameView(
                        frame_id=42,
                        raw_rgb_path=str(tmp_path / "frame_42.png"),
                        bbox_2d=(10, 10, 100, 100),
                    )
                },
            ),
        ],
        valid_frame_ids=[42],
    )
    bundle = SimpleNamespace(
        extra_metadata={
            "scene_catalog": catalog.model_dump(),
            "vg_pending_images": [],
        }
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.seen_image_paths = set()
    rs.skills_loaded = {"scene-exploration-playbook"}
    return rs


def test_mark_frame_with_bbox_errors_when_no_filter(tmp_path: Path):
    rs = _runtime(tmp_path)
    tool = build_mark_frame_with_bbox_tool(rs)
    out = tool.invoke({"frame_id": 42})
    assert out.startswith("ERROR: mark_frame_with_bbox requires at least one of {labels, ids}")


def test_mark_frame_with_bbox_errors_on_empty_lists(tmp_path: Path):
    rs = _runtime(tmp_path)
    tool = build_mark_frame_with_bbox_tool(rs)
    out = tool.invoke({"frame_id": 42, "labels": [], "ids": []})
    assert out.startswith("ERROR: mark_frame_with_bbox requires at least one of {labels, ids}")


def test_mark_frame_with_bbox_errors_on_invalid_frame(tmp_path: Path):
    rs = _runtime(tmp_path)
    tool = build_mark_frame_with_bbox_tool(rs)
    out = tool.invoke({"frame_id": 999, "ids": [4]})
    assert out.startswith("ERROR: frame_id=999 not in valid_frame_ids")


def test_mark_frame_with_bbox_errors_when_filter_matches_nothing(tmp_path: Path):
    rs = _runtime(tmp_path)
    tool = build_mark_frame_with_bbox_tool(rs)
    out = tool.invoke({"frame_id": 42, "ids": [999]})
    assert "no visible proposals matched filters" in out
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/agents/tools/tests/test_mark_frame_with_bbox_errors.py -v
```
Expected: `ImportError: No module named 'agents.tools.mark_frame_with_bbox'`.

- [ ] **Step 3: Implement skeleton with error paths only**

Create `src/agents/tools/mark_frame_with_bbox.py`:

```python
"""v9.1 mark_frame_with_bbox tool: high-contrast annotated single-frame view."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from agents.catalog import SceneProposal
from agents.runtime.scene_runtime import (
    get_scene_catalog,
    queue_pending_image,
)
from agents.tools.scene_perception import _gate


def _norm_category(category: str) -> str:
    return " ".join(str(category).strip().lower().split())


def _filter_visible(
    proposals: list[SceneProposal],
    frame_id: int,
    labels: list[str],
    ids: list[int],
) -> list[SceneProposal]:
    wanted_cat = {_norm_category(c) for c in labels if c}
    wanted_ids = {int(i) for i in ids}
    out: list[SceneProposal] = []
    for p in proposals:
        if frame_id not in p.frame_views:
            continue
        cat_match = bool(wanted_cat) and _norm_category(p.category) in wanted_cat
        id_match = p.proposal_id in wanted_ids
        if cat_match or id_match:
            out.append(p)
    return out


def build_mark_frame_with_bbox_tool(runtime: Any) -> BaseTool:
    @tool
    def mark_frame_with_bbox(
        frame_id: int,
        labels: list[str] | None = None,
        ids: list[int] | None = None,
    ) -> str:
        """Render one first-person frame with high-contrast bboxes for the labels / ids you name.

        Requires at least one of `labels` (category names) or `ids` (proposal ids).
        Detailed usage in 'scene-exploration-playbook'.
        """
        labels_in = [str(c) for c in (labels or []) if c]
        ids_in = [int(i) for i in (ids or [])]
        request = {"frame_id": int(frame_id), "labels": labels_in, "ids": ids_in}
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("mark_frame_with_bbox", request, gate)
            return gate
        if not labels_in and not ids_in:
            err = (
                "ERROR: mark_frame_with_bbox requires at least one of {labels, ids}; "
                "to view plain RGB, request the frame via a selector"
            )
            runtime.record("mark_frame_with_bbox", request, err)
            return err
        catalog = get_scene_catalog(runtime)
        if int(frame_id) not in {int(f) for f in catalog.valid_frame_ids}:
            err = (
                f"ERROR: frame_id={frame_id} not in valid_frame_ids; "
                f"available[:20]={sorted(int(f) for f in catalog.valid_frame_ids)[:20]}"
            )
            runtime.record("mark_frame_with_bbox", request, err)
            return err
        proposals_here = [p for p in catalog.proposals if int(frame_id) in p.frame_views]
        visible = _filter_visible(proposals_here, int(frame_id), labels_in, ids_in)
        if not visible:
            err = (
                f"ERROR: no visible proposals matched filters for frame_id={frame_id}; "
                f"filtered_by={{'labels': {labels_in}, 'ids': {ids_in}}}; "
                f"visible_proposals={[p.proposal_id for p in proposals_here]}"
            )
            runtime.record("mark_frame_with_bbox", request, err)
            return err
        # Rendering implementation comes in Task 3-4. For now, return a placeholder
        # that still meets the response schema so downstream guards parse it.
        placeholder = f"frame_id={frame_id} mark image PLACEHOLDER; visible_proposals=[]; categories=[]; left_to_right=[]; boxes_2d={{}}"
        runtime.record("mark_frame_with_bbox", request, placeholder)
        return placeholder

    return mark_frame_with_bbox


__all__ = ["build_mark_frame_with_bbox_tool"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest src/agents/tools/tests/test_mark_frame_with_bbox_errors.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agents/tools/mark_frame_with_bbox.py src/agents/tools/tests/test_mark_frame_with_bbox_errors.py
git commit -m "feat(tools): add mark_frame_with_bbox skeleton with required-filter validation"
```

---

## Task 3: `mark_frame_with_bbox` rendering — palette + double outline

**Files:**
- Modify: `src/agents/tools/mark_frame_with_bbox.py`
- Test: `src/agents/tools/tests/test_mark_frame_with_bbox_render.py`

- [ ] **Step 1: Write the failing test**

Create `src/agents/tools/tests/test_mark_frame_with_bbox_render.py`:

```python
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from agents.catalog import SceneCatalog, SceneProposal, FrameView
from agents.runtime.base import Stage2RuntimeState
from agents.tools.mark_frame_with_bbox import (
    build_mark_frame_with_bbox_tool,
    BBOX_PALETTE,
    BLACK_OUTLINE_PAD,
)


@pytest.fixture
def tiny_runtime(tmp_path: Path) -> tuple[Stage2RuntimeState, Path]:
    rgb_path = tmp_path / "frame_42.png"
    Image.new("RGB", (400, 300), (200, 200, 200)).save(rgb_path)
    bev_path = tmp_path / "bev.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(bev_path)
    catalog = SceneCatalog(
        scene_id="s_render",
        bev_image_path=str(bev_path),
        proposals=[
            SceneProposal(
                proposal_id=4,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                bbox_3d=(0, 0, 0, 1, 1, 1),
                frame_views={
                    42: FrameView(
                        frame_id=42,
                        raw_rgb_path=str(rgb_path),
                        bbox_2d=(50, 50, 250, 200),
                    )
                },
            ),
        ],
        valid_frame_ids=[42],
    )
    bundle = SimpleNamespace(
        extra_metadata={
            "scene_catalog": catalog.model_dump(),
            "vg_pending_images": [],
        }
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.seen_image_paths = set()
    rs.skills_loaded = {"scene-exploration-playbook"}
    return rs, rgb_path


def test_render_writes_png_with_bbox_pixels(tiny_runtime):
    rs, _ = tiny_runtime
    tool = build_mark_frame_with_bbox_tool(rs)
    out = tool.invoke({"frame_id": 42, "ids": [4]})
    assert "mark image at" in out
    rendered_path = Path(out.split("mark image at")[1].split(";", 1)[0].strip())
    assert rendered_path.exists()
    img = np.asarray(Image.open(rendered_path).convert("RGB"))
    # The first palette colour is green (34, 197, 94). It should appear somewhere on
    # the bbox border (around y=50 / y=200, x=50 / x=250).
    green = np.array(BBOX_PALETTE[0])
    matches = np.all(img == green, axis=2)
    assert matches.any(), "expected at least one pixel of the first palette colour"


def test_render_has_black_outline_outside_colour_stroke(tiny_runtime):
    """The colour stroke must be wrapped by a black silhouette for any-background contrast."""
    rs, _ = tiny_runtime
    tool = build_mark_frame_with_bbox_tool(rs)
    out = tool.invoke({"frame_id": 42, "ids": [4]})
    rendered_path = Path(out.split("mark image at")[1].split(";", 1)[0].strip())
    img = np.asarray(Image.open(rendered_path).convert("RGB"))
    # Walk outward from the colour stroke at the top edge (y just above bbox y1=50).
    green = np.array(BBOX_PALETTE[0])
    black = np.array((0, 0, 0))
    found_green_then_black = False
    for x in (60, 120, 180, 240):
        for y in range(45, 56):
            if np.array_equal(img[y, x], green):
                # Look BLACK_OUTLINE_PAD pixels outside the green band.
                if np.array_equal(img[max(0, y - BLACK_OUTLINE_PAD - 1), x], black):
                    found_green_then_black = True
                    break
        if found_green_then_black:
            break
    assert found_green_then_black, "expected a black outline outside the colour stroke"


def test_image_is_queued_for_injection(tiny_runtime):
    rs, _ = tiny_runtime
    tool = build_mark_frame_with_bbox_tool(rs)
    tool.invoke({"frame_id": 42, "ids": [4]})
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert pending and pending[-1].endswith(".png")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/agents/tools/tests/test_mark_frame_with_bbox_render.py -v
```
Expected: `ImportError: cannot import name 'BBOX_PALETTE' from 'agents.tools.mark_frame_with_bbox'` (and all three tests fail).

- [ ] **Step 3: Implement rendering with palette + outline**

Replace the placeholder body in `src/agents/tools/mark_frame_with_bbox.py`. Add at top:

```python
import cv2
import numpy as np

BBOX_PALETTE: list[tuple[int, int, int]] = [
    (34, 197, 94),   # green
    (239, 68, 68),   # red
    (59, 130, 246),  # blue
    (234, 179, 8),   # yellow
    (168, 85, 247),  # purple
]
BLACK_OUTLINE_PAD: int = 2


def _bbox_stroke_thickness(img_width: int) -> int:
    return max(5, img_width // 200)


def _draw_palette_bbox(
    img: np.ndarray,
    bbox: tuple[float, float, float, float],
    colour: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    stroke = _bbox_stroke_thickness(img.shape[1])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), stroke + 2 * BLACK_OUTLINE_PAD)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, stroke)
```

Then replace the placeholder return in `mark_frame_with_bbox` (the body after `if not visible: ... return err`) with:

```python
        from PIL import Image

        raw_path = Path(visible[0].frame_views[int(frame_id)].raw_rgb_path)
        if not raw_path.exists():
            err = f"ERROR: raw RGB image not found: {raw_path}"
            runtime.record("mark_frame_with_bbox", request, err)
            return err
        img = np.asarray(Image.open(raw_path).convert("RGB")).copy()
        for idx, prop in enumerate(visible):
            colour = BBOX_PALETTE[idx % len(BBOX_PALETTE)]
            _draw_palette_bbox(img, prop.frame_views[int(frame_id)].bbox_2d, colour)
        catalog_dir = Path(catalog.bev_image_path).parent
        cache_dir = catalog_dir / "filtered_marks"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ids_token = "_".join(str(p.proposal_id) for p in sorted(visible, key=lambda q: q.proposal_id))
        out_path = cache_dir / f"frame_{int(frame_id)}_ids_{ids_token}.png"
        Image.fromarray(img).save(out_path, format="PNG")
        queue_pending_image(runtime, str(out_path))
        body = f"frame_id={frame_id} mark image at {out_path}; visible_proposals=[]; categories=[]; left_to_right=[]; boxes_2d={{}}"
        runtime.record("mark_frame_with_bbox", request, body)
        return body
```

(Labels and the proper response schema come in Tasks 4–5; this commit only wires the bbox stroke + outline.)

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest src/agents/tools/tests/test_mark_frame_with_bbox_render.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agents/tools/mark_frame_with_bbox.py src/agents/tools/tests/test_mark_frame_with_bbox_render.py
git commit -m "feat(tools): render palette bbox with black outline in mark_frame_with_bbox"
```

---

## Task 4: `mark_frame_with_bbox` label placement (area ratio < 0.15 → centre, else top-left inside)

**Files:**
- Modify: `src/agents/tools/mark_frame_with_bbox.py`
- Test: `src/agents/tools/tests/test_mark_frame_with_bbox_label.py`

- [ ] **Step 1: Write the failing test**

Create `src/agents/tools/tests/test_mark_frame_with_bbox_label.py`:

```python
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from agents.catalog import SceneCatalog, SceneProposal, FrameView
from agents.runtime.base import Stage2RuntimeState
from agents.tools.mark_frame_with_bbox import (
    build_mark_frame_with_bbox_tool,
    _decide_label_anchor,
)


def _runtime_with_one_box(tmp_path: Path, bbox: tuple[int, int, int, int]) -> Stage2RuntimeState:
    rgb_path = tmp_path / "frame_42.png"
    Image.new("RGB", (400, 300), (200, 200, 200)).save(rgb_path)
    bev_path = tmp_path / "bev.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(bev_path)
    catalog = SceneCatalog(
        scene_id="s",
        bev_image_path=str(bev_path),
        proposals=[
            SceneProposal(
                proposal_id=4,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                bbox_3d=(0, 0, 0, 1, 1, 1),
                frame_views={
                    42: FrameView(
                        frame_id=42, raw_rgb_path=str(rgb_path), bbox_2d=bbox
                    )
                },
            ),
        ],
        valid_frame_ids=[42],
    )
    bundle = SimpleNamespace(
        extra_metadata={"scene_catalog": catalog.model_dump(), "vg_pending_images": []}
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.seen_image_paths = set()
    rs.skills_loaded = {"scene-exploration-playbook"}
    return rs


def test_decide_label_anchor_centre_for_large_bbox():
    anchor = _decide_label_anchor(
        bbox=(50, 50, 250, 200),
        text_w=60,
        text_h=18,
    )
    cx, cy = (50 + 250) / 2, (50 + 200) / 2
    assert anchor.mode == "centre"
    # area_ratio = 60*18 / (200 * 150) = 1080 / 30000 = 0.036 < 0.15
    assert anchor.origin == (int(cx - 60 / 2), int(cy + 18 / 2))


def test_decide_label_anchor_topleft_for_small_bbox():
    anchor = _decide_label_anchor(
        bbox=(50, 50, 110, 80),
        text_w=60,
        text_h=18,
    )
    # area_ratio = 60*18 / (60*30) = 1080 / 1800 = 0.6 > 0.15 -> top-left inside
    assert anchor.mode == "topleft"
    assert anchor.origin == (50 + 4, 50 + 18 + 4)


def test_render_includes_white_label_pixels_in_centre_for_large_bbox(tmp_path: Path):
    rs = _runtime_with_one_box(tmp_path, bbox=(50, 50, 350, 250))
    tool = build_mark_frame_with_bbox_tool(rs)
    out = tool.invoke({"frame_id": 42, "ids": [4]})
    rendered_path = Path(out.split("mark image at")[1].split(";", 1)[0].strip())
    img = np.asarray(Image.open(rendered_path).convert("RGB"))
    # Centre of bbox ~ (200, 150); expect a white-pixel cluster nearby (label text).
    centre_patch = img[140:160, 180:220]
    has_white = (centre_patch == np.array([255, 255, 255])).all(axis=-1).any()
    assert has_white, "expected centred label text in centre of large bbox"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/agents/tools/tests/test_mark_frame_with_bbox_label.py -v
```
Expected: `ImportError: cannot import name '_decide_label_anchor'`.

- [ ] **Step 3: Implement label placement**

Add to `src/agents/tools/mark_frame_with_bbox.py`:

```python
from dataclasses import dataclass

LABEL_AREA_RATIO_THRESHOLD: float = 0.15
LABEL_PADDING_PX: int = 4


@dataclass(frozen=True)
class LabelAnchor:
    mode: str  # 'centre' | 'topleft'
    origin: tuple[int, int]  # cv2.putText origin (x, baseline-y)
    bg_box: tuple[int, int, int, int]  # (x1, y1, x2, y2) for the opaque rect


def _decide_label_anchor(
    bbox: tuple[float, float, float, float],
    text_w: int,
    text_h: int,
) -> LabelAnchor:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    area_ratio = (text_w * text_h) / float(bbox_w * bbox_h)
    if area_ratio < LABEL_AREA_RATIO_THRESHOLD:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        origin = (cx - text_w // 2, cy + text_h // 2)
        bg = (
            origin[0] - LABEL_PADDING_PX,
            origin[1] - text_h - LABEL_PADDING_PX,
            origin[0] + text_w + LABEL_PADDING_PX,
            origin[1] + LABEL_PADDING_PX,
        )
        return LabelAnchor(mode="centre", origin=origin, bg_box=bg)
    origin = (x1 + LABEL_PADDING_PX, y1 + text_h + LABEL_PADDING_PX)
    bg = (
        x1,
        y1,
        x1 + text_w + 2 * LABEL_PADDING_PX,
        y1 + text_h + 2 * LABEL_PADDING_PX,
    )
    return LabelAnchor(mode="topleft", origin=origin, bg_box=bg)


def _draw_label(
    img: np.ndarray,
    text: str,
    bbox: tuple[float, float, float, float],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.7, img.shape[1] / 1800.0)
    thickness = max(2, img.shape[1] // 600)
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    anchor = _decide_label_anchor(bbox, text_w, text_h)
    cv2.rectangle(img, (anchor.bg_box[0], anchor.bg_box[1]), (anchor.bg_box[2], anchor.bg_box[3]), (0, 0, 0), -1)
    cv2.putText(img, text, anchor.origin, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
```

Then in the render path replace the bbox loop with:

```python
        for idx, prop in enumerate(visible):
            colour = BBOX_PALETTE[idx % len(BBOX_PALETTE)]
            view = prop.frame_views[int(frame_id)]
            _draw_palette_bbox(img, view.bbox_2d, colour)
            _draw_label(img, f"#{prop.proposal_id} {prop.category}", view.bbox_2d)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest src/agents/tools/tests/test_mark_frame_with_bbox_label.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agents/tools/mark_frame_with_bbox.py src/agents/tools/tests/test_mark_frame_with_bbox_label.py
git commit -m "feat(tools): mark_frame_with_bbox label placement (centre vs topleft via area ratio)"
```

---

## Task 5: `mark_frame_with_bbox` guard-compatible response schema

**Files:**
- Modify: `src/agents/tools/mark_frame_with_bbox.py`
- Test: `src/agents/tools/tests/test_mark_frame_with_bbox_schema.py`

- [ ] **Step 1: Write the failing test**

Create `src/agents/tools/tests/test_mark_frame_with_bbox_schema.py`:

```python
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from agents.catalog import SceneCatalog, SceneProposal, FrameView
from agents.runtime.base import Stage2RuntimeState
from agents.tools.mark_frame_with_bbox import build_mark_frame_with_bbox_tool


@pytest.fixture
def rs_with_two(tmp_path: Path) -> Stage2RuntimeState:
    rgb = tmp_path / "frame_42.png"
    Image.new("RGB", (400, 300), (200, 200, 200)).save(rgb)
    bev = tmp_path / "bev.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(bev)
    catalog = SceneCatalog(
        scene_id="s",
        bev_image_path=str(bev),
        proposals=[
            SceneProposal(
                proposal_id=4,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                bbox_3d=(0, 0, 0, 1, 1, 1),
                frame_views={42: FrameView(frame_id=42, raw_rgb_path=str(rgb), bbox_2d=(20, 50, 120, 150))},
            ),
            SceneProposal(
                proposal_id=7,
                category="table",
                position_3d=(1.0, 0.0, 0.0),
                bbox_3d=(0, 0, 0, 1, 1, 1),
                frame_views={42: FrameView(frame_id=42, raw_rgb_path=str(rgb), bbox_2d=(220, 50, 320, 150))},
            ),
        ],
        valid_frame_ids=[42],
    )
    bundle = SimpleNamespace(
        extra_metadata={"scene_catalog": catalog.model_dump(), "vg_pending_images": []}
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.seen_image_paths = set()
    rs.skills_loaded = {"scene-exploration-playbook"}
    return rs


def test_response_includes_guard_parseable_fields(rs_with_two):
    tool = build_mark_frame_with_bbox_tool(rs_with_two)
    out = tool.invoke({"frame_id": 42, "ids": [4, 7]})
    assert re.search(r"visible_proposals=\[\s*4\s*,\s*7\s*\]", out)
    assert re.search(r"categories=\['chair'\s*,\s*'table'\]", out)
    assert re.search(r"left_to_right=\['4:chair'\s*,\s*'7:table'\]", out)
    assert re.search(r"boxes_2d=\{4:\s*\[20,\s*50,\s*120,\s*150\]\s*,\s*7:\s*\[220,\s*50,\s*320,\s*150\]\}", out)


def test_response_includes_filter_summary(rs_with_two):
    tool = build_mark_frame_with_bbox_tool(rs_with_two)
    out = tool.invoke({"frame_id": 42, "labels": ["table"]})
    assert "filtered_by={'labels': ['table'], 'ids': []}" in out
    assert "visible_proposals=[7]" in out
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/agents/tools/tests/test_mark_frame_with_bbox_schema.py -v
```
Expected: 2 failed (placeholder body returns empty `visible_proposals=[]`).

- [ ] **Step 3: Replace placeholder schema with real fields**

In `mark_frame_with_bbox`, after `_draw_label` loop, replace the trailing `body = ...` with:

```python
        left_to_right_pairs = sorted(
            visible,
            key=lambda p: (
                (p.frame_views[int(frame_id)].bbox_2d[0] + p.frame_views[int(frame_id)].bbox_2d[2]) / 2.0,
                p.proposal_id,
            ),
        )
        visible_ids = [p.proposal_id for p in left_to_right_pairs]
        categories = [p.category for p in left_to_right_pairs]
        left_to_right = [f"{p.proposal_id}:{p.category}" for p in left_to_right_pairs]
        boxes_2d = {
            p.proposal_id: [int(round(v)) for v in p.frame_views[int(frame_id)].bbox_2d]
            for p in left_to_right_pairs
        }
        body = (
            f"frame_id={frame_id} mark image at {out_path}; "
            f"filtered_by={{'labels': {labels_in}, 'ids': {ids_in}}}; "
            f"visible_proposals={visible_ids}; "
            f"categories={categories}; "
            f"left_to_right={left_to_right}; "
            f"boxes_2d={boxes_2d}"
        )
        runtime.record("mark_frame_with_bbox", request, body)
        return body
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest src/agents/tools/tests/test_mark_frame_with_bbox_schema.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agents/tools/mark_frame_with_bbox.py src/agents/tools/tests/test_mark_frame_with_bbox_schema.py
git commit -m "feat(tools): mark_frame_with_bbox response schema matches guards' regex contract"
```

---

## Task 6: Wire `mark_frame_with_bbox` into the agent

**Files:**
- Modify: `src/agents/runtime/deepagents_agent.py` (around the tool-building block, lines ~131-142 today)
- Test: `src/agents/tests/integration/test_v9_mark_frame_with_bbox_wired.py` (new)

- [ ] **Step 1: Write the failing test**

Create `src/agents/tests/integration/test_v9_mark_frame_with_bbox_wired.py`:

```python
from __future__ import annotations

import pytest

from agents.runtime.deepagents_agent import _collect_tools_for_task_pack
from agents.core.agent_config import Stage2TaskType


def test_mark_frame_with_bbox_is_loaded_for_vg(monkeypatch):
    tools = _collect_tools_for_task_pack(
        task_type=Stage2TaskType.VISUAL_GROUNDING, runtime=object()
    )
    names = {getattr(t, "name", "") for t in tools}
    assert "mark_frame_with_bbox" in names


def test_mark_frame_with_bbox_is_loaded_for_qa(monkeypatch):
    tools = _collect_tools_for_task_pack(
        task_type=Stage2TaskType.QUESTION_ANSWERING, runtime=object()
    )
    names = {getattr(t, "name", "") for t in tools}
    assert "mark_frame_with_bbox" in names
```

If `_collect_tools_for_task_pack` does not exist yet (it's inline today), extract the tool-building block in `deepagents_agent.py` into a free function with that name.

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/agents/tests/integration/test_v9_mark_frame_with_bbox_wired.py -v
```
Expected: `ImportError` or "mark_frame_with_bbox not in names".

- [ ] **Step 3: Wire the new tool**

In `src/agents/runtime/deepagents_agent.py`, locate the section near lines 131–142 that imports `build_view_keyframe_tool`. Extract the tool collection into a function and add the mark tool:

```python
def _collect_tools_for_task_pack(*, task_type, runtime) -> list:
    from agents.tools.mark_frame_with_bbox import build_mark_frame_with_bbox_tool
    from agents.tools.selectors import build_selector_tools
    from agents.tools.scene_perception import build_scene_perception_tools
    # ... keep all the other tool imports that were already there ...

    tools = []
    tools.extend(build_scene_perception_tools(runtime))
    tools.extend(build_selector_tools(runtime))
    tools.append(build_mark_frame_with_bbox_tool(runtime))
    # ... keep the rest of the existing logic for other task-pack-specific tools ...
    return tools
```

Then call `_collect_tools_for_task_pack(...)` from `build_agent`. Leave `view_keyframe` wiring alone for now — Task 14 will remove it.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest src/agents/tests/integration/test_v9_mark_frame_with_bbox_wired.py -v
```
Expected: 2 passed.

Also run the broader agent loader smoke:

```bash
pytest src/agents/runtime/tests/ -v -k 'not snapshot'
```
Expected: no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/agents/runtime/deepagents_agent.py src/agents/tests/integration/test_v9_mark_frame_with_bbox_wired.py
git commit -m "feat(runtime): wire mark_frame_with_bbox tool into deepagents for VG + QA"
```

---

## Task 7: `select_by_text` injects ≤3 RGB frames

**Files:**
- Modify: `src/agents/tools/selectors.py` (the `select_by_text` block)
- Test: `src/agents/tools/tests/test_selectors_text.py` (rewrite)

- [ ] **Step 1: Rewrite the failing test**

Replace `src/agents/tools/tests/test_selectors_text.py` body with:

```python
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from agents.catalog import SceneCatalog, SceneProposal, FrameView
from agents.runtime.base import Stage2RuntimeState
from agents.tools.selectors import build_selector_tools


class _FakeKeyframeSelector:
    def __init__(self, fids: list[int]) -> None:
        self._fids = fids

    def select_keyframes_v2(self, **_kwargs):
        return SimpleNamespace(
            keyframe_indices=list(self._fids),
            metadata={"hypothesis_output": {"hypotheses": [
                {"grounding_query": {"root": {"category": "chair"}}, "kind": "direct"}
            ]}},
        )


def _runtime(tmp_path: Path, fids: list[int]) -> Stage2RuntimeState:
    bev_path = tmp_path / "bev.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(bev_path)
    proposals: list[SceneProposal] = []
    for idx, fid in enumerate(fids):
        rgb = tmp_path / f"frame_{fid}.png"
        Image.new("RGB", (320, 240), (200, 200, 200)).save(rgb)
        proposals.append(
            SceneProposal(
                proposal_id=10 + idx,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                bbox_3d=(0, 0, 0, 1, 1, 1),
                frame_views={fid: FrameView(frame_id=fid, raw_rgb_path=str(rgb), bbox_2d=(0, 0, 20, 20))},
            )
        )
    catalog = SceneCatalog(
        scene_id="s",
        bev_image_path=str(bev_path),
        proposals=proposals,
        valid_frame_ids=list(fids),
    )
    bundle = SimpleNamespace(
        extra_metadata={
            "scene_catalog": catalog.model_dump(),
            "vg_pending_images": [],
            "camera_trajectory_xy_yaw": {f: [float(f), float(f), 0.0] for f in fids},
        }
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.seen_image_paths = set()
    rs.skills_loaded = {"scene-exploration-playbook"}
    rs.keyframe_selector = _FakeKeyframeSelector(fids)
    return rs


def test_select_by_text_returns_image_paths_and_queues_them(tmp_path: Path):
    rs = _runtime(tmp_path, fids=[1, 2, 3])
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    raw = tool.invoke({"query": "wooden chair"})
    payload = json.loads(raw)
    assert len(payload["frames"]) == 3
    for frame, fid in zip(payload["frames"], [1, 2, 3]):
        assert frame["frame_id"] == fid
        assert "image_path" in frame
        assert frame["already_seen"] is False
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert len(pending) == 3


def test_select_by_text_caps_k_at_3(tmp_path: Path):
    rs = _runtime(tmp_path, fids=[1, 2, 3, 4, 5])
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    raw = tool.invoke({"query": "chair", "k": 5})
    payload = json.loads(raw)
    assert len(payload["frames"]) == 3
    assert "k capped at 3" in raw.lower()


def test_select_by_text_marks_already_seen(tmp_path: Path):
    rs = _runtime(tmp_path, fids=[1, 2])
    seen_path = str(tmp_path / "frame_1.png")
    rs.seen_image_paths.add(seen_path)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    raw = tool.invoke({"query": "chair"})
    payload = json.loads(raw)
    by_fid = {f["frame_id"]: f for f in payload["frames"]}
    assert by_fid[1]["already_seen"] is True
    assert by_fid[2]["already_seen"] is False
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert pending == [str(tmp_path / "frame_2.png")]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/agents/tools/tests/test_selectors_text.py -v
```
Expected: 3 failed.

- [ ] **Step 3: Modify `select_by_text` to inject images**

In `src/agents/tools/selectors.py`, find the `select_by_text` block. Replace the inner `result = selector.select_keyframes_v2(...)` → JSON build with:

```python
        k_in = int(k)
        capped = min(k_in, 3)
        k_warning = "" if k_in == capped else f" (k capped at 3 from {k_in})"
        try:
            result = selector.select_keyframes_v2(
                query=str(query),
                k=capped,
                hidden_categories=list(hidden_categories or []),
                use_visual_context=False,
            )
        except Exception as exc:  # noqa: BLE001
            err = f"ERROR: Stage-1 parse/exec failed: {type(exc).__name__}: {exc}"
            runtime.record("select_by_text", request, err)
            return err

        from agents.runtime.scene_runtime import queue_pending_image_if_new

        catalog = get_scene_catalog(runtime)
        frames: list[dict] = []
        for fid in (result.keyframe_indices or [])[:capped]:
            base = _build_frame_payload(
                runtime, catalog, int(fid),
                selected_because=f"select_by_text(query={query!r}){k_warning}",
                hidden_categories=list(hidden_categories or []),
            )
            image_path = _resolve_raw_rgb_path(catalog, int(fid))
            base["image_path"] = str(image_path) if image_path else None
            queued = queue_pending_image_if_new(runtime, base["image_path"] or "")
            base["already_seen"] = not queued and base["image_path"] in runtime.seen_image_paths
            frames.append(base)

        summary = ""
        hyp = (result.metadata or {}).get("hypothesis_output")
        if isinstance(hyp, dict) and hyp.get("hypotheses"):
            first = hyp["hypotheses"][0]
            root = (first.get("grounding_query") or {}).get("root") or {}
            summary = f"target={root.get('category')!r} kind={first.get('kind', 'direct')}"
        payload = {"hypothesis_summary": summary + k_warning, "frames": frames}
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("select_by_text", request, text)
        return text
```

Also add a top-level helper at the bottom of the file (or near `_build_frame_payload`):

```python
def _resolve_raw_rgb_path(catalog: Any, frame_id: int) -> Path | None:
    for p in catalog.proposals:
        view = p.frame_views.get(int(frame_id))
        if view is not None and view.raw_rgb_path:
            return Path(view.raw_rgb_path)
    return None
```

(Import `Path` from `pathlib` at the top of `selectors.py` if not already imported.)

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest src/agents/tools/tests/test_selectors_text.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agents/tools/selectors.py src/agents/tools/tests/test_selectors_text.py
git commit -m "feat(selectors): select_by_text injects <=3 RGB frames with already_seen + k cap"
```

---

## Task 8: `select_by_proposal` injects ≤3 RGB frames

**Files:**
- Modify: `src/agents/tools/selectors.py` (the `select_by_proposal` block)
- Test: `src/agents/tools/tests/test_selectors_proposal.py`

- [ ] **Step 1: Write the failing test**

Create or rewrite `src/agents/tools/tests/test_selectors_proposal.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from agents.catalog import SceneCatalog, SceneProposal, FrameView
from agents.runtime.base import Stage2RuntimeState
from agents.tools.selectors import build_selector_tools


def _runtime(tmp_path: Path) -> Stage2RuntimeState:
    bev = tmp_path / "bev.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(bev)
    rgbs = {f: tmp_path / f"frame_{f}.png" for f in [10, 20, 30, 40]}
    for r in rgbs.values():
        Image.new("RGB", (320, 240), (200, 200, 200)).save(r)
    catalog = SceneCatalog(
        scene_id="s",
        bev_image_path=str(bev),
        proposals=[
            SceneProposal(
                proposal_id=4,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                bbox_3d=(0, 0, 0, 1, 1, 1),
                frame_views={
                    10: FrameView(frame_id=10, raw_rgb_path=str(rgbs[10]), bbox_2d=(0, 0, 20, 20)),
                    20: FrameView(frame_id=20, raw_rgb_path=str(rgbs[20]), bbox_2d=(0, 0, 20, 20)),
                    30: FrameView(frame_id=30, raw_rgb_path=str(rgbs[30]), bbox_2d=(0, 0, 20, 20)),
                    40: FrameView(frame_id=40, raw_rgb_path=str(rgbs[40]), bbox_2d=(0, 0, 20, 20)),
                },
            ),
        ],
        valid_frame_ids=[10, 20, 30, 40],
    )
    bundle = SimpleNamespace(
        extra_metadata={
            "scene_catalog": catalog.model_dump(),
            "vg_pending_images": [],
            "camera_trajectory_xy_yaw": {f: [float(f), 0.0, 0.0] for f in [10, 20, 30, 40]},
        }
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.seen_image_paths = set()
    rs.skills_loaded = {"scene-exploration-playbook"}
    return rs


def test_select_by_proposal_returns_3_frames_with_images(tmp_path: Path):
    rs = _runtime(tmp_path)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_proposal")
    raw = tool.invoke({"proposal_ids": [4]})
    payload = json.loads(raw)
    assert len(payload["frames"]) == 3
    for frame in payload["frames"]:
        assert "image_path" in frame
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert len(pending) == 3


def test_select_by_proposal_respects_k_cap(tmp_path: Path):
    rs = _runtime(tmp_path)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_proposal")
    raw = tool.invoke({"proposal_ids": [4], "k": 5})
    payload = json.loads(raw)
    assert len(payload["frames"]) == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/agents/tools/tests/test_selectors_proposal.py -v
```
Expected: 2 failed (no `image_path`).

- [ ] **Step 3: Modify `select_by_proposal`**

Apply the same pattern as Task 7 — cap `k` at 3, populate `image_path` + `already_seen`, queue via `queue_pending_image_if_new`. Reuse `_resolve_raw_rgb_path` from Task 7.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest src/agents/tools/tests/test_selectors_proposal.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agents/tools/selectors.py src/agents/tools/tests/test_selectors_proposal.py
git commit -m "feat(selectors): select_by_proposal injects <=3 RGB frames"
```

---

## Task 9: `select_by_frame_neighbor` injects ≤3 RGB frames

**Files:**
- Modify: `src/agents/tools/selectors.py` (the `select_by_frame_neighbor` block)
- Test: `src/agents/tools/tests/test_selectors_neighbor.py`

- [ ] **Step 1–5:** Same pattern as Tasks 7–8. Write the failing test (anchor `frame_id=20`, mode `'temporal'`, expect 3 frames with `image_path`), implement, verify, commit:

```bash
git commit -m "feat(selectors): select_by_frame_neighbor injects <=3 RGB frames"
```

---

## Task 10: `select_by_region` injects ≤3 RGB frames

**Files:**
- Modify: `src/agents/tools/selectors.py` (the `select_by_region` block)
- Test: `src/agents/tools/tests/test_selectors_region.py`

- [ ] **Step 1–5:** Same pattern. Test both `region_type='bev_2d'` and `'bbox_3d'` paths each return ≤3 image-bearing frames. Commit:

```bash
git commit -m "feat(selectors): select_by_region injects <=3 RGB frames for both region types"
```

---

## Task 11: `select_by_coverage` injects ≤3 RGB frames

**Files:**
- Modify: `src/agents/tools/selectors.py` (the `select_by_coverage` block)
- Test: `src/agents/tools/tests/test_selectors_coverage.py`

- [ ] **Step 1–5:** Same pattern. Test both `method='obj_iou'` and `'pose_depth'` paths return ≤3 image-bearing frames. Commit:

```bash
git commit -m "feat(selectors): select_by_coverage injects <=3 RGB frames for both methods"
```

---

## Task 12: Delete `select_by_hypothesis`

**Files:**
- Modify: `src/agents/tools/selectors.py`
- Delete: `src/agents/tools/tests/test_selectors_hypothesis.py`

- [ ] **Step 1: Delete the hypothesis test file**

```bash
git rm src/agents/tools/tests/test_selectors_hypothesis.py
```

- [ ] **Step 2: Run the rest of the selector tests to confirm we don't depend on hypothesis anywhere**

```bash
pytest src/agents/tools/tests/ -v
```
Expected: every non-deleted test still passes.

- [ ] **Step 3: Remove the `@tool def select_by_hypothesis` block from `selectors.py`**

Open `src/agents/tools/selectors.py`. Delete the entire `@tool def select_by_hypothesis(...)` body and the `tools.append(select_by_hypothesis)` line. Also remove the `from agents... HypothesisOutputV1` import if it becomes unused.

In `src/query_scene/keyframe_selector.py`, find `def execute_hypothesis_dict(...)` and prepend a comment:

```python
# TODO(v9.2): No remaining consumers after select_by_hypothesis was deleted in v9.1.
# Verify no external benchmark scripts depend on this before removing.
```

- [ ] **Step 4: Run the broad agent tool tests**

```bash
pytest src/agents/tools/tests/ -v
```
Expected: no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/agents/tools/selectors.py src/query_scene/keyframe_selector.py
git commit -m "refactor(tools): delete select_by_hypothesis (no callers post-v9.1)"
```

---

## Task 13: Migrate `evidence_frame_guard` to `mark_frame_with_bbox`

**Files:**
- Modify: `src/agents/skills/evidence_frame_guard.py`
- Test: `src/agents/tests/test_evidence_frame_guard.py`

- [ ] **Step 1: Update the test to use the new tool name**

In `src/agents/tests/test_evidence_frame_guard.py`, every fixture that builds a `view_keyframe` trace entry needs to switch to `mark_frame_with_bbox`. Search-and-replace `tool_name="view_keyframe"` → `tool_name="mark_frame_with_bbox"`. The `tool_input` `mode` field becomes irrelevant; remove it.

Also add one explicit "plain RGB does not satisfy guard" case:

```python
def test_evidence_frame_guard_rejects_plain_rgb_only_evidence(...):
    # Build a runtime where the only frame-bearing tool calls are select_by_text
    # entries (no mark_frame_with_bbox). Submit pid=4 with rationale citing frame 42.
    # Assert decision.blocked is True (target not seen with bbox).
    ...
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/agents/tests/test_evidence_frame_guard.py -v
```
Expected: failures because the guard still filters by `tool_name == "view_keyframe"`.

- [ ] **Step 3: Switch the guard predicate**

In `src/agents/skills/evidence_frame_guard.py`, line ~146:

```python
def _is_marked_view(entry: Any) -> bool:
    return _tool_name(entry) == "mark_frame_with_bbox"
```

(Delete the mode check entirely — `mark_frame_with_bbox` has no mode parameter.)

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/agents/tests/test_evidence_frame_guard.py -v
```
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/agents/skills/evidence_frame_guard.py src/agents/tests/test_evidence_frame_guard.py
git commit -m "refactor(guards): evidence_frame_guard counts mark_frame_with_bbox not view_keyframe"
```

---

## Task 14: Migrate `no_match_guard` to `mark_frame_with_bbox`

**Files:**
- Modify: `src/agents/skills/no_match_guard.py`
- Test: `src/agents/tests/test_no_match_guard.py`

- [ ] **Step 1: Update the test fixtures** the same way as Task 13: replace `view_keyframe` trace entries with `mark_frame_with_bbox`.

- [ ] **Step 2: Run the test to verify it fails.**

- [ ] **Step 3: Switch the predicate**

In `no_match_guard.py` line ~103:

```python
        if name == "mark_frame_with_bbox":
            response = _response_text(entry)
            frame_id_raw = tool_input.get("frame_id")
            ...
```

(Drop the mode check.)

- [ ] **Step 4: Run the test to verify it passes.**

- [ ] **Step 5: Commit**

```bash
git add src/agents/skills/no_match_guard.py src/agents/tests/test_no_match_guard.py
git commit -m "refactor(guards): no_match_guard counts mark_frame_with_bbox not view_keyframe"
```

---

## Task 15: Delete `view_keyframe`

**Files:**
- Delete: `src/agents/tools/view_keyframe.py`
- Delete: `src/agents/tools/tests/test_view_keyframe_v9.py` (and any other `test_view_keyframe*.py`)
- Modify: `src/agents/runtime/deepagents_agent.py` (remove `from agents.tools.view_keyframe ...` and the `tools.append(build_view_keyframe_tool(runtime))` line)
- Modify: `src/agents/tests/test_migration_no_callback_tools.py` (extend must-not-be-loaded list)

- [ ] **Step 1: Strengthen `test_migration_no_callback_tools.py`**

Add assertions:

```python
def test_view_keyframe_not_loaded():
    tools = _collect_tools_for_task_pack(task_type=Stage2TaskType.VISUAL_GROUNDING, runtime=object())
    names = {getattr(t, "name", "") for t in tools}
    assert "view_keyframe" not in names


def test_select_by_hypothesis_not_loaded():
    tools = _collect_tools_for_task_pack(task_type=Stage2TaskType.VISUAL_GROUNDING, runtime=object())
    names = {getattr(t, "name", "") for t in tools}
    assert "select_by_hypothesis" not in names
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest src/agents/tests/test_migration_no_callback_tools.py -v
```
Expected: failure because view_keyframe IS still loaded.

- [ ] **Step 3: Delete files + remove imports**

```bash
git rm src/agents/tools/view_keyframe.py
git rm src/agents/tools/tests/test_view_keyframe_v9.py
```

In `src/agents/runtime/deepagents_agent.py`, remove every reference to `build_view_keyframe_tool` / `view_keyframe`.

- [ ] **Step 4: Run tests**

```bash
pytest src/agents/tests/test_migration_no_callback_tools.py src/agents/runtime/tests/ -v -k 'not snapshot'
```
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/agents/runtime/deepagents_agent.py src/agents/tests/test_migration_no_callback_tools.py
git commit -m "refactor(tools): delete view_keyframe (replaced by mark_frame_with_bbox + selectors)"
```

---

## Task 16: BEV — crop blank canvas

**Files:**
- Modify: `src/query_scene/scene_bev_builder.py`
- Test: `src/query_scene/tests/test_scene_bev_crop.py`

- [ ] **Step 1: Write the failing test**

Create `src/query_scene/tests/test_scene_bev_crop.py`:

```python
import numpy as np

from query_scene.scene_bev_builder import _crop_to_non_white


def test_crop_keeps_eight_pixel_margin():
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    img[100:110, 200:220] = (50, 50, 50)  # tiny dark patch
    cropped, (ox, oy) = _crop_to_non_white(img, margin=8)
    # Patch was at y[100:110), x[200:220). With 8-px margin, expected ~ (192, 92, 228, 118)
    assert cropped.shape[0] == 26  # 10 + 2*8 (height of patch + margins)
    assert cropped.shape[1] == 36  # 20 + 2*8
    assert ox == 192
    assert oy == 92


def test_crop_all_white_returns_original():
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    cropped, (ox, oy) = _crop_to_non_white(img, margin=8)
    assert cropped.shape == img.shape
    assert (ox, oy) == (0, 0)
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest src/query_scene/tests/test_scene_bev_crop.py -v
```
Expected: `ImportError: cannot import name '_crop_to_non_white'`.

- [ ] **Step 3: Implement the crop**

Add to `src/query_scene/scene_bev_builder.py`:

```python
def _crop_to_non_white(
    img: np.ndarray, *, margin: int = 8, threshold: int = 250
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop an image to its non-white bounding box plus a fixed pixel margin.

    Returns (cropped_image, (offset_x, offset_y)). Offsets allow callers to
    translate any pre-computed (u, v) image coordinates into the cropped frame.
    """
    non_white = np.any(img < threshold, axis=2)
    if not non_white.any():
        return img, (0, 0)
    rows = np.where(non_white.any(axis=1))[0]
    cols = np.where(non_white.any(axis=0))[0]
    y_min = max(0, int(rows.min()) - margin)
    y_max = min(img.shape[0], int(rows.max()) + 1 + margin)
    x_min = max(0, int(cols.min()) - margin)
    x_max = min(img.shape[1], int(cols.max()) + 1 + margin)
    return img[y_min:y_max, x_min:x_max], (x_min, y_min)
```

Then in `build_with_labels`, after `_render_mesh_with_traj` returns `(img, view)`, call:

```python
        img, (crop_ox, crop_oy) = _crop_to_non_white(img, margin=8)
        if isinstance(view, dict):
            view["crop_offset"] = (crop_ox, crop_oy)
```

Update `_project_centroid` to subtract `view["crop_offset"]` before returning `(u, v)` when `crop_offset` is set:

```python
        u_out = int(u) - int(view_or_bounds.get("crop_offset", (0, 0))[0])
        v_out = int(v) - int(view_or_bounds.get("crop_offset", (0, 0))[1])
```

Also add `crop_margin: int = 8` to `SceneBEVConfig` so the value is part of `config_hash`.

- [ ] **Step 4: Run tests**

```bash
pytest src/query_scene/tests/test_scene_bev_crop.py -v
pytest src/query_scene/tests/ -v -k 'bev'
```
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/query_scene/scene_bev_builder.py src/query_scene/tests/test_scene_bev_crop.py
git commit -m "feat(bev): crop blank canvas to non-white bbox + propagate crop offset to projection"
```

---

## Task 17: BEV — label legibility (font + outline)

**Files:**
- Modify: `src/query_scene/scene_bev_builder.py` (`_overlay_proposal_labels`)
- Test: `src/query_scene/tests/test_scene_bev_label_contrast.py`

- [ ] **Step 1: Write the failing test**

Create `src/query_scene/tests/test_scene_bev_label_contrast.py`:

```python
import numpy as np
from PIL import Image
from query_scene.scene_bev_builder import ScanNetSceneBEVBuilderBase, SceneBEVConfig
from agents.catalog import SceneProposal


class _Stub(ScanNetSceneBEVBuilderBase):
    def resolve_paths(self, scene_id, data_root):
        raise NotImplementedError


def test_label_has_black_outline_around_white_text():
    """The text glyph must have a black outline so it's readable on any background."""
    img = np.full((300, 400, 3), 200, dtype=np.uint8)  # uniform grey
    proposals = [
        SceneProposal(
            proposal_id=1,
            category="chair",
            position_3d=(0.0, 0.0, 0.0),
            bbox_3d=(0, 0, 0, 1, 1, 1),
            frame_views={},
        )
    ]
    builder = _Stub(config=SceneBEVConfig(image_size=400))
    bounds = (0.0, 0.0, 10.0, 10.0)
    out = builder._overlay_proposal_labels(img, proposals, bounds, highlight_ids=None)
    # Expect: somewhere in the image there's a row where black pixel is adjacent to a white pixel.
    is_black = (out == np.array([0, 0, 0])).all(axis=-1)
    is_white = (out == np.array([255, 255, 255])).all(axis=-1)
    adj_right = is_black[:, :-1] & is_white[:, 1:]
    adj_left = is_white[:, :-1] & is_black[:, 1:]
    assert adj_right.any() or adj_left.any(), "expected black/white adjacency from text outline"
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest src/query_scene/tests/test_scene_bev_label_contrast.py -v
```
Expected: fail (current code paints grey text on white bg, no high-contrast outline).

- [ ] **Step 3: Update `_overlay_proposal_labels`**

In `src/query_scene/scene_bev_builder.py`, replace the per-proposal label block with:

```python
            scale = 0.85
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
            text_org = (u + 6, v - 6)
            bg = (
                text_org[0] - 4,
                text_org[1] - th - 4,
                text_org[0] + tw + 4,
                text_org[1] + baseline + 4,
            )
            cv2.rectangle(img, (bg[0], bg[1]), (bg[2], bg[3]), self.config.label_bg_default if not highlighted else self.config.label_bg_highlight, -1)
            cv2.putText(img, label, text_org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(img, label, text_org, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
```

Add `label_font_scale: float = 0.85` and `label_font_thickness: int = 2` to `SceneBEVConfig` so cache-key reflects the change.

- [ ] **Step 4: Run tests**

```bash
pytest src/query_scene/tests/test_scene_bev_label_contrast.py -v
```
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/query_scene/scene_bev_builder.py src/query_scene/tests/test_scene_bev_label_contrast.py
git commit -m "feat(bev): label font scale 0.85 + black-outline white-fill for legibility"
```

---

## Task 18: BEV — fix `view_bev(highlight=…)` projection bug

**Files:**
- Modify: `src/agents/tools/scene_perception.py` (`_render_highlighted_bev`)
- Test: `src/agents/tools/tests/test_view_bev.py` (add a new pixel-alignment assertion)

- [ ] **Step 1: Add the failing test**

Append to `src/agents/tools/tests/test_view_bev.py`:

```python
import numpy as np
from PIL import Image

from agents.tools.scene_perception import build_scene_perception_tools


def _build_view_params() -> dict:
    """Deterministic perspective camera used by both base + highlight renders."""
    return {
        "R": np.eye(3),
        "t": np.array([0.0, 0.0, 10.0]),
        "f": 800.0,
        "c": 400.0,
        "image_size": 800,
        "crop_offset": (0, 0),
    }


def test_highlight_label_aligns_with_base_bev(tmp_path: Path, monkeypatch):
    """Highlight overlay must project through the same camera as the base BEV.

    Today the highlight path uses linear bounds (fallback in `_project_centroid`)
    while the base BEV uses the perspective camera. Mock both render paths to
    share one view_params dict; assert the projected (u, v) matches.
    """
    rs = _runtime(tmp_path)  # existing helper in this file
    view_params = _build_view_params()

    # Capture base BEV's projected (u, v) for proposal #4.
    base_canvas = np.full((800, 800, 3), 200, dtype=np.uint8)

    def _fake_full_render(self, *, scene_id, data_root, proposals, output_path, highlight_ids=None, use_cache=True):
        from query_scene.scene_bev_builder import ScanNetSceneBEVBuilderBase
        labelled = ScanNetSceneBEVBuilderBase._overlay_proposal_labels(
            self, base_canvas.copy(), proposals, view_params, highlight_ids
        )
        Image.fromarray(labelled).save(output_path)
        return output_path

    monkeypatch.setattr(
        "query_scene.scene_bev_builder.ScanNetSceneBEVBuilderBase.build_with_labels",
        _fake_full_render,
    )

    catalog = rs.bundle.extra_metadata["scene_catalog"]
    proposal = catalog["proposals"][0]
    proposal_id = int(proposal["proposal_id"])

    # Render base BEV (cache miss → uses _fake_full_render).
    base_path = tmp_path / "base.png"
    monkeypatch.setattr(
        "agents.runtime.scene_runtime.get_scene_catalog",
        lambda r: r._catalog,
    )
    rs._catalog = type("C", (), {**catalog, "bev_image_path": str(base_path)})()

    view_tool = next(t for t in build_scene_perception_tools(rs) if t.name == "view_bev")
    view_tool.invoke({"highlight": None})
    base_img = np.asarray(Image.open(base_path).convert("RGB"))

    # Render highlight BEV.
    highlight_path = tmp_path / "highlights" / f"bev_h_{proposal_id}.png"
    view_tool.invoke({"highlight": [proposal_id]})
    hi_img = np.asarray(Image.open(highlight_path).convert("RGB"))

    # Find first non-grey pixel in each (label glyph or bbox border) and assert
    # column / row agree within ±2.
    base_non = np.where(np.any(base_img != 200, axis=2))
    hi_non = np.where(np.any(hi_img != 200, axis=2))
    assert len(base_non[0]) > 0 and len(hi_non[0]) > 0
    base_centroid = (int(np.mean(base_non[1])), int(np.mean(base_non[0])))
    hi_centroid = (int(np.mean(hi_non[1])), int(np.mean(hi_non[0])))
    assert abs(base_centroid[0] - hi_centroid[0]) <= 4, (base_centroid, hi_centroid)
    assert abs(base_centroid[1] - hi_centroid[1]) <= 4, (base_centroid, hi_centroid)
```

The test should fail today because the production `_render_highlighted_bev` uses `_overlay_proposal_labels(img, ..., scene_bounds, ...)` with the linear-bounds fallback — its label centroid lands tens of pixels away from the perspective-projected base label.

- [ ] **Step 2: Run to verify it fails**

```bash
pytest src/agents/tools/tests/test_view_bev.py::test_highlight_label_aligns_with_base_bev -v
```
Expected: fail (label pixels off by tens of pixels).

- [ ] **Step 3: Rewrite `_render_highlighted_bev`**

Replace the body of `_render_highlighted_bev` in `src/agents/tools/scene_perception.py` with a call to `build_with_labels` so the highlight uses the same perspective camera:

```python
def _render_highlighted_bev(catalog, highlight_ids: list[int], output_path: Path) -> Path:
    from query_scene.scene_bev_builder import resolve_benchmark_builder
    builder = resolve_benchmark_builder(catalog.benchmark or "scannet")
    builder.build_with_labels(
        scene_id=catalog.scene_id,
        data_root=Path(catalog.data_root),
        proposals=catalog.proposals,
        output_path=output_path,
        highlight_ids=list(highlight_ids),
        use_cache=True,
    )
    return output_path
```

If `resolve_benchmark_builder` does not yet exist, add it in `src/query_scene/scene_bev_builder.py` as a small factory that maps `"scannet"` to `OpenEQAScanNetBEVBuilder` (or the relevant per-benchmark subclass already used in pack-prep).

You will also need `catalog.benchmark` and `catalog.data_root` accessible. They're already populated by pack-prep — confirm in `agents/catalog.py` that `SceneCatalog` exposes them; if not, add the two optional fields and have pack-prep set them.

- [ ] **Step 4: Run tests**

```bash
pytest src/agents/tools/tests/test_view_bev.py -v
```
Expected: pass, including alignment test.

- [ ] **Step 5: Commit**

```bash
git add src/agents/tools/scene_perception.py src/query_scene/scene_bev_builder.py src/agents/tools/tests/test_view_bev.py
git commit -m "fix(bev): highlight path uses build_with_labels not backdrop+linear (pixel alignment fix)"
```

---

## Task 19: BEV — label declutter

**Files:**
- Modify: `src/query_scene/scene_bev_builder.py`
- Test: `src/query_scene/tests/test_scene_bev_declutter.py`

- [ ] **Step 1: Write the failing test**

Create `src/query_scene/tests/test_scene_bev_declutter.py`:

```python
import numpy as np
from query_scene.scene_bev_builder import _stack_overlapping_anchors


def test_stack_overlapping_anchors_offsets_overlapping_labels():
    # Anchors at the same (u, v) should get stacked vertically.
    anchors = [(100, 100), (100, 100), (200, 100)]
    text_w, text_h = 60, 18
    out = _stack_overlapping_anchors(anchors, ref_w=text_w, ref_h=text_h)
    assert out[0] == (100, 100)
    assert out[1] == (100, 100 + text_h + 6)
    assert out[2] == (200, 100)  # not overlapping; no offset


def test_stack_overlapping_anchors_stacks_three():
    anchors = [(100, 100), (105, 100), (110, 100)]
    text_w, text_h = 60, 18
    out = _stack_overlapping_anchors(anchors, ref_w=text_w, ref_h=text_h)
    assert out[1] == (105, 100 + text_h + 6)
    assert out[2] == (110, 100 + 2 * (text_h + 6))
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest src/query_scene/tests/test_scene_bev_declutter.py -v
```
Expected: `ImportError`.

- [ ] **Step 3: Implement the declutter helper**

Add to `scene_bev_builder.py`:

```python
def _stack_overlapping_anchors(
    anchors: list[tuple[int, int]], *, ref_w: int, ref_h: int
) -> list[tuple[int, int]]:
    """Nudge labels whose anchor (u, v) sits within (ref_w, ref_h) of an earlier
    label's anchor downward by `ref_h + 6` per overlap.
    """
    out: list[tuple[int, int]] = []
    for u, v in anchors:
        offset = 0
        for ou, ov in out:
            if abs(u - ou) <= ref_w and abs(v - ov - offset) <= ref_h:
                offset += ref_h + 6
        out.append((u, v + offset))
    return out
```

Wire it into `_overlay_proposal_labels`: build the anchor list first, run `_stack_overlapping_anchors`, then draw labels at the dispatched anchors.

Add `label_declutter_gap_px: int = 6` to `SceneBEVConfig`.

- [ ] **Step 4: Run tests**

```bash
pytest src/query_scene/tests/test_scene_bev_declutter.py -v
pytest src/query_scene/tests/ -v -k 'bev'
```
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/query_scene/scene_bev_builder.py src/query_scene/tests/test_scene_bev_declutter.py
git commit -m "feat(bev): vertical-stack overlapping labels via shared ref text size"
```

---

## Task 20: Rewrite `scene_exploration_playbook.md`

**Files:**
- Modify: `src/agents/skills/shared_skills/scene_exploration_playbook.md`
- Test: `src/agents/skills/tests/test_scene_exploration_playbook_loadable.py`

- [ ] **Step 1: Update the loadable-playbook test**

In `test_scene_exploration_playbook_loadable.py`, replace the substring assertions:

```python
def test_playbook_mentions_v9_1_tools():
    body = Path("src/agents/skills/shared_skills/scene_exploration_playbook.md").read_text()
    assert "select_by_text" in body
    assert "mark_frame_with_bbox" in body
    assert "select_by_hypothesis" not in body
    assert "view_keyframe" not in body


def test_playbook_says_first_move_is_select_by_text():
    body = Path("src/agents/skills/shared_skills/scene_exploration_playbook.md").read_text()
    assert "First move" in body
    assert body.lower().count("select_by_text") >= 2
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest src/agents/skills/tests/test_scene_exploration_playbook_loadable.py -v
```
Expected: fail (old playbook still mentions `view_keyframe`).

- [ ] **Step 3: Replace the playbook body**

Copy the markdown body in spec §7.1 verbatim into `src/agents/skills/shared_skills/scene_exploration_playbook.md`.

- [ ] **Step 4: Run tests**

```bash
pytest src/agents/skills/tests/test_scene_exploration_playbook_loadable.py -v
```
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/agents/skills/shared_skills/scene_exploration_playbook.md src/agents/skills/tests/test_scene_exploration_playbook_loadable.py
git commit -m "docs(skills): scene_exploration_playbook v9.1 (select_by_text first, mark_frame_with_bbox)"
```

---

## Task 21: Rewrite `vg_grounding_playbook.md` + `vg_spatial_disambiguation.md`

**Files:**
- Modify: `src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md`
- Modify: `src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation.md`
- Test: `src/agents/packs/vg_embodiedscan/skills/tests/test_playbook_v9_consistency.py`

- [ ] **Step 1–5:** Same TDD pattern as Task 20. Update the consistency test to assert presence of `select_by_text` / `mark_frame_with_bbox` and absence of `view_keyframe` / `select_by_hypothesis`. Replace playbook bodies per spec §7.2 + §7.3. Commit:

```bash
git commit -m "docs(skills): VG playbooks v9.1 (mark_frame_with_bbox + first-move text)"
```

---

## Task 22: Rewrite `qa_answering_playbook.md`

**Files:**
- Modify: `src/agents/packs/qa_default/skills/qa_answering_playbook.md`
- Test: `src/agents/packs/qa_default/skills/tests/test_qa_playbook_v9_consistency.py`

- [ ] **Step 1–5:** Same pattern. Replace question taxonomy per spec §7.4. Test must assert `view_keyframe(mode='rgb')` is gone. Commit:

```bash
git commit -m "docs(skills): qa_answering_playbook v9.1 (selectors-as-RGB-source)"
```

---

## Task 23: Refresh `build_system_prompt` + `deepagents_agent` inline strings

**Files:**
- Modify: `src/agents/runtime/base.py` (the `build_system_prompt` body around lines 389-441)
- Modify: `src/agents/runtime/deepagents_agent.py` (the hint strings around lines 207-280)
- Test: `src/agents/runtime/tests/test_build_system_prompt_v9.py` (regenerate snapshots)

- [ ] **Step 1: Update the snapshot test expected substrings**

In `test_build_system_prompt_v9.py`, replace expected fragments:

```python
def test_prompt_describes_selector_first_move():
    prompt = build_system_prompt(...)
    assert "select_by_text" in prompt
    assert "primary entry" in prompt.lower() or "first move" in prompt.lower()


def test_prompt_does_not_mention_deleted_tools():
    prompt = build_system_prompt(...)
    assert "view_keyframe" not in prompt
    assert "select_by_hypothesis" not in prompt
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Apply spec §7.6 and §7.7 changes**

In `runtime/base.py` lines 389-441, replace the enumeration block with the new one from spec §7.6.

In `deepagents_agent.py` lines 207-280, remove the VG/QA "Default view mode" sentences and rewrite the hint strings per spec §7.7.

- [ ] **Step 4: Run tests.**

```bash
pytest src/agents/runtime/tests/ -v
```

- [ ] **Step 5: Commit**

```bash
git add src/agents/runtime/base.py src/agents/runtime/deepagents_agent.py src/agents/runtime/tests/test_build_system_prompt_v9.py
git commit -m "refactor(runtime): system prompt v9.1 (text-first selectors, mark_frame_with_bbox)"
```

---

## Task 24: Update integration mock E2E tests

**Files:**
- Modify: `src/agents/tests/integration/test_v9_vg_end_to_end_mock.py`
- Modify: `src/agents/tests/integration/test_v9_qa_end_to_end_mock.py`

- [ ] **Step 1: Update the tool sequence**

Switch every mocked `view_keyframe` invocation to `mark_frame_with_bbox`. Each `select_by_*` mock should now return a JSON envelope with `frames: [{frame_id, image_path, already_seen, ...}]` matching the v9.1 schema.

Add an assertion that the runtime's `seen_image_paths` grows by ≤3 after each selector call.

- [ ] **Step 2: Run tests to verify they fail (or pass with stale expectations).**

```bash
pytest src/agents/tests/integration/test_v9_vg_end_to_end_mock.py src/agents/tests/integration/test_v9_qa_end_to_end_mock.py -v
```

- [ ] **Step 3: Iterate the mock harness until it matches reality.**

- [ ] **Step 4: Verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/agents/tests/integration/test_v9_vg_end_to_end_mock.py src/agents/tests/integration/test_v9_qa_end_to_end_mock.py
git commit -m "test(integration): E2E mocks use mark_frame_with_bbox + image-returning selectors"
```

---

## Task 25: Update trace HTML generator

**Files:**
- Modify: `scripts/generate_nr3d_langsmith_trace_html.py`

- [ ] **Step 1: Add a `mark` tool family with amber styling and a `legacy` family for `view_keyframe` / `select_by_hypothesis`**

Locate the dict that maps tool names → families (search for `"view_keyframe"` in the file). Add:

```python
TOOL_FAMILIES = {
    # ... existing ...
    "mark_frame_with_bbox": ("view", "#f59e0b"),  # amber
    "view_keyframe": ("legacy", "#94a3b8"),  # legacy badge
    "select_by_hypothesis": ("legacy", "#94a3b8"),
}
```

Extend `summarize_tool` so `mark_frame_with_bbox` displays its `labels` / `ids` filter and the image filename. Make the legacy family render a dashed amber outline + a "DEPRECATED" badge so historical traces still show.

Update `collect_image_refs` to also pick up the `image_path` JSON field that selectors now emit.

- [ ] **Step 2: Regenerate a sample case from the most recent run**

```bash
python scripts/generate_nr3d_langsmith_trace_html.py \
    --run-dir tmp/openeqa_eval_v9_full_random100/ \
    --out docs/benchmark/nr3d/v9_1_smoketest.html \
    --cases-limit 2
```

Inspect that the new layout shows `mark_frame_with_bbox` cards in amber and any legacy `view_keyframe` calls with the deprecation badge.

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_nr3d_langsmith_trace_html.py
git commit -m "feat(trace-html): mark_frame_with_bbox amber family + legacy badge for old tools"
```

---

## Task 26: NR3D random100 verification + SQLite ingest

**Files:**
- Run scripts/launch: `scripts/run_v9_full_nr3d_random100.sh`
- New doc: `docs/benchmark/nr3d/v9_1_selectors_return_images_20260516.md`

- [ ] **Step 1: Snapshot current branch tip**

```bash
git rev-parse --short HEAD
git log --oneline -1
```
Note the commit for the version doc.

- [ ] **Step 2: Launch the run inside tmux**

```bash
tmux new-session -d -s eval-v91 "bash scripts/run_v9_full_nr3d_random100.sh 2>&1 | tee /tmp/v9_1_eval.log"
```

Wait for completion (typically 60–120 min). Periodically peek:

```bash
tmux capture-pane -t eval-v91 -p -S -10
```

- [ ] **Step 3: Verify metrics**

The run should produce `tmp/openeqa_eval_v9_1_random100/` with `summary.json`. Confirm overall accuracy is within ±2 pp of the v9 clean run (86.0%); otherwise diagnose before continuing.

- [ ] **Step 4: SQLite ingest**

```bash
python scripts/ingest_openeqa_run.py \
    --output-dir tmp/openeqa_eval_v9_1_random100/ \
    --run-id v9_1_selectors_return_images \
    --branch feat/v9-1-selectors-return-images \
    --commit "$(git rev-parse --short HEAD)" \
    --judge-model gemini-2.5-pro \
    --notes "v9.1: selectors return RGB, mark_frame_with_bbox replaces view_keyframe, BEV fixes" \
    --db docs/benchmark/nr3d/runs.sqlite
```

- [ ] **Step 5: Write the version doc + commit**

Create `docs/benchmark/nr3d/v9_1_selectors_return_images_20260516.md` with the standard sections (branch + tip commit, CLI, raw artifact dir, judge model, fold size, headline metric, vs v9 clean comparison, caveats). Update `docs/benchmark/nr3d/README.md` and `leaderboard.md`. Commit:

```bash
git add docs/benchmark/nr3d/
git commit -m "docs(benchmark): NR3D v9.1 random100 results + SQLite ingest"
```

---

## Task 27: Final acceptance checklist

- [ ] **Step 1: Run the full test suite**

```bash
pytest src/ -v
```
All green.

- [ ] **Step 2: Confirm no deleted tools remain anywhere**

```bash
rg -n "view_keyframe|select_by_hypothesis" src/agents/ docs/superpowers/specs/2026-05-15-v9-1-selectors-return-images-design.md
```
Only spec / legacy comment hits acceptable. No active imports or call sites.

- [ ] **Step 3: Push branch + open PR**

```bash
git push -u origin feat/v9-1-selectors-return-images
gh pr create --title "v9.1: selectors return RGB + mark_frame_with_bbox + BEV fixes" --body "$(cat <<'EOF'
## Summary
- All five selectors now inject ≤3 RGB candidate frames per call (Stage-1 catalog-first innovation made first-class).
- New `mark_frame_with_bbox` tool replaces `view_keyframe(mode='marked')`; requires explicit `labels` or `ids`.
- `select_by_hypothesis` and `view_keyframe` deleted.
- BEV fixes: non-white crop, label legibility (font + outline), highlight projection bug, label declutter.
- Playbooks + system prompt updated to "first move = select_by_text".
- Guards (`evidence_frame_guard`, `no_match_guard`) switched to `mark_frame_with_bbox` predicate.

## Test plan
- [x] Unit tests pass: `pytest src/ -v`
- [x] NR3D random100 within ±2 pp of v9 clean baseline
- [x] Trace HTML regenerated, legacy tools show deprecation badge
- [x] SQLite ingest of the v9.1 run

## Spec / Plan
- spec: `docs/superpowers/specs/2026-05-15-v9-1-selectors-return-images-design.md`
- plan: `docs/superpowers/plans/2026-05-15-v9-1-selectors-return-images.md`
EOF
)"
```

---

## Self-Review Notes

**Spec coverage:** Tasks 1–6 cover spec §4.2 (mark tool) + §5 (rendering). Tasks 7–11 cover §4.1 (selectors return images). Task 12 covers `select_by_hypothesis` deletion. Tasks 13–14 cover §7.5 (guards). Task 15 covers `view_keyframe` deletion. Tasks 16–19 cover §6 (BEV fixes). Tasks 20–22 cover §7.1–7.4 (playbooks). Task 23 covers §7.6–7.7 (system prompt). Tasks 24–25 cover §8.3 (trace HTML) + §8.2 (E2E mocks). Tasks 26–27 cover §10 (acceptance).

**Placeholder scan:** No "TBD" / "implement later" / "add error handling" stubs. The two `Step 1–5` shortcut tasks (9, 10, 11, 21, 22) repeat the pattern from a referenced earlier task — engineer reads upstream task for the template, applies same TDD shape with the per-selector or per-playbook specifics.

**Type consistency:** `BBOX_PALETTE`, `BLACK_OUTLINE_PAD`, `LABEL_AREA_RATIO_THRESHOLD`, `LABEL_PADDING_PX`, `LabelAnchor`, `_decide_label_anchor`, `_crop_to_non_white`, `_stack_overlapping_anchors`, `queue_pending_image_if_new`, `_collect_tools_for_task_pack`, `build_mark_frame_with_bbox_tool` — all referenced symbol names match across tasks.

**Inter-task dependencies:** queue helper (T1) ships before any selector uses it (T7+). Mark tool (T2-5) ships before guards switch to it (T13-14). Mark + guards + selectors ship before `view_keyframe` deletion (T15). Playbooks + prompt (T20-23) updated after the tool surface stabilises so docs describe shipped reality.
