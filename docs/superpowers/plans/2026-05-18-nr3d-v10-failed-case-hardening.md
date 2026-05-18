# NR3D v10 Failed-Case Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the v10 no-initial-keyframes VG agent flow so masked-selector errors, target-category drift, stale spatial compares, EFG parsing gaps, and incomplete candidate coverage no longer steer NR3D cases to wrong proposal ids.

**Architecture:** Add deterministic tool/guard contracts before introducing any larger planner: `select_by_text` retries masked leaks internally; `submit_final` runs a VG target-category guard; EFG parses query direction and frame citations more robustly; TADG/EFG can bind to explicit relation evidence ids instead of stale latest compares; playbooks require candidate coverage and unsupported-relation visual workflows. Keep first-person evidence acquisition tool-driven only.

**Tech Stack:** Python 3.12, Pydantic, LangChain tools, Stage2RuntimeState tool traces, pytest, ruff, NR3D strat600 artifacts.

---

## File Structure

- Modify `src/agents/tools/selectors.py`
  - Owns `select_by_text`; add masked-leak retry without changing selector output shape for normal calls.
- Modify `src/agents/tools/tests/test_selectors_text.py`
  - Unit tests for masked-leak retry, retry request recording, and query preservation.
- Create `src/agents/skills/target_category_guard.py`
  - New VG-only soft guard that derives target head/category from query text, tool traces, and proposal labels.
- Modify `src/agents/skills/chassis_tools.py`
  - Wire `target_category_guard` into `submit_final` before TADG/EFG finalization.
- Create `src/agents/tests/test_target_category_guard.py`
  - Unit and chassis tests for `pillow -> bed`, `whiteboard -> chair`, and ambiguous-head pass-through.
- Modify `src/agents/skills/evidence_frame_guard.py`
  - Add target-side direction patterns, plural/range frame citation parsing, current `list_scene_proposals` parsing, and optional relation evidence input.
- Modify `src/agents/tests/test_evidence_frame_guard.py`
  - Tests for `right one`, contrastive rationale text, plural frame citations, and current proposal-list shape.
- Modify `src/agents/packs/vg_embodiedscan/tools.py`
  - Add stable `evidence_id` to `compare_proposals_spatial` responses and trace request payloads.
- Modify `src/agents/skills/tadg.py`
  - Accept optional `relation_evidence`, prefer it over latest-compare inference, and reject role/category-inverted evidence.
- Modify `src/agents/tests/test_tadg.py`
  - Tests for bound evidence beating stale compares and support-category candidates not ranking the target.
- Modify `src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md`
- Modify `src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook_no_text.md`
- Modify `src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation.md`
- Modify `src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation_no_text.md`
- Modify `src/agents/skills/shared_skills/scene_exploration_playbook.md`
  - Document candidate coverage and unsupported semantic relation workflows.
- Modify `src/agents/packs/vg_embodiedscan/skills/tests/test_playbook_v9_consistency.py`
- Modify `src/agents/skills/tests/test_scene_exploration_playbook_loadable.py`
  - Static tests for playbook contract.
- Create `docs/benchmark/nr3d/assets/v10_failed_case_audit40_sample_ids_20260518.json`
  - Durable sample-id slice for replay verification.
- Create `scripts/build_nr3d_v10_failed_case_audit_slice.py`
  - Regenerates the audit slice from the fixed case list and validates each sample exists in strat600.

---

### Task 1: `select_by_text` Masked-Leak Auto-Retry

**Files:**
- Modify: `src/agents/tools/selectors.py`
- Modify: `src/agents/tools/tests/test_selectors_text.py`

- [ ] **Step 1: Add failing selector tests**

Append these tests to `src/agents/tools/tests/test_selectors_text.py`:

```python
class _LeakyThenOkTextFrameSelector:
    def __init__(self, fids: list[int]) -> None:
        self.calls: list[dict] = []
        self._fids = list(fids)

    def select_keyframes_v2(self, **kwargs):
        self.calls.append(dict(kwargs))
        hidden = list(kwargs.get("hidden_categories") or [])
        if hidden:
            raise ValueError("Masked category leak detected: 'wall'")
        return SimpleNamespace(
            keyframe_indices=list(self._fids),
            metadata={
                "hypothesis_output": {
                    "hypotheses": [
                        {
                            "grounding_query": {"root": {"category": "picture"}},
                            "kind": "direct",
                        }
                    ]
                }
            },
        )


def test_select_by_text_auto_retries_masked_category_leak_with_empty_mask(
    tmp_path: Path,
) -> None:
    rs = _runtime(tmp_path, fids=[43, 49, 50])
    selector = _LeakyThenOkTextFrameSelector([43, 49, 50])
    rs.text_frame_selector = selector
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")

    raw = tool.invoke(
        {
            "query": "The largest picture in the room.",
            "hidden_categories": ["wall", "floor"],
        }
    )

    payload = json.loads(raw)
    assert payload["masked_category_retry"]["retried_with_hidden_categories"] == []
    assert "Masked category leak detected" in payload["masked_category_retry"]["error"]
    assert [call["hidden_categories"] for call in selector.calls] == [
        ["wall", "floor"],
        [],
    ]
    assert [call["query"] for call in selector.calls] == [
        "The largest picture in the room.",
        "The largest picture in the room.",
    ]
    assert [frame["frame_id"] for frame in payload["frames"]] == [43, 49, 50]


def test_select_by_text_masked_retry_records_original_request(tmp_path: Path) -> None:
    rs = _runtime(tmp_path, fids=[1])
    selector = _LeakyThenOkTextFrameSelector([1])
    rs.text_frame_selector = selector
    recorded: list[tuple[str, dict, str]] = []
    rs.record = lambda name, request, response: recorded.append((name, request, response))  # type: ignore[method-assign]

    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    raw = tool.invoke({"query": "picture on wall", "hidden_categories": ["wall"]})

    assert len(recorded) == 1
    assert recorded[0][0] == "select_by_text"
    assert recorded[0][1]["hidden_categories"] == ["wall"]
    payload = json.loads(raw)
    assert payload["masked_category_retry"]["retried_with_hidden_categories"] == []
```

- [ ] **Step 2: Run red test**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  src/agents/tools/tests/test_selectors_text.py::test_select_by_text_auto_retries_masked_category_leak_with_empty_mask \
  src/agents/tools/tests/test_selectors_text.py::test_select_by_text_masked_retry_records_original_request \
  -q
```

Expected: both tests fail because `select_by_text` currently returns an
`ERROR:` string after the first masked leak and never retries.

- [ ] **Step 3: Implement retry helper**

In `src/agents/tools/selectors.py`, inside `select_by_text`, replace the single
`try/except` around `selector.select_keyframes_v2(...)` with a small helper:

```python
def _select_text_frames_once(hidden: list[str]):
    return selector.select_keyframes_v2(
        query=str(query),
        k=capped,
        hidden_categories=list(hidden),
        use_visual_context=False,
    )

hidden_in = list(hidden_categories or [])
masked_retry: dict | None = None
try:
    result = _select_text_frames_once(hidden_in)
    effective_hidden_categories = hidden_in
except Exception as exc:  # noqa: BLE001 — fail-loud with deterministic retry
    first_err = f"{type(exc).__name__}: {exc}"
    if hidden_in and "Masked category leak detected" in str(exc):
        try:
            result = _select_text_frames_once([])
            effective_hidden_categories = []
            masked_retry = {
                "original_hidden_categories": hidden_in,
                "error": first_err,
                "retried_with_hidden_categories": [],
            }
        except Exception as retry_exc:  # noqa: BLE001
            err = (
                "ERROR: Stage-1 parse/exec failed after masked-category retry: "
                f"first={first_err}; retry={type(retry_exc).__name__}: {retry_exc}"
            )
            runtime.record("select_by_text", request, err)
            return err
    else:
        err = f"ERROR: Stage-1 parse/exec failed: {first_err}"
        runtime.record("select_by_text", request, err)
        return err
```

Then pass `effective_hidden_categories` into `_build_frame_payload(...)`, and
add the retry metadata to the response only when it exists:

```python
payload = {"hypothesis_summary": summary + k_warning, "frames": frames}
if masked_retry is not None:
    payload["masked_category_retry"] = masked_retry
```

- [ ] **Step 4: Run selector tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tools/tests/test_selectors_text.py -q
```

Expected: all selector-text tests pass.

- [ ] **Step 5: Lint and commit**

Run:

```bash
.venv/bin/python -m ruff check src/agents/tools/selectors.py src/agents/tools/tests/test_selectors_text.py
```

Expected: `All checks passed!`

Commit:

```bash
git add src/agents/tools/selectors.py src/agents/tools/tests/test_selectors_text.py
git commit -m "fix(vg): auto-retry masked text selector leaks"
```

---

### Task 2: Target-Head Category Guard

**Files:**
- Create: `src/agents/skills/target_category_guard.py`
- Modify: `src/agents/skills/chassis_tools.py`
- Test: `src/agents/tests/test_target_category_guard.py`

- [ ] **Step 1: Write failing guard tests**

Create `src/agents/tests/test_target_category_guard.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.packs.vg_embodiedscan.ctx import Proposal, VgEmbodiedScanCtx
from agents.runtime.base import Stage2RuntimeState
from agents.skills.target_category_guard import evaluate_target_category_guard


def _runtime(query: str) -> Stage2RuntimeState:
    rs = Stage2RuntimeState(bundle=Stage2EvidenceBundle(stage1_query=query))
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.task_ctx = VgEmbodiedScanCtx(
        proposal_pool_source="unit",
        proposals=[
            Proposal(id=8, category="bed", score=0.9, bbox_3d_9dof=[0] * 9),
            Proposal(id=46, category="pillow", score=0.8, bbox_3d_9dof=[1] * 9),
            Proposal(id=13, category="whiteboard", score=0.7, bbox_3d_9dof=[2] * 9),
            Proposal(id=6, category="chair", score=0.6, bbox_3d_9dof=[3] * 9),
        ],
        frame_index={},
        proposal_index={},
    )
    return rs


def test_target_category_guard_blocks_context_object_submission() -> None:
    rs = _runtime("Staring at both beds from their foot, you want the bed on the right. The pillow is the back right option.")

    decision = evaluate_target_category_guard(rs, {"proposal_id": 8})

    assert decision.blocked is True
    assert decision.expected_category == "pillow"
    assert decision.submitted_category == "bed"
    assert "TARGET_CATEGORY_GUARD" in decision.message


def test_target_category_guard_allows_matching_head_category() -> None:
    rs = _runtime("The pillow is the back right option.")

    decision = evaluate_target_category_guard(rs, {"proposal_id": 46})

    assert decision.blocked is False
    assert decision.expected_category == "pillow"
    assert decision.submitted_category == "pillow"


def test_target_category_guard_passes_ambiguous_head() -> None:
    rs = _runtime("It is the one on the left.")

    decision = evaluate_target_category_guard(rs, {"proposal_id": 8})

    assert decision.blocked is False
    assert decision.expected_category is None


def test_target_category_guard_reads_current_list_scene_proposals_shape() -> None:
    rs = _runtime("The whiteboard that has a blue note on top.")
    rs.tool_trace.append(
        SimpleNamespace(
            tool_name="list_scene_proposals",
            response_text='{"count":1,"proposals":[{"proposal_id":13,"category":"whiteboard"}]}',
        )
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 6})

    assert decision.blocked is True
    assert decision.expected_category == "whiteboard"
    assert decision.submitted_category == "chair"
```

- [ ] **Step 2: Run red test**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tests/test_target_category_guard.py -q
```

Expected: import failure because `agents.skills.target_category_guard` does not
exist.

- [ ] **Step 3: Implement `target_category_guard.py`**

Create `src/agents/skills/target_category_guard.py` with:

```python
"""Target-head/category guard for VG submit_final."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_ALIASES = {
    "bookcase": "bookshelf",
    "book case": "bookshelf",
    "white board": "whiteboard",
    "trashcan": "trash can",
}


@dataclass(frozen=True)
class TargetCategoryDecision:
    blocked: bool
    message: str = ""
    submitted_pid: int | None = None
    expected_category: str | None = None
    submitted_category: str | None = None


def _norm_label(value: str) -> str:
    text = " ".join(str(value).lower().split())
    return _ALIASES.get(text, text)


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm_label(value))


def _payload_dict(payload: dict | Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    inner = payload.get("payload")
    if isinstance(inner, dict) and "proposal_id" in inner:
        return inner
    return payload


def _proposal_category(runtime: Any, proposal_id: int) -> str | None:
    ctx = getattr(runtime, "task_ctx", None)
    for proposal in list(getattr(ctx, "proposals", []) or []):
        if int(getattr(proposal, "id", -999999)) == proposal_id:
            return str(getattr(proposal, "category", "") or "")
    return None


def _categories_from_current_list_scene_trace(runtime: Any) -> list[str]:
    categories: list[str] = []
    for entry in reversed(list(getattr(runtime, "tool_trace", []) or [])):
        if str(getattr(entry, "tool_name", "") or "") != "list_scene_proposals":
            continue
        response_text = str(getattr(entry, "response_text", "") or "")
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            continue
        for row in payload.get("proposals", []) or []:
            if isinstance(row, dict) and row.get("category"):
                cat = _norm_label(str(row["category"]))
                if cat not in categories:
                    categories.append(cat)
        if categories:
            return categories
    return categories


def _query_category(runtime: Any) -> str | None:
    query = str(getattr(getattr(runtime, "bundle", None), "stage1_query", "") or "")
    catalog_categories = [
        _norm_label(str(getattr(p, "category", "") or ""))
        for p in list(getattr(getattr(runtime, "task_ctx", None), "proposals", []) or [])
    ]
    trace_categories = _categories_from_current_list_scene_trace(runtime)
    candidates = trace_categories or sorted(set(catalog_categories), key=len, reverse=True)
    compact_query = _compact(query)
    hits = [cat for cat in candidates if cat and _compact(cat) in compact_query]
    if len(set(hits)) == 1:
        return hits[0]
    if trace_categories and len(set(trace_categories)) == 1:
        return trace_categories[0]
    return None


def _compatible(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    ac = _compact(a)
    bc = _compact(b)
    return ac == bc or ac in bc or bc in ac


def evaluate_target_category_guard(runtime: Any, payload: dict | Any) -> TargetCategoryDecision:
    submitted_pid = _payload_dict(payload).get("proposal_id")
    if not isinstance(submitted_pid, int) or submitted_pid == -1:
        return TargetCategoryDecision(blocked=False)
    expected = _query_category(runtime)
    submitted = _proposal_category(runtime, submitted_pid)
    if expected is None or submitted is None:
        return TargetCategoryDecision(
            blocked=False,
            submitted_pid=submitted_pid,
            expected_category=expected,
            submitted_category=submitted,
        )
    if _compatible(expected, submitted):
        return TargetCategoryDecision(
            blocked=False,
            submitted_pid=submitted_pid,
            expected_category=expected,
            submitted_category=submitted,
        )
    message = (
        "TARGET_CATEGORY_GUARD: query target appears to be "
        f"{expected!r}, but submitted proposal {submitted_pid} is category "
        f"{submitted!r}. Submit a {expected} proposal, or explain an explicit "
        "referent shift with marked evidence."
    )
    return TargetCategoryDecision(
        blocked=True,
        message=message,
        submitted_pid=submitted_pid,
        expected_category=expected,
        submitted_category=submitted,
    )


def target_category_guard_record_fields(
    decision: TargetCategoryDecision,
) -> dict[str, Any]:
    return {
        "target_category_guard_blocked": bool(decision.blocked),
        "target_category_guard_submitted_pid": decision.submitted_pid,
        "target_category_guard_expected_category": decision.expected_category,
        "target_category_guard_submitted_category": decision.submitted_category,
        "target_category_guard_message": decision.message or None,
    }
```

- [ ] **Step 4: Wire guard into `submit_final`**

In `src/agents/skills/chassis_tools.py`:

1. Import:

```python
from agents.skills.target_category_guard import (
    evaluate_target_category_guard,
    target_category_guard_record_fields,
)
```

2. After `gate_payload` is computed and before `evaluate_tadg(...)`, add:

```python
target_category_decision = evaluate_target_category_guard(runtime, gate_payload)
if target_category_decision.blocked:
    runtime.record(
        "submit_final",
        {
            "payload": payload,
            "rationale": rationale,
            "evidence_refs": evidence_refs or [],
            "tool_override_reason": tool_override_reason,
            **target_category_guard_record_fields(target_category_decision),
        },
        target_category_decision.message,
    )
    return target_category_decision.message
```

3. Include `**target_category_guard_record_fields(target_category_decision)` in
   every later `submit_final` `runtime.record(...)` payload, including success.

- [ ] **Step 5: Run guard tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  src/agents/tests/test_target_category_guard.py \
  src/agents/tests/test_tadg.py \
  src/agents/tests/test_evidence_frame_guard.py \
  -q
```

Expected: all tests pass. If existing TADG/EFG tests instantiate runtimes with
queries that now confidently imply a category mismatch, update those test
payloads to use matching proposal categories rather than disabling the guard.

- [ ] **Step 6: Lint and commit**

Run:

```bash
.venv/bin/python -m ruff check \
  src/agents/skills/target_category_guard.py \
  src/agents/skills/chassis_tools.py \
  src/agents/tests/test_target_category_guard.py
```

Expected: `All checks passed!`

Commit:

```bash
git add src/agents/skills/target_category_guard.py src/agents/skills/chassis_tools.py src/agents/tests/test_target_category_guard.py
git commit -m "fix(vg): guard target category drift"
```

---

### Task 3: Evidence-Frame Guard Direction And Citation Parsing

**Files:**
- Modify: `src/agents/skills/evidence_frame_guard.py`
- Modify: `src/agents/tests/test_evidence_frame_guard.py`

- [ ] **Step 1: Add failing EFG tests**

Append to `src/agents/tests/test_evidence_frame_guard.py`:

```python
def test_evidence_frame_guard_query_right_one_beats_contrastive_rationale_left(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(
            stage1_query="When facing the two tables choose the one on the right."
        )
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = True
    _record_view(
        rs,
        frame_id=8,
        visible_ids=[16, 17],
        categories=["table", "table"],
        left_to_right=["16:table@x=200.0", "17:table@x=500.0"],
        boxes_2d={16: [100, 200, 300, 650], 17: [400, 200, 620, 650]},
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 17, "confidence": 0.7},
            "rationale": (
                "Frame 8 shows proposal 17 as the right table; proposal 16 is "
                "the left-hand alternative."
            ),
            "evidence_refs": [],
        }
    )

    assert response.startswith("submitted;")


def test_evidence_frame_guard_parses_plural_frame_citations(tmp_path: Path) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = _runtime()
    rs.use_evidence_frame_guard = True
    _record_view(rs, frame_id=0, visible_ids=[23], categories=["trash can"])
    _record_view(rs, frame_id=1, visible_ids=[27], categories=["trash can"])
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 27, "confidence": 0.7},
            "rationale": "Frames 0, 1, and 2 show proposal 27 by the outlet.",
            "evidence_refs": [],
        }
    )

    assert response.startswith("submitted;")
    submit_records = [t for t in rs.tool_trace if t.tool_name == "submit_final"]
    assert submit_records[0].tool_input["evidence_frame_guard_cited_frame_ids"] == [0, 1]


def test_evidence_frame_guard_parses_frame_range_without_marked_evidence(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = _runtime()
    rs.use_evidence_frame_guard = True
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 27, "confidence": 0.7},
            "rationale": "Frames 0-2 show the final object.",
            "evidence_refs": [],
        }
    )

    assert response.startswith("EVIDENCE_FRAME_GUARD:")
    assert "0, 1, 2" in response
```

- [ ] **Step 2: Run red tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  src/agents/tests/test_evidence_frame_guard.py::test_evidence_frame_guard_query_right_one_beats_contrastive_rationale_left \
  src/agents/tests/test_evidence_frame_guard.py::test_evidence_frame_guard_parses_plural_frame_citations \
  src/agents/tests/test_evidence_frame_guard.py::test_evidence_frame_guard_parses_frame_range_without_marked_evidence \
  -q
```

Expected: at least the first and third tests fail because direction/citation
parsing is too narrow.

- [ ] **Step 3: Implement direction precedence**

In `src/agents/skills/evidence_frame_guard.py`, add target-side patterns:

```python
_LEFT_TARGET_SIDE_RE = re.compile(
    r"\b(?:on|at|in|towards?)\s+the\s+left\b|"
    r"\bleft\s+(?:one|option|side|corner|hand\s+side)\b|"
    r"\b(?:upper|lower|back|front)\s+left\b",
    re.I,
)
_RIGHT_TARGET_SIDE_RE = re.compile(
    r"\b(?:on|at|in|towards?)\s+the\s+right\b|"
    r"\bright\s+(?:one|option|side|corner|hand\s+side)\b|"
    r"\b(?:upper|lower|back|front)\s+right\b",
    re.I,
)
_ALTERNATIVE_DIRECTION_RE = re.compile(
    r"\b(?:alternative|other|not\s+the|proposal\s+\d+)\b[^.?!]{0,80}"
    r"\b(?:left|right)(?:-hand)?\b",
    re.I,
)
```

Then update `_desired_relative_direction(...)`:

```python
def _direction_from_text(text: str) -> str | None:
    if _LEFT_TARGET_SIDE_RE.search(text or ""):
        return "left"
    if _RIGHT_TARGET_SIDE_RE.search(text or ""):
        return "right"
    if _LEFT_RELATION_RE.search(text or ""):
        return "left"
    if _RIGHT_RELATION_RE.search(text or ""):
        return "right"
    return None


def _desired_relative_direction(runtime: Any, rationale: str) -> str | None:
    bundle = getattr(runtime, "bundle", None)
    query = str(getattr(bundle, "stage1_query", "") or "")
    if _anchor_relative_to_target_direction(query) is not None:
        return None
    query_direction = _direction_from_text(query)
    if query_direction is not None:
        return query_direction
    if _anchor_relative_to_target_direction(rationale or "") is not None:
        return None
    if _ALTERNATIVE_DIRECTION_RE.search(rationale or ""):
        return None
    return _direction_from_text(rationale or "")
```

- [ ] **Step 4: Implement plural/range citation parsing**

Replace `_cited_frame_ids(...)` with parser helpers:

```python
_FRAME_RANGE_RE = re.compile(r"\bframes?\s+(\d+)\s*[-–]\s*(\d+)\b", re.I)
_FRAMES_LIST_RE = re.compile(
    r"\bframes?\s+((?:\d+\s*(?:,|and)?\s*){2,})",
    re.I,
)


def _append_frame_id(frame_ids: list[int], frame_id: int) -> None:
    if frame_id not in frame_ids:
        frame_ids.append(frame_id)


def _cited_frame_ids(rationale: str, evidence_refs: list[dict] | None) -> list[int]:
    frame_ids: list[int] = []
    text = rationale or ""
    for match in _FRAME_RANGE_RE.finditer(text):
        start = int(match.group(1))
        end = int(match.group(2))
        lo, hi = sorted((start, end))
        for frame_id in range(lo, hi + 1):
            _append_frame_id(frame_ids, frame_id)
    for match in _FRAMES_LIST_RE.finditer(text):
        for raw in re.findall(r"\d+", match.group(1)):
            _append_frame_id(frame_ids, int(raw))
    for match in _FRAME_CITATION_RE.finditer(text):
        _append_frame_id(frame_ids, int(match.group(1)))
    for ref in evidence_refs or []:
        if not isinstance(ref, dict):
            continue
        for key in ("frame_id", "frame"):
            value = ref.get(key)
            if isinstance(value, int):
                _append_frame_id(frame_ids, value)
    return frame_ids
```

- [ ] **Step 5: Run EFG tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/agents/tests/test_evidence_frame_guard.py -q
```

Expected: all EFG tests pass.

- [ ] **Step 6: Lint and commit**

Run:

```bash
.venv/bin/python -m ruff check src/agents/skills/evidence_frame_guard.py src/agents/tests/test_evidence_frame_guard.py
```

Expected: `All checks passed!`

Commit:

```bash
git add src/agents/skills/evidence_frame_guard.py src/agents/tests/test_evidence_frame_guard.py
git commit -m "fix(vg): harden evidence frame parsing"
```

---

### Task 4: Relation Evidence Binding For TADG And EFG

**Files:**
- Modify: `src/agents/packs/vg_embodiedscan/tools.py`
- Modify: `src/agents/skills/chassis_tools.py`
- Modify: `src/agents/skills/tadg.py`
- Modify: `src/agents/skills/evidence_frame_guard.py`
- Modify: `src/agents/packs/vg_embodiedscan/tests/test_tools.py`
- Modify: `src/agents/tests/test_tadg.py`
- Modify: `src/agents/tests/test_evidence_frame_guard.py`

- [ ] **Step 1: Add failing `compare_proposals_spatial` evidence-id test**

Append to `src/agents/packs/vg_embodiedscan/tests/test_tools.py`:

```python
def test_compare_proposals_spatial_returns_stable_evidence_id(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")

    payload = json.loads(
        tool.invoke({"candidate_ids": [0, 2], "anchor_id": 1, "relation": "left_of"})
    )

    assert payload["evidence_id"].startswith("compare_proposals_spatial:")
    assert payload["candidate_ids"] == [0, 2]
    assert payload["anchor_id"] == 1
    assert payload["relation"] == "left_of"
```

- [ ] **Step 2: Add failing TADG bound-evidence tests**

Append to `src/agents/tests/test_tadg.py`:

```python
def test_tadg_uses_bound_relation_evidence_over_stale_latest_compare() -> None:
    rs = _runtime(bundle=_bundle_with_query("the door nearest the small black chair"))
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=13,
        candidate_ids=[5],
        ranked_ids=[5],
    )
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=37,
        candidate_ids=[39, 5],
        ranked_ids=[39, 5],
    )

    decision = evaluate_tadg(
        rs,
        {"proposal_id": 39, "confidence": 0.8},
        relation_evidence={
            "relation": "closest_to",
            "anchor_id": 37,
            "candidate_ids": [39, 5],
            "ranked_ids": [39, 5],
        },
    )

    assert decision.blocked is False


def test_tadg_rejects_bound_anchor_self_without_forcing_rank1() -> None:
    rs = _runtime(bundle=_bundle_with_query("trash can next to two red chairs"))

    decision = evaluate_tadg(
        rs,
        {"proposal_id": 36, "confidence": 0.8},
        relation_evidence={
            "relation": "next_to",
            "anchor_id": 36,
            "candidate_ids": [39, 40, 41],
            "ranked_ids": [39, 40, 41],
        },
    )

    assert decision.blocked is True
    assert decision.subcase == "anchor_self"
    assert "target/anchor role" in decision.message
    assert "Revise to proposal 39" not in decision.message
```

- [ ] **Step 3: Run red tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  src/agents/packs/vg_embodiedscan/tests/test_tools.py::test_compare_proposals_spatial_returns_stable_evidence_id \
  src/agents/tests/test_tadg.py::test_tadg_uses_bound_relation_evidence_over_stale_latest_compare \
  src/agents/tests/test_tadg.py::test_tadg_rejects_bound_anchor_self_without_forcing_rank1 \
  -q
```

Expected: evidence-id test fails because payload lacks `evidence_id`; TADG tests
fail because `relation_evidence` is not an accepted argument.

- [ ] **Step 4: Add evidence id to compare tool**

In `src/agents/packs/vg_embodiedscan/tools.py`, before building the response:

```python
compare_index = sum(
    1
    for entry in list(getattr(runtime, "tool_trace", []) or [])
    if getattr(entry, "tool_name", None) == "compare_proposals_spatial"
)
evidence_id = f"compare_proposals_spatial:{compare_index}"
```

Add to payload:

```python
"evidence_id": evidence_id,
"candidate_ids": list(candidate_ids),
```

Add to `request` before `runtime.record(...)`:

```python
request["evidence_id"] = evidence_id
```

- [ ] **Step 5: Teach TADG to accept bound relation evidence**

In `src/agents/skills/tadg.py`:

1. Add parameter:

```python
def evaluate_tadg(
    runtime: Any,
    payload: dict | Any,
    *,
    tool_override_reason: str | None = None,
    relation_evidence: dict[str, Any] | None = None,
) -> TADGDecision:
```

2. Add parser:

```python
def _bound_compare(relation_evidence: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(relation_evidence, dict):
        return None
    ranked_ids = relation_evidence.get("ranked_ids") or []
    candidate_ids = relation_evidence.get("candidate_ids") or []
    relation = _canonical_compare_relation(relation_evidence.get("relation", ""))
    anchor_id = relation_evidence.get("anchor_id")
    if not relation or not isinstance(anchor_id, int):
        return None
    return {
        "relation": relation,
        "anchor_id": anchor_id,
        "candidate_ids": [int(pid) for pid in candidate_ids if isinstance(pid, int)],
        "ranked_ids": [int(pid) for pid in ranked_ids if isinstance(pid, int)],
        "supporting_frame_counts": relation_evidence.get("supporting_frame_counts") or [],
        "contradicting_frame_counts": relation_evidence.get("contradicting_frame_counts") or [],
    }
```

3. Replace compare selection with:

```python
compare = _bound_compare(relation_evidence)
if compare is None:
    relevant_relations = _query_relation_set(runtime)
    compare = _last_matching_compare(runtime, relevant_relations)
else:
    relevant_relations = {compare["relation"]}
```

- [ ] **Step 6: Wire `relation_evidence` through chassis**

In `src/agents/skills/chassis_tools.py`, update signature:

```python
def submit_final(
    payload: dict,
    rationale: str,
    evidence_refs: list[dict] | None = None,
    tool_override_reason: str | None = None,
    relation_evidence: dict | None = None,
) -> str:
```

Pass it to `evaluate_tadg(...)`:

```python
decision = evaluate_tadg(
    runtime,
    gate_payload,
    tool_override_reason=tool_override_reason,
    relation_evidence=relation_evidence,
)
```

Add `"relation_evidence": relation_evidence` to every `runtime.record(...)`
payload in `submit_final`.

- [ ] **Step 7: Run relation-binding tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  src/agents/packs/vg_embodiedscan/tests/test_tools.py \
  src/agents/tests/test_tadg.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 8: Extend EFG to accept relation evidence**

In `src/agents/skills/evidence_frame_guard.py`, change signature:

```python
def evaluate_evidence_frame_guard(
    runtime: Any,
    payload: dict | Any,
    *,
    rationale: str,
    evidence_refs: list[dict] | None = None,
    relation_evidence: dict[str, Any] | None = None,
) -> EvidenceFrameGuardDecision:
```

Add:

```python
def _spatial_compare_from_relation_evidence(
    relation_evidence: dict[str, Any] | None,
    submitted_pid: int,
) -> dict[str, Any] | None:
    if not isinstance(relation_evidence, dict):
        return None
    candidate_ids = relation_evidence.get("candidate_ids") or []
    if submitted_pid not in candidate_ids:
        return None
    anchor_id = relation_evidence.get("anchor_id")
    relation = relation_evidence.get("relation")
    ranked_ids = relation_evidence.get("ranked_ids") or []
    if not isinstance(anchor_id, int) or not isinstance(relation, str):
        return None
    return {"anchor_id": anchor_id, "relation": relation, "ranked_ids": ranked_ids}
```

Then replace:

```python
spatial_compare = _latest_spatial_compare_for_submission(runtime, submitted_pid)
```

with:

```python
spatial_compare = _spatial_compare_from_relation_evidence(
    relation_evidence,
    submitted_pid,
) or _latest_spatial_compare_for_submission(runtime, submitted_pid)
```

Wire from `chassis_tools.py` into `evaluate_evidence_frame_guard(...)`.

- [ ] **Step 9: Lint and commit**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  src/agents/packs/vg_embodiedscan/tests/test_tools.py \
  src/agents/tests/test_tadg.py \
  src/agents/tests/test_evidence_frame_guard.py \
  -q
.venv/bin/python -m ruff check \
  src/agents/packs/vg_embodiedscan/tools.py \
  src/agents/skills/chassis_tools.py \
  src/agents/skills/tadg.py \
  src/agents/skills/evidence_frame_guard.py \
  src/agents/packs/vg_embodiedscan/tests/test_tools.py \
  src/agents/tests/test_tadg.py \
  src/agents/tests/test_evidence_frame_guard.py
```

Expected: tests pass and ruff passes.

Commit:

```bash
git add \
  src/agents/packs/vg_embodiedscan/tools.py \
  src/agents/skills/chassis_tools.py \
  src/agents/skills/tadg.py \
  src/agents/skills/evidence_frame_guard.py \
  src/agents/packs/vg_embodiedscan/tests/test_tools.py \
  src/agents/tests/test_tadg.py \
  src/agents/tests/test_evidence_frame_guard.py
git commit -m "fix(vg): bind final answers to relation evidence"
```

---

### Task 5: Candidate Coverage And Unsupported-Relation Playbooks

**Files:**
- Modify: `src/agents/skills/tadg.py`
- Modify: `src/agents/skills/evidence_frame_guard.py`
- Modify: `src/agents/tests/test_tadg.py`
- Modify: `src/agents/tests/test_evidence_frame_guard.py`
- Modify: `src/agents/skills/shared_skills/scene_exploration_playbook.md`
- Modify: `src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md`
- Modify: `src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook_no_text.md`
- Modify: `src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation.md`
- Modify: `src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation_no_text.md`
- Modify: `src/agents/skills/tests/test_scene_exploration_playbook_loadable.py`
- Modify: `src/agents/packs/vg_embodiedscan/skills/tests/test_playbook_v9_consistency.py`

- [ ] **Step 1: Add failing current-list-shape tests**

Append to `src/agents/tests/test_tadg.py`:

```python
def test_tadg_candidate_coverage_reads_current_list_scene_proposals_shape() -> None:
    rs = _runtime(bundle=_bundle_with_query("the chair nearest the whiteboard"))
    rs.tool_trace.append(
        Stage2ToolObservation(
            tool_name="list_scene_proposals",
            tool_input={"category": "chair"},
            response_text=json.dumps(
                {
                    "count": 3,
                    "proposals": [
                        {"proposal_id": 19, "category": "chair"},
                        {"proposal_id": 21, "category": "chair"},
                        {"proposal_id": 22, "category": "chair"},
                    ],
                }
            ),
        )
    )
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=2,
        candidate_ids=[21],
        ranked_ids=[21],
    )

    decision = evaluate_tadg(rs, {"proposal_id": 21, "confidence": 0.8})

    assert decision.blocked is True
    assert decision.subcase == "candidate_coverage_gap"
    assert "proposal 19" in decision.message
    assert "proposal 22" in decision.message
```

Append to `src/agents/tests/test_evidence_frame_guard.py`:

```python
def test_evidence_frame_guard_candidate_ids_reads_current_list_scene_proposals_shape(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(stage1_query="choose the table on the right")
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = True
    rs.tool_trace.append(
        Stage2ToolObservation(
            tool_name="list_scene_proposals",
            tool_input={"category": "table"},
            response_text=json.dumps(
                {
                    "count": 2,
                    "proposals": [
                        {"proposal_id": 16, "category": "table"},
                        {"proposal_id": 17, "category": "table"},
                    ],
                }
            ),
        )
    )
    _record_view(
        rs,
        frame_id=8,
        visible_ids=[16, 17, 99],
        categories=["table", "table", "cabinet"],
        left_to_right=["16:table@x=100.0", "17:table@x=300.0", "99:cabinet@x=500.0"],
        boxes_2d={
            16: [80, 100, 180, 500],
            17: [260, 100, 360, 500],
            99: [450, 100, 650, 500],
        },
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 17, "confidence": 0.7},
            "rationale": "Frame 8 shows proposal 17 as the table on the right.",
            "evidence_refs": [],
        }
    )

    assert response.startswith("submitted;")
```

- [ ] **Step 2: Run red tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  src/agents/tests/test_tadg.py::test_tadg_candidate_coverage_reads_current_list_scene_proposals_shape \
  src/agents/tests/test_evidence_frame_guard.py::test_evidence_frame_guard_candidate_ids_reads_current_list_scene_proposals_shape \
  -q
```

Expected: TADG coverage test fails because `_candidate_coverage_gap` only reads
legacy `proposal_ids`. EFG test may pass after Task 3/4; keep it to pin the
current `proposals[].proposal_id` shape.

- [ ] **Step 3: Add shared proposal-id parser in each guard**

In both `src/agents/skills/tadg.py` and
`src/agents/skills/evidence_frame_guard.py`, add local helper:

```python
def _proposal_ids_from_list_scene_response(response: dict[str, Any]) -> list[int]:
    raw_ids = response.get("proposal_ids")
    if isinstance(raw_ids, list):
        return [int(pid) for pid in raw_ids if isinstance(pid, int)]
    rows = response.get("proposals")
    if isinstance(rows, list):
        ids: list[int] = []
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("proposal_id"), int):
                ids.append(int(row["proposal_id"]))
        return ids
    return []
```

Use it in `_candidate_coverage_gap(...)`, `_ambiguous_anchor_gap(...)`, and
`_candidate_ids_for_submitted_pid(...)`.

- [ ] **Step 4: Update playbooks with unsupported relation workflows**

In both VG playbooks, add this paragraph under spatial disambiguation:

```markdown
Unsupported semantic relations are visual/BEV workflows, not relation strings.
Do not call `compare_proposals_spatial` with `same_side_as`, `between`,
`opposite`, `across_from`, `facing`, `in_front_of`, or `behind`. For those,
mark the target candidates and anchor(s), use BEV/3D positions for room-side or
between/opposite checks, and cite the marked frames used for appearance.
For negated relations like "not closer to X", first compare the positive
relation to identify candidates to avoid, then choose among the remaining
target-category candidates.
```

In `scene_exploration_playbook.md`, add candidate-coverage rule:

```markdown
Before finalizing an ordering, closest/farthest, size, or superlative query,
make sure every plausible same-category candidate in a small candidate set has
marked evidence or an explicit elimination reason. A selector returning frames
for only one spatial cluster is not enough to eliminate unseen same-category
candidates.
```

- [ ] **Step 5: Add static playbook tests**

Append to `src/agents/packs/vg_embodiedscan/skills/tests/test_playbook_v9_consistency.py`:

```python
def test_vg_playbooks_route_unsupported_relations_to_visual_workflow() -> None:
    for path in (_PB, _PB_NO_TEXT, _SD, _SD_NO_TEXT):
        text = path.read_text()
        assert "Unsupported semantic relations" in text
        assert "same_side_as" in text
        assert "between" in text
        assert "Do not call `compare_proposals_spatial`" in text
```

Append to `src/agents/skills/tests/test_scene_exploration_playbook_loadable.py`:

```python
def test_scene_exploration_playbook_requires_candidate_coverage_before_final():
    body = _PLAYBOOK_PATH.read_text()
    assert "every plausible same-category candidate" in body
    assert "marked evidence or an explicit elimination reason" in body
```

- [ ] **Step 6: Run tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  src/agents/tests/test_tadg.py \
  src/agents/tests/test_evidence_frame_guard.py \
  src/agents/packs/vg_embodiedscan/skills/tests/test_playbook_v9_consistency.py \
  src/agents/skills/tests/test_scene_exploration_playbook_loadable.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 7: Lint and commit**

Run:

```bash
.venv/bin/python -m ruff check \
  src/agents/skills/tadg.py \
  src/agents/skills/evidence_frame_guard.py \
  src/agents/tests/test_tadg.py \
  src/agents/tests/test_evidence_frame_guard.py \
  src/agents/packs/vg_embodiedscan/skills/tests/test_playbook_v9_consistency.py \
  src/agents/skills/tests/test_scene_exploration_playbook_loadable.py
```

Expected: `All checks passed!`

Commit:

```bash
git add \
  src/agents/skills/tadg.py \
  src/agents/skills/evidence_frame_guard.py \
  src/agents/tests/test_tadg.py \
  src/agents/tests/test_evidence_frame_guard.py \
  src/agents/skills/shared_skills/scene_exploration_playbook.md \
  src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md \
  src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook_no_text.md \
  src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation.md \
  src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation_no_text.md \
  src/agents/packs/vg_embodiedscan/skills/tests/test_playbook_v9_consistency.py \
  src/agents/skills/tests/test_scene_exploration_playbook_loadable.py
git commit -m "fix(vg): require candidate coverage for semantic relations"
```

---

### Task 6: Audit Slice Artifact And Verification Commands

**Files:**
- Create: `scripts/build_nr3d_v10_failed_case_audit_slice.py`
- Create: `docs/benchmark/nr3d/assets/v10_failed_case_audit40_sample_ids_20260518.json`
- Modify: `docs/benchmark/nr3d/v10_no_initial_keyframes_strat600_20260518.md`

- [ ] **Step 1: Add audit slice builder**

Create `scripts/build_nr3d_v10_failed_case_audit_slice.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

AUDIT_SAMPLE_IDS = [
    "scannet/scene0696_00::21::29050",
    "scannet/scene0704_00::10::7262",
    "scannet/scene0699_00::26::40486",
    "scannet/scene0025_00::26::38922",
    "scannet/scene0095_00::23::37291",
    "scannet/scene0208_00::106::14558",
    "scannet/scene0222_00::20::39339",
    "scannet/scene0025_00::17::37165",
    "scannet/scene0030_00::18::30641",
    "scannet/scene0231_00::45::38236",
    "scannet/scene0565_00::4::9604",
    "scannet/scene0629_00::14::17248",
    "scannet/scene0011_00::23::15020",
    "scannet/scene0030_00::23::28411",
    "scannet/scene0081_00::5::26430",
    "scannet/scene0144_00::17::37726",
    "scannet/scene0221_00::13::4514",
    "scannet/scene0329_00::39::35152",
    "scannet/scene0378_00::41::26109",
    "scannet/scene0249_00::36::39983",
    "scannet/scene0249_00::23::30460",
    "scannet/scene0690_00::17::19402",
    "scannet/scene0221_00::46::36120",
    "scannet/scene0343_00::16::35014",
    "scannet/scene0644_00::35::19090",
    "scannet/scene0208_00::17::21496",
    "scannet/scene0222_00::7::10110",
    "scannet/scene0496_00::29::10819",
    "scannet/scene0565_00::23::30788",
    "scannet/scene0568_00::0::41395",
    "scannet/scene0647_00::17::34181",
    "scannet/scene0591_00::5::17135",
    "scannet/scene0655_00::19::31620",
    "scannet/scene0651_00::7::6342",
    "scannet/scene0222_00::13::738",
    "scannet/scene0351_00::22::23771",
    "scannet/scene0081_00::0::10619",
    "scannet/scene0246_00::39::37886",
    "scannet/scene0329_00::6::26630",
    "scannet/scene0095_00::29::20881",
    "scannet/scene0490_00::13::36559",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strat600", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    strat600 = set(json.loads(args.strat600.read_text()))
    missing = [sid for sid in AUDIT_SAMPLE_IDS if sid not in strat600]
    if missing:
        raise SystemExit(f"audit sample ids missing from strat600: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(AUDIT_SAMPLE_IDS, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(AUDIT_SAMPLE_IDS)} sample ids to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Generate durable audit slice**

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_nr3d_v10_failed_case_audit_slice.py \
  --strat600 tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --output docs/benchmark/nr3d/assets/v10_failed_case_audit40_sample_ids_20260518.json
```

Expected output:

```text
wrote 41 sample ids to docs/benchmark/nr3d/assets/v10_failed_case_audit40_sample_ids_20260518.json
```

The file name keeps `audit40` because the audit was planned as 40 cases; the
actual durable list contains 41 ids due to the positive recovery example
`scene0699_00::26::40486` being kept as a guardrail.

- [ ] **Step 3: Add replay command to benchmark doc**

Append to `docs/benchmark/nr3d/v10_no_initial_keyframes_strat600_20260518.md`:

```markdown
## Failed-case hardening replay slice

The follow-up hardening spec/plan uses a durable replay slice:

- `docs/benchmark/nr3d/assets/v10_failed_case_audit40_sample_ids_20260518.json`

Run after guard/tool-flow changes:

```bash
tmux new-session -d -s nr3d_v10_hardening_audit40 \\
  'cd /Users/bytedance/project/3DVLMReasoning && \\
   PYTHONPATH=src .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \\
     --sample-ids docs/benchmark/nr3d/assets/v10_failed_case_audit40_sample_ids_20260518.json \\
     --data-root data/nr3d/scannet \\
     --pack-name pack_nr3d_v9_catalog_first \\
     --output-dir tmp/nr3d_eval_v10_hardening_audit40_<commit> \\
     --workers 8 \\
     --sample-retries 2 \\
     --use-tool-answer-disagreement-gate \\
     --use-no-match-candidate-guard \\
     --use-evidence-frame-guard 2>&1 | tee /tmp/nr3d_v10_hardening_audit40.log'
```

Use this slice only as behavioral replay, not as a leaderboard claim.
```

- [ ] **Step 4: Run script and lightweight validation**

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_nr3d_v10_failed_case_audit_slice.py \
  --strat600 tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --output docs/benchmark/nr3d/assets/v10_failed_case_audit40_sample_ids_20260518.json
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("docs/benchmark/nr3d/assets/v10_failed_case_audit40_sample_ids_20260518.json")
ids = json.loads(p.read_text())
assert len(ids) == len(set(ids)) == 41
assert "scannet/scene0208_00::17::21496" in ids
print("audit slice ok", len(ids))
PY
```

Expected:

```text
wrote 41 sample ids to docs/benchmark/nr3d/assets/v10_failed_case_audit40_sample_ids_20260518.json
audit slice ok 41
```

- [ ] **Step 5: Full focused test run**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  src/agents/tools/tests/test_selectors_text.py \
  src/agents/tests/test_target_category_guard.py \
  src/agents/tests/test_evidence_frame_guard.py \
  src/agents/tests/test_tadg.py \
  src/agents/packs/vg_embodiedscan/tests/test_tools.py \
  src/agents/packs/vg_embodiedscan/skills/tests/test_playbook_v9_consistency.py \
  src/agents/skills/tests/test_scene_exploration_playbook_loadable.py \
  -q
.venv/bin/python -m ruff check \
  scripts/build_nr3d_v10_failed_case_audit_slice.py \
  src/agents/tools/selectors.py \
  src/agents/skills/chassis_tools.py \
  src/agents/skills/target_category_guard.py \
  src/agents/skills/evidence_frame_guard.py \
  src/agents/skills/tadg.py \
  src/agents/packs/vg_embodiedscan/tools.py
```

Expected: focused tests pass and ruff passes.

- [ ] **Step 6: Commit**

Commit:

```bash
git add \
  scripts/build_nr3d_v10_failed_case_audit_slice.py \
  docs/benchmark/nr3d/assets/v10_failed_case_audit40_sample_ids_20260518.json \
  docs/benchmark/nr3d/v10_no_initial_keyframes_strat600_20260518.md
git commit -m "docs(nr3d): add v10 failed-case replay slice"
```

---

## Final Verification Before Benchmark

- [ ] **Step 1: Confirm worktree is clean after all implementation commits**

Run:

```bash
git status --short
```

Expected: no output.

- [ ] **Step 2: Run focused test bundle once more**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  src/agents/tools/tests/test_selectors_text.py \
  src/agents/tests/test_target_category_guard.py \
  src/agents/tests/test_evidence_frame_guard.py \
  src/agents/tests/test_tadg.py \
  src/agents/packs/vg_embodiedscan/tests/test_tools.py \
  src/agents/packs/vg_embodiedscan/skills/tests/test_playbook_v9_consistency.py \
  src/agents/skills/tests/test_scene_exploration_playbook_loadable.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 3: Launch audit-slice replay in tmux**

Use a run id that includes the current short commit:

```bash
COMMIT=$(git rev-parse --short HEAD)
tmux new-session -d -s nr3d_v10_hardening_audit40 \
  "cd /Users/bytedance/project/3DVLMReasoning && \
   PYTHONPATH=src .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
     --sample-ids docs/benchmark/nr3d/assets/v10_failed_case_audit40_sample_ids_20260518.json \
     --data-root data/nr3d/scannet \
     --pack-name pack_nr3d_v9_catalog_first \
     --output-dir tmp/nr3d_eval_v10_hardening_audit40_${COMMIT} \
     --workers 8 \
     --sample-retries 2 \
     --use-tool-answer-disagreement-gate \
     --use-no-match-candidate-guard \
     --use-evidence-frame-guard 2>&1 | tee /tmp/nr3d_v10_hardening_audit40_${COMMIT}.log"
```

Expected: tmux session starts. Monitor with:

```bash
tmux capture-pane -t nr3d_v10_hardening_audit40 -p -S -30
```

- [ ] **Step 4: Only after audit replay passes, run canonical strat600**

Follow the benchmark pre-run checklist from `CLAUDE.md`: commit pending changes,
capture commit SHA, then launch the canonical strat600 run in tmux:

```bash
COMMIT=$(git rev-parse --short HEAD)
RUN_ID="v11_failed_case_hardening_strat600_$(date +%Y%m%d)"
OUT_DIR="tmp/nr3d_eval_${RUN_ID}_${COMMIT}"
tmux new-session -d -s nr3d_v11_strat600 \
  "cd /Users/bytedance/project/3DVLMReasoning && \
   PYTHONPATH=src .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
     --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
     --data-root data/nr3d/scannet \
     --pack-name pack_nr3d_v9_catalog_first \
     --output-dir ${OUT_DIR} \
     --workers 20 \
     --sample-retries 2 \
     --use-tool-answer-disagreement-gate \
     --use-no-match-candidate-guard \
     --use-evidence-frame-guard 2>&1 | tee /tmp/${RUN_ID}_${COMMIT}.log"
```

After the tmux run completes, compute leaderboard metrics and ingest SQLite:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.nr3d_leaderboard_metrics \
  --side-by-side "${OUT_DIR}/side_by_side.json" \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --output "${OUT_DIR}/leaderboard_metrics.json" \
  --canonical-filter true

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir "${OUT_DIR}" \
  --run-id "${RUN_ID}" \
  --branch "$(git branch --show-current)" \
  --commit "${COMMIT}" \
  --backend pack_v1 \
  --judge-model none \
  --leaderboard-metrics "${OUT_DIR}/leaderboard_metrics.json" \
  --notes "v10 failed-case hardening: masked selector retry, target category guard, relation evidence binding, EFG parsing, and candidate coverage." \
  --db docs/benchmark/nr3d/runs.sqlite
```

Then write `docs/benchmark/nr3d/v11_failed_case_hardening_<YYYYMMDD>.md`, update
`docs/benchmark/nr3d/README.md` and `docs/benchmark/nr3d/leaderboard.md`, and
commit the result doc plus `docs/benchmark/nr3d/runs.sqlite`.
