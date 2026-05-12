# NR3D v4 Agent Guards With Fair Views Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a 100-sample NR3D pilot that reuses the existing v2/v3 full-run baseline, removes GT-target-visible keyframe selection from the new run, enables the ScanRefer v3.20 VG agent guards, and records the result under `docs/benchmark/nr3d/`.

**Architecture:** Keep NR3D on the canonical ScanNet GT bbox proposal pool, but split inference-time evidence from scoring-time GT. Add deterministic subset selection, add a `query_driven` keyframe mode to NR3D pack-prep, port guard/low-memory runner switches from ScanRefer, and add subset-aware leaderboard metrics for fair 100-sample deltas.

**Tech Stack:** Python 3.12 via `.venv` on macOS, pytest, existing Stage2DeepResearchAgent, `KeyframeSelector.select_keyframes_v2`, SQLite ingester, tmux for long-running evaluation.

---

## File Map

- Create `scripts/build_nr3d_v4_random100_fold.py`  
  Builds the frozen 100-sample NR3D v4 pilot fold from the canonical filtered fold and existing full-run predictions.

- Create `src/evaluation/scripts/tests/test_build_nr3d_v4_random100_fold.py`  
  Unit tests for deterministic fold selection and no correctness-based filtering.

- Modify `src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py`  
  Adds `--keyframe-mode gt_target|query_driven`, `--keyframe-llm-model`, query-driven keyframe selection, non-GT fallback, and audit fields in sample JSON.

- Modify `src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d.py`  
  Tests legacy default behavior and new query-driven/no-GT audit behavior.

- Modify `src/evaluation/scripts/run_nr3d_vg_side_by_side.py`  
  Adds Stage-2 guard CLI flags, optional checkpoint-only / max-new-samples execution, low-memory side-by-side assembly, and tool trace extraction.

- Modify `src/evaluation/scripts/tests/test_run_nr3d_vg_side_by_side.py`  
  Tests guard flag propagation, checkpoint-only resume, max-new-samples, and trace preservation.

- Modify `src/evaluation/scripts/nr3d_leaderboard_metrics.py`  
  Adds optional `sample_ids_path` support so pilot metrics use the fixed 100-sample denominator.

- Modify `src/evaluation/scripts/tests/test_nr3d_leaderboard_metrics.py`  
  Tests subset-limited aggregation and failure retention.

- Modify `docs/benchmark/nr3d/README.md`, `docs/benchmark/nr3d/leaderboard.md`, `docs/benchmark/README.md`  
  Records the pilot after the run. `leaderboard.md` must clearly label the row as partial / non-leaderboard if updated.

- Create `docs/benchmark/nr3d/v4_agent_guards_fair_views_20260512.md`  
  Permanent benchmark process record for the v4 pilot.

---

### Task 1: Frozen NR3D v4 Random100 Fold

**Files:**
- Create: `scripts/build_nr3d_v4_random100_fold.py`
- Create: `src/evaluation/scripts/tests/test_build_nr3d_v4_random100_fold.py`

- [ ] **Step 1: Write failing tests for deterministic fold selection**

Create `src/evaluation/scripts/tests/test_build_nr3d_v4_random100_fold.py`:

```python
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    script = Path(__file__).resolve().parents[4] / "scripts" / "build_nr3d_v4_random100_fold.py"
    spec = importlib.util.spec_from_file_location("build_nr3d_v4_random100_fold", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _meta(sample_id: str, target_id: int, target: str = "chair") -> dict:
    scene = sample_id.split("::", 1)[0].split("/")[-1]
    return {
        "sample_id": sample_id,
        "scene_id": scene,
        "target_id": target_id,
        "target": target,
        "mentions_target_class": True,
    }


def test_select_subset_is_deterministic_and_uses_sha1_order() -> None:
    mod = _load_module()
    rows = [_meta(f"scannet/scene{i:04d}_00::{i}::A", i) for i in range(10)]
    predictions = {row["sample_id"] for row in rows}

    first = mod.select_subset(rows, predictions, n=4)
    second = mod.select_subset(list(reversed(rows)), predictions, n=4)

    assert first == second
    assert len(first) == 4
    assert first == sorted(first, key=lambda r: mod.stable_key(r["sample_id"]))[:4]


def test_select_subset_does_not_filter_failed_or_wrong_predictions() -> None:
    mod = _load_module()
    rows = [
        _meta("scannet/scene0001_00::1::A", 1),
        _meta("scannet/scene0002_00::2::A", 2),
        _meta("scannet/scene0003_00::3::A", 3),
    ]
    predictions = {
        "scannet/scene0001_00::1::A",
        "scannet/scene0002_00::2::A",
        "scannet/scene0003_00::3::A",
    }

    out = mod.select_subset(rows, predictions, n=3)

    assert {row["sample_id"] for row in out} == predictions


def test_write_outputs_sample_rows_and_summary(tmp_path: Path) -> None:
    mod = _load_module()
    rows = [_meta(f"scannet/scene{i:04d}_00::{i}::A", i) for i in range(5)]
    sample_out = tmp_path / "sample_ids.json"
    summary_out = tmp_path / "summary.json"

    mod.write_outputs(rows[:3], sample_out=sample_out, summary_out=summary_out, n_candidates=5)

    payload = json.loads(sample_out.read_text(encoding="utf-8"))
    assert set(payload[0]) == {"sample_id", "scene_id", "target_id", "category"}
    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    assert summary["n_selected"] == 3
    assert summary["n_candidates"] == 5
    assert summary["selection_salt"] == mod.SELECTION_SALT
```

- [ ] **Step 2: Run tests and verify they fail because the script does not exist**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest src/evaluation/scripts/tests/test_build_nr3d_v4_random100_fold.py -q
```

Expected: `FileNotFoundError` or import failure for `scripts/build_nr3d_v4_random100_fold.py`.

- [ ] **Step 3: Implement the fold builder**

Create `scripts/build_nr3d_v4_random100_fold.py`:

```python
"""Build the frozen NR3D v4 fair-view random100 pilot fold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.nr3d_loader import Nr3dDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SIDE_BY_SIDE = PROJECT_ROOT / "tmp/nr3d_eval_v1_full/side_by_side.json"
DEFAULT_OUT = PROJECT_ROOT / "tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json"
DEFAULT_SUMMARY = PROJECT_ROOT / "tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_summary.json"
SELECTION_SALT = "nr3d_v4_agent_guards_fair_views_20260512"
N = 100


def stable_key(sample_id: str) -> str:
    return hashlib.sha1((sample_id + SELECTION_SALT).encode("utf-8")).hexdigest()


def load_prediction_ids(side_by_side: Path, backend: str = "pack_v1") -> set[str]:
    payload = json.loads(side_by_side.read_text(encoding="utf-8"))
    per_sample = payload[backend]["per_sample"]
    return {str(row["sample_id"]) for row in per_sample}


def load_canonical_rows(nr3d_root: Path, phase8_data_root: Path) -> list[dict[str, Any]]:
    dataset = Nr3dDataset.from_path(
        data_root=nr3d_root,
        split="test",
        bbox_source="phase8_gt_cg",
        phase8_data_root=phase8_data_root,
        mentions_target_class_only=True,
    )
    return [
        {
            "sample_id": sample.sample_id,
            "scene_id": sample.scene_id,
            "target_id": int(sample.target_id),
            "target": sample.target,
            "mentions_target_class": bool(sample.mentions_target_class),
        }
        for sample in dataset
    ]


def select_subset(rows: list[dict[str, Any]], prediction_ids: set[str], n: int = N) -> list[dict[str, Any]]:
    candidates = [row for row in rows if str(row["sample_id"]) in prediction_ids]
    if len(candidates) < n:
        raise ValueError(f"only {len(candidates)} candidates available, need {n}")
    ordered = sorted(candidates, key=lambda row: stable_key(str(row["sample_id"])))
    return ordered[:n]


def write_outputs(
    selected: list[dict[str, Any]],
    *,
    sample_out: Path,
    summary_out: Path,
    n_candidates: int,
) -> None:
    sample_rows = [
        {
            "sample_id": str(row["sample_id"]),
            "scene_id": str(row["scene_id"]),
            "target_id": int(row["target_id"]),
            "category": str(row["target"]),
        }
        for row in selected
    ]
    sample_out.parent.mkdir(parents=True, exist_ok=True)
    sample_out.write_text(json.dumps(sample_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "selection_salt": SELECTION_SALT,
        "n_selected": len(sample_rows),
        "n_candidates": int(n_candidates),
        "sample_ids_path": str(sample_out),
        "first_10_sample_ids": [row["sample_id"] for row in sample_rows[:10]],
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side-by-side", type=Path, default=DEFAULT_SIDE_BY_SIDE)
    parser.add_argument("--nr3d-root", type=Path, default=Path("data/nr3d"))
    parser.add_argument("--phase8-data-root", type=Path, default=Path("data/nr3d/scannet"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--backend", default="pack_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0:
        raise ValueError("--n must be positive")
    prediction_ids = load_prediction_ids(args.side_by_side, backend=args.backend)
    rows = load_canonical_rows(args.nr3d_root, args.phase8_data_root)
    selected = select_subset(rows, prediction_ids, n=args.n)
    write_outputs(
        selected,
        sample_out=args.output,
        summary_out=args.summary,
        n_candidates=len([row for row in rows if row["sample_id"] in prediction_ids]),
    )
    print(f"wrote {args.output} n={len(selected)} salt={SELECTION_SALT}")
    print(f"summary={args.summary}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run fold-builder tests**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest src/evaluation/scripts/tests/test_build_nr3d_v4_random100_fold.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add scripts/build_nr3d_v4_random100_fold.py src/evaluation/scripts/tests/test_build_nr3d_v4_random100_fold.py
git commit -m "feat(nr3d): add v4 random100 fold builder"
```

---

### Task 2: NR3D Fair Query-Driven Keyframe Mode

**Files:**
- Modify: `src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py`
- Modify: `src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d.py`

- [ ] **Step 1: Add failing tests for legacy default and query-driven audit fields**

Append to `src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d.py`:

```python
def test_prepare_default_records_gt_target_keyframe_mode(tmp_path, monkeypatch) -> None:
    from evaluation.scripts import prepare_pack_v1_inputs_nr3d as prep

    data_root = tmp_path / "scannet"
    _write_phase8_tree(data_root)
    sample_ids = tmp_path / "sample_ids.json"
    _write_sample_ids(sample_ids)
    sample = SimpleNamespace(
        sample_id="scannet/scene0001_00::0::A1",
        scene_id="scene0001_00",
        scan_id="scannet/scene0001_00",
        target_id=0,
        target="chair",
        query="the chair by the table",
        gt_bbox_3d=[0.0, 0.0, 5.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    )
    monkeypatch.setattr(
        prep,
        "load_sample_lookup",
        lambda *, nr3d_root, phase8_data_root, split, sample_ids=None: (
            SimpleNamespace(),
            {sample.sample_id: sample},
        ),
    )

    [path] = prep.prepare_pack_v1_inputs_nr3d(
        sample_ids_path=sample_ids,
        data_root=data_root,
        pack_name="pack_nr3d_v1",
        split="test",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["keyframe_mode"] == "gt_target"
    assert payload["keyframe_selection_uses_gt_target"] is True
```

Append a unit-level test that does not invoke the real LLM:

```python
def test_query_driven_keyframes_do_not_read_target_visibility(tmp_path, monkeypatch) -> None:
    from evaluation.scripts import prepare_pack_v1_inputs_nr3d as prep

    data_root = tmp_path / "scannet"
    _write_phase8_tree(
        data_root,
        visibility={
            "object_to_views": {0: [(0, 0.9)]},
            "view_to_objects": {0: [(0, 0.9)], 1: [(1, 0.7)]},
            "metadata": {},
        },
    )

    class FakeSelector:
        def select_keyframes_v2(self, *, query, k, use_visual_context):
            assert query == "the chair by the table"
            assert k == 3
            assert use_visual_context is False
            return SimpleNamespace(keyframe_indices=[1], target_term="chair")

    keyframes, used_fallback = prep.select_keyframes_query_driven(
        selector=FakeSelector(),
        scene_id="scene0001_00",
        query="the chair by the table",
        raw_frames_root=data_root,
        k=3,
    )

    assert used_fallback is False
    assert [item["frame_id"] for item in keyframes] == [1]
    assert keyframes[0]["image_path"].endswith("000010-rgb.png")
```

Append a fallback test:

```python
def test_query_driven_fallback_uses_scene_density_not_target_id(tmp_path) -> None:
    from evaluation.scripts import prepare_pack_v1_inputs_nr3d as prep

    data_root = tmp_path / "scannet"
    _write_phase8_tree(
        data_root,
        visibility={
            "object_to_views": {0: [(0, 0.9)]},
            "view_to_objects": {0: [(0, 0.9)], 1: [(0, 0.2), (1, 0.8)]},
            "metadata": {},
        },
    )

    class EmptySelector:
        def select_keyframes_v2(self, *, query, k, use_visual_context):
            return SimpleNamespace(keyframe_indices=[], target_term="chair")

    visibility = prep.load_phase8_visibility_index(data_root / "scene0001_00")
    keyframes, used_fallback = prep.select_keyframes_query_driven(
        selector=EmptySelector(),
        scene_id="scene0001_00",
        query="the chair by the table",
        raw_frames_root=data_root,
        k=1,
        fallback_visibility=visibility,
    )

    assert used_fallback is True
    assert [item["frame_id"] for item in keyframes] == [1]
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d.py -q
```

Expected: failures for missing `keyframe_mode`, `keyframe_selection_uses_gt_target`, and `select_keyframes_query_driven`.

- [ ] **Step 3: Add keyframe CLI arguments and function parameters**

Modify `parse_args()`:

```python
    parser.add_argument(
        "--keyframe-mode",
        default="gt_target",
        choices=["gt_target", "query_driven"],
        help=(
            "'gt_target': legacy top visible frames for the target id. "
            "'query_driven': KeyframeSelector.select_keyframes_v2(query, k=3) "
            "with no GT target visibility."
        ),
    )
    parser.add_argument(
        "--keyframe-llm-model",
        default="gemini-2.5-pro",
        help="LLM for query-driven keyframe parsing.",
    )
```

Modify `prepare_pack_v1_inputs_nr3d(...)` signature:

```python
def prepare_pack_v1_inputs_nr3d(
    *,
    sample_ids_path: Path,
    data_root: Path,
    pack_name: str = "pack_nr3d_v1",
    split: str = "test",
    max_samples: int | None = None,
    nr3d_root: Path | None = None,
    keyframe_mode: str = "gt_target",
    keyframe_llm_model: str = "gemini-2.5-pro",
) -> list[Path]:
```

Inside the function, validate:

```python
    if keyframe_mode not in {"gt_target", "query_driven"}:
        raise ValueError(f"Unsupported keyframe_mode={keyframe_mode!r}")
    selector_cache: dict[str, Any] = {}
```

Pass `keyframe_mode`, `keyframe_llm_model`, and `selector_cache` into `write_sample_artifact`.

- [ ] **Step 4: Implement query-driven keyframe helper and fallback**

Add near `select_keyframes_for_sample`:

```python
def _keyframes_from_frame_ids(
    *,
    scene_root: Path,
    frame_ids: Sequence[int],
    k: int,
) -> list[dict[str, Any]]:
    keyframes: list[dict[str, Any]] = []
    for keyframe_idx, frame_id in enumerate(list(frame_ids)[:k]):
        keyframes.append(
            {
                "keyframe_idx": keyframe_idx,
                "image_path": str(resolve_raw_rgb_path(scene_root, int(frame_id))),
                "frame_id": int(frame_id),
            }
        )
    return keyframes


def select_keyframes_by_scene_density(
    *,
    scene_root: Path,
    visibility: Phase8Visibility,
    k: int = 3,
) -> list[dict[str, Any]]:
    ranked = sorted(
        visibility.view_to_objects.items(),
        key=lambda item: (-len(item[1]), int(item[0])),
    )
    return _keyframes_from_frame_ids(
        scene_root=scene_root,
        frame_ids=[frame_id for frame_id, _entries in ranked],
        k=k,
    )


def select_keyframes_query_driven(
    *,
    selector: Any,
    scene_id: str,
    query: str,
    raw_frames_root: Path,
    k: int = 3,
    fallback_visibility: Phase8Visibility | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    res = selector.select_keyframes_v2(
        query=query,
        k=k,
        use_visual_context=False,
    )
    if getattr(res, "keyframe_indices", None):
        return (
            _keyframes_from_frame_ids(
                scene_root=raw_frames_root / scene_id,
                frame_ids=[int(frame_id) for frame_id in res.keyframe_indices],
                k=k,
            ),
            False,
        )
    if fallback_visibility is None:
        return [], True
    return (
        select_keyframes_by_scene_density(
            scene_root=raw_frames_root / scene_id,
            visibility=fallback_visibility,
            k=k,
        ),
        True,
    )
```

- [ ] **Step 5: Wire keyframe selection into sample writing**

Modify `write_sample_artifact(...)` signature:

```python
def write_sample_artifact(
    *,
    request: SampleRequest,
    sample: Nr3dVGSample,
    data_root: Path,
    scene_artifacts: SceneArtifacts,
    keyframe_mode: str = "gt_target",
    keyframe_llm_model: str = "gemini-2.5-pro",
    selector_cache: dict[str, Any] | None = None,
) -> Path:
```

Replace the keyframe block with:

```python
    visibility = load_phase8_visibility_index(data_root / request.scene_id)
    query = getattr(sample, "query", "") or getattr(sample, "text", "")
    if not query:
        raise ValueError(f"Missing query for {request.sample_id}")

    used_fallback = False
    if keyframe_mode == "gt_target":
        keyframes = select_keyframes_for_sample(
            scene_root=data_root / request.scene_id,
            target_id=request.target_id,
            visibility=visibility,
            k=5,
        )
        uses_gt_target = True
    elif keyframe_mode == "query_driven":
        cache = selector_cache if selector_cache is not None else {}
        selector = cache.get(request.scene_id)
        if selector is None:
            from query_scene.keyframe_selector import KeyframeSelector

            selector = KeyframeSelector.from_scene_path(
                str(data_root / request.scene_id / "conceptgraph"),
                stride=1,
                llm_model=keyframe_llm_model,
            )
            cache[request.scene_id] = selector
        keyframes, used_fallback = select_keyframes_query_driven(
            selector=selector,
            scene_id=request.scene_id,
            query=query,
            raw_frames_root=data_root,
            k=3,
            fallback_visibility=visibility,
        )
        uses_gt_target = False
    else:
        raise ValueError(f"Unsupported keyframe_mode={keyframe_mode!r}")
```

Add to `payload`:

```python
        "keyframe_mode": keyframe_mode,
        "keyframe_selection_uses_gt_target": uses_gt_target,
        "keyframe_selection_used_fallback": used_fallback,
```

- [ ] **Step 6: Wire CLI main**

Modify `main()` call:

```python
    written = prepare_pack_v1_inputs_nr3d(
        sample_ids_path=args.sample_ids,
        data_root=args.data_root,
        pack_name=args.pack_name,
        split=args.split,
        max_samples=args.max_samples,
        nr3d_root=args.nr3d_root,
        keyframe_mode=args.keyframe_mode,
        keyframe_llm_model=args.keyframe_llm_model,
    )
```

Update `__all__` to export:

```python
    "select_keyframes_by_scene_density",
    "select_keyframes_query_driven",
```

- [ ] **Step 7: Run focused pack-prep tests**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 2**

```bash
git add src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d.py
git commit -m "feat(nr3d): add fair query-driven keyframes"
```

---

### Task 3: NR3D Runner Guards, Traces, And Low-Memory Resume

**Files:**
- Modify: `src/evaluation/scripts/run_nr3d_vg_side_by_side.py`
- Modify: `src/evaluation/scripts/tests/test_run_nr3d_vg_side_by_side.py`

- [ ] **Step 1: Add failing tests for guard flags**

Append to `src/evaluation/scripts/tests/test_run_nr3d_vg_side_by_side.py`:

```python
def test_main_wires_all_guard_flags(tmp_path, monkeypatch) -> None:
    import sys
    from evaluation.scripts import run_nr3d_vg_side_by_side as runner

    sample_ids = tmp_path / "samples.json"
    sample_ids.write_text("[]", encoding="utf-8")
    captured = {}

    def fake_compare_backends(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(runner, "compare_backends", fake_compare_backends)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_nr3d_vg_side_by_side",
            "--sample-ids", str(sample_ids),
            "--output-dir", str(tmp_path / "out"),
            "--data-root", str(tmp_path),
            "--use-tool-answer-disagreement-gate",
            "--use-no-match-candidate-guard",
            "--use-evidence-frame-guard",
        ],
    )

    runner.main()

    cfg = captured["config"]
    assert cfg.use_tool_answer_disagreement_gate is True
    assert cfg.use_no_match_candidate_guard is True
    assert cfg.use_evidence_frame_guard is True
```

- [ ] **Step 2: Add failing tests for checkpoint-only and max-new-samples**

Append:

```python
def test_compare_backends_checkpoint_only_respects_max_new_samples(monkeypatch, tmp_path) -> None:
    from evaluation.scripts import run_nr3d_vg_side_by_side as runner

    sample_ids = [
        "scannet/scene0001_00::72::A1",
        "scannet/scene0001_00::73::A2",
    ]
    data_root = _write_nr3d_pack_inputs(tmp_path / "data_root", sample_id=sample_ids[0])
    _write_nr3d_pack_inputs(tmp_path / "data_root", sample_id=sample_ids[1])
    calls = []

    def fake_run_one(sample_id, backend, **kwargs):
        calls.append(sample_id)
        return {
            "sample_id": sample_id,
            "backend": backend,
            "status": "completed",
            "iou": 1.0,
            "selected_object_id": 72,
        }

    monkeypatch.setattr(runner, "run_one_sample", fake_run_one)

    out = runner.compare_backends(
        sample_ids=sample_ids,
        output_dir=tmp_path / "out",
        data_root=data_root,
        workers=1,
        write_side_by_side=False,
        return_results=False,
        max_new_samples=1,
    )

    assert out is None
    assert calls == [sample_ids[0]]
    assert not (tmp_path / "out" / "side_by_side.json").exists()
```

- [ ] **Step 3: Add failing test for tool trace extraction**

Append:

```python
def test_run_one_sample_preserves_tool_trace(monkeypatch, tmp_path) -> None:
    from evaluation.scripts import run_nr3d_vg_side_by_side as runner

    data_root = _write_nr3d_pack_inputs(tmp_path)

    class FakeAgent:
        def __init__(self, config):
            self.config = config

        def run(self, task, bundle):
            return SimpleNamespace(
                result=SimpleNamespace(
                    payload={
                        "status": "completed",
                        "selected_object_id": 72,
                        "bbox_3d": [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    },
                    confidence=0.9,
                ),
                tool_trace=[
                    SimpleNamespace(tool_name="submit_final", tool_input={"payload": {"proposal_id": 72}}, response_text="submitted")
                ],
            )

    monkeypatch.setattr(runner, "Stage2DeepResearchAgent", FakeAgent)
    monkeypatch.setattr(
        runner,
        "build_pack_v1_bundle",
        lambda **kwargs: SimpleNamespace(scene_id=kwargs["scene_id"]),
    )

    out = runner.run_one_sample("scannet/scene0001_00::72::A1", "pack_v1", data_root=data_root)

    assert out["tool_trace"] == [
        {
            "tool_name": "submit_final",
            "tool_input": {"payload": {"proposal_id": 72}},
            "response_text": "submitted",
        }
    ]
```

- [ ] **Step 4: Run runner tests and verify failures**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest src/evaluation/scripts/tests/test_run_nr3d_vg_side_by_side.py -q
```

Expected: failures for missing CLI args, missing `write_side_by_side`, missing `max_new_samples`, or missing `tool_trace`.

- [ ] **Step 5: Implement tool trace extraction**

Add to `run_nr3d_vg_side_by_side.py`:

```python
def extract_result_tool_trace(result: Any) -> list[dict[str, Any]]:
    trace = getattr(result, "tool_trace", None)
    if trace is None:
        trace = getattr(getattr(result, "result", None), "tool_trace", None)
    out: list[dict[str, Any]] = []
    for entry in trace or []:
        if isinstance(entry, dict):
            out.append(
                {
                    "tool_name": entry.get("tool_name"),
                    "tool_input": entry.get("tool_input"),
                    "response_text": entry.get("response_text"),
                }
            )
        else:
            out.append(
                {
                    "tool_name": getattr(entry, "tool_name", None),
                    "tool_input": getattr(entry, "tool_input", None),
                    "response_text": getattr(entry, "response_text", None),
                }
            )
    return out
```

In `_run_one_sample_once`, after `raw_result`:

```python
    tool_trace = extract_result_tool_trace(raw_result)
```

Add `"tool_trace": tool_trace` to both completed and failed-marker result payloads.

- [ ] **Step 6: Implement checkpoint-only / max-new-samples support**

Change `compare_backends` signature:

```python
def compare_backends(
    *,
    sample_ids: Sequence[str],
    output_dir: Path,
    data_root: Path,
    pack_name: str = "pack_nr3d_v1",
    config: Stage2DeepAgentConfig | None = None,
    sample_retries: int = 0,
    workers: int = 1,
    return_results: bool = True,
    write_side_by_side: bool = True,
    max_new_samples: int | None = None,
) -> dict[str, Any] | None:
```

Add validation:

```python
    if max_new_samples is not None and max_new_samples < 0:
        raise ValueError("max_new_samples must be non-negative")
```

Process only missing checkpoints:

```python
        missing_sample_ids = [
            sample_id
            for sample_id in sample_ids
            if not sample_result_path(
                output_dir,
                backend,
                sample_id,
                pack_name=pack_name,
            ).exists()
        ]
        if max_new_samples is not None:
            missing_sample_ids = missing_sample_ids[:max_new_samples]
```

After processing, if `write_side_by_side` is false:

```python
        if not write_side_by_side:
            continue
```

- [ ] **Step 7: Add checkpoint assembly helpers**

Add helpers copied in shape from ScanRefer, adapted to NR3D:

```python
def build_backend_payload_from_checkpoints(
    *,
    sample_ids: Sequence[str],
    output_dir: Path,
    backend: BackendName,
    pack_name: str,
    include_per_sample: bool = True,
) -> dict[str, Any]:
    per_sample = []
    for sample_id in sample_ids:
        cached = load_sample_result_checkpoint(
            output_dir,
            backend,
            sample_id,
            pack_name=pack_name,
        )
        if cached is None:
            raise FileNotFoundError(f"missing checkpoint for {sample_id}")
        per_sample.append(cached)
    ious = [float(r["iou"]) for r in per_sample if r.get("iou") is not None]
    payload: dict[str, Any] = {
        "n": len(per_sample),
        "mean_iou": statistics.mean(ious) if ious else 0.0,
        "Acc@0.25": sum(1 for v in ious if v >= 0.25) / max(len(ious), 1),
        "Acc@0.50": sum(1 for v in ious if v >= 0.50) / max(len(ious), 1),
    }
    if include_per_sample:
        payload["per_sample"] = per_sample
    return payload


def stream_side_by_side_from_checkpoints(
    *,
    sample_ids: Sequence[str],
    output_dir: Path,
    backend: BackendName,
    pack_name: str,
) -> None:
    payload = {
        backend: build_backend_payload_from_checkpoints(
            sample_ids=sample_ids,
            output_dir=output_dir,
            backend=backend,
            pack_name=pack_name,
            include_per_sample=True,
        )
    }
    (output_dir / "side_by_side.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
```

Use these helpers at the end of each backend loop:

```python
        if return_results:
            results[backend] = build_backend_payload_from_checkpoints(
                sample_ids=sample_ids,
                output_dir=output_dir,
                backend=backend,
                pack_name=pack_name,
                include_per_sample=True,
            )
        else:
            stream_side_by_side_from_checkpoints(
                sample_ids=sample_ids,
                output_dir=output_dir,
                backend=backend,
                pack_name=pack_name,
            )
```

- [ ] **Step 8: Add CLI args and config wiring**

Modify `parse_args()`:

```python
    parser.add_argument("--checkpoint-only", action="store_true", default=False)
    parser.add_argument("--max-new-samples", type=int, default=None)
    parser.add_argument("--use-tool-answer-disagreement-gate", action="store_true", default=False)
    parser.add_argument("--use-no-match-candidate-guard", action="store_true", default=False)
    parser.add_argument("--use-evidence-frame-guard", action="store_true", default=False)
```

Modify `main()`:

```python
    config = Stage2DeepAgentConfig(
        use_tool_answer_disagreement_gate=args.use_tool_answer_disagreement_gate,
        use_no_match_candidate_guard=args.use_no_match_candidate_guard,
        use_evidence_frame_guard=args.use_evidence_frame_guard,
    )
    compare_backends(
        sample_ids=sample_ids,
        output_dir=args.output_dir,
        data_root=args.data_root,
        pack_name=args.pack_name,
        config=config,
        sample_retries=args.sample_retries,
        workers=args.workers,
        return_results=False,
        write_side_by_side=not args.checkpoint_only,
        max_new_samples=args.max_new_samples,
    )
```

- [ ] **Step 9: Run focused runner tests**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest src/evaluation/scripts/tests/test_run_nr3d_vg_side_by_side.py -q
```

Expected: all tests pass.

- [ ] **Step 10: Commit Task 3**

```bash
git add src/evaluation/scripts/run_nr3d_vg_side_by_side.py src/evaluation/scripts/tests/test_run_nr3d_vg_side_by_side.py
git commit -m "feat(nr3d): port guard and checkpoint runner flags"
```

---

### Task 4: Subset-Aware NR3D Leaderboard Metrics

**Files:**
- Modify: `src/evaluation/scripts/nr3d_leaderboard_metrics.py`
- Modify: `src/evaluation/scripts/tests/test_nr3d_leaderboard_metrics.py`

- [ ] **Step 1: Add failing unit test for explicit subset denominator**

Append to `src/evaluation/scripts/tests/test_nr3d_leaderboard_metrics.py`:

```python
def test_aggregate_with_subset_scores_only_requested_sample_ids():
    from evaluation.scripts.nr3d_leaderboard_metrics import restrict_to_sample_ids

    predictions = {"a": 0, "b": 99, "c": 2}
    meta = [
        _meta("a", 0, 2, ["table"]),
        _meta("b", 1, 2, ["chair"]),
        _meta("c", 2, 2, ["lamp"]),
    ]
    subset_predictions, subset_meta, subset_filtered = restrict_to_sample_ids(
        predictions,
        meta,
        filtered_sample_ids={"a", "b", "c"},
        requested_sample_ids={"a", "c"},
    )
    m = aggregate(subset_predictions, subset_meta, subset_filtered)

    assert m["n_full"] == 2
    assert m["n_filtered"] == 2
    assert m["classification_acc_filtered"] == 1.0
    assert {row["sample_id"] for row in m["per_sample"]} == {"a", "c"}
```

Append:

```python
def test_restrict_to_sample_ids_requires_prediction_for_requested_id():
    from evaluation.scripts.nr3d_leaderboard_metrics import restrict_to_sample_ids

    with pytest.raises(ValueError, match="missing requested sample_ids"):
        restrict_to_sample_ids(
            predictions={"a": 0},
            sample_meta=[_meta("a", 0, 2, ["x"])],
            filtered_sample_ids={"a"},
            requested_sample_ids={"a", "b"},
        )
```

- [ ] **Step 2: Run tests and verify failures**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest src/evaluation/scripts/tests/test_nr3d_leaderboard_metrics.py -q
```

Expected: import failure for `restrict_to_sample_ids`.

- [ ] **Step 3: Add subset helper and loader**

Add to `nr3d_leaderboard_metrics.py`:

```python
def load_requested_sample_ids(path: Path) -> set[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"sample ids JSON must be a list: {path}")
    out: set[str] = set()
    for index, item in enumerate(raw):
        if isinstance(item, str):
            sample_id = item
        elif isinstance(item, dict):
            sample_id = item.get("sample_id")
        else:
            raise ValueError(f"sample_ids[{index}] must be string or object")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"sample_ids[{index}] missing non-empty sample_id")
        out.add(sample_id)
    if not out:
        raise ValueError(f"sample ids JSON is empty: {path}")
    return out


def restrict_to_sample_ids(
    predictions: dict[str, int | None],
    sample_meta: list[dict[str, Any]],
    filtered_sample_ids: set[str],
    requested_sample_ids: set[str],
) -> tuple[dict[str, int | None], list[dict[str, Any]], set[str]]:
    missing_predictions = requested_sample_ids - set(predictions)
    if missing_predictions:
        raise ValueError(
            f"missing requested sample_ids in predictions: {sorted(missing_predictions)[:5]}"
        )
    meta_ids = {str(row["sample_id"]) for row in sample_meta}
    missing_meta = requested_sample_ids - meta_ids
    if missing_meta:
        raise ValueError(
            f"missing requested sample_ids in NR3D metadata: {sorted(missing_meta)[:5]}"
        )
    subset_predictions = {
        sample_id: predictions[sample_id]
        for sample_id in sorted(requested_sample_ids)
    }
    subset_meta = [
        row for row in sample_meta if str(row["sample_id"]) in requested_sample_ids
    ]
    subset_filtered = filtered_sample_ids & requested_sample_ids
    return subset_predictions, subset_meta, subset_filtered
```

- [ ] **Step 4: Wire `sample_ids_path` into IO wrapper and CLI**

Modify signature:

```python
def compute_leaderboard_metrics(
    side_by_side_path: Path,
    nr3d_data_root: Path,
    phase8_data_root: Path,
    canonical_filter: bool = True,
    backend: str = "pack_v1",
    sample_ids_path: Path | None = None,
) -> dict[str, Any]:
```

Before `return aggregate(...)`:

```python
    if sample_ids_path is not None:
        predictions, sample_meta, filtered_sample_ids = restrict_to_sample_ids(
            predictions,
            sample_meta,
            filtered_sample_ids,
            load_requested_sample_ids(sample_ids_path),
        )
```

Add CLI arg:

```python
    parser.add_argument("--sample-ids", type=Path, default=None)
```

Pass it into `compute_leaderboard_metrics`.

- [ ] **Step 5: Run metrics tests**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest src/evaluation/scripts/tests/test_nr3d_leaderboard_metrics.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/evaluation/scripts/nr3d_leaderboard_metrics.py src/evaluation/scripts/tests/test_nr3d_leaderboard_metrics.py
git commit -m "feat(nr3d): support subset leaderboard metrics"
```

---

### Task 5: Run The 100-Sample NR3D v4 Pilot

**Files:**
- Runtime artifacts only under `tmp/nr3d_artifacts/`, `tmp/nr3d_eval_v4_agent_guards_fair_views_random100/`, and `tmp/nr3d_eval_v4_baseline_random100/`

- [ ] **Step 1: Generate frozen sample ids**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/build_nr3d_v4_random100_fold.py
```

Expected:

```text
wrote tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json n=100
summary=tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_summary.json
```

- [ ] **Step 2: Prepare NR3D v4 pack inputs with query-driven keyframes**

Run in tmux:

```bash
tmux new-session -d -s nr3d_v4_prep 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v4_agent_guards_fair_views \
  --split test \
  --keyframe-mode query_driven \
  --keyframe-llm-model gemini-2.5-pro \
  2>&1 | tee tmp/nr3d_eval_v4_prep_random100.log'
```

Check every 30-60 minutes:

```bash
tmux capture-pane -t nr3d_v4_prep -p -S -50
```

Expected final line:

```text
wrote 100 sample artifacts under data/nr3d/scannet/<scene>/pack_nr3d_v4_agent_guards_fair_views/
```

- [ ] **Step 3: Audit prepared samples for no GT-target keyframes**

Run:

```bash
source .venv/bin/activate
python - <<'PY'
import json
from pathlib import Path
ids = json.loads(Path('tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json').read_text())
bad = []
for row in ids:
    sid = row['sample_id']
    scene = row['scene_id']
    safe = sid.replace('/', '__').replace('::', '__')
    p = Path('data/nr3d/scannet') / scene / 'pack_nr3d_v4_agent_guards_fair_views' / 'samples' / f'{safe}.json'
    payload = json.loads(p.read_text())
    if payload.get('keyframe_mode') != 'query_driven' or payload.get('keyframe_selection_uses_gt_target') is not False:
        bad.append((sid, payload.get('keyframe_mode'), payload.get('keyframe_selection_uses_gt_target')))
print('bad_count=', len(bad))
if bad:
    raise SystemExit(bad[:5])
PY
```

Expected: `bad_count= 0`.

- [ ] **Step 4: Run the new agent on 100 samples**

Run in tmux:

```bash
tmux new-session -d -s nr3d_v4_random100 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && PYTHONPATH=src python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v4_agent_guards_fair_views \
  --output-dir tmp/nr3d_eval_v4_agent_guards_fair_views_random100 \
  --workers 15 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard \
  2>&1 | tee tmp/nr3d_eval_v4_agent_guards_fair_views_random100.log'
```

Monitor every 30-60 minutes:

```bash
tmux capture-pane -t nr3d_v4_random100 -p -S -80
ps -o pid,rss,command -ax | rg 'run_nr3d_vg_side_by_side|python src/evaluation/scripts/run_nr3d' || true
```

RSS target: keep the main Python process below 15 GB.

- [ ] **Step 5: If interrupted, resume without reprocessing completed samples**

If `side_by_side.json` was not assembled but checkpoints exist, run:

```bash
tmux new-session -d -s nr3d_v4_resume 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && PYTHONPATH=src python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v4_agent_guards_fair_views \
  --output-dir tmp/nr3d_eval_v4_agent_guards_fair_views_random100 \
  --workers 15 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard \
  2>&1 | tee -a tmp/nr3d_eval_v4_agent_guards_fair_views_random100.log'
```

- [ ] **Step 6: Verify 100 checkpoints and side-by-side output**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
sample_ids = json.loads(Path('tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json').read_text())
ckpts = list(Path('tmp/nr3d_eval_v4_agent_guards_fair_views_random100/per_sample/pack_nr3d_v4_agent_guards_fair_views').glob('*.json'))
side = json.loads(Path('tmp/nr3d_eval_v4_agent_guards_fair_views_random100/side_by_side.json').read_text())
print('sample_ids=', len(sample_ids))
print('checkpoints=', len(ckpts))
print('side_n=', side['pack_v1']['n'])
print('failed=', sum(1 for r in side['pack_v1']['per_sample'] if r.get('status') == 'failed'))
if len(sample_ids) != 100 or len(ckpts) != 100 or side['pack_v1']['n'] != 100:
    raise SystemExit('incomplete run')
PY
```

Expected: `sample_ids=100`, `checkpoints=100`, `side_n=100`.

---

### Task 6: Metrics, SQLite Ingest, Docs, And Final Verification

**Files:**
- Create: `docs/benchmark/nr3d/v4_agent_guards_fair_views_20260512.md`
- Modify: `docs/benchmark/nr3d/README.md`
- Modify: `docs/benchmark/nr3d/leaderboard.md`
- Modify: `docs/benchmark/README.md`
- Modify: `docs/benchmark/nr3d/runs.sqlite`

- [ ] **Step 1: Compute baseline subset metrics**

Run:

```bash
mkdir -p tmp/nr3d_eval_v4_baseline_random100
source .venv/bin/activate
PYTHONPATH=src python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v1_full/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --output tmp/nr3d_eval_v4_baseline_random100/leaderboard_metrics.json \
  --canonical-filter true
```

Expected: `n_full=100 n_filtered=100`.

- [ ] **Step 2: Compute new-run subset metrics**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v4_agent_guards_fair_views_random100/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --output tmp/nr3d_eval_v4_agent_guards_fair_views_random100/leaderboard_metrics.json \
  --canonical-filter true
```

Expected: `n_full=100 n_filtered=100`.

- [ ] **Step 3: Print delta table**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
b = json.loads(Path('tmp/nr3d_eval_v4_baseline_random100/leaderboard_metrics.json').read_text())
n = json.loads(Path('tmp/nr3d_eval_v4_agent_guards_fair_views_random100/leaderboard_metrics.json').read_text())
keys = [
    ('Overall', 'classification_acc_filtered'),
    ('Easy', 'acc_easy'),
    ('Hard', 'acc_hard'),
    ('View-Dep', 'acc_view_dep'),
    ('View-Indep', 'acc_view_indep'),
]
for label, key in keys:
    bv = float(b[key])
    nv = float(n[key])
    print(f'{label:10s} baseline={bv:.4f} new={nv:.4f} delta_pp={(nv-bv)*100:+.2f}')
side = json.loads(Path('tmp/nr3d_eval_v4_agent_guards_fair_views_random100/side_by_side.json').read_text())
print('new_failed=', sum(1 for r in side['pack_v1']['per_sample'] if r.get('status') == 'failed'))
PY
```

Use this output in the version doc.

- [ ] **Step 4: Ingest new run into SQLite**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v4_agent_guards_fair_views_random100 \
  --run-id v4_agent_guards_fair_views_random100_20260512 \
  --branch "$(git branch --show-current)" \
  --commit "$(git rev-parse --short HEAD)" \
  --backend pack_v1 \
  --judge-model none \
  --notes "v4 random100; query-driven fair keyframes; TADG + no-match + evidence-frame guards" \
  --leaderboard-metrics tmp/nr3d_eval_v4_agent_guards_fair_views_random100/leaderboard_metrics.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

Expected:

```text
[ingest] run_id=v4_agent_guards_fair_views_random100_20260512 samples=100 db=docs/benchmark/nr3d/runs.sqlite
```

- [ ] **Step 5: Verify SQLite row**

Run:

```bash
sqlite3 docs/benchmark/nr3d/runs.sqlite "
SELECT run_id, n,
       printf('%.4f', classification_acc_filtered) AS acc,
       printf('%.4f', acc_easy) AS easy,
       printf('%.4f', acc_hard) AS hard,
       printf('%.4f', acc_view_dep) AS vdep,
       printf('%.4f', acc_view_indep) AS vind
FROM runs
WHERE run_id='v4_agent_guards_fair_views_random100_20260512';
"
```

Expected: one row with `n=100`.

- [ ] **Step 6: Generate benchmark version doc from observed metrics**

Run this script after Steps 1-5 have produced metrics and SQLite output:

```bash
python - <<'PY'
import json
import subprocess
from pathlib import Path

branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
baseline = json.loads(Path("tmp/nr3d_eval_v4_baseline_random100/leaderboard_metrics.json").read_text())
new = json.loads(Path("tmp/nr3d_eval_v4_agent_guards_fair_views_random100/leaderboard_metrics.json").read_text())
side = json.loads(Path("tmp/nr3d_eval_v4_agent_guards_fair_views_random100/side_by_side.json").read_text())
failed = sum(1 for row in side["pack_v1"]["per_sample"] if row.get("status") == "failed")

rows = [
    ("Overall", "classification_acc_filtered"),
    ("Easy", "acc_easy"),
    ("Hard", "acc_hard"),
    ("View-Dep", "acc_view_dep"),
    ("View-Indep", "acc_view_indep"),
]
table_lines = [
    "| Metric | Baseline subset | v4 new | Delta |",
    "|---|---:|---:|---:|",
]
for label, key in rows:
    bv = float(baseline[key])
    nv = float(new[key])
    table_lines.append(f"| {label} | {bv * 100:.2f} | {nv * 100:.2f} | {(nv - bv) * 100:+.2f} pp |")
table_lines.append(f"| Failed samples | 0.00 | {failed:.2f} | {failed:+.2f} |")

overall_delta = float(new["classification_acc_filtered"]) - float(baseline["classification_acc_filtered"])
if overall_delta >= 0.01:
    interpretation = "positive signal; expand to 500 samples after a quick case audit"
elif overall_delta <= -0.01:
    interpretation = "negative signal; audit whether query-driven keyframes lost target evidence before scaling"
else:
    interpretation = "neutral signal; inspect case-level errors before deciding whether to scale"

doc = f"""# v4 Agent Guards With Fair Views - 2026-05-12

## Run Identity

- Branch: `{branch}`
- Tip commit: `{commit}`
- Run ID: `v4_agent_guards_fair_views_random100_20260512`
- Eval scale: 100 canonical-filtered NR3D test samples
- Backend: Stage2 pack_v1 via `Stage2DeepAgentConfig`
- Proposal pool: ScanNet GT instance bboxes
- Keyframe mode: `query_driven`
- Guards: TADG, no-match candidate guard, evidence-frame guard

## Why This Version Exists

This pilot tests whether the ScanRefer v3.20 Stage-2 guard stack helps NR3D
after removing the old GT-target-visible keyframe assist.

## No-GT-Inference Checklist

- Proposal pool uses ScanNet annotated bboxes, as expected for NR3D GT-track classification.
- Keyframe selection does not read `target_id` or `object_to_views[target_id]`.
- `target_id` and `gt_bbox_3d_9dof` are used only for offline scoring.
- Guards inspect only agent tool trace and final rationale.

## Commands

The exact commands are the fold-builder, pack-prep, run, metrics, and ingest
commands from `docs/superpowers/plans/2026-05-12-nr3d-v4-agent-guards-fair-views.md`
Tasks 5-6.

## Results

{chr(10).join(table_lines)}

## Interpretation

Overall delta is `{overall_delta * 100:+.2f} pp`, so this pilot is a {interpretation}.

## Raw Artifacts

- Sample ids: `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json`
- Selection summary: `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_summary.json`
- New run: `tmp/nr3d_eval_v4_agent_guards_fair_views_random100/`
- Baseline subset metrics: `tmp/nr3d_eval_v4_baseline_random100/leaderboard_metrics.json`
- SQLite row: `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v4_agent_guards_fair_views_random100_20260512'`

## Caveats

- This is a 100-sample pilot, not a full leaderboard row.
- The baseline reused old v2/v3 outputs rather than rerunning old config.
- The new keyframe mode may reduce target visibility; negative results should trigger evidence-selection audit rather than immediate rejection of Stage-2 guards.
"""

out = Path("docs/benchmark/nr3d/v4_agent_guards_fair_views_20260512.md")
out.write_text(doc, encoding="utf-8")
print(out)
PY
```

Expected: `docs/benchmark/nr3d/v4_agent_guards_fair_views_20260512.md` exists and contains observed metric values.

- [ ] **Step 7: Update NR3D index docs**

Update `docs/benchmark/nr3d/README.md`:

- Add a timeline row for `v4_agent_guards_fair_views_random100_20260512`.
- In "Current Interpretation", keep v3 as the full leaderboard headline and label v4 as partial pilot.

Update `docs/benchmark/nr3d/leaderboard.md`:

- Add a separate "Partial Pilots" table, not the public SOTA table, if one does not exist.
- Include baseline subset and v4 subset rows with `n=100`.

Update `docs/benchmark/README.md` only if its active benchmark table lists current NR3D work; label v4 as pilot.

- [ ] **Step 8: Run focused verification**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest \
  src/evaluation/scripts/tests/test_build_nr3d_v4_random100_fold.py \
  src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_nr3d.py \
  src/evaluation/scripts/tests/test_run_nr3d_vg_side_by_side.py \
  src/evaluation/scripts/tests/test_nr3d_leaderboard_metrics.py \
  src/evaluation/scripts/tests/test_ingest_nr3d_run.py \
  src/agents/tests/test_no_match_guard.py \
  src/agents/tests/test_evidence_frame_guard.py \
  src/agents/tests/test_tadg.py \
  src/agents/tests/test_chassis_tools.py \
  src/agents/tests/test_agent_config_flags.py \
  src/agents/packs/vg_embodiedscan/tests/test_tools.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 9: Run static checks**

Run:

```bash
git diff --check
python -m py_compile \
  scripts/build_nr3d_v4_random100_fold.py \
  src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
  src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  scripts/ingest_nr3d_run.py
```

Expected: both commands exit 0.

- [ ] **Step 10: Commit final code, metrics DB, and docs**

Run:

```bash
git status --short
git add \
  docs/benchmark/nr3d/v4_agent_guards_fair_views_20260512.md \
  docs/benchmark/nr3d/README.md \
  docs/benchmark/nr3d/leaderboard.md \
  docs/benchmark/README.md \
  docs/benchmark/nr3d/runs.sqlite
git commit -m "docs(nr3d): record v4 fair-view random100 pilot"
```

If `docs/benchmark/README.md` or `leaderboard.md` did not need edits, omit them from `git add`.

- [ ] **Step 11: Final status report**

Report:

- Branch and latest commit.
- New v4 Overall / Easy / Hard / View-Dep / View-Indep.
- Baseline subset values and deltas.
- Failure count.
- Whether the result crosses the +1 pp, neutral, or -1 pp threshold.
- Verification commands and pass/fail status.
