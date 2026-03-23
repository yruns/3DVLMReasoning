# Project State Memory

## Scope

This note captures the **current local checkout reality** as of `2026-03-23`.

Use it when repository docs, migration handoff notes, and local files disagree.

If code or disk state changes, update this file instead of letting stale assumptions persist.

## Local Data Reality

### What exists under `data/`

The current checkout has a single populated dataset root:

- `data/OpenEQA/`
  - `scannet/`
  - `89` valid prepared scene directories
  - total size about `37G`

Scene directory names follow:

- `<clip_id>-scannet-<scene_id>`
- example: `002-scannet-scene0709_00`

Each scene is already prepared into a ConceptGraph-style scene package:

- `conceptgraph/000000-rgb.png`
- `conceptgraph/000000-depth.png`
- `conceptgraph/000000.txt`
- `conceptgraph/intrinsic_*.txt`
- `conceptgraph/extrinsic_*.txt`
- `conceptgraph/traj.txt`
- `conceptgraph/mesh.ply`
- `conceptgraph/indices/`
- `conceptgraph/gsa_detections_ram_withbg_allclasses/`
- `conceptgraph/gsa_vis_ram_withbg_allclasses/`
- `conceptgraph/pcd_saves/`
- `conceptgraph/checks/`
- `conceptgraph/scene_info.json`

Frame-count distribution in the current local data:

- `86` scenes have `600` RGB/depth/pose triplets
- `1` scene has `520`
- `1` scene has `536`
- `1` scene has `537`

There are also stale failed-backup directories that inflate disk usage:

- `014-scannet-scene0714_00/conceptgraph_failed_backup_20260322_152647`
- `031-scannet-scene0787_00/conceptgraph_failed_backup_20260322_152647`
- `037-scannet-scene0763_00/conceptgraph_failed_backup_20260322_152647`
- `046-scannet-scene0724_00/conceptgraph_failed_backup_20260322_152647`
- `047-scannet-scene0747_00/conceptgraph_failed_backup_20260322_152647`
- `048-scannet-scene0745_00/conceptgraph_failed_backup_20260322_152647`

### What does not exist locally

The current checkout does **not** contain the official OpenEQA benchmark root expected by the benchmark loader:

- no `data/benchmark/`
- no `data/benchmarks/`
- no `data/OpenEQA/data/open-eqa-v0.json`
- no official `data/frames/<episode_history>/` tree

This matters because `src/benchmarks/openeqa_loader.py` expects:

- `<root>/data/open-eqa-v0.json`
- `<root>/data/frames/...` or `<root>/frames/...`

So the local `data/OpenEQA/` tree should be treated as:

- **prepared scene assets for retrieval / full-pipeline experiments**

not as:

- **official OpenEQA metadata + episode-history benchmark root**

### Scene metadata caveat

`conceptgraph/scene_info.json` records absolute source paths from the machine that originally produced the assets.

Examples point to `/home/ysh/...`, not the current local checkout.

Treat those fields as provenance only. Do not use them as runnable paths.

## Repository Organization Reality

### High-level structure

The intended architecture is still:

1. `src/query_scene/` for Stage 1 evidence retrieval
2. `src/agents/` for Stage 2 agentic reasoning
3. `src/dataset/` for dataset adapters
4. `src/benchmarks/` for benchmark loaders
5. `src/evaluation/` for metrics / scripts / ablations

### Current code-weight distribution

Roughly by Python file count in the local checkout:

- `query_scene`: `62`
- `evaluation`: `52`
- `agents`: `40`
- `scripts`: `12`
- `dataset`: `9`
- `benchmarks`: `8`

Interpretation:

- Stage 1 and evaluation are currently the densest parts of the codebase
- Stage 2 exists and is usable, but the repo still carries migration-era compatibility layers

## Transitional / Compatibility Areas

### Dataset layer

`src/dataset/__init__.py` exposes both:

- the new adapter-based interface
- optional legacy GradSLAM-style dataset classes

This means docs that talk about a single clean dataset API can still be incomplete.

### Stage 1 canonical path

Treat the canonical Stage 1 selector implementation as:

- `src/query_scene/keyframe_selector.py`

Notes:

- `src/query_scene/retrieval/__init__.py` lazily re-exports `KeyframeSelector`
- `src/query_scene/retrieval/keyframe_selector.py` is still present as a legacy duplicate and should not be treated as the authoritative edit target by default

### Stage 1 -> Stage 2 bridge

Treat the canonical bridge as:

- `src/agents/stage1_adapters.py`

Notes:

- `src/agents/adapters.py` is a backward-compatible export shim
- `src/agents/adapters/` and `src/agents/adapters_pkg/` are benchmark-adapter abstractions, not the canonical Stage 1 -> Stage 2 bridge

### Tests

Test layout is split across:

- `src/**/tests`
- `tests/`

But the default pytest config in `pyproject.toml` uses:

- `testpaths = ["src"]`

So plain `pytest` or config-driven test runs do **not** automatically cover every root-level test module.

### Packaging

Wheel packaging currently includes only:

- `src/query_scene`
- `src/agents`
- `src/benchmarks`
- `src/utils`

It currently omits:

- `src/dataset`
- `src/config`
- `src/evaluation`

Do not assume the built wheel mirrors the full source tree.

## Operational Guidance

When working in this repository:

- trust local code and disk state over historical handoff docs
- treat `conceptgraph/*` path references in docs as migration-era vocabulary unless explicitly updated
- verify whether a task needs official benchmark metadata or prepared scene assets; they are not interchangeable here
- prefer updating canonical modules over compatibility shims unless the task is specifically about backward compatibility

## Last Refresh

- refreshed against current local repository and `data/` contents on `2026-03-23`
