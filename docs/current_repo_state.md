# Current Repo State

This note records the **current local checkout reality** as of `2026-03-23`.

Use it together with the architecture and dataset guides when local files do not match older migration handoff notes.

## Summary

The repository has already landed the intended two-stage structure:

- `src/query_scene/` for Stage 1 retrieval
- `src/agents/` for Stage 2 reasoning
- `src/dataset/` for adapters
- `src/benchmarks/` for benchmark loaders
- `src/evaluation/` for metrics, scripts, and ablations

But the checkout is still in a **migration-transition** phase:

- compatibility shims are still present
- some docs still use old `conceptgraph/*` language
- local data layout does not fully match the benchmark loader assumptions

## Local Data Inventory

### What is present

The current `data/` directory contains:

- `data/OpenEQA/`
  - `scannet/`
  - `89` valid prepared scene directories
  - about `37G` on disk

Each scene is already packaged in a ConceptGraph-style prepared layout:

- `conceptgraph/*-rgb.png`
- `conceptgraph/*-depth.png`
- `conceptgraph/*.txt` pose files
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

Most scenes have `600` frames. Three scenes currently have shorter prepared sequences:

- `120-scannet-scene0684_01`: `537`
- `154-scannet-scene0193_01`: `536`
- `156-scannet-scene0461_00`: `520`

There are also stale failed-backup directories under a few scenes. They are historical artifacts and should not be treated as active assets.

### What is absent

The current checkout does **not** include the official OpenEQA benchmark root expected by `src/benchmarks/openeqa_loader.py`:

- no `data/benchmark/`
- no `data/benchmarks/`
- no `data/OpenEQA/data/open-eqa-v0.json`
- no official `data/frames/<episode_history>/` tree

## What This Means Operationally

### For Stage 1 / full-pipeline work

The local `data/OpenEQA/scannet/*/conceptgraph` trees are suitable for:

- prepared-scene retrieval experiments
- scene-graph-based Stage 1 debugging
- full-pipeline experiments that consume prepared ConceptGraph-style scene packages

### For benchmark loaders

The benchmark loader for OpenEQA expects official benchmark metadata and episode-history directories, not prepared scene packages.

In other words:

- `data/OpenEQA/scannet/*/conceptgraph` is **not** a drop-in replacement for the official OpenEQA benchmark repo layout

### For dataset adapters

The standard `ScanNetAdapter` in `src/dataset/scannet_adapter.py` expects raw ScanNet scene folders such as:

- `scene0000_00/color/`
- `scene0000_00/depth/`
- `scene0000_00/pose/`
- `scene0000_00/intrinsic/`

It does not directly consume the prepared `conceptgraph/` scene layout.

## Repository Transition Notes

### Canonical Stage 1 selector

Treat this file as the canonical Stage 1 selector implementation:

- `src/query_scene/keyframe_selector.py`

Notes:

- `src/query_scene/retrieval/__init__.py` lazily re-exports selector types
- `src/query_scene/retrieval/keyframe_selector.py` still exists as a legacy duplicate and should not be assumed canonical by default

### Canonical Stage 1 -> Stage 2 bridge

Treat this file as the canonical Stage 1 -> Stage 2 adapter:

- `src/agents/stage1_adapters.py`

Notes:

- `src/agents/adapters.py` is a compatibility shim
- `src/agents/adapters/` and `src/agents/adapters_pkg/` are benchmark-adapter abstractions, not the main Stage 1 -> Stage 2 bridge

### Test layout

Tests are split across:

- `src/**/tests`
- `tests/`

But `pyproject.toml` currently uses:

```toml
testpaths = ["src"]
```

So config-driven default pytest discovery does not automatically include every root-level test module.

### Packaging

The wheel build currently packages:

- `src/query_scene`
- `src/agents`
- `src/benchmarks`
- `src/utils`

It currently omits:

- `src/dataset`
- `src/config`
- `src/evaluation`

If you need those modules in an installed package workflow, verify packaging first.

## Documentation Interpretation Rule

When older docs mention:

- `conceptgraph/*`
- `data/benchmarks/*`
- `data/benchmark/*`

treat them as historical migration context unless they have been explicitly reconciled with the current checkout.
