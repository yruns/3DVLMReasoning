# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management

**This project uses `uv` for Python package management.** All package operations should use `uv`:

```bash
# Install dependencies (creates .venv automatically)
uv pip install -e ".[dev]"

# Install with all optional dependencies
uv pip install -e ".[dev,full,agents]"

# Add new dependency
uv pip install <package>

# Sync from pyproject.toml
uv pip sync
```

Note: The `agents` optional dependency group requires ByteDance internal pypi for `deepagents>=0.4.0`.

## Development Commands

```bash
# Run all tests
pytest src/ -v

# Run Stage 1 tests only
pytest src/query_scene/tests/ -v

# Run Stage 2 tests only
pytest src/agents/tests/ -v

# Run benchmark tests only
pytest src/benchmarks/tests/ -v

# Run single test file
pytest src/agents/tests/test_stage2_deep_agent.py -v

# Run single test function
pytest src/agents/tests/test_stage2_deep_agent.py::test_function_name -v

# Linting
ruff check src/

# Formatting
black src/

# Type checking
mypy src/
```

## Architecture Overview

This is a **two-stage framework for 3D scene understanding with evidence-seeking VLM agents**:

### Stage 1: Query-Driven Keyframe Retrieval (`src/query_scene/`)

Handles task-conditioned evidence retrieval from 3D scene graphs:

- **Query parsing**: `query_parser.py` - Parses natural language queries into structured `HypothesisOutputV1` using LLM
- **Query execution**: `query_executor.py` - Executes parsed queries against scene indices
- **Keyframe selection**: `keyframe_selector.py` - Main entry point (`select_keyframes_v2()`) for selecting task-relevant keyframes
- **Scene indices**: `index_builder.py` - Multi-granularity CLIP indexing (region → object → point), visibility index, spatial index
- **Spatial relations**: `spatial_relations.py` - Geometric relation checking between objects

The Stage 1 output is treated as **visual evidence entry points** (high recall, not necessarily high precision), not final answers.

### Stage 2: VLM Agentic Reasoning (`src/agents/`)

A ReAct-style VLM agent built on **LangChain v1 + DeepAgents** that reasons over retrieved evidence:

- **Agent core**: `stage2_deep_agent.py` - `Stage2DeepResearchAgent` class
- **Data models**: `models.py` - Pydantic schemas (`Stage2TaskSpec`, `Stage2EvidenceBundle`, `Stage2AgentResult`, etc.)
- **Adapters**: `adapters.py` - Bridge Stage 1 output to Stage 2 input (`build_stage2_evidence_bundle()`)
- **Benchmark adapters**: `benchmark_adapters.py` - Unified interface for OpenEQA, ScanNet, Replica, etc.
- **Tools**: `tools/` - Agent tools like `request_crops.py`, `hypothesis_repair.py`
- **Tracing**: `trace.py`, `trace_server.py` - Execution trace recording and HTML rendering

### Key Design Principles

1. **Hypothesis as soft prior**: Stage 2 treats Stage 1 hypotheses as soft priors to verify/correct, not ground truth
2. **Evidence-seeking**: Agent actively decides what additional evidence to request (more views, crops, BEV)
3. **Unified task interface**: Single agent handles QA, visual grounding, navigation planning, manipulation
4. **Budget-aware reasoning**: Agent operates within fixed token/image budgets

### Benchmark Loaders (`src/benchmarks/`)

- `openeqa_loader.py` - OpenEQA dataset
- `sqa3d_loader.py` - SQA3D dataset
- `scanrefer_loader.py` - ScanRefer dataset

## Current Local Checkout Reality

These notes reflect the local repository state verified on **2026-03-23**.

### Data under `data/`

The local checkout currently has:

- `data/OpenEQA/scannet/`
- `89` valid prepared scene directories
- about `37G` of prepared OpenEQA ScanNet scene assets

Each scene is already packaged into a ConceptGraph-style prepared layout under:

- `data/OpenEQA/scannet/<clip_id>/conceptgraph/`

with assets such as:

- `*-rgb.png`, `*-depth.png`, pose `*.txt`
- `intrinsic_*.txt`, `extrinsic_*.txt`, `traj.txt`
- `mesh.ply`
- `indices/`
- `gsa_detections_ram_withbg_allclasses/`
- `gsa_vis_ram_withbg_allclasses/`
- `pcd_saves/`
- `scene_info.json`

Important:

- this is **not** the official OpenEQA benchmark repo layout
- there is currently **no** `data/benchmark/` or `data/benchmarks/`
- there is currently **no** local `data/open-eqa-v0.json`

Implication:

- use `data/OpenEQA/scannet/*/conceptgraph` for prepared-scene / full-pipeline work
- do **not** assume `src/benchmarks/openeqa_loader.py` can load the local `data/OpenEQA/` tree as-is

### Dataset interface caveat

The standard `ScanNetAdapter` expects raw ScanNet-style scene folders such as:

- `sceneXXXX_XX/color/`
- `sceneXXXX_XX/depth/`
- `sceneXXXX_XX/pose/`
- `sceneXXXX_XX/intrinsic/`

It does **not** directly consume the prepared `conceptgraph/` scene packages under `data/OpenEQA/scannet/`.

### Scene metadata provenance caveat

`conceptgraph/scene_info.json` files may contain old absolute source paths from the machine that originally generated the assets (for example `/home/ysh/...`).

Treat those fields as provenance only, not as runnable local paths.

## Repository Transition Notes

The repository is functionally migrated, but it still carries compatibility layers and historical terminology.

### Canonical modules to prefer

- Stage 1 selector implementation: `src/query_scene/keyframe_selector.py`
- Stage 1 -> Stage 2 bridge: `src/agents/stage1_adapters.py`

### Compatibility areas still present

- `src/query_scene/retrieval/__init__.py` lazily re-exports selector symbols
- `src/query_scene/retrieval/keyframe_selector.py` still exists as a legacy duplicate
- `src/agents/adapters.py` is a backward-compatible export shim
- `src/agents/adapters/` and `src/agents/adapters_pkg/` are benchmark-adapter abstractions, not the canonical Stage 1 -> Stage 2 bridge

### Tests and packaging

Test layout is split across:

- `src/**/tests`
- `tests/`

But `pyproject.toml` currently uses:

- `testpaths = ["src"]`

So default pytest discovery does not automatically include every root-level test module.

Wheel packaging currently includes:

- `src/query_scene`
- `src/agents`
- `src/benchmarks`
- `src/utils`

and currently omits:

- `src/dataset`
- `src/config`
- `src/evaluation`

Do not assume the built wheel mirrors the full source tree.

### Documentation interpretation rule

When docs mention:

- `conceptgraph/*`
- `data/benchmark/*`
- `data/benchmarks/*`

verify whether the note is historical migration context or current local truth before acting on it.

## Stage 2 Backend Configuration

Default VLM backend is `gpt-5.2-2025-12-11` via Azure-compatible endpoint. Configuration is in `Stage2DeepAgentConfig`:

- Uses single-key `AzureChatOpenAI` client (not connection pooling)
- Base URL: internal GenAI endpoint
- Session ID required in `extra_body` for prompt caching
- Gemini available as override but not default (unstable FC with DeepAgents)

## JSON Schemas

Output schemas are in `schema/` directory:
- `hypothesis_output_v1.json` - Schema for Stage 1 query parsing output

## Automation Scripts

### auto-claude.sh

Migration automation script that invokes Claude for multi-phase tasks. **Must use `ttadk` CLI** (ByteDance internal):

```bash
# Run full migration
./auto-claude.sh

# Run specific phase (1-6)
./auto-claude.sh --phase 2

# Preview without changes
./auto-claude.sh --dry-run

# Check migration status
./auto-claude.sh --status
```

The script uses `ttadk code` with piped prompts:
```bash
echo "$prompt" | ttadk code --model "claude-opus-4-5" \
    -a "--dangerously-skip-permissions --print --output-format stream-json"
```

Note: The standard `claude` CLI fallback does not work in this environment.
