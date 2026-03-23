
## Project Thesis

This repository is organized around a two-stage 3D reasoning story:

1. Stage 1 retrieves task-relevant visual evidence from scene assets.
2. Stage 2 reasons over that evidence with an agentic VLM workflow.

The intended scientific value is not just “scene graph + LLM”. The stronger framing is:
- evidence-seeking
- symbolic-to-visual repair
- uncertainty-aware reasoning
- shared policy across multiple task types

## Current Stage Ownership

### Stage 1
- package: `src/query_scene/`
- role: parse query, retrieve candidate objects/views/keyframes, provide soft structured priors
- output should be treated as evidence proposal, not final truth

### Stage 2
- package: `src/agents/`
- role: consume Stage 1 output and perform evidence-grounded reasoning
- current main user-facing class: `Stage2DeepResearchAgent`
- runtime implementation now lives in `src/agents/runtime/`

## Current Stage 2 Architecture Reality

### Main code paths
- `src/agents/stage2_deep_agent.py`
  - compatibility wrapper around runtime
- `src/agents/runtime/base.py`
  - shared runtime state and helpers
- `src/agents/runtime/deepagents_agent.py`
  - main DeepAgents runtime implementation
- `src/agents/runtime/langchain_agent.py`
  - provider / LangChain compatibility glue
- `src/agents/stage1_adapters.py`
  - converts Stage 1 outputs into `Stage2EvidenceBundle`

### Important migration detail
The old Stage 1-to-Stage 2 adapter module is no longer the canonical `agents.adapters.py` file. The repo now uses:
- `src/agents/stage1_adapters.py`

There is also a new `src/agents/adapters/` package for benchmark adapter abstractions, which means examples or docs that still mention `agents.adapters` are stale unless they are explicitly referring to the benchmark package.

## Agent Design Principles

The intended Stage 2 behavior remains:
- Stage 1 hypothesis is a soft prior
- the agent can request more evidence if needed
- the output should expose confidence and uncertainty instead of hallucinated certainty

Useful design vocabulary for papers, docs, or planning:
- adaptive evidence acquisition
- symbolic-to-visual repair
- evidence-grounded uncertainty
- unified task-conditioned reasoning

## Current Practical Constraints

### What is implemented enough to trust
- top-level `agents` package imports successfully
- benchmark and evaluation infrastructure exists
- Stage 2 runtime split has landed in code

### What is still in transition
- some tests still import runtime symbols from old locations
- at least one test expects `Stage2RuntimeState` to be importable from `agents.stage2_deep_agent`, but that symbol currently lives in `agents.runtime.base`
- some docs and examples still point at `conceptgraph.*` or old adapter paths

This means Stage 2 architecture is present, but compatibility polish is incomplete.

## Benchmark / Evaluation Story

The repository already contains a meaningful evaluation layer under `src/evaluation/` and `tests/evaluation/`.

Locally verified on 2026-03-22:
- `tests/config + tests/evaluation` -> `469 passed`

So the evaluation framework is one of the strongest migrated areas of the codebase right now.

## What To Emphasize In Future Work

### High-value directions
- make Stage 2 genuinely exploit evidence refinement instead of one-shot answering
- tighten the interface between Stage 1 metadata and Stage 2 repair loops
- turn uncertainty output into a first-class measurable behavior, not a logging afterthought
- keep the same agent interface usable across QA, grounding, navigation, and manipulation

### Low-value directions to avoid
- overselling Stage 1 as exact truth
- describing the system as static scene-graph reasoning
- relying on old `conceptgraph` package boundaries in docs or examples

## Current Operational Truth

The repository should currently be described as:
- functionally migrated
- architecturally much cleaner than the source layout
- partially validated
- not yet fully reconciled at the compatibility, test, and documentation layers

That is a stronger and more accurate claim than saying “migration fully complete”.
