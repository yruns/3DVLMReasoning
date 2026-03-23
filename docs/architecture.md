# Architecture Overview

This document describes the architecture of the 3DVLMReasoning framework.

For the **current local checkout reality** including dataset inventory, transition notes, and packaging/test caveats, see `docs/current_repo_state.md`.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           3DVLMReasoning                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐   │
│  │   Dataset     │    │  Query Scene  │    │   Agents              │   │
│  │   Adapters    │───▶│  (Stage 1)    │───▶│   (Stage 2)           │   │
│  └───────────────┘    └───────────────┘    └───────────────────────┘   │
│         │                    │                        │                 │
│         │                    ▼                        ▼                 │
│         │            ┌───────────────┐    ┌───────────────────────┐   │
│         │            │  Evidence     │    │   Structured          │   │
│         │            │  Bundle       │    │   Response            │   │
│         │            └───────────────┘    └───────────────────────┘   │
│         │                                                               │
│         ▼                                                               │
│  ┌───────────────┐                                                      │
│  │  Evaluation   │                                                      │
│  │  Framework    │                                                      │
│  └───────────────┘                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module Overview

### 1. Dataset Module (`src/dataset/`)

The dataset module provides a unified interface for loading RGB-D data from various 3D scene datasets.

#### Core Components

- **DatasetAdapter** (`base.py`): Abstract base class defining the adapter interface
- **Registry** (`registry.py`): Adapter registration and factory pattern
- **ReplicaAdapter** (`replica_adapter.py`): Replica dataset implementation
- **ScanNetAdapter** (`scannet_adapter.py`): ScanNet dataset implementation

#### Design Pattern

Uses the **Adapter Pattern** with a **Registry**:

```python
@register_adapter("replica", aliases=["replica-imap", "replica-v1"])
class ReplicaAdapter(DatasetAdapter):
    def iter_frames(self, scene_id, stride=5):
        ...
```

#### Key Data Types

```python
@dataclass
class FrameData:
    """A single RGB-D frame."""
    frame_id: int
    rgb: np.ndarray        # (H, W, 3)
    depth: np.ndarray | None
    pose: np.ndarray | None
    intrinsics: CameraIntrinsics | None
    timestamp: float | None = None

@dataclass
class SceneMetadata:
    """Metadata for a scene."""
    scene_id: str
    dataset_name: str
    num_frames: int
    coordinate_system: CoordinateSystem
    extra: dict[str, Any] = field(default_factory=dict)
```

### 2. Query Scene Module (`src/query_scene/`)

Stage 1 of the pipeline: task-conditioned keyframe retrieval.

#### Core Components

- **QueryParser** (`query_parser.py`): LLM-based natural language query parsing
- **QueryExecutor** (`query_executor.py`): Query execution against scene indices
- **KeyframeSelector** (`keyframe_selector.py`): Main entry point for keyframe selection
- **SpatialRelationChecker** (`spatial_relations.py`): Geometric relation verification

#### Data Flow

```
User Query
    │
    ▼
┌─────────────────┐
│  QueryParser    │──▶ HypothesisOutputV1
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  QueryExecutor  │──▶ ExecutionResult (matched objects)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ KeyframeSelector│──▶ KeyframeResult (views, scores)
└─────────────────┘
```

#### Query Structure Types

```python
@dataclass
class HypothesisOutputV1:
    """Parsed query output."""
    parse_mode: ParseMode  # SINGLE or MULTI
    hypotheses: list[QueryHypothesis]

@dataclass
class QueryHypothesis:
    """A single interpretation of the query."""
    kind: HypothesisKind  # DIRECT, PROXY, or CONTEXT
    rank: int
    grounding_query: GroundingQuery
    lexical_hints: list[str]

@dataclass
class GroundingQuery:
    """Executable grounding query."""
    raw_query: str
    root: QueryNode
    expect_unique: bool
```

#### Indices

The system maintains several indices for efficient retrieval:

- **CLIPIndex**: Semantic similarity using CLIP embeddings
- **VisibilityIndex**: View-object visibility mapping
- **SpatialIndex**: 3D spatial relationships

### 3. Agents Module (`src/agents/`)

Stage 2 of the pipeline: VLM agentic reasoning.

#### Core Components

- **Stage2DeepResearchAgent** (`stage2_deep_agent.py`): Main agent class
- **Adapters** (`adapters.py`): Stage 1 → Stage 2 bridge
- **BenchmarkAdapters** (`benchmark_adapters.py`): Unified benchmark interface

#### Agent Architecture

Built on **LangChain v1 + DeepAgents**:

```
┌─────────────────────────────────────────────────┐
│             Stage2DeepResearchAgent             │
├─────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────┐   │
│  │            ReAct Loop                    │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │   │
│  │  │ Observe │─▶│  Think  │─▶│   Act   │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  │   │
│  └─────────────────────────────────────────┘   │
│                      │                          │
│                      ▼                          │
│  ┌─────────────────────────────────────────┐   │
│  │              Tools                       │   │
│  │  • request_crops   • request_views       │   │
│  │  • hypothesis_repair • request_bev       │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

#### Key Data Types

```python
@dataclass
class Stage2TaskSpec:
    """Task specification for Stage 2."""
    task_type: TaskType  # QA, GROUNDING, NAVIGATION, MANIPULATION
    question: str
    stage1_evidence: Stage2EvidenceBundle
    budget: EvidenceBudget

@dataclass
class Stage2AgentResult:
    """Agent execution result."""
    success: bool
    structured_response: dict
    evidence_used: list[EvidenceItem]
    uncertainty: float
    trace: ExecutionTrace
```

### 4. Evaluation Module (`src/evaluation/`)

Comprehensive evaluation framework.

#### Components

- **BatchEval** (`batch_eval.py`): Batch evaluation runner
- **Metrics** (`metrics.py`): Evaluation metrics
- **Ablations** (`ablations/`): Ablation study scripts
- **Scripts** (`scripts/`): Benchmark-specific runners

### 5. Configuration Module (`src/config/`)

Centralized configuration management.

```yaml
# datasets.yaml
replica:
  data_root: /path/to/replica
  default_stride: 5
  coordinate_system: replica

scannet:
  data_root: /path/to/scannet
  default_stride: 10
  coordinate_system: scannet
```

## Current Transition Notes

The target architecture is already visible in the codebase, but the repository is still carrying migration-era compatibility layers.

### Canonical paths to prefer

- Stage 1 selector: `src/query_scene/keyframe_selector.py`
- Stage 1 -> Stage 2 bridge: `src/agents/stage1_adapters.py`

### Compatibility areas still present

- `src/query_scene/retrieval/__init__.py` re-exports selector symbols lazily
- `src/query_scene/retrieval/keyframe_selector.py` still exists as a legacy duplicate
- `src/agents/adapters.py` is a backward-compatible shim
- `src/agents/adapters/` and `src/agents/adapters_pkg/` are benchmark adapter abstractions, not the canonical Stage 1 -> Stage 2 bridge

### Local data caveat

The current local `data/` directory is populated with prepared OpenEQA ScanNet ConceptGraph-style scene packages under `data/OpenEQA/scannet/*/conceptgraph`.

This is suitable for prepared-scene retrieval and full-pipeline experiments, but it is not the same layout expected by:

- the raw `ScanNetAdapter`
- the official OpenEQA benchmark loader

## Design Principles

### 1. Hypothesis as Soft Prior

Stage 2 treats Stage 1 hypotheses as **soft priors to verify/correct**, not ground truth:

```python
# Stage 1 provides initial hypothesis
hypothesis = stage1.parse("the pillow on the sofa")

# Stage 2 validates and may correct
if not agent.verify_hypothesis(hypothesis, evidence):
    corrected = agent.repair_hypothesis(hypothesis)
```

### 2. Evidence-Seeking

The agent actively decides what additional evidence to request:

```python
while budget.remaining > 0:
    evidence_need = agent.assess_evidence_gap()
    if evidence_need.type == "MORE_VIEWS":
        new_evidence = request_additional_views(...)
    elif evidence_need.type == "CROPS":
        new_evidence = request_object_crops(...)
```

### 3. Unified Task Interface

Single agent handles multiple task types:

```python
# Same agent, different tasks
agent = Stage2DeepResearchAgent()

qa_result = agent.run(qa_task)           # Question answering
ground_result = agent.run(grounding_task) # Visual grounding
nav_result = agent.run(navigation_task)   # Navigation planning
```

### 4. Budget-Aware Reasoning

Agent operates within fixed token/image budgets:

```python
@dataclass
class EvidenceBudget:
    max_images: int = 10
    max_tokens: int = 4000
    max_tool_calls: int = 5
```

## Data Flow Example

Complete pipeline for a query "Find the red pillow on the sofa":

```
1. Dataset Loading
   adapter = get_adapter("replica")
   frames = list(adapter.iter_frames("room0"))

2. Scene Graph Construction (external)
   scene_graph = build_scene_graph(frames)

3. Stage 1: Query Parsing
   parser = QueryParser(scene_categories=categories)
   hypothesis = parser.parse("Find the red pillow on the sofa")
   # HypothesisOutputV1(parse_mode=SINGLE, hypotheses=[
   #   QueryHypothesis(kind=DIRECT, grounding_query=...)
   # ])

4. Stage 1: Query Execution
   executor = QueryExecutor(scene_graph)
   result = executor.execute(hypothesis)
   # ExecutionResult(matched_objects=[obj_42, obj_67])

5. Stage 1: Keyframe Selection
   selector = KeyframeSelector(scene_path)
   keyframes = selector.select_from_objects(result.matched_objects)
   # KeyframeResult(keyframe_indices=[10, 25, 30])

6. Stage 2: Evidence Bundle
   evidence = build_stage2_evidence_bundle(keyframes)

7. Stage 2: Agent Reasoning
   agent = Stage2DeepResearchAgent()
   task = Stage2TaskSpec(
       task_type=TaskType.GROUNDING,
       question="Find the red pillow on the sofa",
       stage1_evidence=evidence
   )
   result = agent.run(task)
   # Stage2AgentResult(success=True, structured_response={
   #   "object_id": 42,
   #   "bounding_box": [...],
   #   "confidence": 0.95
   # })
```

## Extension Points

### Adding a New Dataset

1. Create adapter class implementing `DatasetAdapter`
2. Register with `@register_adapter` decorator
3. Add configuration to `datasets.yaml`

### Adding a New Agent Tool

1. Create tool class in `agents/tools/`
2. Implement `__call__` method
3. Register in agent's tool list

### Adding a New Benchmark

1. Create loader in `benchmarks/`
2. Implement `BenchmarkAdapter` interface
3. Add evaluation script in `evaluation/scripts/`
