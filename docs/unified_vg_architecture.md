# Unified Multi-Task Stage 2 Architecture

**Author**: emd-super
**Date**: 2026-04-06
**Branch**: `feat/embodiedscan-grounding`

---

## Goal

Extend Stage 2 to support EmbodiedScan 3D Visual Grounding (VG) alongside OpenEQA QA, using a pluggable benchmark adapter pattern. Stage 1 unchanged.

## Design Principles

1. **Single agent, multiple tasks** — task_type drives prompt/output schema, not code branching
2. **Pluggable benchmarks** — adding a new benchmark = implementing one adapter class
3. **Shared evidence-seeking** — tools (request_more_views, request_crops) work for ALL tasks
4. **No regression** — OpenEQA QA must remain functional and unaffected

---

## Architecture Overview

```
BenchmarkAdapter (pluggable)
  │
  ├── load_samples()      → list[BenchmarkSample]
  ├── build_task_spec()   → Stage2TaskSpec  (task_type, query, output_schema)
  ├── build_bundle()      → Stage2EvidenceBundle  (via Stage1 or direct)
  ├── extract_prediction() → BenchmarkPrediction  (from Stage2AgentResult)
  └── evaluate()          → dict  (metrics)

                ┌─────────────────────────────┐
Sample ────────>│ Stage 1: KeyframeSelector   │  (unchanged)
                └──────────┬──────────────────┘
                           │ KeyframeResult
                ┌──────────▼──────────────────┐
                │ Adapter.build_bundle()      │  (adapter-specific bridge)
                └──────────┬──────────────────┘
                           │ Stage2EvidenceBundle
                ┌──────────▼──────────────────┐
                │ Stage 2: VLM Agent          │
                │ - system_prompt driven by   │
                │   task_type + payload_schema │
                │ - shared tool set           │
                └──────────┬──────────────────┘
                           │ Stage2AgentResult
                ┌──────────▼──────────────────┐
                │ Adapter.extract_prediction() │
                │ Adapter.evaluate()           │
                └─────────────────────────────┘
```

---

## Phase 1: Benchmark Adapter Base Class

### File: `src/benchmarks/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class BenchmarkSample:
    """Minimal unified sample interface."""
    sample_id: str
    scene_id: str          # e.g., "scene0415_00"
    query: str             # Natural language input
    metadata: dict[str, Any] = field(default_factory=dict)

class BenchmarkAdapter(ABC):
    """Pluggable benchmark adapter interface.
    
    Implement this to add a new benchmark. The pipeline calls:
    1. load_samples() — get evaluation samples
    2. build_task_spec(sample) — create Stage2TaskSpec
    3. extract_prediction(result) — extract prediction from agent output
    4. evaluate(predictions, samples) — compute metrics
    """

    @abstractmethod
    def load_samples(self, split: str = "val", **kwargs) -> list[BenchmarkSample]:
        """Load benchmark samples for a given split."""

    @abstractmethod
    def build_task_spec(self, sample: BenchmarkSample) -> Stage2TaskSpec:
        """Create task specification from a sample."""

    @abstractmethod
    def extract_prediction(self, sample: BenchmarkSample, result: Stage2AgentResult) -> dict[str, Any]:
        """Extract benchmark-specific prediction from agent result."""

    @abstractmethod
    def evaluate(self, predictions: list[dict], samples: list[BenchmarkSample]) -> dict[str, Any]:
        """Compute evaluation metrics."""

    def get_scene_path(self, sample: BenchmarkSample) -> Path:
        """Get path to scene data for Stage 1. Override if needed."""
        raise NotImplementedError
```

---

## Phase 2: EmbodiedScan Loader

### File: `src/benchmarks/embodiedscan_loader.py`

```python
@dataclass
class EmbodiedScanVGSample(BenchmarkSample):
    """EmbodiedScan visual grounding sample."""
    text: str = ""                          # VG description
    target_id: int = -1                     # GT instance ID in PKL
    target: str = ""                        # GT object category
    distractor_ids: list[int] = field(default_factory=list)
    anchors: list[str] = field(default_factory=list)
    anchor_ids: list[int] = field(default_factory=list)
    tokens_positive: list[list[int]] = field(default_factory=list)
    # From PKL — populated on demand:
    gt_bbox_3d: list[float] | None = None   # 9-DOF [cx,cy,cz,dx,dy,dz,a,b,g]
    instances: list[dict] | None = None     # All instances in scene
    images_meta: list[dict] | None = None   # Image paths + poses

class EmbodiedScanDataset:
    """EmbodiedScan VG dataset loader."""
    
    @classmethod
    def from_path(
        cls,
        data_root: str | Path,          # data/embodiedscan/
        split: str = "val",
        source_filter: str | None = "scannet",  # "scannet" | "3rscan" | "matterport3d" | None
        max_samples: int | None = None,
        mini: bool = False,              # Use mini VG set
    ) -> EmbodiedScanDataset: ...
    
    def __iter__(self) -> Iterator[EmbodiedScanVGSample]: ...
    def __len__(self) -> int: ...
    
    def get_scene_info(self, scan_id: str) -> dict:
        """Get PKL scene info for a scan_id."""
    
    def get_gt_bbox(self, scan_id: str, target_id: int) -> list[float]:
        """Get GT 3D bbox for target instance."""
```

**Loading logic:**
1. Load `embodiedscan_infos_{split}.pkl` → scene metadata index
2. Load `embodiedscan_{split}_vg.json` (or `_mini_vg.json`)
3. For each VG entry, join with PKL by `scan_id`
4. Populate `gt_bbox_3d` from PKL instances where `bbox_id == target_id`
5. If `source_filter`, keep only matching `scan_id` prefixes

---

## Phase 3: 3D Oriented BBox IoU

### File: `src/benchmarks/embodiedscan_eval.py`

EmbodiedScan uses **9-DOF oriented bboxes** `[cx, cy, cz, dx, dy, dz, alpha, beta, gamma]` (ZXY Euler angles). Cannot use ScanRefer's axis-aligned IoU.

```python
def euler_to_rotation_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """ZXY Euler angles to 3x3 rotation matrix."""

def oriented_bbox_to_corners(bbox_9dof: list[float]) -> np.ndarray:
    """Convert 9-DOF bbox to 8 corners (8, 3)."""

def compute_oriented_iou_3d(bbox1: list[float], bbox2: list[float]) -> float:
    """Compute 3D IoU for oriented bounding boxes.
    
    Use ConvexHull intersection for rotated boxes.
    Fallback: use EmbodiedScan repo's own eval if available.
    """

def evaluate_vg_predictions(
    predictions: list[dict],   # [{sample_id, bbox_3d: [9 floats]}]
    samples: list[EmbodiedScanVGSample],
) -> dict:
    """
    Returns: {
        "acc_025": float,        # Acc@0.25 IoU
        "acc_050": float,        # Acc@0.50 IoU
        "mean_iou": float,
        "per_category": {...},   # category -> {acc_025, acc_050}
        "num_samples": int,
    }
    """
```

**Implementation note**: The oriented 3D IoU computation is non-trivial. Consider reusing EmbodiedScan's own `GroundingMetric` from `data/EmbodiedScan_repo/embodiedscan/`. If too complex, start with axis-aligned approximation (ignore rotation) and iterate.

---

## Phase 4: VG-Specific Stage 2 Behavior

### 4.1 Update payload schema

In `src/agents/runtime/base.py`, update `default_payload_schema`:

```python
if task_type == Stage2TaskType.VISUAL_GROUNDING:
    return {
        "selected_object_id": "int|str",   # ID of selected object from scene graph
        "bbox_3d": "[cx, cy, cz, dx, dy, dz]",  # 3D bbox (from scene graph)
        "target_description": "str",       # What the agent identified
        "grounding_rationale": "str",      # Why this object matches
        "alternative_candidates": ["str"], # Other considered objects
    }
```

### 4.2 Update output instruction

```python
if task_type == Stage2TaskType.VISUAL_GROUNDING:
    return (
        "Identify and localize the described object in the 3D scene. "
        "Select the best matching object from the scene inventory. "
        "Return its object ID and 3D bounding box. "
        "If multiple candidates exist, explain why you chose one over others."
    )
```

### 4.3 VG-specific system prompt section

Add to `build_system_prompt` when `task_type == VISUAL_GROUNDING`:

```
## Visual Grounding Protocol

You are localizing a target object described in natural language.

### Object Candidate List
The scene inventory below lists all detected objects with their 3D locations.
Your task is to SELECT the object that best matches the description.

### Grounding Strategy (in order):
1. Parse the description for target category + spatial constraints
2. Filter candidates by category match
3. If spatial anchors mentioned (e.g., "near the table"), verify spatial relations
4. Use request_crops to verify visual attributes (color, shape, material)
5. Use request_more_views if target is not visible in current keyframes

### Output Requirements:
- selected_object_id: ID from the scene inventory
- bbox_3d: The selected object's 3D bounding box
- grounding_rationale: Evidence for why this object matches

### MANDATORY Rules:
- NEVER guess an object ID without visual verification
- If multiple candidates of same category, MUST use spatial/attribute evidence to disambiguate
- If description mentions spatial relation, verify BOTH target and anchor are visible
```

### 4.4 Object Candidate Formatting

Extend `_format_scene_inventory` or add new method for VG:

```python
@staticmethod
def _format_vg_candidates(
    object_context: dict[str, str],
    extra_metadata: dict[str, Any],  # Contains candidate bboxes
) -> str:
    """Format object candidates with 3D positions for VG task."""
    candidates = extra_metadata.get("vg_candidates", [])
    if not candidates:
        return ""
    lines = ["## Object Candidates for Grounding"]
    for c in candidates:
        lines.append(
            f"- [ID={c['obj_id']}] {c['category']}: "
            f"position=({c['cx']:.2f}, {c['cy']:.2f}, {c['cz']:.2f}), "
            f"size=({c['dx']:.2f}, {c['dy']:.2f}, {c['dz']:.2f})"
        )
        if c.get('description'):
            lines.append(f"  Description: {c['description'][:100]}")
    return "\n".join(lines)
```

---

## Phase 5: EmbodiedScan Benchmark Adapter

### File: `src/agents/adapters/embodiedscan_adapter.py`

```python
class EmbodiedScanVGAdapter(BenchmarkAdapter):
    """Adapter for EmbodiedScan 3D Visual Grounding."""
    
    def __init__(self, data_root: Path, scene_data_root: Path):
        self.data_root = data_root          # data/embodiedscan/
        self.scene_data_root = scene_data_root  # data/embodiedscan/ (scenes)
        self._dataset: EmbodiedScanDataset | None = None
        self._pkl_cache: dict = {}
    
    def load_samples(self, split="val", source_filter="scannet", **kwargs):
        self._dataset = EmbodiedScanDataset.from_path(
            self.data_root, split=split, source_filter=source_filter, **kwargs
        )
        return list(self._dataset)
    
    def build_task_spec(self, sample: EmbodiedScanVGSample) -> Stage2TaskSpec:
        return Stage2TaskSpec(
            task_type=Stage2TaskType.VISUAL_GROUNDING,
            user_query=sample.text,  # e.g., "find the bag closer to the bathtub"
            max_reasoning_turns=6,
        )
    
    def get_scene_path(self, sample: EmbodiedScanVGSample) -> Path:
        """Map scan_id to local ConceptGraph scene path."""
        # scan_id = "scannet/scene0415_00" → scene_data_root/scene0415_00/conceptgraph
        scene_name = sample.scan_id.split("/")[-1]  # "scene0415_00"
        return self.scene_data_root / scene_name
    
    def build_vg_candidates(self, sample, scene_objects) -> list[dict]:
        """Build candidate list from ConceptGraph objects for VLM selection."""
        candidates = []
        for obj in scene_objects:
            candidates.append({
                "obj_id": obj.obj_id,
                "category": obj.category,
                "cx": obj.centroid[0], "cy": obj.centroid[1], "cz": obj.centroid[2],
                "dx": obj.bbox_extent[0], "dy": obj.bbox_extent[1], "dz": obj.bbox_extent[2],
                "description": getattr(obj, "description", ""),
            })
        return candidates
    
    def extract_prediction(self, sample, result: Stage2AgentResult) -> dict:
        """Extract 3D bbox from agent's selected object."""
        payload = result.result.payload
        selected_id = payload.get("selected_object_id")
        bbox_3d = payload.get("bbox_3d")
        return {
            "sample_id": sample.sample_id,
            "bbox_3d": bbox_3d,
            "selected_object_id": selected_id,
            "confidence": result.result.confidence,
        }
    
    def evaluate(self, predictions, samples) -> dict:
        return evaluate_vg_predictions(predictions, samples)
```

---

## Phase 6: Entry Point

### File: `src/agents/examples/embodiedscan_vg_pilot.py`

Mirror structure of `openeqa_official_question_pilot.py`:

```python
def main():
    args = parse_args()
    
    # 1. Load benchmark
    adapter = EmbodiedScanVGAdapter(
        data_root=Path(args.data_root),
        scene_data_root=Path(args.scene_data_root),
    )
    samples = adapter.load_samples(
        split="val",
        source_filter=args.source_filter,
        max_samples=args.max_samples,
    )
    
    # 2. Run pipeline per sample
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one_sample, s, adapter, args): s for s in samples}
        for future in as_completed(futures):
            results.append(future.result())
    
    # 3. Evaluate
    predictions = [r["prediction"] for r in results]
    metrics = adapter.evaluate(predictions, samples)
    print(f"Acc@0.25: {metrics['acc_025']:.3f}")
    print(f"Acc@0.50: {metrics['acc_050']:.3f}")

def run_one_sample(sample, adapter, args):
    # Stage 1: keyframe selection (same as OpenEQA)
    scene_path = adapter.get_scene_path(sample)
    selector = KeyframeSelector.from_scene_path(scene_path)
    stage1_result = selector.select_keyframes_v2(sample.query, k=args.k)
    
    # Build bundle with VG candidates
    bundle = build_stage2_evidence_bundle(
        stage1_result,
        scene_id=sample.scene_id,
        object_context=build_object_context(selector.objects),
    )
    # Inject VG candidates into extra_metadata
    bundle.extra_metadata["vg_candidates"] = adapter.build_vg_candidates(
        sample, selector.objects
    )
    
    # Stage 2: VLM grounding
    task = adapter.build_task_spec(sample)
    agent = Stage2DeepResearchAgent(config=Stage2DeepAgentConfig(...))
    result = agent.run(task, bundle)
    
    # Extract prediction
    prediction = adapter.extract_prediction(sample, result)
    return {"sample": sample, "prediction": prediction, "result": result}
```

---

## Phase 7: OpenEQA Compatibility Adapter

### File: `src/agents/adapters/openeqa_adapter.py`

Wrap existing OpenEQA logic into the adapter pattern for consistency:

```python
class OpenEQAAdapter(BenchmarkAdapter):
    """Adapter for OpenEQA QA benchmark (wraps existing logic)."""
    
    def build_task_spec(self, sample) -> Stage2TaskSpec:
        return Stage2TaskSpec(
            task_type=Stage2TaskType.QA,
            user_query=sample.question,
        )
    
    def extract_prediction(self, sample, result) -> dict:
        return {
            "question_id": sample.sample_id,
            "answer": result.result.payload.get("answer", result.result.summary),
        }
    
    def evaluate(self, predictions, samples) -> dict:
        # Delegate to existing openeqa_official_eval
        return evaluate_predictions_with_official_llm_match(...)
```

This is a **lightweight wrapper**, not a rewrite. Existing OpenEQA entry point continues to work.

---

## Critical Implementation Notes

1. **Object matching for evaluation**: Our ConceptGraph objects have different IDs/bboxes than EmbodiedScan GT. For evaluation, we compute IoU between our predicted bbox and EmbodiedScan's GT bbox. No ID matching needed — pure geometric evaluation.

2. **Oriented vs axis-aligned**: EmbodiedScan uses 9-DOF oriented bboxes. Our ConceptGraph objects typically use axis-aligned bboxes. For initial implementation, use axis-aligned IoU (ignore rotation). Can upgrade later with oriented IoU from EmbodiedScan's evaluation code.

3. **VG candidate list size**: Typical scene has 30-60 objects. All go into the prompt as candidates. If too many (>80), truncate to those visible in keyframes.

4. **Tool reuse**: request_more_views and request_crops work as-is for VG. The VLM uses them to verify object identity before selection.

5. **No training data needed**: We only evaluate. The VLM (GPT-5.4) is zero-shot.

---

## File Creation Summary

| File | Type | Description |
|------|------|-------------|
| `src/benchmarks/base.py` | NEW | BenchmarkAdapter ABC |
| `src/benchmarks/embodiedscan_loader.py` | NEW | EmbodiedScan data loading |
| `src/benchmarks/embodiedscan_eval.py` | NEW | 3D IoU + VG metrics |
| `src/agents/adapters/embodiedscan_adapter.py` | NEW | VG benchmark adapter |
| `src/agents/adapters/openeqa_adapter.py` | NEW | QA benchmark adapter (thin wrapper) |
| `src/agents/examples/embodiedscan_vg_pilot.py` | NEW | VG evaluation entry point |
| `src/agents/runtime/base.py` | MODIFY | VG prompt, payload schema, output instruction |
| `src/benchmarks/__init__.py` | MODIFY | Export new classes |

---

## Execution Order

1. Phase 1: `base.py` (adapter interface)
2. Phase 2: `embodiedscan_loader.py` (data loading) + unit tests
3. Phase 3: `embodiedscan_eval.py` (evaluation) + unit tests
4. Phase 4: Modify `runtime/base.py` (VG prompt/schema)
5. Phase 5: `embodiedscan_adapter.py` (full adapter)
6. Phase 6: `embodiedscan_vg_pilot.py` (entry point)
7. Phase 7: `openeqa_adapter.py` (compatibility wrapper)
8. Integration test: run 5-10 VG samples end-to-end
