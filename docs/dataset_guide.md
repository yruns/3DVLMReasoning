# Dataset Guide

This guide covers how to use the dataset adapter system in 3DVLMReasoning to work with multiple 3D scene datasets.

## Overview

The dataset module provides a unified interface for loading RGB-D data from various 3D scene datasets. It uses an **adapter pattern** that allows transparent access to different datasets through a common API.

## Current Checkout Reality

As of `2026-03-23`, the local repository state is:

- raw adapter code exists for Replica and ScanNet
- local prepared data under `data/` currently contains only `data/OpenEQA/scannet/*/conceptgraph`
- that prepared layout is useful for ConceptGraph-style full-pipeline work
- it is **not** the same thing as the official OpenEQA benchmark repo layout

Important implications:

- `src/dataset/scannet_adapter.py` expects raw ScanNet-style scene folders such as `sceneXXXX_XX/color`, `depth`, `pose`, and `intrinsic`
- `src/benchmarks/openeqa_loader.py` expects official benchmark metadata such as `data/open-eqa-v0.json` and `data/frames/...`

For a fuller repository-state summary, see `docs/current_repo_state.md`.

## Quick Start

```python
from dataset import get_adapter, list_adapters

# List available dataset adapters
print(list_adapters())
# ['replica', 'replica-imap', 'replica-v1', 'scannet', 'scannet-v2']

# Get a dataset adapter
adapter = get_adapter("replica", data_root="/path/to/replica")

# Get scene metadata
metadata = adapter.load_scene_metadata("room0")
print(f"Scene: {metadata.scene_id}")
print(f"Dataset: {metadata.dataset_name}")
print(f"Frames: {metadata.num_frames}")

# Iterate through frames
for frame in adapter.iter_frames("room0", stride=5):
    print(f"Frame {frame.frame_id}: RGB shape {frame.rgb.shape}")
```

## Supported Datasets

### Replica Dataset

**Adapters:** `replica` (primary), `replica-imap`, `replica-v1`

High-quality indoor scenes with dense RGB-D and ground truth poses.

```python
adapter = get_adapter("replica", data_root="/datasets/replica")

# List scenes
scenes = adapter.get_scene_ids()
# ['room0', 'room1', 'room2', 'office0', ...]

# Get specific frame
frame = adapter.load_frame("room0", frame_id=10)
```

**Expected directory structure:**
```
replica/
├── room0/
│   ├── results/
│   │   ├── frame000000.jpg
│   │   ├── depth000000.png
│   │   └── ...
│   └── traj.txt
├── room1/
│   └── ...
└── classes.txt
```

### ScanNet Dataset

**Adapters:** `scannet` (primary), `scannet-v2`

Real-world indoor scenes with noisy depth and camera poses.

```python
adapter = get_adapter("scannet", data_root="/datasets/scannet")

# Invalid poses are skipped automatically by the adapter
for frame in adapter.iter_frames("scene0000_00", stride=5):
    process(frame)
```

**Expected directory structure:**
```
scannet/
└── scans/
    ├── scene0000_00/
    │   ├── color/
    │   │   ├── 0.jpg
    │   │   └── ...
    │   ├── depth/
    │   │   ├── 0.png
    │   │   └── ...
    │   └── pose/
    │       ├── 0.txt
    │       └── ...
    └── scene0000_01/
        └── ...
```

## API Reference

### DatasetAdapter Base Class

All adapters implement this interface:

```python
class DatasetAdapter(ABC):
    """Abstract base class for dataset adapters."""

    @abstractmethod
    def get_scene_ids(self) -> list[str]:
        """List available scene IDs."""

    @abstractmethod
    def load_scene_metadata(self, scene_id: str) -> SceneMetadata:
        """Load metadata for a scene."""

    @abstractmethod
    def load_frame(self, scene_id: str, frame_id: int) -> FrameData:
        """Load a single frame by index."""

    @abstractmethod
    def iter_frames(
        self,
        scene_id: str,
        stride: int = 1,
        start: int = 0,
        end: int | None = None,
    ) -> Iterator[FrameData]:
        """Iterate through frames with configurable stride."""
```

### FrameData

```python
@dataclass
class FrameData:
    """A single RGB-D frame with camera pose."""

    frame_id: int
    rgb: np.ndarray         # Shape: (H, W, 3), dtype: uint8
    depth: np.ndarray | None = None
    pose: np.ndarray | None = None
    intrinsics: CameraIntrinsics | None = None
    timestamp: float | None = None
```

### SceneMetadata

```python
@dataclass
class SceneMetadata:
    """Metadata about a scene."""

    scene_id: str
    dataset_name: str
    num_frames: int
    has_depth: bool = True
    has_poses: bool = True
    has_mesh: bool = False
    has_semantics: bool = False
    coordinate_system: CoordinateSystem = CoordinateSystem.OPENGL
    intrinsics: CameraIntrinsics | None = None
    scene_bounds: tuple[np.ndarray, np.ndarray] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
```

### CameraIntrinsics

```python
@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""

    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int
    height: int

    def to_matrix(self) -> np.ndarray:
        """Return 3x3 intrinsic matrix."""
```

## Configuration

### Default Configuration

Dataset defaults are configured in `src/config/datasets.yaml`:

```yaml
defaults:
  stride: 1
  start: 0
  end: null

replica:
  camera:
    fx: 600.0
    fy: 600.0
    cx: 599.5
    width: 1200
    height: 680
  depth:
    scale: 6553.5
    format: png
  coordinate_system: opengl

scannet:
  camera:
    width: 1296
    height: 968
  depth:
    scale: 1000.0
    format: png
  coordinate_system: scannet
```

### Per-Call Configuration

Override defaults when getting an adapter:

```python
adapter = get_adapter(
    "replica",
    data_root="/my/custom/path",
)
```

Or when iterating frames:

```python
for frame in adapter.iter_frames("room0", stride=20, start=100, end=500):
    ...
```

## Advanced Usage

### Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor

adapter = get_adapter("replica", data_root="/datasets/replica")
scenes = adapter.get_scene_ids()

def process_scene(scene_id):
    frames = list(adapter.iter_frames(scene_id, stride=10))
    # Process frames...
    return len(frames)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_scene, scenes))
```

### Memory-Efficient Streaming

For large scenes, use generators:

```python
def stream_rgbd(adapter, scene_id, batch_size=10):
    """Stream frames in batches."""
    batch = []
    for frame in adapter.iter_frames(scene_id, stride=5):
        batch.append(frame)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# Process without loading all frames into memory
for batch in stream_rgbd(adapter, "room0", batch_size=20):
    process_batch(batch)
```

### Custom Adapters

Create a new adapter by subclassing:

```python
from dataset import DatasetAdapter, register_adapter, FrameData

@register_adapter("my-dataset", aliases=["my-ds"])
class MyDatasetAdapter(DatasetAdapter):
    """Adapter for my custom dataset."""

    def __init__(self, data_root: str, **kwargs):
        self.data_root = Path(data_root)
        # Initialize...

    def list_scenes(self) -> list[str]:
        return [p.name for p in self.data_root.iterdir() if p.is_dir()]

    def get_scene_metadata(self, scene_id: str) -> SceneMetadata:
        return SceneMetadata(
            scene_id=scene_id,
            num_frames=self._count_frames(scene_id),
            categories=self._load_categories(scene_id),
            coordinate_system=CoordinateSystem.CUSTOM,
        )

    def get_frame(self, scene_id: str, frame_idx: int) -> FrameData:
        # Load single frame...
        pass

    def iter_frames(self, scene_id, stride=1, **kwargs):
        # Iterate frames...
        pass
```

After registering, use like any other adapter:

```python
adapter = get_adapter("my-dataset", data_root="/path/to/data")
```

## Integration with Query Scene

### Using run_query_with_dataset

```python
from query_scene import run_query_with_dataset
from dataset import get_adapter

# Simple usage
adapter = get_adapter("replica", data_root="/datasets/replica")

result = run_query_with_dataset(
    adapter=adapter,
    scene_id="room0",
    query="the pillow on the sofa",
    k=5,
)

# Access results
print(f"Target objects: {result.target_objects}")
print(f"Keyframe indices: {result.keyframe_indices}")
```

### Building Scene Indices

```python
from query_scene import SceneIndices

# Build indices from adapter
indices = SceneIndices.from_adapter(adapter, scene_id="room0")

# Use for queries
selector = KeyframeSelector(
    scene_indices=indices,
    scene_categories=adapter.get_scene_metadata("room0").categories,
)
```

## Troubleshooting

### Missing Frames

If frames are missing, check:
1. File naming convention matches adapter expectations
2. All required files exist (RGB, depth, pose)

```python
# Debug frame loading
try:
    frame = adapter.get_frame("room0", frame_idx=0)
except FileNotFoundError as e:
    print(f"Missing files: {e}")
```

### Invalid Poses

ScanNet often has invalid poses. Handle them:

```python
# Option 1: Skip invalid poses
for frame in adapter.iter_frames("scene0000_00", skip_invalid_poses=True):
    ...

# Option 2: Check manually
for frame in adapter.iter_frames("scene0000_00"):
    if np.isnan(frame.pose).any():
        print(f"Frame {frame.frame_idx} has invalid pose")
        continue
    process(frame)
```

### Coordinate System Mismatches

Different datasets use different coordinate conventions:

```python
from dataset import CoordinateSystem

# Check coordinate system
metadata = adapter.get_scene_metadata("room0")
print(f"Coordinate system: {metadata.coordinate_system}")

# Transform if needed
if metadata.coordinate_system == CoordinateSystem.SCANNET:
    pose = scannet_to_replica_transform(frame.pose)
```

## Performance Tips

1. **Use stride**: Don't process every frame unless necessary
   ```python
   adapter.iter_frames("room0", stride=5)  # Every 5th frame
   ```

2. **Limit range**: Process only relevant portions
   ```python
   adapter.iter_frames("room0", start_idx=100, end_idx=200)
   ```

3. **Lazy loading**: Frames are loaded on-demand, not all at once

4. **Parallel processing**: Use thread pools for I/O-bound loading
