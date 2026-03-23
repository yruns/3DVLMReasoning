# Dataset Configuration Guide

This guide explains how to configure datasets in the 3DVLMReasoning project.

## Configuration File

The main configuration file is located at `src/config/datasets.yaml`. It contains:
- Default values applied to all datasets
- Dataset-specific configurations (Replica, ScanNet, HM3D, etc.)
- Camera parameters, depth scales, coordinate systems
- Coordinate transformation matrices

## Loading Configuration

### Python API

```python
from src.config import get_dataset_config, get_datasets_config

# Load configuration for a specific dataset
config = get_dataset_config("replica")
print(config["camera"])  # Camera parameters
print(config["depth"]["scale"])  # Depth scale

# Load all dataset configurations
all_configs = get_datasets_config()
```

### Dataset Adapters

Dataset adapters automatically load their configuration:

```python
from src.dataset import ReplicaAdapter

# Adapter loads configuration from datasets.yaml
adapter = ReplicaAdapter(
    scene_path="/path/to/replica/room0",
    stride=5  # Optional override
)
```

## Environment Variable Overrides

You can override any configuration value using environment variables. This is useful for:
- Running different experiments with different settings
- Deploying to different environments
- CI/CD pipelines
- Containerized deployments

### Syntax

Environment variables follow the pattern:
```
DATASET_{DATASET_NAME}_{KEY}
```

For nested keys, use double underscores (`__`):
```
DATASET_{DATASET_NAME}_{KEY1}__{KEY2}
```

### Examples

#### Top-level overrides

```bash
# Override frame stride
export DATASET_REPLICA_STRIDE=10

# Override start and end frames
export DATASET_REPLICA_START=100
export DATASET_REPLICA_END=500
```

#### Nested overrides (camera parameters)

```bash
# Override camera dimensions
export DATASET_REPLICA_CAMERA__WIDTH=1920
export DATASET_REPLICA_CAMERA__HEIGHT=1080

# Override camera intrinsics
export DATASET_REPLICA_CAMERA__FX=800.5
export DATASET_REPLICA_CAMERA__FY=800.5
```

#### Nested overrides (depth configuration)

```bash
# Override depth scale
export DATASET_SCANNET_DEPTH__SCALE=2000.0

# Override depth format
export DATASET_SCANNET_DEPTH__FORMAT=exr
```

#### Multiple datasets

```bash
# Configure multiple datasets independently
export DATASET_REPLICA_STRIDE=5
export DATASET_SCANNET_STRIDE=10
export DATASET_HM3D_STRIDE=2
```

### Type Inference

Environment variable values are automatically converted to the appropriate type:

- **Integer**: `"42"` → `42`
- **Float**: `"3.14"` → `3.14`
- **Boolean**: `"true"` / `"false"` → `True` / `False`
- **Null**: `"null"` / `"none"` → `None`
- **String**: Any other value remains a string

Examples:
```bash
export DATASET_REPLICA_STRIDE=10              # int
export DATASET_REPLICA_CAMERA__FX=600.5       # float
export DATASET_REPLICA_ENABLE_CACHING=true    # bool
export DATASET_REPLICA_END=null               # None
export DATASET_REPLICA_COORD_SYSTEM=custom    # str
```

## Dataset-Specific Configurations

### Replica Dataset

```yaml
replica:
  camera:
    fx: 600.0
    fy: 600.0
    cx: 599.5
    cy: 339.5
    width: 1200
    height: 680
  depth:
    scale: 6553.5
    format: png
  coordinate_system: opengl
```

Override examples:
```bash
export DATASET_REPLICA_CAMERA__WIDTH=1920
export DATASET_REPLICA_DEPTH__SCALE=1000.0
```

### ScanNet Dataset

```yaml
scannet:
  camera:
    width: 1296
    height: 968
  depth:
    scale: 1000.0
    format: png
  coordinate_system: scannet
```

Override examples:
```bash
export DATASET_SCANNET_STRIDE=5
export DATASET_SCANNET_CAMERA__WIDTH=1920
export DATASET_SCANNET_DEPTH__SCALE=2000.0
```

## Use Cases

### Local Development

Create a `.env` file for local overrides:
```bash
# .env
DATASET_REPLICA_STRIDE=10
DATASET_REPLICA_START=0
DATASET_REPLICA_END=100
```

Load with python-dotenv:
```python
from dotenv import load_dotenv
load_dotenv()

from src.config import get_dataset_config
config = get_dataset_config("replica")
```

### Docker Containers

Pass environment variables via Docker:
```bash
docker run \
  -e DATASET_REPLICA_STRIDE=5 \
  -e DATASET_REPLICA_CAMERA__WIDTH=1920 \
  your-image:latest
```

Or via docker-compose.yml:
```yaml
services:
  app:
    image: your-image:latest
    environment:
      DATASET_REPLICA_STRIDE: 5
      DATASET_REPLICA_CAMERA__WIDTH: 1920
```

### CI/CD Pipelines

GitHub Actions:
```yaml
env:
  DATASET_REPLICA_STRIDE: 10
  DATASET_REPLICA_START: 0
  DATASET_REPLICA_END: 100

steps:
  - name: Run tests
    run: pytest tests/
```

### Experiment Management

Different configurations for different experiments:
```bash
# Experiment 1: Low resolution, high stride
export DATASET_REPLICA_STRIDE=20
export DATASET_REPLICA_CAMERA__WIDTH=640
export DATASET_REPLICA_CAMERA__HEIGHT=480
python scripts/run_experiment.py --name exp1

# Experiment 2: High resolution, low stride
export DATASET_REPLICA_STRIDE=1
export DATASET_REPLICA_CAMERA__WIDTH=1920
export DATASET_REPLICA_CAMERA__HEIGHT=1080
python scripts/run_experiment.py --name exp2
```

## Disabling Environment Overrides

If you need to ignore environment variables and use only the config file:

```python
config = get_dataset_config("replica", apply_env_overrides=False)
```

## Troubleshooting

### Configuration not loading

Check that the dataset name is correct (case-insensitive):
```python
# These all work
config = get_dataset_config("replica")
config = get_dataset_config("REPLICA")
config = get_dataset_config("Replica")
```

### Environment variables not working

1. Check the variable name format: `DATASET_{NAME}_{KEY}`
2. Use double underscores for nested keys: `DATASET_REPLICA_CAMERA__WIDTH`
3. Verify the environment variable is set: `echo $DATASET_REPLICA_STRIDE`
4. Check if overrides are enabled: `apply_env_overrides=True` (default)

### Type conversion issues

If you get unexpected types, explicitly convert:
```python
config = get_dataset_config("replica")
stride = int(config["stride"])  # Force int conversion
```

## Advanced Usage

### Custom Configuration Files

Load custom configuration files:
```python
from src.config import load_config

custom_config = load_config("/path/to/custom_config.yaml")
```

### Merging Configurations

Merge default and custom configurations:
```python
from src.config import get_dataset_config

default_config = get_dataset_config("replica", apply_env_overrides=False)
custom_config = {"stride": 20, "camera": {"width": 1920}}

# Merge configs (custom overrides default)
merged = {**default_config, **custom_config}
```

### Dynamic Configuration

Generate configuration programmatically:
```python
import os
from src.config import get_dataset_config

# Set environment variables dynamically
os.environ["DATASET_REPLICA_STRIDE"] = "15"
os.environ["DATASET_REPLICA_CAMERA__WIDTH"] = "1920"

# Load with overrides
config = get_dataset_config("replica")
```

## Testing Configuration

Run configuration tests:
```bash
pytest tests/config/test_datasets_config.py -v
```

Test environment variable overrides:
```bash
export DATASET_REPLICA_STRIDE=25
pytest tests/config/test_datasets_config.py::TestEnvironmentOverrides -v
```
