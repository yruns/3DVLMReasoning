"""Dataset module for 3D scene RGB-D data loading.

Provides:
1. New adapter-based interface (DatasetAdapter, ReplicaAdapter, ScanNetAdapter)
2. Legacy PyTorch dataset classes for GradSLAM (ReplicaDataset, ScannetDataset, etc.)

New Adapter Interface (recommended):
    >>> from dataset import get_adapter
    >>> adapter = get_adapter("replica", data_root="/path/to/replica")
    >>> for frame in adapter.iter_frames("room0", stride=5):
    ...     process(frame.rgb, frame.depth, frame.pose)

Legacy GradSLAM Interface (requires full dependencies):
    >>> from dataset import ReplicaDataset, load_dataset_config
    >>> config = load_dataset_config("replica.yaml")
    >>> dataset = ReplicaDataset(config, basedir="/path/to", sequence="room0")
"""

# New adapter-based interface (always available)
from .base import (
    CameraIntrinsics,
    CoordinateSystem,
    DatasetAdapter,
    FrameData,
    SceneMetadata,
)
from .registry import (
    AdapterFactory,
    configure_default_root,
    create_adapter,
    get_adapter,
    get_adapter_class,
    is_registered,
    list_adapters,
    list_primary_adapters,
    register_adapter,
)

# Import adapters to trigger registration
from .replica_adapter import ReplicaAdapter

# Constants (always available)
from .replica_constants import (
    REPLICA_CLASSES,
    REPLICA_SCENE_IDS,
    REPLICA_SCENE_IDS_,
)
from .scannet_adapter import ScanNetAdapter

# Legacy GradSLAM interface (optional, requires full dependencies)
# Import these separately when needed
_LEGACY_IMPORTS_AVAILABLE = False
try:
    from .datasets_common import (  # noqa: F401
        Ai2thorDataset,
        AzureKinectDataset,
        # Base class
        GradSLAMDataset,
        Hm3dDataset,
        Hm3dOpeneqaDataset,
        # Dataset classes
        ICLDataset,
        MultiscanDataset,
        RealsenseDataset,
        Record3DDataset,
        ReplicaDataset,
        ScannetDataset,
        ScannetOpenEQADataset,
        # Utility functions
        as_intrinsics_matrix,
        common_dataset_to_batch,
        from_intrinsics_matrix,
        get_dataset,
        load_dataset_config,
        readEXR_onlydepth,
        update_recursive,
    )

    _LEGACY_IMPORTS_AVAILABLE = True
except ImportError:
    # Legacy dependencies not installed, legacy classes not available
    pass

__all__ = [
    # New adapter interface
    "CoordinateSystem",
    "CameraIntrinsics",
    "SceneMetadata",
    "FrameData",
    "DatasetAdapter",
    "register_adapter",
    "get_adapter",
    "get_adapter_class",
    "list_adapters",
    "list_primary_adapters",
    "is_registered",
    "AdapterFactory",
    "configure_default_root",
    "create_adapter",
    # Adapter implementations
    "ReplicaAdapter",
    "ScanNetAdapter",
    # Constants
    "REPLICA_CLASSES",
    "REPLICA_SCENE_IDS",
    "REPLICA_SCENE_IDS_",
]

# Add legacy exports only if available
if _LEGACY_IMPORTS_AVAILABLE:
    __all__.extend(
        [
            # Utility functions (legacy)
            "as_intrinsics_matrix",
            "from_intrinsics_matrix",
            "readEXR_onlydepth",
            "load_dataset_config",
            "update_recursive",
            "common_dataset_to_batch",
            "get_dataset",
            # Base class (legacy)
            "GradSLAMDataset",
            # Dataset classes (legacy)
            "ICLDataset",
            "ReplicaDataset",
            "ScannetDataset",
            "Ai2thorDataset",
            "AzureKinectDataset",
            "RealsenseDataset",
            "Record3DDataset",
            "MultiscanDataset",
            "Hm3dDataset",
            "Hm3dOpeneqaDataset",
            "ScannetOpenEQADataset",
        ]
    )
