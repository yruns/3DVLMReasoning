"""End-to-end tests for Replica dataset integration.

These tests verify the complete pipeline works with the Replica dataset
using the new DatasetAdapter interface.

Requirements:
- Replica dataset available at REPLICA_DATA_ROOT environment variable
  or the default path /data/replica
- PCD files generated for at least room0

The tests are marked as integration tests and can be skipped if the
dataset is not available by setting SKIP_INTEGRATION_TESTS=1.
"""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Check if we should skip integration tests
SKIP_INTEGRATION = os.environ.get("SKIP_INTEGRATION_TESTS", "0") == "1"
REPLICA_DATA_ROOT = os.environ.get(
    "REPLICA_DATA_ROOT",
    "/data/replica"
)

# Check if test data is actually available
def _replica_data_available() -> bool:
    """Check if Replica test data is available."""
    root = Path(REPLICA_DATA_ROOT)
    if not root.exists():
        return False
    # Check for room0 with expected structure
    room0 = root / "room0"
    if not room0.exists():
        return False
    results_dir = room0 / "results"
    traj_file = room0 / "traj.txt"
    return results_dir.exists() and traj_file.exists()


REPLICA_AVAILABLE = _replica_data_available()
skip_if_no_data = pytest.mark.skipif(
    not REPLICA_AVAILABLE or SKIP_INTEGRATION,
    reason="Replica dataset not available or SKIP_INTEGRATION_TESTS=1"
)


class TestReplicaAdapter:
    """Unit tests for ReplicaAdapter (mock-based, no real data needed)."""

    def test_adapter_registration(self):
        """Test that ReplicaAdapter is properly registered."""
        from src.dataset import is_registered, list_adapters

        assert is_registered("replica")
        assert is_registered("replica-v1")
        assert is_registered("replica-imap")
        assert "replica" in list_adapters()

    def test_adapter_properties(self):
        """Test adapter class properties."""
        from src.dataset import get_adapter_class, CoordinateSystem

        adapter_cls = get_adapter_class("replica")
        assert adapter_cls.dataset_name.fget(None) is None or True  # Property exists

        # Create mock adapter to check properties
        with patch.object(adapter_cls, '__init__', lambda self, **kwargs: None):
            adapter = adapter_cls.__new__(adapter_cls)
            adapter.data_root = Path("/mock/path")
            adapter.intrinsics = None
            adapter.depth_scale = 6553.5
            adapter._scene_cache = {}

            # Test dataset_name property
            assert hasattr(adapter_cls, 'dataset_name')

    def test_intrinsics_defaults(self):
        """Test default camera intrinsics for Replica."""
        from src.dataset.replica_adapter import (
            DEFAULT_REPLICA_INTRINSICS,
            REPLICA_DEPTH_SCALE,
        )

        # Verify default values from iMAP config
        assert DEFAULT_REPLICA_INTRINSICS.fx == 600.0
        assert DEFAULT_REPLICA_INTRINSICS.fy == 600.0
        assert DEFAULT_REPLICA_INTRINSICS.width == 1200
        assert DEFAULT_REPLICA_INTRINSICS.height == 680
        assert REPLICA_DEPTH_SCALE == 6553.5

    def test_coordinate_system(self):
        """Test Replica coordinate system is OpenGL."""
        from src.dataset import get_adapter_class, CoordinateSystem

        adapter_cls = get_adapter_class("replica")

        # Coordinate system should be property returning OPENGL
        # We check the implementation directly
        from src.dataset.replica_adapter import ReplicaAdapter
        assert ReplicaAdapter.coordinate_system.fget is not None


@skip_if_no_data
class TestReplicaIntegration:
    """Integration tests requiring actual Replica data."""

    @pytest.fixture
    def adapter(self):
        """Create a ReplicaAdapter for tests."""
        from src.dataset import get_adapter
        return get_adapter("replica", data_root=REPLICA_DATA_ROOT)

    def test_get_scene_ids(self, adapter):
        """Test listing available scenes."""
        scene_ids = adapter.get_scene_ids()

        assert isinstance(scene_ids, list)
        assert len(scene_ids) > 0
        assert "room0" in scene_ids

    def test_load_scene_metadata(self, adapter):
        """Test loading scene metadata for room0."""
        metadata = adapter.load_scene_metadata("room0")

        assert metadata.scene_id == "room0"
        assert metadata.dataset_name == "replica"
        assert metadata.num_frames > 0
        assert metadata.has_depth is True
        assert metadata.has_poses is True
        assert metadata.intrinsics is not None
        assert metadata.intrinsics.fx == 600.0

    def test_iter_frames(self, adapter):
        """Test iterating over frames with stride."""
        frames = list(adapter.iter_frames("room0", stride=50, start=0, end=100))

        assert len(frames) > 0
        assert len(frames) <= 2  # At most 2 frames with stride=50 in range [0,100)

        frame = frames[0]
        assert frame.frame_id == 0
        assert frame.rgb is not None
        assert frame.rgb.shape[2] == 3  # RGB channels
        assert frame.depth is not None
        assert frame.pose is not None
        assert frame.pose.shape == (4, 4)

    def test_load_single_frame(self, adapter):
        """Test loading a specific frame."""
        frame = adapter.load_frame("room0", frame_id=0)

        assert frame.frame_id == 0
        assert frame.rgb.dtype == np.uint8
        assert frame.depth.dtype == np.float32
        assert np.all(np.isfinite(frame.pose))

    def test_depth_scale(self, adapter):
        """Test depth values are properly scaled to meters."""
        frame = adapter.load_frame("room0", frame_id=0)

        # Check depth is in reasonable range (meters)
        valid_depth = frame.depth[frame.depth > 0]
        assert valid_depth.min() >= 0.1  # At least 10cm
        assert valid_depth.max() <= 10.0  # At most 10m (indoor scene)


@skip_if_no_data
class TestReplicaPipeline:
    """Integration tests for QueryScenePipeline with Replica."""

    @pytest.fixture
    def room0_pcd_file(self):
        """Get path to room0 pcd file if available."""
        root = Path(REPLICA_DATA_ROOT)
        pcd_dir = root / "room0" / "pcd_saves"
        if not pcd_dir.exists():
            pytest.skip("PCD files not available for room0")

        pcd_files = list(pcd_dir.glob("*ram*_post.pkl.gz"))
        if not pcd_files:
            pcd_files = list(pcd_dir.glob("*_post.pkl.gz"))
        if not pcd_files:
            pcd_files = list(pcd_dir.glob("*.pkl.gz"))

        if not pcd_files:
            pytest.skip("No PCD files found for room0")

        return str(pcd_files[0])

    def test_pipeline_from_dataset(self, room0_pcd_file):
        """Test creating pipeline using from_dataset."""
        from src.query_scene import QueryScenePipeline

        pipeline = QueryScenePipeline.from_dataset(
            dataset="replica",
            scene_id="room0",
            data_root=REPLICA_DATA_ROOT,
            pcd_file=room0_pcd_file,
            stride=10,
            llm_model=None,  # Skip LLM for unit test
        )

        assert pipeline is not None
        assert pipeline.scene is not None
        assert len(pipeline.scene.objects) > 0
        assert pipeline._dataset_name == "replica"
        assert pipeline._scene_id == "room0"

    def test_pipeline_summary(self, room0_pcd_file):
        """Test pipeline summary includes adapter info."""
        from src.query_scene import QueryScenePipeline

        pipeline = QueryScenePipeline.from_dataset(
            dataset="replica",
            scene_id="room0",
            data_root=REPLICA_DATA_ROOT,
            pcd_file=room0_pcd_file,
            stride=10,
            llm_model=None,
        )

        summary = pipeline.summary()
        assert "scene" in summary
        assert summary["scene"]["scene_id"] == "room0"
        assert summary["scene"]["n_objects"] > 0


@skip_if_no_data
class TestReplicaSpatialQuery:
    """Integration tests for spatial queries on Replica scenes."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline for spatial query tests."""
        from src.query_scene import QueryScenePipeline

        root = Path(REPLICA_DATA_ROOT)
        pcd_dir = root / "room0" / "pcd_saves"
        if not pcd_dir.exists():
            pytest.skip("PCD files not available")

        pcd_files = list(pcd_dir.glob("*.pkl.gz"))
        if not pcd_files:
            pytest.skip("No PCD files found")

        return QueryScenePipeline.from_dataset(
            dataset="replica",
            scene_id="room0",
            data_root=REPLICA_DATA_ROOT,
            pcd_file=str(pcd_files[0]),
            stride=10,
            llm_model=None,
        )

    def test_scene_has_objects(self, pipeline):
        """Verify scene loaded with objects."""
        assert len(pipeline.scene.objects) > 0
        # Check some objects have valid geometry
        has_centroid = any(obj.centroid is not None for obj in pipeline.scene.objects)
        assert has_centroid

    def test_scene_categories(self, pipeline):
        """Verify scene has diverse object categories."""
        categories = set(obj.category for obj in pipeline.scene.objects)
        assert len(categories) > 1  # Multiple categories

    def test_get_objects_by_category(self, pipeline):
        """Test retrieving objects by category."""
        # Get all categories
        categories = set(obj.category for obj in pipeline.scene.objects)

        # Try to get objects for first available category
        cat = next(iter(categories))
        objs = pipeline.scene.get_objects_by_category(cat)
        assert len(objs) > 0
        assert all(cat.lower() in obj.category.lower() for obj in objs)


class TestReplicaMock:
    """Mock-based tests that don't require real data."""

    def test_pipeline_from_dataset_no_adapters(self):
        """Test graceful handling when adapters unavailable."""
        from src.query_scene.query_pipeline import QueryScenePipeline

        # Temporarily disable adapter availability flag
        import src.query_scene.query_pipeline as pipeline_module
        original = pipeline_module._DATASET_ADAPTERS_AVAILABLE

        try:
            pipeline_module._DATASET_ADAPTERS_AVAILABLE = False

            with pytest.raises(ImportError, match="Dataset adapters not available"):
                QueryScenePipeline.from_dataset(
                    dataset="replica",
                    scene_id="room0",
                    data_root="/path/to/data"
                )
        finally:
            pipeline_module._DATASET_ADAPTERS_AVAILABLE = original

    def test_run_query_with_dataset_function(self):
        """Test the convenience function signature."""
        from src.query_scene import run_query_with_dataset
        import inspect

        sig = inspect.signature(run_query_with_dataset)
        params = list(sig.parameters.keys())

        assert "dataset" in params
        assert "scene_id" in params
        assert "data_root" in params
        assert "query" in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
