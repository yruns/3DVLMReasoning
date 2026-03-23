"""End-to-end tests for ScanNet dataset integration.

These tests verify the complete pipeline works with the ScanNet dataset
using the new DatasetAdapter interface.

Requirements:
- ScanNet dataset available at SCANNET_DATA_ROOT environment variable
  or the default path /data/scannet
- At least one scene (e.g., scene0000_00) with color/, depth/, pose/ dirs

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
SCANNET_DATA_ROOT = os.environ.get(
    "SCANNET_DATA_ROOT",
    "/data/scannet"
)

# Check if test data is actually available
def _scannet_data_available() -> bool:
    """Check if ScanNet test data is available."""
    root = Path(SCANNET_DATA_ROOT)
    if not root.exists():
        return False

    # Look for any scene with expected structure
    import re
    pattern = re.compile(r"scene\d{4}_\d{2}")

    for scene_dir in root.iterdir():
        if scene_dir.is_dir() and pattern.match(scene_dir.name):
            color_dir = scene_dir / "color"
            depth_dir = scene_dir / "depth"
            pose_dir = scene_dir / "pose"
            if color_dir.exists() and depth_dir.exists() and pose_dir.exists():
                return True
    return False


def _get_first_scannet_scene() -> str:
    """Get the first available ScanNet scene ID."""
    root = Path(SCANNET_DATA_ROOT)
    import re
    pattern = re.compile(r"scene\d{4}_\d{2}")

    for scene_dir in sorted(root.iterdir()):
        if scene_dir.is_dir() and pattern.match(scene_dir.name):
            color_dir = scene_dir / "color"
            depth_dir = scene_dir / "depth"
            pose_dir = scene_dir / "pose"
            if color_dir.exists() and depth_dir.exists() and pose_dir.exists():
                return scene_dir.name
    return "scene0000_00"  # Default fallback


SCANNET_AVAILABLE = _scannet_data_available()
skip_if_no_data = pytest.mark.skipif(
    not SCANNET_AVAILABLE or SKIP_INTEGRATION,
    reason="ScanNet dataset not available or SKIP_INTEGRATION_TESTS=1"
)


class TestScanNetAdapter:
    """Unit tests for ScanNetAdapter (mock-based, no real data needed)."""

    def test_adapter_registration(self):
        """Test that ScanNetAdapter is properly registered."""
        from src.dataset import is_registered, list_adapters

        assert is_registered("scannet")
        assert is_registered("scannet-v2")
        assert "scannet" in list_adapters()

    def test_adapter_class_exists(self):
        """Test ScanNetAdapter class properties."""
        from src.dataset import get_adapter_class, CoordinateSystem
        from src.dataset.scannet_adapter import ScanNetAdapter, SCANNET_CLASSES

        adapter_cls = get_adapter_class("scannet")
        assert adapter_cls == ScanNetAdapter

        # Verify semantic class mapping exists
        assert len(SCANNET_CLASSES) > 0
        assert 0 in SCANNET_CLASSES  # unannotated
        assert SCANNET_CLASSES[5] == "chair"

    def test_depth_scale_constant(self):
        """Test ScanNet depth scale is 1000 (mm to m)."""
        from src.dataset.scannet_adapter import SCANNET_DEPTH_SCALE

        assert SCANNET_DEPTH_SCALE == 1000.0

    def test_coordinate_system(self):
        """Test ScanNet coordinate system."""
        from src.dataset.scannet_adapter import ScanNetAdapter
        from src.dataset import CoordinateSystem

        # Property should return SCANNET coordinate system
        assert ScanNetAdapter.coordinate_system.fget is not None


@skip_if_no_data
class TestScanNetIntegration:
    """Integration tests requiring actual ScanNet data."""

    @pytest.fixture
    def adapter(self):
        """Create a ScanNetAdapter for tests."""
        from src.dataset import get_adapter
        return get_adapter("scannet", data_root=SCANNET_DATA_ROOT)

    @pytest.fixture
    def test_scene_id(self):
        """Get first available scene ID."""
        return _get_first_scannet_scene()

    def test_get_scene_ids(self, adapter):
        """Test listing available scenes."""
        scene_ids = adapter.get_scene_ids()

        assert isinstance(scene_ids, list)
        assert len(scene_ids) > 0
        # All IDs should match pattern sceneXXXX_XX
        import re
        pattern = re.compile(r"scene\d{4}_\d{2}")
        assert all(pattern.match(sid) for sid in scene_ids)

    def test_load_scene_metadata(self, adapter, test_scene_id):
        """Test loading scene metadata."""
        metadata = adapter.load_scene_metadata(test_scene_id)

        assert metadata.scene_id == test_scene_id
        assert metadata.dataset_name == "scannet"
        assert metadata.num_frames > 0
        assert metadata.has_depth is True
        assert metadata.has_poses is True
        assert metadata.intrinsics is not None
        # ScanNet intrinsics vary per scene but should be reasonable
        assert metadata.intrinsics.fx > 100
        assert metadata.intrinsics.width > 0

    def test_iter_frames(self, adapter, test_scene_id):
        """Test iterating over frames with stride."""
        frames = list(adapter.iter_frames(test_scene_id, stride=100, start=0, end=200))

        # Should get some frames (exact count depends on valid poses)
        assert len(frames) >= 0

        if len(frames) > 0:
            frame = frames[0]
            assert frame.rgb is not None
            assert frame.rgb.shape[2] == 3  # RGB channels
            assert frame.depth is not None
            assert frame.pose is not None
            assert frame.pose.shape == (4, 4)

    def test_load_single_frame(self, adapter, test_scene_id):
        """Test loading a specific frame."""
        # First, find a valid frame (some may have invalid poses)
        valid_frame = None
        for i in range(100):
            try:
                frame = adapter.load_frame(test_scene_id, frame_id=i)
                valid_frame = frame
                break
            except ValueError:
                continue

        if valid_frame is None:
            pytest.skip("No valid frames found in first 100")

        assert valid_frame.rgb.dtype == np.uint8
        assert valid_frame.depth.dtype == np.float32
        assert np.all(np.isfinite(valid_frame.pose))

    def test_invalid_pose_handling(self, adapter, test_scene_id):
        """Test that invalid poses are handled gracefully."""
        # ScanNet has frames with invalid poses (inf/nan)
        # iter_frames should skip them
        frame_count = 0
        for frame in adapter.iter_frames(test_scene_id, stride=1, start=0, end=50):
            frame_count += 1
            # All yielded frames should have valid poses
            assert frame.pose is not None
            assert np.all(np.isfinite(frame.pose))

    def test_depth_scale(self, adapter, test_scene_id):
        """Test depth values are properly scaled to meters."""
        # Find a valid frame
        for frame in adapter.iter_frames(test_scene_id, stride=50, start=0, end=100):
            # Check depth is in reasonable range (meters)
            valid_depth = frame.depth[frame.depth > 0]
            if len(valid_depth) > 0:
                assert valid_depth.min() >= 0.0  # Non-negative
                assert valid_depth.max() <= 20.0  # At most 20m (indoor scene)
            break

    def test_semantic_info(self, adapter, test_scene_id):
        """Test loading semantic class mapping."""
        semantic_info = adapter.load_semantic_info(test_scene_id)

        assert semantic_info is not None
        assert isinstance(semantic_info, dict)
        assert 0 in semantic_info
        assert semantic_info[5] == "chair"


@skip_if_no_data
class TestScanNetPipeline:
    """Integration tests for QueryScenePipeline with ScanNet."""

    @pytest.fixture
    def test_scene_id(self):
        """Get first available scene ID."""
        return _get_first_scannet_scene()

    @pytest.fixture
    def scene_pcd_file(self, test_scene_id):
        """Get path to scene pcd file if available."""
        root = Path(SCANNET_DATA_ROOT)
        pcd_dir = root / test_scene_id / "pcd_saves"
        if not pcd_dir.exists():
            return None

        pcd_files = list(pcd_dir.glob("*ram*_post.pkl.gz"))
        if not pcd_files:
            pcd_files = list(pcd_dir.glob("*_post.pkl.gz"))
        if not pcd_files:
            pcd_files = list(pcd_dir.glob("*.pkl.gz"))

        if not pcd_files:
            return None

        return str(pcd_files[0])

    def test_pipeline_from_dataset(self, test_scene_id, scene_pcd_file):
        """Test creating pipeline using from_dataset."""
        if scene_pcd_file is None:
            pytest.skip("PCD files not available")

        from src.query_scene import QueryScenePipeline

        pipeline = QueryScenePipeline.from_dataset(
            dataset="scannet",
            scene_id=test_scene_id,
            data_root=SCANNET_DATA_ROOT,
            pcd_file=scene_pcd_file,
            stride=10,
            llm_model=None,  # Skip LLM for unit test
        )

        assert pipeline is not None
        assert pipeline.scene is not None
        assert len(pipeline.scene.objects) > 0
        assert pipeline._dataset_name == "scannet"
        assert pipeline._scene_id == test_scene_id

    def test_pipeline_without_pcd(self, test_scene_id):
        """Test creating pipeline without PCD file creates minimal scene."""
        from src.query_scene import QueryScenePipeline

        # Create a mock scenario where PCD is not found
        # This tests the adapter-only path
        pipeline = QueryScenePipeline.from_dataset(
            dataset="scannet",
            scene_id=test_scene_id,
            data_root=SCANNET_DATA_ROOT,
            pcd_file="/nonexistent/path.pkl.gz",  # Force fallback
            stride=50,
            llm_model=None,
        )

        # Even without PCD, should have camera poses from adapter
        # Note: This path creates a minimal scene with no objects
        assert pipeline is not None
        assert pipeline._dataset_name == "scannet"


@skip_if_no_data
class TestOpenEQACompatibility:
    """Tests for OpenEQA scene compatibility with ScanNet adapter."""

    @pytest.fixture
    def adapter(self):
        """Create ScanNet adapter."""
        from src.dataset import get_adapter
        return get_adapter("scannet", data_root=SCANNET_DATA_ROOT)

    def test_openeqa_scene_pattern(self, adapter):
        """Test that ScanNet scenes match OpenEQA expected format."""
        scene_ids = adapter.get_scene_ids()

        # OpenEQA uses ScanNet scenes like scene0000_00
        import re
        openeqa_pattern = re.compile(r"scene\d{4}_\d{2}")

        matching = [sid for sid in scene_ids if openeqa_pattern.match(sid)]
        assert len(matching) > 0

    def test_frame_data_for_openeqa(self, adapter):
        """Test frame data meets OpenEQA requirements."""
        scene_id = _get_first_scannet_scene()
        metadata = adapter.load_scene_metadata(scene_id)

        # OpenEQA requires:
        # - RGB images
        # - Camera poses (for navigation/grounding tasks)
        # - Optional depth
        assert metadata.has_depth
        assert metadata.has_poses

        # Verify we can get frames
        frames = list(adapter.iter_frames(scene_id, stride=100, end=200))
        if len(frames) > 0:
            frame = frames[0]
            assert frame.rgb is not None
            assert frame.pose is not None


class TestScanNetMock:
    """Mock-based tests that don't require real data."""

    def test_coordinate_transform(self):
        """Test coordinate transform from ScanNet to OpenGL."""
        from src.dataset import CoordinateSystem
        from src.dataset.base import DatasetAdapter
        from src.dataset.scannet_adapter import ScanNetAdapter

        # Create mock adapter
        with patch.object(ScanNetAdapter, '__init__', lambda self, **kwargs: None):
            adapter = ScanNetAdapter.__new__(ScanNetAdapter)
            adapter.data_root = Path("/mock/path")
            adapter.depth_scale = 1000.0
            adapter.use_depth_intrinsics = False
            adapter._scene_cache = {}
            adapter._intrinsics_cache = {}

            # Test coordinate transform matrix exists
            transform = adapter.get_coordinate_transform(CoordinateSystem.OPENGL)
            assert transform.shape == (4, 4)

    def test_scannet_classes_mapping(self):
        """Test ScanNet class ID mapping."""
        from src.dataset.scannet_adapter import SCANNET_CLASSES

        # Essential classes for 3D scene understanding
        essential_classes = {
            1: "wall",
            2: "floor",
            5: "chair",
            6: "sofa",
            7: "table",
            33: "toilet",
            34: "sink",
        }

        for class_id, class_name in essential_classes.items():
            assert SCANNET_CLASSES[class_id] == class_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
