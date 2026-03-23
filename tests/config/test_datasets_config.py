"""Tests for dataset configuration loading and environment variable overrides."""

import os
from pathlib import Path

import pytest

from src.config import get_dataset_config, get_datasets_config, load_config


class TestConfigLoading:
    """Test basic configuration file loading."""

    def test_load_datasets_config(self):
        """Test loading the entire datasets configuration."""
        config = get_datasets_config()

        assert config is not None
        assert isinstance(config, dict)
        assert "defaults" in config
        assert "replica" in config
        assert "scannet" in config

    def test_load_dataset_config_replica(self):
        """Test loading Replica dataset configuration."""
        config = get_dataset_config("replica", apply_env_overrides=False)

        assert config is not None
        assert config["description"] == "Replica: High-quality 3D reconstruction dataset"
        assert config["coordinate_system"] == "opengl"

        # Check camera parameters
        assert "camera" in config
        assert config["camera"]["fx"] == 600.0
        assert config["camera"]["fy"] == 600.0
        assert config["camera"]["width"] == 1200
        assert config["camera"]["height"] == 680

        # Check depth configuration
        assert "depth" in config
        assert config["depth"]["scale"] == 6553.5
        assert config["depth"]["format"] == "png"

    def test_load_dataset_config_scannet(self):
        """Test loading ScanNet dataset configuration."""
        config = get_dataset_config("scannet", apply_env_overrides=False)

        assert config is not None
        assert config["description"] == "ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes"
        assert config["coordinate_system"] == "scannet"

        # Check camera parameters
        assert "camera" in config
        assert config["camera"]["width"] == 1296
        assert config["camera"]["height"] == 968

        # Check depth configuration
        assert config["depth"]["scale"] == 1000.0

    def test_load_nonexistent_dataset(self):
        """Test loading a non-existent dataset returns None."""
        config = get_dataset_config("nonexistent")
        assert config is None

    def test_case_insensitive_dataset_names(self):
        """Test that dataset names are case-insensitive."""
        config1 = get_dataset_config("replica", apply_env_overrides=False)
        config2 = get_dataset_config("REPLICA", apply_env_overrides=False)
        config3 = get_dataset_config("Replica", apply_env_overrides=False)

        assert config1 == config2 == config3

    def test_defaults_applied(self):
        """Test that default values are applied to dataset configs."""
        config = get_dataset_config("replica", apply_env_overrides=False)

        # Check defaults from config
        assert config["stride"] == 1
        assert config["start"] == 0
        assert config["end"] is None


class TestEnvironmentOverrides:
    """Test environment variable override functionality."""

    def setup_method(self):
        """Clean environment before each test."""
        self._original_env = dict(os.environ)
        # Remove any existing dataset env vars
        for key in list(os.environ.keys()):
            if key.startswith("DATASET_"):
                del os.environ[key]

    def teardown_method(self):
        """Restore original environment after each test."""
        os.environ.clear()
        os.environ.update(self._original_env)

    def test_simple_override(self):
        """Test simple top-level override."""
        os.environ["DATASET_REPLICA_STRIDE"] = "10"

        config = get_dataset_config("replica")

        assert config["stride"] == 10

    def test_nested_override_camera(self):
        """Test nested override for camera parameters."""
        os.environ["DATASET_REPLICA_CAMERA__WIDTH"] = "1920"
        os.environ["DATASET_REPLICA_CAMERA__HEIGHT"] = "1080"
        os.environ["DATASET_REPLICA_CAMERA__FX"] = "800.5"

        config = get_dataset_config("replica")

        assert config["camera"]["width"] == 1920
        assert config["camera"]["height"] == 1080
        assert config["camera"]["fx"] == 800.5

    def test_nested_override_depth(self):
        """Test nested override for depth configuration."""
        os.environ["DATASET_SCANNET_DEPTH__SCALE"] = "2000.0"
        os.environ["DATASET_SCANNET_DEPTH__FORMAT"] = "exr"

        config = get_dataset_config("scannet")

        assert config["depth"]["scale"] == 2000.0
        assert config["depth"]["format"] == "exr"

    def test_type_inference_int(self):
        """Test that integer values are correctly inferred."""
        os.environ["DATASET_REPLICA_STRIDE"] = "42"
        os.environ["DATASET_REPLICA_START"] = "-10"

        config = get_dataset_config("replica")

        assert config["stride"] == 42
        assert isinstance(config["stride"], int)
        assert config["start"] == -10
        assert isinstance(config["start"], int)

    def test_type_inference_float(self):
        """Test that float values are correctly inferred."""
        os.environ["DATASET_REPLICA_CAMERA__FX"] = "1234.56"
        os.environ["DATASET_REPLICA_DEPTH__SCALE"] = "3.14159"

        config = get_dataset_config("replica")

        assert config["camera"]["fx"] == 1234.56
        assert isinstance(config["camera"]["fx"], float)
        assert config["depth"]["scale"] == 3.14159

    def test_type_inference_bool(self):
        """Test that boolean values are correctly inferred."""
        os.environ["DATASET_REPLICA_SOME_FLAG"] = "true"
        os.environ["DATASET_REPLICA_OTHER_FLAG"] = "False"

        config = get_dataset_config("replica")

        assert config["some_flag"] is True
        assert isinstance(config["some_flag"], bool)
        assert config["other_flag"] is False

    def test_type_inference_null(self):
        """Test that null/none values are correctly inferred."""
        os.environ["DATASET_REPLICA_END"] = "null"
        os.environ["DATASET_REPLICA_OTHER"] = "None"

        config = get_dataset_config("replica")

        assert config["end"] is None
        assert config["other"] is None

    def test_type_inference_string(self):
        """Test that string values are preserved."""
        os.environ["DATASET_REPLICA_COORDINATE_SYSTEM"] = "custom"
        os.environ["DATASET_REPLICA_DESCRIPTION"] = "My custom description"

        config = get_dataset_config("replica")

        assert config["coordinate_system"] == "custom"
        assert isinstance(config["coordinate_system"], str)
        assert config["description"] == "My custom description"

    def test_multiple_datasets_independent(self):
        """Test that overrides for different datasets don't interfere."""
        os.environ["DATASET_REPLICA_STRIDE"] = "5"
        os.environ["DATASET_SCANNET_STRIDE"] = "10"

        replica_config = get_dataset_config("replica")
        scannet_config = get_dataset_config("scannet")

        assert replica_config["stride"] == 5
        assert scannet_config["stride"] == 10

    def test_override_only_affects_specific_dataset(self):
        """Test that overrides only affect the specified dataset."""
        os.environ["DATASET_REPLICA_STRIDE"] = "20"

        replica_config = get_dataset_config("replica")
        scannet_config = get_dataset_config("scannet")

        assert replica_config["stride"] == 20
        # ScanNet should have default stride
        assert scannet_config["stride"] == 1

    def test_case_insensitive_env_vars(self):
        """Test that environment variable parsing is case-insensitive for keys."""
        os.environ["DATASET_REPLICA_STRIDE"] = "15"

        # Should work with different case variations
        config = get_dataset_config("replica")
        assert config["stride"] == 15

    def test_disable_env_overrides(self):
        """Test that env overrides can be disabled."""
        os.environ["DATASET_REPLICA_STRIDE"] = "99"

        config = get_dataset_config("replica", apply_env_overrides=False)

        # Should have default value, not override
        assert config["stride"] == 1

    def test_create_new_nested_key(self):
        """Test that overrides can create new nested structures."""
        os.environ["DATASET_REPLICA_CUSTOM__NEW_KEY"] = "test_value"

        config = get_dataset_config("replica")

        assert "custom" in config
        assert config["custom"]["new_key"] == "test_value"

    def test_deep_nesting(self):
        """Test multiple levels of nesting in overrides."""
        os.environ["DATASET_REPLICA_LEVEL1__LEVEL2__LEVEL3"] = "deep_value"

        config = get_dataset_config("replica")

        assert config["level1"]["level2"]["level3"] == "deep_value"

    def test_combined_defaults_and_overrides(self):
        """Test that defaults and overrides work together."""
        os.environ["DATASET_REPLICA_STRIDE"] = "25"
        # Don't override 'start' or 'end'

        config = get_dataset_config("replica")

        # Overridden value
        assert config["stride"] == 25
        # Default values
        assert config["start"] == 0
        assert config["end"] is None


class TestConfiguredDatasets:
    """Test that all expected datasets are properly configured."""

    def test_all_expected_datasets_present(self):
        """Test that all expected datasets are in the config."""
        config = get_datasets_config()

        expected_datasets = ["replica", "scannet", "hm3d", "ai2thor"]
        for dataset in expected_datasets:
            assert dataset in config

    def test_all_datasets_have_coordinate_system(self):
        """Test that all datasets specify a coordinate system."""
        config = get_datasets_config()

        for dataset_name, dataset_config in config.items():
            # Skip special keys
            if dataset_name in ["defaults", "coordinate_transforms", "openeqa", "scanrefer", "sqa3d"]:
                continue

            assert "coordinate_system" in dataset_config, f"{dataset_name} missing coordinate_system"

    def test_coordinate_transforms_present(self):
        """Test that coordinate transform matrices are defined."""
        config = get_datasets_config()

        assert "coordinate_transforms" in config
        transforms = config["coordinate_transforms"]

        assert "opencv_to_opengl" in transforms
        assert "scannet_to_opengl" in transforms

        # Check matrix dimensions (4x4)
        for transform_name, matrix in transforms.items():
            assert len(matrix) == 4, f"{transform_name} should have 4 rows"
            for row in matrix:
                assert len(row) == 4, f"{transform_name} rows should have 4 elements"


class TestConfigHelpers:
    """Test configuration helper functions."""

    def test_load_config_file_not_found(self):
        """Test that loading non-existent config raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_backward_compatibility_alias(self):
        """Test that load_dataset_config is available as alias."""
        from src.config import load_dataset_config

        config = load_dataset_config("replica", apply_env_overrides=False)
        assert config is not None
        assert config["coordinate_system"] == "opengl"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
