"""Configuration loading utilities."""

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dictionary with configuration values

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def _apply_env_overrides(config: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    """Apply environment variable overrides to dataset configuration.

    Environment variables follow the pattern: DATASET_{NAME}_{KEY}
    For nested keys, use double underscores: DATASET_{NAME}_{KEY1}__{KEY2}

    Examples:
        DATASET_REPLICA_STRIDE=10  -> config["stride"] = 10
        DATASET_SCANNET_CAMERA__WIDTH=1920  -> config["camera"]["width"] = 1920
        DATASET_REPLICA_DEPTH__SCALE=1000.0  -> config["depth"]["scale"] = 1000.0

    Args:
        config: Configuration dictionary to modify
        dataset_name: Name of the dataset (e.g., "replica")

    Returns:
        Configuration dictionary with environment overrides applied
    """
    prefix = f"DATASET_{dataset_name.upper()}_"

    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue

        # Extract config key path
        key_path = env_key[len(prefix) :].lower()
        keys = key_path.split("__")

        # Navigate to nested dict
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set value with type inference
        final_key = keys[-1]
        try:
            # Try int first
            if env_value.isdigit() or (env_value.startswith("-") and env_value[1:].isdigit()):
                current[final_key] = int(env_value)
            # Try float
            elif "." in env_value:
                current[final_key] = float(env_value)
            # Try boolean
            elif env_value.lower() in ("true", "false"):
                current[final_key] = env_value.lower() == "true"
            # Try null
            elif env_value.lower() in ("null", "none"):
                current[final_key] = None
            # String
            else:
                current[final_key] = env_value
        except (ValueError, AttributeError):
            # Fallback to string
            current[final_key] = env_value

    return config


def get_datasets_config() -> dict[str, Any]:
    """Load the default datasets configuration.

    Returns:
        Dictionary with dataset configurations
    """
    config_path = Path(__file__).parent / "datasets.yaml"
    return load_config(config_path)


def get_dataset_config(
    dataset_name: str, apply_env_overrides: bool = True
) -> dict[str, Any] | None:
    """Get configuration for a specific dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "replica", "scannet")
        apply_env_overrides: Whether to apply environment variable overrides

    Returns:
        Dataset configuration dictionary or None if not found

    Note:
        Environment variable overrides follow the pattern: DATASET_{NAME}_{KEY}
        For nested keys, use double underscores: DATASET_{NAME}_{KEY1}__{KEY2}
    """
    config = get_datasets_config()
    dataset_config = config.get(dataset_name.lower())

    if dataset_config is None:
        return None

    # Apply default values
    defaults = config.get("defaults", {})
    for key, value in defaults.items():
        if key not in dataset_config:
            dataset_config[key] = value

    # Apply environment variable overrides
    if apply_env_overrides:
        dataset_config = _apply_env_overrides(dataset_config, dataset_name)

    return dataset_config


# Alias for backward compatibility
load_dataset_config = get_dataset_config

__all__ = [
    "load_config",
    "get_datasets_config",
    "get_dataset_config",
    "load_dataset_config",
]
