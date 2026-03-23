"""Dataset adapter registry with decorator-based registration.

This module provides a registry pattern for dynamically discovering and
instantiating dataset adapters. Adapters register themselves using the
@register_adapter decorator, enabling a plugin-like architecture.

Example usage:
    # Register an adapter (typically in the adapter module)
    @register_adapter("replica")
    class ReplicaAdapter(DatasetAdapter):
        ...

    # Get an adapter instance
    adapter = get_adapter("replica", data_root="/path/to/data")

    # List available adapters
    print(list_adapters())  # ["replica", "scannet", ...]
"""

from collections.abc import Callable
from pathlib import Path

from .base import DatasetAdapter

# Global registry mapping dataset names to adapter classes
_ADAPTER_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register_adapter(
    name: str,
    aliases: list[str] | None = None,
) -> Callable[[type[DatasetAdapter]], type[DatasetAdapter]]:
    """Decorator to register a dataset adapter.

    Args:
        name: Primary name for the adapter (e.g., "replica", "scannet")
        aliases: Optional list of alternative names

    Returns:
        Decorator function that registers the adapter class

    Example:
        @register_adapter("replica", aliases=["replica-v1"])
        class ReplicaAdapter(DatasetAdapter):
            ...
    """

    def decorator(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
        if not issubclass(cls, DatasetAdapter):
            raise TypeError(
                f"Registered class must be a DatasetAdapter subclass, got {cls}"
            )

        # Register primary name
        if name in _ADAPTER_REGISTRY:
            existing = _ADAPTER_REGISTRY[name]
            raise ValueError(
                f"Adapter '{name}' already registered by {existing.__name__}"
            )
        _ADAPTER_REGISTRY[name] = cls

        # Register aliases
        if aliases:
            for alias in aliases:
                if alias in _ADAPTER_REGISTRY:
                    existing = _ADAPTER_REGISTRY[alias]
                    raise ValueError(
                        f"Adapter alias '{alias}' already registered by {existing.__name__}"
                    )
                _ADAPTER_REGISTRY[alias] = cls

        return cls

    return decorator


def get_adapter(
    name: str,
    data_root: str | Path,
    **kwargs,
) -> DatasetAdapter:
    """Get an adapter instance by name.

    Args:
        name: Adapter name (case-insensitive)
        data_root: Root directory containing the dataset
        **kwargs: Additional arguments passed to the adapter constructor

    Returns:
        Instantiated DatasetAdapter

    Raises:
        ValueError: If adapter name is not found in registry
    """
    name_lower = name.lower()
    if name_lower not in _ADAPTER_REGISTRY:
        available = ", ".join(sorted(list_adapters()))
        raise ValueError(f"Unknown adapter '{name}'. Available adapters: {available}")

    adapter_cls = _ADAPTER_REGISTRY[name_lower]
    return adapter_cls(data_root=data_root, **kwargs)


def get_adapter_class(name: str) -> type[DatasetAdapter]:
    """Get an adapter class by name without instantiating.

    Args:
        name: Adapter name (case-insensitive)

    Returns:
        DatasetAdapter subclass

    Raises:
        ValueError: If adapter name is not found in registry
    """
    name_lower = name.lower()
    if name_lower not in _ADAPTER_REGISTRY:
        available = ", ".join(sorted(list_adapters()))
        raise ValueError(f"Unknown adapter '{name}'. Available adapters: {available}")

    return _ADAPTER_REGISTRY[name_lower]


def list_adapters() -> list[str]:
    """List all registered adapter names.

    Returns:
        Sorted list of registered adapter names (including aliases)
    """
    return sorted(_ADAPTER_REGISTRY.keys())


def list_primary_adapters() -> list[str]:
    """List primary adapter names (excluding aliases).

    Returns:
        List of unique adapter names (one per adapter class)
    """
    # Get unique classes and their first registered name
    seen_classes: dict[type[DatasetAdapter], str] = {}
    for name, cls in sorted(_ADAPTER_REGISTRY.items()):
        if cls not in seen_classes:
            seen_classes[cls] = name
    return sorted(seen_classes.values())


def is_registered(name: str) -> bool:
    """Check if an adapter is registered.

    Args:
        name: Adapter name to check (case-insensitive)

    Returns:
        True if adapter is registered
    """
    return name.lower() in _ADAPTER_REGISTRY


def unregister_adapter(name: str) -> bool:
    """Unregister an adapter (mainly for testing).

    Args:
        name: Adapter name to unregister

    Returns:
        True if adapter was found and removed, False otherwise
    """
    name_lower = name.lower()
    if name_lower in _ADAPTER_REGISTRY:
        del _ADAPTER_REGISTRY[name_lower]
        return True
    return False


def clear_registry() -> None:
    """Clear all registered adapters (mainly for testing)."""
    _ADAPTER_REGISTRY.clear()


class AdapterFactory:
    """Factory class for creating dataset adapters with configuration.

    Provides a more object-oriented interface for adapter creation,
    with support for default configurations and caching.

    Example:
        factory = AdapterFactory()
        factory.set_default_root("replica", "/data/replica")

        # Later, create adapter without specifying root
        adapter = factory.create("replica")
    """

    def __init__(self):
        """Initialize the factory."""
        self._default_roots: dict[str, Path] = {}
        self._default_kwargs: dict[str, dict] = {}
        self._cache: dict[str, DatasetAdapter] = {}
        self._cache_enabled = False

    def set_default_root(self, name: str, data_root: str | Path) -> None:
        """Set default data root for an adapter.

        Args:
            name: Adapter name
            data_root: Default data root path
        """
        self._default_roots[name.lower()] = Path(data_root)

    def set_default_kwargs(self, name: str, **kwargs) -> None:
        """Set default keyword arguments for an adapter.

        Args:
            name: Adapter name
            **kwargs: Default arguments
        """
        self._default_kwargs[name.lower()] = kwargs

    def enable_caching(self, enabled: bool = True) -> None:
        """Enable or disable adapter caching.

        When enabled, adapters are cached by (name, data_root) and reused.

        Args:
            enabled: Whether to enable caching
        """
        self._cache_enabled = enabled
        if not enabled:
            self._cache.clear()

    def create(
        self,
        name: str,
        data_root: str | Path | None = None,
        **kwargs,
    ) -> DatasetAdapter:
        """Create an adapter instance.

        Args:
            name: Adapter name
            data_root: Data root path (uses default if not specified)
            **kwargs: Additional arguments (merged with defaults)

        Returns:
            Instantiated DatasetAdapter

        Raises:
            ValueError: If no data_root is specified and no default is set
        """
        name_lower = name.lower()

        # Resolve data root
        if data_root is None:
            if name_lower not in self._default_roots:
                raise ValueError(
                    f"No data_root specified and no default set for '{name}'"
                )
            data_root = self._default_roots[name_lower]
        else:
            data_root = Path(data_root)

        # Check cache
        cache_key = f"{name_lower}:{data_root}"
        if self._cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        # Merge kwargs with defaults
        merged_kwargs = dict(self._default_kwargs.get(name_lower, {}))
        merged_kwargs.update(kwargs)

        # Create adapter
        adapter = get_adapter(name, data_root, **merged_kwargs)

        # Cache if enabled
        if self._cache_enabled:
            self._cache[cache_key] = adapter

        return adapter

    def clear_cache(self) -> None:
        """Clear the adapter cache."""
        self._cache.clear()


# Global factory instance for convenience
_default_factory = AdapterFactory()


def configure_default_root(name: str, data_root: str | Path) -> None:
    """Configure default data root for an adapter in the global factory.

    Args:
        name: Adapter name
        data_root: Default data root path
    """
    _default_factory.set_default_root(name, data_root)


def create_adapter(
    name: str,
    data_root: str | Path | None = None,
    **kwargs,
) -> DatasetAdapter:
    """Create an adapter using the global factory.

    Args:
        name: Adapter name
        data_root: Data root path (uses configured default if not specified)
        **kwargs: Additional arguments

    Returns:
        Instantiated DatasetAdapter
    """
    return _default_factory.create(name, data_root, **kwargs)
