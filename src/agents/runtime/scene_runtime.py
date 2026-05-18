"""Runtime helpers for v9 catalog-first scene exploration tools.

Hydrates SceneCatalog on demand from bundle.extra_metadata, and exposes helpers
that create trace-local image metadata for images produced by active tools.
"""

from __future__ import annotations

from typing import Any

from agents.catalog import SceneCatalog

_RUNTIME_ATTR = "_v9_scene_catalog"


def get_scene_catalog(runtime: Any) -> SceneCatalog:
    """Lazy-load SceneCatalog from bundle.extra_metadata["scene_catalog"].

    Subsequent calls on the same runtime return the cached instance.
    Raises ValueError when the catalog is not present (fail-loud per
    the no-fallback rule).
    """
    cached = getattr(runtime, _RUNTIME_ATTR, None)
    if cached is not None:
        return cached
    extra = getattr(runtime.bundle, "extra_metadata", None) or {}
    raw = extra.get("scene_catalog")
    if raw is None:
        raise ValueError(
            "bundle.extra_metadata.scene_catalog is missing; v9 pack-prep "
            "must write scene_catalog before invoking scene-perception tools"
        )
    catalog = SceneCatalog(**raw)
    setattr(runtime, _RUNTIME_ATTR, catalog)
    return catalog


def make_tool_image_ref(
    runtime: Any,
    image_path: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return trace metadata for a tool-produced image."""
    path = str(image_path)
    row = {"image_path": path}
    if metadata is not None:
        row.update(dict(metadata))
    return row


def make_tool_image_ref_if_new(
    runtime: Any,
    path: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return trace metadata only if `path` has not already been seen."""
    if not path:
        return None
    if path in runtime.seen_image_paths:
        return None
    return make_tool_image_ref(runtime, path, metadata=metadata)


__all__ = ["get_scene_catalog", "make_tool_image_ref", "make_tool_image_ref_if_new"]
