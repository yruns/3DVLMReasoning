"""Runtime helpers for v9 catalog-first scene exploration tools.

Hydrates SceneCatalog on demand from bundle.extra_metadata, and exposes
queue_pending_image() so every image-injecting tool uses a single code path.
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


def queue_pending_image(runtime: Any, image_path: str) -> None:
    """Append an image to bundle.extra_metadata['vg_pending_images'] and mark evidence updated."""
    extra = dict(runtime.bundle.extra_metadata or {})
    pending = list(extra.get("vg_pending_images") or [])
    pending.append(str(image_path))
    extra["vg_pending_images"] = pending
    runtime.bundle.extra_metadata = extra
    runtime.mark_evidence_updated()


__all__ = ["get_scene_catalog", "queue_pending_image"]
