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


def queue_pending_image(
    runtime: Any,
    image_path: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append an image to the pending queue and mark evidence updated."""
    extra = dict(runtime.bundle.extra_metadata or {})
    pending = list(extra.get("vg_pending_images") or [])
    path = str(image_path)
    pending.append(path)
    extra["vg_pending_images"] = pending
    if metadata is not None:
        metadata_rows = list(extra.get("vg_pending_image_metadata") or [])
        row = {"image_path": path, **dict(metadata)}
        metadata_rows.append(row)
        extra["vg_pending_image_metadata"] = metadata_rows
    runtime.bundle.extra_metadata = extra
    runtime.mark_evidence_updated()


def queue_pending_image_if_new(
    runtime: Any,
    path: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Queue `path` for the next evidence update only if it has not already been seen.

    Returns True if the image was queued (caller should mark `already_seen=False`),
    False if it was a no-op (caller should mark `already_seen=True`). Empty paths
    are silently skipped (returns False).
    """
    if not path:
        return False
    if path in runtime.seen_image_paths:
        return False
    pending = set((runtime.bundle.extra_metadata or {}).get("vg_pending_images") or [])
    if path in pending:
        return False
    queue_pending_image(runtime, path, metadata=metadata)
    return True


__all__ = ["get_scene_catalog", "queue_pending_image", "queue_pending_image_if_new"]
