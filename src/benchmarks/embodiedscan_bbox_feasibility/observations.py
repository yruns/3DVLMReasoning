from __future__ import annotations

from pathlib import Path

from .models import EmbodiedScanTarget, ObservationRecord


def centered_frame_window(center: int, available: list[int], size: int) -> list[int]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not available:
        return []

    ordered = sorted(int(frame_id) for frame_id in available)
    if center in ordered:
        center_idx = ordered.index(center)
    else:
        center_idx = min(range(len(ordered)), key=lambda idx: abs(ordered[idx] - center))

    half = size // 2
    start = max(0, center_idx - half)
    end = min(len(ordered), start + size)
    start = max(0, end - size)
    return ordered[start:end]


def make_observation(
    target: EmbodiedScanTarget,
    *,
    best_frame_id: int,
    available_frame_ids: list[int],
    window_size: int,
) -> ObservationRecord:
    return ObservationRecord(
        policy="target_best_visible_centered_window",
        frame_ids=centered_frame_window(best_frame_id, available_frame_ids, window_size),
        metadata={
            "scan_id": target.scan_id,
            "scene_id": target.scene_id,
            "target_id": target.target_id,
            "best_frame_id": int(best_frame_id),
            "window_size": int(window_size),
        },
    )


def list_raw_frame_ids(scene_root: str | Path) -> list[int]:
    raw = Path(scene_root) / "raw"
    frame_ids: list[int] = []
    for path in sorted(raw.glob("*-rgb.*")):
        stem = path.name.split("-rgb", 1)[0]
        if stem.isdigit():
            frame_ids.append(int(stem))
    return sorted(frame_ids)
