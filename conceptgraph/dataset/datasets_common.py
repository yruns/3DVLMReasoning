"""Minimal dataset utilities needed by the SLAM pipeline."""

from __future__ import annotations

import numpy as np
import torch

from conceptgraph.utils.general import to_scalar


def as_intrinsics_matrix(intrinsics: list[float] | tuple[float, ...]) -> np.ndarray:
    """Build 3x3 intrinsics matrix from (fx, fy, cx, cy)."""
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def from_intrinsics_matrix(K: torch.Tensor) -> tuple[float, float, float, float]:
    """Extract fx, fy, cx, cy from intrinsics matrix."""
    fx = to_scalar(K[0, 0])
    fy = to_scalar(K[1, 1])
    cx = to_scalar(K[0, 2])
    cy = to_scalar(K[1, 2])
    return fx, fy, cx, cy
