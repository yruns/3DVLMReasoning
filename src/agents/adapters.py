"""Backward-compatible Stage-1 adapter exports.

This module preserves the old ``agents.adapters`` import path used in the
source repository. The canonical implementation now lives in
``agents.stage1_adapters``.
"""

from __future__ import annotations

from .stage1_adapters import build_object_context, build_stage2_evidence_bundle

__all__ = ['build_object_context', 'build_stage2_evidence_bundle']
