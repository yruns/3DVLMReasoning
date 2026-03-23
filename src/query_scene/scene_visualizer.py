"""
Scene Visualizer: Generate annotated BEV images for multimodal LLM input.

This module re-exports from bev_builder.py for backward compatibility.
For new code, prefer importing directly from bev_builder.py.

Usage:
    from query_scene.scene_visualizer import SceneBEVGenerator, BEVConfig
    # or
    from query_scene.bev_builder import ReplicaBEVBuilder, create_bev_builder
"""

from .bev_builder import (
    AnnotatedObject,
    # Base class
    BaseBEVBuilder,
    # Configuration
    BEVConfig,
    GenericBEVBuilder,
    # Dataset-specific builders
    ReplicaBEVBuilder,
    # Alias for backward compatibility
    SceneBEVGenerator,
    # Factory
    create_bev_builder,
)

__all__ = [
    "BEVConfig",
    "AnnotatedObject",
    "BaseBEVBuilder",
    "ReplicaBEVBuilder",
    "GenericBEVBuilder",
    "SceneBEVGenerator",
    "create_bev_builder",
]
