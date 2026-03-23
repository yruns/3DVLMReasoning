"""ScanNet data processing utilities.

This module provides tools for extracting and processing ScanNet .sens files,
including RGB-D frame extraction, camera pose export, and intrinsics export.

Based on the official ScanNet SDK:
https://github.com/ScanNet/ScanNet/tree/master/SensReader/python
"""

from .SensorData import RGBDFrame, SensorData

__all__ = ["RGBDFrame", "SensorData"]
