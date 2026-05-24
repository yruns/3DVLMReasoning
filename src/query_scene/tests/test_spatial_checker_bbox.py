from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from query_scene.retrieval.spatial_checker import SpatialRelationChecker


def _box(
    min_xyz: tuple[float, float, float],
    max_xyz: tuple[float, float, float],
) -> np.ndarray:
    x0, y0, z0 = min_xyz
    x1, y1, z1 = max_xyz
    return np.array(
        [
            [x0, y0, z0],
            [x0, y0, z1],
            [x0, y1, z0],
            [x0, y1, z1],
            [x1, y0, z0],
            [x1, y0, z1],
            [x1, y1, z0],
            [x1, y1, z1],
        ],
        dtype=np.float32,
    )


def _obj(
    obj_id: int,
    category: str,
    min_xyz: tuple[float, float, float],
    max_xyz: tuple[float, float, float],
) -> SimpleNamespace:
    bbox = _box(min_xyz, max_xyz)
    return SimpleNamespace(
        obj_id=obj_id,
        category=category,
        object_tag=category,
        centroid=bbox.mean(axis=0),
        bbox_np=bbox,
        bbox_3d=None,
    )


class TestBBoxAwareSpatialChecker(unittest.TestCase):
    def test_inside_uses_bbox_np_anchor(self) -> None:
        checker = SpatialRelationChecker()
        picture = _obj(1, "picture", (1.0, 1.0, 1.0), (1.5, 1.5, 1.5))
        frame = _obj(2, "frame", (0.5, 0.5, 0.5), (2.0, 2.0, 2.0))

        result = checker.check(picture, frame, "inside")

        self.assertTrue(result.satisfies)
        self.assertGreater(result.score, 0.0)
        self.assertTrue(result.details["bbox_used"])

    def test_on_top_of_uses_anchor_footprint_not_centroid_radius(self) -> None:
        checker = SpatialRelationChecker()
        pillow = _obj(1, "pillow", (2.6, 0.2, 0.95), (3.1, 0.7, 1.2))
        bed = _obj(2, "bed", (0.0, 0.0, 0.0), (4.0, 2.0, 1.0))

        result = checker.check(pillow, bed, "on")

        self.assertTrue(result.satisfies)
        self.assertGreater(result.score, 0.0)
        self.assertEqual(result.details["mode"], "bbox_horizontal_support")

    def test_next_to_uses_bbox_gap_not_centroid_distance(self) -> None:
        checker = SpatialRelationChecker()
        window = _obj(1, "window", (2.1, 0.2, 1.2), (2.4, 1.2, 2.5))
        bed = _obj(2, "bed", (0.0, 0.0, 0.0), (2.0, 1.8, 0.7))

        result = checker.check(window, bed, "next_to")

        self.assertTrue(result.satisfies)
        self.assertGreater(result.score, 0.0)
        self.assertEqual(result.details["distance_mode"], "bbox_xy_gap")

    def test_between_searches_all_anchor_pairs(self) -> None:
        checker = SpatialRelationChecker()
        lamp = _obj(1, "lamp", (4.9, 0.0, 0.0), (5.1, 0.2, 0.6))
        wrong_anchor = _obj(2, "bed", (0.0, 5.0, 0.0), (0.5, 5.5, 0.5))
        bed_left = _obj(3, "bed", (0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
        bed_right = _obj(4, "bed", (10.0, 0.0, 0.0), (10.5, 0.5, 0.5))

        result = checker.check(lamp, [wrong_anchor, bed_left, bed_right], "between")

        self.assertTrue(result.satisfies)
        self.assertGreater(result.score, 0.0)
        self.assertEqual(result.details["anchor_pair"], (3, 4))
