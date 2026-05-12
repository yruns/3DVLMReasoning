from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from query_scene.query_executor import QueryExecutor


def test_unknown_category_sentinel_does_not_call_clip_encoder() -> None:
    objects = [
        SimpleNamespace(
            obj_id=0,
            category="sofa",
            object_tag="sofa",
            class_name=[],
        )
    ]

    def fail_if_called(_text: str) -> np.ndarray:
        raise AssertionError("UNKNOW sentinel should not trigger CLIP fallback")

    executor = QueryExecutor(
        objects=objects,
        clip_features=np.ones((1, 3), dtype=np.float32),
        clip_encoder=fail_if_called,
    )

    assert executor._find_by_categories(["UNKNOW"]) == []


def test_unknown_category_sentinel_is_filtered_when_other_categories_exist() -> None:
    sofa = SimpleNamespace(
        obj_id=0,
        category="sofa",
        object_tag="sofa",
        class_name=[],
    )
    executor = QueryExecutor(objects=[sofa])

    assert executor._find_by_categories(["UNKNOW", "sofa"]) == [sofa]
