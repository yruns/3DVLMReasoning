from benchmarks.embodiedscan_bbox_feasibility.targets import (
    deduplicate_targets,
    load_targets,
)
from benchmarks.embodiedscan_loader import EmbodiedScanDataset, EmbodiedScanVGSample


def _sample(sample_id: str, target_id: int, query: str) -> EmbodiedScanVGSample:
    return EmbodiedScanVGSample(
        sample_id=sample_id,
        scene_id="scene0001_00",
        query=query,
        scan_id="scannet/scene0001_00",
        target_id=target_id,
        target="chair",
        gt_bbox_3d=[1, 2, 3, 4, 5, 6, 0, 0, 0],
    )


def test_deduplicate_targets_merges_repeated_referring_expressions() -> None:
    targets = deduplicate_targets(
        [
            _sample("a", 7, "the chair"),
            _sample("b", 7, "the wooden chair"),
            _sample("c", 8, "the table"),
        ]
    )
    assert len(targets) == 2
    assert targets[0].target_id == 7
    assert targets[0].sample_ids == ["a", "b"]
    assert targets[1].target_id == 8


def test_deduplicate_targets_skips_samples_without_gt_bbox() -> None:
    sample = _sample("a", 7, "the chair")
    sample.gt_bbox_3d = None
    assert deduplicate_targets([sample]) == []


def test_load_targets_carries_axis_align_matrix(monkeypatch) -> None:
    class FakeDataset:
        def __iter__(self):
            return iter([_sample("a", 7, "the chair")])

        def get_scene_info(self, scan_id: str) -> dict:
            assert scan_id == "scannet/scene0001_00"
            return {
                "axis_align_matrix": [
                    [1, 0, 0, 10],
                    [0, 1, 0, 20],
                    [0, 0, 1, 30],
                    [0, 0, 0, 1],
                ],
                "images": [
                    {"visible_instance_ids": [7]},
                    {"visible_instance_ids": []},
                    {"visible_instance_ids": [7, 8]},
                ],
            }

    monkeypatch.setattr(
        EmbodiedScanDataset,
        "from_path",
        classmethod(lambda cls, *args, **kwargs: FakeDataset()),
    )

    targets = load_targets("unused")

    assert targets[0].axis_align_matrix == [
        [1.0, 0.0, 0.0, 10.0],
        [0.0, 1.0, 0.0, 20.0],
        [0.0, 0.0, 1.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert targets[0].visible_frame_ids == [0, 2]
