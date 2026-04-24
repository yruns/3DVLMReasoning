from benchmarks.embodiedscan_bbox_feasibility.targets import deduplicate_targets
from benchmarks.embodiedscan_loader import EmbodiedScanVGSample


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
