from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

from benchmarks.embodiedscan_loader import EmbodiedScanDataset, EmbodiedScanVGSample

from .models import EmbodiedScanTarget


def load_targets(
    data_root: str,
    *,
    split: str = "val",
    source_filter: str | None = "scannet",
    max_samples: int | None = None,
    mini: bool = False,
) -> list[EmbodiedScanTarget]:
    dataset = EmbodiedScanDataset.from_path(
        data_root,
        split=split,
        source_filter=source_filter,
        max_samples=max_samples,
        mini=mini,
    )
    return deduplicate_targets(dataset)


def deduplicate_targets(
    samples: Iterable[EmbodiedScanVGSample],
) -> list[EmbodiedScanTarget]:
    by_key: OrderedDict[tuple[str, int], EmbodiedScanTarget] = OrderedDict()
    for sample in samples:
        if sample.gt_bbox_3d is None:
            continue
        key = (sample.scan_id, sample.target_id)
        if key not in by_key:
            by_key[key] = EmbodiedScanTarget(
                sample_ids=[sample.sample_id],
                scan_id=sample.scan_id,
                scene_id=sample.scene_id,
                target_id=sample.target_id,
                target_category=sample.target,
                gt_bbox_3d=sample.gt_bbox_3d,
            )
        else:
            by_key[key].sample_ids.append(sample.sample_id)
    return list(by_key.values())
