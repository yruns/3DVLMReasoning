"""Tests for detector proposal-pool recall evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_synthetic_perfect_pool_has_full_recall(tmp_path: Path) -> None:
    from evaluation.scripts.eval_proposal_pool_recall import eval_proposal_pool_recall

    sample_ids = _write_pack(
        tmp_path,
        scene_id="scene0001_00",
        target_id=7,
        gt_bbox=_unit_bbox(),
        proposals=[_proposal(0, _unit_bbox(), "chair")],
        visibility={"10": [0]},
        keyframes=[10],
    )

    result = eval_proposal_pool_recall(
        data_root=tmp_path,
        pack_name="pack_test",
        sample_ids_path=sample_ids,
        output_json=tmp_path / "out.json",
        use_keyframe_pool=True,
    )

    aggregate = result["aggregate"]
    sample = result["samples"][0]
    assert aggregate["recall_at_0.25"] == 1.0
    assert aggregate["recall_at_0.50"] == 1.0
    assert aggregate["mean_max_iou_3d"] == 1.0
    assert sample["max_iou_3d"] == 1.0
    assert sample["best_proposal_id"] == 0


def test_synthetic_empty_pool_has_zero_recall(tmp_path: Path) -> None:
    from evaluation.scripts.eval_proposal_pool_recall import eval_proposal_pool_recall

    sample_ids = _write_pack(
        tmp_path,
        scene_id="scene0001_00",
        target_id=7,
        gt_bbox=_unit_bbox(),
        proposals=[],
        visibility={"10": []},
        keyframes=[10],
    )

    result = eval_proposal_pool_recall(
        data_root=tmp_path,
        pack_name="pack_test",
        sample_ids_path=sample_ids,
        output_json=tmp_path / "out.json",
        use_keyframe_pool=True,
    )

    aggregate = result["aggregate"]
    sample = result["samples"][0]
    assert aggregate["recall_at_0.25"] == 0.0
    assert aggregate["recall_at_0.50"] == 0.0
    assert aggregate["mean_max_iou_3d"] == 0.0
    assert sample["max_iou_3d"] == 0.0
    assert sample["best_proposal_id"] is None


def test_keyframe_filter_excludes_proposals_from_other_frames(
    tmp_path: Path,
) -> None:
    from evaluation.scripts.eval_proposal_pool_recall import eval_proposal_pool_recall

    sample_ids = _write_pack(
        tmp_path,
        scene_id="scene0001_00",
        target_id=7,
        gt_bbox=_unit_bbox(),
        proposals=[_proposal(0, _unit_bbox(), "chair")],
        visibility={"10": [], "20": [0]},
        keyframes=[10],
    )

    result = eval_proposal_pool_recall(
        data_root=tmp_path,
        pack_name="pack_test",
        sample_ids_path=sample_ids,
        output_json=tmp_path / "out.json",
        use_keyframe_pool=True,
    )

    sample = result["samples"][0]
    assert sample["n_proposals_in_pool"] == 0
    assert sample["hit_at_0.25"] is False
    assert sample["hit_at_0.50"] is False


def test_no_keyframe_filter_counts_scene_proposals(tmp_path: Path) -> None:
    from evaluation.scripts.eval_proposal_pool_recall import eval_proposal_pool_recall

    sample_ids = _write_pack(
        tmp_path,
        scene_id="scene0001_00",
        target_id=7,
        gt_bbox=_unit_bbox(),
        proposals=[_proposal(0, _unit_bbox(), "chair")],
        visibility={"10": [], "20": [0]},
        keyframes=[10],
    )

    result = eval_proposal_pool_recall(
        data_root=tmp_path,
        pack_name="pack_test",
        sample_ids_path=sample_ids,
        output_json=tmp_path / "out.json",
        use_keyframe_pool=False,
    )

    sample = result["samples"][0]
    assert sample["n_proposals_in_pool"] == 1
    assert sample["hit_at_0.25"] is True
    assert sample["hit_at_0.50"] is True


def test_line_delimited_proposal_file_is_supported(tmp_path: Path) -> None:
    from evaluation.scripts.eval_proposal_pool_recall import eval_proposal_pool_recall

    sample_ids = _write_pack(
        tmp_path,
        scene_id="scene0001_00",
        target_id=7,
        gt_bbox=_unit_bbox(),
        proposals=[_proposal(0, _unit_bbox(), "chair")],
        visibility={"10": [0]},
        keyframes=[10],
        proposals_as_jsonl=True,
    )

    result = eval_proposal_pool_recall(
        data_root=tmp_path,
        pack_name="pack_test",
        sample_ids_path=sample_ids,
        output_json=tmp_path / "out.json",
        use_keyframe_pool=True,
    )

    assert result["aggregate"]["recall_at_0.50"] == 1.0


def test_missing_pack_dir_raises(tmp_path: Path) -> None:
    from evaluation.scripts.eval_proposal_pool_recall import eval_proposal_pool_recall

    sample_ids = tmp_path / "sample_ids.json"
    sample_ids.write_text(json.dumps(["scene0001_00::7"]), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="pack directory not found"):
        eval_proposal_pool_recall(
            data_root=tmp_path,
            pack_name="pack_missing",
            sample_ids_path=sample_ids,
            output_json=tmp_path / "out.json",
            use_keyframe_pool=True,
        )


def _write_pack(
    root: Path,
    *,
    scene_id: str,
    target_id: int,
    gt_bbox: list[float],
    proposals: list[dict],
    visibility: dict[str, list[int]],
    keyframes: list[int],
    proposals_as_jsonl: bool = False,
) -> Path:
    pack_dir = root / scene_id / "pack_test"
    samples_dir = pack_dir / "samples"
    samples_dir.mkdir(parents=True)
    sample_id = f"{scene_id}::{target_id}"
    (samples_dir / f"{target_id}.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "target_id": target_id,
                "category": "chair",
                "gt_bbox_3d_9dof": gt_bbox,
                "keyframes": [
                    {
                        "keyframe_idx": idx,
                        "image_path": f"/tmp/frame_{frame_id}.png",
                        "frame_id": frame_id,
                    }
                    for idx, frame_id in enumerate(keyframes)
                ],
            }
        ),
        encoding="utf-8",
    )
    if proposals_as_jsonl:
        (pack_dir / "proposals.jsonl").write_text(
            "\n".join(json.dumps(proposal) for proposal in proposals),
            encoding="utf-8",
        )
    else:
        (pack_dir / "proposals.jsonl").write_text(
            json.dumps(
                {"source": "test", "scene_id": scene_id, "proposals": proposals}
            ),
            encoding="utf-8",
        )
    (pack_dir / "visibility.json").write_text(json.dumps(visibility), encoding="utf-8")
    sample_ids = root / "sample_ids.json"
    sample_ids.write_text(json.dumps([sample_id]), encoding="utf-8")
    return sample_ids


def _proposal(proposal_id: int, bbox: list[float], label: str) -> dict:
    return {
        "id": proposal_id,
        "bbox_3d": bbox,
        "score": 0.99,
        "label": label,
        "metadata": {"class_id": 2},
    }


def _unit_bbox() -> list[float]:
    return [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
