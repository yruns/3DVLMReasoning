"""Tests for merging detector proposal packs."""

from __future__ import annotations

import json
from pathlib import Path


def test_union_recall_is_at_least_each_source_pack(tmp_path: Path) -> None:
    from evaluation.scripts.eval_proposal_pool_recall import eval_proposal_pool_recall
    from evaluation.scripts.merge_proposal_packs import merge_proposal_packs

    sample_ids = _write_two_source_pack_fixture(
        tmp_path,
        pack_a_proposals=[
            _proposal(10, _shifted_bbox(4.0), 0.8, "miss-from-a"),
        ],
        pack_b_proposals=[
            _proposal(20, _unit_bbox(), 0.7, "hit-from-b"),
        ],
        visibility_a={"1": [10]},
        visibility_b={"1": [20]},
    )

    source_a = eval_proposal_pool_recall(
        data_root=tmp_path,
        pack_name="pack_a",
        sample_ids_path=sample_ids,
        output_json=tmp_path / "a.json",
        use_keyframe_pool=True,
    )
    source_b = eval_proposal_pool_recall(
        data_root=tmp_path,
        pack_name="pack_b",
        sample_ids_path=sample_ids,
        output_json=tmp_path / "b.json",
        use_keyframe_pool=True,
    )
    merge_proposal_packs(
        data_root=tmp_path,
        source_packs=["pack_a", "pack_b"],
        output_pack="pack_union",
        sample_ids_path=sample_ids,
        max_proposals_per_scene=2000,
    )
    merged = eval_proposal_pool_recall(
        data_root=tmp_path,
        pack_name="pack_union",
        sample_ids_path=sample_ids,
        output_json=tmp_path / "union.json",
        use_keyframe_pool=True,
    )

    merged_recall = merged["aggregate"]["recall_at_0.50"]
    assert merged["samples"][0]["n_proposals_in_pool"] == 2
    assert merged_recall >= source_a["aggregate"]["recall_at_0.50"]
    assert merged_recall >= source_b["aggregate"]["recall_at_0.50"]


def test_stable_ids_are_sorted_by_score_source_pool_and_original_id(
    tmp_path: Path,
) -> None:
    from evaluation.scripts.merge_proposal_packs import merge_proposal_packs

    sample_ids = _write_two_source_pack_fixture(
        tmp_path,
        pack_a_proposals=[
            _proposal(8, _unit_bbox(), 0.9, "a-high"),
            _proposal(2, _shifted_bbox(2.0), 0.3, "a-low"),
        ],
        pack_b_proposals=[
            _proposal(7, _shifted_bbox(3.0), 0.9, "b-high"),
        ],
        visibility_a={"1": [8, 2]},
        visibility_b={"1": [7]},
    )

    merge_proposal_packs(
        data_root=tmp_path,
        source_packs=["pack_a", "pack_b"],
        output_pack="pack_union",
        sample_ids_path=sample_ids,
        max_proposals_per_scene=2000,
    )
    first = _merged_proposals(tmp_path)
    merge_proposal_packs(
        data_root=tmp_path,
        source_packs=["pack_a", "pack_b"],
        output_pack="pack_union",
        sample_ids_path=sample_ids,
        max_proposals_per_scene=2000,
    )
    second = _merged_proposals(tmp_path)

    assert [
        (
            proposal["id"],
            proposal["metadata"]["source_pool"],
            proposal["metadata"]["original_pool_id"],
        )
        for proposal in first
    ] == [(0, "a", 8), (1, "b", 7), (2, "a", 2)]
    assert first == second


def test_source_pool_tracking_is_added_to_each_proposal(tmp_path: Path) -> None:
    from evaluation.scripts.merge_proposal_packs import merge_proposal_packs

    sample_ids = _write_two_source_pack_fixture(
        tmp_path,
        pack_a_proposals=[_proposal(10, _unit_bbox(), 0.8, "from-a")],
        pack_b_proposals=[_proposal(20, _shifted_bbox(2.0), 0.7, "from-b")],
        visibility_a={"1": [10]},
        visibility_b={"1": [20]},
    )

    merge_proposal_packs(
        data_root=tmp_path,
        source_packs=["pack_a", "pack_b"],
        output_pack="pack_union",
        sample_ids_path=sample_ids,
        max_proposals_per_scene=2000,
    )

    metadata = [proposal["metadata"] for proposal in _merged_proposals(tmp_path)]
    assert metadata == [
        {"existing": "from-a", "source_pool": "a", "original_pool_id": 10},
        {"existing": "from-b", "source_pool": "b", "original_pool_id": 20},
    ]


def test_visibility_union_rewrites_source_ids_to_merged_ids(tmp_path: Path) -> None:
    from evaluation.scripts.merge_proposal_packs import merge_proposal_packs

    sample_ids = _write_two_source_pack_fixture(
        tmp_path,
        pack_a_proposals=[_proposal(10, _unit_bbox(), 0.8, "from-a")],
        pack_b_proposals=[_proposal(20, _shifted_bbox(2.0), 0.7, "from-b")],
        visibility_a={"1": [10], "2": []},
        visibility_b={"1": [], "2": [20]},
    )

    merge_proposal_packs(
        data_root=tmp_path,
        source_packs=["pack_a", "pack_b"],
        output_pack="pack_union",
        sample_ids_path=sample_ids,
        max_proposals_per_scene=2000,
    )

    visibility = json.loads(
        (tmp_path / "scene0001_00" / "pack_union" / "visibility.json").read_text(
            encoding="utf-8"
        )
    )
    assert visibility == {"1": [0], "2": [1]}


def test_merged_pack_satisfies_vg_proposal_pool_schema(tmp_path: Path) -> None:
    from agents.packs.vg_embodiedscan.proposal_pool import build_vg_proposal_pool
    from evaluation.scripts.merge_proposal_packs import merge_proposal_packs

    sample_ids = _write_two_source_pack_fixture(
        tmp_path,
        pack_a_proposals=[_proposal(10, _unit_bbox(), 0.8, "from-a")],
        pack_b_proposals=[_proposal(20, _shifted_bbox(2.0), 0.7, "from-b")],
        visibility_a={"1": [10]},
        visibility_b={"1": [20]},
    )

    merge_proposal_packs(
        data_root=tmp_path,
        source_packs=["pack_a", "pack_b"],
        output_pack="pack_union",
        sample_ids_path=sample_ids,
        max_proposals_per_scene=2000,
    )

    pack_dir = tmp_path / "scene0001_00" / "pack_union"
    pool = build_vg_proposal_pool(
        proposals_jsonl=pack_dir / "proposals.jsonl",
        source="vdetr",
        annotated_image_dir=pack_dir / "annotated",
        frame_visibility={1: [0, 1]},
        axis_align_matrix=None,
    )
    assert [proposal["id"] for proposal in pool["proposals"]] == [0, 1]


def _write_two_source_pack_fixture(
    root: Path,
    *,
    pack_a_proposals: list[dict],
    pack_b_proposals: list[dict],
    visibility_a: dict[str, list[int]],
    visibility_b: dict[str, list[int]],
) -> Path:
    scene_id = "scene0001_00"
    target_id = 7
    sample_id = f"{scene_id}::{target_id}"
    for pack_name, proposals, visibility in [
        ("pack_a", pack_a_proposals, visibility_a),
        ("pack_b", pack_b_proposals, visibility_b),
    ]:
        pack_dir = root / scene_id / pack_name
        (pack_dir / "samples").mkdir(parents=True)
        (pack_dir / "proposals.jsonl").write_text(
            json.dumps(
                {
                    "source": pack_name,
                    "scene_id": scene_id,
                    "proposals": proposals,
                }
            ),
            encoding="utf-8",
        )
        (pack_dir / "visibility.json").write_text(
            json.dumps(visibility), encoding="utf-8"
        )
        (pack_dir / "samples" / f"{target_id}.json").write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "scene_id": scene_id,
                    "target_id": target_id,
                    "category": "chair",
                    "query": "find the chair",
                    "gt_bbox_3d_9dof": _unit_bbox(),
                    "source": pack_name,
                    "keyframes": [
                        {
                            "keyframe_idx": 0,
                            "image_path": f"/tmp/{pack_name}_frame_1.png",
                            "frame_id": 1,
                        }
                    ],
                    "proposals": proposals,
                }
            ),
            encoding="utf-8",
        )
    sample_ids = root / "sample_ids.json"
    sample_ids.write_text(json.dumps([sample_id]), encoding="utf-8")
    return sample_ids


def _merged_proposals(root: Path) -> list[dict]:
    payload = json.loads(
        (root / "scene0001_00" / "pack_union" / "proposals.jsonl").read_text(
            encoding="utf-8"
        )
    )
    return payload["proposals"]


def _proposal(proposal_id: int, bbox: list[float], score: float, label: str) -> dict:
    return {
        "id": proposal_id,
        "bbox_3d": bbox,
        "score": score,
        "label": label,
        "source": "fixture",
        "metadata": {"existing": label},
    }


def _unit_bbox() -> list[float]:
    return [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def _shifted_bbox(x: float) -> list[float]:
    return [x, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
