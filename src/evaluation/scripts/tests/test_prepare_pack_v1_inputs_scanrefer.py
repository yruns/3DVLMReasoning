"""Smoke + unit tests for ScanRefer pack-v1 prep."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_parse_sample_id_canonical():
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import parse_sample_id
    scene, target_id, ann_id = parse_sample_id("scannet/scene0088_00::5::3")
    assert scene == "scene0088_00"
    assert target_id == 5
    assert ann_id == "3"


def test_parse_sample_id_rejects_malformed():
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import parse_sample_id
    with pytest.raises(ValueError, match="format"):
        parse_sample_id("scannet/scene_x::5")


def test_safe_sample_id_normalizes_separators():
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import safe_sample_id
    assert safe_sample_id("scannet/scene0088_00::5::3") == "scannet__scene0088_00__5__3"


def test_load_sample_requests_validates_per_row(tmp_path: Path):
    from evaluation.scripts.prepare_pack_v1_inputs_scanrefer import load_sample_requests
    p = tmp_path / "ids.json"
    p.write_text(json.dumps([
        {"sample_id": "scannet/scene_a::0::0", "scene_id": "scene_a",
         "target_id": 0, "ann_id": "0", "category": "chair"},
    ]))
    reqs = load_sample_requests(p)
    assert len(reqs) == 1
    assert reqs[0].scene_id == "scene_a"
    assert reqs[0].target_id == 0
    assert reqs[0].ann_id == "0"
