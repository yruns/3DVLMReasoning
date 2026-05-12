import json
from pathlib import Path

import pytest

from evaluation.scripts.scanrefer_consensus_side_by_side import (
    build_consensus_payload,
    build_consensus_payload_from_paths,
    choose_consensus_record,
)


def _record(sample_id: str, pid: int, confidence: float, iou: float) -> dict:
    return {
        "sample_id": sample_id,
        "backend": "pack_v1",
        "status": "completed",
        "selected_object_id": pid,
        "confidence": confidence,
        "iou": iou,
        "query": "find the chair",
        "tool_trace": [],
    }


def test_plurality_beats_single_high_confidence_and_iou() -> None:
    chosen = choose_consensus_record(
        "scene::1::0",
        [
            ("a", _record("scene::1::0", 1, 0.60, 0.10)),
            ("b", _record("scene::1::0", 1, 0.70, 0.10)),
            ("c", _record("scene::1::0", 2, 0.99, 0.99)),
        ],
    )

    assert chosen["selected_object_id"] == 1
    assert chosen["consensus_choice_source"] == "b"
    assert chosen["consensus_votes"] == [
        {
            "source": "a",
            "selected_object_id": 1,
            "confidence": 0.60,
            "status": "completed",
        },
        {
            "source": "b",
            "selected_object_id": 1,
            "confidence": 0.70,
            "status": "completed",
        },
        {
            "source": "c",
            "selected_object_id": 2,
            "confidence": 0.99,
            "status": "completed",
        },
    ]


def test_confidence_breaks_plurality_ties() -> None:
    chosen = choose_consensus_record(
        "scene::1::0",
        [
            ("a", _record("scene::1::0", 1, 0.60, 0.99)),
            ("b", _record("scene::1::0", 2, 0.80, 0.00)),
        ],
    )

    assert chosen["selected_object_id"] == 2
    assert chosen["consensus_choice_source"] == "b"


def test_consensus_record_uses_synthetic_trace_instead_of_copying_source_trace() -> None:
    source_record = _record("scene::1::0", 1, 0.60, 0.10)
    source_record["tool_trace"] = [
        {"tool_name": "load_skill", "tool_input": {}, "response_text": "large trace"}
    ]

    chosen = choose_consensus_record(
        "scene::1::0",
        [
            ("a", source_record),
            ("b", _record("scene::1::0", 1, 0.70, 0.10)),
        ],
    )

    assert chosen["tool_trace"] == [
        {
            "tool_name": "consensus_select",
            "tool_input": {
                "policy": "plurality_avg_confidence",
                "choice_source": "b",
                "selected_object_id": 1,
                "group_size": 2,
                "n_sources": 2,
                "votes": chosen["consensus_votes"],
            },
            "response_text": (
                "consensus selected proposal_id=1 from source=b "
                "with group_size=2/2"
            ),
        }
    ]


def test_build_consensus_payload_requires_matching_sample_ids() -> None:
    payloads = {
        "a": {"pack_v1": {"per_sample": [_record("scene::1::0", 1, 0.6, 0.1)]}},
        "b": {"pack_v1": {"per_sample": [_record("scene::2::0", 1, 0.7, 0.2)]}},
    }

    with pytest.raises(ValueError, match="sample_id sets differ"):
        build_consensus_payload(payloads, backend="pack_v1")


def test_build_consensus_payload_recomputes_metrics_from_chosen_records() -> None:
    payloads = {
        "a": {
            "pack_v1": {
                "per_sample": [
                    _record("scene::1::0", 1, 0.6, 0.1),
                    _record("scene::2::0", 4, 0.9, 0.8),
                ]
            }
        },
        "b": {
            "pack_v1": {
                "per_sample": [
                    _record("scene::1::0", 1, 0.7, 0.1),
                    _record("scene::2::0", 5, 0.8, 0.0),
                ]
            }
        },
    }

    payload = build_consensus_payload(payloads, backend="pack_v1")

    assert payload["pack_v1"]["n"] == 2
    assert payload["pack_v1"]["mean_iou"] == pytest.approx(0.45)
    assert payload["pack_v1"]["Acc@0.25"] == pytest.approx(0.5)
    assert payload["pack_v1"]["Acc@0.50"] == pytest.approx(0.5)
    assert [r["selected_object_id"] for r in payload["pack_v1"]["per_sample"]] == [1, 4]


def test_build_consensus_payload_from_paths_streams_without_read_text(
    tmp_path,
    monkeypatch,
) -> None:
    paths: dict[str, Path] = {}
    for source, pid, confidence in [
        ("a", 1, 0.60),
        ("b", 1, 0.70),
        ("c", 2, 0.99),
    ]:
        path = tmp_path / f"{source}.json"
        record = _record("scene::1::0", pid, confidence, iou=float(pid) / 10.0)
        record["tool_trace"] = [{"response_text": "x" * 1000}]
        path.write_text(
            json.dumps({"pack_v1": {"per_sample": [record]}}, indent=2),
            encoding="utf-8",
        )
        paths[source] = path

    original_read_text = Path.read_text

    def guarded_read_text(self: Path, *args, **kwargs):
        if self in set(paths.values()):
            raise AssertionError("consensus inputs must be streamed")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    payload = build_consensus_payload_from_paths(paths, backend="pack_v1")
    chosen = payload["pack_v1"]["per_sample"][0]
    assert chosen["selected_object_id"] == 1
    assert chosen["consensus_choice_source"] == "b"
    assert chosen["tool_trace"][0]["tool_name"] == "consensus_select"
