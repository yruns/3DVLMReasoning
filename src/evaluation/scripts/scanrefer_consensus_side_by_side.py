"""Build a no-GT consensus ScanRefer side_by_side.json from prior runs.

The selector only sees inference outputs: sample id, selected proposal id,
status, and self-reported confidence. IoU / GT fields are copied from the
chosen record after selection so the standard metric scripts can score the
result, but they are not part of the choice.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from evaluation.scripts.scanrefer_leaderboard_metrics import (
    load_side_by_side_compact_records,
)

POLICY_NAME = "plurality_avg_confidence"
CONSENSUS_RECORD_FIELDS = (
    "backend",
    "status",
    "iou",
    "predicted_bbox_3d_9dof",
    "gt_bbox_3d_9dof",
    "selected_object_id",
    "confidence",
    "query",
    "error",
)


def _confidence(record: dict[str, Any]) -> float:
    value = record.get("confidence")
    if value is None:
        return -1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def _scored_iou(record: dict[str, Any]) -> float:
    if str(record.get("status", "")).lower() == "failed":
        return 0.0
    value = record.get("iou")
    if value is None:
        return 0.0
    return float(value)


def choose_consensus_record(
    sample_id: str,
    candidates: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Choose one record by proposal plurality, then confidence.

    The function intentionally does not inspect IoU, GT bbox, or predicted bbox
    while ranking candidates. Those fields remain in the returned copy because
    downstream aggregation expects the original side_by_side record shape.
    """
    if not candidates:
        raise ValueError(f"no candidates for sample_id={sample_id!r}")

    source_order = {source: idx for idx, (source, _) in enumerate(candidates)}
    groups: dict[Any, list[tuple[str, dict[str, Any]]]] = {}
    votes: list[dict[str, Any]] = []
    for source, record in candidates:
        if record.get("sample_id") != sample_id:
            raise ValueError(
                f"candidate from {source!r} has sample_id={record.get('sample_id')!r}; "
                f"expected {sample_id!r}"
            )
        proposal_id = record.get("selected_object_id")
        groups.setdefault(proposal_id, []).append((source, record))
        votes.append(
            {
                "source": source,
                "selected_object_id": proposal_id,
                "confidence": record.get("confidence"),
                "status": record.get("status"),
            }
        )

    def group_score(group: list[tuple[str, dict[str, Any]]]) -> tuple[float, ...]:
        confidences = [_confidence(record) for _, record in group]
        return (
            float(len(group)),
            sum(confidences) / len(confidences),
            max(confidences),
            -float(min(source_order[source] for source, _ in group)),
        )

    chosen_group = max(groups.values(), key=group_score)
    source, record = max(
        chosen_group,
        key=lambda item: (_confidence(item[1]), -source_order[item[0]]),
    )

    chosen = {
        key: copy.deepcopy(value)
        for key, value in record.items()
        if key != "tool_trace"
    }
    chosen["consensus_policy"] = POLICY_NAME
    chosen["consensus_choice_source"] = source
    chosen["consensus_group_size"] = len(chosen_group)
    chosen["consensus_n_sources"] = len(candidates)
    chosen["consensus_votes"] = votes
    chosen["tool_trace"] = [
        {
            "tool_name": "consensus_select",
            "tool_input": {
                "policy": POLICY_NAME,
                "choice_source": source,
                "selected_object_id": record.get("selected_object_id"),
                "group_size": len(chosen_group),
                "n_sources": len(candidates),
                "votes": votes,
            },
            "response_text": (
                "consensus selected "
                f"proposal_id={record.get('selected_object_id')} from source={source} "
                f"with group_size={len(chosen_group)}/{len(candidates)}"
            ),
        }
    ]
    return chosen


def _index_by_sample_id(source: str, payload: dict[str, Any], backend: str) -> dict[str, Any]:
    if backend not in payload:
        raise ValueError(f"{source}: missing backend={backend!r}")
    per_sample = payload[backend].get("per_sample")
    if not isinstance(per_sample, list) or not per_sample:
        raise ValueError(f"{source}: {backend}.per_sample must be a non-empty list")
    out: dict[str, Any] = {}
    for item in per_sample:
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{source}: per_sample item missing sample_id")
        if sample_id in out:
            raise ValueError(f"{source}: duplicate sample_id={sample_id!r}")
        out[sample_id] = item
    return out


def build_consensus_payload(
    payloads: dict[str, dict[str, Any]],
    *,
    backend: str = "pack_v1",
) -> dict[str, Any]:
    """Merge side_by_side payloads into a new consensus payload."""
    if not payloads:
        raise ValueError("at least one input payload is required")

    indexed = {
        source: _index_by_sample_id(source, payload, backend)
        for source, payload in payloads.items()
    }
    first_source = next(iter(indexed))
    sample_order = list(indexed[first_source].keys())
    first_set = set(sample_order)
    for source, by_sample in indexed.items():
        if set(by_sample) != first_set:
            missing = sorted(first_set - set(by_sample))[:5]
            extra = sorted(set(by_sample) - first_set)[:5]
            raise ValueError(
                "sample_id sets differ across inputs; "
                f"source={source!r} missing={missing} extra={extra}"
            )

    chosen_records = [
        choose_consensus_record(
            sample_id,
            [(source, by_sample[sample_id]) for source, by_sample in indexed.items()],
        )
        for sample_id in sample_order
    ]
    ious = [_scored_iou(record) for record in chosen_records]
    n = len(chosen_records)
    return {
        backend: {
            "n": n,
            "mean_iou": sum(ious) / n if n else 0.0,
            "Acc@0.25": sum(iou >= 0.25 for iou in ious) / n if n else 0.0,
            "Acc@0.50": sum(iou >= 0.50 for iou in ious) / n if n else 0.0,
            "per_sample": chosen_records,
            "consensus_policy": POLICY_NAME,
            "consensus_sources": list(payloads.keys()),
        }
    }


def build_consensus_payload_from_paths(
    inputs: dict[str, Path],
    *,
    backend: str = "pack_v1",
) -> dict[str, Any]:
    """Stream compact source records from side_by_side paths and build consensus."""
    payloads: dict[str, dict[str, Any]] = {}
    for source, path in inputs.items():
        payloads[source] = {
            backend: {
                "per_sample": list(
                    load_side_by_side_compact_records(
                        path,
                        backend=backend,
                        fields=CONSENSUS_RECORD_FIELDS,
                    )
                )
            }
        }
    return build_consensus_payload(payloads, backend=backend)


def _parse_input_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--input must use NAME=PATH so consensus votes are auditable"
        )
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("--input NAME cannot be empty")
    path = Path(raw_path)
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=_parse_input_arg,
        required=True,
        help="Consensus source as NAME=path/to/side_by_side.json; repeatable.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backend", default="pack_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_consensus_payload_from_paths(dict(args.input), backend=args.backend)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "side_by_side.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = payload[args.backend]
    print(f"wrote {output_path}")
    print(
        f"n={metrics['n']} mean_iou={metrics['mean_iou']:.4f} "
        f"acc25={metrics['Acc@0.25']:.4f} acc50={metrics['Acc@0.50']:.4f}"
    )


if __name__ == "__main__":
    main()
