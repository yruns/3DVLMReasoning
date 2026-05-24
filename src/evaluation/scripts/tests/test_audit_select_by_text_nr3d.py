"""Tests for ``scripts/audit_select_by_text_nr3d.py`` helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_audit_module():
    script = Path(__file__).resolve().parents[4] / "scripts" / "audit_select_by_text_nr3d.py"
    spec = importlib.util.spec_from_file_location("audit_select_by_text_nr3d", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_summarize_hypothesis_viewpoint_stats_counts_nested_constraints() -> None:
    audit = _load_audit_module()

    stats = audit.summarize_hypothesis_viewpoint_stats(
        {
            "hypotheses": [
                {
                    "grounding_query": {
                        "viewpoint_contexts": [{"id": "vp1"}],
                        "root": {
                            "categories": ["cabinet"],
                            "spatial_constraints": [
                                {
                                    "relation": "right_of",
                                    "reference_frame": "viewer",
                                    "execution_policy": "soft",
                                    "anchors": [
                                        {
                                            "categories": ["door"],
                                            "spatial_constraints": [
                                                {
                                                    "relation": "near",
                                                    "reference_frame": "world",
                                                    "execution_policy": "hard",
                                                    "anchors": [
                                                        {"categories": ["wall"]}
                                                    ],
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                            "select_constraint": {
                                "constraint_type": "superlative",
                                "metric": "x_position",
                                "order": "max",
                                "reference_frame": "ambiguous",
                                "execution_policy": "rank_only",
                            },
                        },
                    }
                }
            ]
        }
    )

    assert stats == {
        "viewpoint_context_count": 1,
        "reference_frame_counts": {"ambiguous": 1, "viewer": 1, "world": 1},
        "execution_policy_counts": {"hard": 1, "rank_only": 1, "soft": 1},
    }


def test_summarize_aggregates_viewpoint_stats() -> None:
    audit = _load_audit_module()

    summary = audit.summarize(
        [
            {
                "unmeasurable": False,
                "hit_at_K": {"3": 1},
                "recall_at_K": {"3": 1.0},
                "gt_coverage": 0.2,
                "first_hit_rank": 1,
                "viewpoint_context_count": 1,
                "reference_frame_counts": {"viewer": 1, "world": 2},
                "execution_policy_counts": {"soft": 1, "hard": 2},
            },
            {
                "unmeasurable": False,
                "hit_at_K": {"3": 0},
                "recall_at_K": {"3": 0.0},
                "gt_coverage": 0.05,
                "viewpoint_context_count": 0,
                "reference_frame_counts": {"ambiguous": 1},
                "execution_policy_counts": {"rank_only": 1},
            },
        ],
        [3],
    )

    assert summary["n_with_viewpoint_contexts"] == 1
    assert summary["reference_frame_counts"] == {
        "ambiguous": 1,
        "viewer": 1,
        "world": 2,
    }
    assert summary["execution_policy_counts"] == {
        "hard": 2,
        "rank_only": 1,
        "soft": 1,
    }
