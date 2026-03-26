#!/usr/bin/env python3
"""Simple query-parsing smoke test against the current parser API."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from query_scene.parsing import QueryParser
from utils.llm_client import DEFAULT_MODEL

TEST_QUERIES = [
    "the red cup",
    "the pillow on the sofa",
    "the lamp near the window",
    "the sofa nearest the door",
    "the largest book on the shelf",
    "the smallest plant",
    "the lamp between the sofa and the TV",
    "the second chair from the left",
    "the first book from the right",
    "the pillow on the sofa nearest the door",
    "the red cup on the table near the window",
]

MOCK_CATEGORIES = [
    "pillow",
    "sofa",
    "cup",
    "door",
    "book",
    "shelf",
    "lamp",
    "tv",
    "chair",
    "table",
    "window",
    "plant",
]


def _dump_node(node, prefix: str = "root") -> None:
    logger.info(
        "{}: categories={}, attrs={}",
        prefix,
        node.categories,
        node.attributes or [],
    )
    for idx, constraint in enumerate(node.spatial_constraints):
        anchor_categories = [anchor.categories for anchor in constraint.anchors]
        logger.info(
            "{}.spatial[{}]: [{}] -> {}",
            prefix,
            idx,
            constraint.relation,
            anchor_categories or "N/A",
        )
        for anchor_idx, anchor in enumerate(constraint.anchors):
            _dump_node(anchor, prefix=f"{prefix}.anchor[{idx}][{anchor_idx}]")

    if node.select_constraint is not None:
        ref = node.select_constraint.reference
        ctype = node.select_constraint.constraint_type
        ctype = ctype.value if hasattr(ctype, "value") else str(ctype)
        logger.info(
            "{}.select: type={}, metric={}, order={}, reference={}",
            prefix,
            ctype,
            node.select_constraint.metric,
            node.select_constraint.order,
            ref.categories if ref else "N/A",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple query parser smoke test")
    parser.add_argument(
        "--llm_model",
        default=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        help="LLM model used by QueryParser",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("QueryParser smoke test")
    logger.info("model={}", args.llm_model)
    logger.info("=" * 60)

    query_parser = QueryParser(
        llm_model=args.llm_model,
        scene_categories=MOCK_CATEGORIES,
        use_pool="gemini" in args.llm_model.lower(),
    )

    failures = 0
    for query in TEST_QUERIES:
        logger.info("Query: {!r}", query)
        logger.info("-" * 40)
        try:
            result = query_parser.parse(query)
            logger.info("parse_mode={}", result.parse_mode.value)
            logger.info("num_hypotheses={}", len(result.hypotheses))
            for hypothesis in result.ordered_hypotheses():
                logger.info(
                    "hypothesis rank={} kind={}", hypothesis.rank, hypothesis.kind.value
                )
                _dump_node(hypothesis.grounding_query.root)
                logger.info(
                    "raw_query={!r}",
                    hypothesis.grounding_query.raw_query or query,
                )
        except Exception as exc:  # pragma: no cover - debugging helper
            failures += 1
            logger.exception("Failed to parse {!r}: {}", query, exc)

    logger.info("=" * 60)
    logger.info("failures={}", failures)
    logger.info("=" * 60)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
