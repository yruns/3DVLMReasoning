#!/usr/bin/env python3
"""Inspect nested query parsing behavior against the current parser."""

from __future__ import annotations

__test__ = False

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
    {
        'query': 'the pillow on the sofa',
        'description': 'Simple spatial relation: pillow -[on]-> sofa',
    },
    {
        'query': 'the red cup',
        'description': 'Simple object with attribute, no spatial relation',
    },
    {
        'query': 'the pillow on the sofa nearest the door',
        'description': 'Nested: pillow -[on]-> sofa -[nearest]-> door',
    },
    {
        'query': 'the sofa nearest the door',
        'description': 'Superlative: sofa -[nearest]-> door',
    },
    {
        'query': 'the largest book on the shelf',
        'description': 'Superlative + spatial: book -[on]-> shelf, select largest',
    },
    {
        'query': 'the lamp on the table near the window',
        'description': 'Multi-level: lamp -[on]-> table -[near]-> window',
    },
    {
        'query': 'the cup on the table in the kitchen',
        'description': 'Multi-level: cup -[on]-> table -[in]-> kitchen',
    },
    {
        'query': 'the lamp between the sofa and the TV',
        'description': 'Multi-anchor: lamp -[between]-> [sofa, TV]',
    },
    {
        'query': 'the second chair from the left',
        'description': 'Ordinal: select 2nd chair by x_position',
    },
    {
        'query': 'the red book on the shelf above the desk',
        'description': 'Complex: book(red) -[on]-> shelf -[above]-> desk',
    },
]

MOCK_SCENE_CATEGORIES = [
    'pillow', 'throw_pillow', 'cushion',
    'sofa', 'couch', 'armchair', 'chair',
    'table', 'coffee_table', 'side_table', 'desk',
    'lamp', 'table_lamp', 'floor_lamp',
    'book', 'bookshelf', 'shelf',
    'cup', 'mug', 'glass',
    'door', 'window',
    'tv', 'television',
    'plant', 'flower', 'vase',
    'kitchen', 'living_room', 'bedroom',
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


def run_parser_demo(queries: list[dict[str, str]], llm_model: str) -> int:
    parser = QueryParser(
        llm_model=llm_model,
        scene_categories=MOCK_SCENE_CATEGORIES,
        use_pool="gemini" in llm_model.lower(),
    )

    failures = 0
    for item in queries:
        query = item["query"]
        description = item["description"]
        logger.info("=" * 70)
        logger.info("Query: {!r}", query)
        logger.info("Description: {}", description)
        logger.info("-" * 70)
        try:
            result = parser.parse(query)
            logger.info("parse_mode={}", result.parse_mode.value)
            logger.info("num_hypotheses={}", len(result.hypotheses))
            for hypothesis in result.ordered_hypotheses():
                logger.info(
                    "hypothesis rank={} kind={}",
                    hypothesis.rank,
                    hypothesis.kind.value,
                )
                _dump_node(hypothesis.grounding_query.root)
        except Exception as exc:  # pragma: no cover - debugging helper
            failures += 1
            logger.exception("Failed to parse {!r}: {}", query, exc)

    return failures


def main() -> None:
    cli = argparse.ArgumentParser(description='Test nested query parsing')
    cli.add_argument(
        '--llm_model',
        type=str,
        default=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        help='LLM model name for QueryParser.',
    )
    args = cli.parse_args()

    logger.info('=' * 70)
    logger.info('Nested Query Parsing Test')
    logger.info('=' * 70)
    logger.info(f'Testing {len(TEST_QUERIES)} queries...')
    logger.info(f'Scene categories: {len(MOCK_SCENE_CATEGORIES)} categories')

    failures = run_parser_demo(TEST_QUERIES, args.llm_model)

    logger.info('=' * 70)
    logger.info('failures={}', failures)
    logger.info('=' * 70)
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == '__main__':
    main()
