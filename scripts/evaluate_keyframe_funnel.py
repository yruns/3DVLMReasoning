#!/usr/bin/env python
"""Retired legacy funnel evaluator.

This file is kept only to make old invocations fail with a clear message.
Modern Stage-2 runs do not receive prep-seeded first-person frames; visual
evidence enters through selector tools and crop/mark tools during the agent run.
"""

raise SystemExit(
    "This legacy funnel evaluator was retired. Use tool-trace analysis for "
    "selector-acquired visual evidence instead."
)
