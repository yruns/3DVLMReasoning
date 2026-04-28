"""Re-evaluate predictions using a single high-quota AK that overrides the
default GeminiClientPool config. Used when the default 3-key pool is being
rate-limited and we have a higher-quota AK available.

Usage:
    PYTHONPATH=src AK_OVERRIDE=<ak> python scripts/reeval_with_custom_ak.py \
        --dataset data/open-eqa-v0.json \
        --predictions tmp/<run>/official_predictions_<column>.json \
        --output    tmp/<run>/official_predictions_<column>-metrics.json \
        --max-workers 16 --max-retries 20
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# IMPORTANT: patch BEFORE importing anything that touches GeminiClientPool.
AK = os.environ.get("AK_OVERRIDE")
if not AK:
    raise SystemExit("AK_OVERRIDE env var required")

import utils.llm_client as llm_client_mod  # noqa: E402

llm_client_mod.GEMINI_POOL_CONFIGS = [
    {
        "endpoint": "https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl",
        "model_name": "gemini-2.5-pro",
        "api_key": AK,
        "api_version": "2024-03-01-preview",
        "weight": 1,
    },
]
# Reset the singleton so it picks up the new config on first use.
llm_client_mod.GeminiClientPool._instance = None  # type: ignore[attr-defined]
print(f"[patch] GeminiClientPool overridden to single AK ...{AK[-12:]}")

# Now import the eval entry.
from benchmarks.openeqa_official_eval import (  # noqa: E402
    evaluate_predictions_with_official_llm_match,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--eval-model", default="gemini-2.5-pro")
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--max-retries", type=int, default=20)
    args = p.parse_args()

    dataset = json.loads(args.dataset.read_text())
    predictions = json.loads(args.predictions.read_text())
    pred_ids = {p["question_id"] for p in predictions}
    filtered = [d for d in dataset if d["question_id"] in pred_ids]
    print(
        f"[reeval] dataset={len(dataset)} predictions={len(predictions)} "
        f"filtered={len(filtered)} output={args.output}"
    )

    result = evaluate_predictions_with_official_llm_match(
        dataset_items=filtered,
        predictions=predictions,
        output_path=args.output,
        eval_model=args.eval_model,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
    )
    print(
        f"[reeval] DONE n={result.get('num_scored')} "
        f"final_score={result.get('final_score')}"
    )


if __name__ == "__main__":
    main()
