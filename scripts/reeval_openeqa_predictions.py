"""Re-evaluate already-saved OpenEQA predictions with Gemini judge.

Used when the inline `--evaluate` step in the pilot crashes (e.g. 503 outage)
and the predictions JSON is intact but the *-metrics.json is missing/partial.

Usage:
    PYTHONPATH=src python scripts/reeval_openeqa_predictions.py \
        --dataset data/open-eqa-v0.json \
        --predictions tmp/<run>/official_predictions_e2e.json \
        --output tmp/<run>/official_predictions_e2e-metrics.json \
        --eval-model gemini-2.5-pro --max-workers 8 --max-retries 20

Reads predictions, restricts dataset to those question_ids, and scores them.
Resumes automatically from any pre-existing partial output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.openeqa_official_eval import (
    evaluate_predictions_with_official_llm_match,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--eval-model", default="gemini-2.5-pro")
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--max-retries", type=int, default=20)
    args = p.parse_args()

    dataset = json.loads(args.dataset.read_text())
    predictions = json.loads(args.predictions.read_text())

    pred_ids = {p["question_id"] for p in predictions}
    filtered_dataset = [d for d in dataset if d["question_id"] in pred_ids]
    print(
        f"[reeval] dataset={len(dataset)} predictions={len(predictions)} "
        f"filtered_dataset={len(filtered_dataset)} output={args.output}"
    )

    result = evaluate_predictions_with_official_llm_match(
        dataset_items=filtered_dataset,
        predictions=predictions,
        output_path=args.output,
        eval_model=args.eval_model,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
    )
    final_score = result.get("final_score")
    n = result.get("num_scored")
    print(f"[reeval] DONE. n={n} final_score={final_score}")


if __name__ == "__main__":
    main()
