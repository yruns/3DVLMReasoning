"""Official OpenEQA evaluation wrapper using the cloned upstream repo.

This module intentionally reuses the official ``openeqa.evaluation.llm_match``
logic from the upstream OpenEQA repository and only swaps the model backend to
the current project's Azure-compatible chat client.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from utils.llm_client import get_langchain_chat_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_REPO_ROOT = PROJECT_ROOT / "external" / "open-eqa"


def _ensure_repo_on_path(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(f"Official OpenEQA repo not found: {repo_root}")
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _normalize_response_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    chunks.append(str(text))
        return "\n".join(chunk for chunk in chunks if chunk).strip()
    return str(content)


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
            continue
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
    return "\n\n".join(part for part in parts if part).strip()


def _make_official_call_adapter(default_model: str):
    client_cache: dict[tuple[str, float, int], Any] = {}

    def call_openai_api(
        messages: list[dict[str, Any]],
        model: str = "gpt-4",
        seed: int | None = None,
        max_tokens: int = 32,
        temperature: float = 0.2,
        verbose: bool = False,
    ) -> str:
        deployment = model or default_model
        cache_key = (deployment, float(temperature), int(max_tokens))
        client = client_cache.get(cache_key)
        if client is None:
            client = get_langchain_chat_model(
                deployment_name=deployment,
                temperature=temperature,
                max_tokens=max_tokens,
                use_pool="gemini" in deployment.lower(),
            )
            client_cache[cache_key] = client

        prompt = _messages_to_prompt(messages)
        if not prompt:
            raise ValueError("Official evaluation received an empty prompt.")

        if seed is not None and verbose:
            logger.info("Official eval backend ignores seed={} for model={}", seed, deployment)

        response = client.invoke(prompt)
        text = _normalize_response_content(getattr(response, "content", response)).strip()
        if verbose:
            logger.info(
                "Official eval response model={} prompt_chars={} text={!r}",
                deployment,
                len(prompt),
                text[:200],
            )
        return text

    return call_openai_api


def evaluate_predictions_with_official_llm_match(
    dataset_items: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    output_path: Path,
    official_repo_root: Path = DEFAULT_OFFICIAL_REPO_ROOT,
    eval_model: str = "gemini-2.5-pro",
    verbose: bool = False,
) -> dict[str, Any]:
    """Score predictions with the upstream OpenEQA LLM-match evaluator.

    ``dataset_items`` and ``predictions`` should describe the same question ids.
    ``output_path`` stores the raw official score mapping ``{question_id: score}``.
    """

    if not dataset_items:
        raise ValueError("No dataset items were provided for evaluation.")
    if not predictions:
        raise ValueError("No predictions were provided for evaluation.")

    _ensure_repo_on_path(official_repo_root)
    official_llm_match = importlib.import_module("openeqa.evaluation.llm_match")

    question_id_to_item = {item["question_id"]: item for item in dataset_items}
    prediction_ids = [item["question_id"] for item in predictions]
    dataset_ids = list(question_id_to_item)
    if set(prediction_ids) != set(dataset_ids):
        missing_predictions = sorted(set(dataset_ids) - set(prediction_ids))
        extra_predictions = sorted(set(prediction_ids) - set(dataset_ids))
        raise ValueError(
            "Prediction ids do not match dataset ids. "
            f"missing={missing_predictions[:5]} extra={extra_predictions[:5]}"
        )

    official_llm_match.call_openai_api = _make_official_call_adapter(eval_model)
    official_llm_match.set_openai_key = lambda key=None: None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_scores: dict[str, int] = {}
    for prediction in predictions:
        question_id = prediction["question_id"]
        item = question_id_to_item[question_id]
        score = official_llm_match.get_llm_match_score(
            question=item["question"],
            answer=item["answer"],
            prediction=prediction.get("answer"),
            extra_answers=item.get("extra_answers"),
            openai_model=eval_model,
            verbose=verbose,
        )
        all_scores[question_id] = int(score)
        output_path.write_text(json.dumps(all_scores, indent=2, ensure_ascii=False) + "\n")

    raw_scores = np.array(list(all_scores.values()), dtype=float)
    scaled_scores = 100.0 * (np.clip(raw_scores, 1, 5) - 1) / 4 if raw_scores.size else []
    final_score = float(np.mean(scaled_scores)) if raw_scores.size else 0.0
    return {
        "official_repo_root": str(official_repo_root),
        "eval_model": eval_model,
        "num_predictions": len(predictions),
        "metrics_path": str(output_path),
        "raw_score_path": str(output_path),
        "final_score": final_score,
        "score_by_question_id": all_scores,
    }
