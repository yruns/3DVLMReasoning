# 05 — Evaluation: MNAS, Judge, Protocol

How MNAS is computed, why the judge is Gemini 2.5 Pro despite the OpenEQA paper using GPT-4, and what the production scoring harness does that the upstream does not. Line numbers against HEAD = `a8e651e`.

## 5.1 MNAS — definition

**MNAS** ("mean normalised automatic score") is a 0–100 float derived from per-question integer scores in `{1, 2, 3, 4, 5}` returned by an LLM judge:

```
MNAS = mean over all questions of [ 100 * (clip(raw_score, 1, 5) - 1) / 4 ]
```

Two live implementation sites:

| Site | Role | Line |
|---|---|---|
| **Upstream canonical** — `external/open-eqa/evaluate-predictions.py:117-118` | Aggregator used by the OpenEQA CVPR 2024 paper | `scores = 100.0 * (np.clip(scores, 1, 5) - 1) / 4 ; print("final score: {:.1f}".format(np.mean(scores)))` |
| **Project re-implementation** — `src/benchmarks/openeqa_official_eval.py:212-216` | What our harness actually returns in `final_score` | `scaled_scores = 100.0 * (np.clip(raw_scores, 1, 5) - 1) / 4 ; final_score = float(np.mean(scaled_scores))` |

The per-question score comes from the upstream LLM-match pipeline at `external/open-eqa/openeqa/evaluation/llm_match.py:29` (`get_llm_match_score`). We reuse that prompt and JSON-parsing logic verbatim and only swap the LLM call.

The literal string `"MNAS"` does **not** appear in the eval module code itself; the project uses the name in results documents (`10_experiment_log/*.md`) and in the related-work catalogue (`src/evaluation/related_work.py:86`).

## 5.2 The judge — why Gemini 2.5 Pro, not GPT-4?

**OpenEQA paper (CVPR 2024) judge**: `gpt-4` with temperature 0.2.
**This project default judge**: `gemini-2.5-pro` with temperature 0.2 (same temperature, different provider).

Default set since the module was first introduced (commit `eb3c6b8`, "codex end"): `evaluate_predictions_with_official_llm_match(…, eval_model="gemini-2.5-pro", …)` at `src/benchmarks/openeqa_official_eval.py:122`. CLI default also `gemini-2.5-pro` in `openeqa_official_question_pilot.py` `--eval-model`.

**Practical reasons for Gemini**:
1. **Access** — the project runs in an environment where Azure-compat OpenAI endpoints are restricted to the VLM `gpt-5.4` deployment; a separate `gpt-4` deployment for judging is not available on the same key pool.
2. **Pool-key rotation** — `utils.llm_client.GeminiClientPool` (instrumented by commit `85545e3`) holds 5 keys and rotates on rate-limit, enabling 12-worker concurrent scoring. The OpenAI pool available to the project does not provide equivalent rotation with `gpt-4`.
3. **Cost / throughput** — scoring 1050 × (2 predictions per question) = 2100 LLM calls per full eval; sequential GPT-4 scoring would be prohibitive in the 5–10 min window the iteration cadence wants.

**Methodological caveat (flagged everywhere)**:
- `10_experiment_log/README.md` line 7: *"**Judge:** Gemini 2.5 Pro (paper uses GPT-4)"*.
- `10_experiment_log/leaderboard.md` line 9: *"Our judge is Gemini 2.5 Pro; all other entries use GPT-4 as judge (per OpenEQA paper). Scores not directly comparable."*
- `10_experiment_log/leaderboard.md §ToDo` item 1: *"Re-evaluate with GPT-4 as judge (same protocol as all baselines)"*.

This caveat is a **REQUIRES** entry on Claim 1 in `00_research_manifest.md`; any paper shipping the 73.1 number must include a matched-judge re-score before camera-ready.

Gemini vs GPT-4 as an LLM-match judge has not been formally validated on OpenEQA; correlation studies on other QA benchmarks suggest the two judges rank comparably but agree to ± 2–3 MNAS at aggregate. Treat the 73.1 as an upper-bound until matched.

## 5.3 Entry point

`evaluate_predictions_with_official_llm_match(…)` at `src/benchmarks/openeqa_official_eval.py:117-225`:

| Parameter | Default | Purpose |
|---|---|---|
| `dataset_items: list[dict]` | — | Must contain matching `question_id`s for every prediction |
| `predictions: list[dict]` | — | `{"question_id": str, "answer": str}` per item |
| `output_path: Path` | — | Persistent `{question_id: int}` score map for resume |
| `official_repo_root: Path` | `external/open-eqa` | The cloned upstream repo |
| `eval_model: str` | `"gemini-2.5-pro"` | Judge |
| `verbose: bool` | `False` | Log each response |
| `max_workers: int` | `12` | Scorer concurrency |
| `max_retries: int` | `10` | Per-request retry budget |

Flow (line references in `openeqa_official_eval.py`):
1. **:142** `_ensure_repo_on_path(official_repo_root)` — raise if cloned repo missing.
2. **:143** `importlib.import_module("openeqa.evaluation.llm_match")` — pulls upstream code.
3. **:156-159** Monkey-patch `llm_match.call_openai_api` with `_make_official_call_adapter(eval_model, max_retries)` and `set_openai_key = lambda: None`.
4. **:164-171** Resume: if `output_path` exists, load existing `{qid: int}` into `all_scores`; skip already-scored predictions.
5. **:195-211** `ThreadPoolExecutor(max_workers=12)` scores remaining predictions concurrently; each worker acquires `save_lock` to write the updated map back to disk **after every** score.
6. **:212-216** Compute MNAS over all scored predictions; return `{final_score, score_by_question_id, …}`.

## 5.4 Pool-key rotation + progressive resume (commit `85545e3`)

`_make_official_call_adapter` at `:67-114`:
- For any `eval_model` whose name contains `"gemini"`: use `utils.llm_client.GeminiClientPool.invoke_with_full_retry`, which:
  - Exponential backoff on 429/timeout/5xx;
  - Rotates through all pool keys between attempts;
  - Caps at `max_retries` total attempts.
- Otherwise: `get_langchain_chat_model(deployment_name=eval_model, use_pool=False)` — single-client path.

**Progressive save rationale**: OpenEQA eval runs are hour-long. Before `85545e3`, a single 429 stall would wedge the whole run; now each newly scored question writes the complete `{qid: int}` map back to disk so a killed job resumes from exactly where it stopped on restart. This is structurally similar to the outer pilot's `official_batch_summary.json` write-before-score pattern.

## 5.5 Per-sample retry at the pilot layer

`openeqa_official_question_pilot.py:632-666` wraps `run_one_sample` in an exponential-backoff retry loop (commit `c8abd4d`):

- Retryable: error strings containing `{"500", "502", "503", "429", "rate limit", "server error", "timeout", "connection", "resource exhausted"}`.
- Backoff schedule: `min(10 * 2^attempt, 120)` seconds = 10, 20, 40, 80, 120.
- Non-retryable: propagated immediately (`run_stage1_ranked` raises on zero keyframes, for example — we want that to surface, not retry).

Retries happen **per sample**, not per prediction; so a crash during Stage 2 for question *q* retries Stage 1 + Stage 2 + E2E, not just the failing LLM call. The scoring retry (§5.4) is independent and lives inside the scoring LLM call.

## 5.6 What the pilot produces and scores

In the pilot's `--evaluate` path (`openeqa_official_question_pilot.py:734-779`):

```
<output-root>/
  runs/<clip>/<qid>/stage2.json            ← raw Stage 2 structured response + tool trace
  runs/<clip>/<qid>/e2e.json               ← raw E2E (only if guard triggered)
  official_batch_summary.json              ← aggregated, written BEFORE scoring
  official_selected_questions.json         ← filtered OpenEQA dataset subset
  official_predictions_stage2.json         ← [{question_id, answer}, ...]  ← scored
  official_predictions_stage2-metrics.json ← {question_id: int 1-5}        ← scored
  official_predictions_e2e.json            ← [{question_id, answer}, ...]  ← scored
  official_predictions_e2e-metrics.json    ← {question_id: int 1-5}        ← scored
```

`build_prediction_file(results, field_name)` at pilot `:583-593` builds the prediction JSON. Both `stage2` and `e2e` fields are scored independently — even when the E2E guard was **not** triggered (in which case `e2e.json == stage2.json`), so the two metrics files are identical. This dual-scoring lets us read off the E2E lift directly from the metrics files.

## 5.7 Category-level MNAS

OpenEQA defines seven categories. Per-category MNAS is computed downstream of the JSON metrics map by joining on `dataset_items[i]["category"]`; see `docs/10_experiment_log/v14_inventory_20260404.md` for the v13 / v14 breakdown (Object State, Localization, Attribute, Functional, World Knowledge, Object Recognition, Spatial). The per-category computation is not in `openeqa_official_eval.py`; the per-version results docs compute it ad hoc from the raw JSON.

## 5.8 Leaderboard context

`docs/10_experiment_log/leaderboard.md` is kept as ground truth; reproduce here only the ScanNet-only top-of-table for quick reference:

| # | Method | ScanNet MNAS | Paper | Judge |
|---|---|:-:|---|---|
| — | Human | 87.7 | OpenEQA CVPR 2024 | — |
| **1** | **Ours (v14)** | **73.1** | This work | **Gemini 2.5 Pro** |
| 2 | GPT-4V 500Q subset | 51.3 | OpenEQA CVPR 2024 | GPT-4 |
| 3 | GPT-4 + LLaVA-1.5 | 45.4 | OpenEQA CVPR 2024 | GPT-4 |
| 4 | GPT-4 + ConceptGraphs | 37.8 | OpenEQA CVPR 2024 | GPT-4 |
| 5 | GPT-4 (blind) | 32.5 | OpenEQA CVPR 2024 | GPT-4 |

Our position paragraph (`leaderboard.md §Our Position`): **+21.8 MNAS over GPT-4V, +35.3 over GPT-4 + ConceptGraphs**, judge caveat in force. Methods reporting only ALL (ScanNet+HM3D) — CoV, 3D-Mem, GraphPad — are not comparable to our ScanNet-only score.

**Exclusion**: AlanaVLM (Gemini 1.5 Flash 74.0 / Gemini 1.5 Pro 66.9 / GPT-4V 50f 57.4) is kept out of the leaderboard per `10_experiment_log/leaderboard.md §Exclusions` due to disputed paper authenticity. If/when that is resolved, v14 (73.1) drops to #2 behind Gemini 1.5 Flash (74.0) on the same benchmark — important for the CVPR framing of Claim 1.

## 5.9 Running an eval (recipes)

All recipes are reproduced from `10_experiment_log/README.md §Evaluation Commands`; do not diverge from the canonical invocation unless debugging.

Full 1050Q run:
```bash
tmux new-session -d -s eval-v14 "PYTHONPATH=src python -m agents.examples.openeqa_official_question_pilot \
    --max-samples 2000 --workers 6 \
    --llm-rewrite --confidence-guard 0.6 \
    --max-reasoning-turns 10 --max-additional-views 2 \
    --evaluate --eval-model gemini-2.5-pro \
    --output-root tmp/openeqa_official_pilot_runs/v14_full 2>&1 | tee /tmp/v14_full.log"
```

Re-score an existing predictions file (no re-inference):
```python
from benchmarks.openeqa_official_eval import evaluate_predictions_with_official_llm_match
from pathlib import Path
evaluate_predictions_with_official_llm_match(
    dataset_items=[...],
    predictions=[...],
    output_path=Path("tmp/.../official_predictions_e2e-metrics.json"),
    eval_model="gemini-2.5-pro",
    max_workers=12,
    max_retries=10,
)
```

Replace `eval_model="gemini-2.5-pro"` with `"gpt-4"` (when a GPT-4 deployment is available) to produce the matched-judge number required by Claim 1.

## 5.10 Evaluation debt — matched-judge re-score, ALL split, per-category fair comparison

Three open items that every paper shipping MNAS 73.1 must address, cross-referenced from `10_experiment_log/leaderboard.md §ToDo`:

1. Re-score predictions with GPT-4 as judge on the same 1050Q split. Minimum sample: enough to produce a stable per-category estimate (~300 samples) for a calibration constant, then apply to the full 1050.
2. Run on HM3D scenes to produce an ALL split score — needed for apples-to-apples comparison with CoV, 3D-Mem, GraphPad.
3. Per-category fair comparison across baselines — AlanaVLM's per-category breakdown is not public; without it, the "we beat Gemini 1.5 Flash on Attribute" claim cannot be stated.

Cross-ref: `09_gotchas.md` for concrete failure modes of this harness (rate-limit wedge, `scene_info.json` provenance paths, resume inconsistency after partial crash).
