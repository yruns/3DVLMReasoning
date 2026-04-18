# 09 — Gotchas

Known foot-guns at HEAD = `a8e651e`. Everything here is actionable: what bites you, where the authoritative fact lives, and what to do about it. Do not add entries without a concrete reproduction or an anchoring commit.

## 9.1 CLAUDE.md is stale (Stage 2 default model)

**Claim**: `CLAUDE.md` §Stage 2 Backend Configuration says *"Default VLM backend is `gpt-5.2-2025-12-11`"*.
**Reality**: `src/agents/core/agent_config.py:44` — `model_name: str = "gpt-5.4-2026-03-05"`. Default changed in commit `2abb404` (v12, 2026-04-02).
**Consequence**: A new agent that trusts `CLAUDE.md` will reason about the wrong backbone. Every `Stage2DeepAgentConfig()` instantiation without an explicit `model_name` picks the HEAD default, which is `gpt-5.4`. Resulting MNAS numbers *will* differ from any v9–v11 comparison predicated on `gpt-5.2`.
**Fix**: Trust `src/agents/core/agent_config.py:44` over `CLAUDE.md`. A CLAUDE.md edit is pending and has not been made in this session (not our scope).

## 9.2 Linux server: GPU 1 is broken

**Source**: commit `f13048b` "docs: document GPU 1 broken on Linux server" (CLAUDE.md §GPU Usage on Linux).
**Failure mode**: `export CUDA_VISIBLE_DEVICES=1` causes CUDA init to fail and takes the whole process down.
**What to do**: Use any of GPUs `0, 2, 3, 4, 5, 6, 7`. In parallel orchestrations, explicitly skip `1`:

```bash
for scene_gpu in "scene_a:0" "scene_b:2" "scene_c:3" "scene_d:4"; do
    ...
done
```

Check availability with `nvidia-smi --query-gpu=index,memory.free --format=csv,noheader` before launching.

## 9.3 Python environment differs by platform

**macOS**: `uv` + `.venv` (see CLAUDE.md §Package Management).

**Linux**: conda env **`conceptgraph`**. The local `.venv` has been **deleted** and **must not be recreated** — a prior `.venv` shadowed conda's python and caused silent import failures (referenced in `docs/0325.md` which itself is deleted from the workspace but captured in commit history).

Always activate the correct env before running the pilot:

```bash
if [[ "$(uname -s)" == "Linux" ]]; then
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
else
    source .venv/bin/activate 2>/dev/null || (uv venv && source .venv/bin/activate)
fi
```

## 9.4 Long-running jobs must go in `tmux`

**Source**: CLAUDE.md §Long-Running Tasks is authoritative and explicit.
**Reason**: Bash tool timeout is 120 s default, 600 s max; a 1050Q full eval takes hours. Network disconnects lose un-tmux'd work instantly.
**Rule**: Every pilot run longer than two minutes is `tmux new-session -d -s <name> "..."`. Recipe in `05_evaluation.md §5.9`.

## 9.5 Strict no-fallback rule (HEAD enforces it)

**Source**: CLAUDE.md §Strict No-Fallback Rule + commit `17cc18c` "fix: eliminate 9 silent fallback patterns in pipeline" + commit `598bbfe` "feat: multi-label category index + remove silent fallback in Stage 1".
**Scope**: every pipeline step either succeeds as designed or raises.
**Concrete enforcers on HEAD**:

| Enforcer | Location |
|---|---|
| `enriched_objects.json` must exist | `src/query_scene/keyframe_selector.py:352-358` raises `FileNotFoundError` |
| Stage 1 must return ≥ 1 keyframe | `src/agents/examples/openeqa_official_question_pilot.py:440` raises `RuntimeError` |
| Required dirs must exist | `ensure_runtime_scene` raises `FileNotFoundError` if `conceptgraph/` or `raw/` missing |
| FAISS required on Linux | `src/query_scene/index_builder.py:29-33` raises `ImportError` on Linux if FAISS missing |
| Upstream OpenEQA repo must be cloned | `src/benchmarks/openeqa_official_eval.py:28` raises `FileNotFoundError` |

**What an agent should NOT do**:
- `try/except ImportError` that substitutes `None` for torch/SAM/open_clip.
- Replace SAM-generated masks with rectangular bboxes on SAM failure.
- Provide a default when a required file is missing.

If you find yourself writing any of these, stop — surface the error, fix the root cause.

## 9.6 `docs/` under version control is intentionally minimal

**Observation**: `git status` on the workspace shows multiple `D docs/<…>.md` entries from prior analyses (`docs/eval_analysis_v10_enriched_20260330.md`, `docs/eval_analysis_v13_full_20260403.md`, `docs/optimization_plan_20260327.md`, and others).
**Intent**: The supervisor explicitly requested **not** to restore these during doc work. Their content is preserved in commit history (use `git show <commit>:<path>`). This `docs/` tree is the replacement.
**What NOT to do**: `git restore docs/eval_analysis_v13_full_20260403.md` or similar. If you need content from those files, reach for `git show` and summarise — never recreate them in the working tree.

## 9.7 OpenEQA loader vs. local layout

**Summary**: `src/benchmarks/openeqa_loader.py` expects a layout that **does not exist** locally (`data/frames/<episode>/`). Detail at `07_data_layout.md §7.2`.
**Consequence**: `OpenEQADataset.from_path(…).load_frames()` returns an empty list; any code that runs Stage 2 on those frames silently has zero images.
**What to do**: Use `src/agents/examples/openeqa_official_question_pilot.py`, which reads `data/open-eqa-v0.json` directly and bridges via `ensure_runtime_scene`. Or use the `OpenEQAAdapter` wrapper (`src/agents/adapters/openeqa_adapter.py`) if you are writing new code.

## 9.8 CLAUDE.md generational drift — a watch list

CLAUDE.md is the human-onboarding doc; it drifts slower than the code. Specific watch items whose staleness would be most surprising:

| Thing | CLAUDE.md says | HEAD says | Drift |
|---|---|---|---|
| Stage 2 default model | `gpt-5.2-2025-12-11` | `gpt-5.4-2026-03-05` (`agent_config.py:44`) | **stale** |
| `.venv` status on Linux | "deleted, do not recreate" | same; confirmed at HEAD by absence | up to date |
| `external/open-eqa` requirement | documented | enforced at `openeqa_official_eval.py:28` | up to date |
| Default `--workers` | 6 (CLAUDE hints parallelism) | 6 (pilot default, commit `25a67fb`) | up to date |

If you edit `CLAUDE.md`, keep this table in sync or add a new row.

## 9.9 `scene_info.json` provenance paths are stale

**Source**: `data/OpenEQA/scannet/<clip>/conceptgraph/scene_info.json` may contain absolute paths like `/home/ysh/…` from the prep host.
**Risk**: An agent that tries to use those paths as runnable will hit file-not-found.
**Rule**: Treat `scene_info.json` fields as *provenance only*. Computed paths come from `ensure_runtime_scene` (see `07_data_layout.md §7.3`).

## 9.10 Concurrency: per-scene lock is mandatory

**Observed failure mode (pre-v9)**: two workers writing the same overlay races on symlink creation → intermittent `FileExistsError` or stale symlinks pointing into a partially-built tree.
**Fix**: `_get_scene_lock(clip_id)` at `src/agents/examples/openeqa_official_question_pilot.py:47-55` wraps every `ensure_runtime_scene` call in a per-clip `threading.Lock`.
**Rule for new code**: any new entry point that instantiates `KeyframeSelector` across multiple threads/workers **must** serialise overlay creation via the same (or an equivalent) per-clip lock. No exceptions.

## 9.11 Eval judge non-comparability

**Claim**: our 73.1 MNAS uses Gemini 2.5 Pro as judge; published OpenEQA baselines use GPT-4.
**Source**: `05_evaluation.md §5.2` + `10_experiment_log/leaderboard.md §Important Notes`.
**What NOT to do**: Put 73.1 in a paper without the matched-judge re-score — reviewers will (rightly) ask for it. It is a REQUIRES on Claim 1 in `00_research_manifest.md`.

## 9.12 Tool-trace writes happen early, MNAS scoring happens late

**Pilot write order** (confirmed at `openeqa_official_question_pilot.py:729-737`): the `official_batch_summary.json` is written **before** scoring starts, so if the LLM judge stalls or crashes the inference-side artefacts are never lost. The scoring-side metrics files are written progressively (§5.4).
**Implication**: If a run appears to "hang" near the end, the inference is usually complete — look for `official_predictions_*.json` already on disk; the judge is probably rate-limited. You can re-score offline using the snippet in `05_evaluation.md §5.9`.

## 9.13 Don't confuse the two `docs/` conventions

- `docs/10_experiment_log/*.md` — primary evidence, treat as **immutable**. If you find an error, open an issue or patch with a commit message that cross-references the discrepancy; do not silently edit.
- `docs/0X_*.md`, `docs/1X_*.md` (this series) — derived documentation. Edit freely with commit attribution; keep cross-refs to `10_experiment_log/` up to date.

Cross-ref: `05_evaluation.md §5.10` lists the three formal open evaluation-debt items; `07_data_layout.md §7.6` lists the layout pitfalls; this file is the full foot-gun superset.
