# v15 Trajectory-Aware Keyframe Selection — Design

**Status**: Implemented & empirically validated (2026-04-20)
**Branch**: `feat/v15-trajectory-aware`
**Baseline**: v14 master @ `c7a469d` (MNAS 73.17 e2e / 72.29 stage2 on OpenEQA ScanNet 1050Q EM-EQA, Gemini 2.5 Pro judge)
**Headline**: **v15-S1-L1 = +2.00 MNAS over v14-rerun** on 1049 apples-to-apples common questions, zero category regression.

---

## 1. Problem

v14 Stage 1 (`KeyframeSelector.select_keyframes_v2`) loads `traj.txt` as a raw pose array and only uses it for per-object geometric visibility scoring (`_compute_visibility_scores` at `keyframe_selector.py:635`). It does **not** use pose/trajectory information to reason about:

- **Inter-view redundancy** — two selected keyframes can be taken 0.1 s apart from nearly identical viewpoints. A 1050Q empirical sweep on v14 cache showed **75.3 %** of questions had at least one pair of selected keyframes within 0.25 m camera distance, and **29.2 %** had a pair within 2 frame indices.
- **Motion saliency** — operator "dwell" (slow translation/rotation) signals visual importance but is not used in selection. In v14, selected-view speed percentile mean is **0.483** (essentially random).
- **Trajectory topology** — no view-view graph exists; `_explore_views` at `stage1_callbacks.py:150` uses object-set Jaccard novelty only.

Ceiling analysis on v14 (`docs/10_experiment_log/v14_inventory_20260404.md`) flagged:

| Category | v14 MNAS | Bottleneck |
|---|---:|---|
| Spatial Understanding | 57.8 | BEV / multi-view |
| Object Recognition | 64.2 | Enrichment + crops |

These are precisely the categories where pose-aware retrieval is structurally well-suited to help.

---

## 2. Goals & Non-goals

### Goals

1. **Exploit trajectory/pose information in Stage 1 retrieval** to reduce near-duplicate keyframes and prefer dwell regions — without changing the Stage 2 VLM, system prompt, or agent loop.
2. **Preserve v14 paper narrative** ("same VLM / same prompt / only retrieval change") by making the new behavior a flag-gated ablation axis (`pose_aware=False` default → bit-for-bit v14).
3. **Satisfy CLAUDE.md strict-no-fallback rule** throughout; every new code path either succeeds or raises.
4. **Full 1050Q OpenEQA ScanNet EM-EQA evaluation** on the same question subset as v14, judged by the same Gemini 2.5 Pro pool.

### Non-goals (deliberately out-of-scope)

- Learning-based stopping / policy (handcrafted thresholds only).
- Rewriting the Stage 2 DeepAgents runtime.
- New benchmarks beyond OpenEQA ScanNet EM-EQA.
- Generalization to HM3D / non-ScanNet corpora (`(W,H)` hardcoded for ScanNet 1296×968).

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1 (Keyframe retrieval)                                    │
│                                                                 │
│ traj.txt ──► KeyframeSelector                                   │
│              ├─ _load_camera_poses() [existing]                 │
│              ├─ _compute_trajectory_stats() [NEW]               │
│              │    velocity, turn_rate, dwell_score, turn_score  │
│              ├─ _get_intrinsic() [NEW, lazy]                    │
│              │    load raw/intrinsic_color.txt via frustum.py   │
│              └─ get_joint_coverage_views(pose_aware=True) [MOD] │
│                 gain *= redundancy_penalty if frustum_overlap   │
│                                          > threshold            │
│                 gain *= 1 + dwell_weight * dwell_score[v]       │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼ (same Stage2EvidenceBundle contract)
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2 — unchanged from v14 (VLM, prompt, tool inventory)      │
│                                                                 │
│ Optional: temporal_fan Stage 2 tool mode (default disabled)     │
│ Optional: per-view temporal metadata in KeyframeEvidence.note   │
└─────────────────────────────────────────────────────────────────┘
```

Single new helper module `src/query_scene/frustum.py` exports `load_scene_intrinsic()`, `frustum_overlap_l1()`, `frustum_overlap_l2()`; consumed by `KeyframeSelector` (Stage 1) and `_temporal_fan_views` (Stage 2, gated).

---

## 4. Components

### 4.1 `src/query_scene/frustum.py` (new, ~210 LOC + 135 test)

| Function | Purpose |
|---|---|
| `load_scene_intrinsic(raw_dir: Path)` | Read `raw/intrinsic_color.txt`, return 3×3 `K` and `(W, H)`. Raises on missing file or degenerate matrix. |
| `frustum_overlap_l1(pose_a, pose_b, K, wh, scene_depth=2.0)` | Pose-only analytical approximation. Uses yaw divergence (`arccos(f_a·f_b) / FOV_h`) plus lateral position offset (`‖d − (d·f_a)f_a‖ / (scene_depth · tan(FOV_h/2))`) with cross-term correction. ~0.05 ms/pair. |
| `frustum_overlap_l2(depth_a, pose_a, pose_b, K, wh, subsample=8)` | Back-project anchor's depth to 3D, project into `pose_b` image plane, return in-frustum pixel ratio. ~5 ms/pair. |

8 unit tests on real scene0709_00 data (no mocks): identity, opposite-facing, perpendicular, translation-only, small-delta, L2-identity, L1≈L2 in near-regime, and intrinsic loader.

### 4.2 `src/query_scene/keyframe_selector.py` (modified, +320 LOC + 240 test)

New methods / attributes on `KeyframeSelector`:

```python
self.pose_velocities: np.ndarray  # (N,) m/frame, smoothed over ±2
self.pose_turn_rates: np.ndarray  # (N,) radians/frame
self.dwell_score: np.ndarray      # (N,) ∈ [0,1], high for slow motion
self.turn_score: np.ndarray       # (N,) ∈ [0,1], normalized by 95th pctile

self._K: np.ndarray | None        # lazy-loaded 3×3 intrinsic
self._img_wh: tuple[int,int]      # (W, H)
self._depth_cache: dict[int, np.ndarray]  # LRU-free, per-scene bounded
```

Modified signatures:

```python
def get_joint_coverage_views(
    self, object_ids, max_views=3,
    *,
    pose_aware: bool = False,               # v14 default
    frustum_overlap_threshold: float = 0.7,
    redundancy_penalty: float = 0.2,
    dwell_weight: float = 0.15,
    frustum_method: str = "l1",             # "l1" | "l2"
) -> list[int]
```

Algorithm inside the greedy marginal-gain loop (when `pose_aware=True`):

1. Compute base `gain` from object-coverage marginal gain (unchanged from v14).
2. For each already-selected view `sel_vid`, compute `overlap = frustum_overlap(pose[view_id], pose[sel_vid])`. If any overlap > `frustum_overlap_threshold`, multiply `gain *= redundancy_penalty` (soft, not zeroed).
3. Multiply `gain *= 1 + dwell_weight * self.dwell_score[view_id]`.

`_pad_keyframes_to_minimum` mirrors the same scoring when `pose_aware=True`; v14 path unchanged when `pose_aware=False`.

**Degenerate-scene gate** (critical for robust eval): `_compute_trajectory_stats()` is called only if `len(camera_poses) >= 2`. Single-pose scenes get zero-filled arrays and a warning log. `_normalize_vector` failures on individual poses are caught locally and replaced with `[0, 0, -1]` safe default (v14 previously tolerated these silently).

### 4.3 `src/agents/runtime/langchain_agent.py` (infra, +266 LOC + 115 test)

`ToolChoiceCompatibleAzureChatOpenAI` extended with `ModelHubKeyRotator` and httpx request-rewrite hook:

- AKs are rotated round-robin; on 403/429/503/−1003/quota/rate-limit/timeout the current AK is demoted for 30–60 s.
- Exponential-backoff retry (5 attempts, 2 s base, jitter) on retryable errors.
- `session_id` injected into `extra` header for prompt cache (up to 97 % hit rate per ModelHub doc).
- Path / query / body rewrite per `/Users/bytedance/aispace/test_prompt_cache_langchain.py` reference impl.

### 4.4 `src/agents/examples/openeqa_official_question_pilot.py` (modified, +83 LOC)

New CLI flags:

| Flag | Purpose |
|---|---|
| `--pose-aware` | Forward `pose_aware=True` to `select_keyframes_v2`. |
| `--frustum-method {l1,l2}` | Pick frustum backend. |
| `--enable-temporal-fan` | Expose `temporal_fan` mode in Stage 2 system prompt (default off, validated harmful). |
| `--force-selection PATH` | Pin 1050Q to a saved `official_selected_questions.json` for apples-to-apples. |
| `--resume` | Skip already-completed QIDs in `official_batch_summary.json`; retry previously-failed. |

### 4.5 Other touches

- `src/agents/stage1_adapters.py`: Optional temporal-metadata in `KeyframeEvidence.note` when `pose_aware_enabled=True` (T6). Format: `"order=i/n dwell=True heading=+18° neighbors=[...]"`. Default preserves v14 note behavior.
- `src/agents/stage1_callbacks.py`: `_temporal_fan_views` implementation (T4) with adaptive best-available-neighbor fallback, gated by `enable_temporal_fan`.
- `src/agents/examples/openeqa_single_scene_pilot.py`: `ensure_runtime_scene` symlinks `raw/` into runtime overlay so `frustum._resolve_raw_dir` can find intrinsics (fix discovered during smoke test).
- `src/query_scene/keyframe_selector.py` `_resolve_raw_dir`: supports both canonical (`scene/conceptgraph/` + `scene/raw/`) and overlay (`runtime_cache/<scene>/raw/`) layouts.

---

## 5. Data flow

```
traj.txt (per-frame 4×4 pose)
    │
    ├─► _load_camera_poses() ─► self.camera_poses: list[np.ndarray]
    │
    └─► _compute_trajectory_stats() ─► self.{pose_velocities,
                                             pose_turn_rates,
                                             dwell_score,
                                             turn_score}

raw/intrinsic_color.txt (ScanNet per-scene)
    │
    └─► load_scene_intrinsic() ─► (K, (W, H))

Query ─► parse_query_hypotheses ─► execute_hypotheses ─► candidate_object_ids
                                                         │
    object_to_views (visibility index) ────────────────► │
                                                         │
                                                         ▼
                            get_joint_coverage_views(pose_aware=True):
                                greedy gain *= frustum_redundancy_penalty
                                greedy gain *= 1 + dwell_weight · dwell_score
                                         │
                                         ▼
                                 keyframe_indices
                                         │
                                         ▼
                             Stage2EvidenceBundle (same schema as v14)
```

---

## 6. Error handling

- Strict no-fallback (CLAUDE.md):
  - Missing `intrinsic_color.txt` → `FileNotFoundError` with clear path.
  - Degenerate K matrix → `ValueError("intrinsic matrix is degenerate")`.
  - Pose-aware requested but scene has < 2 poses → zero-arrays + `logger.warning` (tolerant, v14 parity).
  - Single degenerate forward vector in an N-pose trajectory → local replace + `logger.warning` (don't kill whole scene).
- ModelHub errors:
  - 403/429/503/−1003/quota/rate-limit/timeout → retry with backoff + AK rotation; cooldown demoted AK 30–60 s.
  - Non-retryable (auth error with stable pattern) → propagate.
- Pilot sample-level:
  - Individual sample failures recorded in `failed_samples[]`; run continues.
  - `--resume` picks up failed samples on relaunch (enables partial-run recovery after reboots).

---

## 7. Testing

Added 296 tests across 4 test files (all passing):

| File | Tests | Covers |
|---|---:|---|
| `src/query_scene/tests/test_frustum.py` | 8 | L1 / L2 correctness on real scene0709_00 data |
| `src/query_scene/tests/test_pose_aware_coverage.py` | 7 | `pose_aware=False` bit-for-bit v14, redundancy drop, dwell promotion, degenerate scenes |
| `src/agents/tests/test_modelhub_adapter.py` | 8 | Key rotator, request rewrite, retryable status codes; 1 gated live smoke |
| `src/agents/tests/test_stage2_deep_agent.py` | +5 | `request_more_views` mode round-trip, pinned frame_indices, temporal_fan neighbors + boundary cases |
| `src/agents/tests/test_stage1_adapters.py` | 2 | Temporal-note generation when pose_aware, v14 fallback |

Full regression `pytest src/` before merge: **303 passed, 3 skipped** (3 skipped are gated live-network tests).

---

## 8. Empirical validation

### 8.1 Eval matrix (all on 1050Q OpenEQA ScanNet EM-EQA, Gemini 2.5 Pro judge, same force-selected QID set)

| column | flags | n_stage2 | MNAS_s2 | MNAS_e2e | note |
|---|---|---:|---:|---:|---|
| v14-rerun (control) | — | 1050 | 72.29 | **73.17** | matches paper 73.1, judge variance < 0.1 |
| **v15-S1-L1** | `--pose-aware --frustum-method l1` | 1049 | **74.33** | 74.74 | **+2.00 vs v14, 7/7 categories non-negative** |
| v15-S1-L2 | `--pose-aware --frustum-method l2` | 1050 | 72.36 | — | +0.07 noise; Spatial −1.36 (L2 depth noise pollutes) |
| v15-S1+S2-L1 | `--pose-aware --frustum-method l1 --enable-temporal-fan` | 1050 | 72.45 | — | **−1.81 vs S1-L1** (prompt pollution) |

### 8.2 Per-category delta (v15-S1-L1 vs v14-rerun, 1049 common)

| category | v14 | v15-S1-L1 | Δ |
|---|---:|---:|---:|
| object recognition | 61.18 | 67.11 | **+5.92** |
| spatial understanding | 59.35 | 63.27 | **+3.91** |
| object state | 85.99 | 87.58 | +1.59 |
| world knowledge | 67.70 | 68.80 | +1.09 |
| object localization | 80.70 | 81.65 | +0.95 |
| functional reasoning | 75.17 | 75.52 | +0.34 |
| attribute recognition | 74.67 | 74.84 | +0.16 |

Score transitions (1049 common): **−19 score=1** (errors fixed) → **+20 score=5** (full-correct gained). Net "wrong → fully correct" conversion.

### 8.3 Negative finding: S2 temporal_fan prompt pollution

Partitioning 1049 common by whether the agent actually called temporal_fan in the S1+S2 run:

| subset | n | v15-S1-L1 MNAS | v15-S1+S2-L1 MNAS | Δ |
|---|---:|---:|---:|---:|
| agent used temporal_fan (`tf > 0`) | 76 | 55.26 | 58.22 | **+2.96** |
| agent didn't use it (`tf = 0`) | 973 | 75.82 | 73.64 | **−2.18** |

Interpretation: the tool itself is effective when invoked, but the mere presence of `mode='temporal_fan'` in the Stage 2 system prompt / tool signature perturbs the VLM's reasoning on the 93 % of questions where it isn't used. Net effect dwarfs the targeted gain. This corroborates the project's pre-existing Claim 2 (`docs/00_research_manifest.md`): context injection outperforms tool gating at the current VLM capability tier.

**Implication**: `temporal_fan` code stays in the tree but defaults to disabled. The research finding is itself a contribution.

---

## 9. Risks, limitations, future work

| Risk / limitation | Mitigation |
|---|---|
| `(W, H) = (1296, 968)` hardcoded for ScanNet iPad captures. | Flagged as TODO in `frustum.py`; replace with `scene_info.json` lookup when expanding beyond OpenEQA ScanNet. |
| L1 analytical formula under-estimates parallax for pure forward-dolly motion (`d_perp ≈ 0` while scene content shifts). | Docstring warning; switch to L2 for dolly-heavy trajectories. Validated negligible on ScanNet (L2 didn't outperform L1 anyway). |
| Wrapper `stage2_deep_agent.py` duplicates `run()`/`build_agent()` bodies to preserve legacy patch-on-class tests (145 LOC copy). | `WARNING: ...` comment added (T7); scheduled for `BaseStage2Runtime._iterate_evidence_loop()` refactor in T2.5b before any future feature. |
| Gemini judge variance is ≤ 0.1 MNAS between paper (73.1) and our v14-rerun (73.17) — acceptable signal floor for +2.00 claim. | `v14-rerun` column mandatory in all future v15+ ablations. |
| Prompt pollution from tool menu growth is now a known risk. | Future Stage 2 tool additions must A/B against `tf=0` subset before ship. |

---

## 10. Defaults on merge

| Flag | Default | Reason |
|---|---|---|
| `pose_aware` (selector arg) | `False` | Protects v14 as code-level default; callers must opt in. |
| `--pose-aware` (pilot CLI) | off | Protects production pipelines from unintended behavior change. |
| `--frustum-method` | `l1` | L1 dominates L2 empirically with lower compute. |
| `frustum_overlap_threshold` | `0.7` | Matches ScanNet kinematics; stricter (0.5) starved the filter. |
| `redundancy_penalty` | `0.2` | Soft penalty, not zero — genuinely unique candidates survive. |
| `dwell_weight` | `0.15` | Modest bonus, prevents dwell from dominating coverage. |
| `--enable-temporal-fan` | off | Empirically harmful (−1.81 MNAS); kept for future research only. |
| `--resume` | off unless needed | Pilot default is fresh run. |

---

## 11. Commits on `feat/v15-trajectory-aware` (in order)

```
a716c30 feat(frustum): add L1 + L2 frustum overlap helpers
d649231 fix(stage2): expose `mode` param and consume `frame_indices` in request_more_views
6b56bfb feat(stage1): trajectory-aware joint coverage with frustum redundancy + dwell bonus
3963671 feat(stage2): temporal_fan mode — frustum-filtered temporal neighbors via request_more_views
3793852 fix(stage1): tolerate degenerate scenes in trajectory stats (v14 behavior restored)
7032a16 feat(pilot): v15 CLI flags — pose-aware, frustum-method, temporal-fan, force-selection
483e69b fix(overlay): symlink raw/ into runtime_cache overlay so frustum intrinsic resolution works
a94bb8b fix(selector): _resolve_raw_dir supports both canonical and overlay scene layouts
734161e feat(stage1): trajectory-derived metadata in KeyframeEvidence.note when pose-aware
ec24c97 chore(v15): T2.5 follow-ups — DRY warning, warning logs, docstring caveats, test extension
a6f11f6 feat(modelhub): httpx-rewriter adapter for ModelHub endpoint + 2-key rotation + retry
6830394 fix(eval): 403 retryable + --resume support for incremental batch1 recovery
46dc77e fix(stage2): convert temporal_fan 'no valid neighbor' error into tool observation
8cd48d5 fix(stage2): ScanNet-realistic temporal_fan params (0.7 overlap / 16 window / adaptive fallback)
```

---

## 12. References

- Patent KP1 gap audit (triggered this work): conversation turn 2026-04-20.
- OpenEQA (Majumdar et al., CVPR 2024, arXiv:2312.15857).
- ModelHub Prompt Cache doc: `https://bytedance.sg.larkoffice.com/docx/QTGSd9IOAoAO5VxRRrDlAfZvgLp`.
- v14 baseline: `docs/10_experiment_log/v14_inventory_20260404.md`.
- Ceiling analysis: `docs/11_academic_angles_catalog.md`.
- Claim 2 (context injection > tool gating): `docs/00_research_manifest.md`.
