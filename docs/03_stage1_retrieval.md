# 03 — Stage 1: Task-Conditioned Keyframe Retrieval

Deep dive into `src/query_scene/keyframe_selector.py`. All line numbers are against HEAD = `a8e651e`.

## 3.1 Public contract

**Entry**: `KeyframeSelector.from_scene_path(scene_path, stride, llm_model, use_pool)` → `KeyframeSelector`.
**Main method**: `selector.select_keyframes_v2(query: str, k: int = 3, strategy: str = "joint_coverage", hidden_categories=None, use_visual_context=True)` → `KeyframeResult`.
**Output** (`src/query_scene/keyframe_selector.py:185-218`):

```python
@dataclass
class KeyframeResult:
    query: str
    target_term: str
    anchor_term: str | None
    keyframe_indices: list[int]            # view_id list
    keyframe_paths: list[Path]             # resolved RGB paths
    target_objects: list[SceneObject]
    anchor_objects: list[SceneObject]
    selection_scores: dict[int, float]
    metadata: dict[str, Any]               # status, hypothesis_kind, hypothesis_rank, frame_mappings, hypothesis_output, version, ...
```

## 3.2 Scene loading order (exact)

`KeyframeSelector._load_scene` at `keyframe_selector.py:339-372`:

| # | Step | Source | Fails hard if missing? |
|---|---|---|---|
| 1 | Load objects from PCD | `pcd_saves/*ram*_post.pkl.gz` → `SceneObject` list with `clip_ft`, `pcd_np`, `bbox_np` (`_load_objects_from_pcd`, lines 374–459) | yes if no PCD file at all |
| 2 | Affordance enrichment | `sg_cache[_detect]/object_affordances.json` (optional) | no |
| 3 | **LLM enrichment** | `conceptgraph/enriched_objects.json` (`_load_enrichment`) | **YES — `FileNotFoundError`** at `:352-358` per v10 strict-no-fallback rule |
| 4 | Camera poses | `traj.txt` / pose overlays via `_load_camera_poses` | yes |
| 5 | Image paths | `results/frame*.jpg` under the runtime overlay | implicit via resolve failure |
| 6 | Multi-label category index | `_build_multilabel_categories()` — includes minority detection class names | no (defaults to per-object category) |
| 7 | Visibility index | `indices/visibility_index.pkl` (stride embedded in `metadata.stride`) | loaded lazily, regenerated if stale |

The loader is stride-aware: `infer_stride` reads `indices/visibility_index.pkl` metadata at `openeqa_single_scene_pilot.py:152-160` to match prep-time sampling; mismatched strides silently skew view-id → frame-id mapping, which is why the infer is mandatory before evaluation.

## 3.3 Hypothesis types (schema + semantics)

Formal schema: `schema/hypothesis_output_v1.json`. Python dataclasses: `src/query_scene/core/hypotheses.py`.

| Kind | When produced | Semantics | Example query | Retrieval behaviour |
|---|---|---|---|---|
| `DIRECT_GROUNDED` | Parser matched the literal target category to one or more scene objects with high confidence | "Exactly what you asked for is in the scene" | "the keyboard near the desktop computer" — `keyboard` matches directly | executes immediately; returns top-k joint-coverage views over matched targets + anchors |
| `PROXY_GROUNDED` | Literal target not found; a category-similar or hierarchy-related proxy exists (e.g., `monitor` → `screen`) | "The closest thing to what you asked for" | "the couch" when only `sofa` is labelled | uses the proxy as target; retains BETWEEN anchors (commit `9be20f1`) |
| `CONTEXT_ONLY` | Neither literal nor proxy found; only the spatial anchor(s) and context are usable | "Best we can do is show the neighbourhood" | "what is behind the vacuum cleaner?" when the vacuum isn't labelled | falls through to open-ended mode (v11 `02ea2f3`); returns views weighted by anchor + spatial constraint |

`ParseMode` in the same module distinguishes `SINGLE_HYPOTHESIS` (one candidate) from `MULTI_HYPOTHESIS` (rank-ordered list of up to 3). Both modes feed the same executor.

## 3.4 `select_keyframes_v2` algorithm (step-by-step)

Source: `keyframe_selector.py:1570-1731`. Pseudocode aligned to the actual code.

```python
def select_keyframes_v2(query, k=3, strategy="joint_coverage", ...):
    # Step 1 — parse
    hypothesis_output = self.parse_query_hypotheses(
        query, max_hypotheses=3, use_visual_context=use_visual_context
    )                                             # line 1593

    # Step 2 — ranked execution
    status, selected_hypothesis, result = self.execute_hypotheses(
        hypothesis_output=hypothesis_output,
        hidden_categories=hidden_categories,
    )                                             # line 1603
    if status == "no_evidence" or result.is_empty:
        return empty_result_with_metadata(...)   # line 1611

    target_objects = result.matched_objects      # line 1627

    # Step 3 — collect anchors + between-midpoint objects
    anchor_objects, between_anchors = [], []
    for constraint in selected_hypothesis.grounding_query.root.spatial_constraints:
        for anchor_node in constraint.anchors:
            anchor_objects.extend(
                self._get_query_executor()._execute_node(anchor_node).matched_objects
            )
        if constraint.relation.lower() == "between" and len(anchor_objects) >= 2:
            between_anchors = anchor_objects[-2:]
                                                 # lines 1635-1644

    midpoint_object_ids = []
    if len(between_anchors) >= 2:
        c1, c2 = between_anchors[0].centroid, between_anchors[1].centroid
        midpoint = (c1 + c2) / 2
        midpoint_object_ids = [
            oid for oid, dist in sorted(
                ((o.obj_id, norm(o.centroid - midpoint)) for o in self.objects),
                key=lambda x: x[1]
            )[:5]
        ]                                        # lines 1645-1665 (v12 2abb404)

    # Step 4 — joint-coverage view selection
    all_object_ids = (
        [o.obj_id for o in target_objects[:5]] +
        [o.obj_id for o in anchor_objects[:3]]   +
        midpoint_object_ids
    )
    keyframe_indices = self.get_joint_coverage_views(all_object_ids, max_views=k)

    # Step 5 — pad to minimum (v11 02ea2f3)
    keyframe_indices = self._pad_keyframes_to_minimum(
        keyframe_indices, all_object_ids, min_count=min(k, 3)
    )                                            # line 1684

    # Step 6 — resolve view_id → frame_id → image path
    keyframe_paths, frame_mappings = [], []
    for view_id in keyframe_indices:
        requested_fid = self.map_view_to_frame(view_id)
        path, resolved_vid = self._resolve_keyframe_path(view_id)
        resolved_fid = self.map_view_to_frame(resolved_vid)
        if path: keyframe_paths.append(path)
        frame_mappings.append({...})

    return KeyframeResult(..., metadata={
        "status": status,
        "selected_hypothesis_kind": selected_hypothesis.kind.value,
        "selected_hypothesis_rank": selected_hypothesis.rank,
        "frame_mappings": frame_mappings,
        "hypothesis_output": hypothesis_output.model_dump(),
        "version": "v3",
    })
```

### Step-wise guarantees

| Step | Guarantee | Enforced by |
|---|---|---|
| 1 | Parser output conforms to `HypothesisOutputV1` | `HypothesisOutputV1.model_validate()` inside `parse_query_hypotheses` |
| 2 | Execution walks candidates by `rank` ascending; commits on first non-empty `ExecutionResult` | `execute_hypotheses` |
| 3 | BETWEEN anchors are preserved even when the hypothesis is `PROXY_GROUNDED` | commit `9be20f1` |
| 4 | `get_joint_coverage_views` returns ≤ `k` views optimising a greedy coverage score | `keyframe_selector.py:get_joint_coverage_views` |
| 5 | **Minimum** `min(k, 3)` keyframes are returned when any candidate view exists | `_pad_keyframes_to_minimum` (v11) |
| 6 | `_resolve_keyframe_path` tries neighbouring view_ids if the exact frame file is missing, so the returned index may differ from requested | resolution helper |

## 3.5 Joint-coverage view selection

`get_joint_coverage_views(object_ids, max_views)`:
- Input: union of target + anchor + midpoint object ids.
- Per view, compute a coverage score = fraction of object_ids visible × softmax over per-object visibility.
- Greedy pick without replacement up to `max_views`; ties broken by larger visible bbox area.
- Views come from `view_to_objects` built during `_load_or_build_visibility_index`.
- Strategy `joint_coverage` uses target ∪ anchor ∪ midpoint. Strategy fallback (`else` branch, line 1679) uses only targets.

## 3.6 Pad-to-minimum logic

`_pad_keyframes_to_minimum(selected, all_object_ids, min_count)`:
- If `len(selected) >= min_count`: no-op.
- Else: rank all views in `view_to_objects` by visibility-weighted score over `all_object_ids`; append until `min_count` is reached, skipping already-selected views.
- Rationale: v10 had 18/36 low-score cases caused by a single selected keyframe that did not happen to show the target. Padding with the next-best candidate reduced this specific failure mode to near zero (`10_experiment_log/v11_callbacks_20260330.md`).

## 3.7 LLM query rewrite integration (v9)

The production pilot (`openeqa_official_question_pilot.py:362-448`) layers a *query-rewrite ranking* step above `select_keyframes_v2`:

1. `build_stage1_query_candidates(question)` produces heuristic rewrites (strip `what/where`, retain spatial relations).
2. If `--llm-rewrite` is set, `llm_rewrite_qa_to_retrieval(question)` via Gemini-2.5-pro prepends additional retrieval-oriented queries.
3. `run_stage1_ranked` invokes `select_keyframes_v2` for each candidate; returns immediately on first `DIRECT_GROUNDED`, else keeps the best by `_GROUNDING_RANK = {direct_grounded: 0, proxy_grounded: 1, context_only: 2}`.

Quantitative: `--llm-rewrite` raised `direct_grounded` rate 26 % → 33 % on the 30-scene fold (`b6a8aa6` commit message). The ranked loop is the only reason a single Stage 1 call can return a typed recall-mode signal rather than a fail-silent empty list.

## 3.8 Visibility index + stride semantics

- `indices/visibility_index.pkl` is built at scene prep time; key is `view_id = frame_id // stride`.
- `map_view_to_frame(view_id)` = `view_id * stride` (approximately; actual implementation in `KeyframeSelector.map_view_to_frame`).
- If the pilot passes `--stride 0`, `infer_stride` inspects the pickle metadata. A wrong stride silently offsets every frame path — always defer to `infer_stride` over guessing.

## 3.9 Failure modes surfaced to downstream

A successful `KeyframeResult` may still carry `status == "no_evidence"` if every ranked hypothesis yielded an empty `ExecutionResult`. In that case `keyframe_paths == []`. The pilot treats this as a hard retrieval failure and raises at `run_stage1_ranked:440`. Stage 2 is never invoked with zero keyframes.

## 3.10 Points that Claim 5 (manifest) rides on

- The typed rank is exposed to Stage 2 via `inspect_stage1_metadata` (`src/agents/runtime/deepagents_agent.py:94-108`).
- The system prompt instructs the agent to consult it (`src/agents/runtime/base.py:355-362`).
- No ablation yet measures *whether the agent actually changes behaviour* based on `hypothesis_kind`. This is an open REQUIRES (see `00_research_manifest.md` Claim 5).

Cross-ref: `06_evolution_v9_to_v14.md` — which of the above were v11 / v12 / v14 additions.
