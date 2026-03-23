# Query Scene Memory

## Current Role

`src/query_scene/` is the Stage 1 package. Its job is evidence retrieval, not final reasoning. It turns a natural-language query plus scene assets into:
- a structured parse (`HypothesisOutputV1`)
- a selected executable hypothesis (`direct` / `proxy` / `context`)
- matched scene objects
- a small set of keyframes for Stage 2 or downstream evaluation

The practical Stage 1 entrypoints are:
- `query_scene.keyframe_selector.KeyframeSelector`
- `query_scene.run_query_with_dataset(...)`
- `query_scene.query_pipeline.QueryScenePipeline`

## Canonical Data Model

The canonical parsing schema is `HypothesisOutputV1` in `src/query_scene/core/hypotheses.py`.

Current schema reality:
- `HypothesisKind` values are `direct`, `proxy`, `context`
- `ParseMode` values are `single`, `multi`
- `GroundingQuery` uses `root`
- `QueryNode` uses `categories: list[str]`
- `SpatialConstraint` uses `anchors: list[QueryNode]`
- `SelectConstraint.reference` is optional and used for distance/superlative cases

Ignore any older notes that mention:
- `simple` / `spatial` / `superlative` as hypothesis kinds
- `object_category` as the canonical field
- `target` instead of `root`
- a single `anchor` instead of `anchors`

Those notes are stale and do not match the current repository.

## Parse To Selector Chain

### 1. Entry
`KeyframeSelector.select_keyframes_v2()` in `src/query_scene/keyframe_selector.py` is the main Stage 1 retrieval entrypoint.

High-level flow:
1. `parse_query_hypotheses()`
2. `execute_hypotheses()`
3. collect target/anchor objects from the winning hypothesis
4. choose views with `get_joint_coverage_views()`
5. resolve actual frame paths and return `KeyframeResult`

### 2. Parse stage
`KeyframeSelector.parse_query_hypotheses()` does the following:
- gets or lazily creates `QueryParser`
- optionally generates visual context through `_generate_scene_images()`
- calls `QueryParser.parse(query, scene_images=...)`
- sanitizes every hypothesis so categories are either scene categories or `UNKNOW`
- validates the final `HypothesisOutputV1` against current `scene_categories`

Important implementation details:
- `QueryParser` lives at `src/query_scene/parsing/parser.py`
- prompts and few-shot examples come from `src/query_scene/parsing/structures.py` (`get_system_prompt()` and `get_few_shot_examples()`)
- for Gemini models, parsing uses JSON text output and `_parse_json_response()` for local validation, not `with_structured_output(...)` (see `_do_parse_with_llm()` L439-463)
- for non-Gemini models without images, parsing uses `with_structured_output()` with dynamically-built schema (L487-491)
- for non-Gemini models with images, parsing falls back to JSON text output (L467-485)
- when `use_pool=True` and model contains `gemini`, `parse()` delegates to `_parse_with_pool_retry()` which rotates over `GeminiClientPool` keys on rate limit errors (L329-330, L362-419)

### 3. Visual context path
`parse_query_hypotheses()` currently enables visual context by default (`use_visual_context=True`).

Actual behavior:
- `_generate_scene_images()` caches a BEV image under `scene_path/bev/scene_bev_<hash>.png`
- the generated image is mesh-background BEV only and does not add object labels or circle markers
- **BUT** the parser prompt in `parsing/structures.py` (L122-131) tells the model:
  > "Each object is shown as a labeled circle at its centroid position"
  > "Labels follow format 'NNN: category' (e.g., '001: sofa', '002: pillow')"

This prompt/image mismatch is a **confirmed** current risk. The prompt describes labeled markers that the actual BEV image does not contain. This can cause Gemini to hallucinate object positions or give inconsistent parses across runs.

### 4. Parser output
`QueryParser.parse()` returns `HypothesisOutputV1`.

Expected parser behavior:
- `single` mode: exactly one `direct` hypothesis
- `multi` mode: ranked fallback chain such as `direct -> proxy -> context`
- every hypothesis contains one executable `GroundingQuery`
- node ids are assigned after parse via `_assign_node_ids()` for execution tracing (L493-508)
- node id format: `h{rank}_root`, with nested nodes like `h1_root_sc0_a0` (spatial constraint 0, anchor 0)

### 5. Hypothesis execution stage
`KeyframeSelector.execute_hypotheses()` is the selector-side decision point.

Execution behavior (keyframe_selector.py L1321-1359):
1. normalize payload to `HypothesisOutputV1` via `normalize_hypothesis_output()`
2. validate categories against scene categories
3. sort by rank with `ordered_hypotheses()`
4. for each hypothesis in rank order:
   a. convert to `GroundingQuery` via `to_grounding_query()`
   b. validate categories in scene
   c. validate no hidden-category leak via `validate_no_mask_leak()` (if `hidden_categories` provided)
   d. **skip** if `_has_unknown_anchors()` returns True (anchors or select references contain `UNKNOW`)
   e. execute via `execute_query()`
   f. return first non-empty result

Returned status values are:
- `direct_grounded`
- `proxy_grounded`
- `context_only`
- `no_evidence`

Interpretation:
- `direct_grounded`: literal parse succeeded
- `proxy_grounded`: fallback semantic substitution succeeded
- `context_only`: only coarse context fallback produced evidence
- `no_evidence`: all executable hypotheses failed or were skipped

### 6. Query execution internals
`KeyframeSelector.execute_query()` delegates to `QueryExecutor` in `src/query_scene/query_executor.py`.

`QueryExecutor.execute()` behavior:
- recursively executes from `GroundingQuery.root`
- if `expect_unique=True`, collapses multi-match outputs to `best_object`

`QueryExecutor._execute_node()` pipeline:
1. find candidates by category
2. filter by attributes
3. apply spatial constraints with AND logic
4. apply optional select constraint

Category retrieval behavior in `_find_by_categories()` (query_executor.py L250-310):
- exact category match first (case-insensitive)
- substring fallback second (bidirectional: query in scene OR scene in query)
- CLIP fallback only if `clip_features` and `clip_encoder` are both available
- **Note**: CLIP fallback only uses `categories[0]` (the first category in the list), not all categories

Spatial constraint behavior in `_apply_spatial_constraint()` (query_executor.py L379-475):
1. recursively resolve anchor nodes first via `_execute_node()`
2. if anchor resolution is empty:
   - `strict_mode=True`: return empty (hard failure)
   - `strict_mode=False` (default): return all current candidates as lenient fallback
3. if relation has a quick filter (`QuickFilters.has_filter()`), pre-filter candidates
4. if quick filter removes everything, fall back to the original candidate list
5. run full geometric relation checks with `SpatialRelationChecker.check()`
6. special case: "between" relation passes first 2 anchors together (L454-459)

This means selector behavior is intentionally recall-biased. It prefers fallback behavior over hard failure in several places.

### 7. Keyframe selection after hypothesis win
Once one hypothesis wins in `select_keyframes_v2()`:
- `target_objects = result.matched_objects`
- anchor objects are re-collected from `selected_query.root.spatial_constraints`
- object ids are truncated before view selection
  - up to 5 target objects
  - up to 3 anchor objects
- `get_joint_coverage_views()` is used for joint coverage retrieval
- `_resolve_keyframe_path()` maps sampled view ids back to actual frame files using stride-aware fallback
- `KeyframeResult.metadata` stores selected hypothesis kind/rank, frame mappings, and the full serialized `hypothesis_output`

## Current Practical Conclusions

### Current dominant failure mode
For current Replica room0 E2E behavior, the main instability is parser-side, not a clean selector regression.

Evidence from current investigation:
- current repo baseline backup: `77/96` non-empty hits
- original repo comparison run: `88/96` non-empty hits
- same current code can produce different parses on repeated Gemini runs for the same query
- selector core functions are effectively aligned with the original migrated logic

### Why parse drifts
The main sources of drift are:
- Gemini non-deterministic JSON generation
- rotating multi-key / multi-endpoint Gemini pool
- visual-context prompt claims that do not match the actual BEV image content

### What to trust during debugging
If comparing selector behavior across branches or repos:
- do not rely only on full E2E reruns
- first save the parsed `HypothesisOutputV1`
- then feed the same parse into both selectors
- only then judge selector equivalence

## Current Verification State

**Last verified: 2026-03-22** against commit tree.

Verified locally during the migration/debug cycle:
- `src/query_scene/tests` plus `src/agents/tests` passed at `108 passed, 3 skipped`
- current canonical schema/runtime are aligned around `core/hypotheses.py`, `parsing/parser.py`, and `keyframe_selector.py`
- the current repo and original repo are not yet behavior-identical in E2E Gemini runs

Known current caveats:
- parser output for the same query is not stable enough for strict E2E parity claims
- BEV prompt description (labeled circles with IDs) does not match actual generated BEV image (mesh only)
- selector remains high-recall oriented and will use lenient fallback paths in executor/quick-filter stages
- CLIP fallback only considers first category in `categories` list

## Maintenance Guidance

When updating Stage 1 docs or code, preserve this mental model:
- parser decides the executable hypothesis set
- selector decides which hypothesis wins
- executor is recall-biased and permissive by design
- Stage 1 should be debugged as `parse` and `selector/executor` separately

If behavior changes again, re-check in this order:
1. parser prompt and schema
2. visual context image generation
3. Gemini pool/client routing
4. hypothesis sanitization and `UNKNOW` skipping
5. executor quick-filter and relation-check behavior
6. final joint-coverage view selection
