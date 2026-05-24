# NR3D BBox-Aware Spatial Execution Repair Design

Date: 2026-05-24
Branch: `analysis/nr3d-select-by-text-coverage`

## Context

The latest NR3D strat600 `select_by_text` audit has 106 empty outputs. The
empty-case audit shows that these are not primarily missing-data failures:
target proposals exist and have nonzero GT-visible frames. In 105 / 106 empty
rows, executor logs show a hard spatial constraint reducing candidates to zero
before keyframe ranking can run.

The immediate goal is not to solve every parser error. This design focuses on
cases where the parse is broadly correct, target and anchor candidates exist,
but `QueryExecutor` or `SpatialRelationChecker` falsely rejects a real spatial
relation.

## Goals

- Reduce `select_by_text` empty outputs caused by spatial false negatives.
- Make spatial relation checks bbox-aware instead of centroid-only where bbox
  evidence exists.
- Prevent quick filters from being the final reason a hard relation returns
  empty.
- Add enough executor trace metadata to distinguish parse errors from spatial
  checker false negatives.
- Preserve existing strict executor semantics by default, except where the
  relation implementation itself is objectively wrong or under-informed.

## Non-Goals

- Do not fix target-root parser mistakes in this phase.
- Do not implement negation, count, or full ordinal semantics in this phase.
- Do not globally soften every hard constraint.
- Do not change Stage 2 agent behavior.
- Do not change benchmark pack generation or GT visibility logic.

## Problem Breakdown

The empty cases fall into three broad classes:

1. Parser or schema mistakes: wrong root target, wrong anchor, lost negation,
   reversed relation, or unsupported comparative/count semantics.
2. Spatial execution false negatives: parse is plausible, candidates exist,
   but checker geometry or quick filtering wrongly clears them.
3. Proposal granularity mismatch: the query describes attributes or parts of a
   single proposal, but execution treats the phrase as an object-object
   relation.

This spec addresses class 2 first. Class 1 and class 3 stay visible in trace
metadata and can be handled in later parser / proposal-granularity phases.

## Proposed Architecture

### Phase 1: BBox-Aware Spatial Checker Repair

Add a single bbox access helper used by spatial relations:

- Accept `bbox_np` from `SceneObject`.
- Accept 8-corner bbox arrays by converting to `(min_xyz, max_xyz)`.
- Continue to support any existing `bbox_3d` object if present.
- Return `None` only when no usable bbox evidence exists.

Update high-frequency relations:

- `on_top_of`: use target bottom, anchor top, vertical gap, and XY footprint
  overlap or distance-to-footprint. Do not rely on target centroid being within
  a fixed 0.5 m radius of the anchor centroid.
- `above` / `below`: use bbox vertical ordering plus XY footprint proximity.
  Large supports such as beds, desks, couches, walls, and cabinets should not
  fail simply because centroids are far apart.
- `inside`: use target bbox when available, otherwise target centroid. Use
  anchor bbox. If anchor bbox is missing, report unknown/unsupported relation
  evidence rather than treating it as a confident false.
- `next_to` / `near`: use bbox-to-bbox distance, not centroid-to-centroid
  distance.
- `between`: evaluate all anchor pairs and keep the best score, instead of
  checking only the first two resolved anchors.

The checker should still return `RelationResult(satisfies=False, score=0.0)`
when evidence says the relation is false. It should include details such as
`bbox_used`, distances, vertical gap, overlap ratio, and reason.

### Phase 2: Quick Filter Safety

Quick filters stay as accelerators, not proof of failure.

For hard constraints:

- If quick filter keeps at least one candidate, run full checker on the reduced
  set.
- If quick filter would clear all candidates, do not immediately return empty.
  Run full checker on the original candidates and mark the trace with
  `quick_filter_would_empty=true`.

This preserves speed in common cases while preventing a loose pre-filter from
becoming a final false negative.

### Phase 3: Executor Trace Metadata

Add per-constraint trace entries to `ExecutionResult.metadata`.

Each trace entry should record:

- node id and root/anchor category strings
- relation, reference frame, and execution policy
- candidate ids before quick filtering
- anchor ids
- candidate ids after quick filtering
- candidate ids after full checking
- per-candidate best relation score and checker details when available
- whether quick filter would have emptied
- whether full checker emptied
- whether anchor resolution was empty or unknown

For audit outputs, persist full `hypothesis_output` and a compact copy of this
execution trace for all `no_evidence` and recovery cases.

### Phase 4: Strict-First Recall Fallback

After the spatial checker and trace repair are in place, add an explicit
execution mode interface:

```python
execute(query, mode=ExecutionMode.STRICT)  # default old behavior
execute(query, mode=ExecutionMode.RECALL)  # select_by_text fallback only
```

`STRICT` remains the default. `RECALL` is only used by `select_by_text` after
strict execution returns no evidence. In recall mode, root target candidates are
preserved when a spatial relation would otherwise clear them, and failed
constraints become ranking evidence rather than hard rejection.

This phase is a safety net, not the primary fix for spatial false negatives.

## Data Flow

Current flow:

```text
query
  -> parse_query_hypotheses()
  -> execute_hypotheses()
  -> QueryExecutor.execute()
  -> QueryExecutor._apply_spatial_constraint()
  -> SpatialRelationChecker.check()
  -> matched target objects
  -> get_joint_coverage_views()
```

Repaired flow:

```text
query
  -> parse_query_hypotheses()
  -> execute_hypotheses(mode=STRICT)
  -> QueryExecutor.execute(mode=STRICT)
  -> quick filter as non-final accelerator
  -> bbox-aware SpatialRelationChecker
  -> execution trace metadata
  -> matched target objects
  -> keyframe joint coverage
```

Later recall fallback:

```text
strict no_evidence
  -> execute_hypotheses(mode=RECALL)
  -> preserve root candidates when spatial constraints fail
  -> return recall status + trace
  -> keyframe joint coverage
```

## Error Handling

- Missing bbox should not crash relation checks.
- Malformed bbox arrays should be ignored with trace details rather than
  silently producing wrong geometry.
- Quick filter empty results should be traceable but not final.
- If both target and anchor categories genuinely have no candidates, strict
  execution can still return no evidence.
- Recall fallback must be opt-in and must mark metadata clearly, so benchmark
  reports can separate strict grounded hits from recall-preserved evidence.

## Testing Plan

### Unit Tests

Add focused tests for `SpatialRelationChecker`:

- `inside` works with `bbox_np`.
- `on_top_of` works when target is on a large anchor but centroids are far.
- `below` works using bbox top/bottom and footprint proximity.
- `near` and `next_to` use bbox-to-bbox distance.
- `between` searches all anchor pairs.

Add executor tests:

- Hard quick filter would empty candidates, but full checker can still recover.
- Trace metadata records quick-filter and full-check candidate ids.
- Strict default behavior remains unchanged when full checker truly rejects all
  candidates.

### Case Harness

Build a small read-only harness over representative empty cases:

- `scene0552_00::31::279`: stacked boxes / `on`.
- `scene0645_00::55::16871`: window next to bed / `next_to`.
- `scene0231_00::55::11667`: kitchen window / `inside`.
- `scene0307_00::25::35532`: cabinet with cloth / `below` or parse-anchor
  boundary.
- `scene0645_00::33::32439`: lamp/nightstand between beds / `between`.

For each case, assert:

- target candidate exists before relation
- anchor candidate exists where parse is valid
- if the relation is a spatial false negative, the repaired checker no longer
  clears the target
- if the relation is actually parse/proposal-granularity wrong, the trace labels
  that boundary instead of hiding it

### Benchmark Validation

After unit and case harness tests pass:

1. Run canonical NR3D strat600 with the existing benchmark process.
2. Compare empty rate, hit@3, and per-tier metrics against v15.
3. Report recovered empty cases split by attribution:
   spatial-checker repair, quick-filter repair, recall fallback, parser error,
   proposal-granularity mismatch.
4. Only run full NR3D after strat600 shows lower empty rate without unacceptable
   hit@3 regression.

## Risks

- Looser bbox-aware checks can increase false positives.
- Some cases labeled spatial false negatives may be parser or proposal
  granularity problems.
- Trace metadata can become large if it records every pair for large scenes.
- Compatibility duplicate files under `src/query_scene/retrieval/` can drift if
  only canonical code is updated.

## Mitigations

- Keep strict mode default.
- Add compact trace summaries to benchmark outputs and full trace only where
  needed.
- Write unit tests around both canonical and compatibility import paths if the
  legacy duplicate remains active.
- Use strat600 before full NR3D; do not claim gains smaller than the documented
  pilot variance bands.

## Success Criteria

- Targeted spatial false-negative cases no longer return empty for the checked
  relation.
- `inside` uses `bbox_np` correctly.
- Quick filter no longer directly causes hard no-evidence.
- Strat600 empty rate decreases from v15's 17.67%.
- Strat600 hit@3 is stable or improved relative to v15's 69.17%.
- Audit artifacts can explain every remaining empty as parser, proposal
  granularity, true no-target-candidate, or spatial checker failure.

## Implementation Order

1. Add relation trace scaffolding without changing behavior.
2. Add bbox helper and bbox-aware relation unit tests.
3. Repair `inside`, `on_top_of`, `above`, `below`, `next_to`, `near`, and
   `between`.
4. Change quick filter empty handling and test it.
5. Add case harness for representative empty cases.
6. Run strat600 and write benchmark docs.
7. Add `ExecutionMode.RECALL` fallback only after Phase 1 results are measured.
