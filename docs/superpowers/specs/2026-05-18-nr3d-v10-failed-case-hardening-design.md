# NR3D v10 Failed-Case Guard And Tool-Flow Hardening

**Date:** 2026-05-18
**Status:** Draft — awaiting user spec review.
**Parent design:** `docs/superpowers/specs/2026-05-18-remove-initial-keyframes-design.md`
**Benchmark evidence:** `docs/benchmark/nr3d/v10_no_initial_keyframes_strat600_20260518.md`
**Seed run:** `tmp/nr3d_eval_no_initial_keyframes_strat600_20260518_220f128/`
**Current prep commits:** `8f278bc`, `30c3b16`

## 1. Summary

The no-initial-keyframes v10 run is stable but loses accuracy on NR3D strat600:
64.33 % overall, with 214 / 600 metric misses. The failures are not mostly
script crashes or missing scene assets. They are agent-flow failures caused by
weak contracts between query semantics, selector tools, spatial comparison, and
final-answer guards.

This spec hardens the v10 Stage-2 VG flow after the first 40 failed-case audits
(15 initial cases + 25 additional xhigh subagent cases). The target is not to
restore any initial keyframe path. First-person evidence must still be acquired
only through active tools. The target is to make those tool/guard paths less
misleading:

- recover from `select_by_text` masked-category errors without semantic drift;
- keep the final answer's category aligned with the query's referent head;
- bind TADG/EFG to the relation evidence actually cited for the final answer,
  not whichever compare call happened most recently;
- prevent EFG from interpreting contrastive rationale text as the target
  direction;
- require candidate coverage when same-category contenders are split across
  frames;
- route unsupported semantic relations through explicit visual/BEV workflows
  instead of unsupported tool calls followed by ad-hoc guessing.

The recommended implementation is staged. Phase 1 handles low-risk tool and
guard-contract fixes. Phase 2 introduces explicit relation-evidence binding.
Phase 3 adds a lightweight constraint planner for multi-clause relations.

## 2. Evidence From Audits

### 2.1 Existing trace-level counts

The v10 strat600 record shows these trace patterns:

| Pattern | Count / affected cases | Wrong among affected | Reading |
|---|---:|---:|---|
| Unsupported `compare_proposals_spatial` relation | 105 tool errors / 93 cases | 24 | Agent relation vocabulary was wider than the tool contract. |
| `select_by_text` masked-category guard | 39 tool errors / 33 cases | 19 | Agent often hid anchors/supports such as wall, floor, bed, door. |
| Evidence-frame guard block | 135 blocks / 89 cases | 54 | Guard catches real citation problems, but can steer to wrong ids. |
| TADG block | 15 blocks / 12 cases | 7 | Mostly target/anchor role conflicts or stale compare binding. |
| No-match guard block | 2 blocks / 2 cases | 1 | Rare; not the primary target. |

The first patch set already addressed two low-risk issues:

- `8f278bc`: canonicalized common spatial relation aliases and clarified
  playbooks around `hidden_categories`.
- `30c3b16`: category-gated EFG relative-position alternatives so a target
  table is not replaced by a couch merely because the couch is farther right.

The second xhigh audit shows those patches are necessary but not sufficient.

### 2.2 Additional 25-case audit findings

| Failure class | Representative cases | Root issue | Covered by current commits? |
|---|---|---|---|
| Masked-category recovery | `scene0222_00::7::10110`, `scene0565_00::23::30788`, `scene0647_00::17::34181` | `select_by_text` returns an error and leaves retry discipline to the agent; retries sometimes rewrite the target category. | Partially. Prompt says retry, tool does not auto-retry. |
| Target-head/category drift | `scene0221_00::46::36120`, `scene0490_00::13::36559`, `scene0496_00::29::10819` | Agent submits a context object (`bed`, `chair`, `wall`) rather than the referent head (`pillow`, `whiteboard`, `window`, `cart`). | Not covered. |
| TADG stale or inverted compare | `scene0221_00::13::4514`, `scene0378_00::41::26109`, `scene0249_00::36::39983` | Guard uses a compare where the submitted target was actually the anchor, or where the compare candidates are anchors/supports. | Partially. Message now asks for role repair, but binding is still brittle. |
| EFG direction parsing | `scene0208_00::17::21496` | Query says target is on the right, but rationale mentions a left-hand alternative; guard infers the target should be left. | Not covered. |
| EFG citation parsing | `scene0249_00::23::30460` | Rationale cites "Frames 0, 1, and 2"; current parser can miss plural/range citations, so marked-frame requirements can be bypassed. | Not covered. |
| Unsupported semantic relations | `same_side_as`, `between`, `opposite`, `facing`, viewpoint-relative left/right, `NOT closer` | These are not simple aliases. They need visual/BEV workflows or new relation helpers. | Not covered by alias normalization. |
| Candidate coverage | `scene0081_00::0::10619`, `scene0246_00::39::37886` | Agent finalizes using only the visible candidate cluster and never marks same-category candidates that selectors failed to surface. | Not covered. |
| Attribute / superlative ranking | `scene0647_00::17::34181`, `scene0329_00::6::26630`, `scene0644_00::35::19090` | Agent overweights proximity/visibility and underchecks size, color, width, or material. | Not covered. |

## 3. Goals

1. Preserve the v10 no-initial-keyframes contract. No pack-prepared
   first-person RGB frames may be added back.
2. Make tool/guard failures self-healing where the intended recovery is
   deterministic, especially masked-category leaks.
3. Add final-answer guardrails for target category and relation evidence without
   silently using GT target ids.
4. Reduce guard-induced wrong answers by requiring guards to ask for corrected
   evidence instead of forcing a stale or role-inverted rank-1 id.
5. Keep each phase measurable with focused unit tests and a small replay slice
   before any new strat600 run.

## 4. Non-Goals

- Do not add or restore `initial_keyframe_paths`, `Stage2EvidenceBundle.keyframes`,
  or any seed-frame drain.
- Do not introduce GT target id, GT target category, or sample answer metadata
  into runtime decisions. Target-category checks must derive from the query,
  SceneCatalog labels, and the agent-visible candidate lists.
- Do not attempt a full natural-language semantic parser in this pass.
- Do not add a large black-box planner before fixing the deterministic guard
  bugs that already have clear unit-test targets.
- Do not claim leaderboard improvement until the canonical strat600 fold is
  rerun and ingested.

## 5. Design Alternatives

### Option A — Prompt-only hardening

Update playbooks to tell the agent to retry masked-category errors, preserve the
target head noun, mark all candidates, and avoid unsupported relation strings.

Pros:

- Minimal code risk.
- Cheap to iterate.

Cons:

- Already insufficient. The masked-category audit showed the agent can rewrite
  the query after an error and drift from `window` to `chair`, or from `cart` to
  `wall`.
- Guards still use stale compare state even when prompts are clear.

Verdict: useful as support text, not enough as the main fix.

### Option B — Guard/tool-contract hardening first, planner later

Add deterministic recovery and validation at the tool/guard boundaries:

- `select_by_text` auto-retries masked leaks with empty masks;
- final submissions are checked against target-head/category evidence;
- TADG/EFG use explicit relation-evidence handles instead of latest compare;
- EFG direction and frame-citation parsing are fixed;
- playbooks require candidate coverage before final.

Pros:

- Targets the highest-confidence failure modes.
- Keeps changes unit-testable and interpretable.
- Produces cleaner traces for later planner work.

Cons:

- Does not fully solve `between`, `opposite`, `same_side_as`, or complex
  negation.

Verdict: recommended.

### Option C — Full constraint planner now

Introduce a query-clause planner that extracts target head, candidate set,
anchors, positive constraints, negative constraints, and unsupported semantic
relations before any final answer.

Pros:

- Directly attacks multi-clause failures.
- Could unify prompt, TADG, EFG, and comparison evidence.

Cons:

- Larger blast radius.
- Harder to tell whether any strat600 delta comes from parser quality, guard
  quality, or tool behavior.
- Risky immediately after the no-initial-keyframes cleanup.

Verdict: defer until Phase 3, after deterministic guard gaps are closed.

## 6. Target Contracts

### 6.1 `select_by_text` masked-leak auto-retry

Current behavior:

- If Stage-1 raises `Masked category leak detected`, the tool returns an error
  and the agent decides what to do next.

Target behavior:

- If `hidden_categories` is non-empty and the exception text contains
  `Masked category leak detected`, `select_by_text` automatically retries the
  exact same `query` once with `hidden_categories=[]`.
- The response records both attempts:
  - original hidden categories;
  - masked-leak error text;
  - `retried_with_hidden_categories=[]`;
  - final frame payload or final error.
- The retry must not rewrite the query.
- If the empty-mask retry also fails, return a fail-loud error with both
  exception summaries.

Expected impact:

- Covers cases where `wall`, `floor`, `door`, `bed`, or `desk` masks killed
  useful anchor/support retrieval.
- Prevents semantic drift such as retrying `rightmost chair when facing windows`
  when the original target was a window.

### 6.2 Target-head/category consistency before final

Current behavior:

- `submit_final(proposal_id=N)` validates schema and guard state, but does not
  know whether the selected proposal category matches the referent head.

Target behavior:

- Add a lightweight target-category checkpoint for VG final submissions.
- Derive the expected target category from agent-visible sources, in this order:
  1. explicit `SceneCatalog` candidate lists or `list_scene_proposals` calls the
     agent used for the target head;
  2. a small lexical head-noun matcher over the raw query and visible proposal
     categories;
  3. no check when the head is ambiguous or absent.
- If the submitted proposal category is incompatible with a confident expected
  target category, soft-block with a message like:

```text
TARGET_CATEGORY_GUARD: query target appears to be 'pillow', but submitted
proposal 8 is category 'bed'. Submit a pillow proposal or explain an explicit
referent shift with marked evidence.
```

Compatibility rules:

- Use the same compact label alias style as EFG (`bookcase` == `bookshelf`,
  `white board` == `whiteboard`).
- Allow substring-compatible labels such as `trash can` and `trashcan`.
- Do not block when the query genuinely refers through a pronoun and the head is
  ambiguous.
- Do not use GT category fields hidden in benchmark metadata.

Expected impact:

- Blocks `pillow -> bed`, `cart -> wall`, `whiteboard -> chair`,
  `window -> chair`, and similar context-object submissions.

### 6.3 Relation evidence binding for TADG and EFG

Current behavior:

- TADG walks the recent trace and picks the latest compare whose relation
  matches the query.
- EFG uses the latest compare that includes the submitted id.
- Both can bind to stale compares or role-inverted compares.

Target behavior:

- `compare_proposals_spatial` returns a stable `evidence_id`, derived from the
  runtime trace index or a deterministic per-call counter.
- The compare payload includes:
  - `evidence_id`;
  - `relation`;
  - `anchor_id`;
  - `candidate_ids`;
  - `ranked_ids`;
  - `requested_relation`;
  - `canonical_relation` when an alias was normalized.
- `submit_final` accepts optional relation evidence metadata:

```python
submit_final(
    payload={"proposal_id": 13},
    rationale="...",
    evidence_refs=[{"frame_id": 0}],
    relation_evidence={
        "evidence_id": "compare_17",
        "relation": "next_to",
        "anchor_id": 26,
        "candidate_ids": [13, 24, 30],
    },
)
```

- TADG/EFG prefer `relation_evidence` when supplied.
- If absent, they may fall back to trace inference, but block messages should
  ask the agent to rerun/attach explicit relation evidence rather than force a
  stale top-1 answer.
- If `submitted_pid == anchor_id`, treat it as a role error and request a
  corrected compare. Do not recommend the rank-1 id unless the corrected compare
  supports it.
- If `candidate_ids` are not category-compatible with the target head, treat the
  compare as anchor/support evidence, not target-ranking evidence.

Expected impact:

- Prevents stale red-chair compare from controlling a door query.
- Prevents chair-candidate compares from forcing a trash-can target to change.
- Makes guard blocks actionable: rerun with corrected roles, not "revise to
  rank-1".

### 6.4 EFG direction and frame-citation parsing

Current behavior:

- `_desired_relative_direction` catches `left of` / `right of`, but misses
  common target-side phrasing such as `on the right`, `right one`, `right side`.
- If the query direction is missed, rationale text about an alternative can
  become the target direction.
- Frame citations are mostly singular; "Frames 0, 1, and 2" may not produce the
  intended cited-frame set.

Target behavior:

- Query-side direction has precedence over rationale-side direction.
- Add patterns for:
  - `on the left/right`;
  - `left/right one`;
  - `left/right option`;
  - `left/right side`;
  - `upper/lower left/right` where only horizontal direction is needed.
- When rationale mentions an alternative, do not infer target direction from a
  clause such as `proposal 16 is the left-hand alternative`.
- Parse plural and range frame citations:
  - `frames 0, 1, and 2`;
  - `frames 0-2`;
  - `frame ids 58 and 59`.

Expected impact:

- Prevents the `scene0208_00::17::21496` failure where a correct right-table
  submission was changed to the left table.
- Prevents direct-evidence claims from bypassing marked-frame guard by citing
  plural frames.

### 6.5 Candidate coverage before final

Current behavior:

- The agent can select from whichever same-category candidates were visible in
  the first returned frames, while silently dropping candidates not present in
  that evidence batch.

Target behavior:

- Add a prompt/guard rule: when the final answer depends on ordering,
  closest/farthest, size, or superlative among same-category candidates, the
  trace must show either:
  - marked evidence for every plausible same-category candidate in a small
    candidate set; or
  - explicit elimination evidence for candidates not marked.
- For candidate sets larger than a small threshold, require at least coverage of
  each spatial cluster or top-ranked subset from the relevant tool.
- EFG/TADG block messages should say "mark the omitted candidate(s)" instead of
  implying the visible candidate is necessarily correct.

Expected impact:

- Addresses couch and pillow failures where the target candidate never appeared
  in the marked evidence that justified the final answer.

### 6.6 Unsupported semantic relations

Current behavior:

- The agent calls `compare_proposals_spatial` with unsupported relation strings
  such as `same_side_as`, `between`, `opposite`, `in_front_of`, `facing`, then
  falls back ad hoc.

Target behavior:

- Keep `compare_proposals_spatial` limited to stable canonical relations:
  `closest_to`, `near`, `next_to`, `farthest_from`, `above`, `below`,
  `left_of`, `right_of`.
- Playbooks and guard messages route unsupported semantic relations to explicit
  workflows:
  - `between`: mark target and both anchors; use BEV/3D positions to check
    interpolation or enclosure.
  - `opposite` / `across`: mark the pair and use BEV room-side evidence.
  - `same_side_as`: use BEV room partition or anchor-relative side grouping.
  - `facing`: use first-person appearance and proposal orientation cues where
    available; do not pretend `next_to` proves facing.
  - viewpoint-conditioned `left/right`: establish viewpoint anchor first, then
    evaluate image/BEV direction in that viewpoint.
  - `NOT closer` / negated relations: compute the positive relation as a filter
    to avoid, not as `farthest_from`.

Expected impact:

- Makes unsupported relation errors informative rather than a silent fork into
  arbitrary fallback behavior.

## 7. Implementation Phases

### Phase 1 — Deterministic low-risk hardening

1. Add masked-leak auto-retry in `select_by_text`.
2. Add target-head/category guard as a soft-block in the VG final path.
3. Extend EFG direction patterns and plural/range frame citation parsing.
4. Extend tests for the exact audited cases:
   - masked leak retry with the same query;
   - hidden target category is rejected/ignored;
   - `pillow -> bed`, `whiteboard -> chair`, `cart -> wall` category mismatch;
   - `right one` query direction beats contrastive rationale text;
   - `Frames 0, 1, and 2` citations are parsed.

This phase should not change the spatial comparison algorithm.

### Phase 2 — Explicit relation evidence binding

1. Add `evidence_id` to `compare_proposals_spatial` responses.
2. Add optional `relation_evidence` input to `submit_final`.
3. Teach TADG/EFG to prefer bound evidence over latest trace inference.
4. Change stale/ambiguous compare block messages to request corrected evidence.
5. Add role/category validation for compare calls:
   - submitted id equal to anchor id means role error;
   - compare candidates that are all anchor/support categories cannot rank the
     target.

This phase intentionally keeps the old fallback inference path for backward
compatibility, but makes it weaker than explicit binding.

### Phase 3 — Lightweight constraint planner

Add a small trace-visible planning step, initially as prompt scaffolding and
later as a helper tool if prompt-only discipline is unreliable:

```json
{
  "target_head": "chair",
  "target_candidate_ids": [19, 21],
  "anchors": [{"head": "whiteboard", "ids": [2]}, {"head": "table", "ids": [5]}],
  "positive_constraints": [{"relation": "closest_to", "anchor": "whiteboard"}],
  "negative_constraints": [{"relation": "near", "anchor": "table"}],
  "unsupported_visual_constraints": ["same_side_as", "facing"]
}
```

The first implementation can be a playbook-required scratchpad schema rather
than a new tool. A tool becomes worthwhile only if repeated traces show the
agent does not maintain the schema reliably.

## 8. Testing Strategy

### Unit tests

Add focused tests near the affected modules:

- `src/agents/tools/tests/test_selectors_text.py`
  - masked leak triggers one empty-mask retry;
  - retry preserves query;
  - both attempts are recorded.
- `src/agents/tests/test_evidence_frame_guard.py`
  - right/left target-side query phrases;
  - contrastive rationale does not flip target direction;
  - plural/range frame citation parsing.
- `src/agents/tests/test_tadg.py`
  - bound relation evidence beats stale latest compare;
  - role-inverted compare blocks with "rerun corrected compare";
  - anchor/support candidate sets do not force target changes.
- VG finalizer/chassis tests
  - category mismatch soft-blocks confident target-head cases;
  - ambiguous head cases pass.

### Case replay

Before strat600, replay a fixed 25-case audit slice from this spec plus the 15
previously audited cases. Record per-case outcomes:

- selected id;
- number of guard blocks;
- whether masked retry fired;
- whether final category matched expected target head;
- whether relation evidence was explicit or inferred.

The replay need not claim benchmark improvement. Its purpose is to verify
behavioral traces on known failure modes.

### Benchmark validation

After focused replay passes, run canonical NR3D strat600 with a new run id and
record it under `docs/benchmark/nr3d/`:

- commit must be clean before launch;
- capture head/run-time commit;
- ingest SQLite immediately after metrics;
- compare against v10 no-initial-keyframes and v9.3 baselines.

## 9. Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Target-category guard blocks valid pronoun-shift queries | Only block when target head is confident; otherwise emit warning or pass. |
| Explicit relation evidence makes tool calls more verbose | Keep it optional; use fallback inference for legacy traces but prefer explicit binding. |
| Candidate coverage increases turns and cost | Limit strict coverage to small same-category sets and relation/superlative queries. |
| Unsupported relation workflows become too prompt-heavy | Start with playbook routing; promote only repeated stable patterns to tools. |
| Strat600 changes become hard to attribute | Phase commits and run focused replay before full strat600. |

## 10. Acceptance Criteria

Phase 1 is complete when:

- targeted unit tests pass;
- masked-category errors auto-retry with empty masks and preserve query text;
- final category mismatches are soft-blocked in confident head-noun cases;
- EFG parses `right one/on the right` and plural frame citations.

Phase 2 is complete when:

- compare responses include stable evidence ids;
- `submit_final` can bind relation evidence;
- TADG/EFG no longer force rank-1 from role-inverted or stale compares in unit
  tests.

Phase 3 is complete when:

- the agent trace contains a target/anchor/constraint decomposition before final
  in multi-clause relation cases;
- negated relation cases compare the avoided anchor relation rather than
  replacing `not closer` with `farthest`.

The full feature is benchmark-ready when the 40-case audit slice shows no
regression in the previously fixed cases and no new script-level failures.

## 11. Spec Self-Review

- No implementation code is included in this document.
- The phases are independently testable and do not require restoring initial
  keyframes.
- The design avoids GT leakage by deriving target categories from query text and
  agent-visible catalog labels, not hidden benchmark answers.
- The remaining semantic relation work is explicitly staged after deterministic
  guard/tool fixes.
