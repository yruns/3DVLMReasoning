# VG Spatial Disambiguation

## When to use this skill

Load this skill in addition to `vg-grounding-playbook` whenever the
visual-grounding query contains an explicit spatial relation between
two object referents. Telltale phrases:

- "next to ...", "closest to ...", "nearest to ..." → `closest_to`
- "farthest from ...", "on the opposite side of ..." → `farthest_from`
- "between A and B", "in the middle of ..." → reduces to two
  `closest_to` calls (one against each anchor)
- "above ...", "below ...", "on top of ..." → height-axis variants;
  these still resolve to `closest_to` over the (x, y, z) center, but
  the agent should sanity-check by viewing a marked keyframe before
  submitting (the proposal ranking is symmetric in 3-D, so two
  proposals at the same xy distance will tie regardless of z).

This skill assumes you have already loaded `vg-grounding-playbook` and
you understand the 5 VG tools. It only adds disambiguation patterns;
it does not duplicate the tool descriptions.

## Three-step workflow

A spatial query decomposes into:

1. **target** — the object the user wants found (returned in
   `submit_final`).
2. **relation** — one of `closest_to | farthest_from`. Compound
   relations (between, opposite) reduce to multiple calls.
3. **anchor** — the reference object the relation is measured against.

Resolve target candidates and the anchor independently first; combine
with `compare_proposals_spatial` last. Do NOT use spatial reasoning to
filter candidates before you have a candidate list.

## Tool sequence

For a query "find the X relation Y":

1. `find_proposals_by_category("X")` → `target_candidates`.
2. `find_proposals_by_category("Y")` → `anchor_candidates`.
3. If `len(anchor_candidates) > 1`, the anchor itself is ambiguous
   (see "Ambiguous anchor" below). Otherwise pick the single anchor.
4. `compare_proposals_spatial(candidate_ids=target_candidates,
   anchor_id=picked_anchor, relation="closest_to" | "farthest_from")`
   → `ranked_ids`.
5. The first id in `ranked_ids` is your best guess. Cross-check by
   `view_keyframe_marked(frame_id=...)` on a frame that contains both
   the target and the anchor (use `inspect_proposal` to find their
   shared frames if needed).
6. `submit_final({"proposal_id": ranked_ids[0], "confidence": ...},
   rationale="...")`.

## Ambiguous anchor

If `find_proposals_by_category(anchor)` returns ≥ 2 ids, you cannot
just pick the first. The agent must either:

- (a) Look at the user's full query for additional context that
  narrows the anchor (e.g. "the desk near the window" — apply this
  skill recursively with the windowed-anchor as the next-level
  target), or
- (b) Compute the spatial answer for each anchor candidate
  independently and either pick the most likely one given the query's
  natural-language hint, or report failure if all are equally
  plausible.

In practice (a) covers most cases. Only resort to (b) when no
narrowing context exists, and budget allows multiple
`compare_proposals_spatial` calls.

## Anti-patterns

- Do NOT call `compare_proposals_spatial` with a single-element
  `candidate_ids` — the ranking is trivially that one id, and you
  haven't used the spatial information.
- Do NOT use `compare_proposals_spatial` with `relation="left_of"`
  or any unsupported relation — the tool FAIL-LOUDs immediately. Use
  `closest_to` over a re-projected coordinate axis if you genuinely
  need an axis-aligned filter (out of scope for pack v1).
- Do NOT rely on `compare_proposals_spatial` for "between A and B"
  without explicitly intersecting two `closest_to` calls; bbox-center
  Euclidean distance ranks single-anchor relations only.
- Do NOT skip cross-checking with `view_keyframe_marked` after
  ranking — distances on bbox centers do not always match the visual
  intuition of "next to" (e.g. two boxes touching but with very
  different sizes).
- Do NOT assume the anchor is always disambiguated — always inspect
  the length of `anchor_candidates`.

## Examples

### Example 1: closest_to (1 anchor)

Query: "find the trash can closest to the door."

1. `find_proposals_by_category("trash can")` → `[12, 18, 22]`.
2. `find_proposals_by_category("door")` → `[7]` (single anchor).
3. `compare_proposals_spatial([12, 18, 22], anchor_id=7,
   relation="closest_to")` → `{"ranked_ids": [18, 12, 22], ...}`.
4. `view_keyframe_marked(frame_id=...)` showing both the door and
   proposal 18 to confirm.
5. `submit_final({"proposal_id": 18, "confidence": 0.9}, ...)`.

### Example 2: farthest_from

Query: "find the chair farthest from the TV."

1. `find_proposals_by_category("chair")` → `[2, 4, 11]`.
2. `find_proposals_by_category("tv")` → `[6]`.
3. `compare_proposals_spatial([2, 4, 11], anchor_id=6,
   relation="farthest_from")` → `{"ranked_ids": [11, 4, 2]}`.
4. `submit_final({"proposal_id": 11, "confidence": 0.85}, ...)`.

### Example 3: between (compound relation)

Query: "find the lamp between the sofa and the bookshelf."

1. `find_proposals_by_category("lamp")` → `[5, 9]`.
2. `find_proposals_by_category("sofa")` → `[14]`; `("bookshelf")` → `[19]`.
3. `compare_proposals_spatial([5, 9], anchor_id=14,
   relation="closest_to")` → `[9, 5]`.
4. `compare_proposals_spatial([5, 9], anchor_id=19,
   relation="closest_to")` → `[9, 5]`.
5. The lamp that is the closest-to-both is id 9 (top of both rankings)
   — likely "between" the two anchors.
6. `view_keyframe_marked(frame_id=...)` showing all three to confirm
   visually that lamp 9 is geometrically between sofa 14 and
   bookshelf 19 (not just close to both on the same side).
7. `submit_final({"proposal_id": 9, "confidence": 0.8}, ...)`.

### Example 4: above (height-axis variant)

Query: "find the picture above the bed."

1. `find_proposals_by_category("picture")` → `[3, 7, 12]`.
2. `find_proposals_by_category("bed")` → `[1]`.
3. `compare_proposals_spatial([3, 7, 12], anchor_id=1,
   relation="closest_to")` → `[7, 3, 12]`.
4. `inspect_proposal(proposal_id=7)` and `inspect_proposal(proposal_id=1)`
   to compare bbox centers' z values; pick the picture whose z > bed's z.
5. If id 7's z is below the bed's z (e.g. picture on the side wall),
   try id 3 instead.
6. `view_keyframe_marked(frame_id=...)` to confirm.
7. `submit_final({"proposal_id": chosen_id, "confidence": 0.8}, ...)`.

### Ambiguous anchor

Query: "find the chair next to the desk" — but
`find_proposals_by_category("desk")` returns `[5, 11]`.

1. `inspect_proposal(proposal_id=5)` — score 0.95, in frames `[3, 8]`.
2. `inspect_proposal(proposal_id=11)` — score 0.62, in frames `[15]`.
3. `view_keyframe_marked(frame_id=3)` and `view_keyframe_marked(frame_id=15)`.
4. The user's query likely refers to the more prominent desk (5), so
   apply the main workflow with `anchor_id=5`. If the rendered scene
   makes both desks plausible, run the workflow once per anchor and
   submit the best of the two — or `submit_final({"proposal_id": -1, ...})`
   with a rationale citing the irreducible anchor ambiguity.
