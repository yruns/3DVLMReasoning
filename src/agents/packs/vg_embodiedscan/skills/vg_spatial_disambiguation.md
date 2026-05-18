# VG Spatial Disambiguation (v9.1)

Used to resolve proposal relations that the spatial comparison tool can rank.
Allowed canonical relation values are: `closest_to`, `near`, `next_to`,
`farthest_from`, `above`, `below`, `left_of`, `right_of`.

Workflow:
1. Identify the candidate proposals (`#a`, `#b`) and the anchor (`#x`).
2. Call `compare_proposals_spatial(candidate_ids=[#a, #b], anchor_id=#x, relation='<rel>')`.
   The response returns a stable `evidence_id`.
3. Verify visually with `mark_frame_with_bbox(frame_id, ids=[#a, #b, #x])`. The
   `frame_id` must come from a selector first (e.g. `select_by_text` or
   `select_by_region`) — `mark_frame_with_bbox` does not fetch new frames, it
   annotates one you have already seen.
4. Submit the winning candidate via `submit_final(..., relation_evidence={"evidence_id": ...})`
   using the `evidence_id` copied from the comparison response.

Use `closest_to` for "closer/nearest" phrasing and `farthest_from` for
"farther/furthest" phrasing. Vertical above/below/under relations should use
this tool before finalizing: map "under", "right under", and "beneath" to
`relation='below'`; map "above", "over", and "on top of" to
`relation='above'`. The comparator uses the correct vertical side plus
horizontal alignment, so call it to rank candidates first, then verify the
top-ranked candidate in a marked frame with the anchor visible.

For unsupported natural-language relations
such as front/behind, between, facing, across, or same-side, do not invent
a relation string. Fetch/mark frames that show the target and anchor, then
judge the relation visually.

Unsupported semantic relations are visual/BEV workflows, not relation strings.
Do not call `compare_proposals_spatial` with `same_side_as`, `between`,
`opposite`, `across_from`, `facing`, `in_front_of`, or `behind`. For those,
mark the target candidates and anchor(s), use BEV/3D positions for room-side or
between/opposite checks, and cite the marked frames used for appearance.
For negated relations like "not closer to X", first compare the positive
relation to identify candidates to avoid, then choose among the remaining
target-category candidates.

Nested anchor relations must be resolved in dependency order. For expressions
like "the chair behind the desk closest to the window", resolve the anchor
candidates first: find all desk candidates, identify the window anchor, and call
`compare_proposals_spatial(candidate_ids=[anchor ids], anchor_id=#window, relation='closest_to')`.
Only after that comparison names the desk anchor should you then rank the target
candidates against that resolved anchor. If the target relation is unsupported
by `compare_proposals_spatial` (for example `behind`), use marked frames and
BEV evidence against the resolved anchor instead of comparing against every
desk candidate at once.

Use `select_by_region(region_type='bbox_3d')` to fetch frames that show both the
anchor and the candidates simultaneously. If a single frame does not contain all
three, mark multiple frames (one anchor-focused, one candidate-focused) before
choosing.
