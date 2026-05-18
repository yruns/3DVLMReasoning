# VG Spatial Disambiguation (v9.1)

Used to resolve proposal relations that the spatial comparison tool can rank.
Allowed canonical relation values are: `closest_to`, `near`, `next_to`,
`farthest_from`, `above`, `below`, `left_of`, `right_of`.

Workflow:
1. Identify the candidate proposals (`#a`, `#b`) and the anchor (`#x`).
2. Call `compare_proposals_spatial(candidate_ids=[#a, #b], anchor_id=#x, relation='<rel>')`.
3. Verify visually with `mark_frame_with_bbox(frame_id, ids=[#a, #b, #x])`. The
   `frame_id` must come from a selector first (e.g. `select_by_text` or
   `select_by_region`) — `mark_frame_with_bbox` does not fetch new frames, it
   annotates one you have already seen.
4. Submit the winning candidate via `submit_final`.

Use `closest_to` for "closer/nearest" phrasing and `farthest_from` for
"farther/furthest" phrasing. For unsupported natural-language relations
such as front/behind, between, facing, across, or same-side, do not invent
a relation string. Fetch/mark frames that show the target and anchor, then
judge the relation visually.

Use `select_by_region(region_type='bbox_3d')` to fetch frames that show both the
anchor and the candidates simultaneously. If a single frame does not contain all
three, mark multiple frames (one anchor-focused, one candidate-focused) before
choosing.
