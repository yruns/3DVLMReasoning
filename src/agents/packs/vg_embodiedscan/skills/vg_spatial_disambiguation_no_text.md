# VG Spatial Disambiguation — catalog-first (no `select_by_text`)

Used to resolve "left of / right of / closer to / on / inside" between proposals.

Workflow:
1. Identify the candidate proposals (`#a`, `#b`) and the anchor (`#x`).
2. Call `compare_proposals_spatial(candidate_ids=[#a, #b], anchor_id=#x, relation='<rel>')`.
3. Verify visually with `mark_frame_with_bbox(frame_id, ids=[#a, #b, #x])`. The
   `frame_id` must come from a selector first (e.g. `select_by_proposal` or
   `select_by_region`) — `mark_frame_with_bbox` does not fetch new frames, it
   annotates one you have already seen.
4. Submit the winning candidate via `submit_final`.

Use `select_by_region(region_type='bbox_3d')` to fetch frames that show both the
anchor and the candidates simultaneously, or `select_by_proposal(ids=[#a, #b, #x],
require_all=True)` when you want a single frame containing all three. If no
single frame contains all three, mark multiple frames (one anchor-focused, one
candidate-focused) before choosing.
