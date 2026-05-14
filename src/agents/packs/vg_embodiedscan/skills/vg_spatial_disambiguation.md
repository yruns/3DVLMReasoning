# VG Spatial Disambiguation (v9)

Used to resolve "left of / right of / closer to / on / inside" between proposals.

Workflow:
1. Identify the candidate proposals (`#a`, `#b`) and the anchor (`#x`).
2. Call `compare_proposals_spatial(candidate_ids=[#a, #b], anchor_id=#x, relation='<rel>')`.
3. Inspect the result; verify visually with `view_keyframe(mode='marked')`
   (optionally pass `proposal_ids=[#a, #b, #x]` to narrow the overlays).
4. Submit the winning candidate.

Use `select_by_region(region_type='bbox_3d')` to fetch frames that show both the
anchor and the candidates simultaneously. If a single frame does not contain all
three, view multiple frames (one anchor-focused, one candidate-focused) before
choosing.
