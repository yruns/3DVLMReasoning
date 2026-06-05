---
name: vg-spatial-disambiguation
description: Use when an NR3D/VG query contains spatial relations, anchors, ordering, closest/farthest, left/right, or nested relations.
---

# vg-spatial-disambiguation

This Codex Agent SDK skill is synchronized from the DeepAgents playbook used by the Stage-2 VG runtime. Regenerate with `scripts/sync_codex_playbook_skills.py` after editing the source playbook.

Codex SDK note: this skill is mounted as a `SkillInput`; the `nr3d_tools` MCP runtime preloads the corresponding tool gates.

Source playbook: `src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation.md`

<!-- BEGIN_SYNCED_PLAYBOOK: src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation.md -->
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
"farther/furthest" phrasing. For unsupported natural-language relations
such as front/behind, between, facing, across, or same-side, do not invent
a relation string. Fetch/mark frames that show the target and anchor, then
judge the relation visually.

For multi-anchor closest/farthest/near/next_to cases, use
`compare_candidates_to_anchors(candidate_ids=[target ids], anchor_ids=[anchor ids], relation='<rel>')`.
It returns per-anchor rankings plus `anchor_disagreement`. If the same target
does not win for every plausible anchor, resolve the anchor identity with
marked visual evidence first; do not compare against only one anchor and final
from that partial ranking.

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
<!-- END_SYNCED_PLAYBOOK -->
