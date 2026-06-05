---
name: vg-grounding-playbook
description: Use when selecting one proposal_id for an NR3D visual grounding query with text-first and catalog VG tools.
---

# vg-grounding-playbook

This Codex Agent SDK skill is synchronized from the DeepAgents playbook used by the Stage-2 VG runtime. Regenerate with `scripts/sync_codex_playbook_skills.py` after editing the source playbook.

Codex SDK note: this skill is mounted as a `SkillInput`; the `nr3d_tools` MCP runtime preloads the corresponding tool gates.

Source playbook: `src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md`

<!-- BEGIN_SYNCED_PLAYBOOK: src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md -->
# VG Grounding Playbook (v9.1)

Codex SDK prerequisite: `scene-exploration-playbook` is already mounted as a skill input and the scene-exploration tool gate is preloaded.

You are grounding a natural-language referring expression to **one** proposal in
the SceneCatalog. The output must be a single `proposal_id` (or `-1` if the target
is genuinely absent from the catalog — OOD case).

## Standard flow

1. **Read the BEV image** and the `Proposals by category:` block in the task
   message. Identify candidate `#id`s by category (e.g. "brown chair" → all
   `chair` proposals). Generic category words include subtype labels: a query
   for "chair" can refer to an `office chair`, `desk chair`, `lounge chair`,
   or `armchair`; do not reject a subtype label when relation/visual evidence
   matches the referring expression. If the task message includes
   `Proposal notes:`, use those compact enriched notes for initial
   shortlisting, but still verify visually before submit.
2. **First move (audit-informed)** — pick one:
   - If the BEV shows the candidate categories clearly and the query
     references a relation / direction ("the chair on the right",
     "facing the fridge", "between the beds"), go straight to
     `select_by_proposal(proposal_ids=[candidate ids])`. On NR3D Stage-1
     returns `no_evidence` on ~32 % of these and you save a turn.
   - If the category is rare or the BEV does not have a confident
     shortlist, `select_by_text(query)` is the right first move — it
     runs Stage-1 parsing and returns ≤3 RGB frames in one call.
   - See `docs/benchmark/nr3d/v9_1_select_by_text_audit_20260516.md` for
     the audit data behind this routing.
3. **Verify the chosen candidate** with
   `mark_frame_with_bbox(frame_id, ids=[#a, #b, …])` (or
   `labels=['chair', …]`). This renders the frame with high-contrast bounding
   boxes around the named catalog entries; pixels confirm which proposal the
   referring expression matches.
4. **Disambiguate spatial relations** with `compare_proposals_spatial(
   candidate_ids=[…], anchor_id=#x, relation='left_of'|'right_of'|'closest_to'|…)`
   when the query involves a spatial relation.
   Allowed canonical relation values are: `closest_to`, `near`, `next_to`,
   `farthest_from`, `above`, `below`, `left_of`, `right_of`. Use
   `closest_to` for "closer/nearest" phrasing and `farthest_from` for
   "farther/furthest" phrasing. For unsupported relations like front/behind,
   between, facing, across, or same-side, use marked frames / BEV evidence
   instead of inventing a relation string. The response includes a stable
   `evidence_id`; copy it into `submit_final(..., relation_evidence={"evidence_id": ...})`
   when the final answer relies on that comparison.
   If the anchor category itself has more than one plausible proposal for a
   closest/farthest/near/next_to relation, call
   `compare_candidates_to_anchors(candidate_ids=[target ids], anchor_ids=[anchor ids], relation='<rel>')`.
   Inspect each per-anchor ranking. If `anchor_disagreement` is true, first
   resolve which anchor the expression means with marked visual evidence; do
   not compare against only one anchor and final from that partial ranking.

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

For view-dependent left/right queries ("facing the X", "from behind",
"looking in from the door", "standing at the foot/end"), first establish the
viewer/anchor frame, then compare target candidates in that same marked frame.
Use `select_by_proposal(..., require_all=True)` when the candidate set is
small enough to seek a co-visible frame. Do not combine screen-left/right
evidence from different camera viewpoints or different candidate pairs as if
it were one ordering. If a previous spatial comparison narrowed the set (for
example "the two closest to the table"), apply left/right only inside that
subset, not across every same-category object in the frame.
5. **Submit** with `submit_final(payload={"proposal_id": <id>}, …)`. The
   evidence-frame guard requires that at least one `mark_frame_with_bbox`
   call covered the submitted proposal id.

## Tools at a glance

- `select_by_text(query, k=3)` — **first move**; language → ≤3 RGB
  candidate frames. Omit `hidden_categories` for NR3D / VG unless the
  benchmark explicitly masks the target category. Never hide support,
  anchor, or context categories.
- `select_by_proposal(proposal_ids, require_all=False, k=3)` — fetch
  frames containing specific catalog IDs.
- `select_by_frame_neighbor(anchor_frame_id, mode='temporal'|'viewpoint_diverse')`.
- `select_by_region(region, region_type='bev_2d'|'bbox_3d')`.
- `select_by_coverage(method='obj_iou'|'pose_depth')`.
- `mark_frame_with_bbox(frame_id, labels|ids)` — high-contrast
  annotated zoom; required before `submit_final` for VG (see
  evidence_frame_guard).
- `list_frame_proposals(frame_id)`, `list_scene_proposals(...)`,
  `inspect_proposal(id)` — text-only catalog queries.
- `inspect_proposal(id)` returns proposal metadata plus the full enriched
  object description when available (`description / location / nearby_objects /
  color / usability`). Call it when compact notes or category labels are not
  enough to resolve visual attributes, functional clues, or nearby-object
  context.
- `view_bev(highlight=[ids])` — re-render BEV with text labels only on those proposals.
- `view_bev(categories=["chair", "table"])` — text-label proposals whose
  category matches (case-insensitive exact). Default `view_bev()` is
  a clean overview (mesh + trajectory + small dots, no labels).
- `compare_proposals_spatial(candidate_ids, anchor_id, relation)` —
  spatial disambiguation (TADG-relevant). Returns `evidence_id` plus ranked
  candidates; use that id as `relation_evidence` in `submit_final` to bind
  the final answer to the recorded comparison.
- `compare_candidates_to_anchors(candidate_ids, anchor_ids, relation)` —
  multi-anchor spatial check for closest/farthest/near/next_to relations.
  Returns per-anchor rankings plus `anchor_disagreement`; use it before
  finalizing when multiple same-category anchors are plausible.

## Guards (read this before submit)

- **TADG** (Target-Anchor Disambiguation Guard): if the query mentions an
  anchor (e.g. "next to the kitchen counter"), TADG will block submission
  unless `compare_proposals_spatial` proved the chosen proposal satisfies the
  relation. If TADG fires, run the comparison and resubmit with
  `relation_evidence={"evidence_id": "<compare id>"}` copied from the tool
  response.
- **no_match_guard**: if selectors did not surface any candidate of the
  queried category, submit `proposal_id=-1` (OOD).
- **evidence_frame_guard**: at least one `mark_frame_with_bbox` call whose
  filter (`ids=` or `labels=`) covers the submitted proposal id must appear
  in the trace. Plain RGB injection via a selector does not satisfy this
  guard.

## OOD policy

If after exhaustive selectors + ≥3 `mark_frame_with_bbox` calls you cannot
locate the referent, submit `proposal_id=-1` with a rationale explaining what
categories / regions you searched. Do not invent an `id`.

## Anti-patterns

- Submitting based on BEV labels alone (no `mark_frame_with_bbox`).
- Calling `mark_frame_with_bbox` on a frame you have not seen yet (it must
  come from a selector first).
- Calling more than two selectors before any `mark_frame_with_bbox` — that
  means you are collecting candidates without ever verifying.
- Querying `request_crops` before any selector returned a host frame.
<!-- END_SYNCED_PLAYBOOK -->
