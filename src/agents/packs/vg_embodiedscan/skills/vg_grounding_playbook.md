# VG Grounding Playbook (v9.1)

Prerequisite: `load_skill('scene-exploration-playbook')` first.

You are grounding a natural-language referring expression to **one** proposal in
the SceneCatalog. The output must be a single `proposal_id` (or `-1` if the target
is genuinely absent from the catalog — OOD case).

## Standard flow

1. **Read the BEV image** and the `Proposals by category:` block in the task
   message. Identify candidate `#id`s by category (e.g. "brown chair" → all
   `chair` proposals).
2. **First move: `select_by_text(query)`** — runs Stage-1 query parsing and
   returns ≤3 candidate first-person RGB frames + their `visible_proposal_ids`
   in one call. If catalog IDs are already obvious from the BEV labels,
   `select_by_proposal(proposal_ids=[candidate ids])` is the direct shortcut.
3. **Verify the chosen candidate** with
   `mark_frame_with_bbox(frame_id, ids=[#a, #b, …])` (or
   `labels=['chair', …]`). This renders the frame with high-contrast bounding
   boxes around the named catalog entries; pixels confirm which proposal the
   referring expression matches.
4. **Disambiguate spatial relations** with `compare_proposals_spatial(
   candidate_ids=[…], anchor_id=#x, relation='left_of'|'right_of'|'closer_to'|…)`
   when the query involves a spatial relation.
5. **Submit** with `submit_final(payload={"proposal_id": <id>}, …)`. The
   evidence-frame guard requires that at least one `mark_frame_with_bbox`
   call covered the submitted proposal id.

## Tools at a glance

- `select_by_text(query, k=3, hidden_categories?)` — **first move**;
  language → ≤3 RGB candidate frames.
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
- `view_bev(highlight=[ids])` — re-render BEV focused on subset.
- `compare_proposals_spatial(candidate_ids, anchor_id, relation)` —
  spatial disambiguation (TADG-relevant).

## Guards (read this before submit)

- **TADG** (Target-Anchor Disambiguation Guard): if the query mentions an
  anchor (e.g. "next to the kitchen counter"), TADG will block submission
  unless `compare_proposals_spatial` proved the chosen proposal satisfies the
  relation. If TADG fires, run the comparison and resubmit.
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
