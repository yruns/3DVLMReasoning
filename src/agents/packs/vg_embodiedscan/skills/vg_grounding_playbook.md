# VG Grounding Playbook (v9)

Prerequisite: `load_skill('scene-exploration-playbook')` first.

You are grounding a natural-language referring expression to **one** proposal in
the SceneCatalog. The output must be a single `proposal_id` (or `-1` if the target
is genuinely absent from the catalog — OOD case).

## Standard flow

1. **Read the BEV image** and the `Proposals by category:` block in the task
   message. Identify candidate `#id`s by category (e.g. "brown chair" → all
   `chair` proposals).
2. **Cheap filter** with `select_by_proposal(proposal_ids=[candidate ids])`
   to obtain the frames that contain any candidate.
3. **View the frames** with `view_keyframe(frame_id, mode='marked', proposal_ids=[…])`.
   Marked mode draws boxes labelled `#id` on the candidate proposals; visually
   verify which one matches the query.
4. **Disambiguate spatial relations** with `compare_proposals_spatial(
   candidate_ids=[…], anchor_id=#x, relation='left_of'|'right_of'|'closer_to'|…)`
   when the query involves a spatial relation.
5. **Verify edge attributes** (colour, material, state) with
   `request_crops(request_text='…', object_terms=[…])` only when a frame view is
   ambiguous.
6. **Submit** with `submit_final(payload={"proposal_id": <id>}, …)`.

## Tools at a glance

- `select_by_proposal(proposal_ids, require_all=False, k=8)` — instant catalog lookup.
- `select_by_frame_neighbor(anchor_frame_id, mode='temporal'|'viewpoint_diverse')`
  — expand around a good frame.
- `select_by_region(region, region_type='bev_2d'|'bbox_3d')` — region filter.
- `select_by_coverage(method='obj_iou'|'pose_depth')` — diverse coverage.
- `select_by_text(query)` — fallback when category labels don't narrow enough.
- `view_keyframe(frame_id, mode='auto')` — auto → 'marked' for VG.
- `list_frame_proposals(frame_id)` — text-only proposals on a frame.
- `list_scene_proposals(category=?, region_bev=?)` — scene-wide inventory.
- `inspect_proposal(proposal_id)` — frames_appeared + position + bbox.
- `view_bev(highlight=[ids])` — re-render BEV with subset only.

## Guards (read this before submit)

- **TADG** (Target-Anchor Disambiguation Guard): if the query mentions an
  anchor (e.g. "next to the kitchen counter"), TADG will block submission
  unless `compare_proposals_spatial` proved the chosen proposal satisfies the
  relation. If TADG fires, run the comparison and resubmit.
- **no_match_guard**: if `select_by_*` and `view_keyframe` did not surface any
  candidate of the queried category, submit `proposal_id=-1` (OOD).
- **evidence_frame_guard**: at least one frame visible with the chosen
  proposal must appear in the trace under `view_keyframe(mode='marked')` —
  unmarked RGB views do not count as VG evidence.

## OOD policy

If after exhaustive selectors + ≥3 view_keyframe calls you cannot locate the
referent, submit `proposal_id=-1` with a rationale explaining what categories /
regions you searched. Do not invent an `id`.

## Anti-patterns

- Submitting based on BEV labels alone (no `view_keyframe`).
- Using `select_by_text` when the BEV already shows obvious category labels.
- Forgetting `mode='marked'` and trying to recognise objects from raw RGB.
- Querying `request_crops` before any `view_keyframe`.
