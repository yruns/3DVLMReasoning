# QA Answering Playbook — catalog-first variant

Prerequisite: `load_skill('scene-exploration-playbook')` first (it loads the
catalog-first variant automatically in this run).

You are answering an embodied-QA question. Your final payload follows the
`Stage2QAResult` schema (free-form `answer` + optional `supporting_claims`).

> **Why this variant exists.** The Stage-1 audit
> (`docs/benchmark/nr3d/v9_1_select_by_text_audit_20260516.md`) showed
> Stage-1 language→frame retrieval returning `no_evidence` on 32 % of NR3D
> queries. This run drops the tool entirely so the playbook stays consistent
> with the actual tool surface.

## Question taxonomy & default flow

| Question kind | Default approach |
| --- | --- |
| "what / count of X" | `list_scene_proposals(category='X')` to get candidate IDs, then `select_by_proposal(proposal_ids=[…])` to fetch frames; verify by reading `visible_proposal_ids` and inspecting the most informative one. |
| "colour / material / state of X" | Same as above → `mark_frame_with_bbox(frame_id, labels=['X'])` for explicit attribution → `request_crops` if still ambiguous. |
| "where is X relative to Y" | `compare_proposals_spatial(candidate_ids=[#X], anchor_id=#Y, ...)` after `select_by_proposal` returns one frame containing both. |
| "is X there / are there any" | `list_scene_proposals(category='X')` — an empty result is itself evidence of absence; cross-check with BEV labels. |
| open-ended description | `select_by_coverage(method='pose_depth', k=3)` → mark only the frames that contributed to the answer. |

With v9.1 selectors returning plain first-person RGB frames,
`mark_frame_with_bbox` is opt-in for QA — only call it when you want to
attribute the answer to a specific catalog entry.

## Step-by-step

1. Read the BEV image and the `Proposals by category:` block.
2. **First move:** identify candidate catalog IDs from the inventory, then
   `select_by_proposal(proposal_ids=[…])` returns up to 3 first-person
   RGB frames + their `visible_proposal_ids`. Scan them; pick the most
   informative one.
3. For fine attributes (small text on objects, colour patch, gauge readings)
   call `request_crops(frame_id, bbox_2d=[...])` on a frame you have seen.
4. For explicit attribution (e.g. "the mug labelled with #7"), call
   `mark_frame_with_bbox(frame_id, labels=['mug'])` or pass `ids=[#7]`.
5. Compose the answer in natural language. Populate `supporting_claims` with
   `{frame_id: int, proposal_ids: [#a, #b], note: "…"}` for each piece of
   evidence you used.

## Counting questions

- Trust the catalog Cat-B count if and only if the user query category matches
  a catalog category exactly (e.g. "how many chairs" → count proposals in
  `chair`).
- Otherwise verify with at least one `select_by_proposal` and one
  `list_frame_proposals(frame_id)` confirmation.

## Anti-patterns

- Answering from catalog text without viewing any RGB frame.
- Calling `mark_frame_with_bbox` for QA when the question does not need
  explicit catalog attribution — plain RGB from the selector is usually
  enough.
- Composing `supporting_claims` with frame_ids you did not view (the
  evidence_frame_guard will catch this and force a re-run).
