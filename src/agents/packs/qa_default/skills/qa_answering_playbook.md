# QA Answering Playbook (v9)

Prerequisite: `load_skill('scene-exploration-playbook')` first.

You are answering an embodied-QA question. Your final payload follows the
`Stage2QAResult` schema (free-form `answer` + optional `supporting_claims`).

## Question taxonomy & default flow

| Question kind | Default approach |
| --- | --- |
| "what is in / count of X" | `select_by_text(query='X')` → `view_keyframe(mode='rgb')` |
| "what colour / material / state of X" | `select_by_proposal([#X])` → `view_keyframe(mode='rgb')` → `request_crops` if ambiguous |
| "where is X relative to Y" | `compare_proposals_spatial(candidate_ids=[#X], anchor_id=#Y, …)` |
| "is X there / are there any" | BEV inspection → `select_by_text` → `view_keyframe(mode='rgb')` |
| open-ended description | `select_by_coverage(method='pose_depth', k=4)` → multiple `view_keyframe` |

## Step-by-step

1. Read the BEV image and the `Proposals by category:` block.
2. Pick the cheapest selector matching the question kind (see table above).
3. View the returned frames with `view_keyframe(frame_id, mode='rgb')` —
   raw RGB is preferred for QA so that visual cues (colour, occupancy, state)
   are unobstructed by mask boxes.
4. For fine attributes (small text on objects, colour patch, gauge readings),
   call `request_crops(request_text=…, object_terms=[…])`.
5. Compose the answer in natural language. Populate `supporting_claims` with
   `{frame_id: int, proposal_ids: [#a, #b], note: "…"}` for each piece of
   evidence you used.

## Counting questions

- Trust the catalog Cat-B count if and only if the user query category matches a
  catalog category exactly (e.g. "how many chairs" → count proposals in
  `chair`).
- Otherwise verify with at least one `view_keyframe(mode='rgb')` and one
  `list_frame_proposals(frame_id)` confirmation.

## When to use `select_by_text`

Use it when:
- the question references appearance (colour, material) not in catalog labels;
- the question references a state ("is the door open") that catalog labels
  cannot express;
- proposal/region/neighbour selectors come back empty.

## Anti-patterns

- Answering from catalog text without viewing any RGB frame.
- Using marked mode for QA — masks cover visual evidence.
- Composing `supporting_claims` with frame_ids you did not view (the
  evidence_frame_guard will catch this and force a re-run).
