# QA Answering Playbook (v9.1)

Prerequisite: `load_skill('scene-exploration-playbook')` first.

You are answering an embodied-QA question. Your final payload follows the
`Stage2QAResult` schema (free-form `answer` + optional `supporting_claims`).

## Question taxonomy & default flow

| Question kind | Default approach |
| --- | --- |
| "what / count of X" | `select_by_text(query='X')` returns the frames; verify count by reading `visible_proposal_ids` and inspecting the most informative one. |
| "colour / material / state of X" | `select_by_text(query='X')` → `mark_frame_with_bbox(frame_id, labels=['X'])` for explicit attribution → `request_crops` if still ambiguous. |
| "where is X relative to Y" | `compare_proposals_spatial(candidate_ids=[#X], anchor_id=#Y, ...)` after selectors return one frame containing both. |
| "is X there / are there any" | `select_by_text(query='X')` — empty result is itself the evidence of absence; cross-check with BEV labels. |
| open-ended description | `select_by_coverage(method='pose_depth', k=3)` → mark only the frames that contributed to the answer. |

QA historically needed plain-RGB inspection because mask overlays covered
visual evidence. With v9.1 selectors now returning plain first-person RGB
frames by default, `mark_frame_with_bbox` is opt-in for QA — only call it
when you want to attribute the answer to a specific catalog entry.

## Step-by-step

1. Read the BEV image and the `Proposals by category:` block.
2. **First move:** `select_by_text(query=...)` returns up to 3 first-person
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
- Otherwise verify with at least one `select_by_text` and one
  `list_frame_proposals(frame_id)` confirmation.

## When to use `select_by_text` vs the catalog selectors

`select_by_text` is the default first move because it parses a natural-language
query into a frame ranking in one call. Reach for `select_by_proposal` when you
already know the catalog ids (e.g. after a previous `select_by_text` returned
the right instance and you only want different angles on it).

## Anti-patterns

- Answering from catalog text without viewing any RGB frame.
- Calling `mark_frame_with_bbox` for QA when the question does not need
  explicit catalog attribution — plain RGB from the selector is usually
  enough.
- Composing `supporting_claims` with frame_ids you did not view (the
  evidence_frame_guard will catch this and force a re-run).
