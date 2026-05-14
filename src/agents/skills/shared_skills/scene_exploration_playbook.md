# Scene Exploration Playbook (shared, prerequisite)

You start each scene with two views:

1. A BEV (top-down) image with every `#id category` label drawn at the
   proposal's 3D centre, plus the camera trajectory.
2. A `SceneCatalog` text inventory (`Proposals by category: chair: [#4, #5, ...]`).

Neither is first-person evidence. **You must fetch first-person frames before
finalising any answer that depends on appearance, count, state, or relations.**

## Mental model

- BEV + catalog = scene structure (what / where / how many).
- view_keyframe = ground truth appearance / colour / state / view-dep relations.
- The scene contains N frames (see "Total frames" in the task message). You
  have viewed 0. Plan a small fetching budget (typically 3-8 frames).

## Selector cheat sheet — cheapest first

| Selector | Cost | When to use |
| --- | --- | --- |
| `select_by_proposal(proposal_ids=[#a,#b])` | instant | You already know the catalog IDs. Returns frames containing them. |
| `select_by_frame_neighbor(anchor_frame_id, mode='temporal'\|'viewpoint_diverse')` | instant | You found one good frame, want neighbours or alternate angles. |
| `select_by_region(region, region_type='bev_2d'\|'bbox_3d')` | instant | You see a BEV cluster; expand to all frames overlooking that region. |
| `select_by_coverage(method='obj_iou'\|'pose_depth', seen_frame_ids?)` | cheap | You need diverse coverage of the room. |
| `select_by_text(query)` | ~Stage-1 LLM (2-5 s) | Catalog IDs don't help (e.g. attribute search). |
| `select_by_hypothesis(hypothesis_json)` | instant | You authored a hypothesis dict explicitly. |

**Default order:** try proposal/frame_neighbor/region/coverage first; reach for
text/hypothesis when none of those map onto the query.

## Per-frame inspection

- `view_keyframe(frame_id, mode='auto')` resolves to:
  - `mode='marked'` for visual grounding (boxes overlaid on proposals).
  - `mode='rgb'` for QA / generic perception.
- Use `categories=...` or `proposal_ids=...` to narrow what's drawn in marked mode.
- `list_frame_proposals(frame_id)` returns a text inventory of the same frame
  without injecting an image (cheap dry-run).

## BEV inspection

- `view_bev()` re-injects the original BEV.
- `view_bev(highlight=[#a, #b])` re-renders with only those proposals labelled
  (declutter trick — use this when the original BEV is too dense to read).

## Catalog queries

- `list_scene_proposals(category=?, region_bev=?, limit=?)` — scene-wide.
- `inspect_proposal(proposal_id)` — proposal metadata + frames_appeared.

## Anti-patterns

- Answering from BEV labels alone. The BEV is a map, not ground truth.
- Calling `select_by_text` first when proposal IDs are visible on the BEV.
- Viewing more than 12 frames before consulting `submit_final`. Budget pressure
  is real; trim aggressively.
- Using `request_crops` before `view_keyframe`. Crops are a zoom; the host frame
  must already be in context.

**Cheapest-first** mantra: **proposal → neighbour → region → coverage → text → hypothesis**.
