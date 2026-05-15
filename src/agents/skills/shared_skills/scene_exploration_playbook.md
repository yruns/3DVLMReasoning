# Scene Exploration Playbook (shared, prerequisite)

You start each scene with two views: a BEV (top-down) with `#id category`
labels + camera trajectory, and a `SceneCatalog` text inventory. Neither
is first-person evidence — fetch first-person frames before finalising
any answer that depends on appearance, count, state, or relations.

## First move — almost always `select_by_text(query)`

`select_by_text(query, k≤3)` runs Stage-1 query parsing and returns up
to 3 candidate first-person RGB frames in a single call. This is the
shortest path from language to ground truth and should be your default
entry point. The returned JSON lists each frame's `visible_proposal_ids`
+ camera pose so you can decide which of the ≤3 frames is worth a
closer look.

## Refinement selectors (after the first batch)

Once `select_by_text` returned frames and you know which catalog IDs
you're tracking, the catalog-driven selectors are instant and inject
their own ≤3 RGB frames:

| Selector | When to use |
| --- | --- |
| `select_by_proposal(ids, require_all, k)` | Frames showing specific catalog IDs (e.g. after text returned wrong instance). |
| `select_by_frame_neighbor(anchor_frame_id, mode='temporal'\|'viewpoint_diverse')` | Expand around one good frame. |
| `select_by_region(region, region_type='bev_2d'\|'bbox_3d')` | A BEV cluster looks promising — fetch frames overlooking that region. |
| `select_by_coverage(method='obj_iou'\|'pose_depth', seen_frame_ids?)` | You need diverse coverage of unseen frames. |

All selectors cap `k` at 3. Image dedup is automatic: frames already
injected are listed with `already_seen=true` and not re-injected.

## Looking closer at one frame

- `mark_frame_with_bbox(frame_id, labels=?, ids=?)` — render a single
  frame with high-contrast bounding boxes around the labels / ids you
  named. Requires at least one of `labels` or `ids`. Use this when a
  selector returned a frame and you want to verify the chosen catalog
  entry against the actual pixels.
- `request_crops(frame_id, bbox_2d)` — pixel zoom into a region of a
  frame you have already seen.

## Catalog queries (text-only)

- `list_scene_proposals(category=?, region_bev=?, limit=?)` — scene-wide.
- `list_frame_proposals(frame_id)` — what's visible in one frame.
- `inspect_proposal(proposal_id)` — proposal metadata + frames_appeared.

## BEV inspection

- `view_bev()` — re-inject the original full BEV.
- `view_bev(highlight=[#a, #b])` — re-render the same perspective BEV
  with only those proposals labelled.

## Anti-patterns

- Answering from BEV labels alone. BEV is a map, not ground truth.
- Calling `mark_frame_with_bbox` on a frame you have not seen yet
  (selectors already injected it; mark is for the "verify this one"
  step).
- Calling more than two selectors before any `mark_frame_with_bbox` —
  that means you are collecting candidates without ever verifying.
- `request_crops` before any selector — crops zoom a frame; the host
  frame must already be in context.

## The loop

1. `select_by_text(query)` (or, when you already know the IDs,
   `select_by_proposal`).
2. Scan the ≤3 returned frames + the BEV.
3. `mark_frame_with_bbox` on **the** frame that looks decisive.
4. (Optional) `request_crops` for fine attributes; or refine with
   another selector if step 3 was inconclusive.
5. `submit_final`.
