# Scene Exploration Playbook — catalog-first variant

You start each scene with two views: a BEV (top-down) with `#id category`
labels + camera trajectory, and a `SceneCatalog` text inventory. Neither
is first-person evidence — fetch first-person frames before finalising
any answer that depends on appearance, count, state, or relations.

> **Why this variant exists.** The Stage-1 audit
> (`docs/benchmark/nr3d/v9_1_select_by_text_audit_20260516.md`) found that
> on NR3D random100 the Stage-1 language→frame retrieval tool returns
> `parse=no_evidence` for **32 %** of raw queries, mostly for
> spatial-anchor phrasing ("X next to Y", "facing X", "right of"). In
> this run that tool has been removed so the agent never spends a turn
> on a query Stage-1 cannot ground.

## First move — catalog reading + `select_by_proposal`

1. Read the BEV image labels and the `Proposals by category:` block (Cat-B
   inventory in the task message). Identify candidate catalog IDs from the
   query categories.
2. **Fetch frames** with `select_by_proposal(proposal_ids=[#a, #b, ...],
   require_all=False, k=3)`. This returns up to 3 first-person RGB frames
   that each show at least one of the named IDs, along with
   `visible_proposal_ids` and the camera pose.
3. If the BEV does not give you a confident shortlist (e.g. the query
   names a fine-grained subtype), expand the candidate set via
   `list_scene_proposals(category=…)` or `list_frame_proposals(frame_id)`
   first, then call `select_by_proposal`.

## Refinement selectors (after the first batch)

| Selector | When to use |
| --- | --- |
| `select_by_proposal(ids, require_all, k)` | Frames showing specific catalog IDs (your default). |
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
- If request_crops is available:
  `request_crops(request_text, frame_indices=[...], object_terms=[...])`
  is optional pixel zoom for fine attributes. Only treat it as visual
  evidence when it returns crop image outputs. If it returns `ERROR` or
  `No crops generated`, do not cite it as evidence and rely on
  `mark_frame_with_bbox` / selector frames instead.

## Catalog queries (text-only)

- `list_scene_proposals(category=?, region_bev=?, limit=?)` — scene-wide.
- `list_frame_proposals(frame_id)` — what's visible in one frame.
- `inspect_proposal(proposal_id)` — proposal metadata + frames_appeared.
  When available, it also returns the full enriched object description
  (`description`, `location`, `nearby_objects`, `color`, `usability`);
  call it when compact proposal notes or category labels are not enough.

## BEV inspection

The initial BEV in your context is a **clean overview**: mesh +
camera trajectory + a small unlabeled dot at each proposal centroid.
No text labels by default. Read positions from the dots and the
catalog (`list_scene_proposals` / Cat-B inventory) together.

When you need text labels, ask explicitly:

- `view_bev(highlight=[#a, #b])` — re-render the same perspective
  BEV with text labels ("#id category") **only** on those proposals.
- `view_bev(categories=["chair", "table"])` — text-label every
  proposal whose category matches (case-insensitive exact match).
  Catalog-first variant favours this form over the (omitted) text
  selector when you want to anchor on a category before scanning IDs.
- Both args may be combined; the union is labeled.

Before finalizing an ordering, closest/farthest, size, or superlative query,
make sure every plausible same-category candidate in a small candidate set has
marked evidence or an explicit elimination reason. A selector returning frames
for only one spatial cluster is not enough to eliminate unseen same-category
candidates.

For view-dependent left/right queries ("facing the X", "from behind",
"looking in from the door", "standing at the foot/end"), first establish the
viewer/anchor frame, then compare target candidates in that same marked frame.
Use `select_by_proposal(..., require_all=True)` when the candidate set is small
enough to seek a co-visible frame. Do not combine screen-left/right evidence
from different camera viewpoints or different candidate pairs as if it were one
ordering. If a previous spatial comparison narrowed the set, apply left/right
only inside that subset.

## Anti-patterns

- Answering from BEV labels alone. BEV is a map, not ground truth.
- Calling `mark_frame_with_bbox` on a frame you have not seen yet
  (selectors already injected it; mark is for the "verify this one"
  step).
- Calling more than two selectors before any `mark_frame_with_bbox` —
  that means you are collecting candidates without ever verifying.
- If request_crops is available, calling it before any selector — crops
  zoom a frame; the host frame must already be in context.

## The loop

1. Read BEV labels + Cat-B inventory → pick candidate proposal IDs.
2. `select_by_proposal(ids=[…])` (or `select_by_region` for "in the
   corner / on the island" queries with no clean catalog handle).
3. Scan the ≤3 returned frames + the BEV.
4. `mark_frame_with_bbox` on **the** frame that looks decisive.
5. If request_crops is available, optionally call it for fine attributes;
   otherwise refine with another selector if step 4 was inconclusive.
6. `submit_final`.
