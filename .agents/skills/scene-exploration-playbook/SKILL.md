---
name: scene-exploration-playbook
description: Use when solving NR3D/VG samples with text-first and catalog scene exploration tools plus first-person evidence checks.
---

# scene-exploration-playbook

This Codex Agent SDK skill is synchronized from the DeepAgents playbook used by the Stage-2 VG runtime. Regenerate with `scripts/sync_codex_playbook_skills.py` after editing the source playbook.

Codex SDK note: this skill is mounted as a `SkillInput`; the `nr3d_tools` MCP runtime preloads the corresponding tool gates.

Source playbook: `src/agents/skills/shared_skills/scene_exploration_playbook.md`

<!-- BEGIN_SYNCED_PLAYBOOK: src/agents/skills/shared_skills/scene_exploration_playbook.md -->
# Scene Exploration Playbook (shared, prerequisite)

You start each scene with two views: a BEV (top-down) with `#id category`
labels + camera trajectory, and a `SceneCatalog` text inventory. Neither
is first-person evidence — fetch first-person frames before finalising
any answer that depends on appearance, count, state, or relations.

## First move — `select_by_text` for rare-target / OOD queries, catalog for the rest

`select_by_text(query, k≤3)` runs Stage-1 query parsing and returns up
to 3 candidate first-person RGB frames in a single call when the query
can be cleanly parsed. The returned JSON lists each frame's
`visible_proposal_ids` + camera pose so you can decide which of the
≤3 frames is worth a closer look.

For NR3D / VG, call `select_by_text(query, k≤3)` without
`hidden_categories` unless the benchmark explicitly says the **target**
category must be hidden. Do not hide support/anchor/context categories
such as wall, floor, door, bed, desk, or clock; Stage-1 validates the
whole parsed expression, so masking anchors can raise `Masked category leak detected`
and remove the very evidence needed to ground the target.
If you hit that error, retry once with `hidden_categories=[]`, then fall
back to catalog selectors. Keep target candidates and anchor candidates
separate; do not mix them in one broad `require_all=False` selector call.

**Audit-informed routing** (see `docs/benchmark/nr3d/v9_1_select_by_text_audit_20260516.md`):

- **Use `select_by_text` first** when the target is rare (you don't see
  the category in the BEV) or for natural QA queries like "where is the
  kitchen counter". On hard / low-coverage targets it lifts hit@3 by
  ~31 pp over random.
- **Skip `select_by_text` and start with the catalog** when the query is
  view-dependent ("facing X", "right of Y", "standing in the middle"),
  uses negation ("does NOT have"), or names a fine-grained subtype
  beyond the SceneCatalog categories. Stage-1 returns `no_evidence` on
  ~32 % of such queries and you waste a turn. Instead read the BEV +
  Cat-B inventory and jump straight to `select_by_proposal` /
  `select_by_region`.
- If `select_by_text` returns `frames: []` (empty), do **not** retry the
  same query with bigger `k`; switch to `select_by_proposal` /
  `select_by_region` on the candidate catalog IDs you can read from the
  BEV.

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
No text labels by default — that keeps the map readable even in
scenes with 40+ proposals. Read positions from the dots and the
catalog (`list_scene_proposals` / Cat-B inventory) together.

When you need text labels, ask explicitly:

- `view_bev(highlight=[#a, #b])` — re-render the same perspective
  BEV with text labels ("#id category") **only** on those proposals.
  Other dots stay unlabeled. Use after a selector returns candidate
  IDs and you want to see where they sit in the map.
- `view_bev(categories=["chair", "table"])` — text-label every
  proposal whose category matches (case-insensitive exact match).
  Use when you don't have IDs yet but the query is anchored to a
  specific category (e.g. "the leftmost chair").
- Both args may be combined; the union is labeled.

Anti-pattern: calling `view_bev` without args expecting to see
every category labelled. The clean view is intentional —
narrow the label set explicitly.

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

1. Pick the right first move (see the routing block at the top): for
   rare-target / OOD queries call `select_by_text(query)`; for
   view-dependent / catalog-explicit queries call `select_by_proposal` /
   `select_by_region` on the IDs you read from the BEV + inventory.
2. Scan the ≤3 returned frames + the BEV. If `select_by_text` returned
   `frames: []`, fall through to step 1's catalog branch.
3. `mark_frame_with_bbox` on **the** frame that looks decisive.
4. If request_crops is available, optionally call it for fine attributes;
   otherwise refine with another selector if step 3 was inconclusive.
5. `submit_final`.
<!-- END_SYNCED_PLAYBOOK -->
