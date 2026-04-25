# VG Grounding Playbook

## When to use this skill

Load this skill at the start of every visual-grounding task on
EmbodiedScan ScanNet val. It is the prerequisite for the 5 VG-pack
tools — every one of them returns
`"ERROR: load_skill('vg-grounding-playbook') before calling this tool."`
until this skill is loaded into `runtime.skills_loaded`.

You are answering a query of the form "find the X in the scene", where
X may be a single object referent or a referring expression with
spatial constraints. The scene comes with a pre-computed proposal pool
(V-DETR or 2D-CG) and a set of pre-rendered set-of-marks keyframes.
Your job is to pick the proposal id that matches the referent, or to
mark the sample as failed if no proposal in the pool plausibly matches.

## Decision tree

1. `list_keyframes_with_proposals()` — see which keyframes carry which
   proposal ids and how many proposals each frame shows.
2. Pick 1-3 candidate keyframes (prefer frames where the proposal count
   is small and the referent's category is in the candidate list).
3. `view_keyframe_marked(frame_id=N)` for each chosen frame — this
   queues the annotated image for the next user message and returns
   the visible proposal ids and categories.
4. If you have ≥ 2 plausible candidates after looking at the marks,
   call `inspect_proposal(proposal_id=K)` on each to disambiguate by
   category, score, or which other frames the proposal appears in.
5. If the query has a spatial constraint ("next to the desk", "closest
   to the wall"), load the `vg-spatial-disambiguation` skill and apply
   its workflow before submitting.
6. `submit_final({"proposal_id": K, "confidence": C}, rationale=...)`
   — the chassis validator will reject any unknown id and FAIL-LOUD.
7. If the referent genuinely is not in the proposal pool, submit the
   OOD marker (see "OOD handling" below).

## tool: list_keyframes_with_proposals

Inputs: none.

Returns a JSON list, one entry per keyframe in the bundle:
```
[{"keyframe_idx": int,
  "frame_id": int|None,
  "visible_proposal_ids": list[int],
  "n_proposals": int,
  "annotated_image": str (path to a frame_<frame_id>.png on disk)},
 ...]
```

Use this once at the start of every task to pick frames worth viewing.
A frame with `n_proposals=0` is not useful; a frame with
`n_proposals > 20` will usually need spatial disambiguation, not
look-and-pick.

## tool: view_keyframe_marked

Inputs: `frame_id: int` (must be a key in the frame index).

Returns a text body summarizing the chosen frame:
`frame_id=N marked image at <path>; visible_proposals=[...]; categories=[...]`.

Side effect: appends the annotated image path to
`runtime.bundle.extra_metadata["vg_pending_images"]` and marks evidence
updated. The chassis run loop will inject the actual pixel content on
the next user message turn — you do not need to also request a crop.

Errors: `"ERROR: frame_id={N} not in proposal index; available: [...]"`
if the frame has no proposals; `"ERROR: annotated image not found:
{path}"` if the renderer hasn't produced the file (treat as a Stage 1
data bug — switch to a different frame).

## tool: inspect_proposal

Inputs: `proposal_id: int`.

Returns JSON:
```
{"proposal_id": int,
 "category": str,
 "score": float,
 "bbox_3d_9dof": list[float],   # [cx,cy,cz,dx,dy,dz,rx,ry,rz]
 "frames_appeared": list[int],  # frame ids where this proposal is visible
 "source": "vdetr"|"conceptgraph"}
```

Use this when you have a shortlist of 2+ candidates. It cheaply tells
you: which category the detector assigned, how confident it was, and
which other frames you can cross-check.

Errors: `"ERROR: proposal_id={N} not in pool; available count={K}"`.

## tool: find_proposals_by_category

Inputs: `category: str` (case-insensitive, whitespace stripped).

Returns JSON:
```
{"category": str (echoed),
 "proposal_ids": list[int],
 "available_categories": list[str]  (sorted set of all non-empty categories in the pool)}
```

Unknown categories return an empty `proposal_ids` plus the available
list — use that to retry with a closer category guess instead of
giving up.

## tool: compare_proposals_spatial

Inputs:
- `candidate_ids: list[int]`
- `anchor_id: int`
- `relation: "closest_to" | "farthest_from"` (any other value FAIL-LOUDs)

Returns JSON:
```
{"anchor_id": int,
 "relation": str,
 "ranked_ids": list[int],     # candidates ordered by relation
 "distances": list[float]}    # parallel list, Euclidean over bbox centers
```

Errors: bad relation, missing anchor, or any candidate not in the pool
all FAIL-LOUD with explicit error strings. See the
`vg_spatial_disambiguation` skill for the full workflow.

## VgPayload schema

`submit_final(payload, rationale, evidence_refs=[])`:

```
class VgPayload(BaseModel):
    proposal_id: int       # selected pool id, or -1 to mark sample as failed
    confidence: float      # in [0.0, 1.0]
```

The chassis validates `proposal_id` against the pool. Any int that is
neither in the pool nor `-1` raises `ValueError("proposal_id N not in
pool")`, which `submit_final` returns to you as a tool ERROR string.
Do not invent ids.

## OOD handling

If you have inspected every plausible candidate in the pool and none
of them is the referent, do not guess. Submit the failed-sample marker
verbatim:

```
submit_final({"proposal_id": -1, "confidence": 0.0},
             rationale="GT not in proposal pool")
```

This is the ONLY way to cleanly mark a VG sample as failed. The
adapter will emit `{"status": "failed", "selected_object_id": null,
"bbox_3d": null, ...}` so downstream evaluation knows this is a
proposal-pool miss, not a model bug.

## Anti-patterns

- Do NOT call `submit_final` before viewing at least one annotated
  keyframe — proposal ids alone do not tell you what the proposal
  looks like.
- Do NOT re-call `view_keyframe_marked` on a frame you already viewed
  in this run; the image is already in your context.
- Do NOT invent a `proposal_id` that is not in the pool. Use
  `inspect_proposal` to verify before submitting.
- Do NOT skip `find_proposals_by_category` when the query gives a
  clear category — it cheaply reduces a 200-id pool to 2-5 candidates.
- Do NOT submit `proposal_id=-1` just because the first keyframe
  doesn't show the target. Try at least 2 keyframes first.

## Examples

### Example 1: simple referent

Query: "find the chair near the wall on the left."

1. `list_keyframes_with_proposals()` — frame 12 has 3 proposals,
   frame 27 has 5.
2. `find_proposals_by_category("chair")` → `{"proposal_ids": [4, 9, 17]}`.
3. `view_keyframe_marked(frame_id=12)` — frame 12 shows proposals
   `[4, 9]` (both chairs).
4. `inspect_proposal(proposal_id=4)` shows score 0.92 against a wall;
   `inspect_proposal(proposal_id=9)` shows it is in the centre.
5. `submit_final({"proposal_id": 4, "confidence": 0.85},
   rationale="chair against the left wall, visible in frame 12")`.

### Example 2: OOD case

Query: "find the green plant next to the bookshelf."

1. `list_keyframes_with_proposals()` shows 14 frames, all with
   `n_proposals < 8`.
2. `find_proposals_by_category("plant")` → `{"proposal_ids": []}`.
3. `find_proposals_by_category("flower")` → `{"proposal_ids": []}`.
4. After viewing 3 keyframes and inspecting all visible proposals,
   none look like a plant.
5. `submit_final({"proposal_id": -1, "confidence": 0.0},
   rationale="GT not in proposal pool — no plant-like proposals exist
   in the pool, the largest greenery candidate (id=5, category='lamp')
   is clearly not a plant")`.
