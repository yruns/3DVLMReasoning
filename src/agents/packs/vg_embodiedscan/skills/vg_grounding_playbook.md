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

This is a ReAct loop. The 1-3 initial keyframes are a *starting point*,
not the final evidence. You have **four independent paths** to acquire
fresh visual evidence when the initial keyframes don't show the target,
ordered cheapest-first:

- **`view_keyframe_marked(frame_id=N)`** — instant. For any N in the
  scene-wide frame index (typically 50-300 frames per scene). Use this
  when the initial keyframes show the right *region* but not the
  specific instance — e.g. you have category candidates from
  `find_proposals_by_category` and want to verify each in a frame
  where Mask3D actually marked it.
- **`request_more_views(request_text, mode="targeted"|"explore"|"temporal_fan", object_terms=[...], frame_indices=[...])`**
  — Stage-1 visibility-driven view fetch. Cheap (no LLM). Use when
  you need additional views *centered on specific scene objects*
  (`mode="targeted"` + object_terms), or when you want views maximally
  different from what you've seen (`mode="explore"`), or temporal
  neighbors of an anchor frame (`mode="temporal_fan"` + frame_indices).
- **`request_crops(request_text, object_terms=[...])`** — generates
  a zoomed-in red-bbox crop around each named scene object. Use when
  the proposal is visible but small / occluded in the existing
  frames. CLIP-fallback object matching, ~moderate cost.
- **`switch_or_expand_hypothesis(new_query="...")`** — calls Stage 1
  again with a refined query string and **appends** new keyframes
  (1-3 more). Most expensive (LLM hypothesis parse). Use as last
  resort when the initial keyframes don't cover the right *region*
  of the scene at all — e.g. Stage 1 mis-parsed the utterance, or
  the target's category isn't represented in any of the initial
  frames AND `find_proposals_by_category` returned nothing useful.
  This is the project's **Stage 2 → Stage 1 callback** loop (mirrors
  OpenEQA).

Keep iterating with these four tools until you actually see the target
clearly, or until you have proven the referent is not in the proposal
pool. Prefer cheaper paths first.

1. `list_keyframes_with_proposals()` — see which **initial** keyframes
   carry which proposal ids and how many proposals each frame shows.
2. **Identify target category and find ALL same-category candidates
   scene-wide** — call `find_proposals_by_category("<category>")`. The
   returned `proposal_ids` cover the whole scene's Mask3D pool, not
   just the initial keyframes.
3. **Decide where to look:**
   - If the initial keyframes contain ≥ 1 same-category candidate
     (overlap between their `visible_proposal_ids` and step-2 ids):
     `view_keyframe_marked(frame_id=N)` on those frames.
   - If the initial keyframes carry NO same-category candidate but
     `find_proposals_by_category` returned a non-empty list (the
     candidates exist in the pool, just not in your initial frames):
     navigate via `inspect_proposal(K)` → `frames_appeared` →
     `view_keyframe_marked(frame_id=M)` on those frames.
   - If `find_proposals_by_category` ALSO returned an empty list (no
     candidate of that category anywhere in the pool), use
     `switch_or_expand_hypothesis(new_query="<a more general or
     synonymous phrasing of the utterance>")` to ask Stage 1 for
     entirely new keyframes. Stage 1's hypothesis parser may catch a
     synonym or attribute the original parse missed.
4. If you have ≥ 2 plausible candidates after looking at the marks,
   call `inspect_proposal(proposal_id=K)` on each to disambiguate by
   category, score, or which other frames the proposal appears in.
5. If the query has a spatial constraint ("next to the desk", "closest
   to the wall"), load the `vg-spatial-disambiguation` skill and apply
   its workflow before submitting.
6. **Multi-distractor forced choice (MANDATORY when applicable).**
   Whenever `find_proposals_by_category(target_category).proposal_ids`
   has length ≥ 2 AND you cannot uniquely eliminate all but one via
   spatial filters alone, you MUST call
   `select_among_proposals(candidate_ids=<remaining ids>,
   description=<the user's query verbatim>)` BEFORE `submit_final`.
   The tool runs a single multimodal VLM call with one marked frame
   per candidate and returns the verdict id. Use that id (unless you
   have just-acquired contradictory evidence). This step exists
   because turn-1 ranking on multi-distractor utterances is the main
   failure mode of this pipeline; the forced 1-of-K call is a
   discipline you cannot skip with "I'm pretty sure it's K".
7. `submit_final({"proposal_id": K, "confidence": C}, rationale=...)`
   — the chassis validator will reject any unknown id and FAIL-LOUD.
8. If the referent genuinely is not in the proposal pool **after**
   exhausting same-category candidates, viewing 3+ frames, AND trying
   at least one `switch_or_expand_hypothesis` rewrite, submit the OOD
   marker (see "OOD handling" below).

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

Inputs: `frame_id: int` (must be a key in the **scene-wide frame
index**, NOT just the initial 5 keyframes — the index covers every
frame in the scene where at least one Mask3D proposal is visible,
typically 50-300 frames per scene).

Returns a text body summarizing the chosen frame:
`frame_id=N marked image at <path>; visible_proposals=[...]; categories=[...]`.

Side effect: appends the annotated image path to
`runtime.bundle.extra_metadata["vg_pending_images"]` and marks evidence
updated. The chassis run loop will inject the actual pixel content on
the next user message turn — you do not need to also request a crop.

Use this aggressively for cross-frame navigation:
1. `find_proposals_by_category("X")` → list of proposal ids for class X.
2. `inspect_proposal(K)` → `frames_appeared = [...]` for proposal K.
3. `view_keyframe_marked(frame_id=M)` for any M in `frames_appeared`,
   even if M was not in the initial keyframes shown by
   `list_keyframes_with_proposals`.

Errors: `"ERROR: frame_id={N} not in proposal index; available: [...]"`
— the error message lists the first 20 valid frame_ids; pick one of
those instead of a random integer. `"ERROR: annotated image not found:
{path}"` — treat as a Stage 1 data bug, switch to a different frame.

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
- `relation: "closest_to" | "farthest_from" | "above" | "below"`
  (any other value FAIL-LOUDs)

Returns JSON:
```
{"anchor_id": int,
 "relation": str,
 "ranked_ids": list[int],     # candidates ordered by relation
 "distances": list[float]}    # parallel list, Euclidean over bbox centers
```

For `above` / `below`, the tool ranks by bbox-center z offset relative
to the anchor and also returns `vertical_offsets`; use this for vertical
relations like "cabinet above the refrigerator" or "box below the table"
instead of forcing those cases through `closest_to`. Errors: bad
relation, missing anchor, or any candidate not in the pool all FAIL-LOUD
with explicit error strings. See the `vg_spatial_disambiguation` skill
for the full workflow.

## tool: select_among_proposals (forced 1-of-K disambiguation)

Inputs:
- `candidate_ids: list[int]` — the same-category Mask3D candidates that
  remain after geometric/visibility filtering. MUST have length >= 2.
- `description: str` — the natural-language description of the target
  (typically the user's original query verbatim).

Returns JSON:
```
{"selected_proposal_id": int,    # one of candidate_ids
 "reasoning": str,                # one short visual cue
 "candidate_ids": list[int]}      # echo of input
```

What it does internally: composes one marked frame per candidate
(the most-visible annotated PNG, with the proposal_id labeled), bundles
them with the description into a SINGLE multimodal VLM call, and asks
"which proposal_id matches?". The judge VLM sees all candidates side
by side and picks one. This is a forcing function: the agent cannot
talk itself into a confident wrong answer when 3+ same-category
proposals exist — it must compare them visually in one shot.

When to call:
- BEFORE `submit_final` whenever the target category has >= 2 viable
  proposals after `find_proposals_by_category` + spatial filtering.
  Skip only when `find_proposals_by_category(target).proposal_ids`
  has length 1.
- AFTER you have read the marked frames yourself, so the candidate
  list is already pruned. Don't pass all 7 same-category proposals
  if 4 of them are clearly disqualified (e.g., wrong wall in a
  spatial relation) — pass the 2-3 still in contention.

The returned `selected_proposal_id` is the VLM judge's verdict; you
should usually `submit_final(proposal_id=selected_proposal_id)` next
unless you have new contradictory evidence to inspect.

Errors: candidate_ids length < 2, candidate not in pool, candidate
invisible in any current frame, missing annotated PNG, malformed VLM
JSON, or VLM picking an id outside `candidate_ids` all FAIL-LOUD with
explicit error strings.

## tool: switch_or_expand_hypothesis (chassis tool, Stage 2 → Stage 1 callback)

Inputs:
- `request_text: str` — human rationale for re-running Stage 1.
- `new_query: str` — REQUIRED — the alternative retrieval query string
  (e.g. `"patio chair near the kitchen"` instead of the original
  `"this is a brown chair next to the door"`).
- `preferred_kind: str` — optional hint for hypothesis kind
  (`direct` / `proxy` / `context`).

Calls Stage 1's `KeyframeSelector.select_keyframes_v2(new_query, k=3)`
and **appends** the new keyframes to your bundle. After this returns,
the chassis injects the new annotated images into your next user
message turn — like `view_keyframe_marked` but in batches of 1-3.

When to use:
- **The initial keyframes don't show the right region of the scene at
  all.** E.g., the utterance is "a printer on the desk in the office"
  but Stage 1 picked living-room frames. A re-query with
  `new_query="printer on desk"` may pull office-area views.
- **`find_proposals_by_category` returned an empty list.** The
  detector may have labeled the target with a synonym; a re-query
  with a broader or differently-worded category sometimes pulls in
  Stage 1 hypotheses that catch it.
- **You've viewed 3+ frames and the target category just isn't
  visible anywhere you've looked.** A fresh Stage 1 query with a more
  specific or more general phrasing can break the loop.

When NOT to use:
- The initial keyframes show the right region but you need a
  different *angle* of the same object — use `view_keyframe_marked`
  with another frame from `inspect_proposal[K].frames_appeared`
  instead. Re-running Stage 1 is more expensive (~2-5 s LLM call)
  than `view_keyframe_marked` (instant).
- You haven't yet called `find_proposals_by_category` on the original
  query — try that cheap option first.

Cost: each call triggers a Stage 1 LLM parse (~2-5 s). Treat 1-2
calls per sample as normal; 3+ as a sign you should consider OOD.

Returns text like:
`Re-ran Stage 1 with query 'printer on desk': added 2 new keyframe(s).
Total keyframes now: 5.`

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
- **Do NOT pick a proposal whose category obviously mismatches the
  query just because it appears in the initial 5 keyframes.** If the
  initial 5 carry no same-category candidate, navigate via
  `find_proposals_by_category` → `inspect_proposal[K].frames_appeared`
  → `view_keyframe_marked(frame_id=M)` to a frame that actually
  shows your target. The initial keyframes are a starting point, not
  a constraint.
- **Do NOT submit `proposal_id=-1` if `find_proposals_by_category`
  returned a non-empty list.** OOD only applies when no same-category
  candidate exists in the entire pool — exploring frames beyond the
  initial 5 first is mandatory.

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

### Example 1b: cross-frame navigation when initial 5 miss the target

Query: "this is a brown wooden chair next to the kitchen counter."

Initial 5 keyframes from Stage 1 happen to show the dining area
without the kitchen.

1. `list_keyframes_with_proposals()` — keyframes are frames `[42, 58,
   71, 89, 102]`; their `visible_proposal_ids` cover dining-table-area
   proposals like `[2, 5, 8, 11]` (categories: dining table, dining
   chair, dining chair, lamp). No counter, no kitchen chair.
2. `find_proposals_by_category("chair")` → `{"proposal_ids": [5, 8,
   17, 23]}`. Two of those (`5, 8`) are dining chairs already
   visible; `17, 23` are not in any of the initial 5 keyframes —
   they're the kitchen chairs we want.
3. `inspect_proposal(proposal_id=17)` → `frames_appeared=[155, 168,
   201]`, category `chair`. `inspect_proposal(proposal_id=23)` →
   `frames_appeared=[180, 199]`, category `chair`.
4. `view_keyframe_marked(frame_id=155)` — annotates frame 155 (NOT
   in the initial 5) and shows proposals `[17, 19]` over the kitchen
   counter. The brown wooden chair is `17`.
5. `view_keyframe_marked(frame_id=199)` — confirms `23` is also a
   chair near the counter but plastic, not wooden.
6. `submit_final({"proposal_id": 17, "confidence": 0.85},
   rationale="brown wooden chair next to kitchen counter; verified
   via cross-frame navigation to frame 155")`.

The initial 5 keyframes did not contain proposal 17 or 23, but the
agent recovered by category lookup → frames_appeared → cross-frame
view. **This is the standard ReAct recovery; do not stop at the
initial 5.**

### Example 1c: Stage 2 → Stage 1 callback (re-query for new keyframes)

Query: "this is the printer next to the desk in the office."

Stage 1's initial 3 keyframes are all in the living room — its
hypothesis parser fixated on "desk" and pulled the most-visible desk,
which happens to be a writing desk in the living area.

1. `list_keyframes_with_proposals()` — initial keyframes
   `[18, 34, 47]`; `visible_proposal_ids` cover sofas, lamp, coffee
   table, the writing desk. No printer.
2. `find_proposals_by_category("printer")` → `{"proposal_ids": []}`
   — but `available_categories` lists `["printer-laser",
   "monitor", ...]` so a printer-like proposal does exist.
3. `find_proposals_by_category("printer-laser")` → `{"proposal_ids":
   [29]}`. Found it.
4. `inspect_proposal(proposal_id=29)` → `frames_appeared=[112, 138]`.
   Both frames are NOT in the initial 3.
5. `view_keyframe_marked(frame_id=112)` — annotated frame 112 shows
   the printer (`29`) on a desk in the office.
6. `submit_final({"proposal_id": 29, "confidence": 0.85},
   rationale="laser printer on the office desk, visible in frame
   112; Stage 1 mis-routed initial KFs to the living-room desk so we
   navigated via category lookup")`.

Notice that `switch_or_expand_hypothesis` was *not* needed here —
`find_proposals_by_category` + cross-frame navigation was enough.
**Use `switch_or_expand_hypothesis` only when category lookup ALSO
returns empty / inconclusive.** Here is when it WOULD be needed:

Query: "this is the floor lamp by the bay window."

Initial keyframes don't show any window. `find_proposals_by_category("lamp")`
returns proposals 4, 11, 18 — but you've viewed them all and none is
near a window:

7. `switch_or_expand_hypothesis(request_text="initial KFs don't show
   any window; need bay-window views", new_query="floor lamp near
   the window in the bedroom")` — Stage 1 returns 2 new keyframes
   pulled from a different region.
8. `view_keyframe_marked(frame_id=144)` — the new frame shows lamp
   `18` next to a bay window. Confirmed.
9. `submit_final({"proposal_id": 18, "confidence": 0.80},
   rationale="floor lamp by bay window; verified after Stage 1
   re-query with 'floor lamp near the window'")`.

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
