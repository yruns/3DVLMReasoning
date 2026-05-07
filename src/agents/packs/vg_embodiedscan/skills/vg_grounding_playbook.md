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
6. `submit_final({"proposal_id": K, "confidence": C}, rationale=...)`
   — the chassis validator will reject any unknown id and FAIL-LOUD.
7. If the referent genuinely is not in the proposal pool **after**
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

Returns JSON. The base shape is always present:

```
{"category": str (echoed),
 "proposal_ids": list[int],
 "available_categories": list[str]  (sorted set of all non-empty categories in the pool)}
```

Unknown categories return an empty `proposal_ids` plus the available
list — use that to retry with a closer category guess instead of
giving up.

### CVRA fields (when `use_clip_visible_aug=True`)

When the runtime has CVRA (CLIP-Visible Retrieval Augmentation) enabled,
the response carries four additional fields that surface visible
proposals whose Mask3D label disagrees with the query category but
whose visual content matches:

```
{"category": str,
 "proposal_ids": list[int],         # ordered union of label_hits + clip_visible_aug
 "available_categories": list[str],
 "label_hits": [
   {"proposal_id": int, "source": "label_exact"}
 ],
 "clip_visible_aug": [
   {"proposal_id": int,
    "clip_score": float,            # cosine vs CLIP-text(category) on the proposal crop
    "rank_used": int,               # 1, 2, or 3 (parser hypothesis rank that triggered the lookup)
    "via_category": str,            # the category string that was searched
    "source_frame_id": str,         # frame whose crop was scored
    "source": "clip_visible",
    "cvra_category_source": str,    # 'rank1' | 'rank2' | 'rank3'
    "cvra_overflow": bool}          # True for the always-on label-mismatch tier
 ],
 "rank_fallback_used": bool,        # rank-2 or rank-3 fired
 "n_visible_set": int,              # |cumulative seen frames|
 "cvra_exhausted": bool}            # all 3 ranks dry — consider switch_or_expand_hypothesis
```

**How to treat `clip_visible_aug` entries** (these are the rule the
agent must internalize — the spec § H promise to the reviewer):

1. `clip_visible_aug` candidates have **unreliable Mask3D labels but
   plausible visual match** to the query category. Mask3D may have
   tagged the GT-overlap bbox with an unrelated category (e.g. a
   pillow labeled `ball`, a desk labeled `monitor`); CVRA flags such
   bboxes by visual similarity. **Do not trust the Mask3D label** for
   these proposals — the reason they appear in `clip_visible_aug` is
   because that label is suspected wrong.
2. **Always inspect the crop before submitting** a `clip_visible_aug`
   id. Use `view_keyframe_marked(frame_id=<source_frame_id>)` or
   `request_crops(...)` to verify the visual content matches the query.
   A high `clip_score` is necessary but not sufficient.
3. Prefer `label_hits` candidates first; treat `clip_visible_aug` as
   a **fallback or augmentation** when label_hits is empty, when label
   hits don't match the spatial/attribute constraints, or when crop
   inspection of the label hit shows it visually disagrees with the
   query (e.g. a "cabinet" label_hit that is at floor level when the
   query says "above the refrigerator").
4. `cvra_overflow=True` entries belong to the always-on label-mismatch
   tier (3 extra candidates per call beyond the standard cap). Treat
   them with the same caution as ordinary `clip_visible_aug` — verify
   via crop inspection before trusting.
5. If `cvra_exhausted=True`, the parser tried all 3 hypothesis ranks
   and got no label hits + no augmented hits at any rank. Consider
   `switch_or_expand_hypothesis(new_query="<reworded>")` rather than
   submitting OOD prematurely.

The CVRA fields are added; the legacy `proposal_ids` /
`available_categories` shape is preserved for backward compatibility
and for runs with `use_clip_visible_aug=False` (the default).

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

`submit_final(payload, rationale, evidence_refs=[], tool_override_reason=None)`:

```
class VgPayload(BaseModel):
    proposal_id: int       # selected pool id, or -1 to mark sample as failed
    confidence: float      # in [0.0, 1.0]
```

The chassis validates `proposal_id` against the pool. Any int that is
neither in the pool nor `-1` raises `ValueError("proposal_id N not in
pool")`, which `submit_final` returns to you as a tool ERROR string.
Do not invent ids.

### When TADG (Tool-Answer Disagreement Gate) fires

When the runtime has TADG enabled
(`runtime.use_tool_answer_disagreement_gate=True`) and you call
`submit_final` with a `proposal_id` that disagrees with the most recent
matched-relation `compare_proposals_spatial` rank-1, the chassis returns
a soft-block message of the form:

```
TADG: compare_proposals_spatial ranked proposal X as rank-1 for
relation 'closest_to' against proposal A; you are submitting proposal Y
(<subcase>).

Either:
  (a) Resubmit with `tool_override_reason='<one-sentence reason>'`
      to record the explicit divergence, OR
  (b) Revise to proposal X (or another id from ranked_ids=[...])
      and resubmit.
```

This is **not** an error — it's a forced moment of reflection. Three
ways to respond, in order of preference:

1. **Revise** to the ranked rank-1 if the spatial tool's verdict is
   correct (most common — the agent miscounted distance or had a
   stale reading). Just call `submit_final` again with that pid.
2. **Override with a reason** if you have evidence the spatial tool's
   rank-1 is wrong (e.g. you just inspected its crop and it is a
   different category than the query asks for, or it is the anchor
   itself, or its bbox is occluded). Pass
   `tool_override_reason="<one short sentence>"` (≥6 chars) when you
   resubmit. The reason is recorded in the audit trail.
3. **Inspect more before deciding**: call `view_keyframe_marked` or
   `inspect_proposal` on the spatial tool's rank-1 candidate, then
   submit either rank-1 (if it now looks right) or your original pick
   with `tool_override_reason`.

Anti-loop guard: if you resubmit the same `proposal_id` 3 times
without a `tool_override_reason`, the gate auto-passes the
submission to avoid sample crashes. Use the override path
proactively rather than relying on the force-pass.

Subcases of the soft-block message:

- **"the anchor itself"** — you are submitting the same proposal_id
  you used as `anchor_id`. This is structurally suspicious for queries
  with a single relation; reconsider whether the query is symmetric
  ("X next to another X"), in which case enumerate same-category pairs
  rather than locking the anchor.
- **"not in the spatial-tool's candidate set"** — your submitted
  proposal_id was never passed as a candidate to the spatial tool.
  Either the spatial tool was called with the wrong candidate list,
  or you switched targets without re-running the spatial check.
- **"ranked below position 1"** — your submission is in the candidate
  list but not rank-1. Either the rank-1 is genuinely better and you
  should revise, or you have specific evidence (provide an override).

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
