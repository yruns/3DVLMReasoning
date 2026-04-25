# Evidence Scouting

## When to use this skill

Load this skill whenever your current evidence is insufficient to
answer with confidence. Sufficiency check before submitting:

- Is the target object actually visible in at least one keyframe?
- For spatial queries, is the anchor visible in the same keyframe (or
  in a paired keyframe with overlapping coverage)?
- For relations like "next to" or "between", does the existing
  evidence let you judge the geometric relation, not just the
  presence?
- For category-fine queries ("the orange chair"), is the resolution
  high enough to verify the discriminating attribute?

If any answer is "no", do not submit. Use this skill to acquire what
is missing.

This skill is **task-agnostic**: it documents two chassis tools
(`request_more_views` and `request_crops`) that exist for every
task type. The QA pack will load this same skill once it migrates in
Plan B step 7. Keep the wording neutral — do not embed VG-specific
heuristics.

## tool: request_more_views

Inputs:
```
request_text: str          (free-form description of what you need to see)
frame_indices: list[int]   (preferred pins, capped at max_additional_views; excess dropped)
object_terms: list[str]    (object names to retrieve views for, used by 'targeted' mode)
mode: "targeted" | "explore" | "temporal_fan"   (default 'targeted')
```

Returns: free-form text from the configured Stage-1 callback summarising
which views were retrieved and how the bundle changed. The chassis
auto-injects the new image content on the next user message turn —
you do not have to view them separately.

Mode selection rules:

- **`targeted`** (default) — use when you have specific objects to
  ground (`object_terms=["chair", "desk"]`) or specific frames in mind
  (`frame_indices=[12, 27]`). Stage 1 finds views maximally covering
  those objects/frames. Best when you know exactly what's missing.

- **`explore`** — use when you suspect the existing keyframes share
  the same viewpoint and you need geometric variety (e.g. you have 3
  frames but they all look at the room from the door; you want a view
  from the back). `object_terms` is unused here; `request_text`
  describes the desired viewpoint diversity.

- **`temporal_fan`** — use when you need temporal neighbours around a
  specific frame (e.g. the agent is reasoning about a moving event or
  a viewpoint just before/after a known frame). Requires
  `frame_indices=[anchor_frame_id]`. Note: this mode is gated by
  `Stage2DeepAgentConfig.enable_temporal_fan`. If disabled, the tool
  returns `"temporal_fan mode disabled in this run; use targeted or
  explore"` — fall back to `targeted` with the same anchor frame.

`request_text` patterns that work well:

- `"need a view that shows X in clear focus, ideally from a different
  angle than the existing keyframes"` (targeted)
- `"need a view from the back of the room to verify spatial layout"`
  (explore)
- `"need temporal neighbours around frame 142 to verify the event"`
  (temporal_fan)

## tool: request_crops

Inputs:
```
request_text: str          (what you want cropped and why)
frame_indices: list[int]   (frames to crop from)
object_terms: list[str]    (objects to crop around)
```

Returns: free-form text from the configured callback summarising the
crops produced. Same auto-inject behaviour as `request_more_views`.

Use this when you have the right viewpoint already but need higher
resolution on a specific object. Common patterns:

- A small target in a wide-angle shot — request an object-centric crop
  to read fine-grained category attributes.
- A cluttered region with several candidates — request a region crop
  that includes all of them so you can compare side-by-side.

Do NOT use crops for "different viewpoint" needs — that is what
`request_more_views` is for.

## Sufficiency criteria checklist

Before submitting any final answer:

1. **target visibility**: Have I seen at least one annotated frame
   where the target proposal is visible? If not, run
   `request_more_views(mode="targeted", object_terms=[target_category])`.
2. **anchor visibility** (spatial queries only): Same check for the
   anchor; if missing, request a view that covers both target and
   anchor together.
3. **geometric verification**: For "next to" / "between" / "above",
   can I see the relative positions in a single rendered frame? If
   not, request `mode="explore"` for a wider view, or run
   `request_crops` for a region that contains both.
4. **category verification**: Does the proposal actually look like
   what its `category` field claims? If the category is generic
   ("object"), request a crop on the proposal to verify.
5. **OOD candidate**: If you have run all of the above and still no
   plausible target, the GT is likely not in the proposal pool.
   Submit the failed-sample marker with a rationale citing what you
   tried.

## Anti-patterns

- Do NOT call `request_more_views` 3 times with the same
  `object_terms` and `mode="targeted"` — Stage 1 will return the same
  views every time. Vary `mode`, `frame_indices`, or the
  `object_terms` set.
- Do NOT call `request_crops` to get "a different angle" — crops
  cannot rotate viewpoints. Use `request_more_views(mode="explore")`.
- Do NOT call `request_more_views(mode="temporal_fan")` without
  knowing whether `enable_temporal_fan` is on; if off, the tool will
  tell you, but it costs a round trip. Prefer to check existing
  keyframes for temporal context first.
- Do NOT request more evidence after you already have enough. Each
  call costs token budget. If you can answer with current evidence,
  call `submit_final` instead.
- Do NOT chain `request_more_views` → `request_more_views` → ...
  more than 2-3 times in a row without a `view_keyframe_marked` or
  `inspect_proposal` in between to actually use what you got.

## Examples

### Example 1: missing target

After reviewing the initial 4 keyframes, none shows the small "remote
control" the user asked for.

1. `request_more_views(request_text="need views that show small
   electronics on the coffee table", object_terms=["remote control",
   "remote", "coffee table"], mode="targeted")`.
2. The chassis injects 2 new frames; one of them shows the remote.
3. Continue with the task-specific workflow (e.g. VG: pick a
   `view_keyframe_marked` and proceed).

### Example 2: insufficient resolution

You found the target proposal id 14 ("vase") but at 30 px tall in the
existing keyframe, and the user asked which colour it is.

1. `request_crops(request_text="zoom in on proposal 14 to read its
   colour", frame_indices=[8], object_terms=["vase"])`.
2. The chassis injects a tight crop showing the vase clearly.
3. Read the colour from the crop and submit the final answer.

### Example 3: viewpoint diversity (explore)

All 4 keyframes are taken from the same corner; the user asked
about something on the opposite wall.

1. `request_more_views(request_text="need a view from the opposite
   side of the room to see the back wall", mode="explore")`.
2. The chassis returns 1-2 frames from a different viewpoint.
3. Resume the task-specific workflow on the new frames.
