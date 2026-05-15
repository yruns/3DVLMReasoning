# v9.1 — Selectors Return Images + Mark-Frame-With-Bbox Tool + BEV Quality Fixes

**Date:** 2026-05-15
**Status:** Draft — awaiting user spec review.
**Parent design:** `docs/superpowers/specs/2026-05-14-v9-catalog-first-scene-exploration-design.md`
**Touches:** Stage 2 agent tool surface, `scene_perception.py`, `scene_bev_builder.py`, playbooks, runtime, trace HTML generator.

## 1. Motivation

The v9 catalog-first refactor put BEV + `SceneCatalog` in front of the agent
and forced first-person frames to be fetched on demand. Three things turned
out wrong in practice:

1. **`select_by_text` returns text-only.** The agent then has to call
   `view_keyframe` once per candidate frame to actually see anything.
   That doubles the round trips and creates a default behaviour of
   "look at every frame the selector returned", which is the opposite of
   "decide which frame is worth examining". The `select_by_text` →
   text → `view_keyframe` chain is also the single most predictive Stage-1
   shortcut: language-to-frame retrieval is **the** key innovation, and
   the prompt should reflect that.
2. **`select_by_hypothesis` is dead weight.** It overlaps with
   `select_by_text` (which already executes the parsed hypothesis under
   the hood) and is rarely called by the agent because authoring a
   hypothesis dict in JSON costs more tokens than a free-form query.
3. **BEV is unusable as a planning surface.** ~70% of the canvas is blank
   space (perspective render with a fixed fov, no post-render crop),
   labels at `cv2.FONT_HERSHEY_SIMPLEX scale=0.45 thickness=1` blur out
   when the agent ingests the image, and `view_bev(highlight=[...])` puts
   labels in the wrong place because the highlight path uses a
   linear-bounds projection while the original BEV is rendered with a
   perspective camera.

## 2. Goals

- Make `select_by_text` the universal first move: returns up to 3 first-person
  RGB frames + metadata in a single call.
- Make every selector return images, not text-only fids.
- Replace `view_keyframe` with a single dedicated `mark_frame_with_bbox`
  tool. Use it when the agent has decided a frame is worth examining and
  wants a high-contrast annotated zoom.
- Delete `select_by_hypothesis`.
- Fix BEV: crop blank canvas, double the label legibility, fix the
  highlight projection bug.

## 3. Non-Goals

- Changing the BEV builder's underlying camera model (still perspective).
- Touching `request_crops`, `view_bev` non-highlight path, catalog tools
  (`list_scene_proposals`, `list_frame_proposals`, `inspect_proposal`).
- New budget machinery. Keep relying on `seen_image_paths` +
  `submit_final` guards.

## 4. New Tool Surface

| Tool | Purpose | Images injected | Replaces |
| --- | --- | --- | --- |
| `select_by_text(query, k≤3)` | Stage-1 language → frame, **primary entry** | ≤3 RGB | text-only `select_by_text` |
| `select_by_proposal(proposal_ids, require_all, k≤3)` | Frames containing proposal IDs | ≤3 RGB | text-only |
| `select_by_frame_neighbor(anchor_frame_id, mode, k≤3)` | Temporal / viewpoint-diverse neighbours | ≤3 RGB | text-only |
| `select_by_region(region, region_type, k≤3)` | Frames overlooking a BEV/3D region | ≤3 RGB | text-only |
| `select_by_coverage(method, k≤3, seen_frame_ids?)` | Diverse coverage of unseen frames | ≤3 RGB | text-only |
| `mark_frame_with_bbox(frame_id, labels?, ids?)` | Single-frame annotated zoom | 1 marked PNG | `view_keyframe(mode='marked')` |
| `view_bev(highlight?)` | BEV (full or filtered) | 1 BEV PNG | unchanged signature, internals fixed |
| `list_scene_proposals`, `list_frame_proposals`, `inspect_proposal`, `request_crops`, `submit_final` | (unchanged) | — | — |

**Removed:** `select_by_hypothesis`, `view_keyframe` (both `mode='rgb'` and
`mode='marked'`), the now-unused `_left_to_right_entries` /
`_render_filtered_marked` helpers in `view_keyframe.py`.

### 4.1 Selector return contract

All five selectors return the **same shape** of JSON, with `image_paths`
added so callers can verify which images were just injected this turn:

```json
{
  "hypothesis_summary": "target='chair' kind='direct'",
  "frames": [
    {
      "frame_id": 142,
      "visible_proposal_ids": [4, 7, 12],
      "bev_xy": [3.1, 1.4],
      "camera_yaw": 1.57,
      "selected_because": "select_by_text(query='wooden chair')",
      "image_path": "/path/to/frame_142.png",
      "already_seen": false
    },
    ...
  ]
}
```

- `image_paths` for `already_seen=true` frames are still listed, but those
  paths are **not** queued via `queue_pending_image` — only new frames
  inject into context.
- `k` is hard-capped at 3 inside each selector regardless of caller input.
  `k=4` becomes `k=3` with a one-line warning in the tool response.

### 4.2 `mark_frame_with_bbox` contract

```text
mark_frame_with_bbox(
    frame_id: int,
    labels: list[str] | None = None,    # category names, e.g. ["chair", "wooden chair"]
    ids: list[int] | None = None,       # explicit proposal IDs
) -> str
```

Semantics:
- **At least one of `labels` or `ids` must be non-empty.** Empty input
  returns `ERROR: mark_frame_with_bbox requires at least one of {labels, ids}; to view plain RGB, request the frame via a selector`.
- `frame_id` must be in the catalog's `valid_frame_ids`. Otherwise
  `ERROR: frame_id=N not in valid_frame_ids; available[:20]=[...]`.
- All proposals visible at `frame_id` whose `category` matches any entry
  in `labels` (case-insensitive, whitespace-normalised) OR whose
  `proposal_id` is in `ids` are drawn.
- If filters are non-empty but match nothing in this frame:
  `ERROR: no visible proposals matched filters for frame_id=N; filtered_by={...}; visible_proposals=[...]`.

Rendering spec (see §5):
- Bbox stroke = 5-colour palette, rotating by index of match
- 2-pixel black outer stroke around the colour stroke
- Label `#id category` placement: area-ratio rule
- Opaque black label background, white text
- Cache filename: `frame_{frame_id}_ids_{sorted_id_csv}.png` under
  `<catalog.bev_image_path>.parent / "filtered_marks"`

**Response text schema** (CRITICAL — guards parse this verbatim):

```text
frame_id=<int> mark image at <path>;
 filtered_by={'labels': [...], 'ids': [...]};
 visible_proposals=[<int>, ...];
 categories=[<str>, ...];
 left_to_right=['<id>:<category>', ...];
 boxes_2d={<int>: [x1, y1, x2, y2], ...}
```

The `visible_proposals` / `categories` / `left_to_right` / `boxes_2d`
fields are the same ones `evidence_frame_guard` and `no_match_guard`
parse from the old `view_keyframe(mode='marked')` output. Keeping the
exact field names + delimiter style lets the guards switch tool name
without rewriting their regexes (`_VISIBLE_RE`, `_CATEGORIES_RE`,
`_LEFT_TO_RIGHT_RE`, `_BOXES_2D_RE`).

### 4.3 Image dedup

- Selectors call a new helper
  `agents.runtime.scene_runtime.queue_pending_image_if_new(runtime, path)`
  which is a no-op when `path in runtime.seen_image_paths`, otherwise
  delegates to the existing `queue_pending_image`. The JSON response
  always lists every candidate frame's metadata (including its
  `image_path` and an `already_seen` flag) so the agent knows what's
  available even when no new image is injected.
- `mark_frame_with_bbox` writes a deterministic per-(frame, ids) PNG and
  always queues it (different `ids` ⇒ different file ⇒ different image).

## 5. `mark_frame_with_bbox` Rendering Spec

### 5.1 Bbox style

- Stroke palette (rotates by match index):
  `[(34, 197, 94), (239, 68, 68), (59, 130, 246), (234, 179, 8), (168, 85, 247)]`
  (green, red, blue, yellow, purple).
- Stroke thickness: `max(5, img_width // 200)` px (about 1.5× the legacy
  `view_keyframe(mode='marked')` width).
- **Black 2-pixel outline** drawn first on a stroke 2px thicker, then the
  colour stroke drawn on top. Result: colour bbox sits inside a thin
  black silhouette. Robust against both bright and dark backgrounds.

### 5.2 Label placement

For each drawn bbox compute `area_ratio = text_w * text_h / (bbox_w * bbox_h)`.

- `area_ratio < 0.15` ⇒ **centre placement**:
  - text origin `(cx - text_w/2, cy + text_h/2)` (cv2 baseline-aligned)
  - background rect tight around text +4 px padding all sides
- otherwise ⇒ **top-left INSIDE bbox**:
  - text origin `(x1 + 4, y1 + text_h + 4)`
  - background rect at `(x1, y1, x1 + text_w + 8, y1 + text_h + 8)`

### 5.3 Label style

- Font: `cv2.FONT_HERSHEY_SIMPLEX`
- Scale: `max(0.7, img_width / 1800.0)` (~0.7 on 1296×968 frames, ~0.85
  on 1500-wide BEV)
- Thickness: `max(2, img_width // 600)`
- Foreground: white `(255, 255, 255)`
- Background: opaque black `(0, 0, 0)`
- 4-pixel padding inside background box

### 5.4 Cache

`render_filtered_marked(frame_id, sorted_ids)` writes to
`<bev_dir>/filtered_marks/frame_{frame_id}_ids_{sorted_id_csv}.png`. The
function is purely a function of `(scene, frame_id, set_of_ids)`; same
inputs → cached file.

## 6. BEV Rendering Quality Fixes

### 6.1 Crop blank canvas

After `_render_mesh_with_traj`, find the bounding box of non-white
(`max(rgb) < 250`) pixels and crop the canvas to that bounding box +
8-pixel margin. Propagate the crop offset into the projected `(u, v)`
label coordinates by subtracting `(crop_x_min, crop_y_min)`.

### 6.2 Label legibility

In `_overlay_proposal_labels`:

- Font scale: `0.85` (was `0.45`)
- Thickness: `2` (was `1`)
- **Black 1-pixel outline** drawn around the text via
  `cv2.putText(..., color=(0, 0, 0), thickness=thickness+2)` before the
  inner colour text. Combined with the opaque label background, this
  hits the same "double outline" contrast guarantee as
  `mark_frame_with_bbox`.
- Default text colour stays `(32, 32, 32)` but its background goes from
  `(255, 255, 255)` → `(230, 230, 230)` to avoid pure-white blending into
  background floor pixels.

### 6.3 Highlight projection bug

`agents/tools/scene_perception.py::_render_highlighted_bev` is replaced.

Old (buggy):
```python
base = cv2.imread(catalog.bev_image_path)      # already rendered
img = ...
builder = _BackdropBuilder(image_size=img.shape[1])
scene_bounds = (min(xs), min(ys), max(xs), max(ys))
overlay = builder._overlay_proposal_labels(img, props, scene_bounds, highlight_ids)
```

New: call the real `build_with_labels(scene_id, data_root, proposals,
output_path, highlight_ids=...)` so the highlight uses the same
perspective camera (and crop) as the default BEV. `scene_id` and
`data_root` come from `runtime.bundle.extra_metadata` (already populated
by pack-prep). The legacy backdrop fallback is removed.

To avoid re-rendering the mesh for every highlight call: factor
`build_with_labels` so the mesh + trajectory render is cached per scene
(keyed by `config_hash`) and the highlight overlay step composites on
top.

### 6.4 Label declutter

Use one reference text size for all labels in a single BEV (computed
once from the longest catalog name at the current font scale; call it
`ref_w, ref_h`). When two proposals project to centres within `ref_w`
pixels horizontally and `ref_h` pixels vertically, nudge the later
label downward by `ref_h + 6 px`. For ≥3 overlapping centres, stack
labels vertically with a fixed `ref_h + 6 px` gap. This is intentionally
conservative — full force-directed layout is out of scope; the goal is
"don't render two labels on top of each other".

### 6.5 Cache invalidation

All four fixes change pixel output, so `render_hash` must include them.
Add `font_scale`, `font_thickness`, `label_outline_thickness`,
`crop_margin`, and `declutter_dx` fields to `SceneBEVConfig`. The
`config_hash()` derivation already serialises every dataclass field —
new fields participate automatically, and old cached PNGs become unused
(safe to leave or to `git clean -X data/*/bev_cache/` once).

## 7. Prompt + Playbook Updates

All five places where tool names / call patterns leak into prompts:

| File | What lives there |
| --- | --- |
| `src/agents/skills/shared_skills/scene_exploration_playbook.md` | Shared "first move + selector cheat sheet + anti-patterns" |
| `src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md` | VG standard flow + tool-at-a-glance |
| `src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation.md` | Spatial disambiguation workflow |
| `src/agents/packs/qa_default/skills/qa_answering_playbook.md` | QA question taxonomy + when-to-use-text |
| `src/agents/runtime/base.py` (`build_system_prompt`, lines ~389-441) | Hardcoded tool-name strings in the v9 system prompt |
| `src/agents/runtime/deepagents_agent.py` (lines ~131-142, ~207-280) | Inline tool-loading switch + hint strings |

### 7.1 `scene_exploration_playbook.md` — full rewrite

Target body (replacing the existing "Selector cheat sheet"):

```markdown
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
```

### 7.2 `vg_grounding_playbook.md` — replace existing standard flow

Replace steps 1-6 with the new five-step loop, drop references to
`view_keyframe(mode='marked')`, point `submit_final` evidence requirement
at `mark_frame_with_bbox` (see §7.5 evidence_frame_guard).

Replace the "Tools at a glance" block with:

```markdown
## Tools at a glance

- `select_by_text(query, k=3, hidden_categories?)` — **first move**;
  language → ≤3 RGB candidate frames.
- `select_by_proposal(proposal_ids, require_all=False, k=3)` — fetch
  frames containing specific catalog IDs.
- `select_by_frame_neighbor(anchor_frame_id, mode='temporal'|'viewpoint_diverse')`.
- `select_by_region(region, region_type='bev_2d'|'bbox_3d')`.
- `select_by_coverage(method='obj_iou'|'pose_depth')`.
- `mark_frame_with_bbox(frame_id, labels|ids)` — high-contrast
  annotated zoom; required before `submit_final` for VG (see
  evidence_frame_guard).
- `list_frame_proposals(frame_id)`, `list_scene_proposals(...)`,
  `inspect_proposal(id)` — text-only catalog queries.
- `view_bev(highlight=[ids])` — re-render BEV focused on subset.
- `compare_proposals_spatial(candidate_ids, anchor_id, relation)` —
  spatial disambiguation (TADG-relevant).
```

### 7.3 `vg_spatial_disambiguation.md` — small patch

Step 3 currently says
`view_keyframe(mode='marked', proposal_ids=[#a, #b, #x])`. Replace with
`mark_frame_with_bbox(frame_id, ids=[#a, #b, #x])`. Also remove the
implicit assumption that the frame is already in context — be explicit
that the frame must be returned by a selector first.

### 7.4 `qa_answering_playbook.md` — replace question taxonomy

Replace the taxonomy table with one that puts `select_by_text` first
and drops `view_keyframe(mode='rgb')`:

```markdown
| Question kind | Default approach |
| --- | --- |
| "what / count of X" | `select_by_text(query='X')` returns the frames; verify count by reading `visible_proposal_ids` and inspecting the most informative one. |
| "colour / material / state of X" | `select_by_text(query='X')` → `mark_frame_with_bbox(frame_id, labels=['X'])` for explicit attribution → `request_crops` if still ambiguous. |
| "where is X relative to Y" | `compare_proposals_spatial(candidate_ids=[#X], anchor_id=#Y, ...)` after selectors return one frame containing both. |
| "is X there / are there any" | `select_by_text(query='X')` — empty result is itself the evidence of absence; cross-check with BEV labels. |
| open-ended description | `select_by_coverage(method='pose_depth', k=3)` → mark only the frames that contributed to the answer. |
```

QA was the place that depended on plain-RGB (`view_keyframe(mode='rgb')`)
because masks covered visual evidence. With selectors now returning
plain RGB by default, mark is opt-in for QA — only used when the agent
wants to highlight a specific catalog entry it is attributing.

### 7.5 Guards interaction

The two parsers in `src/agents/skills/` (`evidence_frame_guard.py` and
`no_match_guard.py`) currently filter trace entries by
`tool_name == "view_keyframe"` AND `tool_input.mode in ("marked", "auto")`,
then parse `visible_proposals=`, `categories=`, `left_to_right=`,
`boxes_2d=` from the response text. With §4.2 keeping the response
schema identical, only the tool-name predicate needs to change.

- `evidence_frame_guard.py::_is_marked_view` (line ~146):
  - **Old:** `_tool_name(entry) == "view_keyframe"` + mode check
  - **New:** `_tool_name(entry) == "mark_frame_with_bbox"`
  - All downstream parsing (`_viewed_frame_map`, etc.) stays the same.
- `no_match_guard.py::_candidate_evidence` (line ~103):
  - **Old:** `name == "view_keyframe" and (tool_input.get("mode") or "auto") in ("marked", "auto")`
  - **New:** `name == "mark_frame_with_bbox"`
  - Same regex parsing as before.
- `TADG`: trace-name-agnostic; only inspects
  `compare_proposals_spatial`. No change.

VG evidence requirement after v9.1: at least one
`mark_frame_with_bbox` call with the submitted proposal's id either in
`ids` or matched by `labels`. Plain RGB injection via a selector does
**not** satisfy `evidence_frame_guard`.

### 7.6 `build_system_prompt` (`runtime/base.py`)

The v9 prompt enumeration at line ~418 currently says:

```text
1. Selectors A-F (text/hypothesis/proposal/region/neighbor/coverage)
   - select_by_text(query, k, hidden_categories) — ~Stage-1 LLM parse (2-5s)
   - select_by_hypothesis(hypothesis_json, k, hidden_categories) — instant (no parse)
2. view_keyframe(frame_id, mode='auto', categories?, proposal_ids?) — inject a first-person frame.
```

Replace with:

```text
1. Selectors (each returns ≤3 first-person RGB frames + metadata):
   - select_by_text(query, k≤3, hidden_categories) — Stage-1 language→frame, primary entry
   - select_by_proposal(proposal_ids, require_all, k≤3)
   - select_by_frame_neighbor(anchor_frame_id, mode='temporal'|'viewpoint_diverse', k≤3)
   - select_by_region(region, region_type, k≤3)
   - select_by_coverage(method='obj_iou'|'pose_depth', k≤3, seen_frame_ids?)
2. mark_frame_with_bbox(frame_id, labels?, ids?) — high-contrast annotated zoom.
   Requires at least one of labels / ids.
3. view_bev(highlight=?) — re-inject BEV (full or filtered).
4. Catalog: list_scene_proposals, list_frame_proposals, inspect_proposal.
5. request_crops(frame_id, bbox_2d) — pixel zoom on a seen frame.
```

The "Default view mode" branches at lines 389/393/397 disappear (there
is no mode anymore). Replace with a single line:

```text
Workflow: selectors inject candidate RGB frames; use mark_frame_with_bbox
on the one frame you have decided is worth verifying.
```

### 7.7 `deepagents_agent.py` — inline strings

- Lines ~131-142: remove the `from agents.tools.view_keyframe import build_view_keyframe_tool` import and the conditional that appends it. Add a single `tools.append(build_mark_frame_with_bbox_tool(runtime))` instead.
- Lines ~207-213: remove the VG/QA-specific "Default view mode" sentences.
- Lines ~251-252: replace "Use selectors + view_keyframe to fetch first-person frames" with "Use selectors to fetch ≤3 RGB frames per call; use mark_frame_with_bbox to annotate one frame for verification."
- Lines ~278-280: replace the tool enumeration to match §7.6.

## 8. Migration / Compatibility

### 8.1 Deleted modules

- `src/agents/tools/view_keyframe.py` — file deleted.
- `select_by_hypothesis` block in `src/agents/tools/selectors.py` — block
  deleted. The remaining 5 selectors are reorganised.

### 8.2 Test updates

| Test | Action |
| --- | --- |
| `src/agents/tools/tests/test_view_keyframe*.py` | delete |
| `src/agents/tools/tests/test_selectors_hypothesis.py` | delete |
| `src/agents/tools/tests/test_selectors_text.py` and the other `test_selectors_*.py` | rewrite to assert ≤3 images are queued and that `already_seen=true` skips queue |
| `src/agents/tools/tests/test_view_bev.py` | update highlight test to assert it goes through `build_with_labels` (not the linear-fallback backdrop); add a "label position matches base BEV pixel" assertion |
| `src/agents/skills/tests/test_scene_exploration_playbook_loadable.py` | update list of expected tool names; assert `mark_frame_with_bbox` present, `view_keyframe`/`select_by_hypothesis` absent |
| `src/agents/packs/vg_embodiedscan/skills/tests/test_playbook_v9_consistency.py` | update expected tool names; assert "first move" paragraph mentions `select_by_text` |
| `src/agents/packs/qa_default/skills/tests/test_qa_playbook_v9_consistency.py` | same |
| `src/agents/runtime/tests/test_build_system_prompt_v9.py` | regenerate snapshots to match §7.6 |
| `src/agents/runtime/tests/test_build_task_message_v9.py` | regenerate snapshots if affected |
| `src/agents/tests/test_migration_no_callback_tools.py` | extend the "must-not-be-loaded" list with `view_keyframe` and `select_by_hypothesis` |
| `src/agents/tests/test_evidence_frame_guard.py` | update to assert mark-based evidence acceptance, plain-RGB rejection |
| `src/agents/tests/test_no_match_guard.py` | update to count `mark_frame_with_bbox` instead of `view_keyframe` |
| `src/agents/tests/test_tadg.py` | re-verify (no behavioural change expected) |
| `src/agents/tests/integration/test_v9_vg_end_to_end_mock.py` and `test_v9_qa_end_to_end_mock.py` | switch tool sequence from `select → view_keyframe` to `select_by_text → mark_frame_with_bbox`; assert selectors inject images |

New tests:

- `test_mark_frame_with_bbox.py` — covers no-filter error, missing
  frame_id error, label-fits-centre case, label-fits-topleft case, ID +
  category mixed filter.
- `test_mark_frame_with_bbox_render.py` — render a known input on a tiny
  fake image, assert palette colour at a known pixel and the black
  outline ring around it.
- `test_selectors_return_images.py` — for each of the 5 selectors, mock
  the runtime to verify `seen_image_paths` grows by ≤3.
- `test_bev_crop_blank.py` — render a synthetic mostly-white image, run
  the crop pass, assert the output is tight to the non-white region.
- `test_bev_highlight_projection_matches_base.py` — mock the perspective
  render, call `view_bev(highlight=[id])`, assert the label `(u, v)`
  matches the base BEV's projected `(u, v)` for the same proposal.

### 8.3 Trace HTML

`scripts/generate_nr3d_langsmith_trace_html.py` (and its
`v9_catalog_first_*.html` artefacts):

- Add `mark_frame_with_bbox` to the `view` tool family with a distinct
  amber/orange swatch (was teal "view_keyframe").
- Add a `legacy` family for `view_keyframe` and `select_by_hypothesis`
  so old runs still parse with a deprecation badge.
- Selectors now reliably emit images; the existing image-collection
  regex needs an extra path for the `image_path` JSON field.

### 8.4 Stage-1 connector

`select_by_text` still calls `runtime.keyframe_selector.select_keyframes_v2`.
The only change is that the selector wrapper now also calls
`queue_pending_image_if_new(...)` for each returned `frame_id`'s
`raw_rgb_path`. No change to Stage-1 internals.

`select_by_hypothesis` removal means we no longer call
`selector.execute_hypothesis_dict`. The method becomes dead on the
Stage-2 side; a `rg execute_hypothesis_dict` confirms no other call
sites in the repo. We keep the method in
`src/query_scene/keyframe_selector.py` for this PR (do not touch the
Stage-1 surface), with a `# TODO(v9.2)` comment noting the method has
no remaining consumers. A follow-up PR can delete it after
double-checking external benchmark scripts.

## 9. Risk + Mitigation

| Risk | Mitigation |
| --- | --- |
| Selectors returning images blows the image budget on bad runs (5 selectors × 3 images = 15 images) | `seen_image_paths` already dedups across calls; in practice most agents only invoke 1–2 selectors per question. We do not add a hard image cap because that interferes with the "fetch cheap, mark deep" loop. |
| Old traces / cases break when re-rendered with the new HTML generator | `legacy` family + dashed amber outline already exists for v8-style traces. Same pattern reused. |
| BEV cache directories balloon (new render_hash per config field) | Document a one-liner `git clean -X data/*/bev_cache/` in the version doc. |
| `mark_frame_with_bbox` ID-by-category match returns 0 proposals when category name doesn't match catalog spelling | Error message lists the visible proposal IDs so the agent can fall back to `ids=`. Playbook example shows both `labels=` and `ids=` patterns. |
| Tests across 10+ files need updates | One commit per "category" (selectors / mark / BEV / playbook / trace), so any individual hunk is easy to revert if it regresses. |

## 10. Acceptance Criteria

- All five selectors inject ≤3 first-person frames per call and return
  consistent JSON (frame_id, visible_proposal_ids, bev_xy, camera_yaw,
  selected_because, image_path, already_seen).
- `mark_frame_with_bbox` errors when both `labels` and `ids` are empty;
  renders palette+outline bbox with area-ratio-aware label placement.
- `view_bev(highlight=[id])` labels overlay the base BEV in pixel
  alignment (assert in `test_bev_highlight_projection_matches_base.py`).
- BEV canvas no longer has ≥40% blank margin (assert non-white pixel
  bbox covers ≥75% of cropped canvas in a synthetic test).
- `select_by_hypothesis` and `view_keyframe` symbols are deleted from
  `src/agents/tools` and no longer appear in the system prompt.
- `random100` rerun on `feat/v9-catalog-first-scene-exploration` (or its
  follow-up branch) yields overall accuracy within ±2 pp of the v9
  clean run (86.0%) — i.e., no regression from the tool surface change.

## 11. Out of Scope

- ScanRefer / OpenEQA migration (separate PRs, same shape).
- Vector-graphics / SVG BEV rendering (would solve label illegibility
  permanently but is a separate spec).
- Force-directed BEV label layout. Simple stacking only here.
- Hypothesis JSON authoring tool (replaced by `select_by_text` natural
  language entry).

## 12. Open follow-ups (out of this spec)

- After this lands, audit `request_crops`: the v9 implementation already
  returns a placeholder string (see `stage1_callbacks.py`); decide
  whether to wire it for real or to retire it in v9.2.
- The hidden-categories knob currently lives on `select_by_text` only;
  consider promoting it to a runtime-wide filter (e.g., "always hide
  wall / floor / ceiling").
