<!-- QA default pack skill. -->
# QA Answering Playbook

You answer scene-grounded questions from the current evidence bundle.

## Operating Rules

1. Load this skill before using QA-specific reasoning.
2. Ground every answer in visible evidence or explicit Stage-1 metadata.
3. Prefer a short answer when the question asks for a fact.
4. Use uncertainty when evidence is incomplete.
5. Submit exactly once with `submit_final`.

## Expected Final Payload

Use:

```json
{
  "answer": "short evidence-grounded answer",
  "supporting_claims": ["claim grounded in a frame or metadata"]
}
```

The answer must be non-empty. Supporting claims should be concise strings, not
paragraphs.

## Evidence Flow

1. Inspect the current bundle and task prompt.
2. Look at keyframes in the user message.
3. Call `inspect_stage1_metadata` when object terms, hypotheses, or selector
   status matter.
4. Call `retrieve_object_context` for object-centric summaries.
5. Call `request_more_views` when the target object or relation is not visible.
6. Call `request_crops` when a small attribute, state, text, color, or count is
   hard to read in the full frame.
7. Call `switch_or_expand_hypothesis` when the current hypothesis points at the
   wrong object family or anchor.
8. Submit the final payload once evidence is sufficient.

## Tool Guide

`inspect_stage1_metadata()` reveals the Stage-1 hypothesis, selector status,
frame mapping, and extra metadata. Use it before overriding a hypothesis.

`retrieve_object_context(object_terms: list[str] | None = None)` returns object
summaries. Use targeted terms such as `["chair", "table"]` when known.

`request_more_views(request_text: str, frame_indices: list[int] | None = None,
object_terms: list[str] | None = None, mode: str = "targeted")` asks Stage 1 for
additional views. Use `targeted` for object terms, `explore` when current views
are uninformative, and `temporal_fan` only when the runtime advertises it.

`request_crops(request_text: str, frame_indices: list[int] | None = None,
object_terms: list[str] | None = None)` requests zoomed evidence.

`switch_or_expand_hypothesis(request_text: str, preferred_kind: str | None =
None)` asks Stage 1 to repair or expand the hypothesis.

`list_skills()` lists available skills.

`load_skill(skill_name: str)` loads detailed guidance. Load this playbook first.

`submit_final(payload: dict, rationale: str, evidence_refs: list[dict] | None =
None)` validates and terminates the run.

## Question Types

For existence questions, answer yes/no and cite the visible object or metadata.

For color questions, request crops if the object is small, partially occluded, or
lighting is ambiguous.

For counting questions, compare all visible candidate regions and request more
views if the count may be truncated.

For state questions, cite visual state directly. Do not infer state from object
category alone.

For location questions, answer with relative spatial context grounded in visible
landmarks.

For relation questions, inspect both target and anchor terms. Request more views
when only one side is visible.

## Sufficiency Checks

The evidence is sufficient when the answer can be supported by at least one
clear frame or by reliable scene metadata. It is insufficient when the object is
not visible, the relevant attribute is too small, or multiple candidates remain
unresolved.

## Anti-Examples

Do not answer from common sense without checking evidence.

Do not submit before loading this skill.

Do not use long paragraphs in `supporting_claims`.

Do not claim an exact color from a distant blurry view.

Do not ignore a Stage-1 hypothesis mismatch.

Do not fabricate frame IDs or object IDs.

Do not keep requesting evidence after the answer is already clear.

Do not call `submit_final` with fields outside the expected payload.
