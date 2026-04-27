<!-- shared with packs/vg_embodiedscan/skills/ -->
# Evidence Scouting

Use this skill to decide whether current evidence is enough, what is missing,
and which evidence tool should be called next.

## Scout Checklist

1. Identify the target object, attribute, relation, count, or state.
2. Identify anchor objects or contextual landmarks.
3. Check whether the current keyframes show all required entities.
4. Check whether the visual resolution is sufficient.
5. Check whether Stage-1 metadata agrees with the user query.
6. Pick the smallest additional evidence request that could resolve the gap.

## Request Strategy

Use `request_more_views` for missing objects, missing anchors, or views that do
not cover the relevant room region.

Use `request_crops` for fine-grained attributes, small objects, text, color,
state, or dense clutter.

Use `switch_or_expand_hypothesis` when the selector appears focused on the wrong
object category, relation, or anchor.

Use `inspect_stage1_metadata` before assuming a selector failure.

## Stop Conditions

Stop scouting when the answer can be supported by visible evidence and the
remaining uncertainty would not change the final answer.

Continue scouting when the answer depends on an object not yet visible, an
unreadable attribute, or a spatial relation with missing anchors.

## Subagent Legacy Guidance

This skill replaces the old DeepAgents `evidence_scout` subagent. Keep the same
separation of concerns: focus only on whether current keyframes are sufficient,
which missing views or crops are needed, and what uncertainty remains. Do not
produce the final user-facing answer while scouting.
