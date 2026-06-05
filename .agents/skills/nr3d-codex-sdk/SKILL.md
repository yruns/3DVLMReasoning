---
name: nr3d-codex-sdk
description: Solve one NR3D visual grounding sample from a 3DVLMReasoning proposal pool using the Codex Agent SDK.
---

# NR3D Codex SDK Visual Grounding

You are selecting one object proposal for a single NR3D referring expression.

## Rules

- Choose exactly one `proposal_id` from the provided proposal pool.
- Use `proposal_id: -1` only when the described target is absent from the pool.
- Use the attached BEV image when present, together with the proposal metadata.
- Do not create, edit, or delete files.
- Do not use benchmark ground-truth fields.
- Return only the requested JSON object.

## Reasoning Priorities

1. Match the target category and enriched category first.
2. Use spatial language from the query, especially near, closest, left, right, above, below, by, between, and close to.
3. Use proposal centers and sizes as 3D evidence.
4. Use compact notes for visual attributes such as shape, color, material, and object role.
5. If multiple proposals remain plausible, pick the best-supported one and lower confidence.
