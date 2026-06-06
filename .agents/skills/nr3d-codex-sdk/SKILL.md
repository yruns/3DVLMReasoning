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
- Use the CLI evidence tools printed in the prompt. The playbook skills are
  attached directly by the Codex SDK entrypoint, but the evidence interface for
  this SDK path is the CLI command:

  ```bash
  <python> <repo>/src/agents/mcp/nr3d_tools_cli.py \
    --state <sample.state.json> --trace <sample.trace.json> \
    call inspect_proposal '{"proposal_id": 6}'
  <python> <repo>/src/agents/mcp/nr3d_tools_cli.py \
    --state <sample.state.json> --trace <sample.trace.json> \
    call select_by_proposal '{"proposal_ids": [6], "k": 3}'
  <python> <repo>/src/agents/mcp/nr3d_tools_cli.py \
    --state <sample.state.json> --trace <sample.trace.json> \
    call mark_frame_with_bbox '{"frame_id": 10, "ids": [6]}'
  <python> <repo>/src/agents/mcp/nr3d_tools_cli.py \
    --state <sample.state.json> --trace <sample.trace.json> \
    call compare_proposals_spatial '{"candidate_ids": [6, 7], "anchor_id": 3, "relation": "closest_to"}'
  ```

  The CLI exposes the same names as the MCP server, including
  `list_scene_proposals`, `inspect_proposal`, `compare_proposals_spatial`,
  `list_frame_proposals`, `select_by_text`, `select_by_proposal`,
  `select_by_region`, and `mark_frame_with_bbox`.
- Required CLI evidence policy:
  - Inspect the candidate you plan to select.
  - If multiple same-category candidates are plausible, call
    `select_by_proposal`, then verify a selector-returned frame with
    `mark_frame_with_bbox`.
  - For spatial, ordinal, closest/farthest, left/right, above/below, or
    anchor-based language, call `compare_proposals_spatial` or
    `compare_candidates_to_anchors`.
  - Use `select_by_text` when the natural-language query is the best way to
    initialize a shortlist; it lazily initializes the Stage-1 KeyframeSelector.
- The CLI may write only the provided trace file.
- Do not create, edit, or delete files.
- Do not use benchmark ground-truth fields.
- Return only the requested JSON object.

## Reasoning Priorities

1. Match the target category and enriched category first.
2. Use spatial language from the query, especially near, closest, left, right, above, below, by, between, and close to.
3. Use proposal centers and sizes as 3D evidence.
4. Use compact notes for visual attributes such as shape, color, material, and object role.
5. For spatial or ambiguous references, use the CLI tools such as
   `inspect_proposal`, `compare_proposals_spatial`,
   `compare_candidates_to_anchors`, `list_frame_proposals`, `select_by_text`,
   `select_by_proposal`, and `mark_frame_with_bbox` before deciding.
6. If multiple proposals remain plausible, pick the best-supported one and lower confidence.
