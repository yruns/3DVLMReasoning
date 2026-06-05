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
- When the `nr3d_tools` MCP server is available, use it for evidence checks
  before the final JSON. The playbook skills are attached directly by the
  Codex SDK entrypoint, so call task tools such as `inspect_proposal` directly.
- If MCP tools are not visible in the Codex SDK turn, use the CLI fallback
  command printed in the prompt. Before final JSON, run at least one CLI
  evidence command for the most plausible candidate; for simple queries, use
  `inspect_proposal`. The command has the form:

  ```bash
  <python> <repo>/src/agents/mcp/nr3d_tools_cli.py \
    --state <sample.state.json> --trace <sample.trace.json> \
    call inspect_proposal '{"proposal_id": 6}'
  ```

  The CLI exposes the same names as the MCP server, including
  `list_scene_proposals`, `inspect_proposal`, `compare_proposals_spatial`,
  `list_frame_proposals`, `select_by_text`, `select_by_proposal`,
  `select_by_region`, and `mark_frame_with_bbox`.
- Do not claim the evidence tools are unavailable until you have tried the
  CLI fallback command from the prompt. The CLI may write only the provided
  trace file.
- Do not create, edit, or delete files.
- Do not use benchmark ground-truth fields.
- Return only the requested JSON object.

## Reasoning Priorities

1. Match the target category and enriched category first.
2. Use spatial language from the query, especially near, closest, left, right, above, below, by, between, and close to.
3. Use proposal centers and sizes as 3D evidence.
4. Use compact notes for visual attributes such as shape, color, material, and object role.
5. For spatial or ambiguous references, call MCP tools or the CLI fallback
   tools such as
   `inspect_proposal`, `compare_proposals_spatial`, `list_frame_proposals`,
   `select_by_text`, `select_by_proposal`, and `mark_frame_with_bbox`
   before deciding.
6. If multiple proposals remain plausible, pick the best-supported one and lower confidence.
