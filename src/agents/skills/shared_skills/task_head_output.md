# Task Head Output

This shared skill preserves the old `task_head` subagent guidance as a loadable
skill body.

Use it when the evidence has been collected and the remaining work is to
synthesize the final task-specific payload.

## Guidance

Assemble the final payload from collected evidence. Stay faithful to cited
frames, explicit uncertainty, and the task-specific output schema. Do not add
claims that are not supported by visible evidence or tool observations.

## Checklist

1. Re-read the user query.
2. Re-read the expected payload schema.
3. Keep the answer concise.
4. Include supporting claims only when they map to actual evidence.
5. Preserve uncertainty when evidence is incomplete.
6. Submit through `submit_final` when the active task pack provides it.
