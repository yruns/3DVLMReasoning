# cmux Claude Recovery

Date captured: 2026-05-05
Repo: `/Users/bytedance/project/3DVLMReasoning`
Branch snapshot: `feat/scanrefer-v3-query-driven`

This note records how to restore the active multi-Claude cmux setup after reopening cmux. Surface refs are a snapshot, so verify them before sending commands.

## Sessions To Resume

| Role | Last seen surface | Resume command |
| --- | --- | --- |
| Supervisor / coordination | `surface:14` | `claude --resume 2f3326c3-7f63-4f52-8b46-2e035377f2d3` |
| TADG worker / implementation | `surface:46` | `claude --resume d4e29f55-53e7-4085-816d-12f2a4dfdf46` |
| Reviewer CC / M4 review | `surface:48` | `claude --resume a3dfbd35-5aee-4ada-84d1-a228229bce2d` |

Related current-workspace surfaces observed on 2026-05-05:

| Surface | Status |
| --- | --- |
| `surface:49` | Codex reviewer surface, retired; no Claude resume id captured here. |
| `surface:58` | Current Codex surface. Do not reuse for Claude recovery while Codex is active. |
| `surface:59` | Claude Code surface, but not one of the three resume ids above. |

## Recovery Workflow

1. Open cmux and locate the 3DVLMReasoning workspace.

```bash
cmux tree --all
cmux identify
```

2. Verify the target surfaces are shell prompts before sending anything.

```bash
cmux read-screen --surface surface:14 --lines 30
cmux read-screen --surface surface:46 --lines 30
cmux read-screen --surface surface:48 --lines 30
```

If a surface is still inside Claude, continue using that live session or exit it first. Do not paste `claude --resume ...` into a Claude prompt.

3. If the same surfaces are still available and idle, resume all three sessions.

```bash
cmux send --surface surface:14 "cd /Users/bytedance/project/3DVLMReasoning && claude --resume 2f3326c3-7f63-4f52-8b46-2e035377f2d3"
cmux send-key --surface surface:14 enter

cmux send --surface surface:46 "cd /Users/bytedance/project/3DVLMReasoning && claude --resume d4e29f55-53e7-4085-816d-12f2a4dfdf46"
cmux send-key --surface surface:46 enter

cmux send --surface surface:48 "cd /Users/bytedance/project/3DVLMReasoning && claude --resume a3dfbd35-5aee-4ada-84d1-a228229bce2d"
cmux send-key --surface surface:48 enter
```

4. If the old surface refs changed, create or select three terminal surfaces, rerun `cmux tree --all`, then substitute the new `surface:<id>` values into the commands above.

```bash
cmux new-pane --type terminal --direction right
cmux tree --all
```

5. Confirm each resume reached Claude.

```bash
cmux read-screen --surface surface:14 --lines 20
cmux read-screen --surface surface:46 --lines 20
cmux read-screen --surface surface:48 --lines 20
```

## Notes

- Run recovery from the repo root or use the `cd /Users/bytedance/project/3DVLMReasoning && ...` prefix shown above.
- `cmux send` only types text; `cmux send-key --surface <surface> enter` executes it.
- Surface refs can drift between cmux launches. Trust `cmux tree --all` over this snapshot.

## Last Live Check

After writing this file on 2026-05-05, the three resume commands above were sent to `surface:14`, `surface:46`, and `surface:48`, and all three reached Claude. Subsequent Claude API calls on those sessions reported:

```text
This organization has been disabled.
```

So if recovery reaches the Claude UI but work does not proceed, treat it as an account/organization backend issue rather than a cmux resume-command issue.
