# Codex SDK ModelHub Adapter Config

Date: 2026-06-07

This note records the local ModelHub adapter credential location used for
Codex SDK NR3D runs. It intentionally does not contain AK values.

## Private AK File

The real AKs are stored in the local, gitignored adapter file:

```text
/Users/bytedance/aispace/codex_modelhub_adapter/.modelhub_upstreams.toml
```

File permissions are `0600`. Do not copy AK values into tracked docs, commit
messages, console logs, or benchmark records.

The file contains three `[[upstreams]]` entries for:

```text
model_name = "gpt-5.4-2026-03-05"
url = "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online"
```

Traffic weights:

| Alias | Weight |
|---|---:|
| `gpt54_a` | 5 |
| `gpt54_b` | 1 |
| `gpt54_c` | 5 |

The Codex SDK runtime sends both:

- `extra.session_id`: stable prefix-cache session id
- `extra.chat_run_id`: per-turn id used for weighted upstream selection

The adapter selects one upstream by deterministic weighted hashing over
`extra.chat_run_id` when present, falling back to `session_id` only for clients
that do not send a per-turn id. This keeps prefix caching stable while allowing
concurrent benchmark samples to use the 5:1:5 capacity split.

## Office-Network Smoke

On 2026-06-07, each alias was tested directly against the office endpoint
`/api/modelhub/online/v2/crawl` with a simple `1+1` prompt:

| Alias | Status | Response summary |
|---|---:|---|
| `gpt54_a` | 200 | `2` |
| `gpt54_b` | 200 | `2` |
| `gpt54_c` | 200 | `2` |

## Adapter Launch

Use the private TOML when starting the local adapter:

```bash
cd /Users/bytedance/aispace/codex_modelhub_adapter
export AIDP_MODELHUB_UPSTREAMS_TOML=/Users/bytedance/aispace/codex_modelhub_adapter/.modelhub_upstreams.toml
uv run uvicorn adapter.app:app --host 127.0.0.1 --port 8787
```

The adapter's project Codex home is configured for:

```toml
model = "gpt-5.4-2026-03-05"
```

`gpt-5.4*` and `gpt-5.5*` are routed to ModelHub Chat Completions crawl
(`/v2/crawl`), then mapped back to the Codex SDK Responses wire format.
