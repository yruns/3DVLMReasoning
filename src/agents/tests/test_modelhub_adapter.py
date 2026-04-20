from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.messages import HumanMessage

from agents.runtime.langchain_agent import (
    ModelHubKeyRotator,
    ToolChoiceCompatibleAzureChatOpenAI,
    _is_retryable_modelhub_response,
    _rewrite_modelhub_request,
)


def test_key_rotator_cycles_keys() -> None:
    rotator = ModelHubKeyRotator(["ak1", "ak2", "ak3"])

    seen = [rotator.acquire() for _ in range(6)]

    assert set(seen) == {"ak1", "ak2", "ak3"}


def test_key_rotator_demotes_and_skips() -> None:
    rotator = ModelHubKeyRotator(["ak1", "ak2"], time_fn=lambda: 100.0)
    rotator.demote("ak1", cooldown_s=60.0)

    out = [rotator.acquire() for _ in range(4)]

    assert out == ["ak2", "ak2", "ak2", "ak2"]


def test_rewrite_modelhub_request_updates_path_query_body_and_headers() -> None:
    rotator = ModelHubKeyRotator(["ak1"])
    request = httpx.Request(
        "POST",
        "https://aidp-i18ntt-sg.tiktok-row.net/openai/deployments/foo/chat/completions?api-version=2024-03-01-preview",
        content=json.dumps(
            {
                "model": None,
                "messages": [{"role": "user", "content": "hello"}],
            }
        ).encode("utf-8"),
        headers={"content-type": "application/json"},
    )

    _rewrite_modelhub_request(
        request,
        rotator=rotator,
        modelhub_path="/api/modelhub/online/v2/crawl",
        model_name="gpt-5.4-2026-03-05",
        session_id="v15_session",
        logid_prefix="testmh",
        now_ms=lambda: 1234567890,
    )

    assert request.url.path == "/api/modelhub/online/v2/crawl"
    assert request.url.params["ak"] == "ak1"
    assert "api-version" not in request.url.params
    assert json.loads(request.content)["model"] == "gpt-5.4-2026-03-05"
    assert json.loads(request.headers["extra"]) == {"session_id": "v15_session"}
    assert request.headers["X-TT-LOGID"] == "testmh_v15_session_1234567890"


@pytest.mark.parametrize(
    ("status_code", "body", "expected"),
    [
        (403, {"error": {"message": "quota exhausted"}}, True),
        (429, {"error": {"message": "rate limit"}}, True),
        (503, {"error": {"message": "server busy"}}, True),
        (400, {"code": -1003, "message": "quota exceeded"}, True),
        (401, {"error": {"message": "ak not exist"}}, False),
    ],
)
def test_is_retryable_modelhub_response(
    status_code: int,
    body: dict[str, object],
    expected: bool,
) -> None:
    response = httpx.Response(status_code, json=body)
    assert _is_retryable_modelhub_response(response) is expected


def test_modelhub_http_client_uses_longer_cooldown_for_403() -> None:
    from unittest.mock import patch

    from agents.runtime.langchain_agent import ModelHubHttpClient

    rotator = ModelHubKeyRotator(["ak1", "ak2"], time_fn=lambda: 100.0)
    client = ModelHubHttpClient(
        rotator=rotator,
        modelhub_path="/api/modelhub/online/v2/crawl",
        model_name="gpt-5.4-2026-03-05",
        session_id="v15_session",
        timeout=30.0,
        max_attempts=2,
    )
    request = httpx.Request(
        "POST",
        "https://aidp-i18ntt-sg.tiktok-row.net/openai/deployments/foo/chat/completions?api-version=2024-03-01-preview",
        content=b'{"model": null, "messages": []}',
        headers={"content-type": "application/json"},
    )
    calls = {"count": 0}

    def fake_send(self, attempt_request, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(
                403,
                request=attempt_request,
                json={"error": {"message": "quota exhausted"}},
            )
        return httpx.Response(200, request=attempt_request, json={"ok": True})

    with (
        patch.object(rotator, "demote", wraps=rotator.demote) as demote_mock,
        patch("httpx.Client.send", new=fake_send),
        patch("time.sleep", return_value=None),
    ):
        response = client.send(request)

    assert response.status_code == 200
    demote_mock.assert_called_once()
    assert demote_mock.call_args.kwargs["cooldown_s"] == 60.0


def test_live_smoke_one_question() -> None:
    if not pytest.importorskip("langchain_openai"):
        pytest.skip("langchain_openai unavailable")
    if "CI_RUN_LIVE_SMOKE" not in __import__("os").environ:
        pytest.skip("live smoke requires CI_RUN_LIVE_SMOKE=1")

    llm = ToolChoiceCompatibleAzureChatOpenAI(
        azure_endpoint="https://aidp-i18ntt-sg.tiktok-row.net",
        azure_deployment="gpt-5.4-2026-03-05",
        model="gpt-5.4-2026-03-05",
        api_version="2024-03-01-preview",
        api_keys=[
            "hnJAK3LscxwLcy5OpZGQqQAzNyQmdx0a_GPT_AK",
            "cjodAcZmk7eIwm8wtizk1MfqyEJ7V8lG_GPT_AK",
        ],
        modelhub_path="/api/modelhub/online/v2/crawl",
        session_id="v15_live_smoke",
        max_tokens=32,
        timeout=120,
    )

    response = llm.invoke([HumanMessage(content="Reply with exactly one character: 2")])
    content = response.content
    if isinstance(content, list):
        text = "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    else:
        text = str(content)

    assert text.strip()
    assert "2" in text
