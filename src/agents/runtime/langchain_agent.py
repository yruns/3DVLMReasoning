"""LangChain-based runtime components for Stage-2 agents."""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx
from langchain_openai import AzureChatOpenAI
from loguru import logger


def _now_ms() -> int:
    return int(time.time() * 1000)


class ModelHubKeyRotator:
    """Round-robin through ModelHub AKs with cooldown after retryable failures."""

    def __init__(
        self,
        api_keys: list[str],
        *,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        keys = [key for key in api_keys if key]
        if not keys:
            raise ValueError("ModelHubKeyRotator requires at least one AK")
        self._keys = list(keys)
        self._next_index = 0
        self._bad_until: dict[str, float] = {}
        self._time_fn = time_fn or time.time
        self._lock = threading.Lock()

    def acquire(self) -> str:
        now = self._time_fn()
        with self._lock:
            for _ in range(len(self._keys)):
                key = self._keys[self._next_index]
                self._next_index = (self._next_index + 1) % len(self._keys)
                if self._bad_until.get(key, 0.0) <= now:
                    return key
            return self._keys[0]

    def demote(self, key: str, cooldown_s: float = 30.0) -> None:
        with self._lock:
            self._bad_until[key] = self._time_fn() + cooldown_s
        logger.warning(
            "[ModelHubKeyRotator] demoted AK {}...{} for {:.1f}s",
            key[:6],
            key[-8:],
            cooldown_s,
        )


def _rewrite_modelhub_request(
    request: httpx.Request,
    *,
    rotator: ModelHubKeyRotator,
    modelhub_path: str,
    model_name: str,
    session_id: str,
    logid_prefix: str = "modelhub",
    now_ms: Callable[[], int] | None = None,
) -> None:
    """Rewrite Azure chat-completions requests into ModelHub crawl requests."""
    if not request.url.path.endswith("/chat/completions"):
        return

    ak = rotator.acquire()
    request.extensions["modelhub_ak"] = ak

    params = request.url.params.set("ak", ak)
    if "api-version" in params:
        params = params.remove("api-version")
    request.url = request.url.copy_with(path=modelhub_path, params=params)

    if request.content:
        body = json.loads(request.content)
        body["model"] = model_name
        new_content = json.dumps(body).encode("utf-8")
        request._content = new_content
        request.stream = httpx.ByteStream(new_content)
        request.headers["content-length"] = str(len(new_content))

    request.headers["extra"] = json.dumps({"session_id": session_id})
    request.headers["X-TT-LOGID"] = (
        f"{logid_prefix}_{session_id}_{(now_ms or _now_ms)()}"
    )


def _is_retryable_modelhub_response(response: httpx.Response) -> bool:
    """Return True when ModelHub indicates a retryable quota or transient error."""
    if response.status_code in {429, 503}:
        return True

    try:
        payload = response.json()
    except Exception:
        payload = None

    queue: list[object] = [payload] if payload is not None else []
    while queue:
        item = queue.pop()
        if isinstance(item, dict):
            code = item.get("code")
            if code in {-1003, "-1003"}:
                return True
            message = str(item.get("message", "")).lower()
            if any(
                token in message
                for token in (
                    "quota",
                    "rate limit",
                    "resource exhausted",
                    "server busy",
                    "timeout",
                )
            ):
                return True
            queue.extend(item.values())
        elif isinstance(item, list):
            queue.extend(item)

    return False


class ModelHubHttpClient(httpx.Client):
    """httpx client that rewrites Azure chat requests into ModelHub requests."""

    def __init__(
        self,
        *,
        rotator: ModelHubKeyRotator,
        modelhub_path: str,
        model_name: str,
        session_id: str,
        timeout: Any,
        max_attempts: int = 5,
        base_delay: float = 2.0,
        logid_prefix: str = "modelhub",
    ) -> None:
        super().__init__(timeout=timeout)
        self._rotator = rotator
        self._modelhub_path = modelhub_path
        self._model_name = model_name
        self._session_id = session_id
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._logid_prefix = logid_prefix

    def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        body = request.content if request.stream is not None else None
        headers = request.headers
        extensions = dict(request.extensions)
        last_response: httpx.Response | None = None

        for attempt in range(self._max_attempts):
            attempt_request = httpx.Request(
                request.method,
                request.url,
                headers=headers,
                content=body,
                extensions=dict(extensions),
            )
            _rewrite_modelhub_request(
                attempt_request,
                rotator=self._rotator,
                modelhub_path=self._modelhub_path,
                model_name=self._model_name,
                session_id=self._session_id,
                logid_prefix=self._logid_prefix,
            )
            active_ak = str(attempt_request.extensions.get("modelhub_ak", ""))

            try:
                response = super().send(attempt_request, **kwargs)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self._max_attempts - 1:
                    raise
                cooldown = self._base_delay * (2**attempt)
                if active_ak:
                    self._rotator.demote(active_ak, cooldown_s=cooldown)
                delay = cooldown + random.uniform(0.0, 1.0)
                logger.warning(
                    "[ModelHubHttpClient] transport failure on attempt {}/{}: {}. retry in {:.1f}s",
                    attempt + 1,
                    self._max_attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
                continue

            if not _is_retryable_modelhub_response(response):
                return response

            last_response = response
            if attempt >= self._max_attempts - 1:
                return response

            cooldown = self._base_delay * (2**attempt)
            if active_ak:
                self._rotator.demote(active_ak, cooldown_s=cooldown)
            delay = cooldown + random.uniform(0.0, 1.0)
            logger.warning(
                "[ModelHubHttpClient] retryable status={} on attempt {}/{}. retry in {:.1f}s",
                response.status_code,
                attempt + 1,
                self._max_attempts,
                delay,
            )
            response.read()
            response.close()
            time.sleep(delay)

        if last_response is not None:
            return last_response
        raise RuntimeError("ModelHub retry loop exited without a response")


class ToolChoiceCompatibleAzureChatOpenAI(AzureChatOpenAI):
    """AzureChatOpenAI variant with ModelHub routing and tool-choice normalization."""

    def __init__(
        self,
        *args: Any,
        api_keys: list[str] | None = None,
        modelhub_path: str = "/api/modelhub/online/v2/crawl",
        session_id: str = "v15_eval_default",
        logid_prefix: str = "modelhub",
        **kwargs: Any,
    ) -> None:
        resolved_api_keys = list(api_keys or [])
        fallback_key = kwargs.pop("api_key", None)
        if not resolved_api_keys and fallback_key:
            resolved_api_keys = [str(fallback_key)]
        if not resolved_api_keys:
            raise ValueError("ToolChoiceCompatibleAzureChatOpenAI requires api_keys")

        model_name = str(
            kwargs.get("azure_deployment") or kwargs.get("model") or "gpt-5.4-2026-03-05"
        )
        timeout = kwargs.get("timeout", 120.0)
        max_attempts = max(5, int(kwargs.get("max_retries", 0)) + 1)
        rotator = ModelHubKeyRotator(resolved_api_keys)
        http_client = ModelHubHttpClient(
            rotator=rotator,
            modelhub_path=modelhub_path,
            model_name=model_name,
            session_id=session_id,
            timeout=timeout,
            max_attempts=max_attempts,
            logid_prefix=logid_prefix,
        )

        kwargs["http_client"] = http_client
        kwargs["api_key"] = "unused-ak-is-in-query"
        kwargs["max_retries"] = 0

        super().__init__(*args, **kwargs)

        object.__setattr__(self, "_rotator", rotator)
        object.__setattr__(self, "_modelhub_http_client", http_client)
        object.__setattr__(self, "_session_id", session_id)
        object.__setattr__(self, "_api_keys", resolved_api_keys)
        object.__setattr__(self, "_modelhub_path", modelhub_path)

    def bind_tools(
        self,
        tools,
        *,
        tool_choice: str | bool | dict[str, Any] | None = None,
        **kwargs,
    ):
        """Bind tools to the model with normalized tool_choice.

        Args:
            tools: Tools to bind to the model
            tool_choice: Tool invocation mode ("auto", "any", "required", True, etc.)
            **kwargs: Additional arguments passed to parent bind_tools

        Returns:
            Model with bound tools
        """
        if tool_choice in ("any", "required", True):
            tool_choice = "auto"
        return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)
