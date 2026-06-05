"""Stage-2 Codex Agent SDK entrypoint.

This module intentionally does not replace ``stage2_deep_agent``. It provides a
separate wrapper so benchmark scripts can select the Codex SDK runtime
explicitly while keeping the DeepAgents path byte-stable.
"""

from __future__ import annotations

from .core.task_types import Stage2AgentResult, Stage2EvidenceBundle, Stage2TaskSpec
from .models import Stage2DeepAgentConfig
from .runtime.codex_sdk_agent import CodexSdkStage2Runtime


class Stage2CodexAgent:
    """Compatibility-sized wrapper around ``CodexSdkStage2Runtime``."""

    def __init__(
        self,
        config: Stage2DeepAgentConfig | None = None,
        **runtime_kwargs,
    ) -> None:
        self._runtime = CodexSdkStage2Runtime(config=config, **runtime_kwargs)

    @property
    def config(self) -> Stage2DeepAgentConfig:
        return self._runtime.config

    def run(
        self,
        task: Stage2TaskSpec,
        bundle: Stage2EvidenceBundle,
    ) -> Stage2AgentResult:
        return self._runtime.run(task=task, bundle=bundle)


__all__ = ["Stage2CodexAgent"]
