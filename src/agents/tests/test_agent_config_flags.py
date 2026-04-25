"""New chassis-related flags on Stage2DeepAgentConfig."""
from __future__ import annotations

import pytest

from agents.core.agent_config import Stage2DeepAgentConfig


def test_enable_chassis_tools_defaults_off() -> None:
    cfg = Stage2DeepAgentConfig()
    assert cfg.enable_chassis_tools is False


def test_vg_backend_defaults_legacy() -> None:
    cfg = Stage2DeepAgentConfig()
    assert cfg.vg_backend == "legacy"


def test_chassis_tools_version_default_is_int() -> None:
    cfg = Stage2DeepAgentConfig()
    assert isinstance(cfg.chassis_tools_version, int)


def test_vg_backend_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        Stage2DeepAgentConfig(vg_backend="bogus")
