"""v9 migration smoke: dead config flags should not exist."""

import pytest

from agents.core.agent_config import Stage2DeepAgentConfig


def test_config_does_not_expose_dead_flags():
    cfg = Stage2DeepAgentConfig()
    assert not hasattr(cfg, "enable_temporal_fan")
    assert not hasattr(cfg, "enable_stage1_callback")


def test_config_does_not_accept_dead_flag_kwargs():
    with pytest.raises((TypeError, ValueError)):
        Stage2DeepAgentConfig(enable_temporal_fan=True)
    with pytest.raises((TypeError, ValueError)):
        Stage2DeepAgentConfig(enable_stage1_callback=True)
