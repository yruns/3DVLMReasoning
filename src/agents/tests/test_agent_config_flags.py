"""New chassis-related flags on Stage2DeepAgentConfig."""

from __future__ import annotations

import pytest

from agents.core.agent_config import Stage2DeepAgentConfig


def test_enable_chassis_tools_defaults_off() -> None:
    cfg = Stage2DeepAgentConfig()
    assert cfg.enable_chassis_tools is False


def test_vg_backend_defaults_pack_v1() -> None:
    cfg = Stage2DeepAgentConfig()
    assert cfg.vg_backend == "pack_v1"


def test_chassis_tools_version_default_is_int() -> None:
    cfg = Stage2DeepAgentConfig()
    assert isinstance(cfg.chassis_tools_version, int)
    # Bumped to 21 for text-first VG candidate policy prompt changes.
    # (changes the chassis surface, must invalidate prompt cache).
    assert cfg.chassis_tools_version == 21


def test_default_modelhub_pool_uses_three_weighted_gpt54_keys() -> None:
    cfg = Stage2DeepAgentConfig()
    assert cfg.base_url == "https://aidp-i18ntt-sg.tiktok-row.net"
    assert cfg.model_name == "gpt-5.4-2026-03-05"
    assert len(cfg.api_keys) == 3
    assert cfg.api_key_weights == [5.0, 1.0, 2.5]
    assert cfg.api_key_initial_offset == 0


def test_modelhub_pool_initial_offset_can_come_from_env(monkeypatch) -> None:
    monkeypatch.setenv("MODELHUB_AK_INITIAL_OFFSET", "2")

    cfg = Stage2DeepAgentConfig()

    assert cfg.api_key_initial_offset == 2


def test_modelhub_pool_initial_offset_treats_empty_env_as_zero(monkeypatch) -> None:
    monkeypatch.setenv("MODELHUB_AK_INITIAL_OFFSET", "")

    cfg = Stage2DeepAgentConfig()

    assert cfg.api_key_initial_offset == 0


def test_vg_backend_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        Stage2DeepAgentConfig(vg_backend="bogus")


def test_no_match_candidate_guard_defaults_off() -> None:
    cfg = Stage2DeepAgentConfig()
    assert cfg.use_no_match_candidate_guard is False
    assert cfg.use_evidence_frame_guard is False
    assert cfg.no_match_guard_max_repeats == 3
    assert cfg.no_match_guard_max_viewed == 12
