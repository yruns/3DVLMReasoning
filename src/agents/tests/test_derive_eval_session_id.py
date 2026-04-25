"""derive_eval_session_id includes chassis_tools_version + vg_backend."""
from __future__ import annotations

from pathlib import Path

from agents.examples.openeqa_official_question_pilot import derive_eval_session_id


def test_session_id_changes_when_chassis_tools_version_changes(tmp_path: Path) -> None:
    a = derive_eval_session_id(
        output_root=tmp_path,
        enable_temporal_fan=False,
        chassis_tools_version=1,
        vg_backend="legacy",
    )
    b = derive_eval_session_id(
        output_root=tmp_path,
        enable_temporal_fan=False,
        chassis_tools_version=2,
        vg_backend="legacy",
    )
    assert a != b


def test_session_id_changes_when_vg_backend_changes(tmp_path: Path) -> None:
    a = derive_eval_session_id(
        output_root=tmp_path,
        enable_temporal_fan=False,
        chassis_tools_version=1,
        vg_backend="legacy",
    )
    b = derive_eval_session_id(
        output_root=tmp_path,
        enable_temporal_fan=False,
        chassis_tools_version=1,
        vg_backend="pack_v1",
    )
    assert a != b


def test_explicit_session_id_overrides(tmp_path: Path) -> None:
    sid = derive_eval_session_id(
        output_root=tmp_path,
        enable_temporal_fan=False,
        chassis_tools_version=1,
        vg_backend="legacy",
        explicit_session_id="custom",
    )
    assert sid == "custom"
