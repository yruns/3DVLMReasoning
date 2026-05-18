from pathlib import Path

import pytest

from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState


def test_runtime_state_dataclass_has_no_image_queue_fields() -> None:
    field_names = set(Stage2RuntimeState.__dataclass_fields__)
    assert not [
        name for name in field_names if "pending" in name and "image" in name
    ]


def test_runtime_state_has_no_side_channel_attr_registry() -> None:
    import agents.runtime.base as runtime_base

    registry_names = [
        name
        for name in vars(runtime_base)
        if "BANNED" in name and "SIDE_CHANNEL" in name
    ]

    assert registry_names == []


def test_runtime_state_rejects_dynamic_pending_image_attributes() -> None:
    runtime = Stage2RuntimeState(bundle=Stage2EvidenceBundle(scene_id="scene"))
    attr_name = "_".join(["tool", "pending", "image", "side", "channel"])

    with pytest.raises(AttributeError):
        setattr(runtime, attr_name, ["/tmp/leaked.png"])

    assert not hasattr(runtime, attr_name)


def test_runtime_code_has_no_pending_image_channel_symbols() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    scan_roots = [repo_root / "src", repo_root / "scripts"]
    banned_tokens = [
        "_".join(["pending", "image"]),
        "_".join(["vg", "pending", "images"]),
        "_".join(["queue", "pending", "image"]),
        "legacy_" + "image" + "_channel",
    ]
    offenders: list[str] = []

    for root in scan_roots:
        for path in root.rglob("*.py"):
            if path == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8")
            for token in banned_tokens:
                if token in text:
                    offenders.append(f"{path.relative_to(repo_root)}: {token}")

    assert offenders == []
