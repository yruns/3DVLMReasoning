from pathlib import Path

from agents.runtime.base import Stage2RuntimeState


def test_runtime_state_dataclass_has_no_image_queue_fields() -> None:
    field_names = set(Stage2RuntimeState.__dataclass_fields__)
    banned_exact = {
        "pending_" + "image_paths",
        "pending_" + "image_metadata",
        "vg_" + "pending" + "_images",
        "initial_" + "key" + "frame_paths",
    }

    assert field_names.isdisjoint(banned_exact)
    assert not [
        name for name in field_names if "pending" in name and "image" in name
    ]


def test_runtime_code_has_no_pending_image_channel_symbols() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    scan_roots = [repo_root / "src", repo_root / "scripts"]
    banned_tokens = [
        "pending_" + "image",
        "vg_" + "pending" + "_images",
        "queue_" + "pending" + "_image",
        "legacy_" + "image" + "_channel",
        '"vg_" + "' + "pending" + '"',
        '"queue_" + "' + "pending" + '"',
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
