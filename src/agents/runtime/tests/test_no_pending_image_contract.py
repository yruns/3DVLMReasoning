from pathlib import Path


def test_runtime_code_has_no_pending_image_channel_symbols() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    source_root = repo_root / "src"
    banned_tokens = [
        "pending_" + "image",
        "vg_" + "pending" + "_images",
        "queue_" + "pending" + "_image",
        "legacy_" + "image" + "_channel",
        '"vg_" + "' + "pending" + '"',
        '"queue_" + "' + "pending" + '"',
    ]
    offenders: list[str] = []

    for path in source_root.rglob("*.py"):
        if path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for token in banned_tokens:
            if token in text:
                offenders.append(f"{path.relative_to(repo_root)}: {token}")

    assert offenders == []
