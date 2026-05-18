from agents.runtime.scene_runtime import make_tool_image_ref_if_new


class _FakeRuntime:
    def __init__(self) -> None:
        self.seen_image_paths: set[str] = set()

        class _Bundle:
            extra_metadata = {}

        self.bundle = _Bundle()

    def mark_evidence_updated(self) -> None:
        pass


def test_make_tool_image_ref_if_new_returns_unseen():
    rs = _FakeRuntime()
    image_ref = make_tool_image_ref_if_new(rs, "/tmp/frame_42.png")
    assert image_ref == {"image_path": "/tmp/frame_42.png"}
    assert not any("pending" in name and "image" in name for name in dir(rs))
    assert not any(
        key.startswith("vg_" + "pending") for key in rs.bundle.extra_metadata
    )


def test_make_tool_image_ref_if_new_skips_seen():
    rs = _FakeRuntime()
    rs.seen_image_paths.add("/tmp/frame_42.png")
    assert make_tool_image_ref_if_new(rs, "/tmp/frame_42.png") is None
    assert not any("pending" in name and "image" in name for name in dir(rs))
    assert not any(
        key.startswith("vg_" + "pending") for key in rs.bundle.extra_metadata
    )


def test_make_tool_image_ref_if_new_handles_empty_path():
    rs = _FakeRuntime()
    assert make_tool_image_ref_if_new(rs, "") is None
    assert not any("pending" in name and "image" in name for name in dir(rs))
    assert not any(
        key.startswith("vg_" + "pending") for key in rs.bundle.extra_metadata
    )
