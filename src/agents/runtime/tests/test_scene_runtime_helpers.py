from agents.runtime.scene_runtime import queue_pending_image_if_new


class _FakeRuntime:
    def __init__(self) -> None:
        self.seen_image_paths: set[str] = set()

        class _Bundle:
            extra_metadata = {"vg_pending_images": []}

        self.bundle = _Bundle()

    def mark_evidence_updated(self) -> None:
        pass


def test_queue_pending_image_if_new_queues_unseen():
    rs = _FakeRuntime()
    queue_pending_image_if_new(rs, "/tmp/frame_42.png")
    assert rs.bundle.extra_metadata["vg_pending_images"] == ["/tmp/frame_42.png"]


def test_queue_pending_image_if_new_skips_seen():
    rs = _FakeRuntime()
    rs.seen_image_paths.add("/tmp/frame_42.png")
    queue_pending_image_if_new(rs, "/tmp/frame_42.png")
    assert rs.bundle.extra_metadata["vg_pending_images"] == []


def test_queue_pending_image_if_new_handles_empty_path():
    rs = _FakeRuntime()
    queue_pending_image_if_new(rs, "")
    assert rs.bundle.extra_metadata["vg_pending_images"] == []
