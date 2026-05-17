"""v9.4 Experiment D — seed-keyframe drain flag wiring.

`DeepAgentsStage2Runtime.build_evidence_update_message` normally filters
out pack-prep "seed" keyframes (the 5 GT-target-visible RGBs that
`bundle.keyframes` is populated with at pack-prep time). This filter
was added in commit `8ebf701` ("kill Stage-1 seed-keyframe drain
leak"). Before that fix (commit `d5f40ba`, v9.1_fix FULL REPRO at
82.95 % on NR3D), the seeds silently flowed into agent context on
every evidence-update turn.

The v9.4 Experiment D ablation explores whether that leak was part of
the v9.1_fix +16 pp advantage. To support the ablation cleanly,
`Stage2DeepAgentConfig.restore_stage1_seed_keyframe_drain` (and the
mirrored runtime flag) bypasses the seed filter when True.

This test pins the contract on both halves:
  - Flag default (False): seed keyframes are filtered out (post-8ebf701).
  - Flag True: seed keyframes are auto-injected (pre-8ebf701 d5f40ba bug).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.core.task_types import KeyframeEvidence
from agents.models import (
    Stage2DeepAgentConfig,
    Stage2EvidenceBundle,
    Stage2TaskSpec,
    Stage2TaskType,
)
from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime


class _FakeSelector:
    def select_keyframes_v2(self, **_kwargs):  # pragma: no cover - sanity stub
        return SimpleNamespace(keyframe_indices=[], metadata={})


def _make_png(path: Path) -> str:
    Image.new("RGB", (16, 16), (200, 200, 200)).save(path)
    return str(path)


def _bundle_with_seed_keyframes(tmp_path: Path, n_seeds: int = 3) -> Stage2EvidenceBundle:
    """Build a minimal bundle with `n_seeds` pack-prep-style seed keyframes
    and a BEV image. Mirrors the pack_nr3d_v9_catalog_first layout where
    `bundle.keyframes` contains the GT-target-visible seeds and
    `bev_image_path` is the rendered top-down view."""
    bev = _make_png(tmp_path / "bev.png")
    seeds: list[KeyframeEvidence] = []
    for i in range(n_seeds):
        seed_path = _make_png(tmp_path / f"seed_{i}.png")
        seeds.append(
            KeyframeEvidence(
                keyframe_idx=i,
                image_path=seed_path,
                view_id=i,
                frame_id=i,
                note="GT-target-visible (pack-prep)",
            )
        )
    catalog = SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(
                proposal_id=1,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                source="mask3d",
                frame_views={
                    0: FrameView(
                        frame_id=0, raw_rgb_path=bev, bbox_2d=(0, 0, 1, 1)
                    )
                },
            ),
        ],
        total_frames=1,
        frame_id_range=(0, 0),
        valid_frame_ids=[0],
        bev_image_path=bev,
    )
    return Stage2EvidenceBundle(
        keyframes=seeds,
        bev_image_path=bev,
        extra_metadata={
            "scene_catalog": catalog.model_dump(),
            "vg_pending_images": [],
        },
    )


def _build_runtime_with_bundle(
    tmp_path: Path,
    *,
    restore_drain: bool,
) -> tuple[DeepAgentsStage2Runtime, object]:
    """Build a runtime with the seed-drain flag set as requested and the
    bundle's seeds snapshotted into `initial_keyframe_paths`."""
    cfg = Stage2DeepAgentConfig(
        enable_stage1_text_retrieval=False,
        restore_stage1_seed_keyframe_drain=restore_drain,
    )
    runtime_impl = DeepAgentsStage2Runtime(config=cfg)
    bundle = _bundle_with_seed_keyframes(tmp_path)
    # Use QA task type to keep the build path minimal (the VG path also
    # constructs a `task_ctx` from `bundle.extra_metadata.vg_proposal_pool`
    # which we don't need for the seed-drain assertion).
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.QA,
        user_query="how many chairs are in the room?",
    )
    # build_agent populates Stage2RuntimeState.initial_keyframe_paths from
    # the bundle's keyframes and copies the config flag into the runtime.
    _graph, state = runtime_impl.build_agent(task=task, bundle=bundle)
    return runtime_impl, state


def test_runtime_flag_defaults_to_false() -> None:
    """The restore-drain flag defaults to False on both Stage2DeepAgentConfig
    and the runtime state. Pin this so a future field-default flip doesn't
    silently re-introduce the v9.1_fix-era seed-keyframe leak in production."""
    cfg = Stage2DeepAgentConfig(enable_stage1_text_retrieval=False)
    assert cfg.restore_stage1_seed_keyframe_drain is False
    runtime_impl = DeepAgentsStage2Runtime(config=cfg)
    assert runtime_impl.config.restore_stage1_seed_keyframe_drain is False


def test_runtime_state_flag_copied_from_config(tmp_path: Path) -> None:
    """`BaseStage2Runtime.configure_runtime_state` must propagate the
    restore-drain flag from config to the live runtime state. Without this
    plumbing, the `build_evidence_update_message` guard would never see
    the flag and the experiment would silently no-op."""
    _impl, state_off = _build_runtime_with_bundle(tmp_path, restore_drain=False)
    assert state_off.restore_stage1_seed_keyframe_drain is False
    _impl2, state_on = _build_runtime_with_bundle(tmp_path, restore_drain=True)
    assert state_on.restore_stage1_seed_keyframe_drain is True


def test_default_seed_filter_excludes_seed_keyframes(tmp_path: Path) -> None:
    """Default behaviour (post-8ebf701): build_evidence_update_message
    must NOT auto-inject pack-prep seed keyframes. With the BEV already
    in `seen_image_paths` and no pending images, the message should be
    None (no new evidence to inject)."""
    impl, state = _build_runtime_with_bundle(tmp_path, restore_drain=False)
    # Mark the BEV as already seen so it doesn't add a new image.
    state.seen_image_paths = {state.bundle.bev_image_path}
    msg = impl.build_evidence_update_message(state)
    # With seeds filtered + BEV already seen + no pending: nothing to inject.
    assert msg is None
    # Seeds remain unseen (the filter is read-only).
    for kf in state.bundle.keyframes:
        assert kf.image_path not in state.seen_image_paths


def test_restore_drain_flag_re_injects_seed_keyframes(tmp_path: Path) -> None:
    """v9.4 Experiment D contract: with
    `restore_stage1_seed_keyframe_drain=True`, the same setup must
    inject every seed keyframe path into the agent context. The
    returned HumanMessage carries one `image_url` entry per seed, and
    each seed path is recorded in `seen_image_paths` afterwards (so
    the next call doesn't double-inject)."""
    impl, state = _build_runtime_with_bundle(tmp_path, restore_drain=True)
    seed_paths = [kf.image_path for kf in state.bundle.keyframes]
    assert len(seed_paths) == 3  # _bundle_with_seed_keyframes default
    # Pre-seen BEV: same setup as the default-flag test so the only
    # variable is the flag.
    state.seen_image_paths = {state.bundle.bev_image_path}
    msg = impl.build_evidence_update_message(state)

    # Must produce a message containing all 3 seed image URLs.
    assert msg is not None
    image_blocks = [
        c for c in msg.content if isinstance(c, dict) and c.get("type") == "image_url"
    ]
    assert len(image_blocks) == 3, (
        f"expected 3 seed image_url blocks, got {len(image_blocks)}; "
        f"content={msg.content!r}"
    )

    # Seeds are now in seen_image_paths (so they won't be re-injected).
    for p in seed_paths:
        assert p in state.seen_image_paths


def test_restore_drain_flag_does_not_break_tool_pushed_pending(tmp_path: Path) -> None:
    """The restore-drain flag changes ONLY the seed filter. Tool-pushed
    `vg_pending_images` must still be drained correctly (regression for
    request_crops / mark_frame_with_bbox / selector flows that queue
    frames into the pending list for the next turn)."""
    impl, state = _build_runtime_with_bundle(tmp_path, restore_drain=True)
    # Queue one tool-pushed image (e.g. as a selector would).
    tool_pushed = _make_png(tmp_path / "tool_pushed.png")
    state.bundle = state.bundle.model_copy(
        update={
            "extra_metadata": {
                **state.bundle.extra_metadata,
                "vg_pending_images": [tool_pushed],
            }
        }
    )
    state.seen_image_paths = {state.bundle.bev_image_path}
    msg = impl.build_evidence_update_message(state)
    assert msg is not None
    image_blocks = [
        c for c in msg.content if isinstance(c, dict) and c.get("type") == "image_url"
    ]
    # 3 seeds (restored) + 1 tool-pushed = 4 image URLs.
    assert len(image_blocks) == 4
    assert tool_pushed in state.seen_image_paths
    # And the pending queue is now drained (so the next turn doesn't double-inject).
    assert state.bundle.extra_metadata.get("vg_pending_images") == []
