"""Regression: v9.1 select_by_text needs runtime.keyframe_selector to be wired.

Before v9.1, Stage-1 was only reachable through `create_crop_callback`. The
v9.1 selector tools moved Stage-1 to `select_by_text`, which reads
`runtime.keyframe_selector`. If the production runner does not pass that
selector to `Stage2DeepResearchAgent(...)`, the tool silently returns an error
in every run. This test pins the wiring chain end-to-end.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.models import (
    Stage2DeepAgentConfig,
    Stage2EvidenceBundle,
    Stage2TaskSpec,
    Stage2TaskType,
)
from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime
from agents.stage2_deep_agent import Stage2DeepResearchAgent


class _FakeSelector:
    def select_keyframes_v2(self, **_kwargs):  # pragma: no cover - sanity stub
        return SimpleNamespace(keyframe_indices=[], metadata={})


def _minimal_bundle(tmp_path: Path) -> Stage2EvidenceBundle:
    bev = tmp_path / "bev.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(bev)
    catalog = SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(
                proposal_id=1,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                source="mask3d",
                frame_views={
                    0: FrameView(frame_id=0, raw_rgb_path=str(bev), bbox_2d=(0, 0, 1, 1))
                },
            ),
        ],
        total_frames=1,
        frame_id_range=(0, 0),
        valid_frame_ids=[0],
        bev_image_path=str(bev),
    )
    return Stage2EvidenceBundle(extra_metadata={"scene_catalog": catalog.model_dump()})


def test_stage2_agent_forwards_keyframe_selector_to_runtime_init():
    selector = _FakeSelector()
    agent = Stage2DeepResearchAgent(keyframe_selector=selector)
    assert agent._runtime.keyframe_selector is selector


def test_build_agent_populates_runtime_state_keyframe_selector(tmp_path: Path):
    selector = _FakeSelector()
    runtime_impl = DeepAgentsStage2Runtime(
        config=Stage2DeepAgentConfig(),
        keyframe_selector=selector,
    )
    bundle = _minimal_bundle(tmp_path)
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.QA,
        user_query="how many chairs?",
    )
    _graph, runtime_state = runtime_impl.build_agent(task=task, bundle=bundle)
    assert runtime_state.keyframe_selector is selector


def test_missing_selector_makes_runtime_state_attribute_none(tmp_path: Path):
    runtime_impl = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    bundle = _minimal_bundle(tmp_path)
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.QA,
        user_query="how many chairs?",
    )
    _graph, runtime_state = runtime_impl.build_agent(task=task, bundle=bundle)
    assert runtime_state.keyframe_selector is None
