"""Regression: v9.1 select_by_text needs runtime.keyframe_selector to be wired.

Before v9.1, Stage-1 was only reachable through `create_crop_callback`. The
v9.1 selector tools moved Stage-1 to `select_by_text`, which reads
`runtime.keyframe_selector`. If the production runner does not pass that
selector to `Stage2DeepResearchAgent(...)`, the tool silently returns an error
in every run.

v9.3 (current): the contract is now **symmetric and fail-loud**:
- if `config.enable_stage1_text_retrieval=True` AND `keyframe_selector=None`,
  `Stage2DeepResearchAgent.__init__` (via `BaseStage2Runtime.__init__`) raises
  ValueError. Callers must either pass a selector or explicitly disable text
  retrieval via the config flag.
- `build_selector_tools` additionally drops `select_by_text` whenever
  `runtime.keyframe_selector is None`, regardless of the flag, as a
  belt-and-suspenders safeguard (so even direct Stage2RuntimeState mutation in
  tests can't produce a tool list inconsistent with the runtime).

This test file pins both halves of that contract.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
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


def test_construction_raises_when_text_retrieval_enabled_but_no_selector() -> None:
    """v9.3 contract: text-retrieval=True without a selector must fail loud.

    Previously this silently produced an agent whose `select_by_text` tool
    was registered (system prompt advertised it) but returned an ERROR string
    at every invocation. Caught only at tool-call time, this hid v9.1_fix's
    18-pp NR3D regression for weeks. The construction-time guard is the root
    fix.
    """
    with pytest.raises(ValueError, match="enable_stage1_text_retrieval=True"):
        Stage2DeepResearchAgent()  # default config: text retrieval on, no selector

    with pytest.raises(ValueError, match="enable_stage1_text_retrieval=True"):
        DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())


def test_construction_succeeds_when_text_retrieval_explicitly_disabled() -> None:
    """The escape hatch: callers that genuinely don't need select_by_text."""
    cfg = Stage2DeepAgentConfig(enable_stage1_text_retrieval=False)
    agent = Stage2DeepResearchAgent(config=cfg)
    assert agent._runtime.keyframe_selector is None
    runtime_impl = DeepAgentsStage2Runtime(config=cfg)
    assert runtime_impl.keyframe_selector is None


def test_build_agent_drops_select_by_text_when_runtime_selector_is_none(
    tmp_path: Path,
) -> None:
    """Belt-and-suspenders: even if Stage2RuntimeState ends up without a
    selector (e.g., via direct test mutation that bypasses the construction
    guard), `build_selector_tools` must drop `select_by_text` so the agent's
    tool list stays consistent with what the runtime can actually fulfill.
    """
    from agents.runtime.base import Stage2RuntimeState
    from agents.tools.selectors import build_selector_tools

    bundle = _minimal_bundle(tmp_path)
    state = Stage2RuntimeState(bundle=bundle)
    state.task_type = Stage2TaskType.QA
    # Mirror the "config-says-on but runtime-has-no-selector" condition:
    state.enable_stage1_text_retrieval = True
    assert state.keyframe_selector is None  # this is the dangerous state
    tool_names = {t.name for t in build_selector_tools(state)}
    assert "select_by_text" not in tool_names, (
        "build_selector_tools must drop select_by_text whenever the runtime "
        "has no keyframe_selector, even if the flag is True"
    )


def test_stage2_agent_wrapper_build_agent_populates_runtime_state_keyframe_selector(
    tmp_path: Path, monkeypatch
):
    """Regression for the production code path.

    `Stage2DeepResearchAgent.run()` calls `self.build_agent(...)` (the wrapper,
    NOT the underlying `DeepAgentsStage2Runtime.build_agent`). The wrapper
    constructs its own `Stage2RuntimeState`. Earlier v9.1_fix wired the
    selector through the wrong build_agent; this test exercises the wrapper
    exactly the way production does.
    """
    selector = _FakeSelector()
    agent = Stage2DeepResearchAgent(keyframe_selector=selector)
    # Avoid live LLM / DeepAgents graph construction
    monkeypatch.setattr(agent, "_get_llm", lambda: object())
    monkeypatch.setattr("agents.stage2_deep_agent.create_deep_agent", lambda **_k: object())
    bundle = _minimal_bundle(tmp_path)
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.QA,
        user_query="how many chairs?",
    )
    _graph, runtime_state = agent.build_agent(task=task, bundle=bundle)
    assert runtime_state.keyframe_selector is selector
