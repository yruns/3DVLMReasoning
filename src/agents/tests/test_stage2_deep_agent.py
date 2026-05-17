"""Smoke tests for the v9 Stage2DeepAgentRuntime.

The full pre-v9 test suite here exercised callback-style tools
(`request_more_views`, `switch_or_expand_hypothesis`, `inspect_stage1_metadata`,
`view_keyframe_marked`, `find_proposals_by_category`, `list_keyframes_with_proposals`)
that no longer exist after the v9 catalog-first migration. The full end-to-end
behavioural coverage lives in `test_v9_mock_vlm_vg.py` and `test_v9_mock_vlm_qa.py`.

The smoke tests below verify that the runtime can be constructed and that the
unified output contract is still intact.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.core.agent_config import Stage2DeepAgentConfig, Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle, Stage2TaskSpec
from agents.runtime import ToolChoiceCompatibleAzureChatOpenAI
from agents.runtime.base import Stage2RuntimeState
from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime
from agents.stage2_deep_agent import Stage2DeepResearchAgent


class TestStage2DeepAgent(unittest.TestCase):
    def test_tool_choice_compatible_client_maps_required_tool_choice_to_auto(
        self,
    ) -> None:
        with patch("langchain_openai.AzureChatOpenAI.bind_tools") as bind_tools_mock:
            model = object.__new__(ToolChoiceCompatibleAzureChatOpenAI)
            ToolChoiceCompatibleAzureChatOpenAI.bind_tools(
                model,
                tools=[],
                tool_choice="any",
            )
        self.assertEqual(bind_tools_mock.call_args.kwargs["tool_choice"], "auto")

    def test_runtime_constructs_with_v9_signature(self) -> None:
        runtime = DeepAgentsStage2Runtime(
            config=Stage2DeepAgentConfig(enable_stage1_text_retrieval=False)
        )
        self.assertTrue(hasattr(runtime, "crop_callback"))
        self.assertFalse(hasattr(runtime, "more_views_callback"))
        self.assertFalse(hasattr(runtime, "hypothesis_callback"))

    def test_qa_runtime_tool_list_v9_surface(self) -> None:
        from agents.skills import PACKS

        PACKS.clear()
        import agents.packs.qa_default.registration  # noqa: F401

        runtime = DeepAgentsStage2Runtime(
            config=Stage2DeepAgentConfig(enable_stage1_text_retrieval=False)
        )
        bundle = Stage2EvidenceBundle()
        state = Stage2RuntimeState(bundle=bundle)
        state.task_type = Stage2TaskType.QA

        tool_names = {t.name for t in runtime.build_runtime_tools(state)}
        # The post-v9 base tool surface always includes retrieve_object_context +
        # request_crops; deleted names should never reappear.
        self.assertIn("retrieve_object_context", tool_names)
        self.assertIn("request_crops", tool_names)
        for dead in (
            "request_more_views",
            "switch_or_expand_hypothesis",
            "inspect_stage1_metadata",
            "view_keyframe_marked",
            "find_proposals_by_category",
            "list_keyframes_with_proposals",
        ):
            self.assertNotIn(dead, tool_names)

    def test_wrapper_build_agent_qa_auto_registers_default_pack(self) -> None:
        from agents.skills import PACKS

        PACKS.clear()
        bundle = Stage2EvidenceBundle()
        task = Stage2TaskSpec(task_type=Stage2TaskType.QA, user_query="?")
        agent = Stage2DeepResearchAgent(
            config=Stage2DeepAgentConfig(enable_stage1_text_retrieval=False)
        )

        with (
            patch.object(agent, "_get_llm", return_value=object()),
            patch("agents.stage2_deep_agent.create_deep_agent") as mock_create,
        ):
            mock_create.return_value = object()
            agent.build_agent(task, bundle)
            tool_names = {
                t.name for t in mock_create.call_args.kwargs["tools"]
            }

        self.assertIn(Stage2TaskType.QA, PACKS)
        self.assertIn("retrieve_object_context", tool_names)
        self.assertIn("request_crops", tool_names)


if __name__ == "__main__":
    unittest.main()
