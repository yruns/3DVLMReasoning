"""LangChain-based runtime components for Stage-2 agents."""

from __future__ import annotations

from typing import Any

from langchain_openai import AzureChatOpenAI


class ToolChoiceCompatibleAzureChatOpenAI(AzureChatOpenAI):
    """AzureChatOpenAI variant that normalizes tool-choice for stricter providers.

    Some providers enforce strict validation on tool_choice values. This adapter
    normalizes non-standard values ("any", "required", True) to "auto" which is
    more widely accepted.
    """

    def bind_tools(
        self,
        tools,
        *,
        tool_choice: str | bool | dict[str, Any] | None = None,
        **kwargs,
    ):
        """Bind tools to the model with normalized tool_choice.

        Args:
            tools: Tools to bind to the model
            tool_choice: Tool invocation mode ("auto", "any", "required", True, etc.)
            **kwargs: Additional arguments passed to parent bind_tools

        Returns:
            Model with bound tools
        """
        # Normalize non-standard tool_choice values to "auto"
        if tool_choice in ("any", "required", True):
            tool_choice = "auto"
        return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)
