"""Runtime module for agent execution.

This module provides the runtime components for executing Stage-2 agents,
including base classes, LangChain adapters, and DeepAgents implementations.

## Architecture

The runtime is organized into three layers:

1. **Base Runtime** (`base.py`): Provides shared functionality for all runtime
   implementations including image handling, message building, tool callbacks,
   uncertainty management, and system prompt generation.

2. **LangChain Adapters** (`langchain_agent.py`): LangChain-specific components
   like the ToolChoiceCompatibleAzureChatOpenAI adapter for provider compatibility.

3. **DeepAgents Runtime** (`deepagents_agent.py`): Complete DeepAgents-based
   runtime implementation with iterative evidence refinement loop.

## Usage

```python
from agents.runtime import DeepAgentsStage2Runtime

runtime = DeepAgentsStage2Runtime(config=my_config)
result = runtime.run(task, bundle)
```

For backward compatibility, the main agent class is still available in the parent:

```python
from agents import Stage2DeepResearchAgent
```
"""

from .base import BaseStage2Runtime, Stage2RuntimeState, default_output_instruction, default_payload_schema
from .deepagents_agent import DeepAgentsStage2Runtime
from .langchain_agent import ToolChoiceCompatibleAzureChatOpenAI

__all__ = [
    # Base classes
    "BaseStage2Runtime",
    "Stage2RuntimeState",
    "default_output_instruction",
    "default_payload_schema",
    # LangChain adapters
    "ToolChoiceCompatibleAzureChatOpenAI",
    # DeepAgents runtime
    "DeepAgentsStage2Runtime",
]
