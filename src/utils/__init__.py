"""Utility modules for 3DVLMReasoning.

Provides LLM client initialization, Gemini pool management,
model configuration utilities, and general utility functions.
"""

from .llm_client import (
    DEFAULT_MODEL,
    GEMINI_POOL_CONFIGS,
    MODEL_CONFIGS,
    GeminiClientPool,
    get_available_models,
    get_gemini3_flash,
    get_gemini3_pro,
    get_gemini_client,
    get_gemini_pool,
    get_gemini_pro,
    get_gpt4o,
    get_gpt52,
    get_langchain_chat_model,
    test_vision_request,
)

# Lazy import for general_utils (requires torch/numpy from [full] dependencies)
_GENERAL_UTILS_AVAILABLE = False
try:
    from .general_utils import (
        Timer,
        cfg_to_dict,
        measure_time,
        prjson,
        save_hydra_config,
        to_numpy,
        to_scalar,
        to_tensor,
    )

    _GENERAL_UTILS_AVAILABLE = True
except ImportError:
    # Optional dependencies not installed
    pass

__all__ = [
    # Pool management
    "GeminiClientPool",
    "GEMINI_POOL_CONFIGS",
    "get_gemini_pool",
    "get_gemini_client",
    # Model configs
    "MODEL_CONFIGS",
    "DEFAULT_MODEL",
    "get_available_models",
    "get_langchain_chat_model",
    # Convenience getters
    "get_gpt52",
    "get_gpt4o",
    "get_gemini_pro",
    "get_gemini3_pro",
    "get_gemini3_flash",
    # Testing
    "test_vision_request",
]

# Add general_utils exports only if available
if _GENERAL_UTILS_AVAILABLE:
    __all__.extend(
        [
            # General utilities
            "Timer",
            "to_numpy",
            "to_tensor",
            "to_scalar",
            "prjson",
            "cfg_to_dict",
            "measure_time",
            "save_hydra_config",
        ]
    )
