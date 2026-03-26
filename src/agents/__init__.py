"""Stage-2 agent package built on LangChain v1 and DeepAgents."""

# Import Stage 1 -> Stage 2 adapters from stage1_adapters module
# (adapters.py renamed to stage1_adapters.py to avoid conflict with adapters/ package)
try:
    from .stage1_adapters import build_object_context, build_stage2_evidence_bundle
except ImportError:
    # Fallback for backward compatibility during migration
    import warnings

    warnings.warn(
        "adapters.py should be renamed to stage1_adapters.py",
        DeprecationWarning,
        stacklevel=2,
    )
    # Try direct import (will fail if adapters/ exists)
    raise

from .benchmark_adapters import (
    BenchmarkSampleInfo,
    BenchmarkType,
    MockFrameProvider,
    MultiBenchmarkAdapter,
    OpenEQAFrameProvider,
    ReplicaFrameProvider,
    ScanNetFrameProvider,
    build_evidence_bundle_from_frames,
    build_task_spec_from_sample,
    create_adapter_for_benchmark,
    extract_sample_info,
)
from .models import (
    KeyframeEvidence,
    Stage1HypothesisSummary,
    Stage2AgentResult,
    Stage2DeepAgentConfig,
    Stage2EvidenceBundle,
    Stage2EvidenceCitation,
    Stage2PlanMode,
    Stage2Status,
    Stage2StructuredResponse,
    Stage2TaskSpec,
    Stage2TaskType,
    Stage2ToolObservation,
    Stage2ToolResult,
)
from .stage1_callbacks import (
    Stage1BackendCallbacks,
    create_crop_callback,
    create_hypothesis_callback,
    create_more_views_callback,
)
from .stage2_deep_agent import Stage2DeepResearchAgent
from .trace import (
    ExecutionTrace,
    HTMLTraceRenderer,
    TraceRecorder,
    save_trace_report,
)
from .trace_server import (
    TraceDB,
    TraceServer,
    TracingAgent,
)

__all__ = [
    # Benchmark adapters
    "BenchmarkSampleInfo",
    "BenchmarkType",
    "MockFrameProvider",
    "MultiBenchmarkAdapter",
    "OpenEQAFrameProvider",
    "ReplicaFrameProvider",
    "ScanNetFrameProvider",
    "build_evidence_bundle_from_frames",
    "build_task_spec_from_sample",
    "create_adapter_for_benchmark",
    "extract_sample_info",
    # Stage 1 adapters
    "build_object_context",
    "build_stage2_evidence_bundle",
    "create_crop_callback",
    "create_hypothesis_callback",
    "create_more_views_callback",
    "Stage1BackendCallbacks",
    # Stage 2 agent
    "ExecutionTrace",
    "HTMLTraceRenderer",
    "KeyframeEvidence",
    "Stage1HypothesisSummary",
    "Stage2AgentResult",
    "Stage2DeepAgentConfig",
    "Stage2DeepResearchAgent",
    "Stage2EvidenceBundle",
    "Stage2EvidenceCitation",
    "Stage2PlanMode",
    "Stage2Status",
    "Stage2StructuredResponse",
    "Stage2TaskSpec",
    "Stage2TaskType",
    "Stage2ToolObservation",
    "Stage2ToolResult",
    "TraceDB",
    "TraceRecorder",
    "TraceServer",
    "TracingAgent",
    "save_trace_report",
]
