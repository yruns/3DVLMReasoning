"""Evaluation module for two-stage 3D scene understanding.

This module provides batch evaluation infrastructure for running Stage 1 (keyframe retrieval)
and Stage 2 (VLM agent reasoning) on benchmark datasets.

Academic Innovation Points:
- Adaptive Evidence Acquisition: VLM agent dynamically decides when to request more evidence
- Symbolic-to-Visual Repair: Stage 2 validates and corrects Stage 1 scene graph hypotheses
- Evidence-Grounded Uncertainty: Explicit uncertainty output when evidence is insufficient
- Unified Multi-Task Policy: Single agent architecture handles QA, grounding, navigation, manipulation
"""

from .ablation_config import (
    AblationConfig,
    AgentConfig,
    EvaluationConfig,
    Stage1Config,
    Stage2Config,
    ToolConfig,
    generate_ablation_matrix,
    get_all_presets,
    get_preset_config,
    load_experiment_configs,
    save_ablation_matrix,
)
from .academic_positioning import (
    AcademicPositioning,
    CompetingMethod,
    ContributionType,
    # Enums
    NoveltyLevel,
    PublicationStrategy,
    PublicationVenue,
    # Data structures
    ResearchClaim,
    create_academic_positioning,
    # Claim factories
    create_adaptive_evidence_claim,
    create_all_claims,
    # Competitor factories
    create_all_competitors,
    # Strategy factories
    create_cvpr_strategy,
    create_neurips_strategy,
    create_positioning_summary,
    create_symbolic_repair_claim,
    create_uncertainty_claim,
    create_unified_policy_claim,
    # Document generation
    generate_positioning_document,
    save_positioning_document,
)
from .batch_eval import (
    BatchEvalConfig,
    BatchEvaluator,
    CheckpointManager,
    EvalRunResult,
    EvalSample,
    EvalSampleResult,
    OpenEQASampleAdapter,
    ScanReferSampleAdapter,
    SQA3DSampleAdapter,
    adapt_openeqa_samples,
    adapt_scanrefer_samples,
    adapt_sqa3d_samples,
)
from .experimental_analysis import (
    AblationAnalysis,
    BenchmarkAnalysis,
    ExperimentalAnalysis,
    compute_ablation_analysis,
    compute_benchmark_analysis,
    compute_full_analysis,
    generate_ablation_analysis_text,
    generate_calibration_analysis,
    generate_experimental_analysis_section,
    generate_main_results_analysis,
    generate_robustness_analysis,
    generate_tool_usage_analysis,
)
from .metrics import (
    AblationGroup,
    AggregatedResults,
    BenchmarkMetrics,
    aggregate_multiple_runs,
    aggregate_run_result,
    export_summary_statistics,
    export_to_latex_table,
    export_tool_usage_table,
)
from .related_work import (
    # Data structures
    BenchmarkResult,
    DifferentiationPoint,
    EvidenceAcquisition,
    RelatedMethod,
    RelatedWorkSection,
    RepresentationType,
    # Enums
    TaskType,
    Venue,
    # Method factories
    create_all_methods,
    create_differentiation_points,
    create_our_method,
    create_related_work_summary,
    generate_benchmark_comparison_table,
    # Table generation
    generate_comparison_table,
    # Section generation
    generate_related_work_section,
    # Output
    save_related_work_section,
)
from .result_tables import (
    BenchmarkResultSet,
    MethodResult,
    PaperResults,
    create_mock_results,
    generate_all_tables,
    generate_table1_main_results,
    generate_table2_ablation_results,
    load_results_from_directory,
)
from .trace_integration import (
    EvalTraceManager,
    EvalTraceMetadata,
    TracingBatchEvaluatorMixin,
    create_tracing_evaluator,
    export_run_trace_report,
)
from .visualizations import (
    ConfidenceAccuracyPoint,
    DetectionDropDataPoint,
    ToolUsageData,
    create_all_figures,
    create_confidence_accuracy_figure,
    create_detection_drop_figure,
    create_tool_usage_figure,
    generate_confidence_accuracy_data,
    generate_detection_drop_data,
    generate_tool_usage_data,
)

__all__ = [
    # Batch evaluation
    "BatchEvalConfig",
    "BatchEvaluator",
    "CheckpointManager",
    "EvalSample",
    "EvalSampleResult",
    "EvalRunResult",
    "OpenEQASampleAdapter",
    "SQA3DSampleAdapter",
    "ScanReferSampleAdapter",
    "adapt_openeqa_samples",
    "adapt_sqa3d_samples",
    "adapt_scanrefer_samples",
    # Metrics aggregation
    "BenchmarkMetrics",
    "AblationGroup",
    "AggregatedResults",
    "aggregate_run_result",
    "aggregate_multiple_runs",
    "export_to_latex_table",
    "export_tool_usage_table",
    "export_summary_statistics",
    # Ablation configuration
    "AblationConfig",
    "ToolConfig",
    "AgentConfig",
    "Stage1Config",
    "Stage2Config",
    "EvaluationConfig",
    "get_preset_config",
    "get_all_presets",
    "generate_ablation_matrix",
    "load_experiment_configs",
    "save_ablation_matrix",
    # Trace integration
    "EvalTraceMetadata",
    "EvalTraceManager",
    "TracingBatchEvaluatorMixin",
    "create_tracing_evaluator",
    "export_run_trace_report",
    # Result tables
    "MethodResult",
    "BenchmarkResultSet",
    "PaperResults",
    "create_mock_results",
    "generate_table1_main_results",
    "generate_table2_ablation_results",
    "generate_all_tables",
    "load_results_from_directory",
    # Visualizations
    "DetectionDropDataPoint",
    "ToolUsageData",
    "ConfidenceAccuracyPoint",
    "generate_detection_drop_data",
    "generate_tool_usage_data",
    "generate_confidence_accuracy_data",
    "create_detection_drop_figure",
    "create_tool_usage_figure",
    "create_confidence_accuracy_figure",
    "create_all_figures",
    # Experimental Analysis
    "BenchmarkAnalysis",
    "AblationAnalysis",
    "ExperimentalAnalysis",
    "compute_benchmark_analysis",
    "compute_ablation_analysis",
    "compute_full_analysis",
    "generate_main_results_analysis",
    "generate_ablation_analysis_text",
    "generate_robustness_analysis",
    "generate_tool_usage_analysis",
    "generate_calibration_analysis",
    "generate_experimental_analysis_section",
    # Related Work
    "TaskType",
    "EvidenceAcquisition",
    "RepresentationType",
    "Venue",
    "BenchmarkResult",
    "RelatedMethod",
    "DifferentiationPoint",
    "RelatedWorkSection",
    "create_all_methods",
    "create_our_method",
    "create_differentiation_points",
    "generate_comparison_table",
    "generate_benchmark_comparison_table",
    "generate_related_work_section",
    "save_related_work_section",
    "create_related_work_summary",
    # Academic Positioning
    "NoveltyLevel",
    "ContributionType",
    "PublicationVenue",
    "ResearchClaim",
    "CompetingMethod",
    "PublicationStrategy",
    "AcademicPositioning",
    "create_adaptive_evidence_claim",
    "create_symbolic_repair_claim",
    "create_uncertainty_claim",
    "create_unified_policy_claim",
    "create_all_claims",
    "create_all_competitors",
    "create_cvpr_strategy",
    "create_neurips_strategy",
    "generate_positioning_document",
    "create_academic_positioning",
    "save_positioning_document",
    "create_positioning_summary",
]
