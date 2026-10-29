"""Aegis: Agentic Statistical Evaluation, Protocol Profiling, and FinOps Governance Control Plane."""

__version__ = "0.1.0"

from aegis.core.models import (
    MCPStateTransition,
    ModelTier,
    Span,
    SpanType,
    ToolNode,
    TrajectoryDAG,
)
from aegis.core.pricing import ModelPricing, PricingRegistry
from aegis.eval_engine.sprt import (
    SPRTDecision,
    SPRTResult,
    SequentialProbabilityRatioTest,
)
from aegis.eval_engine.tdi import TDIResult, compute_tdi
from aegis.exporters import HTMLDashboardExporter, JSONLAuditExporter, OTLPTraceExporter
from aegis.finops.ecrt import ECRTResult, compute_ecrt
from aegis.integrations import (
    GovernedMCPSessionProxy,
    LangGraphGovernanceCallback,
    OpenAIAgentsGovernanceHooks,
    governed_tool,
)

__all__ = [
    "__version__",
    "SpanType",
    "ModelTier",
    "ModelPricing",
    "Span",
    "ToolNode",
    "MCPStateTransition",
    "TrajectoryDAG",
    "PricingRegistry",
    "compute_tdi",
    "TDIResult",
    "SequentialProbabilityRatioTest",
    "SPRTDecision",
    "SPRTResult",
    "compute_ecrt",
    "ECRTResult",
    "OTLPTraceExporter",
    "JSONLAuditExporter",
    "HTMLDashboardExporter",
    "governed_tool",
    "LangGraphGovernanceCallback",
    "OpenAIAgentsGovernanceHooks",
    "GovernedMCPSessionProxy",
]
