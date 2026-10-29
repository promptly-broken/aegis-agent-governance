"""Aegis FinOps: Unit economics (ECRT), circuit breakers, and hybrid edge-cloud arbitrage."""

from aegis.finops.circuit_breaker import (
    BreakerTripReason,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerStatus,
    CircuitBreakerTrip,
    StreamingCircuitBreaker,
    StreamingStepContext,
)
from aegis.finops.ecrt import ECRTResult, compute_ecrt
from aegis.finops.router import (
    ExecutionResult,
    HybridComplexityRouter,
    RoutingDecision,
    TaskComplexity,
    TaskDispatcher,
    execute_routed_task,
)

__all__ = [
    "compute_ecrt",
    "ECRTResult",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerStatus",
    "BreakerTripReason",
    "CircuitBreakerTrip",
    "StreamingCircuitBreaker",
    "StreamingStepContext",
    "HybridComplexityRouter",
    "RoutingDecision",
    "TaskComplexity",
    "ExecutionResult",
    "TaskDispatcher",
    "execute_routed_task",
]
