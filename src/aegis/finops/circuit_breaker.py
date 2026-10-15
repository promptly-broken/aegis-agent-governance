"""Runaway Loop Circuit Breaker: cyclic tool-call detection and SLA budget enforcement."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from aegis.core.models import TrajectoryDAG


class CircuitBreakerStatus(str, Enum):
    """Operational status of the governance circuit breaker."""

    CLOSED = "closed"        # Operating normally, execution permitted
    OPEN = "open"            # SLA/Safety tripped, execution halted
    HALF_OPEN = "half_open"  # Probe mode under throttled constraints


class BreakerTripReason(str, Enum):
    """Specific violation that tripped the circuit breaker."""

    CYCLIC_LOOP_DETECTED = "cyclic_loop_detected"
    COST_BUDGET_EXCEEDED = "cost_budget_exceeded"
    TOKEN_VELOCITY_BREACHED = "token_velocity_breached"
    TOOL_CALL_LIMIT_EXCEEDED = "tool_call_limit_exceeded"


class CircuitBreakerTrip(Exception):
    """Raised when an agent trajectory violates safety or FinOps SLA constraints."""

    def __init__(self, reason: BreakerTripReason, message: str, details: Optional[dict] = None) -> None:
        super().__init__(f"Circuit Breaker Tripped [{reason.value}]: {message}")
        self.reason = reason
        self.details = details or {}


class CircuitBreakerConfig(BaseModel):
    """Configuration limits for runaway agent governance."""

    model_config = ConfigDict(extra="forbid")

    max_cost_usd: float = Field(default=0.05, ge=0.0)
    max_total_tokens: int = Field(default=50_000, ge=1)
    max_tool_calls: int = Field(default=20, ge=1)
    max_cycle_length: int = Field(default=4, ge=1)
    max_cycle_repetitions: int = Field(default=2, ge=1)


class CircuitBreaker:
    """Interceptors to detect runaway agent behaviors, negotiation loops, and FinOps breaches."""

    def __init__(self, config: Optional[CircuitBreakerConfig] = None) -> None:
        self.config = config or CircuitBreakerConfig()
        self.status = CircuitBreakerStatus.CLOSED

    def detect_sequence_cycles(
        self, tool_names: List[str]
    ) -> List[Tuple[List[str], int]]:
        """Detect repeated cycles of length <= max_cycle_length in tool execution history."""
        n = len(tool_names)
        detected: List[Tuple[List[str], int]] = []

        for cycle_len in range(1, self.config.max_cycle_length + 1):
            if n < cycle_len * self.config.max_cycle_repetitions:
                continue

            # Check suffix for repeated pattern of length cycle_len
            pattern = tool_names[-cycle_len:]
            repetitions = 0
            idx = n

            while idx >= cycle_len and tool_names[idx - cycle_len : idx] == pattern:
                repetitions += 1
                idx -= cycle_len

            if repetitions >= self.config.max_cycle_repetitions:
                detected.append((pattern, repetitions))

        return detected

    def inspect(self, trajectory: TrajectoryDAG) -> Tuple[bool, Optional[BreakerTripReason], Optional[str]]:
        """Evaluate trajectory telemetry against SLA limits and cycle detectors.

        Returns:
            (is_ok: bool, reason: Optional[BreakerTripReason], explanation: Optional[str])
        """
        # 1. Tool call limit check
        num_tools = len(trajectory.tool_nodes)
        if num_tools > self.config.max_tool_calls:
            self.status = CircuitBreakerStatus.OPEN
            return (
                False,
                BreakerTripReason.TOOL_CALL_LIMIT_EXCEEDED,
                f"Tool invocation count {num_tools} exceeded threshold {self.config.max_tool_calls}.",
            )

        # 2. Token consumption check
        total_tokens = trajectory.total_tokens()
        if total_tokens > self.config.max_total_tokens:
            self.status = CircuitBreakerStatus.OPEN
            return (
                False,
                BreakerTripReason.TOKEN_VELOCITY_BREACHED,
                f"Total tokens {total_tokens} exceeded threshold {self.config.max_total_tokens}.",
            )

        # 3. Cost budget check
        current_cost = trajectory.total_cost_usd()
        if current_cost > self.config.max_cost_usd:
            self.status = CircuitBreakerStatus.OPEN
            return (
                False,
                BreakerTripReason.COST_BUDGET_EXCEEDED,
                f"Cumulative cost ${current_cost:.4f} exceeded budget SLA ${self.config.max_cost_usd:.4f}.",
            )

        # 4. Graph cycles & sequence cycles
        if trajectory.has_cycle():
            cycles = trajectory.find_cycles()
            self.status = CircuitBreakerStatus.OPEN
            return (
                False,
                BreakerTripReason.CYCLIC_LOOP_DETECTED,
                f"Directed graph cycle detected: {cycles}",
            )

        # Sequence repetition check
        tools = trajectory.get_tool_sequence()
        seq_cycles = self.detect_sequence_cycles(tools)
        if seq_cycles:
            pattern, count = seq_cycles[0]
            self.status = CircuitBreakerStatus.OPEN
            return (
                False,
                BreakerTripReason.CYCLIC_LOOP_DETECTED,
                f"Tool sequence repeated pattern {pattern} {count} times consecutively.",
            )

        return (True, None, None)

    def guard(self, trajectory: TrajectoryDAG) -> None:
        """Evaluate trajectory; raises CircuitBreakerTrip if constraints are violated."""
        is_ok, reason, explanation = self.inspect(trajectory)
        if not is_ok:
            raise CircuitBreakerTrip(
                reason=reason or BreakerTripReason.CYCLIC_LOOP_DETECTED,
                message=explanation or "Circuit breaker safety limit tripped.",
                details={
                    "total_tokens": trajectory.total_tokens(),
                    "total_cost": trajectory.total_cost_usd(),
                    "num_tool_nodes": len(trajectory.tool_nodes),
                },
            )
