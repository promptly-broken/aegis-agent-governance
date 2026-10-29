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


class StreamingCircuitBreaker:
    """Real-time streaming interceptor for in-flight tool execution dispatchers."""

    def __init__(self, config: Optional[CircuitBreakerConfig] = None) -> None:
        self.breaker = CircuitBreaker(config=config)
        self.tool_history: List[str] = []
        self.cumulative_tokens = 0
        self.cumulative_cost_usd = 0.0

    @property
    def config(self) -> CircuitBreakerConfig:
        return self.breaker.config

    @property
    def status(self) -> CircuitBreakerStatus:
        return self.breaker.status

    def before_tool_call(self, tool_name: str, arguments: Optional[dict] = None) -> None:
        """Pre-execution validation before dispatching tool. Trips if limit is about to be exceeded."""
        if len(self.tool_history) + 1 > self.config.max_tool_calls:
            self.breaker.status = CircuitBreakerStatus.OPEN
            raise CircuitBreakerTrip(
                reason=BreakerTripReason.TOOL_CALL_LIMIT_EXCEEDED,
                message=f"Dispatching '{tool_name}' would breach tool limit ({self.config.max_tool_calls}).",
                details={"tool_history": list(self.tool_history)},
            )

        # Check lookahead sequence cycle
        lookahead = self.tool_history + [tool_name]
        cycles = self.breaker.detect_sequence_cycles(lookahead)
        if cycles:
            pattern, count = cycles[0]
            self.breaker.status = CircuitBreakerStatus.OPEN
            raise CircuitBreakerTrip(
                reason=BreakerTripReason.CYCLIC_LOOP_DETECTED,
                message=f"Tool call '{tool_name}' causes cyclic repetition of {pattern} {count} times.",
                details={"pattern": pattern, "count": count},
            )

    def after_tool_call(
        self, tool_name: str, tokens_consumed: int = 0, cost_usd: float = 0.0
    ) -> None:
        """Post-execution update. Enforces token velocity and SLA budget caps."""
        self.tool_history.append(tool_name)
        self.cumulative_tokens += tokens_consumed
        self.cumulative_cost_usd += cost_usd

        if self.cumulative_tokens > self.config.max_total_tokens:
            self.breaker.status = CircuitBreakerStatus.OPEN
            raise CircuitBreakerTrip(
                reason=BreakerTripReason.TOKEN_VELOCITY_BREACHED,
                message=f"Cumulative tokens {self.cumulative_tokens} exceeded SLA {self.config.max_total_tokens}.",
                details={"cumulative_tokens": self.cumulative_tokens},
            )

        if self.cumulative_cost_usd > self.config.max_cost_usd:
            self.breaker.status = CircuitBreakerStatus.OPEN
            raise CircuitBreakerTrip(
                reason=BreakerTripReason.COST_BUDGET_EXCEEDED,
                message=f"Cumulative cost ${self.cumulative_cost_usd:.4f} exceeded budget SLA ${self.config.max_cost_usd:.4f}.",
                details={"cumulative_cost_usd": self.cumulative_cost_usd},
            )

    def guard_step(self, tool_name: str, arguments: Optional[dict] = None) -> "StreamingStepContext":
        """Context manager for pre/post tool call execution."""
        return StreamingStepContext(self, tool_name, arguments)


class StreamingStepContext:
    """Context manager executing before_tool_call on entry and after_tool_call on exit."""

    def __init__(
        self, breaker: StreamingCircuitBreaker, tool_name: str, arguments: Optional[dict] = None
    ) -> None:
        self.breaker = breaker
        self.tool_name = tool_name
        self.arguments = arguments
        self.tokens_consumed = 0
        self.cost_usd = 0.0

    def record_usage(self, tokens: int = 0, cost_usd: float = 0.0) -> None:
        self.tokens_consumed += tokens
        self.cost_usd += cost_usd

    def __enter__(self) -> "StreamingStepContext":
        self.breaker.before_tool_call(self.tool_name, self.arguments)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.breaker.after_tool_call(
            self.tool_name, tokens_consumed=self.tokens_consumed, cost_usd=self.cost_usd
        )
