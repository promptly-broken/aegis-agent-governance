"""Unit tests for Runaway Loop Circuit Breaker."""

import pytest

from aegis.core.models import Span, SpanType, ToolNode, TrajectoryDAG
from aegis.finops.circuit_breaker import (
    BreakerTripReason,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerStatus,
    CircuitBreakerTrip,
)


def test_circuit_breaker_normal_execution():
    dag = TrajectoryDAG(trajectory_id="normal_traj")
    n1 = ToolNode(node_id="n1", tool_name="search")
    n2 = ToolNode(node_id="n2", tool_name="extract")
    dag.add_tool_node(n1)
    dag.add_tool_node(n2)
    dag.add_dependency("n1", "n2")

    breaker = CircuitBreaker()
    is_ok, reason, explanation = breaker.inspect(dag)
    assert is_ok is True
    assert reason is None
    assert breaker.status == CircuitBreakerStatus.CLOSED
    # guard should not raise
    breaker.guard(dag)


def test_circuit_breaker_detects_graph_cycles():
    dag = TrajectoryDAG(trajectory_id="cyclic_traj")
    n1 = ToolNode(node_id="agent_a", tool_name="negotiate_offer")
    n2 = ToolNode(node_id="agent_b", tool_name="counter_offer")
    dag.add_tool_node(n1)
    dag.add_tool_node(n2)
    dag.add_dependency("agent_a", "agent_b")
    dag.add_dependency("agent_b", "agent_a")  # Cycle

    breaker = CircuitBreaker()
    is_ok, reason, explanation = breaker.inspect(dag)
    assert is_ok is False
    assert reason == BreakerTripReason.CYCLIC_LOOP_DETECTED
    assert breaker.status == CircuitBreakerStatus.OPEN

    with pytest.raises(CircuitBreakerTrip) as exc_info:
        breaker.guard(dag)
    assert exc_info.value.reason == BreakerTripReason.CYCLIC_LOOP_DETECTED


def test_circuit_breaker_sequence_repetition():
    breaker = CircuitBreaker(CircuitBreakerConfig(max_cycle_length=2, max_cycle_repetitions=2))
    repeated_tools = ["search", "filter", "search", "filter"]
    cycles = breaker.detect_sequence_cycles(repeated_tools)
    assert len(cycles) > 0
    pattern, count = cycles[0]
    assert pattern == ["search", "filter"]
    assert count == 2


def test_circuit_breaker_token_breach():
    dag = TrajectoryDAG(trajectory_id="token_heavy")
    dag.add_span(
        Span(
            name="huge_inference",
            span_type=SpanType.LLM_INFERENCE,
            start_time_ms=0,
            end_time_ms=100,
            prompt_tokens=40_000,
            completion_tokens=20_000,
        )
    )
    # Total tokens = 60,000 > default max 50,000
    breaker = CircuitBreaker(CircuitBreakerConfig(max_total_tokens=50_000))
    is_ok, reason, explanation = breaker.inspect(dag)
    assert is_ok is False
    assert reason == BreakerTripReason.TOKEN_VELOCITY_BREACHED


def test_circuit_breaker_cost_breach():
    dag = TrajectoryDAG(trajectory_id="expensive_traj")
    dag.add_span(
        Span(
            name="expensive_span",
            span_type=SpanType.LLM_INFERENCE,
            start_time_ms=0,
            end_time_ms=100,
            cost_usd=0.25,
        )
    )
    breaker = CircuitBreaker(CircuitBreakerConfig(max_cost_usd=0.10))
    is_ok, reason, explanation = breaker.inspect(dag)
    assert is_ok is False
    assert reason == BreakerTripReason.COST_BUDGET_EXCEEDED
