"""Tests for StreamingCircuitBreaker and StreamingStepContext."""

import pytest

from aegis.finops.circuit_breaker import (
    BreakerTripReason,
    CircuitBreakerConfig,
    CircuitBreakerStatus,
    CircuitBreakerTrip,
    StreamingCircuitBreaker,
)


def test_streaming_circuit_breaker_tool_limit():
    config = CircuitBreakerConfig(max_tool_calls=3)
    scb = StreamingCircuitBreaker(config=config)

    scb.before_tool_call("tool_1")
    scb.after_tool_call("tool_1")

    scb.before_tool_call("tool_2")
    scb.after_tool_call("tool_2")

    scb.before_tool_call("tool_3")
    scb.after_tool_call("tool_3")

    assert scb.status == CircuitBreakerStatus.CLOSED

    # 4th call should trip BEFORE tool execution
    with pytest.raises(CircuitBreakerTrip) as exc_info:
        scb.before_tool_call("tool_4")

    assert exc_info.value.reason == BreakerTripReason.TOOL_CALL_LIMIT_EXCEEDED
    assert scb.status == CircuitBreakerStatus.OPEN


def test_streaming_circuit_breaker_lookahead_cycle():
    config = CircuitBreakerConfig(max_tool_calls=20, max_cycle_repetitions=2)
    scb = StreamingCircuitBreaker(config=config)

    # Sequence: query, fetch, query
    for tool in ["query", "fetch", "query"]:
        scb.before_tool_call(tool)
        scb.after_tool_call(tool)

    # Calling 'fetch' next would create [query, fetch] x 2 consecutively
    with pytest.raises(CircuitBreakerTrip) as exc_info:
        scb.before_tool_call("fetch")

    assert exc_info.value.reason == BreakerTripReason.CYCLIC_LOOP_DETECTED
    assert scb.status == CircuitBreakerStatus.OPEN


def test_streaming_circuit_breaker_token_velocity():
    config = CircuitBreakerConfig(max_total_tokens=1500)
    scb = StreamingCircuitBreaker(config=config)

    scb.before_tool_call("summarizer")
    scb.after_tool_call("summarizer", tokens_consumed=800)

    scb.before_tool_call("analyzer")
    with pytest.raises(CircuitBreakerTrip) as exc_info:
        scb.after_tool_call("analyzer", tokens_consumed=900)  # total 1700 > 1500

    assert exc_info.value.reason == BreakerTripReason.TOKEN_VELOCITY_BREACHED
    assert scb.status == CircuitBreakerStatus.OPEN


def test_streaming_circuit_breaker_cost_budget():
    config = CircuitBreakerConfig(max_cost_usd=0.05)
    scb = StreamingCircuitBreaker(config=config)

    scb.before_tool_call("heavy_search")
    scb.after_tool_call("heavy_search", cost_usd=0.03)

    scb.before_tool_call("heavy_reasoning")
    with pytest.raises(CircuitBreakerTrip) as exc_info:
        scb.after_tool_call("heavy_reasoning", cost_usd=0.03)  # total 0.06 > 0.05

    assert exc_info.value.reason == BreakerTripReason.COST_BUDGET_EXCEEDED


def test_streaming_step_context():
    config = CircuitBreakerConfig(max_tool_calls=5, max_total_tokens=1000)
    scb = StreamingCircuitBreaker(config=config)

    # Successful step context
    with scb.guard_step("db_read", {"table": "users"}) as step:
        step.record_usage(tokens=200, cost_usd=0.001)

    assert scb.cumulative_tokens == 200
    assert scb.cumulative_cost_usd == 0.001
    assert scb.tool_history == ["db_read"]

    # Step that trips after execution due to tokens
    with pytest.raises(CircuitBreakerTrip) as exc_info:
        with scb.guard_step("llm_worker") as step:
            step.record_usage(tokens=900, cost_usd=0.005)  # 200 + 900 = 1100 > 1000

    assert exc_info.value.reason == BreakerTripReason.TOKEN_VELOCITY_BREACHED
