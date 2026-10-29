"""Tests for AegisTracer runtime telemetry and span context management."""

import pytest

from aegis.core.models import ModelTier, SpanType
from aegis.core.pricing import PricingRegistry
from aegis.core.tracer import AegisTracer
from aegis.finops.circuit_breaker import (
    BreakerTripReason,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerTrip,
)


def test_tracer_basic_flow():
    tracer = AegisTracer(scenario_id="scenario_001")
    with tracer:
        with tracer.span("initial_thought", SpanType.INTERNAL_REASONING) as s:
            s.record_tokens(prompt_tokens=100, completion_tokens=50)
            s.set_attribute("agent_role", "planner")

        t1 = tracer.record_tool_call(
            tool_name="database_lookup",
            arguments={"query": "SELECT * FROM orders"},
            output={"rows": 3},
            duration_ms=45.0,
        )
        t2 = tracer.record_tool_call(
            tool_name="format_table",
            arguments={"rows": 3},
            output="formatted_table",
            duration_ms=10.0,
            depends_on=[t1.node_id],
        )

        tracer.record_llm_call(
            name="final_answer_generation",
            model_name="gpt-4o",
            prompt_tokens=400,
            completion_tokens=120,
            duration_ms=250.0,
        )

        tracer.record_mcp_transition(
            server_name="db-mcp",
            is_stateful=True,
            state_payload_bytes=512,
            context_token_overhead=32,
            latency_ms=12.5,
        )
        tracer.mark_resolved(True)

    dag = tracer.get_trajectory()
    assert dag.scenario_id == "scenario_001"
    assert dag.resolved is True
    assert len(dag.spans) == 2  # internal reasoning span + LLM call span
    assert len(dag.tool_nodes) == 2
    assert len(dag.edges) == 1
    assert dag.edges[0] == (t1.node_id, t2.node_id)
    assert len(dag.mcp_transitions) == 1
    assert dag.total_tokens() == 150 + 520
    assert dag.total_cost_usd() > 0.0


def test_tracer_span_with_error():
    tracer = AegisTracer(scenario_id="error_scenario")
    try:
        with tracer.span("failing_step", SpanType.LLM_INFERENCE, model_name="gpt-4o") as s:
            s.record_tokens(50, 10)
            raise ValueError("Inference provider rate limit exceeded")
    except ValueError:
        pass

    dag = tracer.get_trajectory()
    assert len(dag.spans) == 1
    assert dag.spans[0].error == "Inference provider rate limit exceeded"


def test_tracer_circuit_breaker_trips_on_tool_cycle():
    config = CircuitBreakerConfig(max_tool_calls=10, max_cycle_repetitions=2)
    cb = CircuitBreaker(config=config)

    tracer = AegisTracer(scenario_id="looping_agent", circuit_breaker=cb)

    # Alternate tool_a and tool_b twice
    tracer.record_tool_call("tool_a")
    tracer.record_tool_call("tool_b")
    tracer.record_tool_call("tool_a")

    # The 4th call forms the 2nd cycle of [tool_a, tool_b]
    with pytest.raises(CircuitBreakerTrip) as exc_info:
        tracer.record_tool_call("tool_b")

    assert exc_info.value.reason == BreakerTripReason.CYCLIC_LOOP_DETECTED


def test_tracer_circuit_breaker_trips_on_token_velocity():
    config = CircuitBreakerConfig(max_total_tokens=1000)
    cb = CircuitBreaker(config=config)
    tracer = AegisTracer(circuit_breaker=cb)

    tracer.record_llm_call("step1", "gpt-4o", prompt_tokens=400, completion_tokens=200)

    with pytest.raises(CircuitBreakerTrip) as exc_info:
        tracer.record_llm_call("step2", "gpt-4o", prompt_tokens=300, completion_tokens=300)

    assert exc_info.value.reason == BreakerTripReason.TOKEN_VELOCITY_BREACHED


def test_tracer_nested_span_hierarchy():
    tracer = AegisTracer(scenario_id="nested_spans")
    assert tracer.active_span_id is None

    with tracer.span("root_orchestrator", SpanType.ROUTER_DECISION) as root_span:
        assert tracer.active_span_id == root_span.span_id
        assert root_span.parent_span_id is None

        with tracer.span("child_reasoning", SpanType.INTERNAL_REASONING) as child_span:
            assert tracer.active_span_id == child_span.span_id
            assert child_span.parent_span_id == root_span.span_id

            llm_span = tracer.record_llm_call(
                name="nested_llm_call",
                model_name="gpt-4o",
                prompt_tokens=120,
                completion_tokens=40,
            )
            assert llm_span.parent_span_id == child_span.span_id

        # Back to root span after child exits
        assert tracer.active_span_id == root_span.span_id

    # Stack empty after root exits
    assert tracer.active_span_id is None

    dag = tracer.get_trajectory()
    spans_by_name = {s.name: s for s in dag.spans}
    assert spans_by_name["root_orchestrator"].parent_span_id is None
    assert spans_by_name["child_reasoning"].parent_span_id == root_span.span_id
    assert spans_by_name["nested_llm_call"].parent_span_id == child_span.span_id


@pytest.mark.asyncio
async def test_tracer_async_concurrent_spans():
    import asyncio

    tracer = AegisTracer(scenario_id="async_concurrent")

    async def worker(branch_name: str) -> tuple[str, str]:
        async with tracer.span(f"parent_{branch_name}", SpanType.ROUTER_DECISION) as p_span:
            await asyncio.sleep(0.005)
            async with tracer.span(f"child_{branch_name}", SpanType.INTERNAL_REASONING) as c_span:
                await asyncio.sleep(0.005)
                assert c_span.parent_span_id == p_span.span_id
                return p_span.span_id, c_span.span_id

    async with tracer:
        (p1, c1), (p2, c2) = await asyncio.gather(worker("alpha"), worker("beta"))

    dag = tracer.get_trajectory()
    spans_by_id = {s.span_id: s for s in dag.spans}
    assert spans_by_id[p1].parent_span_id is None
    assert spans_by_id[c1].parent_span_id == p1
    assert spans_by_id[p2].parent_span_id is None
    assert spans_by_id[c2].parent_span_id == p2
