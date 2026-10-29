"""Benchmark Scenario 2: Multi-Agent Runaway Negotiation Loop & Circuit Breaker Stress Test."""

from __future__ import annotations

from typing import Any, Dict

from aegis.core.models import ModelTier, Span, SpanType, ToolNode, TrajectoryDAG
from aegis.core.tracer import AegisTracer
from aegis.finops.circuit_breaker import (
    BreakerTripReason,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerTrip,
    StreamingCircuitBreaker,
)


def generate_ungoverned_runaway_trajectory(max_oscillation_turns: int = 16) -> TrajectoryDAG:
    """Simulate an ungoverned multi-agent negotiation that oscillates between two tools until timeout."""
    dag = TrajectoryDAG(
        trajectory_id="traj_ungoverned_negotiation",
        scenario_id="vendor_contract_negotiation_loop",
        resolved=False,
    )
    prev_id = None
    t_ms = 1000.0

    for turn in range(max_oscillation_turns):
        tool_name = (
            "buyer_propose_counter_offer"
            if turn % 2 == 0
            else "vendor_evaluate_margin_floor"
        )
        node_id = f"neg_step_{turn + 1}"
        node = ToolNode(
            node_id=node_id,
            tool_name=tool_name,
            arguments={"turn": turn + 1, "proposed_discount_pct": 18.5 - (0.1 * turn)},
            status="success",
            duration_ms=85.0,
        )
        dag.add_tool_node(node)
        if prev_id is not None:
            dag.add_dependency(prev_id, node_id)
        prev_id = node_id

        # Frontier cloud tokens grow each turn as context accumulates
        prompt_tokens = 1200 + (turn * 350)
        completion_tokens = 180
        cost_usd = (prompt_tokens * 2.50 / 1_000_000.0) + (completion_tokens * 10.00 / 1_000_000.0)
        dag.add_span(
            Span(
                name=f"negotiation_turn_{turn + 1}",
                span_type=SpanType.LLM_INFERENCE,
                model_tier=ModelTier.FRONTIER_CLOUD,
                model_name="gpt-4o",
                start_time_ms=t_ms,
                end_time_ms=t_ms + 85.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=round(cost_usd, 6),
            )
        )
        t_ms += 90.0

    return dag


def run_governed_negotiation_stress_test(max_cycle_repetitions: int = 2) -> Dict[str, Any]:
    """Compare ungoverned runaway loop against Aegis StreamingCircuitBreaker interception."""
    ungoverned_dag = generate_ungoverned_runaway_trajectory(max_oscillation_turns=16)

    config = CircuitBreakerConfig(
        max_cost_usd=0.05,
        max_total_tokens=25_000,
        max_tool_calls=16,
        max_cycle_length=4,
        max_cycle_repetitions=max_cycle_repetitions,
    )
    streaming_breaker = StreamingCircuitBreaker(config=config)
    tracer = AegisTracer(
        scenario_id="vendor_contract_negotiation_loop",
        trajectory_id="traj_governed_negotiation",
        circuit_breaker=CircuitBreaker(config=config),
    )

    tripped_reason = None
    tripped_turn = 0
    prev_id = None

    for turn in range(16):
        tool_name = (
            "buyer_propose_counter_offer"
            if turn % 2 == 0
            else "vendor_evaluate_margin_floor"
        )
        try:
            with streaming_breaker.guard_step(tool_name, {"turn": turn + 1}) as step_ctx:
                prompt_tokens = 1200 + (turn * 350)
                completion_tokens = 180
                span = tracer.record_llm_call(
                    name=f"negotiation_turn_{turn + 1}",
                    model_name="gpt-4o",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    duration_ms=85.0,
                )
                node = tracer.record_tool_call(
                    tool_name=tool_name,
                    arguments={"turn": turn + 1},
                    duration_ms=85.0,
                    depends_on=[prev_id] if prev_id else None,
                )
                prev_id = node.node_id
                step_ctx.record_usage(tokens=span.total_tokens, cost_usd=span.cost_usd)
        except CircuitBreakerTrip as exc:
            tripped_reason = exc.reason
            tripped_turn = turn + 1
            break

    governed_dag = tracer.get_trajectory()
    ungov_cost = ungoverned_dag.total_cost_usd()
    gov_cost = governed_dag.total_cost_usd()
    savings_pct = 100.0 * (1.0 - (gov_cost / ungov_cost)) if ungov_cost > 0 else 0.0

    return {
        "scenario_id": "vendor_contract_negotiation_loop",
        "ungoverned_turns": len(ungoverned_dag.tool_nodes),
        "ungoverned_tokens": ungoverned_dag.total_tokens(),
        "ungoverned_cost_usd": round(ungov_cost, 6),
        "governed_turns_before_trip": len(governed_dag.tool_nodes),
        "tripped_on_attempt": tripped_turn,
        "trip_reason": tripped_reason.value if tripped_reason else None,
        "governed_tokens": governed_dag.total_tokens(),
        "governed_cost_usd": round(gov_cost, 6),
        "cost_savings_pct": round(savings_pct, 1),
        "tokens_saved": ungoverned_dag.total_tokens() - governed_dag.total_tokens(),
        "governed_trajectory": governed_dag,
        "ungoverned_trajectory": ungoverned_dag,
    }
