"""Example: Live Autonomous Agent Governed by Aegis Runtime Control Plane."""

from __future__ import annotations

import time

from aegis.core.models import ModelTier, SpanType
from aegis.core.pricing import PricingRegistry
from aegis.core.tracer import AegisTracer
from aegis.finops.circuit_breaker import (
    BreakerTripReason,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerTrip,
    StreamingCircuitBreaker,
)
from aegis.finops.ecrt import compute_ecrt
from aegis.finops.router import HybridComplexityRouter, execute_routed_task
from aegis.mcp_profiler.state_store import MemoryMCPStateStore


def run_governed_agent_demo() -> None:
    print("=" * 70)
    print("🚀 AEGIS LIVE AGENT GOVERNANCE DEMO")
    print("=" * 70)

    # 1. Initialize Circuit Breaker & Tracer
    config = CircuitBreakerConfig(
        max_cost_usd=0.10,
        max_total_tokens=10_000,
        max_tool_calls=10,
        max_cycle_repetitions=2,
    )
    breaker = CircuitBreaker(config=config)
    pricing = PricingRegistry()
    tracer = AegisTracer(
        scenario_id="enterprise_dispute_arbitration",
        circuit_breaker=breaker,
        pricing_registry=pricing,
    )

    print("\n[Step 1] Initializing AegisTracer with active Circuit Breaker...")

    # 2. Hybrid Edge-Cloud Routing for Prompt Steps
    router = HybridComplexityRouter(entropy_threshold=0.65)

    step1_prompt = "Extract customer ID and order number from raw dispute ticket JSON."
    decision1 = router.route(step1_prompt)
    print(f"\n[Step 2] Step 1 Routing: '{step1_prompt[:40]}...'")
    print(f"         Tier: {decision1.selected_tier.value} | Model: {decision1.recommended_model}")
    print(f"         Rationale: {decision1.rationale}")

    # Execute step 1 via dispatcher
    exec_res1 = execute_routed_task(step1_prompt, router=router)
    print(f"         Executed in {exec_res1.duration_ms:.1f}ms | Cost: ${exec_res1.cost_usd:.6f}")

    # 3. Trace Execution Spans and Tool Calls
    with tracer:
        with tracer.span("ticket_parsing", SpanType.INTERNAL_REASONING) as s:
            s.record_tokens(prompt_tokens=exec_res1.prompt_tokens, completion_tokens=exec_res1.completion_tokens)
            s.set_attribute("tier", exec_res1.tier.value)

        t1 = tracer.record_tool_call(
            tool_name="order_db_query",
            arguments={"order_id": "ORD-99824"},
            output={"item": "Enterprise Server Rack", "amount": 1499.00, "status": "delivered"},
            duration_ms=35.0,
        )

        t2 = tracer.record_tool_call(
            tool_name="carrier_telemetry_fetch",
            arguments={"tracking_id": "TRK-8821990"},
            output={"carrier": "FedEx", "status": "damaged_in_transit"},
            duration_ms=42.0,
            depends_on=[t1.node_id],
        )

        step2_prompt = "Arbitrate dispute liability and decide policy concession tradeoff."
        decision2 = router.route(step2_prompt)
        print(f"\n[Step 3] Step 2 Routing: '{step2_prompt[:40]}...'")
        print(f"         Tier: {decision2.selected_tier.value} | Model: {decision2.recommended_model}")
        print(f"         Rationale: {decision2.rationale}")

        tracer.record_llm_call(
            name="arbitration_reasoning",
            model_name=decision2.recommended_model,
            prompt_tokens=850,
            completion_tokens=220,
            duration_ms=320.0,
            model_tier=decision2.selected_tier,
        )

        # 4. Model Context Protocol (MCP) Stateful Session Persistence
        mcp_store = MemoryMCPStateStore()
        ptr, metrics = mcp_store.save_state(
            "session_dispute_99824",
            {"dispute_id": 99824, "liability": "carrier", "concession_amount": 1499.00},
        )
        tracer.record_mcp_transition(
            server_name="commerce_mcp",
            is_stateful=True,
            state_payload_bytes=metrics.payload_bytes,
            context_token_overhead=32,
            latency_ms=metrics.write_latency_ms,
        )
        print(f"\n[Step 4] MCP State Persisted. Session Pointer: {ptr}")

        tracer.mark_resolved(True)

    dag = tracer.get_trajectory()
    print("\n[Step 5] Trajectory Finalized:")
    print(f"         Total Spans: {len(dag.spans)}")
    print(f"         Total Tool Nodes: {len(dag.tool_nodes)}")
    print(f"         Total Tokens: {dag.total_tokens():,}")
    print(f"         Total Trajectory Cost: ${dag.total_cost_usd():.6f}")

    # 5. Compute ECRT Telemetry
    ecrt_metric = compute_ecrt([dag], pricing_registry=pricing)
    print(f"\n[Step 6] Unit Economics (ECRT): ${ecrt_metric.ecrt_usd:.6f} / resolved trajectory")

    # 6. Streaming Circuit Breaker Lookahead Interception Demo
    print("\n[Step 7] Demonstrating Live Streaming Circuit Breaker Interception:")
    scb = StreamingCircuitBreaker(config=CircuitBreakerConfig(max_cycle_repetitions=2))
    try:
        tools_to_attempt = ["carrier_query", "db_fetch", "carrier_query", "db_fetch"]
        for idx, tool in enumerate(tools_to_attempt, start=1):
            print(f"         Dispatching Tool #{idx}: '{tool}'...")
            with scb.guard_step(tool) as step:
                step.record_usage(tokens=120, cost_usd=0.0005)
    except CircuitBreakerTrip as exc:
        print(f"\n         🛡️ CIRCUIT BREAKER TRIPPED AS DESIGNED!")
        print(f"         Reason: {exc.reason.value}")
        print(f"         Message: {exc}")

    print("\n" + "=" * 70)
    print("✅ DEMO EXECUTION COMPLETE - ALL GOVERNANCE GATES OPERATIONAL")
    print("=" * 70)


if __name__ == "__main__":
    run_governed_agent_demo()
