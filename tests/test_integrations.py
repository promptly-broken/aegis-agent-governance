"""Tests for Agent Framework Middleware: governed_tool, LangGraph, OpenAI Agents SDK, and MCP Session Proxy."""

import pytest

from aegis.core.models import ModelTier, SpanType
from aegis.core.tracer import AegisTracer
from aegis.finops.circuit_breaker import (
    BreakerTripReason,
    CircuitBreakerConfig,
    CircuitBreakerTrip,
    StreamingCircuitBreaker,
)
from aegis.integrations import (
    GovernedMCPSessionProxy,
    LangGraphGovernanceCallback,
    OpenAIAgentsGovernanceHooks,
    governed_tool,
)


def test_governed_tool_sync_and_cycle_guard():
    tracer = AegisTracer(scenario_id="sync_tool_test")
    breaker = StreamingCircuitBreaker(CircuitBreakerConfig(max_cycle_repetitions=2))

    @governed_tool(tracer=tracer, streaming_breaker=breaker)
    def lookup_order(order_id: str, include_items: bool = True) -> dict:
        return {"order_id": order_id, "include_items": include_items}

    res1 = lookup_order("ORD-1")
    assert res1["order_id"] == "ORD-1"

    # Second consecutive call to lookup_order trips max_cycle_repetitions=2
    with pytest.raises(CircuitBreakerTrip) as exc_info:
        lookup_order("ORD-2")

    assert exc_info.value.reason == BreakerTripReason.CYCLIC_LOOP_DETECTED
    dag = tracer.get_trajectory()
    assert len(dag.tool_nodes) == 1
    node = list(dag.tool_nodes.values())[0]
    assert node.tool_name == "lookup_order"
    assert node.arguments == {"order_id": "ORD-1", "include_items": True}


@pytest.mark.asyncio
async def test_governed_tool_async_chaining():
    tracer = AegisTracer(scenario_id="async_tool_chain")
    breaker = StreamingCircuitBreaker(CircuitBreakerConfig(max_cycle_repetitions=3))

    @governed_tool(tracer=tracer, streaming_breaker=breaker)
    async def fetch_customer(customer_id: int) -> dict:
        return {"customer_id": customer_id, "tier": "enterprise"}

    @governed_tool(tracer=tracer, streaming_breaker=breaker)
    async def issue_credit(customer_id: int, amount: float) -> dict:
        return {"credited": amount}

    await fetch_customer(42)
    await issue_credit(42, 150.0)

    dag = tracer.get_trajectory()
    assert len(dag.tool_nodes) == 2
    assert len(dag.edges) == 1
    assert dag.get_tool_sequence() == ["fetch_customer", "issue_credit"]


def test_langgraph_governance_callback_and_node_wrapper():
    cb = LangGraphGovernanceCallback()

    @cb.wrap_node("triage_node")
    def triage_node(state: dict) -> dict:
        run_id = cb.on_llm_start(
            {"name": "gpt-4o"},
            ["Extract order_id from JSON payload"],
            metadata={"ls_model_name": "gpt-4o"},
        )
        cb.on_llm_end(
            {"token_usage": {"prompt_tokens": 110, "completion_tokens": 30}},
            run_id=run_id,
        )

        t_run1 = cb.on_tool_start({"name": "order_search"}, '{"order_id": "ORD-77"}')
        cb.on_tool_end({"status": "found"}, run_id=t_run1)

        t_run2 = cb.on_tool_start({"name": "refund_check"}, {"order_id": "ORD-77"})
        cb.on_tool_end({"eligible": True}, run_id=t_run2)
        return {**state, "resolved": True}

    out = triage_node({"ticket_id": "TCK-1"})
    assert out["resolved"] is True

    dag = cb.tracer.get_trajectory()
    assert len(dag.spans) == 2  # node:triage_node + langgraph_llm:gpt-4o
    assert len(dag.tool_nodes) == 2
    assert len(dag.edges) == 1
    assert dag.total_tokens() == 140
    # Verify LLM span is nested under the node span
    llm_span = [s for s in dag.spans if s.span_type == SpanType.LLM_INFERENCE][0]
    node_span = [s for s in dag.spans if s.name == "node:triage_node"][0]
    assert llm_span.parent_span_id == node_span.span_id


@pytest.mark.asyncio
async def test_openai_agents_governance_hooks():
    hooks = OpenAIAgentsGovernanceHooks()

    await hooks.on_agent_start(context={}, agent="TriageAgent")
    await hooks.on_tool_start(
        context={},
        agent="TriageAgent",
        tool="verify_identity",
        arguments={"user_id": "U-10"},
    )
    await hooks.on_tool_end(
        context={},
        agent="TriageAgent",
        tool="verify_identity",
        result={"verified": True},
        arguments={"user_id": "U-10"},
    )
    await hooks.on_handoff(context={}, from_agent="TriageAgent", to_agent="ArbitrationAgent")
    await hooks.on_agent_end(context={}, agent="TriageAgent", output="Handed off")

    dag = hooks.tracer.get_trajectory()
    assert len(dag.tool_nodes) == 1
    span_names = {s.name for s in dag.spans}
    assert "agent:TriageAgent" in span_names
    assert "handoff:TriageAgent->ArbitrationAgent" in span_names


@pytest.mark.asyncio
async def test_governed_mcp_session_proxy():
    class FakeMCPSession:
        async def call_tool(self, name: str, arguments: dict) -> dict:
            return {"tool": name, "ok": True, "echo": arguments}

        async def read_resource(self, uri: str) -> str:
            return f"resource_payload_for_{uri}"

    proxy = GovernedMCPSessionProxy(
        session=FakeMCPSession(),
        server_name="billing_mcp",
        is_stateful=True,
    )

    res1 = await proxy.call_tool("fetch_invoice", {"invoice_id": "INV-900"})
    assert res1["ok"] is True
    assert proxy.latest_pointer is not None
    assert proxy.latest_pointer.startswith("urn:mcp:state:")

    res2 = await proxy.read_resource("mcp://billing/policy_v2")
    assert "policy_v2" in res2

    dag = proxy.tracer.get_trajectory()
    assert len(dag.tool_nodes) == 1
    assert len(dag.mcp_transitions) == 2
    assert all(t.is_stateful for t in dag.mcp_transitions)
    assert all(t.context_token_overhead == 32 for t in dag.mcp_transitions)

