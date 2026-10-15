"""Unit tests for @assert_agent_governance pytest decorator."""

import pytest

from aegis.core.models import Span, SpanType, ToolNode, TrajectoryDAG
from aegis.pytest_plugin.decorator import (
    GovernanceAssertionError,
    assert_agent_governance,
)


def _make_trajectory(resolved: bool = True, cost: float = 0.005) -> TrajectoryDAG:
    dag = TrajectoryDAG(resolved=resolved)
    dag.add_tool_node(ToolNode(node_id="n1", tool_name="tool_a", arguments={"x": 1}))
    dag.add_span(
        Span(
            name="span1",
            span_type=SpanType.LLM_INFERENCE,
            start_time_ms=0,
            end_time_ms=10,
            cost_usd=cost,
        )
    )
    return dag


def test_decorator_sync_pass():
    @assert_agent_governance(min_tdi=0.85, max_ecrt_usd=0.010, max_loops=2, rollouts=2)
    def agent_test():
        return _make_trajectory()

    result = agent_test()
    assert isinstance(result, list)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_decorator_async_pass():
    @assert_agent_governance(min_tdi=0.85, max_ecrt_usd=0.010, max_loops=2, rollouts=2)
    async def async_agent_test():
        return _make_trajectory()

    result = await async_agent_test()
    assert isinstance(result, list)
    assert len(result) == 2


def test_decorator_fails_on_ecrt_budget():
    @assert_agent_governance(max_ecrt_usd=0.001)  # Budget lower than 0.005
    def agent_test():
        return _make_trajectory(cost=0.005)

    with pytest.raises(GovernanceAssertionError, match="FinOps SLA breach"):
        agent_test()


def test_decorator_fails_on_tdi_regression():
    @assert_agent_governance(min_tdi=0.99)
    def agent_test():
        # Return divergent trajectories
        t1 = TrajectoryDAG(resolved=True)
        t1.add_tool_node(ToolNode(node_id="n1", tool_name="tool_1"))
        t2 = TrajectoryDAG(resolved=True)
        t2.add_tool_node(ToolNode(node_id="n2", tool_name="completely_different_tool"))
        return [t1, t2]

    with pytest.raises(GovernanceAssertionError, match="Trajectory Determinism Index regression"):
        agent_test()
