"""Unit tests for Core Data Models and NetworkX DAG translations."""

import pytest

from aegis.core.models import (
    MCPStateTransition,
    ModelTier,
    Span,
    SpanType,
    ToolNode,
    TrajectoryDAG,
)


def test_span_properties():
    span = Span(
        name="test_span",
        span_type=SpanType.LLM_INFERENCE,
        start_time_ms=100.0,
        end_time_ms=350.0,
        prompt_tokens=400,
        completion_tokens=150,
        ttft_ms=50.0,
    )
    assert span.duration_ms == 250.0
    assert span.total_tokens == 550
    assert span.ttft_ms == 50.0


def test_tool_node_auto_schema_keys():
    node = ToolNode(
        tool_name="database_query",
        arguments={"table": "users", "limit": 10, "offset": 0},
    )
    assert node.schema_keys == ["limit", "offset", "table"]


def test_trajectory_dag_networkx_conversion():
    dag = TrajectoryDAG(trajectory_id="dag-01", scenario_id="scenario-x", resolved=True)
    n1 = ToolNode(node_id="n1", tool_name="fetch_user", arguments={"user_id": 1})
    n2 = ToolNode(node_id="n2", tool_name="send_email", arguments={"to": "a@b.com"})
    dag.add_tool_node(n1)
    dag.add_tool_node(n2)
    dag.add_dependency("n1", "n2")

    g = dag.to_networkx()
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
    assert "n1" in g and "n2" in g
    assert g.nodes["n1"]["tool_name"] == "fetch_user"

    # Test roundtrip from_networkx
    reconstructed = TrajectoryDAG.from_networkx(g)
    assert reconstructed.trajectory_id == "dag-01"
    assert reconstructed.scenario_id == "scenario-x"
    assert reconstructed.resolved is True
    assert len(reconstructed.tool_nodes) == 2
    assert reconstructed.edges == [("n1", "n2")]


def test_trajectory_dag_cycles_and_toposort():
    dag = TrajectoryDAG(trajectory_id="topo_dag")
    n1 = ToolNode(node_id="step1", tool_name="tool_1")
    n2 = ToolNode(node_id="step2", tool_name="tool_2")
    n3 = ToolNode(node_id="step3", tool_name="tool_3")
    dag.add_tool_node(n1)
    dag.add_tool_node(n2)
    dag.add_tool_node(n3)
    dag.add_dependency("step1", "step2")
    dag.add_dependency("step2", "step3")

    assert dag.has_cycle() is False
    assert dag.find_cycles() == []
    assert dag.topological_sort() == ["step1", "step2", "step3"]
    assert dag.get_tool_sequence() == ["tool_1", "tool_2", "tool_3"]

    # Introduce a cycle
    dag.add_dependency("step3", "step1")
    assert dag.has_cycle() is True
    cycles = dag.find_cycles()
    assert len(cycles) > 0


def test_trajectory_dag_token_and_cost_aggregation():
    dag = TrajectoryDAG()
    dag.add_span(
        Span(
            name="s1",
            span_type=SpanType.LLM_INFERENCE,
            model_tier=ModelTier.LOCAL_MLX,
            start_time_ms=0,
            end_time_ms=100,
            prompt_tokens=50,
            completion_tokens=25,
            cost_usd=0.0001,
        )
    )
    dag.add_span(
        Span(
            name="s2",
            span_type=SpanType.LLM_INFERENCE,
            model_tier=ModelTier.FRONTIER_CLOUD,
            start_time_ms=100,
            end_time_ms=300,
            prompt_tokens=200,
            completion_tokens=100,
            cost_usd=0.0020,
        )
    )

    assert dag.total_prompt_tokens() == 250
    assert dag.total_completion_tokens() == 125
    assert dag.total_tokens() == 375
    assert dag.total_latency_ms() == 300.0
    assert pytest.approx(dag.total_cost_usd(), rel=1e-4) == 0.0021
    assert dag.get_model_tiers_used() == {ModelTier.LOCAL_MLX, ModelTier.FRONTIER_CLOUD}
