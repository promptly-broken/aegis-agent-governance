"""Unit tests for Trajectory Determinism Index (TDI)."""

import pytest

from aegis.core.models import ToolNode, TrajectoryDAG
from aegis.eval_engine.tdi import compute_dag_edit_distance, compute_tdi


def create_linear_trajectory(traj_id: str, tools: list[tuple[str, dict]]) -> TrajectoryDAG:
    dag = TrajectoryDAG(trajectory_id=traj_id, resolved=True)
    prev_node_id = None
    for idx, (name, args) in enumerate(tools):
        node_id = f"node_{idx}"
        node = ToolNode(node_id=node_id, tool_name=name, arguments=args)
        dag.add_tool_node(node)
        if prev_node_id is not None:
            dag.add_dependency(prev_node_id, node_id)
        prev_node_id = node_id
    return dag


def test_tdi_single_trajectory():
    dag = create_linear_trajectory("t1", [("search", {"q": "python"}), ("fetch", {"url": "https://python.org"})])
    result = compute_tdi([dag])
    assert result.tdi == 1.0
    assert result.num_trajectories == 1
    assert result.is_deterministic is True
    assert result.mean_normalized_distance == 0.0


def test_tdi_identical_trajectories():
    t1 = create_linear_trajectory("t1", [("tool_a", {"k1": 1}), ("tool_b", {"k2": 2})])
    t2 = create_linear_trajectory("t2", [("tool_a", {"k1": 1}), ("tool_b", {"k2": 2})])
    t3 = create_linear_trajectory("t3", [("tool_a", {"k1": 1}), ("tool_b", {"k2": 2})])

    result = compute_tdi([t1, t2, t3])
    assert result.tdi == 1.0
    assert result.num_trajectories == 3
    assert result.is_deterministic is True
    assert result.mean_normalized_distance == 0.0
    assert result.variance == 0.0


def test_tdi_divergent_tools():
    t1 = create_linear_trajectory("t1", [("tool_a", {}), ("tool_b", {})])
    t2 = create_linear_trajectory("t2", [("tool_x", {}), ("tool_y", {})])

    result = compute_tdi([t1, t2])
    # Different tools with no overlap
    assert result.tdi < 0.5
    assert result.is_deterministic is False
    assert result.num_trajectories == 2


def test_tdi_schema_key_drift():
    # Identical tool names, but differing schema keys
    t1 = create_linear_trajectory("t1", [("order_search", {"order_id": "123", "user_id": "456"})])
    t2 = create_linear_trajectory("t2", [("order_search", {"order_id": "123", "filter_date": "2026"})])

    result = compute_tdi([t1, t2])
    assert 0.0 < result.tdi < 1.0
    # Higher determinism than completely different tools
    assert result.tdi > 0.6


def test_tdi_empty_raises():
    with pytest.raises(ValueError, match="At least one trajectory"):
        compute_tdi([])


def test_tdi_bounded_range():
    # Generate 5 random-ish trajectories
    trajectories = [
        create_linear_trajectory(f"t_{i}", [("tool_a", {}), (f"tool_{i}", {})])
        for i in range(5)
    ]
    res = compute_tdi(trajectories)
    assert 0.0 <= res.tdi <= 1.0
    assert 0.0 <= res.mean_normalized_distance <= 1.0
    assert res.variance >= 0.0


def test_tdi_semantic_argument_value_drift():
    # Identical tool names and identical argument keys, but divergent argument values
    t_exact_1 = create_linear_trajectory(
        "t1",
        [
            ("issue_refund", {"order_id": "ORD-100", "amount": 50.0, "meta": {"tier": "gold"}}),
        ],
    )
    t_exact_2 = create_linear_trajectory(
        "t2",
        [
            ("issue_refund", {"order_id": "ORD-100", "amount": 50.0, "meta": {"tier": "gold"}}),
        ],
    )
    t_value_drift = create_linear_trajectory(
        "t3",
        [
            ("issue_refund", {"order_id": "ORD-999", "amount": 5000.0, "meta": {"tier": "bronze"}}),
        ],
    )

    assert compute_tdi([t_exact_1, t_exact_2]).tdi == 1.0

    # With value distance enabled (default), divergent values lower TDI below 1.0
    res_with_val = compute_tdi([t_exact_1, t_value_drift], include_value_distance=True)
    assert 0.70 <= res_with_val.tdi < 1.0
    assert res_with_val.tdi_score == res_with_val.tdi

    # With value distance disabled (schema-keys-only mode), TDI is 1.0 since keys match
    res_keys_only = compute_tdi([t_exact_1, t_value_drift], include_value_distance=False)
    assert res_keys_only.tdi == 1.0

