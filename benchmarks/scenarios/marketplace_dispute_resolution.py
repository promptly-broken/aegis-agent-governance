"""Synthetic multi-agent commerce benchmark: Marketplace dispute resolution scenario."""

from __future__ import annotations

import random
from typing import List, Tuple
import uuid

from aegis.core.models import ModelTier, Span, SpanType, ToolNode, TrajectoryDAG


def generate_canonical_dispute_trajectory(
    scenario_id: str = "order_dispute_lost_in_transit",
    variation_seed: int = 42,
    divergence_factor: float = 0.0,
    inject_cycle: bool = False,
) -> TrajectoryDAG:
    """Generate a synthetic multi-agent dispute trajectory.

    Args:
        scenario_id: Scenario identifier.
        variation_seed: Seed for random deviations.
        divergence_factor: 0.0 means perfectly canonical, 1.0 means high divergence.
        inject_cycle: If True, injects a runaway negotiation loop.
    """
    rng = random.Random(variation_seed)
    traj_id = f"traj_{uuid.uuid4().hex[:8]}"

    dag = TrajectoryDAG(
        trajectory_id=traj_id,
        scenario_id=scenario_id,
        resolved=True,
    )

    t_start = 1000.0

    # Step 1: Order Lookup
    order_args = {"order_id": "ORD-99824", "customer_id": "CUST-104"}
    if divergence_factor > 0.5 and rng.random() > 0.5:
        order_args["include_historical_orders"] = True

    n1 = ToolNode(
        node_id="lookup_order",
        tool_name="order_lookup",
        arguments=order_args,
        duration_ms=45.0,
    )
    dag.add_tool_node(n1)
    dag.add_span(
        Span(
            name="order_lookup_span",
            span_type=SpanType.TOOL_CALL,
            model_tier=ModelTier.LOCAL_MLX,
            model_name="mlx-community/Qwen2.5-7B-Instruct-4bit",
            start_time_ms=t_start,
            end_time_ms=t_start + 45.0,
            prompt_tokens=420,
            completion_tokens=85,
            cost_usd=0.00002,
        )
    )

    # Step 2: Shipping Status Tracking
    t_step2 = t_start + 50.0
    n2 = ToolNode(
        node_id="track_shipment",
        tool_name="tracking_status",
        arguments={"carrier": "FedEx", "tracking_number": "TRK-77189"},
        duration_ms=60.0,
    )
    dag.add_tool_node(n2)
    dag.add_dependency(n1.node_id, n2.node_id)
    dag.add_span(
        Span(
            name="tracking_status_span",
            span_type=SpanType.TOOL_CALL,
            model_tier=ModelTier.LOCAL_MLX,
            model_name="mlx-community/Qwen2.5-7B-Instruct-4bit",
            start_time_ms=t_step2,
            end_time_ms=t_step2 + 60.0,
            prompt_tokens=310,
            completion_tokens=45,
            cost_usd=0.00001,
        )
    )

    # Step 3: Policy Lookup or Divergent Path
    t_step3 = t_step2 + 65.0
    if divergence_factor > 0.7 and rng.random() > 0.4:
        # Divergent action: unnecessary inventory check
        n3 = ToolNode(
            node_id="inventory_check",
            tool_name="warehouse_inventory_lookup",
            arguments={"sku": "SKU-9901"},
            duration_ms=80.0,
        )
        dag.add_tool_node(n3)
        dag.add_dependency(n2.node_id, n3.node_id)
        prev_node_id = n3.node_id
    else:
        n3 = ToolNode(
            node_id="policy_check",
            tool_name="dispute_policy_lookup",
            arguments={"policy_type": "lost_in_transit", "claim_amount": 149.99},
            duration_ms=50.0,
        )
        dag.add_tool_node(n3)
        dag.add_dependency(n2.node_id, n3.node_id)
        prev_node_id = n3.node_id

    # Step 4: Issue Refund / Settlement (Frontier Cloud reasoning)
    t_step4 = t_step3 + 55.0
    n4 = ToolNode(
        node_id="settlement_action",
        tool_name="issue_customer_refund",
        arguments={"order_id": "ORD-99824", "amount_usd": 149.99, "reason": "lost_package"},
        duration_ms=110.0,
    )
    dag.add_tool_node(n4)
    dag.add_dependency(prev_node_id, n4.node_id)
    dag.add_span(
        Span(
            name="arbitration_decision_span",
            span_type=SpanType.LLM_INFERENCE,
            model_tier=ModelTier.FRONTIER_CLOUD,
            model_name="gpt-4o",
            start_time_ms=t_step4,
            end_time_ms=t_step4 + 110.0,
            prompt_tokens=1850,
            completion_tokens=220,
            cost_usd=0.00682,
        )
    )

    # Step 5: Optional Cycle Injection
    if inject_cycle:
        # Create an artificial loop: notify_parties -> dispute_escalation -> notify_parties
        n5 = ToolNode(
            node_id="notify_parties",
            tool_name="customer_notification",
            arguments={"order_id": "ORD-99824", "status": "refund_approved"},
            duration_ms=40.0,
        )
        n6 = ToolNode(
            node_id="merchant_appeal",
            tool_name="merchant_dispute_appeal",
            arguments={"order_id": "ORD-99824", "appeal_reason": "carrier_liability"},
            duration_ms=45.0,
        )
        dag.add_tool_node(n5)
        dag.add_tool_node(n6)
        dag.add_dependency(n4.node_id, n5.node_id)
        dag.add_dependency(n5.node_id, n6.node_id)
        dag.add_dependency(n6.node_id, n5.node_id)  # Cycle!

    return dag


def generate_benchmark_rollouts(
    num_rollouts: int = 5, divergence_factor: float = 0.1
) -> List[TrajectoryDAG]:
    """Generate K Monte Carlo rollouts for statistical benchmarking."""
    rollouts: List[TrajectoryDAG] = []
    for i in range(num_rollouts):
        traj = generate_canonical_dispute_trajectory(
            variation_seed=100 + i,
            divergence_factor=divergence_factor,
        )
        rollouts.append(traj)
    return rollouts
