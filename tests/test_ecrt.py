"""Unit tests for Effective Cost per Resolved Trajectory (ECRT)."""

import pytest

from aegis.core.models import ModelTier, Span, SpanType, TrajectoryDAG
from aegis.core.pricing import PricingRegistry
from aegis.finops.ecrt import compute_ecrt


def create_sample_trajectory(
    traj_id: str,
    resolved: bool,
    prompt_tokens: int = 1000,
    completion_tokens: int = 200,
    model_name: str = "gpt-4o",
    model_tier: ModelTier = ModelTier.FRONTIER_CLOUD,
) -> TrajectoryDAG:
    dag = TrajectoryDAG(trajectory_id=traj_id, resolved=resolved)
    dag.add_span(
        Span(
            name="inference_span",
            span_type=SpanType.LLM_INFERENCE,
            model_name=model_name,
            model_tier=model_tier,
            start_time_ms=0.0,
            end_time_ms=100.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    )
    return dag


def test_ecrt_perfect_resolution():
    # 2 trajectories, both resolved
    t1 = create_sample_trajectory("t1", resolved=True)
    t2 = create_sample_trajectory("t2", resolved=True)

    res = compute_ecrt([t1, t2])
    assert res.resolution_rate == 1.0
    assert res.resolved_trajectories == 2
    assert res.total_trajectories == 2
    assert res.ecrt_usd == res.average_cost_per_attempt_usd
    assert res.is_budget_compliant is True


def test_ecrt_partial_resolution():
    # 4 trajectories, 2 resolved, 2 failed
    # Each run has identical cost C. Total cost = 4C.
    # Resolution rate = 0.5.
    # ECRT = (4C) / 0.5 = 2 * Total Cost = Total Cost / Resolved = 4C / 2 = 2C.
    t1 = create_sample_trajectory("t1", resolved=True)
    t2 = create_sample_trajectory("t2", resolved=True)
    t3 = create_sample_trajectory("t3", resolved=False)
    t4 = create_sample_trajectory("t4", resolved=False)

    res = compute_ecrt([t1, t2, t3, t4])
    assert res.resolution_rate == 0.5
    assert res.resolved_trajectories == 2
    assert res.total_trajectories == 4
    # ECRT is twice the average cost per attempt
    assert pytest.approx(res.ecrt_usd, rel=1e-3) == res.average_cost_per_attempt_usd / 0.5


def test_ecrt_zero_resolution_returns_inf():
    t1 = create_sample_trajectory("t1", resolved=False)
    t2 = create_sample_trajectory("t2", resolved=False)

    res = compute_ecrt([t1, t2])
    assert res.resolution_rate == 0.0
    assert res.ecrt_usd == float("inf")
    assert res.confidence_interval_95 == (float("inf"), float("inf"))


def test_ecrt_with_infra_overhead():
    t1 = create_sample_trajectory("t1", resolved=True)
    infra_usd = 0.10

    res_no_infra = compute_ecrt([t1], infra_overhead_usd=0.0)
    res_with_infra = compute_ecrt([t1], infra_overhead_usd=infra_usd)

    assert pytest.approx(res_with_infra.total_cost_usd, rel=1e-4) == res_no_infra.total_cost_usd + infra_usd
    assert res_with_infra.infra_overhead_usd == infra_usd


def test_ecrt_sla_budget_breach():
    t1 = create_sample_trajectory("t1", resolved=True)
    # Total cost of gpt-4o with 1000 in, 200 out is 1000 * 2.5e-6 + 200 * 10e-6 = 0.0025 + 0.002 = 0.0045 USD
    res = compute_ecrt([t1], max_budget_usd=0.001)  # Budget tighter than cost
    assert res.is_budget_compliant is False

    res_ok = compute_ecrt([t1], max_budget_usd=0.010)
    assert res_ok.is_budget_compliant is True


def test_ecrt_empty_dataset_raises():
    with pytest.raises(ValueError, match="Cannot compute ECRT on empty"):
        compute_ecrt([])
