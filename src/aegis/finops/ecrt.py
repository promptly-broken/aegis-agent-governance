"""Effective Cost per Resolved Trajectory (ECRT) calculator and FinOps unit economics."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from aegis.core.models import ModelTier, TrajectoryDAG
from aegis.core.pricing import PricingRegistry


class ECRTResult(BaseModel):
    """FinOps unit economic telemetry for agent trajectories."""

    model_config = ConfigDict(extra="forbid")

    ecrt_usd: float = Field(..., description="Effective Cost per Resolved Trajectory in USD")
    total_cost_usd: float = Field(..., ge=0.0)
    average_cost_per_attempt_usd: float = Field(..., ge=0.0)
    resolution_rate: float = Field(..., ge=0.0, le=1.0)
    total_trajectories: int = Field(..., ge=1)
    resolved_trajectories: int = Field(..., ge=0)
    infra_overhead_usd: float = Field(default=0.0, ge=0.0)
    cost_by_tier: Dict[str, float] = Field(default_factory=dict)
    confidence_interval_95: Tuple[float, float] = Field(
        ..., description="(Lower bound ECRT, Upper bound ECRT) based on Wilson score interval"
    )
    is_budget_compliant: bool = Field(
        default=True, description="Whether ECRT is within SLA budget threshold"
    )


def _compute_wilson_score_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute Wilson score interval for binomial proportion."""
    if trials == 0:
        return (0.0, 1.0)

    z = stats.norm.ppf((1.0 + confidence) / 2.0)
    p_hat = successes / trials
    denominator = 1.0 + (z**2 / trials)
    center = (p_hat + (z**2 / (2.0 * trials))) / denominator
    half_width = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) / trials) + (z**2 / (4.0 * (trials**2))))
        / denominator
    )

    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return (lower, upper)


def compute_ecrt(
    trajectories: List[TrajectoryDAG],
    pricing_registry: Optional[PricingRegistry] = None,
    infra_overhead_usd: float = 0.0,
    max_budget_usd: Optional[float] = None,
) -> ECRTResult:
    """Compute Effective Cost per Resolved Trajectory (ECRT).

    Formula:
        ECRT = (Total Model Costs across all attempts + C_infra) / P(Task Success)

    Parameters:
        trajectories: List of executed TrajectoryDAG instances.
        pricing_registry: Registry to re-price spans if not pre-computed.
        infra_overhead_usd: Fixed compute or orchestration costs.
        max_budget_usd: Maximum permissible ECRT SLA threshold.
    """
    n = len(trajectories)
    if n == 0:
        raise ValueError("Cannot compute ECRT on empty trajectory dataset.")

    registry = pricing_registry or PricingRegistry()
    total_model_cost = 0.0
    resolved_count = 0
    cost_by_tier: Dict[str, float] = {
        ModelTier.LOCAL_MLX.value: 0.0,
        ModelTier.LOCAL_OLLAMA.value: 0.0,
        ModelTier.FRONTIER_CLOUD.value: 0.0,
        "unspecified": 0.0,
    }

    for traj in trajectories:
        if traj.resolved:
            resolved_count += 1

        for span in traj.spans:
            span_cost = registry.calculate_span_cost(span)
            total_model_cost += span_cost

            tier_key = span.model_tier.value if span.model_tier else "unspecified"
            cost_by_tier[tier_key] = cost_by_tier.get(tier_key, 0.0) + span_cost

    total_expenditure = total_model_cost + infra_overhead_usd
    p_success = resolved_count / n
    avg_cost_per_attempt = total_expenditure / n

    if p_success == 0.0:
        ecrt = float("inf")
        ci_lower = float("inf")
        ci_upper = float("inf")
    else:
        # ECRT = Total Expenditure / P(Task Success) on average basis
        # Note: (total_expenditure / n) / (resolved_count / n) = total_expenditure / resolved_count
        ecrt = round(avg_cost_per_attempt / p_success, 6)

        # Confidence intervals based on Wilson score for P(Task Success)
        p_lower, p_upper = _compute_wilson_score_interval(resolved_count, n, confidence=0.95)
        ci_lower = round(avg_cost_per_attempt / p_upper, 6) if p_upper > 0 else float("inf")
        ci_upper = round(avg_cost_per_attempt / p_lower, 6) if p_lower > 0 else float("inf")

    compliant = True
    if max_budget_usd is not None:
        compliant = ecrt <= max_budget_usd

    # Clean up empty tier records
    clean_tier_costs = {k: round(v, 6) for k, v in cost_by_tier.items() if v > 0.0}

    return ECRTResult(
        ecrt_usd=ecrt,
        total_cost_usd=round(total_expenditure, 6),
        average_cost_per_attempt_usd=round(avg_cost_per_attempt, 6),
        resolution_rate=round(p_success, 4),
        total_trajectories=n,
        resolved_trajectories=resolved_count,
        infra_overhead_usd=round(infra_overhead_usd, 6),
        cost_by_tier=clean_tier_costs,
        confidence_interval_95=(ci_lower, ci_upper),
        is_budget_compliant=compliant,
    )
