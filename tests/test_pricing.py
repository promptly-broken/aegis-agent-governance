"""Unit tests for pricing registry and Apple Silicon hardware TCO model."""

import pytest

from aegis.core.models import ModelTier, Span, SpanType, TrajectoryDAG
from aegis.core.pricing import ModelPricing, PricingRegistry


def test_pricing_registry_default_models():
    registry = PricingRegistry()
    gpt4o = registry.get("gpt-4o")
    assert gpt4o is not None
    assert gpt4o.tier == ModelTier.FRONTIER_CLOUD
    assert gpt4o.input_cost_per_million == 2.50
    assert gpt4o.output_cost_per_million == 10.00


def test_pricing_frontier_cloud_calculation():
    registry = PricingRegistry()
    span = Span(
        name="test_span",
        span_type=SpanType.LLM_INFERENCE,
        model_name="gpt-4o",
        model_tier=ModelTier.FRONTIER_CLOUD,
        start_time_ms=0.0,
        end_time_ms=1000.0,
        prompt_tokens=1_000_000,
        completion_tokens=500_000,
    )
    cost = registry.calculate_span_cost(span)
    # 1M in = $2.50, 0.5M out = $5.00 => $7.50
    assert pytest.approx(cost, rel=1e-3) == 7.50


def test_pricing_apple_silicon_tco():
    registry = PricingRegistry()
    # 10 seconds of inference on MLX model (0 token cost, pure hardware + electricity)
    span = Span(
        name="local_inference",
        span_type=SpanType.LLM_INFERENCE,
        model_name="mlx-community/Qwen2.5-7B-Instruct-4bit",
        model_tier=ModelTier.LOCAL_MLX,
        start_time_ms=0.0,
        end_time_ms=10000.0,  # 10 seconds
        prompt_tokens=500,
        completion_tokens=250,
    )
    cost = registry.calculate_span_cost(span)
    assert cost > 0.0
    # 10s should cost a small fraction of a cent (e.g. ~$0.0003)
    assert cost < 0.01


def test_pricing_custom_model_registration():
    registry = PricingRegistry()
    custom = ModelPricing(
        model_name="custom-fast-llm",
        tier=ModelTier.FRONTIER_CLOUD,
        input_cost_per_million=1.00,
        output_cost_per_million=2.00,
        fixed_cost_per_invocation=0.001,
    )
    registry.register(custom)
    retrieved = registry.get("custom-fast-llm")
    assert retrieved is not None

    cost = custom.compute_cost(1_000_000, 1_000_000)
    assert pytest.approx(cost, rel=1e-3) == 1.00 + 2.00 + 0.001


def test_pricing_trajectory_cost_aggregation():
    registry = PricingRegistry()
    dag = TrajectoryDAG()
    dag.add_span(
        Span(
            name="s1",
            span_type=SpanType.LLM_INFERENCE,
            model_name="gpt-4o-mini",
            start_time_ms=0,
            end_time_ms=100,
            prompt_tokens=1000,
            completion_tokens=100,
        )
    )
    total_cost = registry.calculate_trajectory_cost(dag, infra_overhead_usd=0.002)
    assert total_cost > 0.002
