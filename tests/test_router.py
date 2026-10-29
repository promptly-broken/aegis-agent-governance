"""Unit tests for Hybrid Complexity Router (Apple Silicon vs Frontier Cloud)."""

from aegis.core.models import ModelTier
from aegis.finops.router import (
    HybridComplexityRouter,
    TaskComplexity,
)


def test_router_deterministic_subtasks_to_local_mlx():
    router = HybridComplexityRouter(entropy_threshold=0.65)
    prompt = "Please extract the JSON fields and validate the schema for customer record."
    decision = router.route(prompt)

    assert decision.selected_tier == ModelTier.LOCAL_MLX
    assert decision.complexity == TaskComplexity.DETERMINISTIC_LOW
    assert decision.estimated_cost_usd < 0.0001
    assert "Offloaded to local Apple Silicon MLX" in decision.rationale


def test_router_complex_reasoning_to_frontier_cloud():
    router = HybridComplexityRouter(entropy_threshold=0.65)
    prompt = "Arbitrate this complex dispute between buyer and merchant, evaluate conflicting evidence and decide liability."
    decision = router.route(prompt)

    assert decision.selected_tier == ModelTier.FRONTIER_CLOUD
    assert decision.complexity == TaskComplexity.REASONING_HIGH
    assert decision.estimated_cost_usd > 0.001
    assert "Frontier Cloud" in decision.rationale


def test_router_explicit_metadata_override():
    router = HybridComplexityRouter()
    decision = router.route("Ambiguous prompt text", metadata={"task_type": "extraction"})
    assert decision.selected_tier == ModelTier.LOCAL_MLX

    decision_high = router.route("Ambiguous prompt text", metadata={"task_type": "dispute_resolution"})
    assert decision_high.selected_tier == ModelTier.FRONTIER_CLOUD


def test_task_dispatcher_local_fallback():
    from aegis.finops.router import TaskDispatcher, execute_routed_task

    # When Ollama is offline, dispatcher gracefully returns deterministic fallback
    dispatcher = TaskDispatcher(ollama_base_url="http://127.0.0.1:59999")
    res = dispatcher.execute_task("Extract postal code from address string", force_tier=ModelTier.LOCAL_OLLAMA)

    assert res.tier == ModelTier.LOCAL_OLLAMA
    assert res.is_fallback is True
    assert res.prompt_tokens > 0
    assert "processed" in res.content
    assert res.cost_usd >= 0.0


def test_task_dispatcher_cloud_execution():
    from aegis.finops.router import execute_routed_task

    res = execute_routed_task(
        "Arbitrate strategic dispute settlement",
        metadata={"task_type": "dispute_resolution"},
    )
    assert res.tier == ModelTier.FRONTIER_CLOUD
    assert res.is_fallback is False
    assert res.cost_usd > 0.0

