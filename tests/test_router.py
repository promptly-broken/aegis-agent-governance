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


def test_router_structural_and_logprob_entropy():
    router = HybridComplexityRouter(entropy_threshold=0.65)

    # 1. Token logprob entropy signal: low uncertainty -> LOCAL_MLX
    low_uncertainty_decision = router.route(
        "General prompt without keywords",
        metadata={"logprobs": [-0.05, -0.02, -0.04, -0.03]},
    )
    assert low_uncertainty_decision.selected_tier == ModelTier.LOCAL_MLX
    assert low_uncertainty_decision.complexity == TaskComplexity.DETERMINISTIC_LOW

    # 2. Token logprob entropy signal: high uncertainty -> FRONTIER_CLOUD
    high_uncertainty_decision = router.route(
        "General prompt without keywords",
        metadata={"logprobs": [-2.2, -1.9, -2.5, -2.1]},
    )
    assert high_uncertainty_decision.selected_tier == ModelTier.FRONTIER_CLOUD
    assert high_uncertainty_decision.complexity == TaskComplexity.REASONING_HIGH

    # 3. Structural complexity: large tool fanout + deeply nested JSON schema -> FRONTIER_CLOUD
    nested_schema = {
        "properties": {
            "level1": {
                "properties": {
                    "level2": {
                        "properties": {
                            "level3": {"properties": {"level4": {"type": "string"}}}
                        }
                    }
                }
            }
        }
    }
    structural_decision = router.route(
        "Coordinate workflow across available tools",
        metadata={
            "tools": [f"tool_{i}" for i in range(8)],
            "json_schema": nested_schema,
        },
    )
    assert structural_decision.selected_tier == ModelTier.FRONTIER_CLOUD
    assert structural_decision.complexity == TaskComplexity.REASONING_HIGH


def test_task_dispatcher_mlx_live_execution_and_caching(monkeypatch):
    import sys
    import types
    from aegis.finops.router import TaskDispatcher

    load_calls = []

    class FakeTokenizer:
        def encode(self, text: str):
            return text.split()

    fake_mlx = types.ModuleType("mlx_lm")

    def fake_load(model_name: str):
        load_calls.append(model_name)
        return ("fake_model_weights", FakeTokenizer())

    def fake_generate(model_obj, tokenizer, prompt: str, max_tokens: int = 256, verbose: bool = False):
        return '{"customer_id": "CUST-42", "order_id": "ORD-99"}'

    fake_mlx.load = fake_load  # type: ignore[attr-defined]
    fake_mlx.generate = fake_generate  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx)
    TaskDispatcher._mlx_model_cache.clear()

    dispatcher = TaskDispatcher()
    res1 = dispatcher.execute_task("Extract customer ID from JSON", force_tier=ModelTier.LOCAL_MLX)
    res2 = dispatcher.execute_task("Extract order ID from JSON", force_tier=ModelTier.LOCAL_MLX)

    assert res1.tier == ModelTier.LOCAL_MLX
    assert res1.is_fallback is False
    assert res1.metadata["source"] == "mlx_live"
    assert "CUST-42" in res1.content
    assert res1.prompt_tokens == 5
    # Verify model was loaded only once and cached across invocations
    assert len(load_calls) == 1
    assert res2.is_fallback is False


def test_task_dispatcher_live_cloud_providers(monkeypatch):
    import io
    import json
    import urllib.request
    from aegis.finops.router import TaskDispatcher

    def fake_urlopen(req: urllib.request.Request, timeout: float = 1.0):
        url = req.full_url
        if "anthropic.com" in url:
            payload = {
                "content": [{"type": "text", "text": "Anthropic arbitration ruling"}],
                "usage": {"input_tokens": 42, "output_tokens": 18},
            }
        elif "generativelanguage.googleapis.com" in url:
            payload = {
                "candidates": [{"content": {"parts": [{"text": "Gemini arbitration ruling"}]}}],
                "usageMetadata": {"promptTokenCount": 36, "candidatesTokenCount": 14},
            }
        else:
            payload = {
                "choices": [{"message": {"content": "OpenAI arbitration ruling"}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            }

        resp = io.BytesIO(json.dumps(payload).encode("utf-8"))
        resp. __enter__ = lambda s: s  # type: ignore[attr-defined]
        resp.__exit__ = lambda *args: None  # type: ignore[attr-defined]
        return resp

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    dispatcher = TaskDispatcher(
        cloud_api_keys={
            "openai": "sk-test-openai",
            "anthropic": "sk-ant-test",
            "gemini": "ai-test-gemini",
        }
    )

    res_oai = dispatcher.execute_task(
        "Arbitrate dispute", force_tier=ModelTier.FRONTIER_CLOUD, force_model="gpt-4o"
    )
    assert res_oai.content == "OpenAI arbitration ruling"
    assert res_oai.prompt_tokens == 50
    assert res_oai.completion_tokens == 20
    assert res_oai.metadata["source"] == "cloud_live"
    assert res_oai.metadata["provider"] == "openai"

    res_ant = dispatcher.execute_task(
        "Arbitrate dispute",
        force_tier=ModelTier.FRONTIER_CLOUD,
        force_model="claude-3-5-sonnet-20241022",
    )
    assert res_ant.content == "Anthropic arbitration ruling"
    assert res_ant.prompt_tokens == 42
    assert res_ant.completion_tokens == 18
    assert res_ant.metadata["provider"] == "anthropic"

    res_gem = dispatcher.execute_task(
        "Arbitrate dispute",
        force_tier=ModelTier.FRONTIER_CLOUD,
        force_model="gemini-2.0-flash",
    )
    assert res_gem.content == "Gemini arbitration ruling"
    assert res_gem.prompt_tokens == 36
    assert res_gem.completion_tokens == 14
    assert res_gem.metadata["provider"] == "gemini"


