"""Live Ollama-backed Frontier Cloud wire emulation tests.

Uses Ollama's native OpenAI (/v1/chat/completions) and Anthropic (/v1/messages)
wire-compatibility endpoints on Apple Silicon so TaskDispatcher._dispatch_cloud
executes real socket I/O and neural inference at $0 cloud cost.
Automatically skips live daemon tests when local Ollama is not running.
"""

from __future__ import annotations

import json
from typing import List
import urllib.request

import pytest

from aegis.core.models import ModelTier
from aegis.core.pricing import ModelPricing, PricingRegistry
from aegis.finops.router import TaskDispatcher

OLLAMA_HOST = "http://127.0.0.1:11434"


def _get_available_ollama_models() -> List[str]:
    """Return list of locally installed Ollama models if daemon is reachable, else []."""
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


def test_ollama_frontier_cloud_wire_emulation_config():
    """Verify TaskDispatcher can be configured to route FRONTIER_CLOUD wire calls to Ollama's /v1 endpoints."""
    registry = PricingRegistry()
    # Map local model alias to GPT-4o frontier pricing for realistic FinOps simulation
    registry.register(
        ModelPricing(
            model_name="qwen2.5-coder:7b-instruct",
            tier=ModelTier.FRONTIER_CLOUD,
            input_cost_per_million=2.50,
            output_cost_per_million=10.00,
            cached_input_cost_per_million=1.25,
        )
    )
    dispatcher = TaskDispatcher(
        pricing_registry=registry,
        cloud_api_keys={"openai": "ollama-local-key", "anthropic": "ollama-local-key"},
        cloud_base_urls={
            "openai": f"{OLLAMA_HOST}/v1",
            "anthropic": f"{OLLAMA_HOST}/v1",
        },
        cloud_timeout_seconds=30.0,
    )
    provider, key = dispatcher._resolve_cloud_provider_and_key("qwen2.5-coder:7b-instruct")
    assert provider == "openai"
    assert key == "ollama-local-key"
    assert dispatcher.cloud_timeout_seconds == 30.0


def test_live_ollama_openai_and_anthropic_wire_endpoints():
    """Run a live neural request through TaskDispatcher._dispatch_cloud against Ollama's /v1 endpoints."""
    models = _get_available_ollama_models()
    if not models:
        pytest.skip("Local Ollama daemon is not running or has no models installed")
    chosen_model = "gemma3:270m" if "gemma3:270m" in models else models[0]

    dispatcher = TaskDispatcher(
        cloud_api_keys={"openai": "ollama-local-key", "anthropic": "ollama-local-key"},
        cloud_base_urls={
            "openai": f"{OLLAMA_HOST}/v1",
            "anthropic": f"{OLLAMA_HOST}/v1",
        },
        cloud_timeout_seconds=30.0,
    )

    res = dispatcher.execute_task(
        "Reply with the single word: READY",
        metadata={"max_tokens": 16, "temperature": 0.0},
        force_tier=ModelTier.FRONTIER_CLOUD,
        force_model=chosen_model,
    )
    assert res.is_fallback is False
    assert res.tier == ModelTier.FRONTIER_CLOUD
    assert res.metadata["source"] == "cloud_live"
    assert len(res.content.strip()) > 0
    assert res.prompt_tokens > 0
    assert res.completion_tokens > 0
