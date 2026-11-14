"""End-to-end wire-contract tests against the local HTTP Frontier Wire Twin server."""

from __future__ import annotations

import pytest

from aegis.core.models import ModelTier
from aegis.finops.router import TaskDispatcher
from tests.fixtures.frontier_wire_twin import FrontierWireTwinServer


def test_frontier_wire_twin_standard_text_and_headers():
    with FrontierWireTwinServer() as twin:
        dispatcher = TaskDispatcher(
            cloud_api_keys={
                "openai": "sk-live-wire-oai",
                "anthropic": "sk-ant-live-wire",
                "gemini": "ai-studio-wire-key",
            },
            cloud_base_urls=twin.cloud_base_urls,
        )

        # 1. OpenAI standard chat completion
        res_oai = dispatcher.execute_task(
            "Arbitrate multi-party B2B escrow dispute",
            metadata={"system": "You are an escrow arbitrator.", "temperature": 0.2},
            force_tier=ModelTier.FRONTIER_CLOUD,
            force_model="gpt-4.1",
        )
        assert res_oai.is_fallback is False
        assert res_oai.content == "OpenAI wire-verified resolution via gpt-4.1"
        assert res_oai.prompt_tokens == 64
        assert res_oai.completion_tokens == 24
        assert res_oai.metadata["provider"] == "openai"

        req_oai = twin.state.requests[-1]
        assert req_oai.headers["authorization"] == "Bearer sk-live-wire-oai"
        assert req_oai.body["model"] == "gpt-4.1"
        assert "max_tokens" in req_oai.body
        assert req_oai.body["messages"][0] == {"role": "system", "content": "You are an escrow arbitrator."}

        # 2. Anthropic /v1/messages contract
        res_ant = dispatcher.execute_task(
            "Reconcile conflicting vendor invoices",
            metadata={"system": "Strict financial auditor."},
            force_tier=ModelTier.FRONTIER_CLOUD,
            force_model="claude-sonnet-4-20250514",
        )
        assert res_ant.is_fallback is False
        assert res_ant.content == "Anthropic wire-verified resolution via claude-sonnet-4-20250514"
        assert res_ant.prompt_tokens == 60
        assert res_ant.completion_tokens == 20
        assert res_ant.metadata["provider"] == "anthropic"

        req_ant = twin.state.requests[-1]
        assert req_ant.headers["x-api-key"] == "sk-ant-live-wire"
        assert req_ant.headers["anthropic-version"] == "2023-06-01"
        assert req_ant.body["system"] == "Strict financial auditor."

        # 3. Gemini :generateContent contract
        res_gem = dispatcher.execute_task(
            "Evaluate cross-border compliance",
            force_tier=ModelTier.FRONTIER_CLOUD,
            force_model="gemini-2.0-flash",
        )
        assert res_gem.is_fallback is False
        assert res_gem.content == "Gemini wire-verified resolution"
        assert res_gem.prompt_tokens == 48
        assert res_gem.completion_tokens == 16


def test_openai_reasoning_models_use_max_completion_tokens():
    """Verify o1/o3 reasoning models send max_completion_tokens instead of max_tokens (which OpenAI rejects with 400)."""
    with FrontierWireTwinServer() as twin:
        dispatcher = TaskDispatcher(
            cloud_api_keys={"openai": "sk-live-wire-oai"},
            cloud_base_urls=twin.cloud_base_urls,
        )

        for model_name in ("o1-mini", "o3-mini", "o3"):
            res = dispatcher.execute_task(
                "Prove formal deadlock invariant across multi-agent state machine",
                metadata={"max_tokens": 1024, "temperature": 0.0},
                force_tier=ModelTier.FRONTIER_CLOUD,
                force_model=model_name,
            )
            assert res.is_fallback is False, f"Failed for reasoning model {model_name}: {res.metadata}"
            last_req = twin.state.requests[-1]
            assert "max_completion_tokens" in last_req.body
            assert last_req.body["max_completion_tokens"] == 1024
            assert "max_tokens" not in last_req.body
            assert "temperature" not in last_req.body


def test_tool_call_wire_payloads_openai_null_content_and_anthropic_tool_use():
    """Verify OpenAI 'content: null' tool_calls and Anthropic 'tool_use' blocks parse without crashing."""
    with FrontierWireTwinServer() as twin:
        dispatcher = TaskDispatcher(
            cloud_api_keys={
                "openai": "sk-live-wire-oai",
                "anthropic": "sk-ant-live-wire",
            },
            cloud_base_urls=twin.cloud_base_urls,
        )

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_escrow_release",
                    "description": "Release disputed escrow funds",
                    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
                },
            }
        ]
        res_oai = dispatcher.execute_task(
            "Release escrow for ORD-99182",
            metadata={"tools": openai_tools, "tool_choice": "auto"},
            force_tier=ModelTier.FRONTIER_CLOUD,
            force_model="gpt-4o",
        )
        assert res_oai.is_fallback is False
        assert len(res_oai.tool_calls) == 1
        assert res_oai.tool_calls[0]["name"] == "execute_escrow_release"
        assert res_oai.tool_calls[0]["arguments"] == {"order_id": "ORD-99182", "action": "escrow_refund"}
        assert "execute_escrow_release" in res_oai.content

        # Anthropic pure tool_use response (no accompanying text block)
        twin.state.simulate_tool_only_response = True
        anthropic_tools = [
            {
                "name": "verify_invoice_signature",
                "description": "Verify cryptographic invoice signature",
                "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}},
            }
        ]
        res_ant = dispatcher.execute_task(
            "Verify signature on ORD-99182",
            metadata={"tools": anthropic_tools},
            force_tier=ModelTier.FRONTIER_CLOUD,
            force_model="claude-3-5-sonnet-20241022",
        )
        assert res_ant.is_fallback is False
        assert len(res_ant.tool_calls) == 1
        assert res_ant.tool_calls[0]["name"] == "verify_invoice_signature"
        assert res_ant.tool_calls[0]["arguments"] == {"order_id": "ORD-99182", "verify_signature": True}
        assert "verify_invoice_signature" in res_ant.content


def test_prompt_caching_finops_discount_across_openai_and_anthropic():
    """Verify cached prompt tokens from OpenAI and Anthropic responses apply exact PricingRegistry discounts."""
    with FrontierWireTwinServer() as twin:
        twin.state.simulate_cached_tokens = True
        dispatcher = TaskDispatcher(
            cloud_api_keys={
                "openai": "sk-live-wire-oai",
                "anthropic": "sk-ant-live-wire",
            },
            cloud_base_urls=twin.cloud_base_urls,
        )

        # OpenAI gpt-4o: 1000 total prompt (800 cached @ $1.25/M, 200 uncached @ $2.50/M) + 50 output (@ $10/M)
        # Expected cost = (200 * 2.50 + 800 * 1.25 + 50 * 10.0) / 1e6 = $0.002000 (vs $0.003000 uncached)
        res_oai = dispatcher.execute_task(
            "Evaluate contract with cached system context",
            force_tier=ModelTier.FRONTIER_CLOUD,
            force_model="gpt-4o",
        )
        assert res_oai.prompt_tokens == 1000
        assert res_oai.cached_prompt_tokens == 800
        assert res_oai.completion_tokens == 50
        assert res_oai.cost_usd == pytest.approx(0.002000, abs=1e-7)

        # Anthropic claude-3-5-sonnet-20241022: 200 uncached (@ $3.00/M) + 800 cache_read (@ $0.30/M) + 50 output (@ $15.00/M)
        # Expected cost = (200 * 3.00 + 800 * 0.30 + 50 * 15.0) / 1e6 = $0.001590 (vs $0.003750 uncached)
        res_ant = dispatcher.execute_task(
            "Evaluate contract with Anthropic prompt caching",
            force_tier=ModelTier.FRONTIER_CLOUD,
            force_model="claude-3-5-sonnet-20241022",
        )
        assert res_ant.prompt_tokens == 1000
        assert res_ant.cached_prompt_tokens == 800
        assert res_ant.completion_tokens == 50
        assert res_ant.cost_usd == pytest.approx(0.001590, abs=1e-7)


def test_http_error_handling_and_github_models_token_resolution(monkeypatch):
    """Verify HTTP 401/429/529 error capture and cardless GitHub Models GITHUB_TOKEN support."""
    with FrontierWireTwinServer() as twin:
        # 1. Invalid key -> HTTP 401 captured in fallback metadata
        dispatcher_bad = TaskDispatcher(
            cloud_api_keys={"openai": "invalid-key"},
            cloud_base_urls=twin.cloud_base_urls,
        )
        res_401 = dispatcher_bad.execute_task(
            "Test invalid key", force_tier=ModelTier.FRONTIER_CLOUD, force_model="gpt-4o"
        )
        assert res_401.is_fallback is True
        assert res_401.metadata["http_status"] == 401
        assert "invalid_api_key" in res_401.metadata["error_body"]

        # 2. OpenAI HTTP 429 Rate Limit -> graceful fallback with status 429
        twin.state.force_status_code = 429
        dispatcher_ok = TaskDispatcher(
            cloud_api_keys={"openai": "sk-valid", "anthropic": "sk-ant-valid"},
            cloud_base_urls=twin.cloud_base_urls,
        )
        res_429 = dispatcher_ok.execute_task(
            "Test rate limit", force_tier=ModelTier.FRONTIER_CLOUD, force_model="gpt-4o"
        )
        assert res_429.is_fallback is True
        assert res_429.metadata["http_status"] == 429
        assert "rate_limit_exceeded" in res_429.metadata["error_body"]

        # 3. Anthropic HTTP 529 Overloaded -> graceful fallback with status 529
        twin.state.force_status_code = 529
        res_529 = dispatcher_ok.execute_task(
            "Test overloaded",
            force_tier=ModelTier.FRONTIER_CLOUD,
            force_model="claude-3-5-sonnet-20241022",
        )
        assert res_529.is_fallback is True
        assert res_529.metadata["http_status"] == 529
        assert "overloaded_error" in res_529.metadata["error_body"]

    # 4. Cardless GitHub Models GITHUB_TOKEN resolution
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_cardless_github_models_token")
    gh_dispatcher = TaskDispatcher(
        cloud_base_urls={"openai": "https://models.inference.ai.azure.com"}
    )
    provider, resolved_key = gh_dispatcher._resolve_cloud_provider_and_key("gpt-4o")
    assert provider == "openai"
    assert resolved_key == "ghp_cardless_github_models_token"

