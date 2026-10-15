"""Unit tests for MCP Profiler and Token Amplification Factor auditing."""

import pytest

from aegis.mcp_profiler.amplification import MCPAmplificationAuditor


def test_mcp_profiler_simulation():
    auditor = MCPAmplificationAuditor(pointer_overhead_tokens=32)
    payloads = [
        {"turn": 1, "data": "initial user inquiry about order #12345"},
        {"turn": 2, "data": "agent tool response with invoice details and items"},
        {"turn": 3, "data": "user follow-up question regarding tracking details"},
        {"turn": 4, "data": "shipping carrier response showing delay in transit"},
        {"turn": 5, "data": "resolution: dispute refund issued to customer"},
    ]

    report = auditor.simulate_trajectory(payloads, system_prompt_tokens=500)
    assert report.total_stateless_tokens > 0
    assert report.total_stateful_tokens > 0
    assert report.token_amplification_factor >= 1.0
    assert len(report.turns) == 5
    assert report.tokens_saved > 0
    assert report.percentage_token_reduction > 0.0


def test_mcp_profiler_empty_raises():
    auditor = MCPAmplificationAuditor()
    with pytest.raises(ValueError, match="at least one turn"):
        auditor.simulate_trajectory([])
