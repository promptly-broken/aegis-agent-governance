"""Model Context Protocol (MCP) telemetry profiler: Token Amplification Factor (TAF) auditor."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from pydantic import BaseModel, ConfigDict, Field

from aegis.core.models import TrajectoryDAG


class MCPTurnTelemetry(BaseModel):
    """Per-turn context transmission telemetry comparing stateless vs stateful protocol modes."""

    model_config = ConfigDict(extra="forbid")

    turn_index: int = Field(..., ge=0)
    stateless_tokens: int = Field(..., ge=0)
    stateful_tokens: int = Field(..., ge=0)
    payload_bytes: int = Field(..., ge=0)
    serialization_overhead_ms: float = Field(default=0.0, ge=0.0)


class MCPAmplificationReport(BaseModel):
    """Aggregate assessment of token amplification and state serialization overhead."""

    model_config = ConfigDict(extra="forbid")

    total_stateless_tokens: int = Field(..., ge=0)
    total_stateful_tokens: int = Field(..., ge=0)
    token_amplification_factor: float = Field(..., ge=0.0)
    tokens_saved: int = Field(..., ge=0)
    percentage_token_reduction: float = Field(..., ge=0.0, le=100.0)
    turns: List[MCPTurnTelemetry] = Field(default_factory=list)


class MCPAmplificationAuditor:
    """Harness evaluating Token Amplification Factor (TAF) across multi-turn agent interactions."""

    def __init__(self, avg_bytes_per_token: float = 4.0, pointer_overhead_tokens: int = 32) -> None:
        self.avg_bytes_per_token = avg_bytes_per_token
        self.pointer_overhead_tokens = pointer_overhead_tokens

    def _estimate_tokens(self, payload: Union[str, Dict[str, Any]]) -> int:
        if isinstance(payload, str):
            text = payload
        else:
            text = json.dumps(payload)
        byte_len = len(text.encode("utf-8"))
        return max(1, int(byte_len / self.avg_bytes_per_token))

    def simulate_trajectory(
        self, turn_payloads: List[Union[str, Dict[str, Any]]], system_prompt_tokens: int = 500
    ) -> MCPAmplificationReport:
        """Simulate multi-turn interaction and calculate Token Amplification Factor (TAF).

        In stateless mode: Each turn re-transmits all historical payloads plus current payload.
        In stateful mode: Each turn transmits only system prompt + session pointer + current payload.
        """
        if not turn_payloads:
            raise ValueError("Must provide at least one turn payload to audit.")

        turns_telemetry: List[MCPTurnTelemetry] = []
        cumulative_history_tokens = 0
        total_stateless = 0
        total_stateful = 0

        for turn_idx, payload in enumerate(turn_payloads):
            delta_tokens = self._estimate_tokens(payload)
            payload_str = payload if isinstance(payload, str) else json.dumps(payload)
            payload_bytes = len(payload_str.encode("utf-8"))

            # Stateless transmission: system prompt + all past turns + current turn
            stateless_current = system_prompt_tokens + cumulative_history_tokens + delta_tokens
            cumulative_history_tokens += delta_tokens

            # Stateful transmission: system prompt + session pointer (turn > 0) + current delta
            pointer_overhead = self.pointer_overhead_tokens if turn_idx > 0 else 0
            stateful_current = system_prompt_tokens + pointer_overhead + delta_tokens

            total_stateless += stateless_current
            total_stateful += stateful_current

            turns_telemetry.append(
                MCPTurnTelemetry(
                    turn_index=turn_idx,
                    stateless_tokens=stateless_current,
                    stateful_tokens=stateful_current,
                    payload_bytes=payload_bytes,
                    serialization_overhead_ms=round(payload_bytes / 50_000.0, 3),  # Approx 50MB/s JSON parse
                )
            )

        taf = total_stateless / max(1, total_stateful)
        saved = max(0, total_stateless - total_stateful)
        pct_reduction = (saved / max(1, total_stateless)) * 100.0

        return MCPAmplificationReport(
            total_stateless_tokens=total_stateless,
            total_stateful_tokens=total_stateful,
            token_amplification_factor=round(taf, 3),
            tokens_saved=saved,
            percentage_token_reduction=round(pct_reduction, 2),
            turns=turns_telemetry,
        )

    def audit_trajectory_dag(self, trajectory: TrajectoryDAG) -> MCPAmplificationReport:
        """Extract MCP transitions from a TrajectoryDAG and compute protocol amplification."""
        if not trajectory.mcp_transitions:
            # Fallback using tool nodes as payloads
            payloads = [node.arguments for node in trajectory.tool_nodes.values()]
            if not payloads:
                payloads = [{"action": "initialize"}]
        else:
            payloads = [
                {"server": t.server_name, "state_bytes": t.state_payload_bytes, "meta": t.metadata}
                for t in trajectory.mcp_transitions
            ]
        return self.simulate_trajectory(payloads)
