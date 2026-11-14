"""Runtime tracer and context manager for live agent trajectory telemetry capture."""

from __future__ import annotations

import contextvars
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

from aegis.core.models import (
    MCPStateTransition,
    ModelTier,
    Span,
    SpanType,
    ToolNode,
    TrajectoryDAG,
)
from aegis.core.pricing import PricingRegistry
from aegis.finops.circuit_breaker import CircuitBreaker


class SpanContext:
    """Active execution span context manager (supports sync and async contexts)."""

    def __init__(
        self,
        tracer: "AegisTracer",
        name: str,
        span_type: SpanType,
        model_name: Optional[str] = None,
        model_tier: Optional[ModelTier] = None,
        parent_span_id: Optional[str] = None,
    ) -> None:
        self.tracer = tracer
        self.name = name
        self.span_type = span_type
        self.model_name = model_name
        self.model_tier = model_tier
        self._explicit_parent_span_id = parent_span_id
        self.parent_span_id: Optional[str] = parent_span_id
        self.span_id = str(uuid.uuid4())
        self.start_time_ms = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_prompt_tokens = 0
        self.ttft_ms: Optional[float] = None
        self.attributes: Dict[str, Any] = {}
        self.error: Optional[str] = None
        self._ctx_token: Optional[contextvars.Token[Tuple[str, ...]]] = None

    def record_tokens(
        self, prompt_tokens: int, completion_tokens: int, cached_prompt_tokens: int = 0
    ) -> None:
        """Update token counts for this span."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cached_prompt_tokens += cached_prompt_tokens
        if self.cached_prompt_tokens > 0:
            self.attributes["cached_tokens"] = self.cached_prompt_tokens

    def set_attribute(self, key: str, value: Any) -> None:
        """Add custom metadata attribute."""
        self.attributes[key] = value

    def set_error(self, err: str) -> None:
        """Record error string."""
        self.error = err

    def __enter__(self) -> "SpanContext":
        self.start_time_ms = time.time() * 1000.0
        stack = self.tracer._span_stack_var.get()
        if self._explicit_parent_span_id is None:
            self.parent_span_id = stack[-1] if stack else None
        self._ctx_token = self.tracer._span_stack_var.set((*stack, self.span_id))
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._ctx_token is not None:
            self.tracer._span_stack_var.reset(self._ctx_token)
            self._ctx_token = None

        end_time_ms = time.time() * 1000.0
        if exc_val is not None:
            self.error = str(exc_val)

        cost_usd = 0.0
        if self.tracer.pricing_registry and self.model_name:
            pricing = self.tracer.pricing_registry.get(self.model_name)
            if pricing:
                cached = self.cached_prompt_tokens or int(self.attributes.get("cached_tokens", 0))
                cost_usd = pricing.compute_cost(
                    prompt_tokens=self.prompt_tokens,
                    completion_tokens=self.completion_tokens,
                    duration_ms=max(0.0, end_time_ms - self.start_time_ms),
                    cached_prompt_tokens=cached,
                )

        span = Span(
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            name=self.name,
            span_type=self.span_type,
            model_tier=self.model_tier,
            model_name=self.model_name,
            start_time_ms=self.start_time_ms,
            end_time_ms=end_time_ms,
            ttft_ms=self.ttft_ms,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            cost_usd=cost_usd,
            attributes=self.attributes,
            error=self.error,
        )
        self.tracer.trajectory.add_span(span)
        if self.tracer.circuit_breaker:
            self.tracer.circuit_breaker.guard(self.tracer.trajectory)

    async def __aenter__(self) -> "SpanContext":
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


class AegisTracer:
    """High-level telemetry tracer capturing DAG transitions and checking live governance SLAs."""

    def __init__(
        self,
        scenario_id: str = "default_scenario",
        trajectory_id: Optional[str] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        pricing_registry: Optional[PricingRegistry] = None,
    ) -> None:
        self.trajectory = TrajectoryDAG(
            trajectory_id=trajectory_id or str(uuid.uuid4()),
            scenario_id=scenario_id,
            resolved=False,
        )
        self.circuit_breaker = circuit_breaker
        self.pricing_registry = pricing_registry or PricingRegistry()
        self._span_stack_var: contextvars.ContextVar[Tuple[str, ...]] = contextvars.ContextVar(
            f"aegis_span_stack_{self.trajectory.trajectory_id}", default=()
        )

    @property
    def active_span_id(self) -> Optional[str]:
        """Return the currently active span ID in the execution context, if any."""
        stack = self._span_stack_var.get()
        return stack[-1] if stack else None

    @property
    def _active_span_id(self) -> Optional[str]:
        """Backwards-compatible alias for active_span_id."""
        return self.active_span_id

    def __enter__(self) -> "AegisTracer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_val is not None:
            self.trajectory.metadata["error"] = str(exc_val)

    async def __aenter__(self) -> "AegisTracer":
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)

    def span(
        self,
        name: str,
        span_type: SpanType = SpanType.INTERNAL_REASONING,
        model_name: Optional[str] = None,
        model_tier: Optional[ModelTier] = None,
        parent_span_id: Optional[str] = None,
    ) -> SpanContext:
        """Create a managed execution span context manager."""
        return SpanContext(
            tracer=self,
            name=name,
            span_type=span_type,
            model_name=model_name,
            model_tier=model_tier,
            parent_span_id=parent_span_id,
        )

    def record_tool_call(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        output: Any = None,
        duration_ms: float = 0.0,
        depends_on: Optional[List[str]] = None,
        status: str = "success",
        node_id: Optional[str] = None,
    ) -> ToolNode:
        """Record an executed tool invocation and append causal dependencies in the DAG."""
        nid = node_id or str(uuid.uuid4())
        node = ToolNode(
            node_id=nid,
            tool_name=tool_name,
            arguments=arguments or {},
            output=output,
            status=status,
            duration_ms=duration_ms,
        )
        self.trajectory.add_tool_node(node)

        if depends_on:
            for parent_id in depends_on:
                self.trajectory.add_dependency(parent_id, nid)

        # Enforce live circuit breaker check on each tool registration
        if self.circuit_breaker:
            self.circuit_breaker.guard(self.trajectory)

        return node

    def record_llm_call(
        self,
        name: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: float = 0.0,
        model_tier: Optional[ModelTier] = None,
        ttft_ms: Optional[float] = None,
        attributes: Optional[Dict[str, Any]] = None,
        parent_span_id: Optional[str] = None,
        cached_prompt_tokens: int = 0,
    ) -> Span:
        """Record a single LLM inference call with automatic pricing resolution."""
        attrs = dict(attributes) if attributes else {}
        cached = cached_prompt_tokens or int(attrs.get("cached_tokens", 0))
        if cached > 0:
            attrs["cached_tokens"] = cached

        pricing = self.pricing_registry.get(model_name)
        tier = model_tier or (pricing.tier if pricing else ModelTier.FRONTIER_CLOUD)
        cost_usd = 0.0
        if pricing:
            cost_usd = pricing.compute_cost(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
                cached_prompt_tokens=cached,
            )

        now_ms = time.time() * 1000.0
        start_ms = now_ms - duration_ms if duration_ms > 0 else now_ms

        span = Span(
            name=name,
            parent_span_id=parent_span_id if parent_span_id is not None else self.active_span_id,
            span_type=SpanType.LLM_INFERENCE,
            model_tier=tier,
            model_name=model_name,
            start_time_ms=start_ms,
            end_time_ms=now_ms,
            ttft_ms=ttft_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            attributes=attrs,
        )
        self.trajectory.add_span(span)

        if self.circuit_breaker:
            self.circuit_breaker.guard(self.trajectory)

        return span

    def record_mcp_transition(
        self,
        server_name: str,
        is_stateful: bool,
        state_payload_bytes: int = 0,
        context_token_overhead: int = 0,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MCPStateTransition:
        """Record a Model Context Protocol interaction state transition."""
        trans = MCPStateTransition(
            server_name=server_name,
            is_stateful=is_stateful,
            state_payload_bytes=state_payload_bytes,
            context_token_overhead=context_token_overhead,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        self.trajectory.mcp_transitions.append(trans)
        return trans

    def mark_resolved(self, resolved: bool = True) -> None:
        """Set final outcome resolution status for the trajectory."""
        self.trajectory.resolved = resolved

    def get_trajectory(self) -> TrajectoryDAG:
        """Return the finalized TrajectoryDAG."""
        return self.trajectory
