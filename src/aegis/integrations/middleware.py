"""Agent framework middleware and auto-instrumentation hooks for LangGraph, OpenAI Agents SDK, and MCP."""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import time
from typing import Any, Callable, Dict, List, Optional, Union
import uuid

from aegis.core.models import ModelTier, SpanType, ToolNode, TrajectoryDAG
from aegis.core.tracer import AegisTracer
from aegis.finops.circuit_breaker import (
    CircuitBreakerConfig,
    StreamingCircuitBreaker,
)
from aegis.finops.router import HybridComplexityRouter, RoutingDecision
from aegis.mcp_profiler.state_store import BaseMCPStateStore, MemoryMCPStateStore


def governed_tool(
    tracer: AegisTracer,
    streaming_breaker: Optional[StreamingCircuitBreaker] = None,
    tool_name: Optional[str] = None,
    auto_chain_dependencies: bool = True,
) -> Callable:
    """Decorator that auto-instruments a sync or async tool function with AegisTracer and StreamingCircuitBreaker."""

    def decorator(func: Callable) -> Callable:
        resolved_name = tool_name or func.__name__
        sig = inspect.signature(func)
        is_async = inspect.iscoroutinefunction(func)

        def _extract_args(args: tuple, kwargs: dict) -> Dict[str, Any]:
            try:
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                return {
                    k: v
                    for k, v in bound.arguments.items()
                    if k not in ("self", "cls")
                }
            except Exception:
                return dict(kwargs)

        def _get_last_node_ids() -> Optional[List[str]]:
            if not auto_chain_dependencies or not tracer.trajectory.tool_nodes:
                return None
            last_id = list(tracer.trajectory.tool_nodes.keys())[-1]
            return [last_id]

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            arg_dict = _extract_args(args, kwargs)
            if streaming_breaker is not None:
                streaming_breaker.before_tool_call(resolved_name, arg_dict)

            deps = _get_last_node_ids()
            t0 = time.time()
            status = "success"
            output: Any = None
            try:
                async with tracer.span(resolved_name, SpanType.TOOL_CALL) as span_ctx:
                    span_ctx.set_attribute("tool_name", resolved_name)
                    output = await func(*args, **kwargs)
                    return output
            except Exception as exc:
                status = "error"
                output = {"error": str(exc)}
                raise
            finally:
                duration_ms = (time.time() - t0) * 1000.0
                tracer.record_tool_call(
                    tool_name=resolved_name,
                    arguments=arg_dict,
                    output=output,
                    duration_ms=round(duration_ms, 3),
                    depends_on=deps,
                    status=status,
                )
                if streaming_breaker is not None:
                    streaming_breaker.after_tool_call(resolved_name)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            arg_dict = _extract_args(args, kwargs)
            if streaming_breaker is not None:
                streaming_breaker.before_tool_call(resolved_name, arg_dict)

            deps = _get_last_node_ids()
            t0 = time.time()
            status = "success"
            output: Any = None
            try:
                with tracer.span(resolved_name, SpanType.TOOL_CALL) as span_ctx:
                    span_ctx.set_attribute("tool_name", resolved_name)
                    output = func(*args, **kwargs)
                    return output
            except Exception as exc:
                status = "error"
                output = {"error": str(exc)}
                raise
            finally:
                duration_ms = (time.time() - t0) * 1000.0
                tracer.record_tool_call(
                    tool_name=resolved_name,
                    arguments=arg_dict,
                    output=output,
                    duration_ms=round(duration_ms, 3),
                    depends_on=deps,
                    status=status,
                )
                if streaming_breaker is not None:
                    streaming_breaker.after_tool_call(resolved_name)

        return async_wrapper if is_async else sync_wrapper

    return decorator


class LangGraphGovernanceCallback:
    """Callback handler and node wrapper for LangChain / LangGraph agent workflows.

    Compatible with LangChain's BaseCallbackHandler event signatures without requiring
    langchain-core as a hard installation dependency.
    """

    def __init__(
        self,
        tracer: Optional[AegisTracer] = None,
        streaming_breaker: Optional[StreamingCircuitBreaker] = None,
        router: Optional[HybridComplexityRouter] = None,
        auto_chain_tools: bool = True,
    ) -> None:
        self.tracer = tracer or AegisTracer(scenario_id="langgraph_workflow")
        self.streaming_breaker = streaming_breaker or StreamingCircuitBreaker(
            config=self.tracer.circuit_breaker.config
            if self.tracer.circuit_breaker
            else CircuitBreakerConfig()
        )
        self.router = router or HybridComplexityRouter()
        self.auto_chain_tools = auto_chain_tools
        self._inflight_runs: Dict[str, Dict[str, Any]] = {}
        self._last_tool_node_id: Optional[str] = None

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Optional[Union[str, uuid.UUID]] = None,
        parent_run_id: Optional[Union[str, uuid.UUID]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        rid = str(run_id or uuid.uuid4())
        prompt_text = "\n".join(prompts) if prompts else ""
        model_name = (
            (metadata or {}).get("ls_model_name")
            or kwargs.get("invocation_params", {}).get("model_name")
            or serialized.get("kwargs", {}).get("model_name")
            or serialized.get("name")
            or self.router.default_cloud_model
        )
        decision = self.router.route(prompt_text, metadata=metadata)
        self._inflight_runs[rid] = {
            "type": "llm",
            "start_ms": time.time() * 1000.0,
            "prompt_text": prompt_text,
            "model_name": str(model_name),
            "routing_decision": decision,
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
        }
        return rid

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Optional[Union[str, uuid.UUID]] = None,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id) if run_id else ""
        info = self._inflight_runs.pop(rid, None)
        now_ms = time.time() * 1000.0
        start_ms = info["start_ms"] if info else now_ms
        duration_ms = max(0.0, now_ms - start_ms)
        model_name = info["model_name"] if info else self.router.default_cloud_model
        decision: Optional[RoutingDecision] = info.get("routing_decision") if info else None

        prompt_tokens = 0
        completion_tokens = 0
        if isinstance(response, dict):
            usage = response.get("token_usage") or response.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
        elif hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
            usage = response.llm_output.get("token_usage", {})
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))

        if prompt_tokens == 0 and info and info.get("prompt_text"):
            prompt_tokens = max(1, len(info["prompt_text"].split()))
        if completion_tokens == 0:
            completion_tokens = 16

        span = self.tracer.record_llm_call(
            name=f"langgraph_llm:{model_name}",
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=round(duration_ms, 3),
            model_tier=decision.selected_tier if decision else None,
            attributes={
                "recommended_tier": decision.selected_tier.value if decision else "unknown",
                "recommended_model": decision.recommended_model if decision else model_name,
            },
        )
        self.streaming_breaker.cumulative_tokens += span.total_tokens
        self.streaming_breaker.cumulative_cost_usd += span.cost_usd

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: Union[str, Dict[str, Any]],
        *,
        run_id: Optional[Union[str, uuid.UUID]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        rid = str(run_id or uuid.uuid4())
        tool_name = serialized.get("name", "langgraph_tool")
        if inputs and isinstance(inputs, dict):
            args_dict = inputs
        elif isinstance(input_str, dict):
            args_dict = input_str
        else:
            try:
                parsed = json.loads(input_str)
                args_dict = parsed if isinstance(parsed, dict) else {"input": input_str}
            except Exception:
                args_dict = {"input": str(input_str)}

        self.streaming_breaker.before_tool_call(tool_name, args_dict)
        self._inflight_runs[rid] = {
            "type": "tool",
            "tool_name": tool_name,
            "arguments": args_dict,
            "start_ms": time.time() * 1000.0,
        }
        return rid

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Optional[Union[str, uuid.UUID]] = None,
        **kwargs: Any,
    ) -> ToolNode:
        rid = str(run_id) if run_id else ""
        info = self._inflight_runs.pop(rid, None) or {
            "tool_name": kwargs.get("name", "langgraph_tool"),
            "arguments": {},
            "start_ms": time.time() * 1000.0,
        }
        duration_ms = max(0.0, (time.time() * 1000.0) - info["start_ms"])
        deps = (
            [self._last_tool_node_id]
            if self.auto_chain_tools and self._last_tool_node_id
            else None
        )
        node = self.tracer.record_tool_call(
            tool_name=info["tool_name"],
            arguments=info["arguments"],
            output=output,
            duration_ms=round(duration_ms, 3),
            depends_on=deps,
            status="success",
        )
        self._last_tool_node_id = node.node_id
        self.streaming_breaker.after_tool_call(info["tool_name"])
        return node

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[Union[str, uuid.UUID]] = None,
        **kwargs: Any,
    ) -> ToolNode:
        rid = str(run_id) if run_id else ""
        info = self._inflight_runs.pop(rid, None) or {
            "tool_name": kwargs.get("name", "langgraph_tool"),
            "arguments": {},
            "start_ms": time.time() * 1000.0,
        }
        duration_ms = max(0.0, (time.time() * 1000.0) - info["start_ms"])
        deps = (
            [self._last_tool_node_id]
            if self.auto_chain_tools and self._last_tool_node_id
            else None
        )
        node = self.tracer.record_tool_call(
            tool_name=info["tool_name"],
            arguments=info["arguments"],
            output={"error": str(error)},
            duration_ms=round(duration_ms, 3),
            depends_on=deps,
            status="error",
        )
        self._last_tool_node_id = node.node_id
        self.streaming_breaker.after_tool_call(info["tool_name"])
        return node

    def wrap_node(self, node_name: Optional[str] = None) -> Callable:
        """Decorator for LangGraph state-graph node functions to capture span hierarchy."""

        def decorator(func: Callable) -> Callable:
            name = node_name or func.__name__
            is_async = inspect.iscoroutinefunction(func)

            @functools.wraps(func)
            async def async_node(state: Any, *args: Any, **kwargs: Any) -> Any:
                async with self.tracer.span(f"node:{name}", SpanType.INTERNAL_REASONING) as s:
                    if isinstance(state, dict):
                        s.set_attribute("state_keys", sorted(list(state.keys())))
                    return await func(state, *args, **kwargs)

            @functools.wraps(func)
            def sync_node(state: Any, *args: Any, **kwargs: Any) -> Any:
                with self.tracer.span(f"node:{name}", SpanType.INTERNAL_REASONING) as s:
                    if isinstance(state, dict):
                        s.set_attribute("state_keys", sorted(list(state.keys())))
                    return func(state, *args, **kwargs)

            return async_node if is_async else sync_node

        return decorator


class OpenAIAgentsGovernanceHooks:
    """Lifecycle hooks compatible with the OpenAI Agents SDK (`RunHooks` / `AgentHooks`)."""

    def __init__(
        self,
        tracer: Optional[AegisTracer] = None,
        streaming_breaker: Optional[StreamingCircuitBreaker] = None,
        router: Optional[HybridComplexityRouter] = None,
    ) -> None:
        self.tracer = tracer or AegisTracer(scenario_id="openai_agents_workflow")
        self.streaming_breaker = streaming_breaker or StreamingCircuitBreaker(
            config=self.tracer.circuit_breaker.config
            if self.tracer.circuit_breaker
            else CircuitBreakerConfig()
        )
        self.router = router or HybridComplexityRouter()
        self._agent_span_contexts: Dict[str, Any] = {}
        self._tool_starts: Dict[str, float] = {}
        self._last_tool_node_id: Optional[str] = None

    def _agent_name(self, agent: Any) -> str:
        if isinstance(agent, str):
            return agent
        return getattr(agent, "name", None) or agent.__class__.__name__

    def _tool_name(self, tool: Any) -> str:
        if isinstance(tool, str):
            return tool
        return getattr(tool, "name", None) or getattr(tool, "__name__", "agent_tool")

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        name = self._agent_name(agent)
        span_ctx = self.tracer.span(f"agent:{name}", SpanType.INTERNAL_REASONING)
        await span_ctx.__aenter__()
        span_ctx.set_attribute("agent_name", name)
        self._agent_span_contexts[name] = span_ctx

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        name = self._agent_name(agent)
        span_ctx = self._agent_span_contexts.pop(name, None)
        if span_ctx is not None:
            span_ctx.set_attribute("output_preview", str(output)[:120])
            await span_ctx.__aexit__(None, None, None)

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        src = self._agent_name(from_agent)
        dst = self._agent_name(to_agent)
        async with self.tracer.span(f"handoff:{src}->{dst}", SpanType.ROUTER_DECISION) as s:
            s.set_attribute("from_agent", src)
            s.set_attribute("to_agent", dst)

    async def on_tool_start(
        self,
        context: Any,
        agent: Any,
        tool: Any,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> None:
        t_name = self._tool_name(tool)
        self.streaming_breaker.before_tool_call(t_name, arguments or {})
        self._tool_starts[t_name] = time.time() * 1000.0

    async def on_tool_end(
        self,
        context: Any,
        agent: Any,
        tool: Any,
        result: Any,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> ToolNode:
        t_name = self._tool_name(tool)
        start_ms = self._tool_starts.pop(t_name, time.time() * 1000.0)
        duration_ms = max(0.0, (time.time() * 1000.0) - start_ms)
        deps = [self._last_tool_node_id] if self._last_tool_node_id else None
        node = self.tracer.record_tool_call(
            tool_name=t_name,
            arguments=arguments or {},
            output=result,
            duration_ms=round(duration_ms, 3),
            depends_on=deps,
        )
        self._last_tool_node_id = node.node_id
        self.streaming_breaker.after_tool_call(t_name)
        return node


class GovernedMCPSessionProxy:
    """Proxy middleware wrapping an MCP ClientSession with stateful pointer persistence and Aegis telemetry."""

    def __init__(
        self,
        session: Any,
        server_name: str = "mcp_server",
        session_id: Optional[str] = None,
        tracer: Optional[AegisTracer] = None,
        streaming_breaker: Optional[StreamingCircuitBreaker] = None,
        state_store: Optional[BaseMCPStateStore] = None,
        is_stateful: bool = True,
        pointer_token_overhead: int = 32,
    ) -> None:
        self.session = session
        self.server_name = server_name
        self.session_id = session_id or f"mcp_sess_{uuid.uuid4().hex[:8]}"
        self.tracer = tracer or AegisTracer(scenario_id=f"mcp_{server_name}")
        self.streaming_breaker = streaming_breaker or StreamingCircuitBreaker(
            config=self.tracer.circuit_breaker.config
            if self.tracer.circuit_breaker
            else CircuitBreakerConfig()
        )
        self.state_store = state_store or MemoryMCPStateStore()
        self.is_stateful = is_stateful
        self.pointer_token_overhead = pointer_token_overhead
        self.latest_pointer: Optional[str] = None
        self._last_tool_node_id: Optional[str] = None
        self._cumulative_stateless_bytes = 0

    async def call_tool(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Intercept MCP `call_tool`, guard with circuit breaker, persist state delta, and record DAG telemetry."""
        args = arguments or {}
        self.streaming_breaker.before_tool_call(name, args)

        t0 = time.time()
        async with self.tracer.span(f"mcp.tool:{name}", SpanType.MCP_TOOL_CALL) as s:
            s.set_attribute("mcp_server", self.server_name)
            raw_call = getattr(self.session, "call_tool")
            if inspect.iscoroutinefunction(raw_call):
                result = await raw_call(name, args)
            else:
                result = raw_call(name, args)

        duration_ms = (time.time() - t0) * 1000.0
        deps = [self._last_tool_node_id] if self._last_tool_node_id else None
        node = self.tracer.record_tool_call(
            tool_name=name,
            arguments=args,
            output=result,
            duration_ms=round(duration_ms, 3),
            depends_on=deps,
        )
        self._last_tool_node_id = node.node_id
        self.streaming_breaker.after_tool_call(name)

        # Record MCP state transition and persist delta if stateful
        delta_payload = {"last_tool": name, "arguments": args, "result": str(result)}
        serialized_bytes = len(json.dumps(delta_payload, default=str).encode("utf-8"))
        self._cumulative_stateless_bytes += serialized_bytes

        if self.is_stateful:
            pointer, _, metrics = self.state_store.update_delta(self.session_id, delta_payload)
            self.latest_pointer = pointer
            token_overhead = self.pointer_token_overhead
            latency_ms = metrics.write_latency_ms
        else:
            token_overhead = max(1, self._cumulative_stateless_bytes // 4)
            latency_ms = round(duration_ms, 3)

        self.tracer.record_mcp_transition(
            server_name=self.server_name,
            is_stateful=self.is_stateful,
            state_payload_bytes=serialized_bytes,
            context_token_overhead=token_overhead,
            latency_ms=latency_ms,
            metadata={"pointer": self.latest_pointer} if self.latest_pointer else {},
        )
        return result

    async def read_resource(self, uri: str) -> Any:
        """Intercept MCP `read_resource` and record span + MCP transition telemetry."""
        t0 = time.time()
        async with self.tracer.span(f"mcp.resource:{uri}", SpanType.MCP_RESOURCE_READ) as s:
            s.set_attribute("mcp_server", self.server_name)
            s.set_attribute("resource_uri", uri)
            raw_read = getattr(self.session, "read_resource")
            if inspect.iscoroutinefunction(raw_read):
                result = await raw_read(uri)
            else:
                result = raw_read(uri)

        duration_ms = (time.time() - t0) * 1000.0
        payload_bytes = len(str(result).encode("utf-8"))
        self._cumulative_stateless_bytes += payload_bytes

        if self.is_stateful:
            pointer, _, metrics = self.state_store.update_delta(
                self.session_id, {"resource_uri": uri, "content_preview": str(result)[:256]}
            )
            self.latest_pointer = pointer
            token_overhead = self.pointer_token_overhead
            latency_ms = metrics.write_latency_ms
        else:
            token_overhead = max(1, self._cumulative_stateless_bytes // 4)
            latency_ms = round(duration_ms, 3)

        self.tracer.record_mcp_transition(
            server_name=self.server_name,
            is_stateful=self.is_stateful,
            state_payload_bytes=payload_bytes,
            context_token_overhead=token_overhead,
            latency_ms=latency_ms,
            metadata={"uri": uri, "pointer": self.latest_pointer},
        )
        return result

