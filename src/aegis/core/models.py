"""Core telemetry and trajectory graph models using strict Pydantic v2 and NetworkX."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SpanType(str, Enum):
    """Categorical classification of execution spans."""

    LLM_INFERENCE = "llm_inference"
    TOOL_CALL = "tool_call"
    MCP_RESOURCE_READ = "mcp_resource_read"
    MCP_TOOL_CALL = "mcp_tool_call"
    MCP_PROMPT_GET = "mcp_prompt_get"
    ROUTER_DECISION = "router_decision"
    INTERNAL_REASONING = "internal_reasoning"


class ModelTier(str, Enum):
    """Model execution tier distinguishing edge Apple Silicon from frontier cloud."""

    LOCAL_MLX = "local_mlx"
    LOCAL_OLLAMA = "local_ollama"
    FRONTIER_CLOUD = "frontier_cloud"


class Span(BaseModel):
    """Execution span representing an LLM inference, tool invocation, or MCP interaction."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    span_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    name: str
    span_type: SpanType
    model_tier: Optional[ModelTier] = None
    model_name: Optional[str] = None
    start_time_ms: float
    end_time_ms: float
    ttft_ms: Optional[float] = None
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Calculate total span duration in milliseconds."""
        return max(0.0, self.end_time_ms - self.start_time_ms)

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed in this span."""
        return self.prompt_tokens + self.completion_tokens


class ToolNode(BaseModel):
    """Discrete node in an agent tool execution graph."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    schema_keys: List[str] = Field(default_factory=list)
    output: Optional[Any] = None
    status: str = Field(default="success")
    duration_ms: float = Field(default=0.0, ge=0.0)
    span_id: Optional[str] = None

    @field_validator("schema_keys", mode="before")
    @classmethod
    def sort_schema_keys(cls, v: Any) -> List[str]:
        """Ensure schema keys are always a sorted list of strings."""
        if v is None:
            return []
        if isinstance(v, (list, set, tuple)):
            return sorted(str(k) for k in v)
        return [str(v)]

    @model_validator(mode="before")
    @classmethod
    def populate_schema_keys(cls, data: Any) -> Any:
        """Automatically derive schema keys from arguments if omitted."""
        if isinstance(data, dict):
            args = data.get("arguments", {})
            keys = data.get("schema_keys")
            if (keys is None or len(keys) == 0) and isinstance(args, dict) and args:
                data["schema_keys"] = sorted(list(args.keys()))
        return data


class MCPStateTransition(BaseModel):
    """Telemetry capture for a Model Context Protocol state change."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    transition_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    server_name: str
    is_stateful: bool
    state_payload_bytes: int = Field(default=0, ge=0)
    context_token_overhead: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrajectoryDAG(BaseModel):
    """Directed Acyclic Graph (DAG) representing an agent trajectory."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    trajectory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scenario_id: str = "default_scenario"
    resolved: bool = False
    spans: List[Span] = Field(default_factory=list)
    tool_nodes: Dict[str, ToolNode] = Field(default_factory=dict)
    edges: List[Tuple[str, str]] = Field(default_factory=list)
    mcp_transitions: List[MCPStateTransition] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_tool_node(self, node: ToolNode) -> None:
        """Add a tool node to the trajectory."""
        self.tool_nodes[node.node_id] = node

    def add_dependency(self, from_node_id: str, to_node_id: str) -> None:
        """Add a directed edge representing causal or execution dependency (u -> v)."""
        if from_node_id not in self.tool_nodes:
            raise KeyError(f"Source node '{from_node_id}' not found in tool_nodes.")
        if to_node_id not in self.tool_nodes:
            raise KeyError(f"Target node '{to_node_id}' not found in tool_nodes.")
        edge = (from_node_id, to_node_id)
        if edge not in self.edges:
            self.edges.append(edge)

    def add_span(self, span: Span) -> None:
        """Append an execution span."""
        self.spans.append(span)

    def to_networkx(self) -> nx.DiGraph:
        """Convert the trajectory tool nodes and causal edges into a NetworkX DiGraph."""
        graph = nx.DiGraph(
            trajectory_id=self.trajectory_id,
            scenario_id=self.scenario_id,
            resolved=self.resolved,
        )

        for node_id, node in self.tool_nodes.items():
            graph.add_node(
                node_id,
                tool_name=node.tool_name,
                arguments=node.arguments,
                schema_keys=node.schema_keys,
                status=node.status,
                duration_ms=node.duration_ms,
                span_id=node.span_id,
            )

        for u, v in self.edges:
            graph.add_edge(u, v)

        return graph

    @classmethod
    def from_networkx(
        cls,
        graph: nx.DiGraph,
        trajectory_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        resolved: Optional[bool] = None,
    ) -> "TrajectoryDAG":
        """Reconstruct a TrajectoryDAG from a NetworkX DiGraph."""
        t_id = trajectory_id or graph.graph.get("trajectory_id", str(uuid.uuid4()))
        s_id = scenario_id or graph.graph.get("scenario_id", "default_scenario")
        res = resolved if resolved is not None else graph.graph.get("resolved", False)

        nodes: Dict[str, ToolNode] = {}
        for n, data in graph.nodes(data=True):
            tool_name = data.get("tool_name", "unknown_tool")
            arguments = data.get("arguments", {})
            schema_keys = data.get("schema_keys", [])
            status = data.get("status", "success")
            duration_ms = data.get("duration_ms", 0.0)
            span_id = data.get("span_id", None)
            nodes[str(n)] = ToolNode(
                node_id=str(n),
                tool_name=tool_name,
                arguments=arguments,
                schema_keys=schema_keys,
                status=status,
                duration_ms=duration_ms,
                span_id=span_id,
            )

        edges: List[Tuple[str, str]] = [(str(u), str(v)) for u, v in graph.edges()]

        return cls(
            trajectory_id=t_id,
            scenario_id=s_id,
            resolved=res,
            tool_nodes=nodes,
            edges=edges,
        )

    def has_cycle(self) -> bool:
        """Check if the trajectory graph contains any cycles."""
        g = self.to_networkx()
        return not nx.is_directed_acyclic_graph(g)

    def find_cycles(self) -> List[List[str]]:
        """Find and return all simple cycles within the trajectory graph."""
        g = self.to_networkx()
        return list(nx.simple_cycles(g))

    def topological_sort(self) -> List[str]:
        """Return topological order of tool nodes. Raises error if graph has cycles."""
        g = self.to_networkx()
        return list(nx.topological_sort(g))

    def total_prompt_tokens(self) -> int:
        """Aggregate prompt tokens across all execution spans."""
        return sum(s.prompt_tokens for s in self.spans)

    def total_completion_tokens(self) -> int:
        """Aggregate completion tokens across all execution spans."""
        return sum(s.completion_tokens for s in self.spans)

    def total_tokens(self) -> int:
        """Aggregate total tokens (prompt + completion) across all execution spans."""
        return self.total_prompt_tokens() + self.total_completion_tokens()

    def total_latency_ms(self) -> float:
        """Aggregate duration across all execution spans in milliseconds."""
        return sum(s.duration_ms for s in self.spans)

    def total_cost_usd(self) -> float:
        """Sum of cost_usd recorded directly in the spans."""
        return sum(s.cost_usd for s in self.spans)

    def get_tool_sequence(self) -> List[str]:
        """Return ordered or chronological sequence of tool names executed."""
        if not self.has_cycle():
            topo = self.topological_sort()
            return [self.tool_nodes[nid].tool_name for nid in topo if nid in self.tool_nodes]
        return [node.tool_name for node in self.tool_nodes.values()]

    def get_model_tiers_used(self) -> Set[ModelTier]:
        """Return all distinct model tiers utilized in this trajectory."""
        return {s.model_tier for s in self.spans if s.model_tier is not None}
