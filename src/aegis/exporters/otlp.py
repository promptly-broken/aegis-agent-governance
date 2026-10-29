"""OpenTelemetry (OTLP/JSON) exporter for TrajectoryDAG spans, tool nodes, and FinOps telemetry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import urllib.request

from aegis.core.models import Span, ToolNode, TrajectoryDAG
from aegis.eval_engine.tdi import TDIResult
from aegis.finops.ecrt import ECRTResult


def _to_hex_id(raw_id: str, byte_len: int) -> str:
    """Deterministically derive an OTLP hex traceId (16 bytes) or spanId (8 bytes) from an arbitrary ID."""
    cleaned = raw_id.replace("-", "").lower()
    hex_chars = byte_len * 2
    if len(cleaned) >= hex_chars and all(c in "0123456789abcdef" for c in cleaned[:hex_chars]):
        return cleaned[:hex_chars]
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:hex_chars]


def _otlp_attr(key: str, value: Any) -> Dict[str, Any]:
    """Format a key-value pair as an OTLP JSON attribute."""
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


class OTLPTraceExporter:
    """Exports Aegis TrajectoryDAG telemetry to OpenTelemetry OTLP/JSON (`resourceSpans`).

    Follows OpenInference and OpenTelemetry GenAI semantic conventions for direct
    compatibility with Arize Phoenix, Langfuse, Jaeger, and standard OTLP collectors.
    """

    def __init__(
        self,
        service_name: str = "aegis-agent-governance",
        service_version: str = "0.1.0",
        endpoint: str = "http://localhost:4318/v1/traces",
        timeout_seconds: float = 2.0,
    ) -> None:
        self.service_name = service_name
        self.service_version = service_version
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def _span_to_otlp(self, trace_id_hex: str, span: Span, scenario_id: str) -> Dict[str, Any]:
        span_id_hex = _to_hex_id(span.span_id, 8)
        parent_hex = _to_hex_id(span.parent_span_id, 8) if span.parent_span_id else ""

        start_ns = int(max(0.0, span.start_time_ms) * 1_000_000)
        end_ns = int(max(span.start_time_ms, span.end_time_ms) * 1_000_000)

        attrs: List[Dict[str, Any]] = [
            _otlp_attr("aegis.span_type", span.span_type.value),
            _otlp_attr("aegis.scenario_id", scenario_id),
            _otlp_attr("aegis.cost_usd", span.cost_usd),
            _otlp_attr("gen_ai.usage.input_tokens", span.prompt_tokens),
            _otlp_attr("gen_ai.usage.output_tokens", span.completion_tokens),
            _otlp_attr("gen_ai.usage.total_tokens", span.total_tokens),
        ]
        if span.model_name:
            attrs.append(_otlp_attr("gen_ai.request.model", span.model_name))
        if span.model_tier:
            attrs.append(_otlp_attr("aegis.model_tier", span.model_tier.value))
        if span.ttft_ms is not None:
            attrs.append(_otlp_attr("gen_ai.latency.ttft_ms", span.ttft_ms))
        for k, v in span.attributes.items():
            attrs.append(_otlp_attr(f"aegis.attr.{k}", v))

        status = (
            {"code": 2, "message": span.error}
            if span.error
            else {"code": 1}
        )

        return {
            "traceId": trace_id_hex,
            "spanId": span_id_hex,
            "parentSpanId": parent_hex,
            "name": span.name,
            "kind": 3 if span.span_type.value == "llm_inference" else 1,
            "startTimeUnixNano": str(start_ns),
            "endTimeUnixNano": str(end_ns),
            "attributes": attrs,
            "status": status,
        }

    def _tool_node_to_otlp(
        self,
        trace_id_hex: str,
        root_span_id_hex: str,
        node: ToolNode,
        upstream_ids: List[str],
        base_time_ms: float,
    ) -> Dict[str, Any]:
        span_id_hex = _to_hex_id(node.node_id, 8)
        parent_hex = _to_hex_id(node.span_id, 8) if node.span_id else root_span_id_hex
        start_ns = int(base_time_ms * 1_000_000)
        end_ns = int((base_time_ms + node.duration_ms) * 1_000_000)

        attrs: List[Dict[str, Any]] = [
            _otlp_attr("aegis.span_type", "tool_node"),
            _otlp_attr("tool.name", node.tool_name),
            _otlp_attr("tool.parameters", json.dumps(node.arguments, sort_keys=True, default=str)),
            _otlp_attr("tool.schema_keys", ",".join(node.schema_keys)),
            _otlp_attr("tool.status", node.status),
        ]
        if upstream_ids:
            attrs.append(_otlp_attr("aegis.depends_on", ",".join(upstream_ids)))

        return {
            "traceId": trace_id_hex,
            "spanId": span_id_hex,
            "parentSpanId": parent_hex,
            "name": f"tool:{node.tool_name}",
            "kind": 1,
            "startTimeUnixNano": str(start_ns),
            "endTimeUnixNano": str(end_ns),
            "attributes": attrs,
            "status": {"code": 1 if node.status == "success" else 2},
        }

    def to_otlp_payload(
        self,
        trajectories: Union[TrajectoryDAG, Sequence[TrajectoryDAG]],
        tdi_result: Optional[TDIResult] = None,
        ecrt_result: Optional[ECRTResult] = None,
    ) -> Dict[str, Any]:
        """Serialize one or more TrajectoryDAGs into an OTLP `resourceSpans` JSON dictionary."""
        traj_list = [trajectories] if isinstance(trajectories, TrajectoryDAG) else list(trajectories)

        resource_attrs: List[Dict[str, Any]] = [
            _otlp_attr("service.name", self.service_name),
            _otlp_attr("service.version", self.service_version),
        ]
        if tdi_result is not None:
            resource_attrs.append(_otlp_attr("aegis.governance.tdi", tdi_result.tdi))
            resource_attrs.append(
                _otlp_attr("aegis.governance.is_deterministic", tdi_result.is_deterministic)
            )
        if ecrt_result is not None:
            resource_attrs.append(_otlp_attr("aegis.finops.ecrt_usd", ecrt_result.ecrt_usd))
            resource_attrs.append(
                _otlp_attr("aegis.finops.resolution_rate", ecrt_result.resolution_rate)
            )

        otlp_spans: List[Dict[str, Any]] = []

        for dag in traj_list:
            trace_id_hex = _to_hex_id(dag.trajectory_id, 16)
            root_span_id_hex = _to_hex_id(f"root:{dag.trajectory_id}", 8)

            base_start_ms = (
                min((s.start_time_ms for s in dag.spans), default=1_700_000_000_000.0)
            )
            base_end_ms = (
                max(
                    (s.end_time_ms for s in dag.spans),
                    default=base_start_ms + max(1.0, dag.total_latency_ms()),
                )
            )

            # Trajectory root summary span
            root_attrs = [
                _otlp_attr("aegis.trajectory.id", dag.trajectory_id),
                _otlp_attr("aegis.trajectory.scenario_id", dag.scenario_id),
                _otlp_attr("aegis.trajectory.resolved", dag.resolved),
                _otlp_attr("aegis.trajectory.total_tokens", dag.total_tokens()),
                _otlp_attr("aegis.trajectory.total_cost_usd", dag.total_cost_usd()),
                _otlp_attr("aegis.trajectory.tool_count", len(dag.tool_nodes)),
            ]
            otlp_spans.append(
                {
                    "traceId": trace_id_hex,
                    "spanId": root_span_id_hex,
                    "parentSpanId": "",
                    "name": f"trajectory:{dag.scenario_id}",
                    "kind": 1,
                    "startTimeUnixNano": str(int(base_start_ms * 1_000_000)),
                    "endTimeUnixNano": str(int(base_end_ms * 1_000_000)),
                    "attributes": root_attrs,
                    "status": {"code": 1 if dag.resolved else 2},
                }
            )

            for s in dag.spans:
                otlp_s = self._span_to_otlp(trace_id_hex, s, dag.scenario_id)
                if not otlp_s["parentSpanId"]:
                    otlp_s["parentSpanId"] = root_span_id_hex
                otlp_spans.append(otlp_s)

            # Map causal dependencies u -> v
            upstream_map: Dict[str, List[str]] = {}
            for u, v in dag.edges:
                upstream_map.setdefault(v, []).append(u)

            for node_id, node in dag.tool_nodes.items():
                otlp_spans.append(
                    self._tool_node_to_otlp(
                        trace_id_hex=trace_id_hex,
                        root_span_id_hex=root_span_id_hex,
                        node=node,
                        upstream_ids=upstream_map.get(node_id, []),
                        base_time_ms=base_start_ms,
                    )
                )

        return {
            "resourceSpans": [
                {
                    "resource": {"attributes": resource_attrs},
                    "scopeSpans": [
                        {
                            "scope": {
                                "name": "aegis.telemetry.tracer",
                                "version": self.service_version,
                            },
                            "spans": otlp_spans,
                        }
                    ],
                }
            ]
        }

    def export_to_file(
        self,
        path: Union[str, Path],
        trajectories: Union[TrajectoryDAG, Sequence[TrajectoryDAG]],
        tdi_result: Optional[TDIResult] = None,
        ecrt_result: Optional[ECRTResult] = None,
    ) -> Path:
        """Write OTLP JSON payload to disk."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_otlp_payload(trajectories, tdi_result=tdi_result, ecrt_result=ecrt_result)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path

    def export_http(
        self,
        trajectories: Union[TrajectoryDAG, Sequence[TrajectoryDAG]],
        tdi_result: Optional[TDIResult] = None,
        ecrt_result: Optional[ECRTResult] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> bool:
        """POST OTLP JSON payload to an OpenTelemetry Collector HTTP `/v1/traces` endpoint."""
        payload = self.to_otlp_payload(trajectories, tdi_result=tdi_result, ecrt_result=ecrt_result)
        body = json.dumps(payload).encode("utf-8")
        req_headers = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(self.endpoint, data=body, headers=req_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False

