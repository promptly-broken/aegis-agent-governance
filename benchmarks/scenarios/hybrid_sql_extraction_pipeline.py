"""Benchmark Scenario 3: Hybrid Edge-Cloud SQL & Schema Normalization Pipeline."""

from __future__ import annotations

from typing import Any, Dict, List

from aegis.core.models import ModelTier
from aegis.core.pricing import PricingRegistry
from aegis.core.tracer import AegisTracer
from aegis.finops.router import HybridComplexityRouter
from aegis.mcp_profiler.amplification import MCPAmplificationAuditor


PIPELINE_STEPS: List[Dict[str, Any]] = [
    {
        "name": "ingest_invoice_schema",
        "prompt": "Extract vendor_id, invoice_number, and line_items JSON schema from raw payload.",
        "prompt_tokens": 950,
        "completion_tokens": 180,
        "duration_ms": 140.0,
        "payload_bytes": 1200,
    },
    {
        "name": "validate_tax_fields",
        "prompt": "Validate tax_id format and normalize ISO currency code fields in JSON table.",
        "prompt_tokens": 820,
        "completion_tokens": 120,
        "duration_ms": 110.0,
        "payload_bytes": 950,
    },
    {
        "name": "deduplicate_line_items",
        "prompt": "Deduplicate SKU list and filter zero-quantity rows from invoice dictionary.",
        "prompt_tokens": 780,
        "completion_tokens": 140,
        "duration_ms": 105.0,
        "payload_bytes": 1100,
    },
    {
        "name": "generate_sql_lookup",
        "prompt": "Convert normalized vendor filter parameters into parameterized SQL lookup query.",
        "prompt_tokens": 640,
        "completion_tokens": 95,
        "duration_ms": 90.0,
        "payload_bytes": 800,
    },
    {
        "name": "arbitrate_compliance_exception",
        "prompt": "Evaluate cross-border tax compliance conflict, hypothesize root cause, and decide settlement strategy.",
        "prompt_tokens": 1650,
        "completion_tokens": 340,
        "duration_ms": 480.0,
        "payload_bytes": 2400,
    },
    {
        "name": "format_audit_record",
        "prompt": "Format compliance decision into strict JSON schema and extract audit keys.",
        "prompt_tokens": 720,
        "completion_tokens": 130,
        "duration_ms": 95.0,
        "payload_bytes": 1050,
    },
]


def run_hybrid_pipeline_benchmark() -> Dict[str, Any]:
    """Compare 100% Cloud + Stateless MCP against Aegis Hybrid Edge-Cloud + Stateful MCP."""
    registry = PricingRegistry()
    router = HybridComplexityRouter(entropy_threshold=0.65)
    cloud_profile = registry.get("gpt-4o")
    assert cloud_profile is not None

    cloud_only_cost_usd = 0.0
    hybrid_tracer = AegisTracer(scenario_id="hybrid_sql_compliance_pipeline", pricing_registry=registry)
    local_steps = 0
    cloud_steps = 0
    prev_node_id = None

    with hybrid_tracer:
        for step in PIPELINE_STEPS:
            # 1. Compute baseline 100% Frontier Cloud cost for this step
            cloud_only_cost_usd += cloud_profile.compute_cost(
                prompt_tokens=step["prompt_tokens"],
                completion_tokens=step["completion_tokens"],
                duration_ms=step["duration_ms"],
            )

            # 2. Route via Aegis HybridComplexityRouter
            decision = router.route(step["prompt"])
            if decision.selected_tier in (ModelTier.LOCAL_MLX, ModelTier.LOCAL_OLLAMA):
                local_steps += 1
            else:
                cloud_steps += 1

            hybrid_tracer.record_llm_call(
                name=step["name"],
                model_name=decision.recommended_model,
                prompt_tokens=step["prompt_tokens"],
                completion_tokens=step["completion_tokens"],
                duration_ms=step["duration_ms"],
                model_tier=decision.selected_tier,
            )
            node = hybrid_tracer.record_tool_call(
                tool_name=step["name"],
                arguments={"step": step["name"], "tier": decision.selected_tier.value},
                duration_ms=step["duration_ms"],
                depends_on=[prev_node_id] if prev_node_id else None,
            )
            prev_node_id = node.node_id

        hybrid_tracer.mark_resolved(True)

    hybrid_dag = hybrid_tracer.get_trajectory()
    hybrid_cost_usd = hybrid_dag.total_cost_usd()
    cost_reduction_pct = (
        100.0 * (1.0 - (hybrid_cost_usd / cloud_only_cost_usd))
        if cloud_only_cost_usd > 0
        else 0.0
    )

    auditor = MCPAmplificationAuditor()
    mcp_turns = [{"step": s["name"], "payload": "x" * s["payload_bytes"]} for s in PIPELINE_STEPS]
    mcp_report = auditor.simulate_trajectory(mcp_turns)

    return {
        "scenario_id": "hybrid_sql_compliance_pipeline",
        "total_steps": len(PIPELINE_STEPS),
        "local_mlx_steps": local_steps,
        "frontier_cloud_steps": cloud_steps,
        "cloud_only_cost_usd": round(cloud_only_cost_usd, 6),
        "hybrid_cost_usd": round(hybrid_cost_usd, 6),
        "arbitrage_savings_pct": round(cost_reduction_pct, 1),
        "mcp_taf": round(mcp_report.token_amplification_factor, 2),
        "mcp_token_reduction_pct": round(mcp_report.percentage_token_reduction, 1),
        "trajectory": hybrid_dag,
    }

