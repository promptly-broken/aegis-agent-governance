"""Tests for OpenTelemetry (OTLP/JSON) and JSONL Audit Exporters."""

import json
from pathlib import Path

from aegis.cli import main as cli_main
from aegis.core.models import ModelTier, SpanType
from aegis.core.tracer import AegisTracer
from aegis.eval_engine.tdi import compute_tdi
from aegis.exporters import HTMLDashboardExporter, JSONLAuditExporter, OTLPTraceExporter
from aegis.finops.ecrt import compute_ecrt


def _build_sample_trajectory(scenario_id: str = "export_test", resolved: bool = True):
    tracer = AegisTracer(scenario_id=scenario_id)
    with tracer:
        with tracer.span("orchestrate", SpanType.ROUTER_DECISION) as root:
            root.set_attribute("router", "hybrid")
            tracer.record_llm_call(
                name="reasoning_call",
                model_name="gpt-4o",
                prompt_tokens=200,
                completion_tokens=80,
                duration_ms=120.0,
                model_tier=ModelTier.FRONTIER_CLOUD,
            )

        n1 = tracer.record_tool_call(
            tool_name="fetch_order",
            arguments={"order_id": "ORD-500"},
            output={"total": 299.0},
            duration_ms=25.0,
        )
        tracer.record_tool_call(
            tool_name="issue_refund",
            arguments={"order_id": "ORD-500", "amount": 299.0},
            output={"refunded": True},
            duration_ms=30.0,
            depends_on=[n1.node_id],
        )
        tracer.mark_resolved(resolved)
    return tracer.get_trajectory()


def test_otlp_exporter_payload_and_file(tmp_path: Path):
    dag1 = _build_sample_trajectory()
    dag2 = _build_sample_trajectory()

    tdi_res = compute_tdi([dag1, dag2])
    ecrt_res = compute_ecrt([dag1, dag2])

    exporter = OTLPTraceExporter(service_name="aegis-test-service")
    payload = exporter.to_otlp_payload([dag1, dag2], tdi_result=tdi_res, ecrt_result=ecrt_res)

    assert "resourceSpans" in payload
    res_span = payload["resourceSpans"][0]
    res_attrs = {a["key"]: a["value"] for a in res_span["resource"]["attributes"]}
    assert res_attrs["service.name"]["stringValue"] == "aegis-test-service"
    assert "aegis.governance.tdi" in res_attrs
    assert "aegis.finops.ecrt_usd" in res_attrs

    spans = res_span["scopeSpans"][0]["spans"]
    # Each trajectory produces: 1 root span + 2 execution spans + 2 tool node spans = 5 spans (10 total)
    assert len(spans) == 10
    span_names = [s["name"] for s in spans]
    assert "trajectory:export_test" in span_names
    assert "tool:fetch_order" in span_names
    assert "tool:issue_refund" in span_names

    out_file = tmp_path / "otlp_traces.json"
    exporter.export_to_file(out_file, dag1)
    assert out_file.exists()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert "resourceSpans" in loaded


def test_otlp_exporter_http_push(monkeypatch):
    import urllib.request

    dag = _build_sample_trajectory()
    exporter = OTLPTraceExporter(endpoint="http://127.0.0.1:4318/v1/traces")

    class FakeResp:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=2.0: FakeResp())
    assert exporter.export_http(dag) is True


def test_jsonl_audit_exporter_roundtrip(tmp_path: Path):
    dag1 = _build_sample_trajectory("scen_a")
    dag2 = _build_sample_trajectory("scen_b")

    audit_path = tmp_path / "audit_log.jsonl"
    exporter = JSONLAuditExporter(audit_path)

    rec = exporter.export_trajectory(dag1)
    assert rec["scenario_id"] == "scen_a"
    assert rec["summary"]["total_tokens"] == 280
    assert rec["summary"]["tool_sequence"] == ["fetch_order", "issue_refund"]

    written = exporter.export_batch([dag2])
    assert written == 1

    reloaded = exporter.load_trajectories()
    assert len(reloaded) == 2
    assert reloaded[0].trajectory_id == dag1.trajectory_id
    assert reloaded[1].trajectory_id == dag2.trajectory_id


def test_cli_export_otlp_and_jsonl(tmp_path: Path):
    dag = _build_sample_trajectory("cli_export_scenario")
    in_file = tmp_path / "input_traj.json"
    in_file.write_text(json.dumps(dag.model_dump(mode="json")), encoding="utf-8")

    otlp_out = tmp_path / "out_otlp.json"
    rc1 = cli_main(["export", "-f", str(in_file), "--format", "otlp", "-o", str(otlp_out)])
    assert rc1 == 0
    assert otlp_out.exists()

    jsonl_out = tmp_path / "out_audit.jsonl"
    rc2 = cli_main(["export", "-f", str(in_file), "--format", "jsonl", "-o", str(jsonl_out)])
    assert rc2 == 0
    assert jsonl_out.exists()
    reloaded = JSONLAuditExporter(jsonl_out).load_trajectories()
    assert len(reloaded) == 1
    assert reloaded[0].scenario_id == "cli_export_scenario"


def test_html_dashboard_exporter_and_cli(tmp_path: Path):
    dag = _build_sample_trajectory("dashboard_test")
    dash = HTMLDashboardExporter(title="Aegis Custom Dashboard")
    out_html = tmp_path / "dashboard.html"
    dash.export_to_file(out_html, [dag])

    assert out_html.exists()
    content = out_html.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "Aegis Custom Dashboard" in content
    assert "fetch_order" in content

    cli_dash_out = tmp_path / "cli_dashboard.html"
    rc = cli_main(["dashboard", "-o", str(cli_dash_out)])
    assert rc == 0
    assert cli_dash_out.exists()
    cli_html = cli_dash_out.read_text(encoding="utf-8")
    assert "Runaway Negotiation Loop Stress Test" in cli_html
    assert "Hybrid SQL &amp; Schema Pipeline" in cli_html


