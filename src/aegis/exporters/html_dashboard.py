"""Self-contained Interactive HTML Governance & FinOps Dashboard Generator."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from aegis.core.models import TrajectoryDAG
from aegis.core.pricing import PricingRegistry
from aegis.eval_engine.sprt import SPRTResult
from aegis.eval_engine.tdi import TDIResult, compute_tdi
from aegis.finops.ecrt import ECRTResult, compute_ecrt
from aegis.mcp_profiler.amplification import (
    MCPAmplificationAuditor,
    MCPAmplificationReport,
)


class HTMLDashboardExporter:
    """Renders an interactive, zero-external-dependency HTML dashboard for Aegis telemetry."""

    def __init__(self, title: str = "Aegis Agent Governance & FinOps Control Plane") -> None:
        self.title = title

    def render_dashboard(
        self,
        trajectories: Sequence[TrajectoryDAG],
        tdi_result: Optional[TDIResult] = None,
        divergent_tdi_result: Optional[TDIResult] = None,
        ecrt_result: Optional[ECRTResult] = None,
        sprt_result: Optional[SPRTResult] = None,
        mcp_report: Optional[MCPAmplificationReport] = None,
        negotiation_benchmark: Optional[Dict[str, Any]] = None,
        hybrid_benchmark: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a complete self-contained HTML5 document summarizing governance and FinOps metrics."""
        registry = PricingRegistry()
        traj_list = list(trajectories)
        tdi_res = tdi_result or (compute_tdi(traj_list) if traj_list else None)
        ecrt_res = ecrt_result or (
            compute_ecrt(traj_list, pricing_registry=registry) if traj_list else None
        )
        if mcp_report is None:
            mcp_report = MCPAmplificationAuditor().simulate_trajectory(
                [
                    {"step": "order_lookup", "data": "Order #ORD-99824 retrieved"},
                    {"step": "tracking", "data": "FedEx package delayed in Chicago transit"},
                    {"step": "policy", "data": "Policy 402 allows instant refund up to $200"},
                    {"step": "arbitration", "data": "Refund issued to customer card ending 4410"},
                ]
            )

        tdi_val = f"{tdi_res.tdi:.4f}" if tdi_res else "N/A"
        tdi_status = (
            "PASSED" if (tdi_res and tdi_res.is_deterministic) else "FAILED"
        )
        ecrt_val = f"${ecrt_res.ecrt_usd:.5f}" if ecrt_res else "$0.00000"
        res_rate = f"{ecrt_res.resolution_rate * 100:.1f}%" if ecrt_res else "0.0%"
        taf_val = f"{mcp_report.token_amplification_factor:.2f}x"
        taf_savings = f"{mcp_report.percentage_token_reduction:.1f}%"

        # Build tool DAG rows for representative trajectory
        sample_dag = traj_list[0] if traj_list else None
        dag_rows_html = ""
        if sample_dag:
            upstream_map: Dict[str, List[str]] = {}
            for u, v in sample_dag.edges:
                upstream_map.setdefault(v, []).append(u)
            for nid, node in sample_dag.tool_nodes.items():
                deps = ", ".join(upstream_map.get(nid, [])) or "root"
                keys = ", ".join(node.schema_keys) or "-"
                dag_rows_html += (
                    f"<tr>"
                    f"<td><code>{html.escape(nid)}</code></td>"
                    f"<td><strong>{html.escape(node.tool_name)}</strong></td>"
                    f"<td><code>{html.escape(keys)}</code></td>"
                    f"<td><code>{html.escape(deps)}</code></td>"
                    f"<td>{node.duration_ms:.1f} ms</td>"
                    f"<td><span class='badge pass'>{html.escape(node.status.upper())}</span></td>"
                    f"</tr>"
                )

        # Build MCP turn bars
        mcp_rows_html = ""
        max_stateless = max((t.stateless_tokens for t in mcp_report.turns), default=1)
        for t in mcp_report.turns:
            sl_pct = min(100.0, (t.stateless_tokens / max_stateless) * 100.0)
            sf_pct = min(100.0, (t.stateful_tokens / max_stateless) * 100.0)
            mcp_rows_html += (
                f"<tr>"
                f"<td>Turn {t.turn_index + 1}</td>"
                f"<td>{t.stateless_tokens:,} tok "
                f"<div class='bar-track'><div class='bar-fill warn' style='width:{sl_pct:.1f}%'></div></div></td>"
                f"<td>{t.stateful_tokens:,} tok "
                f"<div class='bar-track'><div class='bar-fill ok' style='width:{sf_pct:.1f}%'></div></div></td>"
                f"</tr>"
            )

        # Multi-scenario benchmark summary rows
        scenario_rows_html = ""
        if negotiation_benchmark:
            scenario_rows_html += (
                f"<tr>"
                f"<td><strong>Runaway Negotiation Loop Stress Test</strong></td>"
                f"<td>Streaming Circuit Breaker (Cycle Limit=2)</td>"
                f"<td>{negotiation_benchmark['ungoverned_turns']} turns (${negotiation_benchmark['ungoverned_cost_usd']:.5f})</td>"
                f"<td>{negotiation_benchmark['governed_turns_before_trip']} turns (${negotiation_benchmark['governed_cost_usd']:.5f})</td>"
                f"<td><span class='badge pass'>{negotiation_benchmark['cost_savings_pct']:.1f}% Cost Saved</span></td>"
                f"</tr>"
            )
        if hybrid_benchmark:
            scenario_rows_html += (
                f"<tr>"
                f"<td><strong>Hybrid SQL &amp; Schema Pipeline</strong></td>"
                f"<td>Apple Silicon MLX ({hybrid_benchmark['local_mlx_steps']} steps) + Cloud ({hybrid_benchmark['frontier_cloud_steps']} step)</td>"
                f"<td>100% Cloud (${hybrid_benchmark['cloud_only_cost_usd']:.5f})</td>"
                f"<td>Hybrid Edge-Cloud (${hybrid_benchmark['hybrid_cost_usd']:.5f})</td>"
                f"<td><span class='badge pass'>{hybrid_benchmark['arbitrage_savings_pct']:.1f}% Arbitrage Savings</span></td>"
                f"</tr>"
            )

        sprt_section = ""
        if sprt_result:
            sprt_section = (
                f"<div class='card'>"
                f"<h3>Wald's SPRT Early-Stopping Gate</h3>"
                f"<p class='metric'>{html.escape(sprt_result.decision.value.upper())}</p>"
                f"<p class='sub'>Terminated at rollout <strong>#{sprt_result.step}</strong> "
                f"(Empirical reliability: {sprt_result.empirical_rate * 100:.1f}%)</p>"
                f"</div>"
            )

        div_tdi_str = (
            f"Stochastic Divergent Cohort TDI: <strong>{divergent_tdi_result.tdi:.4f}</strong> (FAILED)"
            if divergent_tdi_result
            else "Threshold: &ge; 0.8500"
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(self.title)}</title>
  <style>
    :root {{
      color-scheme: dark light;
      --bg: #0f172a;
      --surface: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --pass: #10b981;
      --warn: #f59e0b;
      --fail: #ef4444;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 2rem;
      line-height: 1.5;
    }}
    .container {{ max-width: 1180px; margin: 0 auto; }}
    header {{
      border-bottom: 1px solid var(--border);
      padding-bottom: 1rem;
      margin-bottom: 2rem;
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      flex-wrap: wrap;
    }}
    h1 {{ margin: 0; font-size: 1.65rem; }}
    h2 {{ font-size: 1.2rem; margin-top: 2rem; color: var(--accent); }}
    h3 {{ margin: 0 0 0.5rem 0; font-size: 0.95rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1.25rem;
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.25rem;
    }}
    .metric {{ font-size: 2rem; font-weight: 700; margin: 0.25rem 0; }}
    .sub {{ font-size: 0.85rem; color: var(--muted); margin: 0; }}
    .badge {{
      display: inline-block;
      padding: 0.15rem 0.55rem;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .badge.pass {{ background: rgba(16, 185, 129, 0.2); color: var(--pass); }}
    .badge.fail {{ background: rgba(239, 68, 68, 0.2); color: var(--fail); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      overflow: hidden;
      margin-top: 0.75rem;
    }}
    th, td {{
      text-align: left;
      padding: 0.75rem 1rem;
      border-bottom: 1px solid var(--border);
      font-size: 0.9rem;
    }}
    th {{ color: var(--muted); font-weight: 600; background: rgba(15, 23, 42, 0.4); }}
    .bar-track {{
      width: 100%;
      height: 8px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 4px;
      margin-top: 4px;
      overflow: hidden;
    }}
    .bar-fill {{ height: 100%; border-radius: 4px; }}
    .bar-fill.ok {{ background: var(--pass); }}
    .bar-fill.warn {{ background: var(--warn); }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85em; color: var(--accent); }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>{html.escape(self.title)}</h1>
        <p class="sub">Statistical Trajectory Determinism, Protocol Profiling &amp; Apple Silicon FinOps Control Plane</p>
      </div>
      <span class="badge {'pass' if tdi_status == 'PASSED' else 'fail'}">GATE STATUS: {tdi_status}</span>
    </header>

    <section class="grid">
      <div class="card">
        <h3>Trajectory Determinism (TDI)</h3>
        <p class="metric">{tdi_val}</p>
        <p class="sub">{div_tdi_str}</p>
      </div>
      <div class="card">
        <h3>Effective Cost / Resolved (ECRT)</h3>
        <p class="metric">{ecrt_val}</p>
        <p class="sub">Empirical Resolution Rate: <strong>{res_rate}</strong></p>
      </div>
      <div class="card">
        <h3>MCP Token Amplification (TAF)</h3>
        <p class="metric">{taf_val}</p>
        <p class="sub">Stateful Pointer Token Savings: <strong>{taf_savings}</strong></p>
      </div>
      {sprt_section}
    </section>

    <h2>Multi-Scenario Governance &amp; FinOps Benchmarks</h2>
    <table>
      <thead>
        <tr>
          <th>Scenario</th>
          <th>Active Governance Mechanism</th>
          <th>Ungoverned / Cloud Baseline</th>
          <th>Aegis Governed Outcome</th>
          <th>Unit Economics Impact</th>
        </tr>
      </thead>
      <tbody>
        {scenario_rows_html or "<tr><td colspan='5'>Standard Marketplace Dispute Resolution Cohort Evaluated</td></tr>"}
      </tbody>
    </table>

    <h2>Canonical Trajectory Tool-Call DAG Topology</h2>
    <table>
      <thead>
        <tr>
          <th>Node ID</th>
          <th>Tool Name</th>
          <th>Schema Keys</th>
          <th>Causal Dependencies</th>
          <th>Latency</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {dag_rows_html}
      </tbody>
    </table>

    <h2>Stateless vs. Stateful MCP Context Token Profile</h2>
    <table>
      <thead>
        <tr>
          <th>Turn</th>
          <th>Stateless MCP (Full History Re-transmission)</th>
          <th>Stateful MCP (Session Pointer URN)</th>
        </tr>
      </thead>
      <tbody>
        {mcp_rows_html}
      </tbody>
    </table>
  </div>
</body>
</html>
"""

    def export_to_file(
        self,
        path: Union[str, Path],
        trajectories: Sequence[TrajectoryDAG],
        **kwargs: Any,
    ) -> Path:
        """Render and save the HTML dashboard to a file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        html_str = self.render_dashboard(trajectories, **kwargs)
        out_path.write_text(html_str, encoding="utf-8")
        return out_path
