"""Generate executive governance and FinOps telemetry report with Markdown tables."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aegis.core.pricing import PricingRegistry
from aegis.eval_engine.sprt import SequentialProbabilityRatioTest
from aegis.eval_engine.tdi import compute_tdi
from aegis.exporters.html_dashboard import HTMLDashboardExporter
from aegis.finops.ecrt import compute_ecrt
from aegis.mcp_profiler.amplification import MCPAmplificationAuditor
from benchmarks.scenarios.hybrid_sql_extraction_pipeline import (
    run_hybrid_pipeline_benchmark,
)
from benchmarks.scenarios.marketplace_dispute_resolution import (
    generate_benchmark_rollouts,
)
from benchmarks.scenarios.runaway_negotiation_loop import (
    run_governed_negotiation_stress_test,
)


def generate_report(output_format: str = "markdown") -> str:
    registry = PricingRegistry()

    # 1. Trajectory Determinism Index (TDI) Evaluation
    canonical_rollouts = generate_benchmark_rollouts(num_rollouts=6, divergence_factor=0.0)
    divergent_rollouts = generate_benchmark_rollouts(num_rollouts=6, divergence_factor=0.85)

    tdi_canonical = compute_tdi(canonical_rollouts, threshold=0.85)
    tdi_divergent = compute_tdi(divergent_rollouts, threshold=0.85)

    # 2. SPRT Early Stopping Simulation
    sprt = SequentialProbabilityRatioTest(p0=0.70, p1=0.90, alpha=0.05, beta=0.10)
    sample_stream = [True, True, True, True, True, True, False, True, True, True, True]
    sprt_res = sprt.batch(sample_stream)

    # 3. FinOps ECRT Evaluation
    ecrt_res = compute_ecrt(canonical_rollouts, pricing_registry=registry, infra_overhead_usd=0.005)

    # 4. MCP Token Amplification Factor
    auditor = MCPAmplificationAuditor()
    mcp_report = auditor.simulate_trajectory(
        [
            {"step": "order_lookup", "data": "Order #ORD-99824 retrieved"},
            {"step": "tracking", "data": "FedEx package delayed in Chicago transit"},
            {"step": "policy", "data": "Policy 402 allows instant refund up to $200"},
            {"step": "arbitration", "data": "Refund issued to customer card ending 4410"},
        ]
    )

    # 5. Additional Multi-Scenario Benchmarks
    negotiation_bench = run_governed_negotiation_stress_test(max_cycle_repetitions=2)
    hybrid_bench = run_hybrid_pipeline_benchmark()

    if output_format.lower() == "html":
        dashboard = HTMLDashboardExporter()
        return dashboard.render_dashboard(
            trajectories=canonical_rollouts,
            tdi_result=tdi_canonical,
            divergent_tdi_result=tdi_divergent,
            ecrt_result=ecrt_res,
            sprt_result=sprt_res,
            mcp_report=mcp_report,
            negotiation_benchmark=negotiation_bench,
            hybrid_benchmark=hybrid_bench,
        )

    report = f"""# Aegis Executive Governance & FinOps Telemetry Report

**Generated:** Automatic CI/CD Benchmark Harness  
**Scope:** Multi-Agent Marketplace Dispute Resolution Benchmark  

---

## 1. Statistical Trajectory Stability (TDI)

| Scenario Cohort | Rollouts ($K$) | Mean Distance | Variance | TDI Score | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Canonical Baseline** | {tdi_canonical.num_trajectories} | {tdi_canonical.mean_normalized_distance:.4f} | {tdi_canonical.variance:.6f} | **{tdi_canonical.tdi:.4f}** | {'PASSED' if tdi_canonical.is_deterministic else 'FAILED'} |
| **Stochastic Divergent** | {tdi_divergent.num_trajectories} | {tdi_divergent.mean_normalized_distance:.4f} | {tdi_divergent.variance:.6f} | **{tdi_divergent.tdi:.4f}** | {'PASSED' if tdi_divergent.is_deterministic else 'FAILED'} |

---

## 2. Wald's SPRT CI/CD Early-Stopping Gate

| Metric | Configured Target / Observed Value |
|:---|:---|
| **Hypothesis Null ($H_0$)** | $p_0 = {sprt.p0}$ (Unacceptable Reliability) |
| **Hypothesis Alt ($H_1$)** | $p_1 = {sprt.p1}$ (Production Target) |
| **Significance ($\alpha, \beta$)** | $\alpha = {sprt.alpha}$, $\beta = {sprt.beta}$ |
| **Rollouts Evaluated** | {sprt_res.step} attempts |
| **Decision Reached** | `{sprt_res.decision.value}` |
| **Empirical Success Rate** | {sprt_res.empirical_rate * 100:.1f}% |
| **Compute Cycles Saved** | Early stopped at step {sprt_res.step} vs 100 fixed trials |

---

## 3. Unit Economics & ECRT Telemetry

| Metric | Value (USD) |
|:---|:---|
| **Effective Cost / Resolved Trajectory (ECRT)** | **${ecrt_res.ecrt_usd:.5f}** |
| **Empirical Resolution Rate** | {ecrt_res.resolution_rate * 100:.1f}% ({ecrt_res.resolved_trajectories}/{ecrt_res.total_trajectories}) |
| **Average Cost per Attempt** | ${ecrt_res.average_cost_per_attempt_usd:.5f} |
| **Total Batch Expenditure** | ${ecrt_res.total_cost_usd:.5f} |
| **95% Confidence Interval (ECRT)** | [${ecrt_res.confidence_interval_95[0]:.5f}, ${ecrt_res.confidence_interval_95[1]:.5f}] |
| **SLA Budget Compliance** | {'COMPLIANT' if ecrt_res.is_budget_compliant else 'BREACHED'} |

---

## 4. MCP Protocol Profiling & Token Amplification (TAF)

| Protocol Mode | Context Tokens Transmitted | Efficiency Gain |
|:---|:---:|:---:|
| **Stateless MCP (Full History Re-transmission)** | {mcp_report.total_stateless_tokens} tokens | Baseline |
| **Stateful MCP (Redis Session Pointer)** | {mcp_report.total_stateful_tokens} tokens | **{mcp_report.percentage_token_reduction:.1f}% reduction** |
| **Token Amplification Factor (TAF)** | **{mcp_report.token_amplification_factor:.2f}x** | Context Bloat Quantified |

---

## 5. Multi-Scenario Governance & Arbitrage Stress Benchmarks

| Scenario Suite | Ungoverned / Cloud Baseline | Aegis Governed Outcome | Net Efficiency Gain |
|:---|:---|:---|:---|
| **Runaway Negotiation Loop Stress Test** | {negotiation_bench['ungoverned_turns']} turns (${negotiation_bench['ungoverned_cost_usd']:.5f}) | Tripped at turn #{negotiation_bench['tripped_on_attempt']} (${negotiation_bench['governed_cost_usd']:.5f}) | **{negotiation_bench['cost_savings_pct']:.1f}% cost saved** ({negotiation_bench['tokens_saved']:,} tokens) |
| **Hybrid SQL & Schema Normalization Pipeline** | 100% Cloud (${hybrid_bench['cloud_only_cost_usd']:.5f}) | {hybrid_bench['local_mlx_steps']} MLX + {hybrid_bench['frontier_cloud_steps']} Cloud (${hybrid_bench['hybrid_cost_usd']:.5f}) | **{hybrid_bench['arbitrage_savings_pct']:.1f}% FinOps arbitrage** |
"""
    return report


if __name__ == "__main__":
    report_md = generate_report()
    print(report_md)
    sys.exit(0)
