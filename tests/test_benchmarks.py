"""Tests for Multi-Scenario Benchmark Suite (Negotiation Loop & Hybrid SQL Pipeline)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.generate_executive_report import generate_report
from benchmarks.scenarios.hybrid_sql_extraction_pipeline import (
    run_hybrid_pipeline_benchmark,
)
from benchmarks.scenarios.runaway_negotiation_loop import (
    run_governed_negotiation_stress_test,
)


def test_runaway_negotiation_loop_stress_benchmark():
    res = run_governed_negotiation_stress_test(max_cycle_repetitions=2)
    assert res["ungoverned_turns"] == 16
    assert res["governed_turns_before_trip"] == 3
    assert res["tripped_on_attempt"] == 4
    assert res["trip_reason"] == "cyclic_loop_detected"
    assert res["cost_savings_pct"] > 80.0
    assert res["tokens_saved"] > 20_000


def test_hybrid_sql_extraction_pipeline_benchmark():
    res = run_hybrid_pipeline_benchmark()
    assert res["total_steps"] == 6
    assert res["local_mlx_steps"] == 5
    assert res["frontier_cloud_steps"] == 1
    assert res["hybrid_cost_usd"] < res["cloud_only_cost_usd"]
    assert res["arbitrage_savings_pct"] > 50.0
    assert res["mcp_taf"] > 1.5


def test_executive_report_includes_all_scenarios():
    md = generate_report(output_format="markdown")
    assert "Runaway Negotiation Loop Stress Test" in md
    assert "Hybrid SQL & Schema Normalization Pipeline" in md
