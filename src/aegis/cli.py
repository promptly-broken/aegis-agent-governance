"""Aegis Agent Governance Command-Line Interface (CLI)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from aegis.core.models import TrajectoryDAG
from aegis.core.pricing import PricingRegistry
from aegis.eval_engine.sprt import SequentialProbabilityRatioTest
from aegis.eval_engine.tdi import compute_tdi
from aegis.finops.ecrt import compute_ecrt
from aegis.finops.router import HybridComplexityRouter
from aegis.mcp_profiler.amplification import MCPAmplificationAuditor


def cmd_report(args: argparse.Namespace) -> int:
    """Execute marketplace dispute benchmark and output executive governance report."""
    try:
        from benchmarks.generate_executive_report import generate_report
    except ModuleNotFoundError:
        # Auto-detect benchmarks directory in current working dir or repository root
        cwd = Path.cwd()
        for candidate in [cwd, cwd.parent, Path(__file__).resolve().parent.parent.parent]:
            if (candidate / "benchmarks").is_dir():
                sys.path.insert(0, str(candidate.resolve()))
                break
        from benchmarks.generate_executive_report import generate_report


    report_markdown = generate_report()
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_markdown, encoding="utf-8")
        print(f"Executive governance report written to: {out_path.resolve()}", file=sys.stderr)
    else:
        print(report_markdown)
    return 0


def cmd_profile_mcp(args: argparse.Namespace) -> int:
    """Profile Stateful vs Stateless MCP context amplification."""
    auditor = MCPAmplificationAuditor()
    turns_data: List[Dict[str, Any]] = [
        {
            "step": f"mcp_step_{i + 1}",
            "payload_data": "x" * args.payload_bytes,
        }
        for i in range(args.turns)
    ]

    report = auditor.simulate_trajectory(turns_data)

    print("\n# Aegis MCP Context Amplification Audit\n")
    print(f"- **Simulated Turns:** {len(report.turns)}")
    print(f"- **Payload per Turn:** {args.payload_bytes} bytes")
    print(f"- **Stateless Cumulative Tokens:** {report.total_stateless_tokens:,}")
    print(f"- **Stateful Cumulative Tokens:** {report.total_stateful_tokens:,}")
    print(f"- **Token Amplification Factor (TAF):** {report.token_amplification_factor:.2f}x")
    print(f"- **Net Token Savings:** {report.percentage_token_reduction:.1f}% ({report.tokens_saved:,} tokens)\n")

    print("| Turn | Payload Bytes | Stateless Tokens | Stateful Tokens | Savings % |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for t in report.turns:
        savings = (
            100.0 * (1.0 - (t.stateful_tokens / t.stateless_tokens))
            if t.stateless_tokens > 0
            else 0.0
        )
        print(
            f"| {t.turn_index + 1} | {t.payload_bytes:,} | {t.stateless_tokens:,} | {t.stateful_tokens:,} | {savings:.1f}% |"
        )
    print()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate trajectory DAG against TDI and ECRT governance criteria."""
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: Trajectory file '{file_path}' does not exist.", file=sys.stderr)
        return 2

    try:
        raw_data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Error reading JSON from '{file_path}': {exc}", file=sys.stderr)
        return 2

    trajectories: List[TrajectoryDAG] = []
    if isinstance(raw_data, list):
        for item in raw_data:
            trajectories.append(TrajectoryDAG.model_validate(item))
    elif isinstance(raw_data, dict):
        trajectories.append(TrajectoryDAG.model_validate(raw_data))
    else:
        print("Error: Input file must be a JSON object or array of TrajectoryDAGs.", file=sys.stderr)
        return 2

    registry = PricingRegistry()
    ecrt_result = compute_ecrt(trajectories, pricing_registry=registry)

    # Compute TDI if multiple trajectories or reference provided
    tdi_score: Optional[float] = None
    if len(trajectories) >= 2:
        tdi_res = compute_tdi(trajectories, threshold=args.min_tdi)
        tdi_score = tdi_res.tdi_score

    print("\n# Aegis Trajectory Governance Evaluation\n")
    print(f"- **Evaluated Trajectories:** {len(trajectories)}")
    print(f"- **Resolution Rate:** {ecrt_result.resolution_rate * 100:.1f}%")
    print(f"- **Aggregate Cost:** ${ecrt_result.total_cost_usd:.4f}")
    print(f"- **ECRT:** ${ecrt_result.ecrt_usd:.4f} (Max threshold: ${args.max_ecrt:.4f})")
    if tdi_score is not None:
        print(f"- **TDI Score:** {tdi_score:.4f} (Min threshold: {args.min_tdi:.4f})")

    violations: List[str] = []
    if ecrt_result.ecrt_usd > args.max_ecrt:
        violations.append(
            f"ECRT ${ecrt_result.ecrt_usd:.4f} exceeds maximum threshold ${args.max_ecrt:.4f}"
        )
    if tdi_score is not None and tdi_score < args.min_tdi:
        violations.append(
            f"TDI {tdi_score:.4f} falls below minimum threshold {args.min_tdi:.4f}"
        )

    if violations:
        print("\n❌ **GOVERNANCE VIOLATIONS DETECTED:**")
        for v in violations:
            print(f"  - {v}")
        if args.strict:
            return 1
    else:
        print("\n✅ **ALL GOVERNANCE GATES PASSED**\n")

    return 0


def cmd_route(args: argparse.Namespace) -> int:
    """Interactively route a prompt step between Apple Silicon and Frontier Cloud."""
    router = HybridComplexityRouter()
    metadata: Dict[str, Any] = {}
    if args.task_type:
        metadata["task_type"] = args.task_type

    decision = router.route(args.prompt, metadata)
    print("\n# Aegis Hybrid Routing Decision\n")
    print(f"- **Target Tier:** {decision.selected_tier.value}")
    print(f"- **Model Recommendation:** {decision.recommended_model}")
    print(f"- **Task Complexity:** {decision.complexity.value}")
    print(f"- **Estimated Cost:** ${decision.estimated_cost_usd:.5f}")
    print(f"- **Confidence Score:** {decision.confidence_score:.3f}")
    print(f"- **Rationale:** {decision.rationale}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="aegis",
        description="Aegis Agent Governance: Statistical Evaluation & FinOps Control Plane",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to execute")

    # report
    p_report = subparsers.add_parser("report", help="Generate executive FinOps & governance report")
    p_report.add_argument("-o", "--output", help="Optional output path for markdown report")
    p_report.set_defaults(func=cmd_report)

    # profile-mcp
    p_mcp = subparsers.add_parser("profile-mcp", help="Audit MCP Token Amplification Factor (TAF)")
    p_mcp.add_argument("-t", "--turns", type=int, default=5, help="Number of turns to simulate")
    p_mcp.add_argument(
        "-p", "--payload-bytes", type=int, default=1024, help="Payload size per turn in bytes"
    )
    p_mcp.set_defaults(func=cmd_profile_mcp)

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate trajectory file against governance thresholds")
    p_eval.add_argument("-f", "--file", required=True, help="Path to JSON trajectory DAG or array")
    p_eval.add_argument(
        "--min-tdi", type=float, default=0.85, help="Minimum allowable Trajectory Determinism Index"
    )
    p_eval.add_argument(
        "--max-ecrt", type=float, default=0.10, help="Maximum allowable Effective Cost per Resolved Trajectory ($)"
    )
    p_eval.add_argument(
        "--strict", action="store_true", help="Exit with non-zero code if governance thresholds fail"
    )
    p_eval.set_defaults(func=cmd_eval)

    # route
    p_route = subparsers.add_parser("route", help="Evaluate complexity and route prompt to optimal tier")
    p_route.add_argument("prompt", help="Prompt or task instruction string to route")
    p_route.add_argument("--task-type", help="Optional task categorization hint")
    p_route.set_defaults(func=cmd_route)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
