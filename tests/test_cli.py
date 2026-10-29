"""Tests for the aegis CLI interface and subcommands."""

import json
from pathlib import Path

import pytest

from aegis.cli import main
from aegis.core.models import TrajectoryDAG


def test_cli_route_deterministic(capsys):
    exit_code = main(["route", "Extract JSON fields from order"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "deterministic_low" in captured
    assert "local_mlx" in captured


def test_cli_route_reasoning(capsys):
    exit_code = main(["route", "Arbitrate customer refund dispute and decide policy settlement"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "reasoning_high" in captured
    assert "frontier_cloud" in captured


def test_cli_profile_mcp(capsys):
    exit_code = main(["profile-mcp", "--turns", "3", "--payload-bytes", "256"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Aegis MCP Context Amplification Audit" in captured
    assert "Simulated Turns:" in captured
    assert "Token Amplification Factor" in captured


def test_cli_report(tmp_path, capsys):
    out_file = tmp_path / "executive_report.md"
    exit_code = main(["report", "-o", str(out_file)])
    assert exit_code == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "# Aegis Executive Governance & FinOps Telemetry Report" in content
    assert "Statistical Trajectory Stability (TDI)" in content



def test_cli_eval_success(tmp_path, capsys):
    # Create valid synthetic trajectory
    traj = TrajectoryDAG(trajectory_id="t1", scenario_id="s1", resolved=True)
    traj_path = tmp_path / "trajectory.json"
    traj_path.write_text(traj.model_dump_json(), encoding="utf-8")

    exit_code = main(["eval", "-f", str(traj_path), "--max-ecrt", "1.00"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "ALL GOVERNANCE GATES PASSED" in captured


def test_cli_eval_violation_strict(tmp_path, capsys):
    traj = TrajectoryDAG(trajectory_id="t_fail", scenario_id="s1", resolved=False)
    traj_path = tmp_path / "trajectory_fail.json"
    traj_path.write_text(traj.model_dump_json(), encoding="utf-8")

    # Force failure by setting max-ecrt to $0.0000001
    exit_code = main(["eval", "-f", str(traj_path), "--max-ecrt", "0.0000001", "--strict"])
    assert exit_code == 1
    captured = capsys.readouterr().out
    assert "GOVERNANCE VIOLATIONS DETECTED" in captured


def test_cli_eval_missing_file():
    exit_code = main(["eval", "-f", "non_existent_file_xyz.json"])
    assert exit_code == 2
