"""Append-only JSON Lines (JSONL) audit log exporter and loader for TrajectoryDAG telemetry."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from aegis.core.models import TrajectoryDAG
from aegis.eval_engine.tdi import TDIResult
from aegis.finops.ecrt import ECRTResult


class JSONLAuditExporter:
    """Persists and reloads governed agent trajectories in append-only JSONL audit logs."""

    def __init__(self, file_path: Union[str, Path]) -> None:
        self.file_path = Path(file_path)

    def format_record(
        self,
        trajectory: TrajectoryDAG,
        tdi_result: Optional[TDIResult] = None,
        ecrt_result: Optional[ECRTResult] = None,
    ) -> Dict[str, Any]:
        """Build a structured compliance audit record for a single trajectory."""
        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trajectory_id": trajectory.trajectory_id,
            "scenario_id": trajectory.scenario_id,
            "resolved": trajectory.resolved,
            "summary": {
                "total_tokens": trajectory.total_tokens(),
                "total_cost_usd": round(trajectory.total_cost_usd(), 8),
                "total_latency_ms": round(trajectory.total_latency_ms(), 3),
                "num_spans": len(trajectory.spans),
                "num_tool_nodes": len(trajectory.tool_nodes),
                "tool_sequence": trajectory.get_tool_sequence(),
                "has_cycle": trajectory.has_cycle(),
                "tiers_used": sorted(t.value for t in trajectory.get_model_tiers_used()),
            },
            "trajectory": trajectory.model_dump(mode="json"),
        }
        if tdi_result is not None:
            record["governance_tdi"] = {
                "tdi": tdi_result.tdi,
                "is_deterministic": tdi_result.is_deterministic,
            }
        if ecrt_result is not None:
            record["finops_ecrt"] = {
                "ecrt_usd": ecrt_result.ecrt_usd,
                "resolution_rate": ecrt_result.resolution_rate,
                "is_budget_compliant": ecrt_result.is_budget_compliant,
            }
        return record

    def export_trajectory(
        self,
        trajectory: TrajectoryDAG,
        tdi_result: Optional[TDIResult] = None,
        ecrt_result: Optional[ECRTResult] = None,
    ) -> Dict[str, Any]:
        """Append a single trajectory record to the JSONL audit file."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        record = self.format_record(trajectory, tdi_result=tdi_result, ecrt_result=ecrt_result)
        with self.file_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return record

    def export_batch(
        self,
        trajectories: Sequence[TrajectoryDAG],
        tdi_result: Optional[TDIResult] = None,
        ecrt_result: Optional[ECRTResult] = None,
    ) -> int:
        """Append a batch of trajectories to the JSONL audit file. Returns count written."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with self.file_path.open("a", encoding="utf-8") as fh:
            for traj in trajectories:
                rec = self.format_record(traj, tdi_result=tdi_result, ecrt_result=ecrt_result)
                fh.write(json.dumps(rec) + "\n")
                count += 1
        return count

    def load_trajectories(self) -> List[TrajectoryDAG]:
        """Deserialize all TrajectoryDAG objects from the JSONL audit file."""
        if not self.file_path.exists():
            return []
        loaded: List[TrajectoryDAG] = []
        with self.file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                raw = json.loads(stripped)
                traj_data = raw.get("trajectory", raw)
                loaded.append(TrajectoryDAG.model_validate(traj_data))
        return loaded

