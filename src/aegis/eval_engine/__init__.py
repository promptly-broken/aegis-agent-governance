"""Aegis Evaluation Engine: Trajectory Determinism Index (TDI) and SPRT early stopping."""

from aegis.eval_engine.sprt import (
    SPRTDecision,
    SPRTResult,
    SequentialProbabilityRatioTest,
)
from aegis.eval_engine.tdi import (
    TDIResult,
    compute_dag_edit_distance,
    compute_tdi,
)

__all__ = [
    "compute_tdi",
    "compute_dag_edit_distance",
    "TDIResult",
    "SequentialProbabilityRatioTest",
    "SPRTDecision",
    "SPRTResult",
]
