"""Trajectory Determinism Index (TDI) calculation using Graph Edit Distance and schema alignment."""

from __future__ import annotations

from difflib import SequenceMatcher
import itertools
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import linear_sum_assignment

from aegis.core.models import TrajectoryDAG


class TDIResult(BaseModel):
    """Statistical outcome of Trajectory Determinism Index computation."""

    model_config = ConfigDict(extra="forbid")

    tdi: float = Field(..., ge=0.0, le=1.0, description="Trajectory Determinism Index score")
    pairwise_distances: List[Dict[str, Any]] = Field(default_factory=list)
    mean_normalized_distance: float = Field(..., ge=0.0)
    variance: float = Field(default=0.0, ge=0.0)
    std_dev: float = Field(default=0.0, ge=0.0)
    num_trajectories: int = Field(..., ge=1)
    is_deterministic: bool = Field(description="Whether TDI meets or exceeds target threshold")

    @property
    def tdi_score(self) -> float:
        """Backwards-compatible alias for `tdi`."""
        return self.tdi


def _argument_value_distance(val1: Any, val2: Any) -> float:
    """Compute normalized semantic distance in [0.0, 1.0] between two tool argument values."""
    if type(val1) is type(val2) and val1 == val2:
        return 0.0
    if isinstance(val1, bool) or isinstance(val2, bool):
        return 0.0 if val1 is val2 else 1.0
    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
        if val1 == val2:
            return 0.0
        denom = max(abs(float(val1)), abs(float(val2)), 1e-9)
        return min(1.0, abs(float(val1) - float(val2)) / denom)
    if isinstance(val1, str) and isinstance(val2, str):
        if val1 == val2:
            return 0.0
        return max(0.0, min(1.0, 1.0 - SequenceMatcher(None, val1, val2).ratio()))
    if isinstance(val1, dict) and isinstance(val2, dict):
        k1, k2 = set(val1.keys()), set(val2.keys())
        if not k1 and not k2:
            return 0.0
        union = len(k1 | k2)
        shared = k1 & k2
        key_dist = 1.0 - (len(shared) / union)
        val_dist = (
            sum(_argument_value_distance(val1[k], val2[k]) for k in shared) / len(shared)
            if shared
            else 1.0
        )
        return 0.5 * key_dist + 0.5 * val_dist
    if isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
        if not val1 and not val2:
            return 0.0
        max_len = max(len(val1), len(val2))
        min_len = min(len(val1), len(val2))
        elem_dist = (
            sum(_argument_value_distance(val1[i], val2[i]) for i in range(min_len)) / max_len
        )
        len_penalty = (max_len - min_len) / max_len
        return min(1.0, elem_dist + len_penalty)
    return 1.0


def _node_subst_cost(
    node1_attr: Dict[str, Any],
    node2_attr: Dict[str, Any],
    include_value_distance: bool = True,
) -> float:
    """Calculate substitution cost between two tool nodes combining schema key and value drift."""
    tool1 = node1_attr.get("tool_name", "")
    tool2 = node2_attr.get("tool_name", "")

    if tool1 != tool2:
        return 1.0

    # Same tool: compare schema keys using Jaccard distance
    args1 = node1_attr.get("arguments") or {}
    args2 = node2_attr.get("arguments") or {}
    keys1 = set(node1_attr.get("schema_keys") or args1.keys())
    keys2 = set(node2_attr.get("schema_keys") or args2.keys())

    if not keys1 and not keys2:
        return 0.0

    intersection_keys = keys1.intersection(keys2)
    union = len(keys1.union(keys2))

    if union == 0:
        return 0.0

    jaccard_distance = 1.0 - (len(intersection_keys) / union)

    if not include_value_distance or not intersection_keys or not (args1 and args2):
        return 0.5 * jaccard_distance

    shared_val_dist = sum(
        _argument_value_distance(args1.get(k), args2.get(k)) for k in intersection_keys
    ) / len(intersection_keys)

    if jaccard_distance == 0.0:
        return 0.25 * shared_val_dist

    return (0.35 * jaccard_distance) + (0.15 * shared_val_dist)


def _fast_dag_edit_distance(
    g1: nx.DiGraph, g2: nx.DiGraph, include_value_distance: bool = True
) -> float:
    """Fast bipartite node-matching and edge-difference approximation for Graph Edit Distance.

    Uses Hungarian matching (linear sum assignment) on cost matrix followed by edge symmetric difference.
    """
    nodes1 = list(g1.nodes(data=True))
    nodes2 = list(g2.nodes(data=True))
    n1 = len(nodes1)
    n2 = len(nodes2)

    if n1 == 0 and n2 == 0:
        return 0.0
    if n1 == 0:
        return float(n2 + g2.number_of_edges())
    if n2 == 0:
        return float(n1 + g1.number_of_edges())

    # Build cost matrix of size (n1 + n2) x (n1 + n2) for insertion/deletion/substitution
    max_dim = max(n1, n2)
    cost_matrix = np.zeros((max_dim, max_dim), dtype=float)

    for i in range(max_dim):
        for j in range(max_dim):
            if i < n1 and j < n2:
                cost_matrix[i, j] = _node_subst_cost(
                    nodes1[i][1], nodes2[j][1], include_value_distance=include_value_distance
                )
            elif i < n1 and j >= n2:
                # Deletion cost from g1
                cost_matrix[i, j] = 1.0
            elif i >= n1 and j < n2:
                # Insertion cost to g2
                cost_matrix[i, j] = 1.0
            else:
                cost_matrix[i, j] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    node_cost = float(cost_matrix[row_ind, col_ind].sum())

    # Edge mismatch penalty
    # Map matched nodes
    node_map: Dict[Any, Any] = {}
    for r, c in zip(row_ind, col_ind):
        if r < n1 and c < n2:
            node_map[nodes1[r][0]] = nodes2[c][0]

    # Compare edges under mapping
    mapped_edges_g1 = set()
    for u, v in g1.edges():
        u_mapped = node_map.get(u)
        v_mapped = node_map.get(v)
        if u_mapped is not None and v_mapped is not None:
            mapped_edges_g1.add((u_mapped, v_mapped))

    edges_g2 = set(g2.edges())
    edge_diff = len(mapped_edges_g1.symmetric_difference(edges_g2))
    # Add unmapped edges from g1
    unmapped_edges_g1 = g1.number_of_edges() - len(mapped_edges_g1)
    total_edge_cost = float(edge_diff + max(0, unmapped_edges_g1))

    return node_cost + total_edge_cost


def compute_dag_edit_distance(
    g1: nx.DiGraph,
    g2: nx.DiGraph,
    exact: bool = False,
    timeout_seconds: Optional[float] = 1.0,
    include_value_distance: bool = True,
) -> float:
    """Calculate Graph Edit Distance between two trajectory DAGs."""
    # Check isomorphism first for fast path
    if nx.is_isomorphic(g1, g2, node_match=lambda a, b: a == b):
        return 0.0

    if exact and (g1.number_of_nodes() + g2.number_of_nodes() <= 16):
        try:
            # Use networkx graph_edit_distance for small graphs
            ged_iter = nx.graph_edit_distance(
                g1,
                g2,
                node_subst_cost=lambda a, b: _node_subst_cost(
                    a, b, include_value_distance=include_value_distance
                ),
                node_del_cost=lambda a: 1.0,
                node_ins_cost=lambda a: 1.0,
                edge_subst_cost=lambda a, b: 0.0,
                edge_del_cost=lambda a: 1.0,
                edge_ins_cost=lambda a: 1.0,
                timeout=timeout_seconds,
            )
            # Take the optimal result
            return float(ged_iter)
        except Exception:
            # Fallback to fast DAG edit distance if timed out or failed
            pass

    return _fast_dag_edit_distance(g1, g2, include_value_distance=include_value_distance)


def compute_tdi(
    trajectories: List[Union[TrajectoryDAG, nx.DiGraph]],
    threshold: float = 0.85,
    exact: bool = False,
    timeout_per_pair: Optional[float] = 1.0,
    include_value_distance: bool = True,
) -> TDIResult:
    """Compute the Trajectory Determinism Index (TDI) across K Monte Carlo rollouts.

    Formula:
        TDI = 1 - (1 / binom(K, 2)) * sum_{i < j} [ d_edit(tau_i, tau_j) / max(|tau_i|, |tau_j|) ]
    """
    k = len(trajectories)
    if k == 0:
        raise ValueError("At least one trajectory is required to compute TDI.")

    if k == 1:
        return TDIResult(
            tdi=1.0,
            pairwise_distances=[],
            mean_normalized_distance=0.0,
            variance=0.0,
            std_dev=0.0,
            num_trajectories=1,
            is_deterministic=True,
        )

    # Convert to NetworkX graphs
    graphs: List[nx.DiGraph] = []
    traj_ids: List[str] = []
    for idx, t in enumerate(trajectories):
        if isinstance(t, TrajectoryDAG):
            graphs.append(t.to_networkx())
            traj_ids.append(t.trajectory_id)
        elif isinstance(t, nx.DiGraph):
            graphs.append(t)
            traj_ids.append(t.graph.get("trajectory_id", f"traj_{idx}"))
        else:
            raise TypeError(f"Unsupported trajectory type: {type(t)}")

    num_pairs = (k * (k - 1)) // 2
    normalized_distances: List[float] = []
    pairwise_records: List[Dict[str, Any]] = []

    for (i, g1), (j, g2) in itertools.combinations(enumerate(graphs), 2):
        size1 = g1.number_of_nodes() + g1.number_of_edges()
        size2 = g2.number_of_nodes() + g2.number_of_edges()
        max_size = max(size1, size2)

        if max_size == 0:
            norm_dist = 0.0
            raw_dist = 0.0
        else:
            raw_dist = compute_dag_edit_distance(
                g1,
                g2,
                exact=exact,
                timeout_seconds=timeout_per_pair,
                include_value_distance=include_value_distance,
            )
            # Bound normalized distance to [0.0, 1.0]
            norm_dist = min(1.0, raw_dist / max_size)

        normalized_distances.append(norm_dist)
        pairwise_records.append(
            {
                "pair": (traj_ids[i], traj_ids[j]),
                "raw_edit_distance": raw_dist,
                "max_size": max_size,
                "normalized_distance": norm_dist,
            }
        )

    mean_dist = float(np.mean(normalized_distances))
    var_dist = float(np.var(normalized_distances)) if len(normalized_distances) > 1 else 0.0
    std_dist = float(np.std(normalized_distances)) if len(normalized_distances) > 1 else 0.0

    tdi_score = max(0.0, min(1.0, 1.0 - mean_dist))

    return TDIResult(
        tdi=round(tdi_score, 4),
        pairwise_distances=pairwise_records,
        mean_normalized_distance=round(mean_dist, 4),
        variance=round(var_dist, 6),
        std_dev=round(std_dist, 6),
        num_trajectories=k,
        is_deterministic=tdi_score >= threshold,
    )
