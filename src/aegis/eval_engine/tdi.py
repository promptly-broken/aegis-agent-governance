"""Trajectory Determinism Index (TDI) calculation using Graph Edit Distance and schema alignment."""

from __future__ import annotations

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


def _node_subst_cost(node1_attr: Dict[str, Any], node2_attr: Dict[str, Any]) -> float:
    """Calculate substitution cost between two tool nodes."""
    tool1 = node1_attr.get("tool_name", "")
    tool2 = node2_attr.get("tool_name", "")

    if tool1 != tool2:
        return 1.0

    # Same tool: compare schema keys using Jaccard distance
    keys1 = set(node1_attr.get("schema_keys", []))
    keys2 = set(node2_attr.get("schema_keys", []))

    if not keys1 and not keys2:
        return 0.0

    intersection = len(keys1.intersection(keys2))
    union = len(keys1.union(keys2))

    if union == 0:
        return 0.0

    jaccard_distance = 1.0 - (intersection / union)
    # Weight schema key divergence by 0.5 when tool names match
    return 0.5 * jaccard_distance


def _fast_dag_edit_distance(g1: nx.DiGraph, g2: nx.DiGraph) -> float:
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
                cost_matrix[i, j] = _node_subst_cost(nodes1[i][1], nodes2[j][1])
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
                node_subst_cost=lambda a, b: _node_subst_cost(a, b),
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

    return _fast_dag_edit_distance(g1, g2)


def compute_tdi(
    trajectories: List[Union[TrajectoryDAG, nx.DiGraph]],
    threshold: float = 0.85,
    exact: bool = False,
    timeout_per_pair: Optional[float] = 1.0,
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
                g1, g2, exact=exact, timeout_seconds=timeout_per_pair
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
