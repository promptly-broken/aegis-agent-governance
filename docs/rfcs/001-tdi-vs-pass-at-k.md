# RFC 001: Trajectory Determinism Index (TDI) vs. Pass@k in Agent Evaluation

**Author:** Principal Distributed Systems & AI Infrastructure Architect  
**Status:** Accepted  
**Target:** `aegis.eval_engine.tdi`

---

## 1. Problem Statement

Standard LLM benchmark suites evaluate model capabilities using metrics derived from code generation benchmarks, notably $\text{Pass}@k$:

$$\text{Pass}@k = \mathbb{E}\left[1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}\right]$$

where $n$ total samples are generated and $c$ correct solutions are obtained. 

While suitable for single-turn code generation, $\text{Pass}@k$ fails dramatically when applied to autonomous multi-agent systems executing in enterprise production environments for three fundamental reasons:
1. **Outcome-Blind Process Flaws:** An agent may achieve resolution ($c \ge 1$) through an erroneous, unsafe, or non-compliant sequence of side-effects (e.g. deleting production database records, triggering unauthorized refunds, or oscillating 50 times between tools before accidentally finding the right argument).
2. **Regulatory & Audit Invalidation:** Enterprise compliance mandates determinism. Financial institutions and healthcare systems require that identical inputs produce structurally verifiable, deterministic execution traces.
3. **Flaky Rollouts:** An agent with $\text{Pass}@1 = 0.80$ might execute completely disjoint tool-call paths across attempts, preventing deterministic caching, security sandboxing, and regression tracking.

---

## 2. Mathematical Definition of TDI

The **Trajectory Determinism Index (TDI)** quantifies the structural, semantic, and schema stability of an agent's execution DAG across $K$ Monte Carlo rollouts $\mathcal{T} = \{\tau_1, \tau_2, \dots, \tau_K\}$ under identical initial conditions:

$$TDI = 1 - \frac{1}{\binom{K}{2}} \sum_{1 \le i < j \le K} \frac{d_{\text{edit}}(\tau_i, \tau_j)}{\max(|\tau_i|, |\tau_j|)}$$

where:
- $\tau_i = (V_i, E_i)$ is the directed acyclic graph of tool invocations.
- $|\tau_i| = |V_i| + |E_i|$ represents graph complexity.
- $d_{\text{edit}}(\tau_i, \tau_j)$ is the normalized Graph Edit Distance.

### Node Cost Metric
For tool nodes $u \in V_i$ and $v \in V_j$:
$$c_{\text{node}}(u, v) = \begin{cases}
1.0 & \text{if } \text{tool\_name}(u) \neq \text{tool\_name}(v) \\
0.5 \cdot \left(1.0 - \frac{|\text{keys}(u) \cap \text{keys}(v)|}{|\text{keys}(u) \cup \text{keys}(v)|}\right) & \text{if } \text{tool\_name}(u) = \text{tool\_name}(v)
\end{cases}$$

---

## 3. Comparative Matrix

| Evaluation Dimension | Pass@k | LLM-as-a-Judge | Trajectory Determinism Index (TDI) |
|---|---|---|---|
| **Underlying Substrate** | Final output correctness | Judge prompt heuristic | Tool invocation DAG topology |
| **Statistical Rigor** | Combinatorial expectation | Qualitative subjective score | Exact Graph Edit Distance |
| **Process Observability** | Zero (black box) | Low (summary only) | High (every tool node & edge) |
| **CI/CD Determinism** | Flaky | High variance | Deterministic & reproducible |
| **Compute Overhead** | Low | High (requires frontier LLM calls) | Near zero (analytical graph matching) |

---

## 4. Architectural Recommendation

CI/CD pipelines should couple Wald's SPRT (for sample efficiency) with TDI:
1. Run $K$ rollouts.
2. If $TDI < 0.85$, fail the build immediately, regardless of task success rate, preventing stochastic regression from entering staging environments.
