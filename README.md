# Aegis Agent Governance (`aegis-agent-governance`)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/Architecture-Cloud_%2B_Apple_Silicon_(MLX%2FOllama)-orange.svg)]()
[![Code Style: Pydantic v2](https://img.shields.io/badge/schema-Pydantic_v2-green.svg)](https://docs.pydantic.dev/)

**Aegis** is an enterprise-grade **Agentic Statistical Evaluation, Protocol Profiling, and FinOps Governance Control Plane** optimized for both frontier cloud APIs (OpenAI, Anthropic, Gemini) and local Apple Silicon (MLX / Ollama) execution.

Repository: `https://github.com/promptly-broken/aegis-agent-governance`

---

## Key Pillars

1. **`aegis.core` — Trajectory DAG & Telemetry Representation**
   - Strictly typed execution spans and tool nodes via `Pydantic v2` and `networkx`.
   - Cycle detection, topological serialization, and multi-tier token/latency accounting.
   - Comprehensive pricing registry supporting frontier cloud APIs and local Apple Silicon (MLX / Ollama) amortized hardware & electricity TCO.

2. **`aegis.eval_engine` — Statistical Trajectory Evaluation Beyond LLM-as-a-Judge**
   - **Trajectory Determinism Index (TDI):** Quantifies structural and schema stability of agent tool-call DAGs across $K$ Monte Carlo rollouts:
     $$TDI = 1 - \frac{1}{\binom{K}{2}} \sum_{i < j} \frac{d_{\text{edit}}(\tau_i, \tau_j)}{\max(|\tau_i|, |\tau_j|)}$$
   - **Sequential Probability Ratio Testing (SPRT):** Implements Wald's SPRT to terminate CI/CD evaluation loops early with statistical power guarantees ($\alpha=0.05, \beta=0.10$).

3. **`aegis.finops` — Unit Economics, ECRT & Hybrid Arbitrage**
   - **Effective Cost per Resolved Trajectory (ECRT):** Computes true business unit economics factoring in empirical task resolution probability:
     $$ECRT = \frac{\sum_{m \in \mathcal{M}} (T_{\text{in}, m} \cdot c_{\text{in}, m} + T_{\text{out}, m} \cdot c_{\text{out}, m}) + C_{\text{infra}}}{\mathbb{P}(\text{Task Success})}$$
   - **Runaway Loop Circuit Breaker:** Detects cyclic tool-call patterns and halts runaway token consumption before SLA budgets are breached.
   - **Hybrid Edge/Cloud Complexity Router:** Automatically offloads deterministic sub-tasks to local Apple Silicon engines while preserving frontier cloud tokens for high-entropy reasoning.

4. **`aegis.mcp_profiler` — Stateful vs. Stateless MCP Telemetry**
   - Benchmarks Token Amplification Factor (TAF) and serialization overhead between stateless context re-transmission and Redis-backed session persistence.

5. **`aegis.pytest_plugin` — Automated CI/CD Regression Gates**
   - Decorate async agent tests with `@assert_agent_governance(min_tdi=0.85, max_ecrt_usd=0.015, max_loops=2)` to block regressions in GitHub Actions.

6. **`aegis.integrations` — Framework Middleware & Auto-Instrumentation**
   - Drop-in callbacks and interceptors for **LangGraph** (`LangGraphGovernanceCallback`), **OpenAI Agents SDK** (`OpenAIAgentsGovernanceHooks`), **MCP Client Sessions** (`GovernedMCPSessionProxy`), and function tools (`@governed_tool`).

7. **`aegis.exporters` — OpenTelemetry (OTLP/JSON) & JSONL Audit Exporters**
   - Export governed `TrajectoryDAG` traces, TDI scores, and ECRT metrics to **OpenTelemetry (`OTLPTraceExporter`)** (compatible with Arize Phoenix, Langfuse, and Jaeger) and append-only **JSONL compliance logs (`JSONLAuditExporter`)**.

---

## Directory Layout

```text
aegis-agent-governance/
├── .github/
│   └── workflows/
│       └── agent-governance-gate.yml         # GitHub Actions governance gate
├── docs/
│   └── rfcs/
│       ├── 001-tdi-vs-pass-at-k.md           # Deep dive: TDI vs Pass@k
│       ├── 002-stateful-vs-stateless-mcp-tco.md # MCP architecture analysis
│       └── 003-hybrid-edge-cloud-finops.md   # Apple Silicon MLX/Ollama TCO
├── src/
│   └── aegis/
│       ├── __init__.py                       # Package exports and versioning
│       ├── core/
│       │   ├── __init__.py
│       │   ├── models.py                     # Pydantic v2 Trace, Span, ToolNode, TrajectoryDAG
│       │   └── pricing.py                    # Token cost registry (Cloud APIs + Apple Silicon TCO)
│       ├── eval_engine/
│       │   ├── __init__.py
│       │   ├── tdi.py                        # Trajectory Determinism Index
│       │   └── sprt.py                       # Wald's SPRT early-stopping
│       ├── finops/
│       │   ├── __init__.py
│       │   ├── ecrt.py                       # Effective Cost per Resolved Trajectory
│       │   ├── circuit_breaker.py            # Cycle detection & SLA budget governor
│       │   └── router.py                     # Local MLX/Ollama vs Frontier Cloud arbitrage
│       ├── mcp_profiler/
│       │   ├── __init__.py
│       │   └── amplification.py              # Stateless vs Stateful MCP TAF auditor
│       └── pytest_plugin/
│           ├── __init__.py
│           └── decorator.py                  # @assert_agent_governance decorator
├── benchmarks/
│   ├── scenarios/
│   │   └── marketplace_dispute_resolution.py # Multi-agent commerce dispute benchmark
│   └── generate_executive_report.py          # Markdown telemetry reporting
├── tests/
│   ├── test_tdi.py                           # TDI unit and property tests
│   ├── test_ecrt.py                          # ECRT calculation & edge-case tests
│   ├── test_sprt.py                          # SPRT hypothesis testing & ASN tests
│   └── test_circuit_breaker.py               # Cycle detection & breaker tests
├── pyproject.toml                            # Modern pyproject.toml with uv/hatchling
└── README.md                                 # Architecture, quickstart, and telemetry guide
```

---

## Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/promptly-broken/aegis-agent-governance.git
cd aegis-agent-governance

# Install with uv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Aegis Control Plane CLI

Aegis includes a built-in command-line tool `aegis` to inspect, profile, and evaluate agent trajectories:

```bash
# 1. Run the Multi-Agent Marketplace Dispute Benchmark and print the Executive Report
uv run aegis report

# 2. Generate the Interactive HTML Governance & FinOps Dashboard
uv run aegis dashboard -o aegis_dashboard.html

# 2. Audit Model Context Protocol (MCP) Token Amplification Factor (TAF)
uv run aegis profile-mcp --turns 5 --payload-bytes 1024

# 3. Interactively score task entropy and route prompt to optimal tier (Apple Silicon vs Cloud)
uv run aegis route "Extract customer ID and order number from JSON schema"

# 4. Enforce CI/CD governance gate on a recorded trajectory file
uv run aegis eval -f path/to/trajectory.json --min-tdi 0.85 --max-ecrt 0.05 --strict

# 5. Export trajectories to OpenTelemetry (OTLP/JSON) or JSONL audit trails
uv run aegis export -f path/to/trajectory.json --format otlp -o traces_otlp.json
uv run aegis export -f path/to/trajectory.json --format jsonl -o audit_log.jsonl
```

### Runtime Telemetry & Circuit Breaker Usage

```python
from aegis.core.tracer import AegisTracer
from aegis.finops.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerTrip

# 1. Configure circuit breaker with runaway cycle and SLA budget caps
config = CircuitBreakerConfig(max_cost_usd=0.05, max_cycle_repetitions=2)
tracer = AegisTracer(scenario_id="customer_dispute", circuit_breaker=CircuitBreaker(config))

# 2. Execute and trace agent actions
with tracer:
    with tracer.span("reasoning_step") as s:
        s.record_tokens(prompt_tokens=450, completion_tokens=120)

    # Automatically guarded: repeating tool cycles trips CircuitBreakerTrip
    t1 = tracer.record_tool_call("db_query", arguments={"user": 42})
    t2 = tracer.record_tool_call("api_lookup", depends_on=[t1.node_id])

    tracer.mark_resolved(True)

# 3. Access finalized Trajectory DAG with exact NetworkX representation & cost
dag = tracer.get_trajectory()
print(f"Total Trajectory Cost: ${dag.total_cost_usd():.5f}")
```

### Live End-to-End Demo

Execute the live demonstration scenario showcasing real-time circuit breaker trip interception, hybrid router arbitrage, and MCP pointer persistence:

```bash
uv run python examples/live_governed_agent.py
```

### Running the Test Suite

```bash
# Run all 84 comprehensive unit and property tests
uv run pytest -v

# Run with test coverage
uv run pytest --cov=aegis
```

---

## License

Apache-2.0 License. See LICENSE for details.

