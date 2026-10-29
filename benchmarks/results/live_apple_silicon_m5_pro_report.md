# Aegis Live Apple Silicon Hardware Benchmark Report

**Timestamp (UTC):** 2026-10-29T07:34:23.972739+00:00  
**Hardware Substrate:** Apple M5 Pro (48 GB Unified Memory) — macOS 26.6.2 (`arm64`)  
**Inference Runtime:** Local Ollama REST Engine (`http://127.0.0.1:11434`)  

---

## 1. Live Trajectory Determinism Index (TDI) — Temperature Sensitivity ($K=5$)

| Cohort Configuration | Model | Temp ($T$) | Mean Speed | Schema-Key TDI | Full Semantic TDI (Keys + Values) | Gate Status ($\ge 0.85$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Governed Deterministic** | `qwen2.5-coder:7b-instruct` | `0.0` | 58.0 tok/s | 1.0000 | **1.0000** | **PASSED** |
| **Unconstrained Stochastic** | `qwen2.5-coder:7b-instruct` | `0.8` | 58.3 tok/s | 0.6156 | **0.6189** | **FAILED** |

---

## 2. Live Unit Economics & ECRT: Apple M5 Pro vs. Frontier Cloud (`gpt-4o`)

| Metric | Local Apple M5 Pro (`qwen2.5-coder:7b-instruct`) | Frontier Cloud Equivalent (`gpt-4o`) |
|:---|:---|:---|
| **Batch Expenditure (5 Rollouts)** | **$0.000434** (Hardware CapEx + 28W Power TCO) | $0.010513 |
| **Effective Cost / Resolved Trajectory (ECRT)** | **$0.000087** | $0.002103 |
| **Empirical Resolution Rate** | 100.0% (5/5) | 100.0% (Reference) |
| **Net Unit Cost Reduction** | **95.9% TCO Reduction** | Baseline |

---

## 3. Multi-Model Apple Silicon Throughput & FinOps Arbitrage Matrix

| Local Ollama Model | Generation Speed | Total Latency | TTFT | Apple Silicon TCO | `gpt-4o` Cloud Cost | Cost Reduction | Schema Gate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `gemma3:270m` | **377.1 tok/s** | 710.6 ms | 22.3 ms | $0.000023 | $0.000705 | **96.8%** | VALID |
| `phi3:mini` | **104.7 tok/s** | 2735.8 ms | 60.2 ms | $0.000089 | $0.000852 | **89.6%** | VALID |
| `deepseek-coder:6.7b` | **64.1 tok/s** | 2537.3 ms | 188.6 ms | $0.000083 | $0.000992 | **91.6%** | VALID |
| `qwen2.5-coder:7b-instruct` | **58.7 tok/s** | 1866.8 ms | 149.1 ms | $0.000061 | $0.000770 | **92.0%** | VALID |

---

## 4. Exported Telemetry Artifacts
- **Markdown Report:** `benchmarks/results/live_apple_silicon_m5_pro_report.md`
- **Interactive HTML Dashboard:** `benchmarks/results/live_apple_silicon_dashboard.html`
- **OpenTelemetry (OTLP/JSON):** `benchmarks/results/live_apple_silicon_otlp.json`
- **Append-Only JSONL Audit Log:** `benchmarks/results/live_apple_silicon_trajectories.jsonl`
