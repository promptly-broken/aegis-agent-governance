# Aegis Frontier Cloud & Wire-Contract Verification Report

**Timestamp (UTC):** 2026-11-14T06:47:59.296982+00:00  
**Live Cloud Endpoint:** Google AI Studio REST API (`https://generativelanguage.googleapis.com/v1beta`)  
**Verified Cloud Model:** `gemini-flash-lite-latest` (`FRONTIER_CLOUD` Tier)  

---

## 1. Live Frontier Cloud Smoke Test Telemetry

| Metric | Live Cloud Result |
|:---|:---|
| **Provider / Source** | `gemini` (`cloud_live`) |
| **Model** | `gemini-flash-lite-latest` |
| **Prompt / Completion Tokens** | `20` in / `13` out |
| **Round-Trip Latency (IPv4 Preferred)** | **813.9 ms** |
| **Computed Inference Cost (USD)** | **$0.000005** |
| **Effective Cost / Resolved Trajectory (ECRT)** | **$0.000005** |
| **Trajectory Determinism Index (TDI)** | **1.0000** (`PASSED`, Stochastic Baseline: `0.2667`) |
| **Wald's SPRT Early-Stopping Gate** | **CONTINUE** (Terminated at rollout `#5`, 100.0% reliability) |
| **Raw JSON Payload Verified** | `{"status": "verified", "engine": "gemini"}` |

---

## 2. 3-Layer Frontier API Verification Matrix

| Verification Layer | Target Protocols & Models | Key Contracts Verified | Status |
|:---|:---|:---|:---:|
| **Layer 1: Local HTTP Wire-Contract Twin** (`tests/test_frontier_wire_contracts.py`) | OpenAI (`/v1/chat/completions`: `gpt-4o`, `gpt-4.1`, `o1-mini`, `o3-mini`), Anthropic (`/v1/messages`: `claude-3-5-sonnet-20241022`, `claude-sonnet-4-20250514`), Gemini (`:generateContent`) | Auth headers (`Bearer`, `x-api-key`, `anthropic-version`), `o1`/`o3` `max_completion_tokens`, OpenAI `content: null` tool calls, Anthropic `tool_use` blocks, cached-token FinOps discounts, HTTP `401`/`429`/`529` fallbacks | **PASSED** |
| **Layer 2: Apple Silicon Ollama `/v1` Neural Emulation** (`tests/test_ollama_cloud_emulation.py`) | Local Ollama `/v1` OpenAI & Anthropic wire compatibility (`gemma3:270m`, `qwen2.5-coder:7b-instruct`) | Live socket I/O + neural token generation through `TaskDispatcher._dispatch_cloud` | **PASSED** |
| **Layer 3: Live External Cloud Smoke Test** (`tests/test_cardless_cloud_smoke.py`) | Google AI Studio (`gemini-flash-lite-latest`) via `_IPv4PreferredHTTPSHandler` | Real TLS/HTTPS wire execution, token usage extraction, and ECRT accounting | **PASSED** |

---

## 3. Exported Artifacts
- **Markdown Report:** `benchmarks/results/live_frontier_cloud_smoke_report.md`
- **Interactive HTML Dashboard:** `benchmarks/results/live_frontier_cloud_dashboard.html`
- **OpenTelemetry (OTLP/JSON):** `benchmarks/results/live_frontier_cloud_otlp.json`
- **Append-Only JSONL Audit Log:** `benchmarks/results/live_frontier_cloud_trajectories.jsonl`
