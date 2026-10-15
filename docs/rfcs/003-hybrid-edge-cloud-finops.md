# RFC 003: Hybrid Edge-Cloud FinOps & Apple Silicon Hardware TCO Modeling

**Author:** Principal Distributed Systems & AI Infrastructure Architect  
**Status:** Accepted  
**Target:** `aegis.core.pricing` & `aegis.finops.router`

---

## 1. Motivation

Enterprise agent pipelines route 100% of tasks—regardless of complexity—to frontier cloud models (e.g., GPT-4o, Claude 3.5 Sonnet). Over 65% of intermediate tool-use turns consist of low-entropy operations:
- JSON schema normalization
- Extractive summarization
- Regex-based state filtering
- Parameter validation

Executing low-entropy operations on frontier cloud models generates unwarranted cost with zero quality advantage. Conversely, modern Apple Silicon hardware (M3/M4 Pro/Max) features Unified Memory Architectures (UMA) capable of running quantized 7B-14B models (e.g. Qwen 2.5 7B, Llama 3.1 8B via MLX or Ollama) at over 60 tokens/sec with negligible incremental costs.

---

## 2. Hardware TCO Formulation for Apple Silicon

Unlike cloud APIs with per-token pricing, local edge compute incurs a combined capital expenditure (CapEx) amortization and operational expenditure (OpEx) power cost:

$$C_{\text{local}}(\Delta t) = \left( \frac{\text{CapEx}}{L_{\text{sec}}} \cdot \Delta t \right) + \left( \frac{P_{\text{watts}}}{1000} \cdot \frac{\Delta t}{3600} \cdot R_{\text{kWh}} \right)$$

where:
- $\Delta t$: Active inference wall-clock duration in seconds.
- $\text{CapEx}$: Hardware procurement cost (e.g. $3,000 for Apple Silicon workstation).
- $L_{\text{sec}}$: Useful economic lifespan in seconds (e.g. 3 years $\approx 94,608,000$ seconds $\implies \approx \$0.0000317/\text{sec}$).
- $P_{\text{watts}}$: Average SoC package power draw under active neural engine / GPU load (25W - 35W).
- $R_{\text{kWh}}$: Commercial electricity rate (e.g. $\$0.15/\text{kWh}$).

### Unit Economics Comparison
- **Frontier Cloud (GPT-4o):** ~\$2.50 / MTok input + \$10.00 / MTok output. A 2,000-token extraction costs $\approx \$0.0065$.
- **Local Apple Silicon (MLX Qwen 2.5 7B):** 2,000 tokens generated in 30 seconds at 30W consumes $\approx 0.00025 \text{ kWh} \times \$0.15 = \$0.0000375$ power + $\$0.00095$ amortized hardware $\approx \$0.00099$.
- **Cost Reduction:** **84.8% unit cost reduction** per deterministic step.

---

## 3. Dynamic Arbitrage Routing

`aegis.finops.router.HybridComplexityRouter` implements real-time task entropy estimation:
- **Low Entropy ($H < 0.65$):** Schema extraction, formatting, JSON repair $\implies$ Routed to local Apple Silicon (MLX / Ollama).
- **High Entropy ($H \ge 0.65$):** Multi-agent negotiation, legal dispute analysis $\implies$ Routed to Frontier Cloud.
