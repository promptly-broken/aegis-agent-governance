# RFC 002: Stateful vs. Stateless Model Context Protocol (MCP) TCO & Token Amplification

**Author:** Principal Distributed Systems & AI Infrastructure Architect  
**Status:** Accepted  
**Target:** `aegis.mcp_profiler.amplification`

---

## 1. Context & Architecture

Anthropic's Model Context Protocol (MCP) establishes a client-server architecture enabling LLM agents to access tools, prompts, and data resources. However, naive implementations treat MCP connections as **stateless transports**, re-transmitting the entire interaction history and resource states on every turn $t \in \{1, \dots, T\}$.

This RFC analyzes the Total Cost of Ownership (TCO) and introduces the **Token Amplification Factor (TAF)** metric.

---

## 2. Mathematical Modeling of Token Amplification

Let:
- $S_0$ be the base system prompt and tool definitions token count.
- $\delta_t$ be the token length of the payload exchanged at turn $t$.
- $p_{\text{ptr}}$ be the fixed token overhead of a session state pointer (e.g., 32 tokens).

### Stateless Mode
On turn $t$, the context window must consume the cumulative sum of all prior interactions:
$$K_{\text{stateless}}(t) = S_0 + \sum_{k=1}^t \delta_k$$
Total cumulative tokens consumed over a $T$-turn trajectory:
$$T_{\text{stateless}} = \sum_{t=1}^T K_{\text{stateless}}(t) = T \cdot S_0 + \sum_{t=1}^T (T - t + 1) \delta_t$$

### Redis-Backed Stateful Mode
By utilizing a distributed state store (e.g. Redis) and passing ephemeral session pointers, the agent only transmits the current delta:
$$K_{\text{stateful}}(t) = S_0 + p_{\text{ptr}} + \delta_t$$
Total cumulative tokens:
$$T_{\text{stateful}} = T \cdot (S_0 + p_{\text{ptr}}) + \sum_{t=1}^T \delta_t$$

### Token Amplification Factor (TAF)
$$TAF = \frac{T_{\text{stateless}}}{T_{\text{stateful}}}$$

For deep multi-turn agents ($T \ge 15$), $TAF$ frequently exceeds $6.0\times$ to $12.0\times$, resulting in an exponential cost explosion and context-length degradation.

---

## 3. Protocol Telemetry & Profiling in Aegis

`aegis.mcp_profiler.amplification` instruments both paradigms to provide engineering teams with:
1. Exact per-turn token diffing.
2. Latency impact of Redis session hydration vs. LLM prefill processing time.
3. Quantified dollar savings under cloud token pricing registries.
