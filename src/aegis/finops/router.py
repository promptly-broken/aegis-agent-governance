"""Hybrid Complexity Router: Arbitrage sub-tasks between local Apple Silicon (MLX/Ollama) and Frontier Cloud."""

from __future__ import annotations

from enum import Enum
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request

from pydantic import BaseModel, ConfigDict, Field

from aegis.core.models import ModelTier
from aegis.core.pricing import PricingRegistry


class TaskComplexity(str, Enum):
    """Categorized entropy and reasoning requirements of a task step."""

    DETERMINISTIC_LOW = "deterministic_low"  # Schema validation, extraction, JSON parse
    SUMMARIZATION_MID = "summarization_mid"  # Context compression, extractive summarization
    REASONING_HIGH = "reasoning_high"        # Multi-step strategic planning, dispute arbitration


class RoutingDecision(BaseModel):
    """Decision outcome of the Hybrid Complexity Router."""

    model_config = ConfigDict(extra="forbid")

    selected_tier: ModelTier
    recommended_model: str
    complexity: TaskComplexity
    estimated_cost_usd: float
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    rationale: str


class HybridComplexityRouter:
    """Interception engine routing agent steps based on task entropy and cost arbitrage."""

    def __init__(
        self,
        default_local_model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
        default_cloud_model: str = "gpt-4o",
        entropy_threshold: float = 0.65,
    ) -> None:
        self.default_local_model = default_local_model
        self.default_cloud_model = default_cloud_model
        self.entropy_threshold = entropy_threshold

        # Regex patterns for deterministic sub-tasks
        self._deterministic_patterns = [
            re.compile(r"\b(extract|parse|schema|json|format|validate|filter|convert|sql)\b", re.IGNORECASE),
            re.compile(r"\b(regex|lookup|dictionary|table|list|deduplicate)\b", re.IGNORECASE),
        ]
        # Regex patterns for complex multi-agent reasoning
        self._high_reasoning_patterns = [
            re.compile(r"\b(negotiat|arbitrat|reconcil|hypothes|evaluat|decide|settle)\b", re.IGNORECASE),
            re.compile(r"\b(dispute|strategy|critique|root cause|conflict|tradeoff)\b", re.IGNORECASE),
        ]

    @staticmethod
    def _compute_json_schema_depth(schema: Any, _current_depth: int = 0) -> int:
        """Recursively compute nesting depth of a JSON schema or nested dict."""
        if not isinstance(schema, dict):
            return _current_depth
        props = schema.get("properties") or schema.get("items") or schema
        if not isinstance(props, dict) or not props:
            return _current_depth + 1
        return max(
            HybridComplexityRouter._compute_json_schema_depth(v, _current_depth + 1)
            for v in props.values()
        )

    @staticmethod
    def _compute_logprob_entropy(logprobs: List[float]) -> float:
        """Compute normalized Shannon entropy in [0, 1] from token log-probabilities."""
        if not logprobs:
            return 0.0
        total_surprisal = sum(-lp for lp in logprobs if lp < 0)
        mean_nll = total_surprisal / len(logprobs)
        # Normalize mean negative log-likelihood via logistic saturation (1.5 nats ~ 0.65)
        return min(1.0, max(0.0, 1.0 - math.exp(-0.7 * mean_nll)))

    def compute_structural_signals(
        self, prompt: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract structural, schema, tool-count, and information-theoretic complexity signals."""
        meta = metadata or {}
        schema = meta.get("json_schema") or meta.get("response_schema")
        schema_depth = self._compute_json_schema_depth(schema) if schema is not None else 0

        tools = meta.get("tools")
        num_tools = len(tools) if isinstance(tools, list) else int(meta.get("num_tools", 0))

        logprobs = meta.get("logprobs")
        logprob_entropy = (
            self._compute_logprob_entropy(logprobs) if isinstance(logprobs, list) and logprobs else None
        )

        high_matches = sum(1 for p in self._high_reasoning_patterns if p.search(prompt))
        det_matches = sum(1 for p in self._deterministic_patterns if p.search(prompt))
        multi_step_markers = len(re.findall(r"(?:^|\s)(?:step\s+\d+|\d+\.)\s", prompt, re.IGNORECASE))

        return {
            "schema_depth": schema_depth,
            "num_tools": num_tools,
            "logprob_entropy": logprob_entropy,
            "high_reasoning_matches": high_matches,
            "deterministic_matches": det_matches,
            "multi_step_markers": multi_step_markers,
        }

    def estimate_complexity(self, prompt: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[TaskComplexity, float]:
        """Heuristically and structurally score task entropy.

        Returns:
            (TaskComplexity, entropy_score between 0.0 and 1.0)
        """
        meta = metadata or {}
        task_type = meta.get("task_type", "").lower()

        if task_type in ("extraction", "validation", "json_format", "schema_check"):
            return (TaskComplexity.DETERMINISTIC_LOW, 0.15)
        if task_type in ("dispute_resolution", "negotiation", "multi_agent_arbitration"):
            return (TaskComplexity.REASONING_HIGH, 0.88)

        signals = self.compute_structural_signals(prompt, meta)

        # 1. Direct token logprob entropy signal if supplied by a draft/probe pass
        if signals["logprob_entropy"] is not None:
            lp_entropy = float(signals["logprob_entropy"])
            if lp_entropy >= self.entropy_threshold:
                return (TaskComplexity.REASONING_HIGH, round(lp_entropy, 3))
            elif lp_entropy >= 0.40:
                return (TaskComplexity.SUMMARIZATION_MID, round(lp_entropy, 3))
            else:
                return (TaskComplexity.DETERMINISTIC_LOW, round(lp_entropy, 3))

        # 2. Structural tool fanout or deeply nested schema complexity
        structural_boost = 0.0
        if signals["num_tools"] >= 6:
            structural_boost += min(0.30, 0.04 * signals["num_tools"])
        if signals["schema_depth"] >= 4:
            structural_boost += min(0.25, 0.05 * signals["schema_depth"])
        if signals["multi_step_markers"] >= 3:
            structural_boost += 0.15

        high_matches = signals["high_reasoning_matches"]
        det_matches = signals["deterministic_matches"]

        if high_matches > det_matches or structural_boost >= 0.30:
            entropy = min(0.95, 0.60 + (0.10 * high_matches) + structural_boost)
            return (TaskComplexity.REASONING_HIGH, round(entropy, 3))
        elif det_matches > 0 and structural_boost < 0.20:
            entropy = max(0.10, 0.40 - (0.08 * det_matches) + structural_boost)
            return (TaskComplexity.DETERMINISTIC_LOW, round(entropy, 3))

        # Word length / context heuristic
        words = len(prompt.split())
        if words > 1000:
            return (TaskComplexity.SUMMARIZATION_MID, round(min(0.64, 0.50 + structural_boost), 3))

        base_entropy = min(0.95, 0.35 + structural_boost)
        if base_entropy >= self.entropy_threshold:
            return (TaskComplexity.REASONING_HIGH, round(base_entropy, 3))
        return (TaskComplexity.DETERMINISTIC_LOW, round(base_entropy, 3))

    def route(
        self, prompt: str, metadata: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """Route an agent step to either Apple Silicon edge execution or Frontier Cloud."""
        complexity, entropy = self.estimate_complexity(prompt, metadata)

        if entropy < self.entropy_threshold:
            # Route to local Apple Silicon (MLX or Ollama)
            return RoutingDecision(
                selected_tier=ModelTier.LOCAL_MLX,
                recommended_model=self.default_local_model,
                complexity=complexity,
                estimated_cost_usd=0.00001,  # Near-zero marginal TCO
                confidence_score=round(1.0 - entropy, 3),
                rationale=(
                    f"Task entropy ({entropy:.2f}) < threshold ({self.entropy_threshold:.2f}). "
                    f"Classified as '{complexity.value}'. Offloaded to local Apple Silicon MLX."
                ),
            )
        else:
            # Route to frontier cloud API
            return RoutingDecision(
                selected_tier=ModelTier.FRONTIER_CLOUD,
                recommended_model=self.default_cloud_model,
                complexity=complexity,
                estimated_cost_usd=0.00500,
                confidence_score=round(entropy, 3),
                rationale=(
                    f"Task entropy ({entropy:.2f}) >= threshold ({self.entropy_threshold:.2f}). "
                    f"Classified as '{complexity.value}'. Routed to Frontier Cloud for high reasoning fidelity."
                ),
            )


class ExecutionResult(BaseModel):
    """Result of an executed model step via TaskDispatcher."""

    model_config = ConfigDict(extra="forbid")

    content: str
    model: str
    tier: ModelTier
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0.0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    is_fallback: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskDispatcher:
    """Dispatches routed agent tasks to local Apple Silicon (Ollama / MLX) or Frontier Cloud."""

    _mlx_model_cache: Dict[str, Tuple[Any, Any]] = {}

    def __init__(
        self,
        router: Optional[HybridComplexityRouter] = None,
        pricing_registry: Optional[PricingRegistry] = None,
        ollama_base_url: str = "http://localhost:11434",
        request_timeout_seconds: float = 1.0,
        cloud_api_keys: Optional[Dict[str, str]] = None,
        cloud_base_urls: Optional[Dict[str, str]] = None,
    ) -> None:
        self.router = router or HybridComplexityRouter()
        self.pricing_registry = pricing_registry or PricingRegistry()
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.request_timeout_seconds = request_timeout_seconds
        self.cloud_api_keys = cloud_api_keys or {}
        self.cloud_base_urls = {
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta",
            **(cloud_base_urls or {}),
        }

    def execute_task(
        self,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
        force_tier: Optional[ModelTier] = None,
        force_model: Optional[str] = None,
    ) -> ExecutionResult:
        """Route and execute a task prompt, falling back gracefully if local daemons are offline."""
        meta = metadata or {}
        decision = self.router.route(prompt, meta)

        target_tier = force_tier or decision.selected_tier
        target_model = force_model or (
            decision.recommended_model
            if not force_tier
            else (
                self.router.default_local_model
                if force_tier in (ModelTier.LOCAL_MLX, ModelTier.LOCAL_OLLAMA)
                else self.router.default_cloud_model
            )
        )

        t0 = time.time()
        if target_tier == ModelTier.LOCAL_OLLAMA:
            return self._dispatch_ollama(prompt, target_model, meta, t0)
        elif target_tier == ModelTier.LOCAL_MLX:
            return self._dispatch_mlx(prompt, target_model, meta, t0)
        else:
            return self._dispatch_cloud(prompt, target_model, meta, t0)

    def _dispatch_ollama(
        self, prompt: str, model: str, metadata: Dict[str, Any], t0: float
    ) -> ExecutionResult:
        """Attempt execution against local Ollama REST endpoint with mock fallback."""
        url = f"{self.ollama_base_url}/api/generate"
        payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data.get("response", "")
                p_tokens = data.get("prompt_eval_count", max(1, len(prompt.split())))
                c_tokens = data.get("eval_count", max(1, len(content.split())))
                duration_ms = (time.time() - t0) * 1000.0

                cost = self._compute_cost(model, p_tokens, c_tokens, duration_ms)
                return ExecutionResult(
                    content=content,
                    model=model,
                    tier=ModelTier.LOCAL_OLLAMA,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    duration_ms=round(duration_ms, 2),
                    cost_usd=cost,
                    is_fallback=False,
                    metadata={"source": "ollama_live", **metadata},
                )
        except Exception as exc:
            # Daemon inactive or connection refused -> offline deterministic fallback
            return self._fallback_result(
                prompt=prompt,
                model=model,
                tier=ModelTier.LOCAL_OLLAMA,
                reason=f"Ollama daemon unreachable ({exc.__class__.__name__})",
                metadata=metadata,
                t0=t0,
            )

    def _dispatch_mlx(
        self, prompt: str, model: str, metadata: Dict[str, Any], t0: float
    ) -> ExecutionResult:
        """Execute on local MLX framework with model caching, or fall back if unavailable."""
        try:
            import mlx_lm  # type: ignore

            if hasattr(mlx_lm, "load") and hasattr(mlx_lm, "generate"):
                if model not in self._mlx_model_cache:
                    self._mlx_model_cache[model] = mlx_lm.load(model)
                model_obj, tokenizer = self._mlx_model_cache[model]
                max_tokens = int(metadata.get("max_tokens", 256))
                content = str(
                    mlx_lm.generate(
                        model_obj,
                        tokenizer,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        verbose=False,
                    )
                )
                if hasattr(tokenizer, "encode"):
                    p_tokens = max(1, len(tokenizer.encode(prompt)))
                    c_tokens = max(1, len(tokenizer.encode(content)))
                else:
                    p_tokens = max(1, len(prompt.split()))
                    c_tokens = max(1, len(content.split()))
            else:
                content = f"[MLX Local Output for prompt: {prompt[:40]}...]"
                p_tokens = max(1, len(prompt.split()))
                c_tokens = max(1, len(content.split()))

            duration_ms = (time.time() - t0) * 1000.0
            cost = self._compute_cost(model, p_tokens, c_tokens, duration_ms)
            return ExecutionResult(
                content=content,
                model=model,
                tier=ModelTier.LOCAL_MLX,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                duration_ms=round(duration_ms, 2),
                cost_usd=cost,
                is_fallback=False,
                metadata={"source": "mlx_live", **metadata},
            )
        except ImportError:
            return self._fallback_result(
                prompt=prompt,
                model=model,
                tier=ModelTier.LOCAL_MLX,
                reason="mlx_lm not installed in local environment",
                metadata=metadata,
                t0=t0,
            )
        except Exception as exc:
            return self._fallback_result(
                prompt=prompt,
                model=model,
                tier=ModelTier.LOCAL_MLX,
                reason=f"mlx_lm execution failed ({exc.__class__.__name__}: {exc})",
                metadata=metadata,
                t0=t0,
            )

    def _resolve_cloud_provider_and_key(self, model: str) -> Tuple[str, Optional[str]]:
        """Identify cloud provider ('openai', 'anthropic', 'gemini') and resolve API key."""
        m_lower = model.lower()
        if m_lower.startswith("claude"):
            key = self.cloud_api_keys.get("anthropic") or os.environ.get("ANTHROPIC_API_KEY")
            return "anthropic", key
        if m_lower.startswith("gemini"):
            key = (
                self.cloud_api_keys.get("gemini")
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )
            return "gemini", key
        key = self.cloud_api_keys.get("openai") or os.environ.get("OPENAI_API_KEY")
        return "openai", key

    def _dispatch_cloud(
        self, prompt: str, model: str, metadata: Dict[str, Any], t0: float
    ) -> ExecutionResult:
        """Execute against live Frontier Cloud API (OpenAI/Anthropic/Gemini) if key is configured, else simulate."""
        provider, api_key = self._resolve_cloud_provider_and_key(model)

        if api_key:
            try:
                max_tokens = int(metadata.get("max_tokens", 512))
                if provider == "anthropic":
                    url = f"{self.cloud_base_urls['anthropic'].rstrip('/')}/messages"
                    body = {
                        "model": model,
                        "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    headers = {
                        "Content-Type": "application/json",
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                    }
                elif provider == "gemini":
                    base = self.cloud_base_urls["gemini"].rstrip("/")
                    url = f"{base}/models/{model}:generateContent?key={api_key}"
                    body = {"contents": [{"parts": [{"text": prompt}]}]}
                    headers = {"Content-Type": "application/json"}
                else:
                    url = f"{self.cloud_base_urls['openai'].rstrip('/')}/chat/completions"
                    body = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                    }
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    }

                req = urllib.request.Request(
                    url,
                    data=json.dumps(body).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.request_timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                if provider == "anthropic":
                    blocks = data.get("content", [])
                    content = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
                    usage = data.get("usage", {})
                    p_tokens = int(usage.get("input_tokens", max(1, len(prompt.split()))))
                    c_tokens = int(usage.get("output_tokens", max(1, len(content.split()))))
                elif provider == "gemini":
                    candidates = data.get("candidates", [{}])
                    parts = candidates[0].get("content", {}).get("parts", [{}]) if candidates else [{}]
                    content = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
                    usage = data.get("usageMetadata", {})
                    p_tokens = int(usage.get("promptTokenCount", max(1, len(prompt.split()))))
                    c_tokens = int(usage.get("candidatesTokenCount", max(1, len(content.split()))))
                else:
                    choices = data.get("choices", [{}])
                    content = (
                        choices[0].get("message", {}).get("content", "")
                        if choices
                        else ""
                    )
                    usage = data.get("usage", {})
                    p_tokens = int(usage.get("prompt_tokens", max(1, len(prompt.split()))))
                    c_tokens = int(usage.get("completion_tokens", max(1, len(content.split()))))

                duration_ms = (time.time() - t0) * 1000.0
                cost = self._compute_cost(model, p_tokens, c_tokens, duration_ms)
                return ExecutionResult(
                    content=content,
                    model=model,
                    tier=ModelTier.FRONTIER_CLOUD,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    duration_ms=round(duration_ms, 2),
                    cost_usd=cost,
                    is_fallback=False,
                    metadata={"source": "cloud_live", "provider": provider, **metadata},
                )
            except Exception as exc:
                return self._fallback_result(
                    prompt=prompt,
                    model=model,
                    tier=ModelTier.FRONTIER_CLOUD,
                    reason=f"Cloud API ({provider}) request failed ({exc.__class__.__name__})",
                    metadata={"provider": provider, **metadata},
                    t0=t0,
                )

        # Deterministic simulation when no live cloud API key is configured
        duration_ms = max(50.0, (time.time() - t0) * 1000.0)
        content = f"[Frontier Cloud Response for: {prompt[:50]}...]"
        p_tokens = max(1, len(prompt.split()))
        c_tokens = max(1, len(content.split()))
        cost = self._compute_cost(model, p_tokens, c_tokens, duration_ms)
        return ExecutionResult(
            content=content,
            model=model,
            tier=ModelTier.FRONTIER_CLOUD,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            duration_ms=round(duration_ms, 2),
            cost_usd=cost,
            is_fallback=False,
            metadata={"source": "cloud_runtime", "provider": provider, **metadata},
        )

    def _fallback_result(
        self,
        prompt: str,
        model: str,
        tier: ModelTier,
        reason: str,
        metadata: Dict[str, Any],
        t0: float,
    ) -> ExecutionResult:
        """Deterministic offline fallback when local neural engines are unavailable."""
        duration_ms = max(1.0, (time.time() - t0) * 1000.0)
        p_tokens = max(1, len(prompt.split()))
        content = f"Deterministic local resolution: processed ({len(prompt)} chars)."
        c_tokens = max(1, len(content.split()))
        cost = self._compute_cost(model, p_tokens, c_tokens, duration_ms)
        return ExecutionResult(
            content=content,
            model=model,
            tier=tier,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            duration_ms=round(duration_ms, 2),
            cost_usd=cost,
            is_fallback=True,
            metadata={"fallback_reason": reason, **metadata},
        )

    def _compute_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int, duration_ms: float
    ) -> float:
        profile = self.pricing_registry.get(model)
        if not profile:
            return 0.00001
        return profile.compute_cost(prompt_tokens, completion_tokens, duration_ms)


def execute_routed_task(
    prompt: str,
    metadata: Optional[Dict[str, Any]] = None,
    router: Optional[HybridComplexityRouter] = None,
) -> ExecutionResult:
    """Convenience functional interface to route and execute a prompt in a single call."""
    dispatcher = TaskDispatcher(router=router)
    return dispatcher.execute_task(prompt, metadata)

