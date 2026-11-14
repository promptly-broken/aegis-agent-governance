"""Token cost and hardware TCO registry for Cloud APIs and Apple Silicon local execution."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from aegis.core.models import ModelTier, Span, TrajectoryDAG


class ModelPricing(BaseModel):
    """Cost profile and physical TCO parameters for an LLM execution engine."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model_name: str
    tier: ModelTier
    input_cost_per_million: float = Field(default=0.0, ge=0.0)
    output_cost_per_million: float = Field(default=0.0, ge=0.0)
    cached_input_cost_per_million: float = Field(default=0.0, ge=0.0)
    fixed_cost_per_invocation: float = Field(default=0.0, ge=0.0)
    device_power_watts: float = Field(default=0.0, ge=0.0)
    kwh_cost_usd: float = Field(default=0.15, ge=0.0)
    amortized_hardware_cost_per_second: float = Field(default=0.0, ge=0.0)

    def compute_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: float = 0.0,
        cached_prompt_tokens: int = 0,
    ) -> float:
        """Compute the total economic and hardware cost for a given inference call."""
        cached_tokens = min(prompt_tokens, max(0, cached_prompt_tokens))
        uncached_tokens = max(0, prompt_tokens - cached_tokens)

        token_cost = (
            (uncached_tokens * (self.input_cost_per_million / 1_000_000.0))
            + (cached_tokens * (self.cached_input_cost_per_million / 1_000_000.0))
            + (completion_tokens * (self.output_cost_per_million / 1_000_000.0))
            + self.fixed_cost_per_invocation
        )

        infra_cost = 0.0
        if duration_ms > 0:
            duration_sec = duration_ms / 1000.0
            if self.device_power_watts > 0:
                kwh_consumed = (self.device_power_watts / 1000.0) * (duration_sec / 3600.0)
                infra_cost += kwh_consumed * self.kwh_cost_usd

            if self.amortized_hardware_cost_per_second > 0:
                infra_cost += duration_sec * self.amortized_hardware_cost_per_second

        return round(token_cost + infra_cost, 8)


class PricingRegistry:
    """Registry maintaining pricing configurations across Frontier Cloud and Local Apple Silicon."""

    def __init__(self, include_defaults: bool = True) -> None:
        self._registry: Dict[str, ModelPricing] = {}
        if include_defaults:
            self._load_default_profiles()

    def _load_default_profiles(self) -> None:
        # Frontier Cloud Providers
        defaults = [
            # OpenAI
            ModelPricing(
                model_name="gpt-4o",
                tier=ModelTier.FRONTIER_CLOUD,
                input_cost_per_million=2.50,
                output_cost_per_million=10.00,
                cached_input_cost_per_million=1.25,
            ),
            ModelPricing(
                model_name="gpt-4o-mini",
                tier=ModelTier.FRONTIER_CLOUD,
                input_cost_per_million=0.15,
                output_cost_per_million=0.60,
                cached_input_cost_per_million=0.075,
            ),
            ModelPricing(
                model_name="o1-preview",
                tier=ModelTier.FRONTIER_CLOUD,
                input_cost_per_million=15.00,
                output_cost_per_million=60.00,
                cached_input_cost_per_million=7.50,
            ),
            ModelPricing(
                model_name="o1-mini",
                tier=ModelTier.FRONTIER_CLOUD,
                input_cost_per_million=3.00,
                output_cost_per_million=12.00,
                cached_input_cost_per_million=1.50,
            ),
            ModelPricing(
                model_name="o3-mini",
                tier=ModelTier.FRONTIER_CLOUD,
                input_cost_per_million=1.10,
                output_cost_per_million=4.40,
                cached_input_cost_per_million=0.55,
            ),
            ModelPricing(
                model_name="o3",
                tier=ModelTier.FRONTIER_CLOUD,
                input_cost_per_million=2.00,
                output_cost_per_million=8.00,
                cached_input_cost_per_million=0.50,
            ),
            ModelPricing(
                model_name="gpt-4.1",
                tier=ModelTier.FRONTIER_CLOUD,
                input_cost_per_million=2.00,
                output_cost_per_million=8.00,
                cached_input_cost_per_million=0.50,
            ),
            ModelPricing(
                model_name="gpt-4.1-mini",
                tier=ModelTier.FRONTIER_CLOUD,
                input_cost_per_million=0.40,
                output_cost_per_million=1.60,
                cached_input_cost_per_million=0.10,
            ),
            # Anthropic
            ModelPricing(
                model_name="claude-3-5-sonnet-20241022",
                tier=ModelTier.FRONTIER_CLOUD,
                input_cost_per_million=3.00,
                output_cost_per_million=15.00,
                cached_input_cost_per_million=0.30,
            ),
            ModelPricing(
                model_name="claude-3-7-sonnet-latest",
                tier=ModelTier.FRONTIER_CLOUD,
                input_cost_per_million=3.00,
                output_cost_per_million=15.00,
                cached_input_cost_per_million=0.30,
            ),
            ModelPricing(
                model_name="claude-sonnet-4-20250514",
                tier=ModelTier.FRONTIER_CLOUD,
                input_cost_per_million=3.00,
                output_cost_per_million=15.00,
                cached_input_cost_per_million=0.30,
            ),
            ModelPricing(
                model_name="claude-opus-4-20250514",
                tier=ModelTier.FRONTIER_CLOUD,
                input_cost_per_million=15.00,
                output_cost_per_million=75.00,
                cached_input_cost_per_million=1.50,
            ),
            ModelPricing(
                model_name="claude-3-5-haiku-20241022",
                tier=ModelTier.FRONTIER_CLOUD,
                input_cost_per_million=0.80,
                output_cost_per_million=4.00,
                cached_input_cost_per_million=0.08,
            ),
            # Google Gemini
            ModelPricing(
                model_name="gemini-1.5-pro",
                tier=ModelTier.FRONTIER_CLOUD,
                input_cost_per_million=3.50,
                output_cost_per_million=10.50,
            ),
            ModelPricing(
                model_name="gemini-1.5-flash",
                tier=ModelTier.FRONTIER_CLOUD,
                input_cost_per_million=0.075,
                output_cost_per_million=0.30,
            ),
            ModelPricing(
                model_name="gemini-2.0-flash",
                tier=ModelTier.FRONTIER_CLOUD,
                input_cost_per_million=0.10,
                output_cost_per_million=0.40,
            ),
            ModelPricing(
                model_name="gemini-3.8-flash",
                tier=ModelTier.FRONTIER_CLOUD,
                input_cost_per_million=0.15,
                output_cost_per_million=0.60,
                cached_input_cost_per_million=0.0375,
            ),
            ModelPricing(
                model_name="gemini-flash-lite-latest",
                tier=ModelTier.FRONTIER_CLOUD,
                input_cost_per_million=0.075,
                output_cost_per_million=0.30,
                cached_input_cost_per_million=0.01875,
            ),
            # Apple Silicon Local Models (MLX) - e.g. M3/M4 Max active 30W, $3000 CapEx amortized over 3 yrs
            ModelPricing(
                model_name="mlx-community/Qwen2.5-7B-Instruct-4bit",
                tier=ModelTier.LOCAL_MLX,
                input_cost_per_million=0.0,
                output_cost_per_million=0.0,
                device_power_watts=28.0,
                kwh_cost_usd=0.15,
                amortized_hardware_cost_per_second=0.0000317,  # $3000 / (3 yrs * 365 days * 24 hrs * 3600 s)
            ),
            ModelPricing(
                model_name="mlx-community/Llama-3.1-8B-Instruct-4bit",
                tier=ModelTier.LOCAL_MLX,
                input_cost_per_million=0.0,
                output_cost_per_million=0.0,
                device_power_watts=32.0,
                kwh_cost_usd=0.15,
                amortized_hardware_cost_per_second=0.0000317,
            ),
            # Apple Silicon Local Models (Ollama)
            ModelPricing(
                model_name="ollama/llama3.2:3b",
                tier=ModelTier.LOCAL_OLLAMA,
                input_cost_per_million=0.0,
                output_cost_per_million=0.0,
                device_power_watts=18.0,
                kwh_cost_usd=0.15,
                amortized_hardware_cost_per_second=0.0000211,  # $2000 Mac Mini M4 Pro
            ),
            ModelPricing(
                model_name="ollama/mistral:7b",
                tier=ModelTier.LOCAL_OLLAMA,
                input_cost_per_million=0.0,
                output_cost_per_million=0.0,
                device_power_watts=30.0,
                kwh_cost_usd=0.15,
                amortized_hardware_cost_per_second=0.0000317,
            ),
            ModelPricing(
                model_name="qwen2.5-coder:7b-instruct",
                tier=ModelTier.LOCAL_OLLAMA,
                input_cost_per_million=0.0,
                output_cost_per_million=0.0,
                device_power_watts=28.0,
                kwh_cost_usd=0.15,
                amortized_hardware_cost_per_second=0.0000317,  # Apple M5 Pro 48GB ($3000 / 3 yrs)
            ),
            ModelPricing(
                model_name="deepseek-coder:6.7b",
                tier=ModelTier.LOCAL_OLLAMA,
                input_cost_per_million=0.0,
                output_cost_per_million=0.0,
                device_power_watts=26.0,
                kwh_cost_usd=0.15,
                amortized_hardware_cost_per_second=0.0000317,
            ),
            ModelPricing(
                model_name="phi3:mini",
                tier=ModelTier.LOCAL_OLLAMA,
                input_cost_per_million=0.0,
                output_cost_per_million=0.0,
                device_power_watts=20.0,
                kwh_cost_usd=0.15,
                amortized_hardware_cost_per_second=0.0000317,
            ),
            ModelPricing(
                model_name="gemma3:270m",
                tier=ModelTier.LOCAL_OLLAMA,
                input_cost_per_million=0.0,
                output_cost_per_million=0.0,
                device_power_watts=12.0,
                kwh_cost_usd=0.15,
                amortized_hardware_cost_per_second=0.0000317,
            ),
        ]
        for p in defaults:
            self.register(p)

    def register(self, pricing: ModelPricing) -> None:
        """Register or overwrite a model pricing profile."""
        self._registry[pricing.model_name.lower()] = pricing

    def get(self, model_name: str) -> Optional[ModelPricing]:
        """Lookup model pricing profile by exact or normalized name."""
        target = model_name.lower().strip()
        if target in self._registry:
            return self._registry[target]

        # Prefix/partial match fallback
        for name, profile in self._registry.items():
            if target.startswith(name) or name.startswith(target):
                return profile
        return None

    def calculate_span_cost(self, span: Span) -> float:
        """Calculate the total financial cost of a single span."""
        if not span.model_name:
            return span.cost_usd

        profile = self.get(span.model_name)
        if not profile:
            return span.cost_usd

        cached_tokens = span.attributes.get("cached_tokens", 0)
        return profile.compute_cost(
            prompt_tokens=span.prompt_tokens,
            completion_tokens=span.completion_tokens,
            duration_ms=span.duration_ms,
            cached_prompt_tokens=cached_tokens,
        )

    def calculate_trajectory_cost(
        self, trajectory: TrajectoryDAG, infra_overhead_usd: float = 0.0
    ) -> float:
        """Calculate aggregate cost of a trajectory including span inferences and infrastructure."""
        total = infra_overhead_usd
        for span in trajectory.spans:
            total += self.calculate_span_cost(span)
        return round(total, 6)
