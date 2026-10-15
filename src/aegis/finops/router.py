"""Hybrid Complexity Router: Arbitrage sub-tasks between local Apple Silicon (MLX/Ollama) and Frontier Cloud."""

from __future__ import annotations

from enum import Enum
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from aegis.core.models import ModelTier


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

    def estimate_complexity(self, prompt: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[TaskComplexity, float]:
        """Heuristically or structurally score task entropy.

        Returns:
            (TaskComplexity, entropy_score between 0.0 and 1.0)
        """
        meta = metadata or {}
        task_type = meta.get("task_type", "").lower()

        if task_type in ("extraction", "validation", "json_format", "schema_check"):
            return (TaskComplexity.DETERMINISTIC_LOW, 0.15)
        if task_type in ("dispute_resolution", "negotiation", "multi_agent_arbitration"):
            return (TaskComplexity.REASONING_HIGH, 0.88)

        # Content regex analysis
        high_matches = sum(1 for p in self._high_reasoning_patterns if p.search(prompt))
        det_matches = sum(1 for p in self._deterministic_patterns if p.search(prompt))

        if high_matches > det_matches:
            entropy = min(0.95, 0.60 + (0.10 * high_matches))
            return (TaskComplexity.REASONING_HIGH, entropy)
        elif det_matches > 0:
            entropy = max(0.10, 0.40 - (0.08 * det_matches))
            return (TaskComplexity.DETERMINISTIC_LOW, entropy)

        # Word length / context heuristic
        words = len(prompt.split())
        if words > 1000:
            return (TaskComplexity.SUMMARIZATION_MID, 0.50)

        return (TaskComplexity.DETERMINISTIC_LOW, 0.35)

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
