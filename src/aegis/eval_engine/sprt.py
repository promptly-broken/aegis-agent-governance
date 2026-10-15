"""Sequential Probability Ratio Testing (SPRT) for statistical early-stopping of agent evaluation."""

from __future__ import annotations

from enum import Enum
import math
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SPRTDecision(str, Enum):
    """SPRT decision outcomes."""

    ACCEPT_H1 = "accept_h1"  # Agent passes reliability threshold
    ACCEPT_H0 = "accept_h0"  # Agent fails reliability threshold
    CONTINUE = "continue"    # Continue sampling (indifference region)


class SPRTResult(BaseModel):
    """Telemetry and decision state from an SPRT evaluation rollout."""

    model_config = ConfigDict(extra="forbid")

    decision: SPRTDecision
    step: int = Field(..., ge=0)
    successes: int = Field(..., ge=0)
    failures: int = Field(..., ge=0)
    log_likelihood_ratio: float
    lower_bound_ln_b: float
    upper_bound_ln_a: float
    empirical_rate: float = Field(..., ge=0.0, le=1.0)
    passed: Optional[bool] = None


class SequentialProbabilityRatioTest:
    """Wald's Sequential Probability Ratio Test for Bernoulli evaluation trials.

    Hypotheses:
        H0: p <= p0 (Unacceptable agent reliability)
        H1: p >= p1 (Target production reliability, p1 > p0)

    Boundaries:
        A = (1 - beta) / alpha  ==> ln(A)
        B = beta / (1 - alpha)  ==> ln(B)
    """

    def __init__(
        self,
        p0: float = 0.70,
        p1: float = 0.90,
        alpha: float = 0.05,
        beta: float = 0.10,
        max_steps: Optional[int] = 100,
    ) -> None:
        if not (0.0 < p0 < p1 < 1.0):
            raise ValueError(f"Invalid parameters: required 0 < p0 < p1 < 1, got p0={p0}, p1={p1}")
        if not (0.0 < alpha < 1.0) or not (0.0 < beta < 1.0):
            raise ValueError(f"Significance levels must be in (0, 1): alpha={alpha}, beta={beta}")

        self.p0 = p0
        self.p1 = p1
        self.alpha = alpha
        self.beta = beta
        self.max_steps = max_steps

        # Stopping boundaries
        self.ln_a = math.log((1.0 - beta) / alpha)
        self.ln_b = math.log(beta / (1.0 - alpha))

        # Log likelihood increments
        self.delta_s = math.log(p1 / p0)
        self.delta_f = math.log((1.0 - p1) / (1.0 - p0))

        # Running state
        self.step_count = 0
        self.successes = 0
        self.failures = 0
        self.current_llr = 0.0

    def reset(self) -> None:
        """Reset SPRT state for a new test scenario."""
        self.step_count = 0
        self.successes = 0
        self.failures = 0
        self.current_llr = 0.0

    def step(self, success: bool) -> SPRTResult:
        """Process a single trajectory outcome and return SPRT decision."""
        self.step_count += 1
        if success:
            self.successes += 1
            self.current_llr += self.delta_s
        else:
            self.failures += 1
            self.current_llr += self.delta_f

        empirical_rate = self.successes / self.step_count

        # Check stopping boundaries
        if self.current_llr >= self.ln_a:
            decision = SPRTDecision.ACCEPT_H1
            passed = True
        elif self.current_llr <= self.ln_b:
            decision = SPRTDecision.ACCEPT_H0
            passed = False
        elif self.max_steps is not None and self.step_count >= self.max_steps:
            # Truncated SPRT tie-break
            if self.current_llr >= 0:
                decision = SPRTDecision.ACCEPT_H1
                passed = True
            else:
                decision = SPRTDecision.ACCEPT_H0
                passed = False
        else:
            decision = SPRTDecision.CONTINUE
            passed = None

        return SPRTResult(
            decision=decision,
            step=self.step_count,
            successes=self.successes,
            failures=self.failures,
            log_likelihood_ratio=round(self.current_llr, 5),
            lower_bound_ln_b=round(self.ln_b, 5),
            upper_bound_ln_a=round(self.ln_a, 5),
            empirical_rate=round(empirical_rate, 4),
            passed=passed,
        )

    def batch(self, outcomes: List[bool]) -> SPRTResult:
        """Evaluate a sequence of binary outcomes sequentially until decision or end."""
        last_result: Optional[SPRTResult] = None
        for outcome in outcomes:
            last_result = self.step(outcome)
            if last_result.decision != SPRTDecision.CONTINUE:
                break

        if last_result is None:
            return SPRTResult(
                decision=SPRTDecision.CONTINUE,
                step=0,
                successes=0,
                failures=0,
                log_likelihood_ratio=0.0,
                lower_bound_ln_b=round(self.ln_b, 5),
                upper_bound_ln_a=round(self.ln_a, 5),
                empirical_rate=0.0,
                passed=None,
            )
        return last_result

    def expected_sample_size(self, p: Optional[float] = None) -> float:
        """Calculate theoretical Average Sample Number (ASN) at probability p.

        If p is None, calculates ASN at p = p1.
        """
        eval_p = p if p is not None else self.p1
        denom = (eval_p * self.delta_s) + ((1.0 - eval_p) * self.delta_f)

        if abs(denom) < 1e-9:
            # Indifference point where denom is near zero: approximate via L'Hopital
            return (self.ln_a * abs(self.ln_b)) / (self.delta_s * abs(self.delta_f))

        # Expected value of LLR numerator
        if abs(eval_p - self.p1) < 1e-5:
            # Under H1
            numerator = (self.beta * self.ln_b) + ((1.0 - self.beta) * self.ln_a)
        elif abs(eval_p - self.p0) < 1e-5:
            # Under H0
            numerator = ((1.0 - self.alpha) * self.ln_b) + (self.alpha * self.ln_a)
        else:
            # General approximation
            numerator = ((1.0 - self.alpha) * self.ln_b) + (self.alpha * self.ln_a)

        return max(1.0, abs(numerator / denom))
