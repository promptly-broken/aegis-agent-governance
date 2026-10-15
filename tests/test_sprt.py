"""Unit tests for Wald's Sequential Probability Ratio Testing (SPRT)."""

import pytest

from aegis.eval_engine.sprt import (
    SPRTDecision,
    SequentialProbabilityRatioTest,
)


def test_sprt_invalid_parameters():
    with pytest.raises(ValueError):
        SequentialProbabilityRatioTest(p0=0.9, p1=0.7)  # p0 > p1

    with pytest.raises(ValueError):
        SequentialProbabilityRatioTest(alpha=1.5)

    with pytest.raises(ValueError):
        SequentialProbabilityRatioTest(beta=-0.1)


def test_sprt_early_accept_h1():
    # Consecutive successes should quickly accept H1 (agent passes)
    sprt = SequentialProbabilityRatioTest(p0=0.70, p1=0.90, alpha=0.05, beta=0.10)
    decision = SPRTDecision.CONTINUE

    for _ in range(25):
        res = sprt.step(success=True)
        if res.decision != SPRTDecision.CONTINUE:
            decision = res.decision
            break

    assert decision == SPRTDecision.ACCEPT_H1
    assert res.passed is True
    assert res.empirical_rate == 1.0


def test_sprt_early_accept_h0():
    # Consecutive failures should quickly accept H0 (agent fails)
    sprt = SequentialProbabilityRatioTest(p0=0.70, p1=0.90, alpha=0.05, beta=0.10)
    decision = SPRTDecision.CONTINUE

    for _ in range(20):
        res = sprt.step(success=False)
        if res.decision != SPRTDecision.CONTINUE:
            decision = res.decision
            break

    assert decision == SPRTDecision.ACCEPT_H0
    assert res.passed is False
    assert res.empirical_rate == 0.0


def test_sprt_batch_evaluation():
    sprt = SequentialProbabilityRatioTest(p0=0.70, p1=0.90, alpha=0.05, beta=0.10)
    # Stream with failures first
    outcomes = [False, False, False, False, False, False]
    res = sprt.batch(outcomes)
    assert res.decision == SPRTDecision.ACCEPT_H0


def test_sprt_reset():
    sprt = SequentialProbabilityRatioTest()
    sprt.step(True)
    sprt.step(True)
    assert sprt.step_count == 2
    sprt.reset()
    assert sprt.step_count == 0
    assert sprt.successes == 0
    assert sprt.current_llr == 0.0


def test_sprt_expected_sample_size():
    sprt = SequentialProbabilityRatioTest(p0=0.70, p1=0.90)
    asn_h1 = sprt.expected_sample_size(p=0.90)
    asn_h0 = sprt.expected_sample_size(p=0.70)

    assert asn_h1 > 0
    assert asn_h0 > 0
    # Expected samples under both hypotheses should be substantially less than 100
    assert asn_h1 < 50
    assert asn_h0 < 50


def test_sprt_truncation_tie_breaker():
    # Force max_steps to 3
    sprt = SequentialProbabilityRatioTest(p0=0.70, p1=0.90, max_steps=3)
    sprt.step(True)
    sprt.step(True)
    res = sprt.step(False)
    # At step 3 (max_steps), decision must be reached (not CONTINUE)
    assert res.decision in (SPRTDecision.ACCEPT_H1, SPRTDecision.ACCEPT_H0)
