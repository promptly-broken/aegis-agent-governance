"""Pytest plugin and governance decorator to enforce TDI, ECRT, and loop SLAs in CI/CD."""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any, Callable, Coroutine, List, Optional, Union

from aegis.core.models import TrajectoryDAG
from aegis.eval_engine.tdi import compute_tdi
from aegis.finops.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from aegis.finops.ecrt import compute_ecrt


class GovernanceAssertionError(AssertionError):
    """Raised when an agent test violates governance thresholds."""
    pass


def assert_agent_governance(
    min_tdi: Optional[float] = 0.85,
    max_ecrt_usd: Optional[float] = 0.02,
    max_loops: int = 2,
    rollouts: int = 1,
) -> Callable:
    """Governance decorator enforcing statistical determinism, FinOps SLA, and cycle limits.

    Can be applied to both synchronous and asynchronous pytest test functions.
    The decorated test function must return either a TrajectoryDAG or a List[TrajectoryDAG].
    """
    def decorator(func: Callable) -> Callable:
        is_async = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            collected_trajectories: List[TrajectoryDAG] = []

            # Execute rollouts
            for _ in range(max(1, rollouts)):
                res = await func(*args, **kwargs)
                if isinstance(res, list) and all(isinstance(x, TrajectoryDAG) for x in res):
                    collected_trajectories.extend(res)
                    break
                elif isinstance(res, TrajectoryDAG):
                    collected_trajectories.append(res)
                else:
                    raise TypeError(
                        f"Function '{func.__name__}' decorated with @assert_agent_governance "
                        f"must return TrajectoryDAG or List[TrajectoryDAG], got {type(res)}."
                    )

            _evaluate_governance(collected_trajectories, min_tdi, max_ecrt_usd, max_loops)
            return collected_trajectories if len(collected_trajectories) > 1 else collected_trajectories[0]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            collected_trajectories: List[TrajectoryDAG] = []

            for _ in range(max(1, rollouts)):
                res = func(*args, **kwargs)
                if isinstance(res, list) and all(isinstance(x, TrajectoryDAG) for x in res):
                    collected_trajectories.extend(res)
                    break
                elif isinstance(res, TrajectoryDAG):
                    collected_trajectories.append(res)
                else:
                    raise TypeError(
                        f"Function '{func.__name__}' decorated with @assert_agent_governance "
                        f"must return TrajectoryDAG or List[TrajectoryDAG], got {type(res)}."
                    )

            _evaluate_governance(collected_trajectories, min_tdi, max_ecrt_usd, max_loops)
            return collected_trajectories if len(collected_trajectories) > 1 else collected_trajectories[0]

        return async_wrapper if is_async else sync_wrapper

    return decorator


def _evaluate_governance(
    trajectories: List[TrajectoryDAG],
    min_tdi: Optional[float],
    max_ecrt_usd: Optional[float],
    max_loops: int,
) -> None:
    """Perform governance assertions on the collected trajectories."""
    if not trajectories:
        raise GovernanceAssertionError("No trajectories returned from test execution.")

    # 1. Loop and cycle detection
    breaker = CircuitBreaker(CircuitBreakerConfig(max_cycle_repetitions=max_loops))
    for idx, traj in enumerate(trajectories):
        is_ok, reason, explanation = breaker.inspect(traj)
        if not is_ok:
            raise GovernanceAssertionError(
                f"Trajectory[{idx}] failed loop governance: {reason.value if reason else 'violation'} - {explanation}"
            )

    # 2. TDI assertion (when >= 2 trajectories available)
    if min_tdi is not None and len(trajectories) >= 2:
        tdi_result = compute_tdi(trajectories, threshold=min_tdi)
        if tdi_result.tdi < min_tdi:
            raise GovernanceAssertionError(
                f"Trajectory Determinism Index regression: TDI {tdi_result.tdi:.4f} < required {min_tdi:.4f}."
            )

    # 3. ECRT FinOps assertion
    if max_ecrt_usd is not None:
        ecrt_result = compute_ecrt(trajectories, max_budget_usd=max_ecrt_usd)
        if ecrt_result.ecrt_usd > max_ecrt_usd:
            raise GovernanceAssertionError(
                f"FinOps SLA breach: ECRT ${ecrt_result.ecrt_usd:.4f} > maximum budget SLA ${max_ecrt_usd:.4f} "
                f"(Resolution rate: {ecrt_result.resolution_rate * 100:.1f}%)."
            )
