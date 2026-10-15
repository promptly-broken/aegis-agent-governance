"""Aegis Pytest Plugin: CI/CD governance assertions for autonomous agents."""

from aegis.pytest_plugin.decorator import (
    GovernanceAssertionError,
    assert_agent_governance,
)

__all__ = [
    "assert_agent_governance",
    "GovernanceAssertionError",
]
