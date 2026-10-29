"""Framework integrations and auto-instrumentation middleware."""

from aegis.integrations.middleware import (
    GovernedMCPSessionProxy,
    LangGraphGovernanceCallback,
    OpenAIAgentsGovernanceHooks,
    governed_tool,
)

__all__ = [
    "governed_tool",
    "LangGraphGovernanceCallback",
    "OpenAIAgentsGovernanceHooks",
    "GovernedMCPSessionProxy",
]

