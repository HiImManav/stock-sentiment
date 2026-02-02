"""Data classes for orchestration execution results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class AgentStatus(Enum):
    """Status of an agent execution."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"


class RouteType(Enum):
    """Type of routing decision."""

    NEWS_ONLY = "news_only"
    SEC_ONLY = "sec_only"
    BOTH = "both"


@dataclass
class AgentResult:
    """Result from executing a sub-agent.

    Attributes:
        agent_name: Name of the agent ('news_agent' or 'sec_agent').
        status: Execution status (success, timeout, or error).
        response: The agent's text response if successful.
        error_message: Error description if status is error or timeout.
        execution_time_ms: Time taken to execute in milliseconds.
    """

    agent_name: Literal["news_agent", "sec_agent"]
    status: AgentStatus
    response: str | None = None
    error_message: str | None = None
    execution_time_ms: float = 0.0

    @property
    def is_success(self) -> bool:
        """Check if the agent executed successfully."""
        return self.status == AgentStatus.SUCCESS

    @property
    def is_timeout(self) -> bool:
        """Check if the agent timed out."""
        return self.status == AgentStatus.TIMEOUT

    @property
    def is_error(self) -> bool:
        """Check if the agent encountered an error."""
        return self.status == AgentStatus.ERROR

    def to_dict(self) -> dict[str, str | float | None]:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "status": self.status.value,
            "response": self.response,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class QueryClassification:
    """Classification result for a user query.

    Attributes:
        route_type: Which agent(s) to route the query to.
        confidence: Confidence score for the classification (0.0 to 1.0).
        matched_patterns: List of patterns that matched the query.
        reasoning: Optional explanation for the classification decision.
    """

    route_type: RouteType
    confidence: float = 1.0
    matched_patterns: list[str] = field(default_factory=list)
    reasoning: str | None = None

    @property
    def needs_news_agent(self) -> bool:
        """Check if the news agent should be called."""
        return self.route_type in (RouteType.NEWS_ONLY, RouteType.BOTH)

    @property
    def needs_sec_agent(self) -> bool:
        """Check if the SEC agent should be called."""
        return self.route_type in (RouteType.SEC_ONLY, RouteType.BOTH)

    @property
    def needs_both(self) -> bool:
        """Check if both agents should be called."""
        return self.route_type == RouteType.BOTH
