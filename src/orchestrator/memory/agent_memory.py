"""Unified orchestrator memory for tracking multi-agent queries.

Stores orchestrated query results including user queries, routing decisions,
individual agent results, and synthesized responses for session context.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


@dataclass
class AgentResultEntry:
    """Stored result from a sub-agent execution.

    Attributes:
        agent_name: Name of the agent ('news_agent' or 'sec_agent').
        status: Execution status ('success', 'timeout', 'error').
        response: The agent's response text if successful.
        error_message: Error description if failed.
        execution_time_ms: Time taken in milliseconds.
    """

    agent_name: Literal["news_agent", "sec_agent"]
    status: Literal["success", "timeout", "error"]
    response: str | None = None
    error_message: str | None = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentResultEntry:
        """Create from dictionary."""
        return cls(
            agent_name=data["agent_name"],
            status=data["status"],
            response=data.get("response"),
            error_message=data.get("error_message"),
            execution_time_ms=data.get("execution_time_ms", 0.0),
        )


@dataclass
class OrchestratedQueryEntry:
    """A single orchestrated query record stored in memory.

    Attributes:
        query_id: Unique identifier for this query.
        user_query: The original user query.
        ticker: Optional ticker symbol associated with the query.
        route_type: Routing decision ('news_only', 'sec_only', 'both').
        agents_called: List of agents that were called.
        agent_results: Results from each agent called.
        synthesized_response: The final synthesized response.
        confidence: Confidence score of the response.
        had_discrepancies: Whether discrepancies were found.
        total_execution_time_ms: Total time for the orchestration.
        timestamp: When the query was processed.
    """

    user_query: str
    route_type: Literal["news_only", "sec_only", "both"]
    agents_called: list[str]
    agent_results: list[AgentResultEntry]
    synthesized_response: str
    confidence: float
    query_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ticker: str | None = None
    had_discrepancies: bool = False
    total_execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "user_query": self.user_query,
            "ticker": self.ticker,
            "route_type": self.route_type,
            "agents_called": self.agents_called,
            "agent_results": [r.to_dict() for r in self.agent_results],
            "synthesized_response": self.synthesized_response,
            "confidence": self.confidence,
            "had_discrepancies": self.had_discrepancies,
            "total_execution_time_ms": self.total_execution_time_ms,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrchestratedQueryEntry:
        """Create from dictionary."""
        return cls(
            query_id=data.get("query_id", uuid.uuid4().hex[:12]),
            user_query=data["user_query"],
            ticker=data.get("ticker"),
            route_type=data["route_type"],
            agents_called=data.get("agents_called", []),
            agent_results=[AgentResultEntry.from_dict(r) for r in data.get("agent_results", [])],
            synthesized_response=data.get("synthesized_response", ""),
            confidence=data.get("confidence", 0.0),
            had_discrepancies=data.get("had_discrepancies", False),
            total_execution_time_ms=data.get("total_execution_time_ms", 0.0),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )

    @property
    def successful_agents(self) -> list[str]:
        """Return list of agents that succeeded."""
        return [r.agent_name for r in self.agent_results if r.status == "success"]

    @property
    def failed_agents(self) -> list[str]:
        """Return list of agents that failed or timed out."""
        return [r.agent_name for r in self.agent_results if r.status != "success"]


class OrchestratorMemory:
    """In-process session memory for the orchestrator agent.

    Stores orchestrated query entries keyed by session. Tracks which agents
    were called, their results, and the synthesized responses. Compatible
    with the news_agent and sec_agent memory patterns.
    """

    def __init__(self, session_id: str | None = None) -> None:
        """Initialize memory with optional session ID.

        Args:
            session_id: Unique session identifier. Generated if not provided.
        """
        self._session_id = session_id or uuid.uuid4().hex
        self._entries: list[OrchestratedQueryEntry] = []

    @property
    def session_id(self) -> str:
        """Return the session ID."""
        return self._session_id

    @property
    def entry_count(self) -> int:
        """Return the number of stored entries."""
        return len(self._entries)

    def store_query(
        self,
        user_query: str,
        route_type: Literal["news_only", "sec_only", "both"],
        agents_called: list[str],
        agent_results: list[AgentResultEntry],
        synthesized_response: str,
        confidence: float,
        ticker: str | None = None,
        had_discrepancies: bool = False,
        total_execution_time_ms: float = 0.0,
    ) -> str:
        """Store an orchestrated query result in memory.

        Args:
            user_query: The original user query.
            route_type: The routing decision made.
            agents_called: List of agent names called.
            agent_results: Results from each agent.
            synthesized_response: The final synthesized response.
            confidence: Confidence score.
            ticker: Optional ticker symbol.
            had_discrepancies: Whether discrepancies were detected.
            total_execution_time_ms: Total execution time.

        Returns:
            The query_id of the stored entry.
        """
        entry = OrchestratedQueryEntry(
            user_query=user_query,
            route_type=route_type,
            agents_called=agents_called,
            agent_results=agent_results,
            synthesized_response=synthesized_response,
            confidence=confidence,
            ticker=ticker,
            had_discrepancies=had_discrepancies,
            total_execution_time_ms=total_execution_time_ms,
        )
        self._entries.append(entry)
        return entry.query_id

    def get_recent_queries(
        self,
        ticker: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return recent queries, optionally filtered by ticker.

        Args:
            ticker: Optional ticker to filter by (case-insensitive).
            limit: Optional maximum number of entries to return.

        Returns:
            List of query entries as dictionaries, newest first.
        """
        entries = self._entries
        if ticker:
            entries = [
                e for e in entries
                if e.ticker and e.ticker.upper() == ticker.upper()
            ]
        # Return newest first
        entries = list(reversed(entries))
        if limit is not None:
            entries = entries[:limit]
        return [e.to_dict() for e in entries]

    def get_query_by_id(self, query_id: str) -> dict[str, Any] | None:
        """Retrieve a specific query by its ID.

        Args:
            query_id: The unique query identifier.

        Returns:
            Query entry as dictionary, or None if not found.
        """
        for entry in self._entries:
            if entry.query_id == query_id:
                return entry.to_dict()
        return None

    def get_last_query(self) -> dict[str, Any] | None:
        """Return the most recent query entry.

        Returns:
            Most recent query as dictionary, or None if empty.
        """
        if not self._entries:
            return None
        return self._entries[-1].to_dict()

    def get_session_context(self) -> dict[str, Any]:
        """Return full session memory as a dict.

        Returns:
            Dictionary with session_id and all entries.
        """
        return {
            "session_id": self._session_id,
            "entries": [e.to_dict() for e in self._entries],
        }

    def get_session_summary(self) -> dict[str, Any]:
        """Return a summary of the session for context.

        Returns:
            Summary with query count, tickers analyzed, and discrepancy stats.
        """
        tickers = set(e.ticker for e in self._entries if e.ticker)
        discrepancy_count = sum(1 for e in self._entries if e.had_discrepancies)
        avg_confidence = (
            sum(e.confidence for e in self._entries) / len(self._entries)
            if self._entries
            else 0.0
        )

        return {
            "session_id": self._session_id,
            "query_count": len(self._entries),
            "tickers_analyzed": sorted(tickers),
            "queries_with_discrepancies": discrepancy_count,
            "average_confidence": round(avg_confidence, 2),
        }

    def clear(self) -> None:
        """Clear all memory entries."""
        self._entries = []
