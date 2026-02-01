"""Session memory for the News Sentiment Agent.

Stores and retrieves analysis results so the agent can reference
previous work within a session.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class MemoryEntry:
    """A single analysis record stored in memory."""

    ticker: str
    company_name: str
    summary: str
    sentiment: str
    sentiment_score: float
    material_events: list[str]
    articles_analyzed: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        return cls(
            ticker=data["ticker"],
            company_name=data["company_name"],
            summary=data["summary"],
            sentiment=data["sentiment"],
            sentiment_score=data["sentiment_score"],
            material_events=data.get("material_events", []),
            articles_analyzed=data.get("articles_analyzed", 0),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )


class AgentMemory:
    """In-process short-term memory for the News Sentiment agent.

    Stores analysis entries keyed by session. Compatible with the
    sec_agent memory pattern for future orchestrator integration.
    """

    def __init__(self, session_id: str | None = None) -> None:
        self._session_id = session_id or uuid.uuid4().hex
        self._entries: list[MemoryEntry] = []

    @property
    def session_id(self) -> str:
        return self._session_id

    def store_analysis(
        self,
        ticker: str,
        company_name: str,
        summary: str,
        sentiment: str,
        sentiment_score: float,
        material_events: list[str],
        articles_analyzed: int,
    ) -> None:
        """Store a sentiment analysis result in memory."""
        self._entries.append(
            MemoryEntry(
                ticker=ticker,
                company_name=company_name,
                summary=summary,
                sentiment=sentiment,
                sentiment_score=sentiment_score,
                material_events=material_events,
                articles_analyzed=articles_analyzed,
            )
        )

    def get_recent_analyses(self, ticker: str | None = None) -> list[dict[str, Any]]:
        """Return recent analyses, optionally filtered by ticker."""
        entries = self._entries
        if ticker:
            entries = [e for e in entries if e.ticker.upper() == ticker.upper()]
        return [e.to_dict() for e in entries]

    def get_session_context(self) -> dict[str, Any]:
        """Return full session memory as a dict."""
        return {
            "session_id": self._session_id,
            "entries": [e.to_dict() for e in self._entries],
        }

    def clear(self) -> None:
        """Clear all memory entries."""
        self._entries = []
