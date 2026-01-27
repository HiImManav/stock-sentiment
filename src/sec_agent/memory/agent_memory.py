"""AgentCore memory wrapper for short-term session context.

Stores and retrieves analysis results (tickers analyzed, key findings,
sections examined) so the agent can reference previous work within a session.
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
    filing_type: str
    summary: str
    sections_analyzed: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        return cls(
            ticker=data["ticker"],
            filing_type=data["filing_type"],
            summary=data["summary"],
            sections_analyzed=data["sections_analyzed"],
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )


class AgentMemory:
    """In-process short-term memory for the SEC filings agent.

    Stores analysis entries keyed by session. Designed to be replaced
    with the AgentCore memory service when available.
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
        filing_type: str,
        summary: str,
        sections_analyzed: list[str],
    ) -> None:
        """Store an analysis result in memory."""
        self._entries.append(
            MemoryEntry(
                ticker=ticker,
                filing_type=filing_type,
                summary=summary,
                sections_analyzed=sections_analyzed,
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
