"""Tests for OrchestratorMemory and related dataclasses."""

from __future__ import annotations

import pytest

from src.orchestrator.memory import (
    AgentResultEntry,
    OrchestratedQueryEntry,
    OrchestratorMemory,
)


# ---------------------------------------------------------------------------
# AgentResultEntry tests
# ---------------------------------------------------------------------------


class TestAgentResultEntry:
    """Tests for AgentResultEntry dataclass."""

    def test_create_success_result(self):
        """Test creating a successful agent result."""
        entry = AgentResultEntry(
            agent_name="news_agent",
            status="success",
            response="Positive sentiment for AAPL",
            execution_time_ms=150.5,
        )
        assert entry.agent_name == "news_agent"
        assert entry.status == "success"
        assert entry.response == "Positive sentiment for AAPL"
        assert entry.error_message is None
        assert entry.execution_time_ms == 150.5

    def test_create_error_result(self):
        """Test creating an error agent result."""
        entry = AgentResultEntry(
            agent_name="sec_agent",
            status="error",
            error_message="Connection failed",
            execution_time_ms=50.0,
        )
        assert entry.agent_name == "sec_agent"
        assert entry.status == "error"
        assert entry.response is None
        assert entry.error_message == "Connection failed"

    def test_create_timeout_result(self):
        """Test creating a timeout agent result."""
        entry = AgentResultEntry(
            agent_name="news_agent",
            status="timeout",
            error_message="Exceeded 60s limit",
            execution_time_ms=60000.0,
        )
        assert entry.status == "timeout"
        assert entry.error_message == "Exceeded 60s limit"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        entry = AgentResultEntry(
            agent_name="news_agent",
            status="success",
            response="Test response",
            execution_time_ms=100.0,
        )
        d = entry.to_dict()
        assert d["agent_name"] == "news_agent"
        assert d["status"] == "success"
        assert d["response"] == "Test response"
        assert d["error_message"] is None
        assert d["execution_time_ms"] == 100.0

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "agent_name": "sec_agent",
            "status": "success",
            "response": "SEC analysis complete",
            "error_message": None,
            "execution_time_ms": 200.0,
        }
        entry = AgentResultEntry.from_dict(data)
        assert entry.agent_name == "sec_agent"
        assert entry.status == "success"
        assert entry.response == "SEC analysis complete"
        assert entry.execution_time_ms == 200.0

    def test_from_dict_minimal(self):
        """Test creation from minimal dictionary."""
        data = {
            "agent_name": "news_agent",
            "status": "error",
        }
        entry = AgentResultEntry.from_dict(data)
        assert entry.agent_name == "news_agent"
        assert entry.status == "error"
        assert entry.response is None
        assert entry.error_message is None
        assert entry.execution_time_ms == 0.0

    def test_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        original = AgentResultEntry(
            agent_name="sec_agent",
            status="timeout",
            error_message="Request timed out",
            execution_time_ms=60500.0,
        )
        restored = AgentResultEntry.from_dict(original.to_dict())
        assert restored.agent_name == original.agent_name
        assert restored.status == original.status
        assert restored.error_message == original.error_message
        assert restored.execution_time_ms == original.execution_time_ms


# ---------------------------------------------------------------------------
# OrchestratedQueryEntry tests
# ---------------------------------------------------------------------------


class TestOrchestratedQueryEntry:
    """Tests for OrchestratedQueryEntry dataclass."""

    def test_create_entry(self):
        """Test creating an orchestrated query entry."""
        results = [
            AgentResultEntry("news_agent", "success", "News response", None, 100.0),
            AgentResultEntry("sec_agent", "success", "SEC response", None, 150.0),
        ]
        entry = OrchestratedQueryEntry(
            user_query="What is the outlook for AAPL?",
            route_type="both",
            agents_called=["news_agent", "sec_agent"],
            agent_results=results,
            synthesized_response="Synthesized analysis of AAPL",
            confidence=0.85,
            ticker="AAPL",
            had_discrepancies=False,
            total_execution_time_ms=300.0,
        )
        assert entry.user_query == "What is the outlook for AAPL?"
        assert entry.route_type == "both"
        assert len(entry.agent_results) == 2
        assert entry.confidence == 0.85
        assert entry.ticker == "AAPL"
        assert entry.query_id  # auto-generated

    def test_query_id_auto_generated(self):
        """Test that query_id is auto-generated."""
        entry = OrchestratedQueryEntry(
            user_query="Test query",
            route_type="news_only",
            agents_called=["news_agent"],
            agent_results=[],
            synthesized_response="Response",
            confidence=0.5,
        )
        assert entry.query_id is not None
        assert len(entry.query_id) == 12

    def test_timestamp_auto_generated(self):
        """Test that timestamp is auto-generated."""
        entry = OrchestratedQueryEntry(
            user_query="Test query",
            route_type="sec_only",
            agents_called=["sec_agent"],
            agent_results=[],
            synthesized_response="Response",
            confidence=0.5,
        )
        assert entry.timestamp is not None
        assert "T" in entry.timestamp  # ISO format

    def test_successful_agents_property(self):
        """Test successful_agents property."""
        results = [
            AgentResultEntry("news_agent", "success", "Response", None, 100.0),
            AgentResultEntry("sec_agent", "error", None, "Failed", 50.0),
        ]
        entry = OrchestratedQueryEntry(
            user_query="Test",
            route_type="both",
            agents_called=["news_agent", "sec_agent"],
            agent_results=results,
            synthesized_response="Partial response",
            confidence=0.6,
        )
        assert entry.successful_agents == ["news_agent"]

    def test_failed_agents_property(self):
        """Test failed_agents property."""
        results = [
            AgentResultEntry("news_agent", "timeout", None, "Timeout", 60000.0),
            AgentResultEntry("sec_agent", "success", "Response", None, 100.0),
        ]
        entry = OrchestratedQueryEntry(
            user_query="Test",
            route_type="both",
            agents_called=["news_agent", "sec_agent"],
            agent_results=results,
            synthesized_response="Partial response",
            confidence=0.6,
        )
        assert entry.failed_agents == ["news_agent"]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        results = [AgentResultEntry("news_agent", "success", "Response", None, 100.0)]
        entry = OrchestratedQueryEntry(
            user_query="Test query",
            route_type="news_only",
            agents_called=["news_agent"],
            agent_results=results,
            synthesized_response="Synthesized",
            confidence=0.8,
            ticker="TSLA",
            had_discrepancies=True,
            total_execution_time_ms=200.0,
        )
        d = entry.to_dict()
        assert d["user_query"] == "Test query"
        assert d["route_type"] == "news_only"
        assert d["ticker"] == "TSLA"
        assert d["had_discrepancies"] is True
        assert len(d["agent_results"]) == 1
        assert d["agent_results"][0]["agent_name"] == "news_agent"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "query_id": "test123",
            "user_query": "What about MSFT?",
            "ticker": "MSFT",
            "route_type": "both",
            "agents_called": ["news_agent", "sec_agent"],
            "agent_results": [
                {"agent_name": "news_agent", "status": "success", "response": "News"},
            ],
            "synthesized_response": "Final response",
            "confidence": 0.9,
            "had_discrepancies": False,
            "total_execution_time_ms": 500.0,
            "timestamp": "2026-01-30T10:00:00+00:00",
        }
        entry = OrchestratedQueryEntry.from_dict(data)
        assert entry.query_id == "test123"
        assert entry.user_query == "What about MSFT?"
        assert entry.ticker == "MSFT"
        assert len(entry.agent_results) == 1
        assert entry.agent_results[0].agent_name == "news_agent"

    def test_from_dict_minimal(self):
        """Test creation from minimal dictionary."""
        data = {
            "user_query": "Minimal query",
            "route_type": "news_only",
        }
        entry = OrchestratedQueryEntry.from_dict(data)
        assert entry.user_query == "Minimal query"
        assert entry.route_type == "news_only"
        assert entry.agent_results == []
        assert entry.synthesized_response == ""
        assert entry.confidence == 0.0

    def test_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        results = [
            AgentResultEntry("news_agent", "success", "News", None, 100.0),
            AgentResultEntry("sec_agent", "timeout", None, "Timeout", 60000.0),
        ]
        original = OrchestratedQueryEntry(
            user_query="Complex query",
            route_type="both",
            agents_called=["news_agent", "sec_agent"],
            agent_results=results,
            synthesized_response="Final",
            confidence=0.7,
            ticker="GOOG",
            had_discrepancies=True,
            total_execution_time_ms=60500.0,
        )
        restored = OrchestratedQueryEntry.from_dict(original.to_dict())
        assert restored.user_query == original.user_query
        assert restored.route_type == original.route_type
        assert restored.ticker == original.ticker
        assert restored.had_discrepancies == original.had_discrepancies
        assert len(restored.agent_results) == len(original.agent_results)


# ---------------------------------------------------------------------------
# OrchestratorMemory tests
# ---------------------------------------------------------------------------


class TestOrchestratorMemory:
    """Tests for OrchestratorMemory class."""

    def test_session_id_auto_generated(self):
        """Test that session_id is auto-generated when not provided."""
        mem = OrchestratorMemory()
        assert mem.session_id is not None
        assert isinstance(mem.session_id, str)
        assert len(mem.session_id) == 32  # hex uuid

    def test_session_id_custom(self):
        """Test custom session_id."""
        mem = OrchestratorMemory(session_id="custom-session-123")
        assert mem.session_id == "custom-session-123"

    def test_entry_count_empty(self):
        """Test entry_count when empty."""
        mem = OrchestratorMemory()
        assert mem.entry_count == 0

    def test_store_query(self):
        """Test storing a query."""
        mem = OrchestratorMemory()
        results = [AgentResultEntry("news_agent", "success", "Response", None, 100.0)]

        query_id = mem.store_query(
            user_query="Test query",
            route_type="news_only",
            agents_called=["news_agent"],
            agent_results=results,
            synthesized_response="Synthesized",
            confidence=0.8,
            ticker="AAPL",
        )

        assert query_id is not None
        assert mem.entry_count == 1

    def test_store_multiple_queries(self):
        """Test storing multiple queries."""
        mem = OrchestratorMemory()

        for i in range(5):
            mem.store_query(
                user_query=f"Query {i}",
                route_type="both",
                agents_called=["news_agent", "sec_agent"],
                agent_results=[],
                synthesized_response=f"Response {i}",
                confidence=0.5 + i * 0.1,
            )

        assert mem.entry_count == 5

    def test_get_recent_queries(self):
        """Test getting recent queries."""
        mem = OrchestratorMemory()

        mem.store_query(
            user_query="First query",
            route_type="news_only",
            agents_called=["news_agent"],
            agent_results=[],
            synthesized_response="First response",
            confidence=0.7,
        )
        mem.store_query(
            user_query="Second query",
            route_type="sec_only",
            agents_called=["sec_agent"],
            agent_results=[],
            synthesized_response="Second response",
            confidence=0.8,
        )

        queries = mem.get_recent_queries()
        assert len(queries) == 2
        # Should be newest first
        assert queries[0]["user_query"] == "Second query"
        assert queries[1]["user_query"] == "First query"

    def test_get_recent_queries_filter_by_ticker(self):
        """Test filtering queries by ticker."""
        mem = OrchestratorMemory()

        mem.store_query(
            user_query="AAPL query",
            route_type="both",
            agents_called=["news_agent", "sec_agent"],
            agent_results=[],
            synthesized_response="AAPL response",
            confidence=0.8,
            ticker="AAPL",
        )
        mem.store_query(
            user_query="MSFT query",
            route_type="both",
            agents_called=["news_agent", "sec_agent"],
            agent_results=[],
            synthesized_response="MSFT response",
            confidence=0.8,
            ticker="MSFT",
        )
        mem.store_query(
            user_query="Another AAPL query",
            route_type="news_only",
            agents_called=["news_agent"],
            agent_results=[],
            synthesized_response="Another AAPL response",
            confidence=0.7,
            ticker="AAPL",
        )

        aapl_queries = mem.get_recent_queries(ticker="AAPL")
        assert len(aapl_queries) == 2
        assert all(q["ticker"] == "AAPL" for q in aapl_queries)

        msft_queries = mem.get_recent_queries(ticker="msft")  # case-insensitive
        assert len(msft_queries) == 1

    def test_get_recent_queries_with_limit(self):
        """Test limiting query results."""
        mem = OrchestratorMemory()

        for i in range(10):
            mem.store_query(
                user_query=f"Query {i}",
                route_type="both",
                agents_called=["news_agent", "sec_agent"],
                agent_results=[],
                synthesized_response=f"Response {i}",
                confidence=0.5,
            )

        queries = mem.get_recent_queries(limit=3)
        assert len(queries) == 3
        # Should be newest first
        assert queries[0]["user_query"] == "Query 9"
        assert queries[1]["user_query"] == "Query 8"
        assert queries[2]["user_query"] == "Query 7"

    def test_get_recent_queries_filter_and_limit(self):
        """Test combined ticker filter and limit."""
        mem = OrchestratorMemory()

        for i in range(5):
            mem.store_query(
                user_query=f"AAPL query {i}",
                route_type="both",
                agents_called=[],
                agent_results=[],
                synthesized_response=f"Response {i}",
                confidence=0.5,
                ticker="AAPL",
            )

        queries = mem.get_recent_queries(ticker="AAPL", limit=2)
        assert len(queries) == 2
        assert queries[0]["user_query"] == "AAPL query 4"
        assert queries[1]["user_query"] == "AAPL query 3"

    def test_get_query_by_id(self):
        """Test retrieving a specific query by ID."""
        mem = OrchestratorMemory()

        query_id = mem.store_query(
            user_query="Specific query",
            route_type="news_only",
            agents_called=["news_agent"],
            agent_results=[],
            synthesized_response="Specific response",
            confidence=0.85,
        )

        query = mem.get_query_by_id(query_id)
        assert query is not None
        assert query["user_query"] == "Specific query"
        assert query["query_id"] == query_id

    def test_get_query_by_id_not_found(self):
        """Test retrieving a non-existent query by ID."""
        mem = OrchestratorMemory()

        mem.store_query(
            user_query="Some query",
            route_type="news_only",
            agents_called=[],
            agent_results=[],
            synthesized_response="Response",
            confidence=0.5,
        )

        query = mem.get_query_by_id("nonexistent-id")
        assert query is None

    def test_get_last_query(self):
        """Test getting the most recent query."""
        mem = OrchestratorMemory()

        mem.store_query(
            user_query="First",
            route_type="news_only",
            agents_called=[],
            agent_results=[],
            synthesized_response="First response",
            confidence=0.5,
        )
        mem.store_query(
            user_query="Second",
            route_type="sec_only",
            agents_called=[],
            agent_results=[],
            synthesized_response="Second response",
            confidence=0.6,
        )

        last = mem.get_last_query()
        assert last is not None
        assert last["user_query"] == "Second"

    def test_get_last_query_empty(self):
        """Test getting last query when memory is empty."""
        mem = OrchestratorMemory()
        assert mem.get_last_query() is None

    def test_get_session_context(self):
        """Test getting full session context."""
        mem = OrchestratorMemory(session_id="test-session")

        mem.store_query(
            user_query="Query 1",
            route_type="both",
            agents_called=["news_agent", "sec_agent"],
            agent_results=[],
            synthesized_response="Response 1",
            confidence=0.8,
        )

        ctx = mem.get_session_context()
        assert ctx["session_id"] == "test-session"
        assert len(ctx["entries"]) == 1
        assert ctx["entries"][0]["user_query"] == "Query 1"

    def test_get_session_context_empty(self):
        """Test getting session context when empty."""
        mem = OrchestratorMemory(session_id="empty-session")
        ctx = mem.get_session_context()
        assert ctx["session_id"] == "empty-session"
        assert ctx["entries"] == []

    def test_get_session_summary(self):
        """Test getting session summary."""
        mem = OrchestratorMemory(session_id="summary-session")

        mem.store_query(
            user_query="AAPL query",
            route_type="both",
            agents_called=["news_agent", "sec_agent"],
            agent_results=[],
            synthesized_response="Response",
            confidence=0.8,
            ticker="AAPL",
            had_discrepancies=True,
        )
        mem.store_query(
            user_query="MSFT query",
            route_type="news_only",
            agents_called=["news_agent"],
            agent_results=[],
            synthesized_response="Response",
            confidence=0.6,
            ticker="MSFT",
            had_discrepancies=False,
        )
        mem.store_query(
            user_query="Another AAPL query",
            route_type="sec_only",
            agents_called=["sec_agent"],
            agent_results=[],
            synthesized_response="Response",
            confidence=0.7,
            ticker="AAPL",
            had_discrepancies=True,
        )

        summary = mem.get_session_summary()
        assert summary["session_id"] == "summary-session"
        assert summary["query_count"] == 3
        assert set(summary["tickers_analyzed"]) == {"AAPL", "MSFT"}
        assert summary["queries_with_discrepancies"] == 2
        assert summary["average_confidence"] == 0.7  # (0.8 + 0.6 + 0.7) / 3

    def test_get_session_summary_empty(self):
        """Test session summary when empty."""
        mem = OrchestratorMemory()
        summary = mem.get_session_summary()
        assert summary["query_count"] == 0
        assert summary["tickers_analyzed"] == []
        assert summary["queries_with_discrepancies"] == 0
        assert summary["average_confidence"] == 0.0

    def test_clear(self):
        """Test clearing memory."""
        mem = OrchestratorMemory()

        mem.store_query(
            user_query="Query 1",
            route_type="both",
            agents_called=[],
            agent_results=[],
            synthesized_response="Response 1",
            confidence=0.5,
        )
        mem.store_query(
            user_query="Query 2",
            route_type="both",
            agents_called=[],
            agent_results=[],
            synthesized_response="Response 2",
            confidence=0.5,
        )

        assert mem.entry_count == 2
        mem.clear()
        assert mem.entry_count == 0
        assert mem.get_recent_queries() == []

    def test_clear_preserves_session_id(self):
        """Test that clear preserves the session ID."""
        mem = OrchestratorMemory(session_id="persistent-session")

        mem.store_query(
            user_query="Query",
            route_type="both",
            agents_called=[],
            agent_results=[],
            synthesized_response="Response",
            confidence=0.5,
        )
        mem.clear()

        assert mem.session_id == "persistent-session"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestOrchestratorMemoryIntegration:
    """Integration tests for OrchestratorMemory with full workflow."""

    def test_full_query_workflow(self):
        """Test a full orchestration query workflow."""
        mem = OrchestratorMemory()

        # Simulate a dual-agent query
        news_result = AgentResultEntry(
            agent_name="news_agent",
            status="success",
            response="Apple stock shows positive sentiment based on recent product announcements.",
            execution_time_ms=1500.0,
        )
        sec_result = AgentResultEntry(
            agent_name="sec_agent",
            status="success",
            response="Apple's 10-K shows strong revenue growth but highlights supply chain risks.",
            execution_time_ms=2000.0,
        )

        query_id = mem.store_query(
            user_query="What is the outlook for Apple?",
            route_type="both",
            agents_called=["news_agent", "sec_agent"],
            agent_results=[news_result, sec_result],
            synthesized_response="Both sources agree Apple has positive momentum...",
            confidence=0.85,
            ticker="AAPL",
            had_discrepancies=False,
            total_execution_time_ms=3500.0,
        )

        # Verify storage
        query = mem.get_query_by_id(query_id)
        assert query is not None
        assert len(query["agent_results"]) == 2
        assert query["confidence"] == 0.85

        # Check summary
        summary = mem.get_session_summary()
        assert summary["query_count"] == 1
        assert "AAPL" in summary["tickers_analyzed"]

    def test_partial_failure_workflow(self):
        """Test workflow when one agent fails."""
        mem = OrchestratorMemory()

        news_result = AgentResultEntry(
            agent_name="news_agent",
            status="success",
            response="Tesla news analysis complete.",
            execution_time_ms=1200.0,
        )
        sec_result = AgentResultEntry(
            agent_name="sec_agent",
            status="timeout",
            error_message="Request exceeded 60s timeout",
            execution_time_ms=60000.0,
        )

        query_id = mem.store_query(
            user_query="What are the risks for Tesla?",
            route_type="both",
            agents_called=["news_agent", "sec_agent"],
            agent_results=[news_result, sec_result],
            synthesized_response="Based on available news data (SEC data unavailable)...",
            confidence=0.6,  # Lower due to partial data
            ticker="TSLA",
            had_discrepancies=False,
            total_execution_time_ms=61200.0,
        )

        query = mem.get_query_by_id(query_id)
        entry = OrchestratedQueryEntry.from_dict(query)

        assert entry.successful_agents == ["news_agent"]
        assert entry.failed_agents == ["sec_agent"]

    def test_multi_ticker_session(self):
        """Test session with multiple tickers."""
        mem = OrchestratorMemory()

        tickers = ["AAPL", "MSFT", "GOOG", "AAPL", "AMZN"]
        for i, ticker in enumerate(tickers):
            mem.store_query(
                user_query=f"Query about {ticker}",
                route_type="both",
                agents_called=["news_agent", "sec_agent"],
                agent_results=[],
                synthesized_response=f"Analysis for {ticker}",
                confidence=0.5 + i * 0.05,
                ticker=ticker,
                had_discrepancies=i % 2 == 0,
            )

        summary = mem.get_session_summary()
        assert summary["query_count"] == 5
        assert set(summary["tickers_analyzed"]) == {"AAPL", "MSFT", "GOOG", "AMZN"}
        assert summary["queries_with_discrepancies"] == 3  # indices 0, 2, 4

        # Filter by ticker
        aapl_queries = mem.get_recent_queries(ticker="AAPL")
        assert len(aapl_queries) == 2

    def test_no_ticker_queries(self):
        """Test queries without ticker specified."""
        mem = OrchestratorMemory()

        mem.store_query(
            user_query="What are the market trends?",
            route_type="news_only",
            agents_called=["news_agent"],
            agent_results=[],
            synthesized_response="General market analysis...",
            confidence=0.7,
            # No ticker
        )

        summary = mem.get_session_summary()
        assert summary["tickers_analyzed"] == []

        # Filtering by ticker should return nothing
        queries = mem.get_recent_queries(ticker="AAPL")
        assert len(queries) == 0
