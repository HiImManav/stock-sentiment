"""Tests for the Orchestration Agent FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.orchestrator.api.server import app, get_agent
from src.orchestrator.agent import OrchestrationResult
from src.orchestrator.comparison.discrepancy import (
    Agreement,
    ComparisonResult,
    Discrepancy,
    DiscrepancySeverity,
    DiscrepancyType,
)
from src.orchestrator.comparison.signals import (
    ExtractedSignal,
    SignalDirection,
    SignalExtractionResult,
    SignalType,
)
from src.orchestrator.execution.result import AgentResult, AgentStatus

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_healthy(self) -> None:
        """Health check returns healthy status."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "orchestrator"


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    """Tests for the /query endpoint."""

    @patch("src.orchestrator.api.server.get_agent")
    def test_query_success(self, mock_get_agent: MagicMock) -> None:
        """Query endpoint returns successful response."""
        mock_agent = MagicMock()
        mock_agent.session_id = "test-session-123"
        mock_agent.query.return_value = OrchestrationResult(
            response="Apple shows strong performance.",
            route_type="both",
            agents_used=["news_agent", "sec_agent"],
            had_discrepancies=False,
            confidence=0.85,
            execution_time_ms=1234.5,
            query_id="query-abc123",
        )
        mock_get_agent.return_value = mock_agent

        resp = client.post("/query", json={"query": "What's the outlook for Apple?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Apple shows strong performance."
        assert data["agents_used"] == ["news_agent", "sec_agent"]
        assert data["had_discrepancies"] is False
        assert data["confidence"] == 0.85
        assert data["execution_time_ms"] == 1234.5
        assert data["session_id"] == "test-session-123"
        assert data["query_id"] == "query-abc123"

    @patch("src.orchestrator.api.server.get_agent")
    def test_query_with_ticker(self, mock_get_agent: MagicMock) -> None:
        """Query endpoint passes ticker to agent."""
        mock_agent = MagicMock()
        mock_agent.session_id = "test-session"
        mock_agent.query.return_value = OrchestrationResult(
            response="Analysis for TSLA.",
            route_type="both",
            agents_used=["news_agent"],
            had_discrepancies=False,
            confidence=0.8,
            execution_time_ms=1000.0,
            query_id="query-123",
        )
        mock_get_agent.return_value = mock_agent

        resp = client.post(
            "/query",
            json={"query": "What are the risks?", "ticker": "TSLA"},
        )
        assert resp.status_code == 200
        mock_agent.query.assert_called_once_with(
            user_message="What are the risks?",
            ticker="TSLA",
            force_route=None,
        )

    @patch("src.orchestrator.api.server.get_agent")
    def test_query_with_sources_news_only(self, mock_get_agent: MagicMock) -> None:
        """Query with sources=['news'] forces news_only route."""
        mock_agent = MagicMock()
        mock_agent.session_id = "test-session"
        mock_agent.query.return_value = OrchestrationResult(
            response="News analysis.",
            route_type="news_only",
            agents_used=["news_agent"],
            had_discrepancies=False,
            confidence=0.75,
            execution_time_ms=500.0,
            query_id="query-456",
        )
        mock_get_agent.return_value = mock_agent

        resp = client.post(
            "/query",
            json={"query": "Recent news?", "sources": ["news"]},
        )
        assert resp.status_code == 200
        mock_agent.query.assert_called_once_with(
            user_message="Recent news?",
            ticker=None,
            force_route="news_only",
        )

    @patch("src.orchestrator.api.server.get_agent")
    def test_query_with_sources_sec_only(self, mock_get_agent: MagicMock) -> None:
        """Query with sources=['sec'] forces sec_only route."""
        mock_agent = MagicMock()
        mock_agent.session_id = "test-session"
        mock_agent.query.return_value = OrchestrationResult(
            response="SEC analysis.",
            route_type="sec_only",
            agents_used=["sec_agent"],
            had_discrepancies=False,
            confidence=0.7,
            execution_time_ms=800.0,
            query_id="query-789",
        )
        mock_get_agent.return_value = mock_agent

        resp = client.post(
            "/query",
            json={"query": "10-K risk factors?", "sources": ["sec"]},
        )
        assert resp.status_code == 200
        mock_agent.query.assert_called_once_with(
            user_message="10-K risk factors?",
            ticker=None,
            force_route="sec_only",
        )

    @patch("src.orchestrator.api.server.get_agent")
    def test_query_with_sources_both(self, mock_get_agent: MagicMock) -> None:
        """Query with sources=['news', 'sec'] forces both route."""
        mock_agent = MagicMock()
        mock_agent.session_id = "test-session"
        mock_agent.query.return_value = OrchestrationResult(
            response="Combined analysis.",
            route_type="both",
            agents_used=["news_agent", "sec_agent"],
            had_discrepancies=True,
            confidence=0.9,
            execution_time_ms=2000.0,
            query_id="query-abc",
        )
        mock_get_agent.return_value = mock_agent

        resp = client.post(
            "/query",
            json={"query": "Full analysis?", "sources": ["news", "sec"]},
        )
        assert resp.status_code == 200
        mock_agent.query.assert_called_once_with(
            user_message="Full analysis?",
            ticker=None,
            force_route="both",
        )

    @patch("src.orchestrator.api.server.get_agent")
    def test_query_with_discrepancies(self, mock_get_agent: MagicMock) -> None:
        """Query response includes discrepancy flag."""
        mock_agent = MagicMock()
        mock_agent.session_id = "test-session"
        mock_agent.query.return_value = OrchestrationResult(
            response="Found some conflicts.",
            route_type="both",
            agents_used=["news_agent", "sec_agent"],
            had_discrepancies=True,
            confidence=0.65,
            execution_time_ms=1500.0,
            query_id="query-disc",
        )
        mock_get_agent.return_value = mock_agent

        resp = client.post("/query", json={"query": "Any issues?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["had_discrepancies"] is True
        assert data["confidence"] == 0.65

    @patch("src.orchestrator.api.server.get_agent")
    def test_query_error_handling(self, mock_get_agent: MagicMock) -> None:
        """Query endpoint handles errors gracefully."""
        mock_agent = MagicMock()
        mock_agent.query.side_effect = Exception("Bedrock connection failed")
        mock_get_agent.return_value = mock_agent

        resp = client.post("/query", json={"query": "Test query"})
        assert resp.status_code == 500
        assert "Bedrock connection failed" in resp.json()["detail"]

    def test_query_missing_query_field(self) -> None:
        """Query endpoint returns 422 for missing query field."""
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_query_empty_query(self) -> None:
        """Query endpoint accepts empty query string."""
        # FastAPI/Pydantic should accept empty string (no minLength constraint)
        with patch("src.orchestrator.api.server.get_agent") as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.session_id = "test"
            mock_agent.query.return_value = OrchestrationResult(
                response="",
                route_type="both",
                agents_used=[],
                had_discrepancies=False,
                confidence=0.0,
                execution_time_ms=100.0,
                query_id="q1",
            )
            mock_get_agent.return_value = mock_agent

            resp = client.post("/query", json={"query": ""})
            # Empty query is valid (agent may handle it)
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /compare
# ---------------------------------------------------------------------------


class TestCompareEndpoint:
    """Tests for the /compare endpoint."""

    @patch("src.orchestrator.api.server.get_agent")
    def test_compare_success(self, mock_get_agent: MagicMock) -> None:
        """Compare endpoint returns successful response."""
        mock_agent = MagicMock()
        mock_agent.session_id = "test-session"
        mock_agent.compare.return_value = OrchestrationResult(
            response="Comparison shows alignment.",
            route_type="both",
            agents_used=["news_agent", "sec_agent"],
            had_discrepancies=False,
            confidence=0.9,
            execution_time_ms=2500.0,
            query_id="compare-123",
            comparison=None,
        )
        mock_get_agent.return_value = mock_agent

        resp = client.post("/compare", json={"ticker": "AAPL"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Comparison shows alignment."
        assert data["ticker"] == "AAPL"
        assert data["had_discrepancies"] is False
        assert data["confidence"] == 0.9
        assert data["execution_time_ms"] == 2500.0
        assert data["session_id"] == "test-session"
        assert data["query_id"] == "compare-123"

    @patch("src.orchestrator.api.server.get_agent")
    def test_compare_with_discrepancies(self, mock_get_agent: MagicMock) -> None:
        """Compare endpoint includes discrepancy details."""
        # Create mock signals
        news_signal = ExtractedSignal(
            signal_type=SignalType.SENTIMENT,
            direction=SignalDirection.POSITIVE,
            topic="outlook",
            description="Strong growth reported",
            confidence=0.8,
            source="news_agent",
        )
        sec_signal = ExtractedSignal(
            signal_type=SignalType.SENTIMENT,
            direction=SignalDirection.NEGATIVE,
            topic="outlook",
            description="Declining revenue trend",
            confidence=0.85,
            source="sec_agent",
        )

        # Create comparison result with discrepancies
        comparison = ComparisonResult(
            news_result=MagicMock(spec=SignalExtractionResult),
            sec_result=MagicMock(spec=SignalExtractionResult),
            discrepancies=[
                Discrepancy(
                    discrepancy_type=DiscrepancyType.SENTIMENT_CONFLICT,
                    severity=DiscrepancySeverity.HIGH,
                    topic="outlook",
                    news_signal=news_signal,
                    sec_signal=sec_signal,
                    description="News is positive but SEC is negative.",
                    confidence=0.82,
                )
            ],
            agreements=[
                Agreement(
                    topic="growth",
                    direction=SignalDirection.POSITIVE,
                    news_signal=news_signal,
                    sec_signal=sec_signal,
                    description="Both agree on positive growth.",
                    confidence=0.75,
                )
            ],
            overall_alignment=-0.3,
            has_critical_discrepancies=True,
            summary="Found conflicts.",
        )

        mock_agent = MagicMock()
        mock_agent.session_id = "test-session"
        mock_agent.compare.return_value = OrchestrationResult(
            response="Found discrepancies between sources.",
            route_type="both",
            agents_used=["news_agent", "sec_agent"],
            had_discrepancies=True,
            confidence=0.6,
            execution_time_ms=3000.0,
            query_id="compare-disc",
            comparison=comparison,
        )
        mock_get_agent.return_value = mock_agent

        resp = client.post("/compare", json={"ticker": "TSLA"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["had_discrepancies"] is True
        assert data["alignment_score"] == -0.3
        assert len(data["discrepancies"]) == 1
        assert data["discrepancies"][0]["type"] == "sentiment_conflict"
        assert data["discrepancies"][0]["severity"] == "high"
        assert data["discrepancies"][0]["topic"] == "outlook"
        assert len(data["agreements"]) == 1
        assert data["agreements"][0]["topic"] == "growth"
        assert data["agreements"][0]["direction"] == "positive"

    @patch("src.orchestrator.api.server.get_agent")
    def test_compare_lowercase_ticker(self, mock_get_agent: MagicMock) -> None:
        """Compare endpoint uppercases ticker in response."""
        mock_agent = MagicMock()
        mock_agent.session_id = "test"
        mock_agent.compare.return_value = OrchestrationResult(
            response="Analysis.",
            route_type="both",
            agents_used=[],
            had_discrepancies=False,
            confidence=0.5,
            execution_time_ms=1000.0,
            query_id="q1",
            comparison=None,
        )
        mock_get_agent.return_value = mock_agent

        resp = client.post("/compare", json={"ticker": "aapl"})
        assert resp.status_code == 200
        assert resp.json()["ticker"] == "AAPL"

    @patch("src.orchestrator.api.server.get_agent")
    def test_compare_error_handling(self, mock_get_agent: MagicMock) -> None:
        """Compare endpoint handles errors gracefully."""
        mock_agent = MagicMock()
        mock_agent.compare.side_effect = Exception("Agent timeout")
        mock_get_agent.return_value = mock_agent

        resp = client.post("/compare", json={"ticker": "XYZ"})
        assert resp.status_code == 500
        assert "Agent timeout" in resp.json()["detail"]

    def test_compare_missing_ticker(self) -> None:
        """Compare endpoint returns 422 for missing ticker."""
        resp = client.post("/compare", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /session/summary
# ---------------------------------------------------------------------------


class TestSessionSummaryEndpoint:
    """Tests for the /session/summary endpoint."""

    @patch("src.orchestrator.api.server.get_agent")
    def test_session_summary_success(self, mock_get_agent: MagicMock) -> None:
        """Session summary endpoint returns stats."""
        mock_agent = MagicMock()
        mock_agent.get_session_summary.return_value = {
            "session_id": "session-xyz",
            "query_count": 5,
            "tickers_analyzed": ["AAPL", "TSLA", "MSFT"],
            "discrepancy_rate": 0.4,
            "average_confidence": 0.78,
        }
        mock_get_agent.return_value = mock_agent

        resp = client.get("/session/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "session-xyz"
        assert data["query_count"] == 5
        assert data["tickers_analyzed"] == ["AAPL", "TSLA", "MSFT"]
        assert data["discrepancy_rate"] == 0.4
        assert data["average_confidence"] == 0.78

    @patch("src.orchestrator.api.server.get_agent")
    def test_session_summary_empty_session(self, mock_get_agent: MagicMock) -> None:
        """Session summary for empty session."""
        mock_agent = MagicMock()
        mock_agent.get_session_summary.return_value = {
            "session_id": "new-session",
            "query_count": 0,
            "tickers_analyzed": [],
            "discrepancy_rate": 0.0,
            "average_confidence": 0.0,
        }
        mock_get_agent.return_value = mock_agent

        resp = client.get("/session/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["query_count"] == 0
        assert data["tickers_analyzed"] == []


# ---------------------------------------------------------------------------
# GET /session/history
# ---------------------------------------------------------------------------


class TestSessionHistoryEndpoint:
    """Tests for the /session/history endpoint."""

    @patch("src.orchestrator.api.server.get_agent")
    def test_session_history_success(self, mock_get_agent: MagicMock) -> None:
        """Session history returns query list."""
        mock_agent = MagicMock()
        mock_agent.get_recent_queries.return_value = [
            {"query_id": "q1", "user_query": "Test 1", "ticker": "AAPL"},
            {"query_id": "q2", "user_query": "Test 2", "ticker": "TSLA"},
        ]
        mock_get_agent.return_value = mock_agent

        resp = client.get("/session/history")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["query_id"] == "q1"
        assert data[1]["ticker"] == "TSLA"

    @patch("src.orchestrator.api.server.get_agent")
    def test_session_history_with_ticker_filter(self, mock_get_agent: MagicMock) -> None:
        """Session history with ticker filter."""
        mock_agent = MagicMock()
        mock_agent.get_recent_queries.return_value = [
            {"query_id": "q1", "user_query": "Apple query", "ticker": "AAPL"},
        ]
        mock_get_agent.return_value = mock_agent

        resp = client.get("/session/history?ticker=AAPL")
        assert resp.status_code == 200
        mock_agent.get_recent_queries.assert_called_once_with(ticker="AAPL", limit=None)

    @patch("src.orchestrator.api.server.get_agent")
    def test_session_history_with_limit(self, mock_get_agent: MagicMock) -> None:
        """Session history with limit parameter."""
        mock_agent = MagicMock()
        mock_agent.get_recent_queries.return_value = [
            {"query_id": "q1", "user_query": "Latest", "ticker": "AAPL"},
        ]
        mock_get_agent.return_value = mock_agent

        resp = client.get("/session/history?limit=1")
        assert resp.status_code == 200
        mock_agent.get_recent_queries.assert_called_once_with(ticker=None, limit=1)

    @patch("src.orchestrator.api.server.get_agent")
    def test_session_history_with_both_filters(self, mock_get_agent: MagicMock) -> None:
        """Session history with both ticker and limit."""
        mock_agent = MagicMock()
        mock_agent.get_recent_queries.return_value = []
        mock_get_agent.return_value = mock_agent

        resp = client.get("/session/history?ticker=MSFT&limit=5")
        assert resp.status_code == 200
        mock_agent.get_recent_queries.assert_called_once_with(ticker="MSFT", limit=5)

    @patch("src.orchestrator.api.server.get_agent")
    def test_session_history_empty(self, mock_get_agent: MagicMock) -> None:
        """Session history returns empty list for new session."""
        mock_agent = MagicMock()
        mock_agent.get_recent_queries.return_value = []
        mock_get_agent.return_value = mock_agent

        resp = client.get("/session/history")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# POST /reset
# ---------------------------------------------------------------------------


class TestResetEndpoint:
    """Tests for the /reset endpoint."""

    @patch("src.orchestrator.api.server._agent", None)
    def test_reset_no_agent(self) -> None:
        """Reset when no agent exists."""
        resp = client.post("/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["message"] == "Session reset"

    def test_reset_with_agent(self) -> None:
        """Reset clears existing agent."""
        import src.orchestrator.api.server as mod

        mock_agent = MagicMock()
        mod._agent = mock_agent

        resp = client.post("/reset")
        assert resp.status_code == 200
        mock_agent.reset.assert_called_once()
        data = resp.json()
        assert data["status"] == "ok"

        # Cleanup
        mod._agent = None


# ---------------------------------------------------------------------------
# Singleton Agent
# ---------------------------------------------------------------------------


class TestAgentSingleton:
    """Tests for the agent singleton pattern."""

    @patch("src.orchestrator.api.server.OrchestrationAgent")
    def test_get_agent_creates_singleton(self, mock_cls: MagicMock) -> None:
        """get_agent creates singleton instance."""
        import src.orchestrator.api.server as mod

        mod._agent = None
        mock_cls.return_value = MagicMock()

        a1 = mod.get_agent()
        a2 = mod.get_agent()

        assert a1 is a2
        mock_cls.assert_called_once()

        # Cleanup
        mod._agent = None


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @patch("src.orchestrator.api.server.get_agent")
    def test_query_with_all_options(self, mock_get_agent: MagicMock) -> None:
        """Query with all optional parameters."""
        mock_agent = MagicMock()
        mock_agent.session_id = "test"
        mock_agent.query.return_value = OrchestrationResult(
            response="Full query.",
            route_type="both",
            agents_used=["news_agent", "sec_agent"],
            had_discrepancies=True,
            confidence=0.88,
            execution_time_ms=2200.0,
            query_id="full-q",
        )
        mock_get_agent.return_value = mock_agent

        resp = client.post(
            "/query",
            json={
                "query": "Full analysis for Apple",
                "ticker": "AAPL",
                "sources": ["news", "sec"],
                "enable_comparison": True,
            },
        )
        assert resp.status_code == 200

    @patch("src.orchestrator.api.server.get_agent")
    def test_compare_no_comparison_result(self, mock_get_agent: MagicMock) -> None:
        """Compare endpoint handles None comparison gracefully."""
        mock_agent = MagicMock()
        mock_agent.session_id = "test"
        mock_agent.compare.return_value = OrchestrationResult(
            response="Analysis.",
            route_type="both",
            agents_used=["news_agent", "sec_agent"],
            had_discrepancies=False,
            confidence=0.75,
            execution_time_ms=1500.0,
            query_id="q1",
            comparison=None,
        )
        mock_get_agent.return_value = mock_agent

        resp = client.post("/compare", json={"ticker": "GOOGL"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["discrepancies"] == []
        assert data["agreements"] == []
        assert data["alignment_score"] is None

    @patch("src.orchestrator.api.server.get_agent")
    def test_confidence_bounds(self, mock_get_agent: MagicMock) -> None:
        """Confidence values are within valid bounds."""
        mock_agent = MagicMock()
        mock_agent.session_id = "test"
        mock_agent.query.return_value = OrchestrationResult(
            response="Test.",
            route_type="both",
            agents_used=[],
            had_discrepancies=False,
            confidence=1.0,  # Maximum confidence
            execution_time_ms=100.0,
            query_id="q1",
        )
        mock_get_agent.return_value = mock_agent

        resp = client.post("/query", json={"query": "Test"})
        assert resp.status_code == 200
        assert resp.json()["confidence"] == 1.0

    @patch("src.orchestrator.api.server.get_agent")
    def test_sources_order_independent(self, mock_get_agent: MagicMock) -> None:
        """Sources list order doesn't matter for 'both' route."""
        mock_agent = MagicMock()
        mock_agent.session_id = "test"
        mock_agent.query.return_value = OrchestrationResult(
            response="Test.",
            route_type="both",
            agents_used=[],
            had_discrepancies=False,
            confidence=0.5,
            execution_time_ms=100.0,
            query_id="q1",
        )
        mock_get_agent.return_value = mock_agent

        # Test with sec before news
        resp = client.post(
            "/query",
            json={"query": "Test", "sources": ["sec", "news"]},
        )
        assert resp.status_code == 200
        mock_agent.query.assert_called_with(
            user_message="Test",
            ticker=None,
            force_route="both",
        )
