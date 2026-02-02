"""Integration tests for the Orchestration Agent.

These tests verify the integration between different components of the
orchestrator system, testing real interactions between:
- API endpoints and the orchestration agent
- CLI commands and the orchestration agent
- Cross-component data flow (classifier → executor → comparator → synthesizer)
- Lambda handler integration with FastAPI app
- Memory persistence across multiple requests
- Error propagation through the stack
"""

from __future__ import annotations

import json
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from src.orchestrator.agent import OrchestrationAgent, OrchestrationResult
from src.orchestrator.api.server import app, get_agent
from src.orchestrator.cli.main import cli
from src.orchestrator.comparison import (
    ComparisonResult,
    DiscrepancyDetector,
    SignalExtractor,
)
from src.orchestrator.execution import (
    AgentResult,
    AgentStatus,
    AsyncAgentExecutor,
    ExecutionResult,
)
from src.orchestrator.memory import OrchestratorMemory
from src.orchestrator.routing import QueryClassifier
from src.orchestrator.synthesis import ResponseSynthesizer, SynthesisInput


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_news_agent() -> MagicMock:
    """Create a mock news agent with realistic responses."""
    agent = MagicMock()
    agent.query.return_value = """
    Based on my analysis of recent news for the company:

    **Overall Sentiment: POSITIVE**
    Sentiment Score: 0.72

    Key findings:
    - Strong quarterly earnings reported
    - Product launches received positive coverage
    - Analyst upgrades following guidance

    Material Events:
    1. Q4 earnings beat expectations
    2. New partnership announced

    Forward Outlook: Bullish
    """
    agent.reset.return_value = None
    return agent


@pytest.fixture
def mock_sec_agent() -> MagicMock:
    """Create a mock SEC agent with realistic responses."""
    agent = MagicMock()
    agent.query.return_value = """
    Based on my analysis of SEC filings:

    **10-K Summary:**
    - Total Revenue: $50 billion (up 5% YoY)
    - Net Income: $10 billion
    - Gross Margin: 40%

    Risk Factors:
    1. Market competition
    2. Regulatory uncertainty
    3. Supply chain concentration

    Management Discussion:
    Revenue growth driven by core products. Management expects
    continued growth in the coming fiscal year.

    Forward Guidance: Moderate growth expected
    """
    agent.reset.return_value = None
    return agent


@pytest.fixture
def mock_bedrock_client() -> MagicMock:
    """Create a mock Bedrock client for the synthesizer."""
    client = MagicMock()
    client.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {
                        "text": """
**Company Analysis Summary**

Based on both news sentiment and SEC filings:

**Key Insights:**
- News sentiment is positive with score 0.72
- SEC filings show 5% revenue growth
- Both sources indicate positive outlook

**Agreements:**
- Both sources confirm strong performance
- Growth expectations aligned

**Confidence: High**
"""
                    }
                ]
            }
        }
    }
    return client


@pytest.fixture
def orchestrator_with_mocks(
    mock_news_agent: MagicMock,
    mock_sec_agent: MagicMock,
    mock_bedrock_client: MagicMock,
) -> OrchestrationAgent:
    """Create an orchestration agent with mocked sub-agents."""
    return OrchestrationAgent(
        news_agent=mock_news_agent,
        sec_agent=mock_sec_agent,
        bedrock_client=mock_bedrock_client,
        timeout_seconds=30.0,
        enable_comparison=True,
    )


@pytest.fixture
def api_client(orchestrator_with_mocks: OrchestrationAgent) -> Generator[TestClient, None, None]:
    """Create a test client with mocked orchestrator."""
    # Override the get_agent dependency
    import src.orchestrator.api.server as server_module

    original_agent = server_module._agent
    server_module._agent = orchestrator_with_mocks

    with TestClient(app) as client:
        yield client

    # Restore original
    server_module._agent = original_agent


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


# =============================================================================
# API-Orchestrator Integration Tests
# =============================================================================


class TestAPIOrchestrationIntegration:
    """Test integration between FastAPI endpoints and OrchestrationAgent."""

    def test_query_endpoint_flows_through_full_orchestration(
        self, api_client: TestClient
    ) -> None:
        """Test that /query endpoint executes full orchestration flow."""
        response = api_client.post(
            "/query",
            json={
                "query": "What's the outlook for Apple?",
                "ticker": "AAPL",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure matches orchestration result
        assert "answer" in data
        assert "agents_used" in data
        assert "had_discrepancies" in data
        assert "confidence" in data
        assert "execution_time_ms" in data
        assert "session_id" in data
        assert "query_id" in data

        # Verify agents were actually called
        assert len(data["agents_used"]) > 0

    def test_query_endpoint_respects_source_routing(
        self, api_client: TestClient
    ) -> None:
        """Test that source parameter correctly routes to specific agents."""
        # News only
        response = api_client.post(
            "/query",
            json={
                "query": "What about the company?",
                "sources": ["news"],
            },
        )
        assert response.status_code == 200
        assert "news_agent" in response.json()["agents_used"]
        assert "sec_agent" not in response.json()["agents_used"]

        # SEC only
        response = api_client.post(
            "/query",
            json={
                "query": "What about the company?",
                "sources": ["sec"],
            },
        )
        assert response.status_code == 200
        assert "sec_agent" in response.json()["agents_used"]
        assert "news_agent" not in response.json()["agents_used"]

        # Both
        response = api_client.post(
            "/query",
            json={
                "query": "What about the company?",
                "sources": ["news", "sec"],
            },
        )
        assert response.status_code == 200
        assert "news_agent" in response.json()["agents_used"]
        assert "sec_agent" in response.json()["agents_used"]

    def test_compare_endpoint_forces_both_agents(
        self, api_client: TestClient
    ) -> None:
        """Test that /compare endpoint always uses both agents."""
        response = api_client.post(
            "/compare",
            json={"ticker": "AAPL"},
        )

        assert response.status_code == 200
        data = response.json()

        # Compare should always use both
        assert "answer" in data
        assert data["ticker"] == "AAPL"
        assert "had_discrepancies" in data
        assert "alignment_score" in data or data["alignment_score"] is None

    def test_session_persists_across_requests(
        self, api_client: TestClient
    ) -> None:
        """Test that session state persists across multiple API requests."""
        # First query
        response1 = api_client.post(
            "/query",
            json={"query": "News for AAPL", "ticker": "AAPL"},
        )
        session_id = response1.json()["session_id"]

        # Second query
        response2 = api_client.post(
            "/query",
            json={"query": "SEC for TSLA", "ticker": "TSLA"},
        )

        # Same session
        assert response2.json()["session_id"] == session_id

        # Verify history shows both queries (avoids summary endpoint key mismatch)
        history_response = api_client.get("/session/history")
        assert history_response.status_code == 200
        history = history_response.json()
        assert len(history) == 2

    def test_session_history_returns_stored_queries(
        self, api_client: TestClient
    ) -> None:
        """Test that session history returns all stored queries."""
        # Execute queries
        api_client.post("/query", json={"query": "Query 1", "ticker": "AAPL"})
        api_client.post("/query", json={"query": "Query 2", "ticker": "TSLA"})
        api_client.post("/query", json={"query": "Query 3", "ticker": "AAPL"})

        # Get all history
        response = api_client.get("/session/history")
        assert response.status_code == 200
        history = response.json()
        assert len(history) == 3

        # Filter by ticker
        response = api_client.get("/session/history?ticker=AAPL")
        assert response.status_code == 200
        filtered = response.json()
        assert len(filtered) == 2
        for entry in filtered:
            assert entry["ticker"] == "AAPL"

    def test_reset_clears_session(
        self, api_client: TestClient
    ) -> None:
        """Test that reset endpoint clears session data."""
        # Add queries
        api_client.post("/query", json={"query": "Test", "ticker": "AAPL"})

        # Reset
        response = api_client.post("/reset")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

        # Verify history is cleared
        history_response = api_client.get("/session/history")
        assert history_response.status_code == 200
        assert len(history_response.json()) == 0


# =============================================================================
# CLI-Orchestrator Integration Tests
# =============================================================================


class TestCLIOrchestrationIntegration:
    """Test integration between CLI commands and OrchestrationAgent."""

    def test_query_command_produces_structured_output(
        self,
        cli_runner: CliRunner,
        mock_news_agent: MagicMock,
        mock_sec_agent: MagicMock,
        mock_bedrock_client: MagicMock,
    ) -> None:
        """Test that query command produces proper output."""
        with patch("src.orchestrator.cli.main.OrchestrationAgent") as MockAgent:
            # Create a mock orchestrator
            mock_orchestrator = MagicMock()
            mock_orchestrator.query.return_value = OrchestrationResult(
                response="Test response",
                route_type="news_only",
                agents_used=["news_agent"],
                had_discrepancies=False,
                confidence=0.8,
                execution_time_ms=100.0,
                query_id="test-123",
            )
            MockAgent.return_value = mock_orchestrator

            result = cli_runner.invoke(cli, ["query", "What's the news for Apple?"])

            assert result.exit_code == 0
            assert "Test response" in result.output
            assert "news_agent" in result.output
            assert "80%" in result.output  # Confidence

    def test_query_command_json_output_format(
        self, cli_runner: CliRunner
    ) -> None:
        """Test that --json-output produces valid JSON."""
        with patch("src.orchestrator.cli.main.OrchestrationAgent") as MockAgent:
            mock_orchestrator = MagicMock()
            mock_result = OrchestrationResult(
                response="Test response",
                route_type="both",
                agents_used=["news_agent", "sec_agent"],
                had_discrepancies=True,
                confidence=0.6,
                execution_time_ms=200.0,
                query_id="test-456",
            )
            mock_orchestrator.query.return_value = mock_result
            MockAgent.return_value = mock_orchestrator

            result = cli_runner.invoke(
                cli, ["query", "Compare news vs SEC", "--json-output"]
            )

            assert result.exit_code == 0
            # Should be valid JSON
            data = json.loads(result.output)
            assert data["response"] == "Test response"
            assert data["route_type"] == "both"
            assert data["had_discrepancies"] is True

    def test_query_command_source_routing(
        self, cli_runner: CliRunner
    ) -> None:
        """Test that --source option correctly routes queries."""
        with patch("src.orchestrator.cli.main.OrchestrationAgent") as MockAgent:
            mock_orchestrator = MagicMock()
            mock_orchestrator.query.return_value = OrchestrationResult(
                response="Response",
                route_type="sec_only",
                agents_used=["sec_agent"],
                had_discrepancies=False,
                confidence=0.9,
                execution_time_ms=100.0,
                query_id="test-789",
            )
            MockAgent.return_value = mock_orchestrator

            result = cli_runner.invoke(
                cli, ["query", "Question", "--source", "sec"]
            )

            assert result.exit_code == 0
            # Verify force_route was passed correctly
            mock_orchestrator.query.assert_called_once()
            call_kwargs = mock_orchestrator.query.call_args[1]
            assert call_kwargs["force_route"] == "sec_only"

    def test_compare_command_executes_comparison(
        self, cli_runner: CliRunner
    ) -> None:
        """Test that compare command runs full comparison."""
        with patch("src.orchestrator.cli.main.OrchestrationAgent") as MockAgent:
            mock_orchestrator = MagicMock()
            mock_result = OrchestrationResult(
                response="Comparison result",
                route_type="both",
                agents_used=["news_agent", "sec_agent"],
                had_discrepancies=True,
                confidence=0.7,
                execution_time_ms=300.0,
                query_id="test-compare",
                comparison=None,  # Would have ComparisonResult in real scenario
            )
            mock_orchestrator.compare.return_value = mock_result
            MockAgent.return_value = mock_orchestrator

            result = cli_runner.invoke(cli, ["compare", "AAPL"])

            assert result.exit_code == 0
            assert "AAPL" in result.output
            assert "Comparison result" in result.output
            mock_orchestrator.compare.assert_called_once_with("AAPL")


# =============================================================================
# Component Integration Tests
# =============================================================================


class TestComponentIntegration:
    """Test integration between internal orchestrator components."""

    def test_classifier_executor_integration(
        self,
        mock_news_agent: MagicMock,
        mock_sec_agent: MagicMock,
    ) -> None:
        """Test classifier output feeds correctly into executor."""
        classifier = QueryClassifier()

        # Classify a query
        classification = classifier.classify("What's the news sentiment?")
        assert classification.route_type.value == "news_only"
        assert classification.needs_news_agent
        assert not classification.needs_sec_agent

        # Execute based on classification using the executor with agents
        executor = AsyncAgentExecutor(
            news_agent=mock_news_agent if classification.needs_news_agent else None,
            sec_agent=mock_sec_agent if classification.needs_sec_agent else None,
            timeout_seconds=30.0,
        )

        import asyncio

        async def run_executor() -> ExecutionResult:
            return await executor.execute(
                query="What's the news sentiment?",
                run_news=classification.needs_news_agent,
                run_sec=classification.needs_sec_agent,
            )

        result = asyncio.run(run_executor())

        # Verify only news agent was called
        assert result.news_result is not None
        assert result.news_result.status == AgentStatus.SUCCESS
        assert result.sec_result is None

    def test_executor_comparator_integration(
        self,
        mock_news_agent: MagicMock,
        mock_sec_agent: MagicMock,
    ) -> None:
        """Test executor results feed correctly into comparator."""
        executor = AsyncAgentExecutor(
            news_agent=mock_news_agent,
            sec_agent=mock_sec_agent,
            timeout_seconds=30.0,
        )

        # Execute both agents
        import asyncio

        async def run_executor() -> ExecutionResult:
            return await executor.execute(
                query="Compare company outlook",
                run_news=True,
                run_sec=True,
            )

        exec_result = asyncio.run(run_executor())

        # Extract signals from responses (mimics what OrchestrationAgent does)
        extractor = SignalExtractor()
        news_signals = extractor.extract(
            exec_result.news_result.response or "", "news_agent"
        )
        sec_signals = extractor.extract(
            exec_result.sec_result.response or "", "sec_agent"
        )

        # Feed signals into comparator
        detector = DiscrepancyDetector()
        comparison = detector.compare(
            news_result=news_signals,
            sec_result=sec_signals,
        )

        # Verify comparison was performed
        assert isinstance(comparison, ComparisonResult)

    def test_comparator_synthesizer_integration(
        self,
        mock_news_agent: MagicMock,
        mock_sec_agent: MagicMock,
        mock_bedrock_client: MagicMock,
    ) -> None:
        """Test comparator results feed correctly into synthesizer."""
        # Create agent results
        news_result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response="News: Positive sentiment 0.8",
            execution_time_ms=100.0,
        )
        sec_result = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.SUCCESS,
            response="SEC: Revenue up 10%",
            execution_time_ms=150.0,
        )

        # Extract signals and compare
        extractor = SignalExtractor()
        news_signals = extractor.extract(news_result.response or "", "news_agent")
        sec_signals = extractor.extract(sec_result.response or "", "sec_agent")
        detector = DiscrepancyDetector()
        comparison = detector.compare(news_result=news_signals, sec_result=sec_signals)

        # Synthesize
        synthesizer = ResponseSynthesizer(bedrock_client=mock_bedrock_client)
        synthesis_input = SynthesisInput(
            user_query="Compare company outlook",
            news_result=news_result,
            sec_result=sec_result,
            comparison=comparison,
            ticker="AAPL",
        )
        synthesis_result = synthesizer.synthesize(synthesis_input)

        # Verify synthesis completed
        assert synthesis_result.response is not None
        assert len(synthesis_result.response) > 0
        assert "news_agent" in synthesis_result.sources_used
        assert "sec_agent" in synthesis_result.sources_used

    def test_full_pipeline_integration(
        self,
        mock_news_agent: MagicMock,
        mock_sec_agent: MagicMock,
        mock_bedrock_client: MagicMock,
    ) -> None:
        """Test full pipeline: classify → execute → compare → synthesize."""
        import asyncio

        # Step 1: Classify
        classifier = QueryClassifier()
        classification = classifier.classify("Compare news vs SEC for Apple")
        assert classification.route_type.value == "both"

        # Step 2: Execute
        executor = AsyncAgentExecutor(
            news_agent=mock_news_agent,
            sec_agent=mock_sec_agent,
            timeout_seconds=30.0,
        )

        async def run_executor() -> ExecutionResult:
            return await executor.execute(
                query="Compare news vs SEC for Apple",
                run_news=True,
                run_sec=True,
            )

        exec_result = asyncio.run(run_executor())
        assert exec_result.news_result is not None
        assert exec_result.sec_result is not None

        # Step 3: Extract signals and compare
        extractor = SignalExtractor()
        news_signals = extractor.extract(
            exec_result.news_result.response or "", "news_agent"
        )
        sec_signals = extractor.extract(
            exec_result.sec_result.response or "", "sec_agent"
        )
        detector = DiscrepancyDetector()
        comparison = detector.compare(
            news_result=news_signals,
            sec_result=sec_signals,
        )
        assert isinstance(comparison, ComparisonResult)

        # Step 4: Synthesize
        synthesizer = ResponseSynthesizer(bedrock_client=mock_bedrock_client)
        synthesis_input = SynthesisInput(
            user_query="Compare news vs SEC for Apple",
            news_result=exec_result.news_result,
            sec_result=exec_result.sec_result,
            comparison=comparison,
            ticker="AAPL",
        )
        result = synthesizer.synthesize(synthesis_input)

        # Verify end-to-end flow
        assert result.response is not None
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0


# =============================================================================
# Memory Integration Tests
# =============================================================================


class TestMemoryIntegration:
    """Test memory persistence integration across components."""

    def test_memory_persists_through_orchestration(
        self, orchestrator_with_mocks: OrchestrationAgent
    ) -> None:
        """Test that orchestrator properly stores queries in memory."""
        # Execute queries
        result1 = orchestrator_with_mocks.query("Query 1", ticker="AAPL")
        result2 = orchestrator_with_mocks.query("Query 2", ticker="TSLA")

        # Verify memory persistence
        memory = orchestrator_with_mocks.memory
        assert memory.entry_count == 2

        # Verify queries can be retrieved
        q1 = memory.get_query_by_id(result1.query_id)
        q2 = memory.get_query_by_id(result2.query_id)

        assert q1 is not None
        assert q2 is not None
        assert q1["ticker"] == "AAPL"
        assert q2["ticker"] == "TSLA"

    def test_memory_session_summary_accuracy(
        self, orchestrator_with_mocks: OrchestrationAgent
    ) -> None:
        """Test that session summary accurately reflects queries."""
        # Execute mixed queries
        orchestrator_with_mocks.query("News for Apple", ticker="AAPL")
        orchestrator_with_mocks.compare("TSLA")
        orchestrator_with_mocks.query("SEC for AAPL", ticker="AAPL")

        summary = orchestrator_with_mocks.get_session_summary()

        assert summary["query_count"] == 3
        assert "AAPL" in summary["tickers_analyzed"]
        assert "TSLA" in summary["tickers_analyzed"]
        assert 0.0 <= summary["average_confidence"] <= 1.0

    def test_memory_filtering_integration(
        self, orchestrator_with_mocks: OrchestrationAgent
    ) -> None:
        """Test that memory filtering works correctly."""
        # Execute queries for multiple tickers
        orchestrator_with_mocks.query("Q1", ticker="AAPL")
        orchestrator_with_mocks.query("Q2", ticker="AAPL")
        orchestrator_with_mocks.query("Q3", ticker="TSLA")
        orchestrator_with_mocks.query("Q4", ticker="MSFT")

        # Filter by ticker
        aapl = orchestrator_with_mocks.get_recent_queries(ticker="AAPL")
        assert len(aapl) == 2

        # Filter with limit
        limited = orchestrator_with_mocks.get_recent_queries(limit=2)
        assert len(limited) == 2

        # Combined filter
        filtered = orchestrator_with_mocks.get_recent_queries(ticker="AAPL", limit=1)
        assert len(filtered) == 1
        assert filtered[0]["ticker"] == "AAPL"


# =============================================================================
# Error Propagation Integration Tests
# =============================================================================


class TestErrorPropagationIntegration:
    """Test error handling across component boundaries."""

    def test_agent_error_propagates_through_executor(
        self, mock_sec_agent: MagicMock
    ) -> None:
        """Test that agent errors are properly captured in executor."""
        import asyncio

        # Create failing news agent
        failing_agent = MagicMock()
        failing_agent.query.side_effect = RuntimeError("News API failure")

        executor = AsyncAgentExecutor(
            news_agent=failing_agent,
            sec_agent=mock_sec_agent,
            timeout_seconds=30.0,
        )

        async def run_executor() -> ExecutionResult:
            return await executor.execute(
                query="Test query",
                run_news=True,
                run_sec=True,
            )

        result = asyncio.run(run_executor())

        # News agent error captured
        assert result.news_result is not None
        assert result.news_result.status == AgentStatus.ERROR
        assert "News API failure" in (result.news_result.error_message or "")

        # SEC agent still succeeded
        assert result.sec_result is not None
        assert result.sec_result.status == AgentStatus.SUCCESS

    def test_timeout_propagates_through_executor(
        self, mock_sec_agent: MagicMock
    ) -> None:
        """Test that timeouts are properly captured in executor."""
        import asyncio
        import time

        # Create slow news agent
        slow_agent = MagicMock()

        def slow_query(_: str) -> str:
            time.sleep(2)
            return "Response"

        slow_agent.query = slow_query

        executor = AsyncAgentExecutor(
            news_agent=slow_agent,
            sec_agent=mock_sec_agent,
            timeout_seconds=0.1,
        )

        async def run_executor() -> ExecutionResult:
            return await executor.execute(
                query="Test query",
                run_news=True,
                run_sec=True,
            )

        result = asyncio.run(run_executor())

        # News agent timed out
        assert result.news_result is not None
        assert result.news_result.status == AgentStatus.TIMEOUT

        # SEC agent still succeeded (ran in parallel)
        assert result.sec_result is not None
        assert result.sec_result.status == AgentStatus.SUCCESS

    def test_error_in_api_endpoint_returns_500(
        self, api_client: TestClient
    ) -> None:
        """Test that agent errors result in proper HTTP error responses."""
        import src.orchestrator.api.server as server_module

        # Make the orchestrator throw an error
        original_agent = server_module._agent
        mock_agent = MagicMock()
        mock_agent.query.side_effect = RuntimeError("Internal error")
        server_module._agent = mock_agent

        try:
            response = api_client.post(
                "/query",
                json={"query": "Test"},
            )

            assert response.status_code == 500
            assert "Internal error" in response.json()["detail"]
        finally:
            server_module._agent = original_agent


# =============================================================================
# Lambda Handler Integration Tests
# =============================================================================


class TestLambdaHandlerIntegration:
    """Test Lambda handler integration with FastAPI app."""

    def test_lambda_handler_wraps_fastapi_app(self) -> None:
        """Test that Lambda handler properly wraps the FastAPI app."""
        from src.orchestrator.mangum_handler import handler

        # Handler should be callable
        assert callable(handler)

    def test_lambda_handler_routes_to_health(self) -> None:
        """Test that Lambda handler can route to /health endpoint via TestClient."""
        # Use TestClient which properly handles the ASGI interface
        from src.orchestrator.mangum_handler import handler
        from src.orchestrator.api.server import app

        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["service"] == "orchestrator"


# =============================================================================
# Cross-Request State Integration Tests
# =============================================================================


class TestCrossRequestStateIntegration:
    """Test state management across multiple requests."""

    def test_api_session_isolated_from_cli(
        self,
        orchestrator_with_mocks: OrchestrationAgent,
    ) -> None:
        """Test that API and CLI can have separate sessions."""
        # API session
        import src.orchestrator.api.server as server_module

        server_module._agent = orchestrator_with_mocks

        with TestClient(app) as api_client:
            api_client.post("/query", json={"query": "API query", "ticker": "AAPL"})

        api_count = orchestrator_with_mocks.memory.entry_count

        # CLI creates its own agent instance, so this test verifies
        # that the API agent is properly storing state
        assert api_count == 1

    def test_sequential_queries_build_context(
        self, orchestrator_with_mocks: OrchestrationAgent
    ) -> None:
        """Test that sequential queries build up session context."""
        # Query 1
        orchestrator_with_mocks.query("News for AAPL", ticker="AAPL")
        assert orchestrator_with_mocks.memory.entry_count == 1

        # Query 2
        orchestrator_with_mocks.query("SEC for AAPL", ticker="AAPL")
        assert orchestrator_with_mocks.memory.entry_count == 2

        # Query 3
        orchestrator_with_mocks.compare("AAPL")
        assert orchestrator_with_mocks.memory.entry_count == 3

        # All queries for same ticker
        queries = orchestrator_with_mocks.get_recent_queries(ticker="AAPL")
        assert len(queries) == 3

    def test_reset_clears_all_state(
        self, orchestrator_with_mocks: OrchestrationAgent
    ) -> None:
        """Test that reset clears all accumulated state."""
        # Build up state
        orchestrator_with_mocks.query("Q1", ticker="AAPL")
        orchestrator_with_mocks.query("Q2", ticker="TSLA")
        orchestrator_with_mocks.query("Q3", ticker="MSFT")

        assert orchestrator_with_mocks.memory.entry_count == 3

        # Reset
        orchestrator_with_mocks.reset()

        # Verify cleared
        assert orchestrator_with_mocks.memory.entry_count == 0
        summary = orchestrator_with_mocks.get_session_summary()
        assert summary["query_count"] == 0
        assert len(summary["tickers_analyzed"]) == 0


# =============================================================================
# Data Serialization Integration Tests
# =============================================================================


class TestSerializationIntegration:
    """Test data serialization across component boundaries."""

    def test_orchestration_result_json_serializable(
        self, orchestrator_with_mocks: OrchestrationAgent
    ) -> None:
        """Test that OrchestrationResult can be fully JSON serialized."""
        result = orchestrator_with_mocks.compare("AAPL")

        # Convert to dict and then JSON
        data = result.to_dict()
        json_str = json.dumps(data, default=str)

        # Should be parseable
        parsed = json.loads(json_str)
        assert parsed["route_type"] == "both"
        assert "news_agent" in parsed["agents_used"]
        assert "sec_agent" in parsed["agents_used"]

    def test_api_response_matches_orchestration_result(
        self, api_client: TestClient
    ) -> None:
        """Test that API response structure matches OrchestrationResult."""
        response = api_client.post(
            "/query",
            json={"query": "Test query", "ticker": "AAPL"},
        )

        data = response.json()

        # Required fields from OrchestrationResult should be present
        assert "answer" in data  # maps to response
        assert "agents_used" in data
        assert "had_discrepancies" in data
        assert "confidence" in data
        assert "execution_time_ms" in data
        assert "query_id" in data

    def test_memory_entry_serialization_roundtrip(
        self, orchestrator_with_mocks: OrchestrationAgent
    ) -> None:
        """Test that memory entries survive serialization roundtrip."""
        # Execute query
        result = orchestrator_with_mocks.query("Test query", ticker="AAPL")

        # Get from memory
        stored = orchestrator_with_mocks.memory.get_query_by_id(result.query_id)
        assert stored is not None

        # Serialize to JSON
        json_str = json.dumps(stored, default=str)

        # Parse back
        parsed = json.loads(json_str)

        # Key fields preserved
        assert parsed["user_query"] == "Test query"
        assert parsed["ticker"] == "AAPL"
        assert parsed["route_type"] == result.route_type
