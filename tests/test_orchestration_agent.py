"""Tests for the main OrchestrationAgent class."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.orchestrator.agent import OrchestrationAgent, OrchestrationResult
from src.orchestrator.comparison import ComparisonResult
from src.orchestrator.execution import AgentResult, AgentStatus, ExecutionResult
from src.orchestrator.memory import OrchestratorMemory
from src.orchestrator.synthesis import SynthesisResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_news_agent() -> MagicMock:
    """Create a mock news agent."""
    agent = MagicMock()
    agent.query.return_value = "News response: Positive sentiment for AAPL"
    agent.reset.return_value = None
    return agent


@pytest.fixture
def mock_sec_agent() -> MagicMock:
    """Create a mock SEC agent."""
    agent = MagicMock()
    agent.query.return_value = "SEC response: Strong financials in latest 10-K"
    agent.reset.return_value = None
    return agent


@pytest.fixture
def mock_bedrock_client() -> MagicMock:
    """Create a mock Bedrock client."""
    client = MagicMock()
    client.converse.return_value = {
        "output": {
            "message": {
                "content": [{"text": "Synthesized response based on news and SEC data."}]
            }
        }
    }
    return client


@pytest.fixture
def orchestrator(
    mock_news_agent: MagicMock,
    mock_sec_agent: MagicMock,
    mock_bedrock_client: MagicMock,
) -> OrchestrationAgent:
    """Create an orchestration agent with mocked dependencies."""
    return OrchestrationAgent(
        news_agent=mock_news_agent,
        sec_agent=mock_sec_agent,
        bedrock_client=mock_bedrock_client,
        timeout_seconds=10.0,
        enable_comparison=True,
    )


# =============================================================================
# OrchestrationResult Tests
# =============================================================================


class TestOrchestrationResult:
    """Tests for OrchestrationResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating an OrchestrationResult."""
        result = OrchestrationResult(
            response="Test response",
            route_type="both",
            agents_used=["news_agent", "sec_agent"],
            had_discrepancies=False,
            confidence=0.9,
            execution_time_ms=1500.0,
            query_id="test-123",
        )

        assert result.response == "Test response"
        assert result.route_type == "both"
        assert result.agents_used == ["news_agent", "sec_agent"]
        assert result.had_discrepancies is False
        assert result.confidence == 0.9
        assert result.execution_time_ms == 1500.0
        assert result.query_id == "test-123"
        assert result.news_result is None
        assert result.sec_result is None
        assert result.comparison is None

    def test_to_dict_basic(self) -> None:
        """Test converting result to dictionary."""
        result = OrchestrationResult(
            response="Test response",
            route_type="news_only",
            agents_used=["news_agent"],
            had_discrepancies=False,
            confidence=0.8,
            execution_time_ms=500.0,
            query_id="test-456",
        )

        data = result.to_dict()

        assert data["response"] == "Test response"
        assert data["route_type"] == "news_only"
        assert data["agents_used"] == ["news_agent"]
        assert data["had_discrepancies"] is False
        assert data["confidence"] == 0.8
        assert data["execution_time_ms"] == 500.0
        assert data["query_id"] == "test-456"
        assert data["news_result"] is None
        assert data["sec_result"] is None
        assert data["comparison"] is None

    def test_to_dict_with_agent_results(self) -> None:
        """Test converting result with agent results to dictionary."""
        news_result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response="News data",
            execution_time_ms=200.0,
        )

        result = OrchestrationResult(
            response="Test response",
            route_type="news_only",
            agents_used=["news_agent"],
            had_discrepancies=False,
            confidence=0.8,
            execution_time_ms=500.0,
            query_id="test-789",
            news_result=news_result,
        )

        data = result.to_dict()

        assert data["news_result"] is not None
        assert data["news_result"]["agent_name"] == "news_agent"
        assert data["news_result"]["status"] == "success"


# =============================================================================
# OrchestrationAgent Initialization Tests
# =============================================================================


class TestOrchestrationAgentInit:
    """Tests for OrchestrationAgent initialization."""

    def test_init_defaults(self, mock_bedrock_client: MagicMock) -> None:
        """Test initialization with defaults."""
        with patch.dict("os.environ", {}, clear=True):
            agent = OrchestrationAgent(bedrock_client=mock_bedrock_client)

        assert agent.model_id == "anthropic.claude-opus-4-5-20251101-v1:0"
        assert agent.timeout_seconds == 60.0
        assert agent.enable_comparison is True
        assert agent._news_agent is None
        assert agent._sec_agent is None
        assert agent._agents_initialized is False

    def test_init_with_env_vars(self, mock_bedrock_client: MagicMock) -> None:
        """Test initialization with environment variables."""
        env_vars = {
            "BEDROCK_MODEL_ID": "custom-model-id",
            "ORCHESTRATOR_TIMEOUT_SECONDS": "120",
            "ORCHESTRATOR_ENABLE_COMPARISON": "false",
        }
        with patch.dict("os.environ", env_vars, clear=True):
            agent = OrchestrationAgent(bedrock_client=mock_bedrock_client)

        assert agent.model_id == "custom-model-id"
        assert agent.timeout_seconds == 120.0
        assert agent.enable_comparison is False

    def test_init_with_custom_values(
        self,
        mock_news_agent: MagicMock,
        mock_sec_agent: MagicMock,
        mock_bedrock_client: MagicMock,
    ) -> None:
        """Test initialization with custom values."""
        memory = OrchestratorMemory(session_id="test-session")

        agent = OrchestrationAgent(
            model_id="custom-model",
            bedrock_client=mock_bedrock_client,
            news_agent=mock_news_agent,
            sec_agent=mock_sec_agent,
            timeout_seconds=30.0,
            enable_comparison=False,
            memory=memory,
        )

        assert agent.model_id == "custom-model"
        assert agent.bedrock_client is mock_bedrock_client
        assert agent._news_agent is mock_news_agent
        assert agent._sec_agent is mock_sec_agent
        assert agent.timeout_seconds == 30.0
        assert agent.enable_comparison is False
        assert agent._memory is memory
        assert agent._agents_initialized is True

    def test_memory_property(self, orchestrator: OrchestrationAgent) -> None:
        """Test memory property access."""
        assert isinstance(orchestrator.memory, OrchestratorMemory)

    def test_session_id_property(self, orchestrator: OrchestrationAgent) -> None:
        """Test session_id property access."""
        assert isinstance(orchestrator.session_id, str)
        assert len(orchestrator.session_id) > 0


# =============================================================================
# Ticker Extraction Tests
# =============================================================================


class TestTickerExtraction:
    """Tests for ticker symbol extraction from queries."""

    def test_extract_ticker_company_name(self, orchestrator: OrchestrationAgent) -> None:
        """Test extracting ticker from company name."""
        assert orchestrator._extract_ticker_from_query("What about Apple?") == "AAPL"
        assert orchestrator._extract_ticker_from_query("How is Tesla doing?") == "TSLA"
        assert orchestrator._extract_ticker_from_query("Tell me about Microsoft") == "MSFT"

    def test_extract_ticker_dollar_sign(self, orchestrator: OrchestrationAgent) -> None:
        """Test extracting ticker with dollar sign prefix."""
        assert orchestrator._extract_ticker_from_query("What about $AAPL?") == "AAPL"
        assert orchestrator._extract_ticker_from_query("$TSLA stock price") == "TSLA"

    def test_extract_ticker_explicit(self, orchestrator: OrchestrationAgent) -> None:
        """Test extracting explicitly mentioned ticker."""
        assert orchestrator._extract_ticker_from_query("ticker: NVDA analysis") == "NVDA"

    def test_extract_ticker_with_stock(self, orchestrator: OrchestrationAgent) -> None:
        """Test extracting ticker followed by 'stock'."""
        result = orchestrator._extract_ticker_from_query("AMZN stock analysis")
        assert result == "AMZN"

    def test_extract_ticker_no_match(self, orchestrator: OrchestrationAgent) -> None:
        """Test when no ticker is found."""
        assert orchestrator._extract_ticker_from_query("General market overview") is None

    def test_extract_ticker_filters_common_words(
        self, orchestrator: OrchestrationAgent
    ) -> None:
        """Test that common words are not extracted as tickers."""
        # "THE" and "FOR" should be filtered out
        result = orchestrator._extract_ticker_from_query("What are the risks for this?")
        assert result is None


# =============================================================================
# Query Routing Tests
# =============================================================================


class TestQueryRouting:
    """Tests for query classification and routing."""

    def test_route_news_query(self, orchestrator: OrchestrationAgent) -> None:
        """Test routing a news-focused query."""
        result = orchestrator.query("What's the latest news sentiment for AAPL?")

        assert result.route_type == "news_only"
        assert "news_agent" in result.agents_used

    def test_route_sec_query(self, orchestrator: OrchestrationAgent) -> None:
        """Test routing an SEC-focused query."""
        result = orchestrator.query("What are the risk factors in Tesla's 10-K?")

        assert result.route_type == "sec_only"
        assert "sec_agent" in result.agents_used

    def test_route_both_query(self, orchestrator: OrchestrationAgent) -> None:
        """Test routing a query requiring both agents."""
        result = orchestrator.query("Compare news sentiment vs SEC filings for AAPL")

        assert result.route_type == "both"
        assert "news_agent" in result.agents_used
        assert "sec_agent" in result.agents_used

    def test_force_route_news_only(self, orchestrator: OrchestrationAgent) -> None:
        """Test forcing route to news_only."""
        result = orchestrator.query(
            "What are the 10-K risk factors?",  # Would normally route to SEC
            force_route="news_only",
        )

        assert result.route_type == "news_only"
        assert "news_agent" in result.agents_used
        assert "sec_agent" not in result.agents_used

    def test_force_route_sec_only(self, orchestrator: OrchestrationAgent) -> None:
        """Test forcing route to sec_only."""
        result = orchestrator.query(
            "What's the news sentiment?",  # Would normally route to news
            force_route="sec_only",
        )

        assert result.route_type == "sec_only"
        assert "sec_agent" in result.agents_used
        assert "news_agent" not in result.agents_used

    def test_force_route_both(self, orchestrator: OrchestrationAgent) -> None:
        """Test forcing route to both agents."""
        result = orchestrator.query(
            "What's the news?",  # Would normally route to news only
            force_route="both",
        )

        assert result.route_type == "both"
        assert "news_agent" in result.agents_used
        assert "sec_agent" in result.agents_used


# =============================================================================
# Query Execution Tests
# =============================================================================


class TestQueryExecution:
    """Tests for query execution."""

    def test_query_returns_orchestration_result(
        self, orchestrator: OrchestrationAgent
    ) -> None:
        """Test that query returns an OrchestrationResult."""
        result = orchestrator.query("What's happening with AAPL?")

        assert isinstance(result, OrchestrationResult)
        assert isinstance(result.response, str)
        assert len(result.response) > 0

    def test_query_with_ticker(self, orchestrator: OrchestrationAgent) -> None:
        """Test query with explicit ticker."""
        result = orchestrator.query("What's the outlook?", ticker="TSLA")

        assert result.query_id is not None

    def test_query_extracts_ticker(self, orchestrator: OrchestrationAgent) -> None:
        """Test that ticker is extracted from query."""
        result = orchestrator.query("How is Apple performing?")

        # The query should have stored the ticker in memory
        last_query = orchestrator.memory.get_last_query()
        assert last_query is not None
        assert last_query.get("ticker") == "AAPL"

    def test_query_stores_in_memory(self, orchestrator: OrchestrationAgent) -> None:
        """Test that query is stored in memory."""
        initial_count = orchestrator.memory.entry_count

        orchestrator.query("What's the news for AAPL?")

        assert orchestrator.memory.entry_count == initial_count + 1

    def test_query_returns_query_id(self, orchestrator: OrchestrationAgent) -> None:
        """Test that query returns a valid query_id."""
        result = orchestrator.query("What's happening?")

        assert result.query_id is not None
        assert len(result.query_id) > 0

        # Verify query can be retrieved by ID
        stored = orchestrator.memory.get_query_by_id(result.query_id)
        assert stored is not None

    def test_query_tracks_execution_time(self, orchestrator: OrchestrationAgent) -> None:
        """Test that execution time is tracked."""
        result = orchestrator.query("What's happening?")

        assert result.execution_time_ms > 0

    def test_query_includes_confidence(self, orchestrator: OrchestrationAgent) -> None:
        """Test that confidence score is included."""
        result = orchestrator.query("What's happening?")

        assert 0.0 <= result.confidence <= 1.0


# =============================================================================
# Async Query Tests
# =============================================================================


class TestAsyncQuery:
    """Tests for async query execution."""

    def test_query_async(self, orchestrator: OrchestrationAgent) -> None:
        """Test async query execution."""
        async def run_test() -> OrchestrationResult:
            return await orchestrator.query_async("What's happening with AAPL?")

        result = asyncio.run(run_test())

        assert isinstance(result, OrchestrationResult)
        assert isinstance(result.response, str)

    def test_query_async_with_force_route(
        self, orchestrator: OrchestrationAgent
    ) -> None:
        """Test async query with forced routing."""
        async def run_test() -> OrchestrationResult:
            return await orchestrator.query_async(
                "What's happening?",
                ticker="TSLA",
                force_route="both",
            )

        result = asyncio.run(run_test())

        assert result.route_type == "both"


# =============================================================================
# Comparison Tests
# =============================================================================


class TestComparison:
    """Tests for comparison functionality."""

    def test_compare_method(self, orchestrator: OrchestrationAgent) -> None:
        """Test the compare convenience method."""
        result = orchestrator.compare("AAPL")

        assert result.route_type == "both"
        assert "news_agent" in result.agents_used
        assert "sec_agent" in result.agents_used

    def test_comparison_disabled(
        self,
        mock_news_agent: MagicMock,
        mock_sec_agent: MagicMock,
        mock_bedrock_client: MagicMock,
    ) -> None:
        """Test when comparison is disabled."""
        agent = OrchestrationAgent(
            news_agent=mock_news_agent,
            sec_agent=mock_sec_agent,
            bedrock_client=mock_bedrock_client,
            enable_comparison=False,
        )

        result = agent.query("Compare news vs SEC for AAPL", force_route="both")

        # Comparison should be None when disabled
        assert result.comparison is None


# =============================================================================
# Reset Tests
# =============================================================================


class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_memory(self, orchestrator: OrchestrationAgent) -> None:
        """Test that reset clears memory."""
        orchestrator.query("Test query 1")
        orchestrator.query("Test query 2")

        assert orchestrator.memory.entry_count > 0

        orchestrator.reset()

        assert orchestrator.memory.entry_count == 0

    def test_reset_calls_sub_agent_reset(
        self,
        orchestrator: OrchestrationAgent,
        mock_news_agent: MagicMock,
        mock_sec_agent: MagicMock,
    ) -> None:
        """Test that reset calls reset on sub-agents."""
        orchestrator.reset()

        mock_news_agent.reset.assert_called_once()
        mock_sec_agent.reset.assert_called_once()


# =============================================================================
# Session Management Tests
# =============================================================================


class TestSessionManagement:
    """Tests for session management functionality."""

    def test_get_session_summary(self, orchestrator: OrchestrationAgent) -> None:
        """Test getting session summary."""
        orchestrator.query("What about AAPL?")
        orchestrator.query("Tell me about TSLA")

        summary = orchestrator.get_session_summary()

        assert "query_count" in summary
        assert summary["query_count"] == 2
        assert "tickers_analyzed" in summary

    def test_get_recent_queries(self, orchestrator: OrchestrationAgent) -> None:
        """Test getting recent queries."""
        orchestrator.query("Query 1 about AAPL")
        orchestrator.query("Query 2 about TSLA")
        orchestrator.query("Query 3 about AAPL")

        queries = orchestrator.get_recent_queries()

        assert len(queries) == 3

    def test_get_recent_queries_with_ticker_filter(
        self, orchestrator: OrchestrationAgent
    ) -> None:
        """Test filtering recent queries by ticker."""
        orchestrator.query("Query about Apple", ticker="AAPL")
        orchestrator.query("Query about Tesla", ticker="TSLA")
        orchestrator.query("Another Apple query", ticker="AAPL")

        aapl_queries = orchestrator.get_recent_queries(ticker="AAPL")

        assert len(aapl_queries) == 2
        for q in aapl_queries:
            assert q["ticker"] == "AAPL"

    def test_get_recent_queries_with_limit(
        self, orchestrator: OrchestrationAgent
    ) -> None:
        """Test limiting recent queries."""
        for i in range(5):
            orchestrator.query(f"Query {i}")

        queries = orchestrator.get_recent_queries(limit=3)

        assert len(queries) == 3


# =============================================================================
# Lazy Initialization Tests
# =============================================================================


class TestLazyInitialization:
    """Tests for lazy agent initialization."""

    def test_agents_not_initialized_on_creation(
        self, mock_bedrock_client: MagicMock
    ) -> None:
        """Test that agents are not initialized on creation."""
        agent = OrchestrationAgent(bedrock_client=mock_bedrock_client)

        assert agent._news_agent is None
        assert agent._sec_agent is None
        assert agent._agents_initialized is False

    @patch("src.orchestrator.agent.NewsSentimentAgent")
    @patch("src.orchestrator.agent.SECFilingsAgent")
    def test_agents_initialized_on_first_query(
        self,
        mock_sec_class: MagicMock,
        mock_news_class: MagicMock,
        mock_bedrock_client: MagicMock,
    ) -> None:
        """Test that agents are lazily initialized on first query."""
        mock_news_instance = MagicMock()
        mock_news_instance.query.return_value = "News response"
        mock_news_class.return_value = mock_news_instance

        mock_sec_instance = MagicMock()
        mock_sec_instance.query.return_value = "SEC response"
        mock_sec_class.return_value = mock_sec_instance

        agent = OrchestrationAgent(bedrock_client=mock_bedrock_client)

        assert not agent._agents_initialized

        # Trigger initialization by calling query
        agent.query("Test query", force_route="news_only")

        assert agent._agents_initialized
        mock_news_class.assert_called_once()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_handles_agent_timeout(
        self,
        mock_sec_agent: MagicMock,
        mock_bedrock_client: MagicMock,
    ) -> None:
        """Test handling of agent timeout."""
        # Create a news agent that times out
        slow_news_agent = MagicMock()

        def slow_query(_: str) -> str:
            import time

            time.sleep(2)  # Sleep longer than timeout
            return "Response"

        slow_news_agent.query = slow_query

        agent = OrchestrationAgent(
            news_agent=slow_news_agent,
            sec_agent=mock_sec_agent,
            bedrock_client=mock_bedrock_client,
            timeout_seconds=0.1,  # Very short timeout
        )

        result = agent.query("Test", force_route="news_only")

        # Should still return a result (with timeout error handled)
        assert isinstance(result, OrchestrationResult)
        assert result.news_result is not None
        assert result.news_result.status == AgentStatus.TIMEOUT

    def test_handles_agent_exception(
        self,
        mock_sec_agent: MagicMock,
        mock_bedrock_client: MagicMock,
    ) -> None:
        """Test handling of agent exception."""
        # Create a news agent that raises an exception
        failing_news_agent = MagicMock()
        failing_news_agent.query.side_effect = Exception("Agent failed")

        agent = OrchestrationAgent(
            news_agent=failing_news_agent,
            sec_agent=mock_sec_agent,
            bedrock_client=mock_bedrock_client,
        )

        result = agent.query("Test", force_route="news_only")

        # Should still return a result (with error handled)
        assert isinstance(result, OrchestrationResult)
        assert result.news_result is not None
        assert result.news_result.status == AgentStatus.ERROR
        assert "Agent failed" in (result.news_result.error_message or "")


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the orchestration agent."""

    def test_full_orchestration_flow(self, orchestrator: OrchestrationAgent) -> None:
        """Test full orchestration flow from query to response."""
        result = orchestrator.query(
            "Compare the news sentiment with SEC filings for Apple",
            ticker="AAPL",
        )

        # Verify result structure
        assert isinstance(result, OrchestrationResult)
        assert result.response is not None
        assert len(result.response) > 0
        assert result.route_type == "both"
        assert "news_agent" in result.agents_used
        assert "sec_agent" in result.agents_used
        assert result.query_id is not None
        assert result.execution_time_ms > 0

        # Verify memory storage
        stored_query = orchestrator.memory.get_query_by_id(result.query_id)
        assert stored_query is not None
        assert stored_query["user_query"] == "Compare the news sentiment with SEC filings for Apple"
        assert stored_query["ticker"] == "AAPL"

    def test_multiple_queries_session(self, orchestrator: OrchestrationAgent) -> None:
        """Test multiple queries in a session."""
        # Query 1: News only
        result1 = orchestrator.query("What's the news for AAPL?")

        # Query 2: SEC only
        result2 = orchestrator.query("What are the 10-K risk factors for TSLA?")

        # Query 3: Both
        result3 = orchestrator.query("Compare news vs SEC for MSFT")

        # Verify all queries stored
        assert orchestrator.memory.entry_count == 3

        # Verify each query routed correctly
        assert result1.route_type == "news_only"
        assert result2.route_type == "sec_only"
        assert result3.route_type == "both"

        # Verify session summary
        summary = orchestrator.get_session_summary()
        assert summary["query_count"] == 3
