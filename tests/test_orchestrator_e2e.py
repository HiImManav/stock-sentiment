"""End-to-end integration tests for the OrchestrationAgent.

These tests validate the full orchestration workflow including:
- Complete query lifecycle from input to response
- Parallel execution of sub-agents
- Signal extraction and discrepancy detection
- Response synthesis
- Memory tracking and session management
- Error handling and partial failure scenarios
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.agent import OrchestrationAgent, OrchestrationResult
from orchestrator.comparison import ComparisonResult
from orchestrator.execution import AgentResult, AgentStatus
from orchestrator.memory import OrchestratorMemory


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def realistic_news_response() -> str:
    """Create a realistic news agent response with extractable signals."""
    return """
Based on my analysis of recent news for Apple (AAPL):

**Overall Sentiment: POSITIVE**

The news coverage for Apple over the past 30 days has been predominantly positive,
with a sentiment score of 0.65.

**Key Findings:**
- Strong iPhone 15 sales reported across multiple markets
- Apple's services revenue continues to grow at 15% year-over-year
- Market analysts have upgraded price targets following Q4 earnings beat

**Material Events:**
1. Q4 2024 earnings exceeded expectations with EPS of $2.18 vs $2.10 expected
2. iPhone 15 Pro Max experiencing supply constraints due to high demand
3. New partnership with major enterprise clients announced

**Forward Outlook:**
Analysts remain bullish on Apple's prospects, citing strong ecosystem lock-in
and growing services revenue. The consensus 12-month price target is $210.

**Confidence Level:** High (based on 45 articles analyzed)
"""


@pytest.fixture
def realistic_sec_response() -> str:
    """Create a realistic SEC agent response with extractable signals."""
    return """
Based on my analysis of Apple Inc.'s (AAPL) recent SEC filings:

**Latest 10-K Summary:**

**Financial Highlights:**
- Total Revenue: $383.3 billion (down 3% YoY)
- Net Income: $97.0 billion
- Gross Margin: 44.1%
- Cash and equivalents: $61.6 billion

**Risk Factors (Key Items):**
1. Global economic conditions and consumer spending uncertainty
2. Supply chain concentration in specific regions
3. Intense competition in smartphone and services markets
4. Regulatory scrutiny of App Store practices in multiple jurisdictions

**Management Discussion:**
Management highlighted the continued growth in Services revenue as a strategic
priority. Hardware revenue faced headwinds due to economic conditions, but
the installed base of active devices reached an all-time high of 2.2 billion.

**Forward Guidance:**
The company expects modest revenue growth in the coming fiscal year, with
Services revenue expected to grow at double-digit rates.

**Key Concerns:**
- Revenue decline year-over-year indicates potential market saturation
- Ongoing regulatory challenges in EU and US regarding App Store
"""


@pytest.fixture
def conflicting_news_response() -> str:
    """Create a news response that conflicts with SEC data."""
    return """
Based on my analysis of recent news for Apple (AAPL):

**Overall Sentiment: VERY POSITIVE**

Apple is experiencing explosive growth according to recent media coverage.

**Key Findings:**
- Multiple reports cite "unprecedented demand" for Apple products
- Revenue growth expected to accelerate significantly
- Analysts predict "record-breaking" quarter ahead

**Forward Outlook:**
The narrative is extremely bullish, with expectations of 25%+ revenue growth.

**Sentiment Score: 0.9 (Very Positive)**
"""


@pytest.fixture
def mock_news_agent_realistic(realistic_news_response: str) -> MagicMock:
    """Create a mock news agent with realistic response."""
    agent = MagicMock()
    agent.query.return_value = realistic_news_response
    agent.reset.return_value = None
    return agent


@pytest.fixture
def mock_sec_agent_realistic(realistic_sec_response: str) -> MagicMock:
    """Create a mock SEC agent with realistic response."""
    agent = MagicMock()
    agent.query.return_value = realistic_sec_response
    agent.reset.return_value = None
    return agent


@pytest.fixture
def mock_bedrock_client_synthesizer() -> MagicMock:
    """Create a mock Bedrock client for the synthesizer."""
    client = MagicMock()
    client.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {
                        "text": """
**Apple (AAPL) Analysis Summary**

Based on analysis from both news sources and SEC filings:

**Overall Assessment: CAUTIOUSLY POSITIVE**

**Agreement Points:**
- Both sources confirm strong Services revenue growth
- iPhone remains the key revenue driver
- Company maintains strong cash position

**Key Discrepancies Identified:**
1. News reports "strong growth" narrative, but SEC 10-K shows 3% revenue decline YoY
2. News sentiment is highly positive (0.65), but SEC filings highlight significant risk factors
3. Forward guidance in SEC is "modest growth" vs news expectation of acceleration

**Recommendation:**
While near-term sentiment is positive, investors should note the discrepancy between
media narrative and actual financial performance disclosed in SEC filings.

**Confidence: Medium** (due to narrative vs. filing discrepancies)
"""
                    }
                ]
            }
        }
    }
    return client


@pytest.fixture
def orchestrator_e2e(
    mock_news_agent_realistic: MagicMock,
    mock_sec_agent_realistic: MagicMock,
    mock_bedrock_client_synthesizer: MagicMock,
) -> OrchestrationAgent:
    """Create an orchestrator for end-to-end testing."""
    return OrchestrationAgent(
        news_agent=mock_news_agent_realistic,
        sec_agent=mock_sec_agent_realistic,
        bedrock_client=mock_bedrock_client_synthesizer,
        timeout_seconds=30.0,
        enable_comparison=True,
    )


# =============================================================================
# End-to-End Query Lifecycle Tests
# =============================================================================


class TestE2EQueryLifecycle:
    """Test the complete query lifecycle from input to response."""

    def test_full_query_lifecycle_both_agents(
        self, orchestrator_e2e: OrchestrationAgent
    ) -> None:
        """Test complete query lifecycle routing to both agents."""
        result = orchestrator_e2e.query(
            "Compare the news sentiment with SEC filings for Apple",
            ticker="AAPL",
        )

        # Verify result structure is complete
        assert isinstance(result, OrchestrationResult)
        assert result.response is not None
        assert len(result.response) > 0
        assert result.route_type == "both"
        assert "news_agent" in result.agents_used
        assert "sec_agent" in result.agents_used
        assert result.query_id is not None
        assert result.execution_time_ms >= 0

        # Verify both agent results are present
        assert result.news_result is not None
        assert result.sec_result is not None
        assert result.news_result.status == AgentStatus.SUCCESS
        assert result.sec_result.status == AgentStatus.SUCCESS

        # Verify comparison was performed
        assert result.comparison is not None
        assert isinstance(result.comparison, ComparisonResult)

        # Verify confidence is calculated
        assert 0.0 <= result.confidence <= 1.0

    def test_query_lifecycle_news_only(
        self, orchestrator_e2e: OrchestrationAgent
    ) -> None:
        """Test query lifecycle with news-only routing."""
        result = orchestrator_e2e.query(
            "What's the latest news sentiment for Apple?",
            ticker="AAPL",
        )

        assert result.route_type == "news_only"
        assert "news_agent" in result.agents_used
        assert "sec_agent" not in result.agents_used
        assert result.news_result is not None
        assert result.sec_result is None
        # No comparison when only one agent runs
        assert result.comparison is None

    def test_query_lifecycle_sec_only(
        self, orchestrator_e2e: OrchestrationAgent
    ) -> None:
        """Test query lifecycle with SEC-only routing."""
        result = orchestrator_e2e.query(
            "What are the risk factors in Apple's 10-K filing?",
            ticker="AAPL",
        )

        assert result.route_type == "sec_only"
        assert "sec_agent" in result.agents_used
        assert "news_agent" not in result.agents_used
        assert result.sec_result is not None
        assert result.news_result is None
        assert result.comparison is None

    def test_query_result_serialization(
        self, orchestrator_e2e: OrchestrationAgent
    ) -> None:
        """Test that query results can be fully serialized."""
        result = orchestrator_e2e.query(
            "Compare news vs SEC for Apple",
            ticker="AAPL",
            force_route="both",
        )

        # Convert to dict
        data = result.to_dict()

        # Verify all fields are present and serializable
        assert isinstance(data, dict)
        assert data["response"] == result.response
        assert data["route_type"] == "both"
        assert data["agents_used"] == ["news_agent", "sec_agent"]
        assert data["had_discrepancies"] == result.had_discrepancies
        assert data["confidence"] == result.confidence
        assert data["execution_time_ms"] == result.execution_time_ms
        assert data["query_id"] == result.query_id

        # Verify nested results are serialized
        assert data["news_result"] is not None
        assert data["sec_result"] is not None
        assert data["news_result"]["status"] == "success"
        assert data["sec_result"]["status"] == "success"

        # Verify comparison is serialized
        assert data["comparison"] is not None


# =============================================================================
# Signal Extraction and Comparison Tests
# =============================================================================


class TestE2ESignalExtraction:
    """Test signal extraction and discrepancy detection in e2e flow."""

    def test_signals_extracted_from_realistic_responses(
        self, orchestrator_e2e: OrchestrationAgent
    ) -> None:
        """Test that signals are properly extracted from realistic responses."""
        result = orchestrator_e2e.query(
            "Compare Apple news vs filings",
            ticker="AAPL",
            force_route="both",
        )

        # Comparison should be present when both agents run
        assert result.comparison is not None

        # Check that comparison has analyzed the sources
        assert result.comparison.news_result is not None or result.comparison.sec_result is not None

    def test_discrepancy_detection_with_conflicting_data(
        self,
        mock_sec_agent_realistic: MagicMock,
        conflicting_news_response: str,
        mock_bedrock_client_synthesizer: MagicMock,
    ) -> None:
        """Test discrepancy detection when news conflicts with SEC data."""
        # Create news agent with conflicting optimistic response
        conflicting_news_agent = MagicMock()
        conflicting_news_agent.query.return_value = conflicting_news_response
        conflicting_news_agent.reset.return_value = None

        orchestrator = OrchestrationAgent(
            news_agent=conflicting_news_agent,
            sec_agent=mock_sec_agent_realistic,
            bedrock_client=mock_bedrock_client_synthesizer,
            enable_comparison=True,
        )

        result = orchestrator.query(
            "Compare news vs SEC for Apple",
            ticker="AAPL",
            force_route="both",
        )

        # Comparison should detect discrepancies between overly positive news
        # and more measured SEC data
        assert result.comparison is not None
        # The comparison should have been performed
        assert result.had_discrepancies is not None

    def test_comparison_disabled_skips_signal_extraction(
        self,
        mock_news_agent_realistic: MagicMock,
        mock_sec_agent_realistic: MagicMock,
        mock_bedrock_client_synthesizer: MagicMock,
    ) -> None:
        """Test that comparison is skipped when disabled."""
        orchestrator = OrchestrationAgent(
            news_agent=mock_news_agent_realistic,
            sec_agent=mock_sec_agent_realistic,
            bedrock_client=mock_bedrock_client_synthesizer,
            enable_comparison=False,
        )

        result = orchestrator.query(
            "Compare news vs SEC for Apple",
            ticker="AAPL",
            force_route="both",
        )

        # Comparison should be None when disabled
        assert result.comparison is None


# =============================================================================
# Memory and Session Management Tests
# =============================================================================


class TestE2EMemoryManagement:
    """Test memory tracking and session management in e2e flow."""

    def test_query_stored_in_memory_with_full_details(
        self, orchestrator_e2e: OrchestrationAgent
    ) -> None:
        """Test that queries are stored in memory with all details."""
        result = orchestrator_e2e.query(
            "What's the outlook for Apple?",
            ticker="AAPL",
            force_route="both",
        )

        # Retrieve from memory
        stored = orchestrator_e2e.memory.get_query_by_id(result.query_id)

        assert stored is not None
        assert stored["user_query"] == "What's the outlook for Apple?"
        assert stored["ticker"] == "AAPL"
        assert stored["route_type"] == "both"
        assert stored["synthesized_response"] == result.response
        assert stored["confidence"] == result.confidence
        assert "agent_results" in stored
        assert len(stored["agent_results"]) == 2

    def test_multi_query_session_tracking(
        self, orchestrator_e2e: OrchestrationAgent
    ) -> None:
        """Test tracking of multiple queries in a session."""
        # Execute multiple queries
        result1 = orchestrator_e2e.query("News for Apple?", ticker="AAPL")
        result2 = orchestrator_e2e.query("SEC filings for Tesla?", ticker="TSLA")
        result3 = orchestrator_e2e.query("Compare MSFT", ticker="MSFT", force_route="both")

        # Verify all queries are tracked
        assert orchestrator_e2e.memory.entry_count == 3

        # Verify session summary
        summary = orchestrator_e2e.get_session_summary()
        assert summary["query_count"] == 3
        assert "AAPL" in summary["tickers_analyzed"]
        assert "TSLA" in summary["tickers_analyzed"]
        assert "MSFT" in summary["tickers_analyzed"]

    def test_filter_queries_by_ticker(
        self, orchestrator_e2e: OrchestrationAgent
    ) -> None:
        """Test filtering queries by ticker symbol."""
        # Execute queries for different tickers
        orchestrator_e2e.query("News for Apple?", ticker="AAPL")
        orchestrator_e2e.query("SEC for Apple?", ticker="AAPL")
        orchestrator_e2e.query("News for Tesla?", ticker="TSLA")

        # Filter by AAPL
        aapl_queries = orchestrator_e2e.get_recent_queries(ticker="AAPL")
        assert len(aapl_queries) == 2
        for q in aapl_queries:
            assert q["ticker"] == "AAPL"

        # Filter by TSLA
        tsla_queries = orchestrator_e2e.get_recent_queries(ticker="TSLA")
        assert len(tsla_queries) == 1
        assert tsla_queries[0]["ticker"] == "TSLA"

    def test_reset_clears_all_session_data(
        self, orchestrator_e2e: OrchestrationAgent
    ) -> None:
        """Test that reset clears all session data."""
        # Execute queries
        orchestrator_e2e.query("News for Apple?", ticker="AAPL")
        orchestrator_e2e.query("SEC for Tesla?", ticker="TSLA")

        assert orchestrator_e2e.memory.entry_count == 2

        # Reset
        orchestrator_e2e.reset()

        # Verify cleared
        assert orchestrator_e2e.memory.entry_count == 0
        summary = orchestrator_e2e.get_session_summary()
        assert summary["query_count"] == 0


# =============================================================================
# Error Handling and Partial Failure Tests
# =============================================================================


class TestE2EErrorHandling:
    """Test error handling and graceful degradation in e2e flow."""

    def test_partial_failure_news_timeout(
        self,
        mock_sec_agent_realistic: MagicMock,
        mock_bedrock_client_synthesizer: MagicMock,
    ) -> None:
        """Test graceful handling when news agent times out."""
        # Create slow news agent
        slow_news_agent = MagicMock()

        def slow_query(_: str) -> str:
            import time
            time.sleep(2)
            return "Response"

        slow_news_agent.query = slow_query
        slow_news_agent.reset.return_value = None

        orchestrator = OrchestrationAgent(
            news_agent=slow_news_agent,
            sec_agent=mock_sec_agent_realistic,
            bedrock_client=mock_bedrock_client_synthesizer,
            timeout_seconds=0.1,  # Very short timeout
            enable_comparison=True,
        )

        result = orchestrator.query(
            "Compare news vs SEC for Apple",
            ticker="AAPL",
            force_route="both",
        )

        # Should still return a result
        assert isinstance(result, OrchestrationResult)
        assert result.response is not None

        # News agent should have timed out
        assert result.news_result is not None
        assert result.news_result.status == AgentStatus.TIMEOUT

        # SEC agent should have succeeded
        assert result.sec_result is not None
        assert result.sec_result.status == AgentStatus.SUCCESS

    def test_partial_failure_sec_exception(
        self,
        mock_news_agent_realistic: MagicMock,
        mock_bedrock_client_synthesizer: MagicMock,
    ) -> None:
        """Test graceful handling when SEC agent throws exception."""
        # Create failing SEC agent
        failing_sec_agent = MagicMock()
        failing_sec_agent.query.side_effect = RuntimeError("SEC API unavailable")
        failing_sec_agent.reset.return_value = None

        orchestrator = OrchestrationAgent(
            news_agent=mock_news_agent_realistic,
            sec_agent=failing_sec_agent,
            bedrock_client=mock_bedrock_client_synthesizer,
            enable_comparison=True,
        )

        result = orchestrator.query(
            "Compare news vs SEC for Apple",
            ticker="AAPL",
            force_route="both",
        )

        # Should still return a result
        assert isinstance(result, OrchestrationResult)
        assert result.response is not None

        # News agent should have succeeded
        assert result.news_result is not None
        assert result.news_result.status == AgentStatus.SUCCESS

        # SEC agent should have errored
        assert result.sec_result is not None
        assert result.sec_result.status == AgentStatus.ERROR
        assert "SEC API unavailable" in (result.sec_result.error_message or "")

    def test_both_agents_fail(
        self,
        mock_bedrock_client_synthesizer: MagicMock,
    ) -> None:
        """Test handling when both agents fail."""
        # Create failing agents
        failing_news_agent = MagicMock()
        failing_news_agent.query.side_effect = RuntimeError("News API down")
        failing_news_agent.reset.return_value = None

        failing_sec_agent = MagicMock()
        failing_sec_agent.query.side_effect = RuntimeError("SEC API down")
        failing_sec_agent.reset.return_value = None

        orchestrator = OrchestrationAgent(
            news_agent=failing_news_agent,
            sec_agent=failing_sec_agent,
            bedrock_client=mock_bedrock_client_synthesizer,
            enable_comparison=True,
        )

        result = orchestrator.query(
            "Compare news vs SEC for Apple",
            ticker="AAPL",
            force_route="both",
        )

        # Should still return a result (synthesizer handles no data gracefully)
        assert isinstance(result, OrchestrationResult)
        assert result.news_result is not None
        assert result.sec_result is not None
        assert result.news_result.status == AgentStatus.ERROR
        assert result.sec_result.status == AgentStatus.ERROR


# =============================================================================
# Async Execution Tests
# =============================================================================


class TestE2EAsyncExecution:
    """Test async execution patterns in e2e flow."""

    def test_async_query_produces_same_result(
        self, orchestrator_e2e: OrchestrationAgent
    ) -> None:
        """Test that async and sync queries produce equivalent results."""
        # Sync query
        sync_result = orchestrator_e2e.query(
            "What's the outlook for Apple?",
            ticker="AAPL",
            force_route="both",
        )

        # Reset to clear state
        orchestrator_e2e.reset()

        # Async query
        async def run_async() -> OrchestrationResult:
            return await orchestrator_e2e.query_async(
                "What's the outlook for Apple?",
                ticker="AAPL",
                force_route="both",
            )

        async_result = asyncio.run(run_async())

        # Results should be structurally equivalent
        assert sync_result.route_type == async_result.route_type
        assert sync_result.agents_used == async_result.agents_used
        assert len(sync_result.response) > 0
        assert len(async_result.response) > 0

    def test_multiple_async_queries_in_session(
        self, orchestrator_e2e: OrchestrationAgent
    ) -> None:
        """Test multiple async queries in a session."""
        async def run_queries() -> list[OrchestrationResult]:
            results = []
            results.append(
                await orchestrator_e2e.query_async("News for AAPL?", ticker="AAPL")
            )
            results.append(
                await orchestrator_e2e.query_async("SEC for TSLA?", ticker="TSLA")
            )
            return results

        results = asyncio.run(run_queries())

        assert len(results) == 2
        assert orchestrator_e2e.memory.entry_count == 2


# =============================================================================
# Compare Method Tests
# =============================================================================


class TestE2ECompareMethod:
    """Test the compare convenience method in e2e flow."""

    def test_compare_forces_both_agents(
        self, orchestrator_e2e: OrchestrationAgent
    ) -> None:
        """Test that compare method forces both agents to run."""
        result = orchestrator_e2e.compare("AAPL")

        assert result.route_type == "both"
        assert "news_agent" in result.agents_used
        assert "sec_agent" in result.agents_used
        assert result.news_result is not None
        assert result.sec_result is not None

    def test_compare_includes_comparison_result(
        self, orchestrator_e2e: OrchestrationAgent
    ) -> None:
        """Test that compare includes comparison analysis."""
        result = orchestrator_e2e.compare("AAPL")

        assert result.comparison is not None
        assert isinstance(result.comparison, ComparisonResult)

    def test_compare_stores_ticker_in_memory(
        self, orchestrator_e2e: OrchestrationAgent
    ) -> None:
        """Test that compare stores the ticker in memory."""
        result = orchestrator_e2e.compare("TSLA")

        stored = orchestrator_e2e.memory.get_query_by_id(result.query_id)
        assert stored is not None
        assert stored["ticker"] == "TSLA"


# =============================================================================
# Ticker Extraction Integration Tests
# =============================================================================


class TestE2ETickerExtraction:
    """Test ticker extraction in the full e2e flow."""

    def test_ticker_extracted_from_company_name(
        self, orchestrator_e2e: OrchestrationAgent
    ) -> None:
        """Test ticker extraction from company name in query."""
        result = orchestrator_e2e.query("What's happening with Apple?")

        stored = orchestrator_e2e.memory.get_query_by_id(result.query_id)
        assert stored is not None
        assert stored["ticker"] == "AAPL"

    def test_ticker_extracted_from_dollar_sign(
        self, orchestrator_e2e: OrchestrationAgent
    ) -> None:
        """Test ticker extraction from $SYMBOL format."""
        result = orchestrator_e2e.query("How is $TSLA performing?")

        stored = orchestrator_e2e.memory.get_query_by_id(result.query_id)
        assert stored is not None
        assert stored["ticker"] == "TSLA"

    def test_explicit_ticker_overrides_extraction(
        self, orchestrator_e2e: OrchestrationAgent
    ) -> None:
        """Test that explicit ticker parameter overrides extraction."""
        result = orchestrator_e2e.query(
            "What's happening with Apple?",  # Would extract AAPL
            ticker="GOOGL",  # But we explicitly say GOOGL
        )

        stored = orchestrator_e2e.memory.get_query_by_id(result.query_id)
        assert stored is not None
        assert stored["ticker"] == "GOOGL"


# =============================================================================
# Full Workflow Integration Tests
# =============================================================================


class TestE2EFullWorkflow:
    """Test complete real-world workflow scenarios."""

    def test_analyst_research_workflow(
        self, orchestrator_e2e: OrchestrationAgent
    ) -> None:
        """Test a typical analyst research workflow."""
        # Step 1: Check recent news
        news_result = orchestrator_e2e.query(
            "What's the recent news sentiment for Apple?",
            ticker="AAPL",
        )
        assert news_result.route_type == "news_only"

        # Step 2: Check SEC filings
        sec_result = orchestrator_e2e.query(
            "What are the key risk factors in Apple's 10-K?",
            ticker="AAPL",
        )
        assert sec_result.route_type == "sec_only"

        # Step 3: Run comparison
        compare_result = orchestrator_e2e.compare("AAPL")
        assert compare_result.route_type == "both"
        assert compare_result.comparison is not None

        # Verify session tracked all queries
        assert orchestrator_e2e.memory.entry_count == 3

        # All queries should be for AAPL
        aapl_queries = orchestrator_e2e.get_recent_queries(ticker="AAPL")
        assert len(aapl_queries) == 3

    def test_multi_company_comparison_workflow(
        self, orchestrator_e2e: OrchestrationAgent
    ) -> None:
        """Test comparing multiple companies in a session."""
        # Analyze multiple companies
        tickers = ["AAPL", "TSLA", "MSFT"]
        results = []

        for ticker in tickers:
            result = orchestrator_e2e.compare(ticker)
            results.append(result)

        # Verify all comparisons completed
        assert len(results) == 3
        for result in results:
            assert result.route_type == "both"
            assert result.comparison is not None

        # Verify session summary
        summary = orchestrator_e2e.get_session_summary()
        assert summary["query_count"] == 3
        for ticker in tickers:
            assert ticker in summary["tickers_analyzed"]

    def test_error_recovery_workflow(
        self,
        mock_news_agent_realistic: MagicMock,
        mock_bedrock_client_synthesizer: MagicMock,
    ) -> None:
        """Test recovery from errors in a workflow."""
        # Create SEC agent that fails initially then works
        sec_agent = MagicMock()
        sec_agent.reset.return_value = None
        call_count = 0

        def conditional_response(query: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Temporary failure")
            return "SEC response: Strong financials"

        sec_agent.query = conditional_response

        orchestrator = OrchestrationAgent(
            news_agent=mock_news_agent_realistic,
            sec_agent=sec_agent,
            bedrock_client=mock_bedrock_client_synthesizer,
            enable_comparison=True,
        )

        # First query - SEC fails
        result1 = orchestrator.query("Compare Apple", ticker="AAPL", force_route="both")
        assert result1.sec_result is not None
        assert result1.sec_result.status == AgentStatus.ERROR

        # Second query - SEC works
        result2 = orchestrator.query("Compare Apple again", ticker="AAPL", force_route="both")
        assert result2.sec_result is not None
        assert result2.sec_result.status == AgentStatus.SUCCESS

        # Both queries tracked
        assert orchestrator.memory.entry_count == 2
