"""Tests for the orchestrator query classifier."""

from __future__ import annotations

import pytest

from src.orchestrator.execution.result import QueryClassification, RouteType
from src.orchestrator.routing.classifier import QueryClassifier


class TestQueryClassifier:
    """Tests for QueryClassifier."""

    @pytest.fixture
    def classifier(self) -> QueryClassifier:
        """Create a classifier instance for testing."""
        return QueryClassifier()

    # -------------------------------------------------------------------------
    # Comparison pattern tests (highest priority)
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "query",
        [
            "Compare Apple's news sentiment with SEC filings",
            "What's the comparison between news and filings?",
            "News vs SEC for Tesla",
            "Are there any discrepancies in the data?",
            "Show me the differences between sources",
            "How does news contrast with the 10-K?",
            "Do the sources align on Apple's outlook?",
            "Is the sentiment consistent with filings?",
            "Check if news and SEC filings match",
            "Both sources for AAPL",
        ],
    )
    def test_comparison_patterns_route_to_both(
        self, classifier: QueryClassifier, query: str
    ) -> None:
        """Comparison keywords should route to both agents."""
        result = classifier.classify(query)

        assert result.route_type == RouteType.BOTH
        assert result.confidence == 1.0
        assert result.needs_both is True
        assert result.needs_news_agent is True
        assert result.needs_sec_agent is True
        assert len(result.matched_patterns) > 0
        assert result.reasoning is not None
        assert "comparison" in result.reasoning.lower()

    # -------------------------------------------------------------------------
    # SEC-only pattern tests
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "query",
        [
            "What does the 10-K say about risks?",
            "Show me the 10Q filing",
            "What's in the 8-K?",
            "What are the risk factors for Apple?",
            "Get the SEC filing for Tesla",
            "Show me the financial statements",
            "What are the earnings?",
            "Revenue trends from filings",
            "Balance sheet analysis",
            "Cash flow statement",
            "What's in the MD&A section?",
            "Management discussion",
            "Forward-looking statements",
            "Any regulatory concerns?",
        ],
    )
    def test_sec_patterns_route_to_sec_only(
        self, classifier: QueryClassifier, query: str
    ) -> None:
        """SEC keywords should route to SEC agent only."""
        result = classifier.classify(query)

        assert result.route_type == RouteType.SEC_ONLY
        assert result.confidence >= 0.6
        assert result.needs_sec_agent is True
        assert result.needs_news_agent is False
        assert result.needs_both is False
        assert len(result.matched_patterns) > 0
        assert result.reasoning is not None
        assert "sec" in result.reasoning.lower()

    # -------------------------------------------------------------------------
    # News-only pattern tests
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "query",
        [
            "What's the latest news on Apple?",
            "What's the sentiment around Tesla?",
            "Show me recent headlines for AAPL",
            "Any breaking news about Microsoft?",
            "What's the media saying about the company?",
            "Recent reports on the stock",
            "What are analysts saying?",
            "Market reaction to the announcement",
            "Stock price movement today",
            "Trading activity for NVDA",
            "What's the latest press release?",
        ],
    )
    def test_news_patterns_route_to_news_only(
        self, classifier: QueryClassifier, query: str
    ) -> None:
        """News keywords should route to news agent only."""
        result = classifier.classify(query)

        assert result.route_type == RouteType.NEWS_ONLY
        assert result.confidence >= 0.6
        assert result.needs_news_agent is True
        assert result.needs_sec_agent is False
        assert result.needs_both is False
        assert len(result.matched_patterns) > 0
        assert result.reasoning is not None
        assert "news" in result.reasoning.lower()

    # -------------------------------------------------------------------------
    # Mixed patterns (both SEC and news keywords)
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "query",
        [
            "What's the news about the 10-K filing?",
            "Latest headlines on the earnings report",
            "Sentiment on the quarterly report",
            "News about Apple's financial statements",
            "Recent analyst reports on the SEC filing",
        ],
    )
    def test_mixed_patterns_route_to_both(
        self, classifier: QueryClassifier, query: str
    ) -> None:
        """Queries with both SEC and news keywords should route to both agents."""
        result = classifier.classify(query)

        assert result.route_type == RouteType.BOTH
        assert result.confidence == 0.9
        assert result.needs_both is True
        assert len(result.matched_patterns) >= 2
        assert result.reasoning is not None
        assert "both" in result.reasoning.lower()

    # -------------------------------------------------------------------------
    # Ambiguous/default cases
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "query",
        [
            "What's happening with Apple?",
            "Tell me about Tesla",
            "AAPL outlook",
            "Is Microsoft a good investment?",
            "What should I know about NVDA?",
        ],
    )
    def test_ambiguous_queries_default_to_both(
        self, classifier: QueryClassifier, query: str
    ) -> None:
        """Ambiguous queries with no clear patterns should default to both agents."""
        result = classifier.classify(query)

        assert result.route_type == RouteType.BOTH
        assert result.confidence == 0.5
        assert result.needs_both is True
        assert len(result.matched_patterns) == 0
        assert result.reasoning is not None
        assert "default" in result.reasoning.lower()

    # -------------------------------------------------------------------------
    # Case insensitivity tests
    # -------------------------------------------------------------------------

    def test_case_insensitive_sec_patterns(self, classifier: QueryClassifier) -> None:
        """SEC patterns should match regardless of case."""
        queries = ["10-K FILING", "10-k filing", "10-K Filing"]
        for query in queries:
            result = classifier.classify(query)
            assert result.route_type == RouteType.SEC_ONLY

    def test_case_insensitive_news_patterns(self, classifier: QueryClassifier) -> None:
        """News patterns should match regardless of case."""
        queries = ["LATEST NEWS", "latest news", "Latest News"]
        for query in queries:
            result = classifier.classify(query)
            assert result.route_type == RouteType.NEWS_ONLY

    def test_case_insensitive_comparison_patterns(
        self, classifier: QueryClassifier
    ) -> None:
        """Comparison patterns should match regardless of case."""
        queries = ["COMPARE sources", "compare SOURCES", "Compare Sources"]
        for query in queries:
            result = classifier.classify(query)
            assert result.route_type == RouteType.BOTH
            assert result.confidence == 1.0

    # -------------------------------------------------------------------------
    # Confidence scoring tests
    # -------------------------------------------------------------------------

    def test_confidence_increases_with_more_matches(
        self, classifier: QueryClassifier
    ) -> None:
        """Confidence should increase when more patterns match."""
        # Single match
        result_single = classifier.classify("Show me the 10-K")
        # Multiple matches
        result_multiple = classifier.classify(
            "Show me the 10-K with risk factors from the annual report"
        )

        assert result_multiple.confidence >= result_single.confidence
        assert len(result_multiple.matched_patterns) > len(result_single.matched_patterns)

    def test_confidence_capped_at_one(self, classifier: QueryClassifier) -> None:
        """Confidence should never exceed 1.0."""
        # Query with many SEC keywords
        query = (
            "10-K filing with risk factors, earnings, revenue, "
            "balance sheet, cash flow, MD&A, and regulatory concerns"
        )
        result = classifier.classify(query)

        assert result.confidence <= 1.0

    # -------------------------------------------------------------------------
    # QueryClassification property tests
    # -------------------------------------------------------------------------

    def test_needs_news_agent_property(self, classifier: QueryClassifier) -> None:
        """needs_news_agent should be True for NEWS_ONLY and BOTH."""
        news_result = classifier.classify("latest news")
        both_result = classifier.classify("compare sources")
        sec_result = classifier.classify("10-K filing")

        assert news_result.needs_news_agent is True
        assert both_result.needs_news_agent is True
        assert sec_result.needs_news_agent is False

    def test_needs_sec_agent_property(self, classifier: QueryClassifier) -> None:
        """needs_sec_agent should be True for SEC_ONLY and BOTH."""
        sec_result = classifier.classify("10-K filing")
        both_result = classifier.classify("compare sources")
        news_result = classifier.classify("latest news")

        assert sec_result.needs_sec_agent is True
        assert both_result.needs_sec_agent is True
        assert news_result.needs_sec_agent is False

    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------

    def test_empty_query(self, classifier: QueryClassifier) -> None:
        """Empty query should default to both agents."""
        result = classifier.classify("")

        assert result.route_type == RouteType.BOTH
        assert result.confidence == 0.5

    def test_whitespace_only_query(self, classifier: QueryClassifier) -> None:
        """Whitespace-only query should default to both agents."""
        result = classifier.classify("   \t\n  ")

        assert result.route_type == RouteType.BOTH
        assert result.confidence == 0.5

    def test_special_characters_query(self, classifier: QueryClassifier) -> None:
        """Query with special characters should be handled gracefully."""
        result = classifier.classify("!@#$%^&*()")

        assert result.route_type == RouteType.BOTH
        assert result.confidence == 0.5

    def test_comparison_priority_over_other_patterns(
        self, classifier: QueryClassifier
    ) -> None:
        """Comparison patterns should take priority even when other patterns exist."""
        # Query has SEC keyword (10-K) but also comparison keyword (compare)
        result = classifier.classify("Compare the 10-K with the news")

        # Should match comparison first, not SEC
        assert result.route_type == RouteType.BOTH
        assert result.confidence == 1.0
        assert result.reasoning is not None
        assert "comparison" in result.reasoning.lower()

    # -------------------------------------------------------------------------
    # Pattern matching tests
    # -------------------------------------------------------------------------

    def test_matched_patterns_are_returned(self, classifier: QueryClassifier) -> None:
        """Matched patterns should be included in the result."""
        result = classifier.classify("What are the risk factors in the 10-K?")

        assert len(result.matched_patterns) > 0
        # Should have matched at least "10-K" and "risk factors"
        pattern_str = " ".join(result.matched_patterns)
        assert "10-?[KQ]" in pattern_str or "risk" in pattern_str.lower()

    def test_word_boundary_matching(self, classifier: QueryClassifier) -> None:
        """Patterns should use word boundaries to avoid false matches."""
        # "news" should not match "renewsing" (not a real word, but tests boundary)
        result = classifier.classify("The company is renewing contracts")

        # "renewing" should not trigger news pattern
        assert result.route_type == RouteType.BOTH
        assert result.confidence == 0.5

    def test_pattern_with_optional_suffix(self, classifier: QueryClassifier) -> None:
        """Patterns with optional suffixes should match variations."""
        # "filing" and "filings"
        result1 = classifier.classify("Show me the filing")
        result2 = classifier.classify("Show me the filings")

        assert result1.route_type == RouteType.SEC_ONLY
        assert result2.route_type == RouteType.SEC_ONLY
