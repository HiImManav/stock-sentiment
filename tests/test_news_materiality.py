"""Tests for materiality filter."""

from datetime import datetime, timezone

import pytest

from news_agent.analysis.materiality import (
    TIER_1_SOURCES,
    TIER_2_SOURCES,
    MaterialityFilter,
)
from news_agent.models import NewsArticle


@pytest.fixture
def materiality_filter() -> MaterialityFilter:
    """Create a materiality filter instance."""
    return MaterialityFilter()


@pytest.fixture
def sample_article() -> NewsArticle:
    """Create a sample article for testing."""
    return NewsArticle(
        article_id="test123",
        title="Apple Reports Strong Earnings",
        description="Apple Inc reported better than expected quarterly results",
        content="Apple Inc reported earnings per share of $2.50, beating estimates.",
        source_name="Reuters",
        source_domain="reuters.com",
        author="Test Author",
        url="https://reuters.com/article",
        published_at=datetime.now(timezone.utc),
        fetched_at=datetime.now(timezone.utc),
        ticker="AAPL",
        company_name="Apple Inc",
    )


class TestSourceTiers:
    """Tests for source tier classification."""

    def test_tier_1_sources(self, materiality_filter: MaterialityFilter) -> None:
        """Test Tier 1 source detection."""
        assert materiality_filter.get_source_tier("reuters.com") == 1
        assert materiality_filter.get_source_tier("bloomberg.com") == 1
        assert materiality_filter.get_source_tier("wsj.com") == 1
        assert materiality_filter.get_source_tier("ft.com") == 1
        assert materiality_filter.get_source_tier("cnbc.com") == 1

    def test_tier_2_sources(self, materiality_filter: MaterialityFilter) -> None:
        """Test Tier 2 source detection."""
        assert materiality_filter.get_source_tier("marketwatch.com") == 2
        assert materiality_filter.get_source_tier("seekingalpha.com") == 2
        assert materiality_filter.get_source_tier("fool.com") == 2

    def test_tier_3_sources(self, materiality_filter: MaterialityFilter) -> None:
        """Test Tier 3 (unknown) source detection."""
        assert materiality_filter.get_source_tier("random-blog.com") == 3
        assert materiality_filter.get_source_tier("unknown-site.net") == 3

    def test_source_weight(self, materiality_filter: MaterialityFilter) -> None:
        """Test source weight calculation."""
        assert materiality_filter.get_source_weight("reuters.com") == 1.0
        assert materiality_filter.get_source_weight("marketwatch.com") == 0.7
        assert materiality_filter.get_source_weight("random-blog.com") == 0.3

    def test_www_prefix_handling(self, materiality_filter: MaterialityFilter) -> None:
        """Test that www. prefix is handled correctly."""
        assert materiality_filter.get_source_tier("www.reuters.com") == 1


class TestKeywordDetection:
    """Tests for material keyword detection."""

    def test_earnings_keywords(self, materiality_filter: MaterialityFilter) -> None:
        """Test earnings keyword detection."""
        text = "Company reports record quarterly earnings and raises guidance"
        keywords = materiality_filter.detect_keywords(text)
        assert keywords["earnings"] is True

    def test_legal_keywords(self, materiality_filter: MaterialityFilter) -> None:
        """Test legal keyword detection."""
        text = "Company faces SEC investigation over accounting practices"
        keywords = materiality_filter.detect_keywords(text)
        assert keywords["legal"] is True

    def test_leadership_keywords(self, materiality_filter: MaterialityFilter) -> None:
        """Test leadership keyword detection."""
        text = "CEO announces resignation effective immediately"
        keywords = materiality_filter.detect_keywords(text)
        assert keywords["leadership"] is True

    def test_m_and_a_keywords(self, materiality_filter: MaterialityFilter) -> None:
        """Test M&A keyword detection."""
        text = "Company announces acquisition of competitor for $1 billion"
        keywords = materiality_filter.detect_keywords(text)
        assert keywords["m_and_a"] is True

    def test_no_keywords(self, materiality_filter: MaterialityFilter) -> None:
        """Test when no material keywords are found."""
        text = "The weather was nice today"
        keywords = materiality_filter.detect_keywords(text)
        assert all(not v for v in keywords.values())

    def test_get_detected_topics(self, materiality_filter: MaterialityFilter) -> None:
        """Test topic detection returns list."""
        text = "CEO announces layoffs and restructuring due to earnings miss"
        topics = materiality_filter.get_detected_topics(text)
        assert "leadership" in topics
        assert "operations" in topics
        assert "earnings" in topics


class TestLayerFiltering:
    """Tests for layer-based filtering."""

    def test_passes_layer_1_tier_1(
        self, materiality_filter: MaterialityFilter, sample_article: NewsArticle
    ) -> None:
        """Test Layer 1 passes for Tier 1 source."""
        passes, reason = materiality_filter.passes_layer_1(sample_article)
        assert passes is True
        assert "tier 1" in reason.lower()

    def test_passes_layer_1_tier_3(
        self, materiality_filter: MaterialityFilter, sample_article: NewsArticle
    ) -> None:
        """Test Layer 1 fails for Tier 3 source."""
        sample_article.source_domain = "random-blog.com"
        passes, reason = materiality_filter.passes_layer_1(sample_article)
        assert passes is False
        assert "low-tier" in reason.lower()

    def test_passes_layer_2_with_keywords(
        self, materiality_filter: MaterialityFilter, sample_article: NewsArticle
    ) -> None:
        """Test Layer 2 passes when keywords found."""
        passes, reason = materiality_filter.passes_layer_2(sample_article)
        assert passes is True
        assert "earnings" in reason.lower()

    def test_passes_layer_2_without_keywords(
        self, materiality_filter: MaterialityFilter, sample_article: NewsArticle
    ) -> None:
        """Test Layer 2 fails when no keywords found."""
        sample_article.title = "Nice day for a walk"
        sample_article.description = "The weather is great"
        sample_article.content = "Nothing happening"
        passes, reason = materiality_filter.passes_layer_2(sample_article)
        assert passes is False


class TestClickbaitDetection:
    """Tests for clickbait detection."""

    def test_clickbait_patterns(self, materiality_filter: MaterialityFilter) -> None:
        """Test clickbait headline detection."""
        assert materiality_filter.is_clickbait("You Won't Believe What This CEO Did")
        assert materiality_filter.is_clickbait("10 Reasons Why This Stock Will Explode")
        assert materiality_filter.is_clickbait("SHOCKING: Company Makes Big Move")

    def test_normal_headlines(self, materiality_filter: MaterialityFilter) -> None:
        """Test normal headline detection."""
        assert not materiality_filter.is_clickbait("Apple Reports Q4 Earnings")
        assert not materiality_filter.is_clickbait("Tesla CEO Announces New Factory")
        assert not materiality_filter.is_clickbait("Microsoft Acquires Gaming Company")


class TestArticleFiltering:
    """Tests for full article filtering."""

    def test_filter_articles(
        self, materiality_filter: MaterialityFilter
    ) -> None:
        """Test filtering a list of articles."""
        now = datetime.now(timezone.utc)

        articles = [
            NewsArticle(
                article_id="1",
                title="Apple earnings beat",
                description="Strong results",
                content="Revenue up",
                source_name="Reuters",
                source_domain="reuters.com",
                author=None,
                url="https://reuters.com/1",
                published_at=now,
                fetched_at=now,
                ticker="AAPL",
                company_name="Apple Inc",
            ),
            NewsArticle(
                article_id="2",
                title="Random article",
                description="Not important",
                content="Just stuff",
                source_name="Blog",
                source_domain="random-blog.com",
                author=None,
                url="https://blog.com/2",
                published_at=now,
                fetched_at=now,
                ticker="AAPL",
                company_name="Apple Inc",
            ),
        ]

        # Filter with min_tier=2 (exclude tier 3)
        results = materiality_filter.filter_articles(articles, min_tier=2)

        # Should only include the Reuters article
        assert len(results) == 1
        assert results[0][0].article_id == "1"
        assert results[0][1] == 1.0  # Full weight for tier 1
