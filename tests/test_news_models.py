"""Tests for news_agent data models."""

from datetime import date, datetime, timezone

import pytest

from news_agent.models import (
    DailySentiment,
    NewsArticle,
    NewsSentimentResult,
    SentimentAnalysis,
)


class TestNewsArticle:
    """Tests for NewsArticle model."""

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization round-trip."""
        now = datetime.now(timezone.utc)
        article = NewsArticle(
            article_id="abc123",
            title="Test Article",
            description="Test description",
            content="Full content here",
            source_name="Reuters",
            source_domain="reuters.com",
            author="John Doe",
            url="https://reuters.com/article",
            published_at=now,
            fetched_at=now,
            ticker="AAPL",
            company_name="Apple Inc",
        )

        data = article.to_dict()
        restored = NewsArticle.from_dict(data)

        assert restored.article_id == article.article_id
        assert restored.title == article.title
        assert restored.source_name == article.source_name
        assert restored.ticker == article.ticker

    def test_from_dict_with_none_author(self) -> None:
        """Test deserialization with None author."""
        data = {
            "article_id": "test",
            "title": "Title",
            "description": "Desc",
            "content": "Content",
            "source_name": "Test",
            "source_domain": "test.com",
            "author": None,
            "url": "https://test.com",
            "published_at": "2025-01-01T00:00:00+00:00",
            "fetched_at": "2025-01-01T00:00:00+00:00",
            "ticker": "TEST",
            "company_name": "Test Corp",
        }

        article = NewsArticle.from_dict(data)
        assert article.author is None


class TestSentimentAnalysis:
    """Tests for SentimentAnalysis model."""

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization round-trip."""
        now = datetime.now(timezone.utc)
        analysis = SentimentAnalysis(
            article_id="abc123",
            sentiment="positive",
            confidence=0.85,
            magnitude=0.7,
            key_claims=["Claim 1", "Claim 2"],
            topics=["earnings", "product"],
            is_material=True,
            materiality_reason="Strong earnings beat",
            analysis_timestamp=now,
        )

        data = analysis.to_dict()
        restored = SentimentAnalysis.from_dict(data)

        assert restored.sentiment == "positive"
        assert restored.confidence == 0.85
        assert len(restored.key_claims) == 2
        assert restored.is_material is True


class TestDailySentiment:
    """Tests for DailySentiment model."""

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization round-trip."""
        daily = DailySentiment(
            ticker="AAPL",
            date=date(2025, 1, 15),
            article_count=10,
            positive_count=5,
            negative_count=3,
            neutral_count=2,
            avg_sentiment_score=0.25,
            material_articles=4,
            top_topics=["earnings", "product"],
        )

        data = daily.to_dict()
        restored = DailySentiment.from_dict(data)

        assert restored.ticker == "AAPL"
        assert restored.date == date(2025, 1, 15)
        assert restored.article_count == 10
        assert restored.avg_sentiment_score == 0.25


class TestNewsSentimentResult:
    """Tests for NewsSentimentResult model."""

    def test_to_dict(self) -> None:
        """Test serialization."""
        result = NewsSentimentResult(
            ticker="AAPL",
            company_name="Apple Inc",
            time_period="2025-01-01 to 2025-01-31",
            overall_sentiment="positive",
            sentiment_score=0.45,
            confidence=0.8,
            article_count=50,
            material_article_count=15,
            trend_direction="improving",
            trend_magnitude=0.2,
            key_claims=[{"claim": "Revenue up 10%", "source": "Reuters"}],
            top_topics=["earnings"],
            narrative_summary="Test summary",
            material_events=["Earnings beat"],
            sources=[{"title": "Article", "url": "https://test.com"}],
        )

        data = result.to_dict()

        assert data["ticker"] == "AAPL"
        assert data["overall_sentiment"] == "positive"
        assert data["sentiment_score"] == 0.45
        assert len(data["key_claims"]) == 1

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        result = NewsSentimentResult(
            ticker="AAPL",
            company_name="Apple Inc",
            time_period="2025-01-01 to 2025-01-31",
            overall_sentiment="neutral",
            sentiment_score=0.0,
            confidence=0.5,
            article_count=5,
            material_article_count=0,
            trend_direction="stable",
            trend_magnitude=0.0,
        )

        assert result.key_claims == []
        assert result.top_topics == []
        assert result.narrative_summary == ""
        assert result.material_events == []
        assert result.sources == []
        assert result.status == "ok"
        assert result.status_message == ""
