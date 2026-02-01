"""Tests for trend analysis."""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from news_agent.analysis.trends import TrendAnalyzer
from news_agent.models import DailySentiment, SentimentAnalysis


@pytest.fixture
def mock_cache() -> MagicMock:
    """Create a mock cache."""
    mock = MagicMock()
    mock.get_trend_data.return_value = []
    mock.save_trend_data.return_value = None
    return mock


@pytest.fixture
def trend_analyzer(mock_cache: MagicMock) -> TrendAnalyzer:
    """Create a trend analyzer with mock cache."""
    return TrendAnalyzer(cache=mock_cache)


class TestDailySentimentCalculation:
    """Tests for daily sentiment calculation."""

    def test_calculate_daily_sentiment_empty(
        self, trend_analyzer: TrendAnalyzer
    ) -> None:
        """Test calculation with no analyses."""
        result = trend_analyzer.calculate_daily_sentiment("AAPL", [])

        assert result.ticker == "AAPL"
        assert result.article_count == 0
        assert result.avg_sentiment_score == 0.0

    def test_calculate_daily_sentiment_positive(
        self, trend_analyzer: TrendAnalyzer
    ) -> None:
        """Test calculation with positive analyses."""
        now = datetime.now(timezone.utc)
        analyses = [
            SentimentAnalysis(
                article_id="1",
                sentiment="positive",
                confidence=0.9,
                magnitude=0.8,
                key_claims=[],
                topics=["earnings"],
                is_material=True,
                materiality_reason="",
                analysis_timestamp=now,
            ),
            SentimentAnalysis(
                article_id="2",
                sentiment="positive",
                confidence=0.8,
                magnitude=0.7,
                key_claims=[],
                topics=["earnings", "product"],
                is_material=True,
                materiality_reason="",
                analysis_timestamp=now,
            ),
        ]

        result = trend_analyzer.calculate_daily_sentiment("AAPL", analyses)

        assert result.article_count == 2
        assert result.positive_count == 2
        assert result.negative_count == 0
        assert result.avg_sentiment_score > 0
        assert result.material_articles == 2
        assert "earnings" in result.top_topics

    def test_calculate_daily_sentiment_mixed(
        self, trend_analyzer: TrendAnalyzer
    ) -> None:
        """Test calculation with mixed analyses."""
        now = datetime.now(timezone.utc)
        analyses = [
            SentimentAnalysis(
                article_id="1",
                sentiment="positive",
                confidence=0.9,
                magnitude=0.8,
                key_claims=[],
                topics=[],
                is_material=True,
                materiality_reason="",
                analysis_timestamp=now,
            ),
            SentimentAnalysis(
                article_id="2",
                sentiment="negative",
                confidence=0.8,
                magnitude=0.7,
                key_claims=[],
                topics=[],
                is_material=False,
                materiality_reason="",
                analysis_timestamp=now,
            ),
            SentimentAnalysis(
                article_id="3",
                sentiment="neutral",
                confidence=0.7,
                magnitude=0.5,
                key_claims=[],
                topics=[],
                is_material=False,
                materiality_reason="",
                analysis_timestamp=now,
            ),
        ]

        result = trend_analyzer.calculate_daily_sentiment("AAPL", analyses)

        assert result.article_count == 3
        assert result.positive_count == 1
        assert result.negative_count == 1
        assert result.neutral_count == 1
        assert result.material_articles == 1


class TestTrendCalculation:
    """Tests for trend calculation."""

    def test_calculate_trend_empty(self, trend_analyzer: TrendAnalyzer) -> None:
        """Test trend calculation with no data."""
        result = trend_analyzer.calculate_trend([])

        assert result["trend_direction"] == "stable"
        assert result["trend_magnitude"] == 0.0

    def test_calculate_trend_improving(self, trend_analyzer: TrendAnalyzer) -> None:
        """Test trend calculation with improving sentiment."""
        today = datetime.now(timezone.utc).date()
        daily_scores = [
            DailySentiment(
                ticker="AAPL",
                date=today - timedelta(days=10),
                article_count=5,
                positive_count=1,
                negative_count=3,
                neutral_count=1,
                avg_sentiment_score=-0.3,
                material_articles=2,
                top_topics=[],
            ),
            DailySentiment(
                ticker="AAPL",
                date=today - timedelta(days=5),
                article_count=5,
                positive_count=2,
                negative_count=2,
                neutral_count=1,
                avg_sentiment_score=0.0,
                material_articles=2,
                top_topics=[],
            ),
            DailySentiment(
                ticker="AAPL",
                date=today,
                article_count=5,
                positive_count=4,
                negative_count=0,
                neutral_count=1,
                avg_sentiment_score=0.5,
                material_articles=3,
                top_topics=[],
            ),
        ]

        result = trend_analyzer.calculate_trend(daily_scores)

        assert result["trend_direction"] == "improving"
        assert result["trend_magnitude"] > 0
        assert result["current_score"] > result["score_period_ago"]

    def test_calculate_trend_worsening(self, trend_analyzer: TrendAnalyzer) -> None:
        """Test trend calculation with worsening sentiment."""
        today = datetime.now(timezone.utc).date()
        daily_scores = [
            DailySentiment(
                ticker="AAPL",
                date=today - timedelta(days=10),
                article_count=5,
                positive_count=4,
                negative_count=0,
                neutral_count=1,
                avg_sentiment_score=0.5,
                material_articles=3,
                top_topics=[],
            ),
            DailySentiment(
                ticker="AAPL",
                date=today,
                article_count=5,
                positive_count=1,
                negative_count=3,
                neutral_count=1,
                avg_sentiment_score=-0.3,
                material_articles=2,
                top_topics=[],
            ),
        ]

        result = trend_analyzer.calculate_trend(daily_scores)

        assert result["trend_direction"] == "worsening"
        assert result["current_score"] < result["score_period_ago"]

    def test_calculate_trend_stable(self, trend_analyzer: TrendAnalyzer) -> None:
        """Test trend calculation with stable sentiment."""
        today = datetime.now(timezone.utc).date()
        daily_scores = [
            DailySentiment(
                ticker="AAPL",
                date=today - timedelta(days=10),
                article_count=5,
                positive_count=2,
                negative_count=2,
                neutral_count=1,
                avg_sentiment_score=0.05,
                material_articles=2,
                top_topics=[],
            ),
            DailySentiment(
                ticker="AAPL",
                date=today,
                article_count=5,
                positive_count=2,
                negative_count=2,
                neutral_count=1,
                avg_sentiment_score=0.0,
                material_articles=2,
                top_topics=[],
            ),
        ]

        result = trend_analyzer.calculate_trend(daily_scores)

        assert result["trend_direction"] == "stable"
        assert result["trend_magnitude"] < 0.1

    def test_calculate_trend_inflection_points(
        self, trend_analyzer: TrendAnalyzer
    ) -> None:
        """Test that inflection points are detected."""
        today = datetime.now(timezone.utc).date()
        daily_scores = [
            DailySentiment(
                ticker="AAPL",
                date=today - timedelta(days=2),
                article_count=5,
                positive_count=3,
                negative_count=1,
                neutral_count=1,
                avg_sentiment_score=0.3,
                material_articles=2,
                top_topics=[],
            ),
            DailySentiment(
                ticker="AAPL",
                date=today - timedelta(days=1),
                article_count=5,
                positive_count=0,
                negative_count=4,
                neutral_count=1,
                avg_sentiment_score=-0.5,  # Big drop
                material_articles=3,
                top_topics=["legal"],
            ),
            DailySentiment(
                ticker="AAPL",
                date=today,
                article_count=5,
                positive_count=0,
                negative_count=3,
                neutral_count=2,
                avg_sentiment_score=-0.4,
                material_articles=2,
                top_topics=[],
            ),
        ]

        result = trend_analyzer.calculate_trend(daily_scores)

        # Should detect the significant drop as an inflection point
        assert len(result["inflection_points"]) > 0


class TestGroupAnalysesByDate:
    """Tests for grouping analyses by date."""

    def test_group_analyses_by_date(self, trend_analyzer: TrendAnalyzer) -> None:
        """Test grouping analyses by publication date."""
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)

        analyses = [
            SentimentAnalysis(
                article_id="1",
                sentiment="positive",
                confidence=0.9,
                magnitude=0.8,
                key_claims=[],
                topics=[],
                is_material=True,
                materiality_reason="",
                analysis_timestamp=datetime.now(timezone.utc),
            ),
            SentimentAnalysis(
                article_id="2",
                sentiment="negative",
                confidence=0.8,
                magnitude=0.7,
                key_claims=[],
                topics=[],
                is_material=True,
                materiality_reason="",
                analysis_timestamp=datetime.now(timezone.utc),
            ),
            SentimentAnalysis(
                article_id="3",
                sentiment="neutral",
                confidence=0.7,
                magnitude=0.5,
                key_claims=[],
                topics=[],
                is_material=False,
                materiality_reason="",
                analysis_timestamp=datetime.now(timezone.utc),
            ),
        ]

        articles_by_id = {
            "1": today,
            "2": today,
            "3": yesterday,
        }

        result = trend_analyzer.group_analyses_by_date(analyses, articles_by_id)

        assert len(result[today]) == 2
        assert len(result[yesterday]) == 1
