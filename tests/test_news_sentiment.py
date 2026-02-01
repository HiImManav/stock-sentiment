"""Tests for sentiment analysis."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from news_agent.analysis.sentiment import SentimentAnalyzer
from news_agent.models import NewsArticle, SentimentAnalysis


@pytest.fixture
def mock_bedrock_client() -> MagicMock:
    """Create a mock Bedrock client."""
    mock = MagicMock()
    return mock


@pytest.fixture
def sample_article() -> NewsArticle:
    """Create a sample article for testing."""
    return NewsArticle(
        article_id="test123",
        title="Apple Reports Strong Earnings",
        description="Apple Inc reported better than expected quarterly results",
        content="Apple Inc reported earnings per share of $2.50, beating estimates by 10%.",
        source_name="Reuters",
        source_domain="reuters.com",
        author="Test Author",
        url="https://reuters.com/article",
        published_at=datetime.now(timezone.utc),
        fetched_at=datetime.now(timezone.utc),
        ticker="AAPL",
        company_name="Apple Inc",
    )


class TestSentimentAnalyzer:
    """Tests for SentimentAnalyzer."""

    def test_parse_json_response_clean(self) -> None:
        """Test parsing clean JSON response."""
        analyzer = SentimentAnalyzer(bedrock_client=MagicMock())

        response = '{"sentiment": "positive", "confidence": 0.9}'
        result = analyzer._parse_json_response(response)

        assert result["sentiment"] == "positive"
        assert result["confidence"] == 0.9

    def test_parse_json_response_with_code_blocks(self) -> None:
        """Test parsing JSON wrapped in markdown code blocks."""
        analyzer = SentimentAnalyzer(bedrock_client=MagicMock())

        response = """```json
{"sentiment": "negative", "confidence": 0.8}
```"""
        result = analyzer._parse_json_response(response)

        assert result["sentiment"] == "negative"

    def test_parse_json_response_invalid(self) -> None:
        """Test parsing invalid JSON returns empty dict."""
        analyzer = SentimentAnalyzer(bedrock_client=MagicMock())

        response = "This is not JSON at all"
        result = analyzer._parse_json_response(response)

        assert result == {}

    def test_analyze_article(
        self, mock_bedrock_client: MagicMock, sample_article: NewsArticle
    ) -> None:
        """Test single article analysis."""
        # Mock the Bedrock response
        mock_bedrock_client.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": """{
                            "sentiment": "positive",
                            "confidence": 0.9,
                            "magnitude": 0.8,
                            "is_material": true,
                            "materiality_reason": "Strong earnings beat",
                            "key_claims": ["EPS beat by 10%"],
                            "topics": ["earnings"]
                        }"""
                        }
                    ]
                }
            }
        }

        analyzer = SentimentAnalyzer(bedrock_client=mock_bedrock_client)
        result = analyzer.analyze_article(sample_article)

        assert isinstance(result, SentimentAnalysis)
        assert result.sentiment == "positive"
        assert result.confidence == 0.9
        assert result.is_material is True
        assert "earnings" in result.topics

    def test_analyze_article_invalid_sentiment(
        self, mock_bedrock_client: MagicMock, sample_article: NewsArticle
    ) -> None:
        """Test that invalid sentiment defaults to neutral."""
        mock_bedrock_client.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": '{"sentiment": "invalid_value", "confidence": 0.5}'
                        }
                    ]
                }
            }
        }

        analyzer = SentimentAnalyzer(bedrock_client=mock_bedrock_client)
        result = analyzer.analyze_article(sample_article)

        assert result.sentiment == "neutral"

    def test_calculate_aggregate_sentiment_positive(self) -> None:
        """Test aggregate calculation for positive sentiment."""
        analyzer = SentimentAnalyzer(bedrock_client=MagicMock())

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
                sentiment="positive",
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

        result = analyzer.calculate_aggregate_sentiment(analyses)

        assert result["overall_sentiment"] == "positive"
        assert result["sentiment_score"] > 0
        assert result["positive_count"] == 2
        assert result["neutral_count"] == 1

    def test_calculate_aggregate_sentiment_negative(self) -> None:
        """Test aggregate calculation for negative sentiment."""
        analyzer = SentimentAnalyzer(bedrock_client=MagicMock())

        analyses = [
            SentimentAnalysis(
                article_id="1",
                sentiment="negative",
                confidence=0.9,
                magnitude=0.9,
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
                magnitude=0.8,
                key_claims=[],
                topics=[],
                is_material=True,
                materiality_reason="",
                analysis_timestamp=datetime.now(timezone.utc),
            ),
        ]

        result = analyzer.calculate_aggregate_sentiment(analyses)

        assert result["overall_sentiment"] == "negative"
        assert result["sentiment_score"] < 0
        assert result["negative_count"] == 2

    def test_calculate_aggregate_sentiment_mixed(self) -> None:
        """Test aggregate calculation for mixed sentiment."""
        analyzer = SentimentAnalyzer(bedrock_client=MagicMock())

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
                confidence=0.9,
                magnitude=0.8,
                key_claims=[],
                topics=[],
                is_material=True,
                materiality_reason="",
                analysis_timestamp=datetime.now(timezone.utc),
            ),
        ]

        result = analyzer.calculate_aggregate_sentiment(analyses)

        # With equal positive and negative, should be neutral or mixed
        assert result["overall_sentiment"] in ["neutral", "mixed"]

    def test_calculate_aggregate_sentiment_empty(self) -> None:
        """Test aggregate calculation with no analyses."""
        analyzer = SentimentAnalyzer(bedrock_client=MagicMock())

        result = analyzer.calculate_aggregate_sentiment([])

        assert result["overall_sentiment"] == "neutral"
        assert result["sentiment_score"] == 0.0
        assert result["confidence"] == 0.0
