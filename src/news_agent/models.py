"""Data models for the News Sentiment Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal


@dataclass
class NewsArticle:
    """A single news article fetched from NewsAPI."""

    article_id: str  # Hash of URL
    title: str
    description: str
    content: str  # Full article text (if available)
    source_name: str  # "Reuters", "Bloomberg", etc.
    source_domain: str  # "reuters.com"
    author: str | None
    url: str
    published_at: datetime
    fetched_at: datetime
    ticker: str  # Associated ticker
    company_name: str  # Official company name

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "article_id": self.article_id,
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "source_name": self.source_name,
            "source_domain": self.source_domain,
            "author": self.author,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "fetched_at": self.fetched_at.isoformat(),
            "ticker": self.ticker,
            "company_name": self.company_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> NewsArticle:
        """Deserialize from a dict."""
        return cls(
            article_id=data["article_id"],
            title=data["title"],
            description=data["description"],
            content=data["content"],
            source_name=data["source_name"],
            source_domain=data["source_domain"],
            author=data.get("author"),
            url=data["url"],
            published_at=datetime.fromisoformat(data["published_at"]),
            fetched_at=datetime.fromisoformat(data["fetched_at"]),
            ticker=data["ticker"],
            company_name=data["company_name"],
        )


@dataclass
class SentimentAnalysis:
    """Sentiment analysis result for a single article."""

    article_id: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float  # 0.0 to 1.0
    magnitude: float  # Strength of sentiment 0.0 to 1.0
    key_claims: list[str]  # Extracted factual claims
    topics: list[str]  # "earnings", "lawsuit", "product", etc.
    is_material: bool  # Passed materiality filter
    materiality_reason: str  # Why it's material (or not)
    analysis_timestamp: datetime

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "article_id": self.article_id,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "magnitude": self.magnitude,
            "key_claims": self.key_claims,
            "topics": self.topics,
            "is_material": self.is_material,
            "materiality_reason": self.materiality_reason,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SentimentAnalysis:
        """Deserialize from a dict."""
        return cls(
            article_id=data["article_id"],
            sentiment=data["sentiment"],
            confidence=data["confidence"],
            magnitude=data["magnitude"],
            key_claims=data["key_claims"],
            topics=data["topics"],
            is_material=data["is_material"],
            materiality_reason=data["materiality_reason"],
            analysis_timestamp=datetime.fromisoformat(data["analysis_timestamp"]),
        )


@dataclass
class DailySentiment:
    """Aggregated sentiment data for a single day."""

    ticker: str
    date: date
    article_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    avg_sentiment_score: float  # -1.0 to 1.0
    material_articles: int
    top_topics: list[str]  # Most frequent topics that day

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "ticker": self.ticker,
            "date": self.date.isoformat(),
            "article_count": self.article_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "neutral_count": self.neutral_count,
            "avg_sentiment_score": self.avg_sentiment_score,
            "material_articles": self.material_articles,
            "top_topics": self.top_topics,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DailySentiment:
        """Deserialize from a dict."""
        return cls(
            ticker=data["ticker"],
            date=date.fromisoformat(data["date"]),
            article_count=data["article_count"],
            positive_count=data["positive_count"],
            negative_count=data["negative_count"],
            neutral_count=data["neutral_count"],
            avg_sentiment_score=data["avg_sentiment_score"],
            material_articles=data["material_articles"],
            top_topics=data["top_topics"],
        )


@dataclass
class NewsSentimentResult:
    """Combined structured + natural language output for sentiment analysis."""

    ticker: str
    company_name: str
    time_period: str  # "2025-01-02 to 2025-02-01"

    # Structured data
    overall_sentiment: Literal["positive", "negative", "neutral", "mixed"]
    sentiment_score: float  # -1.0 to 1.0
    confidence: float
    article_count: int
    material_article_count: int

    # Trend data
    trend_direction: Literal["improving", "worsening", "stable"]
    trend_magnitude: float  # How much change

    # Claims and topics
    key_claims: list[dict] = field(default_factory=list)  # [{claim, source, date, sentiment}]
    top_topics: list[str] = field(default_factory=list)

    # Natural language
    narrative_summary: str = ""  # 2-3 paragraph summary
    material_events: list[str] = field(default_factory=list)  # List of significant events

    # Sources
    sources: list[dict] = field(default_factory=list)  # [{title, url, date, sentiment}]

    # Status
    status: Literal["ok", "insufficient_data"] = "ok"
    status_message: str = ""

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "time_period": self.time_period,
            "overall_sentiment": self.overall_sentiment,
            "sentiment_score": self.sentiment_score,
            "confidence": self.confidence,
            "article_count": self.article_count,
            "material_article_count": self.material_article_count,
            "trend_direction": self.trend_direction,
            "trend_magnitude": self.trend_magnitude,
            "key_claims": self.key_claims,
            "top_topics": self.top_topics,
            "narrative_summary": self.narrative_summary,
            "material_events": self.material_events,
            "sources": self.sources,
            "status": self.status,
            "status_message": self.status_message,
        }
