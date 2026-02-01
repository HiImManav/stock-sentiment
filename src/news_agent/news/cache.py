"""S3 caching layer for news articles and analysis results."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

from news_agent.models import DailySentiment, NewsArticle, SentimentAnalysis


class NewsCache:
    """S3 cache for news articles, analysis results, and trend data."""

    def __init__(
        self,
        bucket: str | None = None,
        s3_client: Any | None = None,
        ttl_hours: int = 24,
    ) -> None:
        self._bucket = bucket or os.environ.get(
            "NEWS_SENTIMENT_BUCKET", "news-sentiment-cache"
        )
        self._s3 = s3_client or boto3.client("s3")
        self._ttl_hours = ttl_hours

    # -- Article caching --------------------------------------------------

    def _articles_key(self, ticker: str, date_str: str) -> str:
        """Build the S3 key for cached articles."""
        return f"news/{ticker}/{date_str}/articles.json"

    def _analysis_key(self, ticker: str, date_str: str) -> str:
        """Build the S3 key for cached analysis."""
        return f"news/{ticker}/{date_str}/analysis.json"

    def _trends_key(self, ticker: str) -> str:
        """Build the S3 key for trend data."""
        return f"trends/{ticker}/daily_sentiment.json"

    def _rate_limits_key(self) -> str:
        """Build the S3 key for rate limit tracking."""
        return "metadata/rate_limits.json"

    def get_cached_articles(
        self, ticker: str, days_back: int = 30
    ) -> tuple[list[NewsArticle], bool, datetime | None]:
        """Get cached articles for a ticker.

        Args:
            ticker: Stock ticker
            days_back: Lookback period

        Returns:
            Tuple of (articles, is_stale, cache_time)
            - articles: List of cached articles (empty if no cache)
            - is_stale: True if cache is older than TTL
            - cache_time: When the cache was created
        """
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = self._articles_key(ticker.upper(), date_str)

        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=key)
            data = json.loads(resp["Body"].read().decode("utf-8"))

            # Check cache metadata
            cache_time = datetime.fromisoformat(data.get("cached_at", ""))
            is_stale = datetime.now(timezone.utc) - cache_time > timedelta(
                hours=self._ttl_hours
            )

            articles = [NewsArticle.from_dict(a) for a in data.get("articles", [])]

            return articles, is_stale, cache_time

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return [], True, None
            raise
        except (KeyError, ValueError):
            return [], True, None

    def cache_articles(
        self, ticker: str, articles: list[NewsArticle], days_back: int = 30
    ) -> None:
        """Cache articles for a ticker.

        Args:
            ticker: Stock ticker
            articles: List of articles to cache
            days_back: Lookback period used
        """
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = self._articles_key(ticker.upper(), date_str)

        data = {
            "ticker": ticker.upper(),
            "days_back": days_back,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "article_count": len(articles),
            "articles": [a.to_dict() for a in articles],
        }

        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=json.dumps(data, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )

    # -- Analysis caching -------------------------------------------------

    def get_cached_analysis(
        self, ticker: str
    ) -> tuple[list[SentimentAnalysis], bool, datetime | None]:
        """Get cached sentiment analysis for a ticker.

        Returns:
            Tuple of (analyses, is_stale, cache_time)
        """
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = self._analysis_key(ticker.upper(), date_str)

        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=key)
            data = json.loads(resp["Body"].read().decode("utf-8"))

            cache_time = datetime.fromisoformat(data.get("cached_at", ""))
            is_stale = datetime.now(timezone.utc) - cache_time > timedelta(
                hours=self._ttl_hours
            )

            analyses = [SentimentAnalysis.from_dict(a) for a in data.get("analyses", [])]

            return analyses, is_stale, cache_time

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return [], True, None
            raise
        except (KeyError, ValueError):
            return [], True, None

    def cache_analysis(self, ticker: str, analyses: list[SentimentAnalysis]) -> None:
        """Cache sentiment analysis results for a ticker."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = self._analysis_key(ticker.upper(), date_str)

        data = {
            "ticker": ticker.upper(),
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "analysis_count": len(analyses),
            "analyses": [a.to_dict() for a in analyses],
        }

        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=json.dumps(data, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )

    # -- Trend data caching -----------------------------------------------

    def get_trend_data(self, ticker: str) -> list[DailySentiment]:
        """Get historical daily sentiment data for a ticker."""
        key = self._trends_key(ticker.upper())

        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=key)
            data = json.loads(resp["Body"].read().decode("utf-8"))

            return [DailySentiment.from_dict(d) for d in data.get("daily_scores", [])]

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return []
            raise

    def save_trend_data(self, ticker: str, daily_scores: list[DailySentiment]) -> None:
        """Save daily sentiment data for trending."""
        key = self._trends_key(ticker.upper())

        # Load existing data and merge
        existing = self.get_trend_data(ticker)
        existing_by_date = {d.date: d for d in existing}

        # Update with new scores
        for score in daily_scores:
            existing_by_date[score.date] = score

        # Sort by date and keep last 90 days
        all_scores = sorted(existing_by_date.values(), key=lambda x: x.date)
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=90)
        all_scores = [s for s in all_scores if s.date >= cutoff]

        data = {
            "ticker": ticker.upper(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "daily_scores": [s.to_dict() for s in all_scores],
        }

        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=json.dumps(data, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )

    # -- Rate limit tracking ----------------------------------------------

    def get_rate_limit_status(self) -> dict:
        """Get current rate limit usage tracking."""
        key = self._rate_limits_key()

        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=key)
            return json.loads(resp["Body"].read().decode("utf-8"))
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return {"requests_today": 0, "date": "", "limit": 500}
            raise

    def increment_rate_limit(self) -> int:
        """Increment the daily request counter.

        Returns:
            Current request count for today
        """
        status = self.get_rate_limit_status()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if status.get("date") != today:
            status = {"requests_today": 0, "date": today, "limit": 500}

        status["requests_today"] += 1

        self._s3.put_object(
            Bucket=self._bucket,
            Key=self._rate_limits_key(),
            Body=json.dumps(status).encode("utf-8"),
            ContentType="application/json",
        )

        return status["requests_today"]

    def is_rate_limited(self) -> bool:
        """Check if we've exceeded the daily rate limit."""
        status = self.get_rate_limit_status()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if status.get("date") != today:
            return False

        return status.get("requests_today", 0) >= status.get("limit", 500)
