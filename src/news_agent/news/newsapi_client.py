"""NewsAPI.org client with rate limiting and caching support."""

from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timedelta, timezone

import httpx

from news_agent.models import NewsArticle

_BASE_URL = "https://newsapi.org/v2"
_MIN_INTERVAL = 0.2  # 5 requests per second max


class NewsAPIError(Exception):
    """Error from NewsAPI."""

    pass


class RateLimitError(NewsAPIError):
    """Rate limit exceeded."""

    pass


class NewsAPIClient:
    """Client for NewsAPI.org with rate limiting."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("NEWSAPI_KEY")
        if not self._api_key:
            raise ValueError("NEWSAPI_KEY environment variable or api_key parameter required")
        self._last_request_time: float = 0.0
        self._client = httpx.Client(
            headers={"X-Api-Key": self._api_key},
            timeout=30.0,
        )

    def _throttle(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    def _get(self, endpoint: str, params: dict) -> dict:
        """Make a rate-limited GET request to NewsAPI."""
        self._throttle()
        url = f"{_BASE_URL}/{endpoint}"
        resp = self._client.get(url, params=params)

        if resp.status_code == 429:
            raise RateLimitError("NewsAPI rate limit exceeded")

        data = resp.json()
        if data.get("status") == "error":
            code = data.get("code", "unknown")
            message = data.get("message", "Unknown error")
            if code == "rateLimited":
                raise RateLimitError(message)
            raise NewsAPIError(f"{code}: {message}")

        return data

    @staticmethod
    def _generate_article_id(url: str) -> str:
        """Generate a unique ID for an article based on its URL."""
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from a URL."""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            return parsed.netloc.lower().replace("www.", "")
        except Exception:
            return ""

    def fetch_news(
        self,
        query: str,
        ticker: str,
        company_name: str,
        days_back: int = 30,
        page_size: int = 100,
        language: str = "en",
    ) -> list[NewsArticle]:
        """Fetch news articles for a query.

        Args:
            query: Search query (e.g., '"Apple Inc" OR AAPL')
            ticker: Stock ticker for association
            company_name: Company name for association
            days_back: Number of days to look back
            page_size: Maximum articles to fetch (max 100)
            language: Language filter (default: English)

        Returns:
            List of NewsArticle objects
        """
        from_date = datetime.now(timezone.utc) - timedelta(days=days_back)

        params = {
            "q": query,
            "from": from_date.strftime("%Y-%m-%d"),
            "language": language,
            "sortBy": "publishedAt",
            "pageSize": min(page_size, 100),
        }

        data = self._get("everything", params)

        articles: list[NewsArticle] = []
        now = datetime.now(timezone.utc)

        for item in data.get("articles", []):
            # Skip articles without essential fields
            url = item.get("url", "")
            title = item.get("title", "")
            if not url or not title or title == "[Removed]":
                continue

            # Parse published date
            published_str = item.get("publishedAt", "")
            try:
                published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                published_at = now

            source = item.get("source", {})
            article = NewsArticle(
                article_id=self._generate_article_id(url),
                title=title,
                description=item.get("description") or "",
                content=item.get("content") or "",
                source_name=source.get("name") or "",
                source_domain=self._extract_domain(url),
                author=item.get("author"),
                url=url,
                published_at=published_at,
                fetched_at=now,
                ticker=ticker,
                company_name=company_name,
            )
            articles.append(article)

        return articles

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> NewsAPIClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
