"""Tool for retrieving sentiment trend data."""

from __future__ import annotations

from news_agent.analysis.trends import TrendAnalyzer
from news_agent.news.cache import NewsCache


def get_trends(
    ticker: str,
    days_back: int = 30,
    cache: NewsCache | None = None,
) -> dict:
    """Get sentiment trend data for a ticker.

    Args:
        ticker: Stock ticker symbol
        days_back: Number of days to look back
        cache: Optional cache instance

    Returns:
        Dict with trend analysis and daily scores
    """
    news_cache = cache or NewsCache()
    trend_analyzer = TrendAnalyzer(cache=news_cache)

    return trend_analyzer.get_trend_data(ticker, days_back)
