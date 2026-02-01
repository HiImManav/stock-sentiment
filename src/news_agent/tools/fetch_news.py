"""Tool for fetching news articles from NewsAPI."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from news_agent.news.cache import NewsCache
from news_agent.news.entity_resolver import EntityResolver
from news_agent.news.newsapi_client import NewsAPIClient, RateLimitError


def fetch_news(
    ticker: str,
    days_back: int = 30,
    force_refresh: bool = False,
    newsapi_client: NewsAPIClient | None = None,
    entity_resolver: EntityResolver | None = None,
    cache: NewsCache | None = None,
) -> dict:
    """Fetch news articles for a company.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        days_back: Number of days to look back (default: 30)
        force_refresh: Force refresh from API even if cached
        newsapi_client: Optional NewsAPI client instance
        entity_resolver: Optional entity resolver instance
        cache: Optional cache instance

    Returns:
        Dict with status, article count, sources, etc.
    """
    # Initialize components
    resolver = entity_resolver or EntityResolver()
    news_cache = cache or NewsCache()

    # Resolve ticker to company info
    try:
        company_info = resolver.resolve(ticker)
    except ValueError as e:
        return {
            "status": "error",
            "ticker": ticker.upper(),
            "message": str(e),
        }

    company_name = company_info["company_name"]
    ticker_upper = company_info["ticker"]

    # Check cache first (unless force refresh)
    if not force_refresh:
        cached_articles, is_stale, cache_time = news_cache.get_cached_articles(
            ticker_upper, days_back
        )

        if cached_articles and not is_stale:
            # Return cached data
            sources = list(set(a.source_name for a in cached_articles if a.source_name))
            date_range = _calculate_date_range(days_back)

            return {
                "status": "ok",
                "ticker": ticker_upper,
                "company_name": company_name,
                "article_count": len(cached_articles),
                "date_range": date_range,
                "sources": sources[:10],  # Top 10 sources
                "cache_hit": True,
                "cache_time": cache_time.isoformat() if cache_time else None,
            }

    # Check rate limits before making API call
    if news_cache.is_rate_limited():
        # Try to use stale cache if available
        cached_articles, _, cache_time = news_cache.get_cached_articles(
            ticker_upper, days_back
        )
        if cached_articles:
            sources = list(set(a.source_name for a in cached_articles if a.source_name))
            return {
                "status": "ok",
                "ticker": ticker_upper,
                "company_name": company_name,
                "article_count": len(cached_articles),
                "date_range": _calculate_date_range(days_back),
                "sources": sources[:10],
                "cache_hit": True,
                "stale": True,
                "message": "Using cached data due to rate limits",
            }
        return {
            "status": "rate_limited",
            "ticker": ticker_upper,
            "message": "Daily API rate limit reached and no cached data available",
        }

    # Fetch from NewsAPI
    client = newsapi_client or NewsAPIClient()
    try:
        query = resolver.build_search_query(ticker)
        articles = client.fetch_news(
            query=query,
            ticker=ticker_upper,
            company_name=company_name,
            days_back=days_back,
        )

        # Track rate limit usage
        news_cache.increment_rate_limit()

    except RateLimitError:
        return {
            "status": "rate_limited",
            "ticker": ticker_upper,
            "message": "NewsAPI rate limit exceeded",
        }
    except Exception as e:
        return {
            "status": "error",
            "ticker": ticker_upper,
            "message": f"Failed to fetch news: {str(e)}",
        }
    finally:
        if newsapi_client is None:
            client.close()

    # Check for insufficient coverage
    if len(articles) < 3:
        return {
            "status": "insufficient_data",
            "ticker": ticker_upper,
            "company_name": company_name,
            "article_count": len(articles),
            "message": (
                f"Insufficient news coverage for meaningful analysis. "
                f"Only {len(articles)} article(s) found in the past {days_back} days. "
                f"Consider expanding the time window or checking if the ticker is correct."
            ),
            "date_range": _calculate_date_range(days_back),
        }

    # Cache the results
    news_cache.cache_articles(ticker_upper, articles, days_back)

    # Build response
    sources = list(set(a.source_name for a in articles if a.source_name))
    date_range = _calculate_date_range(days_back)

    return {
        "status": "ok",
        "ticker": ticker_upper,
        "company_name": company_name,
        "article_count": len(articles),
        "date_range": date_range,
        "sources": sources[:10],
        "cache_hit": False,
    }


def _calculate_date_range(days_back: int) -> str:
    """Calculate the date range string for the lookback period."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)
    return f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
