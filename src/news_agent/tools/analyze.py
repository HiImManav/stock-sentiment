"""Tool for analyzing sentiment of fetched news articles."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from news_agent.analysis.materiality import MaterialityFilter
from news_agent.analysis.sentiment import SentimentAnalyzer
from news_agent.analysis.trends import TrendAnalyzer
from news_agent.models import NewsSentimentResult
from news_agent.news.cache import NewsCache


def analyze_sentiment(
    ticker: str,
    question: str | None = None,
    filter_material_only: bool = True,
    max_articles: int = 20,
    cache: NewsCache | None = None,
    sentiment_analyzer: SentimentAnalyzer | None = None,
) -> dict:
    """Analyze sentiment of cached news articles.

    Args:
        ticker: Stock ticker symbol
        question: Optional question to focus analysis
        filter_material_only: Only analyze material articles
        max_articles: Maximum articles to analyze
        cache: Optional cache instance
        sentiment_analyzer: Optional sentiment analyzer instance

    Returns:
        Dict with sentiment analysis results
    """
    news_cache = cache or NewsCache()
    analyzer = sentiment_analyzer or SentimentAnalyzer()
    materiality_filter = MaterialityFilter()
    trend_analyzer = TrendAnalyzer(cache=news_cache)

    # Load cached articles
    articles, is_stale, _ = news_cache.get_cached_articles(ticker)

    if not articles:
        return {
            "status": "no_articles",
            "ticker": ticker.upper(),
            "message": "No cached articles found. Use fetch_news first.",
        }

    # Apply materiality filter (Layers 1 & 2)
    filtered = materiality_filter.filter_articles(
        articles, min_tier=2 if filter_material_only else 3
    )

    # Sort by weight and limit
    filtered.sort(key=lambda x: x[1], reverse=True)
    articles_to_analyze = [a for a, _, _ in filtered[:max_articles]]

    if not articles_to_analyze:
        return {
            "status": "no_material_articles",
            "ticker": ticker.upper(),
            "message": "No material articles found after filtering.",
            "total_articles": len(articles),
        }

    # Analyze sentiment (includes Layer 3 LLM materiality check)
    analyses = analyzer.analyze_articles_batch(articles_to_analyze)

    # Calculate aggregate metrics
    aggregate = analyzer.calculate_aggregate_sentiment(analyses)

    # Filter to material only for final results
    material_analyses = [a for a in analyses if a.is_material]

    # Collect all claims and topics
    all_claims: list[dict] = []
    all_topics: list[str] = []
    material_events: list[str] = []

    article_by_id = {a.article_id: a for a in articles_to_analyze}

    for analysis in analyses:
        article = article_by_id.get(analysis.article_id)
        if not article:
            continue

        # Add claims with source attribution
        for claim in analysis.key_claims:
            all_claims.append({
                "claim": claim,
                "source": article.source_name,
                "date": article.published_at.strftime("%Y-%m-%d"),
                "sentiment": analysis.sentiment,
            })

        all_topics.extend(analysis.topics)

        # Track material events
        if analysis.is_material and analysis.materiality_reason:
            material_events.append(analysis.materiality_reason)

    # Get top topics
    topic_counts = Counter(all_topics)
    top_topics = [t for t, _ in topic_counts.most_common(5)]

    # Build sources list
    sources = [
        {
            "title": article_by_id[a.article_id].title,
            "url": article_by_id[a.article_id].url,
            "date": article_by_id[a.article_id].published_at.strftime("%Y-%m-%d"),
            "sentiment": a.sentiment,
        }
        for a in analyses
        if a.article_id in article_by_id
    ]

    # Get trend data
    trend_data = trend_analyzer.get_trend_data(ticker)
    trend_direction = trend_data.get("trend_direction", "stable")
    trend_magnitude = trend_data.get("trend_magnitude", 0.0)

    # Calculate date range from articles
    if articles:
        min_date = min(a.published_at for a in articles)
        max_date = max(a.published_at for a in articles)
        time_period = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
    else:
        time_period = ""

    # Store daily sentiment for trending
    today = datetime.now(timezone.utc).date()
    articles_by_date = {a.article_id: a.published_at.date() for a in articles_to_analyze}
    by_date = trend_analyzer.group_analyses_by_date(analyses, articles_by_date)

    for date_val, date_analyses in by_date.items():
        daily_sentiment = trend_analyzer.calculate_daily_sentiment(
            ticker, date_analyses, date_val
        )
        trend_analyzer.store_daily_sentiment(ticker, daily_sentiment)

    # Cache the analysis results
    news_cache.cache_analysis(ticker, analyses)

    # Build narrative summary
    narrative = _build_narrative_summary(
        ticker=ticker,
        company_name=articles[0].company_name if articles else ticker,
        aggregate=aggregate,
        material_events=material_events[:5],
        all_claims=all_claims[:10],
        trend_direction=trend_direction,
    )

    return {
        "status": "ok",
        "result": {
            "ticker": ticker.upper(),
            "company_name": articles[0].company_name if articles else ticker,
            "time_period": time_period,
            "overall_sentiment": aggregate["overall_sentiment"],
            "sentiment_score": aggregate["sentiment_score"],
            "confidence": aggregate["confidence"],
            "article_count": len(articles),
            "articles_analyzed": len(analyses),
            "material_article_count": len(material_analyses),
            "trend_direction": trend_direction,
            "trend_magnitude": trend_magnitude,
            "key_claims": all_claims[:15],
            "top_topics": top_topics,
            "narrative_summary": narrative,
            "material_events": list(set(material_events))[:10],
            "sources": sources[:20],
        },
    }


def _build_narrative_summary(
    ticker: str,
    company_name: str,
    aggregate: dict,
    material_events: list[str],
    all_claims: list[dict],
    trend_direction: str,
) -> str:
    """Build a natural language narrative summary."""
    sentiment = aggregate["overall_sentiment"]
    score = aggregate["sentiment_score"]
    pos_count = aggregate["positive_count"]
    neg_count = aggregate["negative_count"]
    neutral_count = aggregate["neutral_count"]

    # Opening paragraph - overall sentiment
    if sentiment == "positive":
        opening = f"Recent news coverage of {company_name} ({ticker}) has been predominantly positive."
    elif sentiment == "negative":
        opening = f"Recent news coverage of {company_name} ({ticker}) has been predominantly negative."
    elif sentiment == "mixed":
        opening = f"Recent news coverage of {company_name} ({ticker}) has been mixed, with both positive and negative stories."
    else:
        opening = f"Recent news coverage of {company_name} ({ticker}) has been largely neutral."

    # Add sentiment breakdown
    opening += f" Of the articles analyzed, {pos_count} were positive, {neg_count} were negative, and {neutral_count} were neutral."

    # Trend paragraph
    if trend_direction == "improving":
        trend_text = "Sentiment has been improving over the analysis period."
    elif trend_direction == "worsening":
        trend_text = "Sentiment has been worsening over the analysis period."
    else:
        trend_text = "Sentiment has remained relatively stable over the analysis period."

    # Material events paragraph
    if material_events:
        events_text = "Key material events include: " + "; ".join(material_events[:3]) + "."
    else:
        events_text = "No major material events were identified in the coverage."

    # Key claims
    if all_claims:
        claims_text = "Notable claims from the coverage: "
        claim_strings = [f'"{c["claim"]}" ({c["source"]})' for c in all_claims[:3]]
        claims_text += "; ".join(claim_strings) + "."
    else:
        claims_text = ""

    paragraphs = [opening, trend_text, events_text]
    if claims_text:
        paragraphs.append(claims_text)

    return "\n\n".join(paragraphs)
