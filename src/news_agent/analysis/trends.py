"""Daily sentiment tracking and trend calculation."""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Literal

from news_agent.models import DailySentiment, SentimentAnalysis
from news_agent.news.cache import NewsCache


class TrendAnalyzer:
    """Analyzes sentiment trends over time."""

    def __init__(self, cache: NewsCache | None = None) -> None:
        self._cache = cache or NewsCache()

    def calculate_daily_sentiment(
        self,
        ticker: str,
        analyses: list[SentimentAnalysis],
        target_date: date | None = None,
    ) -> DailySentiment:
        """Calculate aggregated sentiment for a single day.

        Args:
            ticker: Stock ticker
            analyses: List of sentiment analyses for articles from that day
            target_date: The date being analyzed (defaults to today)

        Returns:
            DailySentiment aggregation
        """
        if target_date is None:
            target_date = datetime.now(timezone.utc).date()

        if not analyses:
            return DailySentiment(
                ticker=ticker.upper(),
                date=target_date,
                article_count=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                avg_sentiment_score=0.0,
                material_articles=0,
                top_topics=[],
            )

        # Count sentiments
        positive = sum(1 for a in analyses if a.sentiment == "positive")
        negative = sum(1 for a in analyses if a.sentiment == "negative")
        neutral = sum(1 for a in analyses if a.sentiment == "neutral")
        material = sum(1 for a in analyses if a.is_material)

        # Calculate weighted sentiment score
        total_weight = 0.0
        weighted_score = 0.0
        all_topics: list[str] = []

        for analysis in analyses:
            weight = analysis.confidence * analysis.magnitude
            if analysis.sentiment == "positive":
                weighted_score += weight
            elif analysis.sentiment == "negative":
                weighted_score -= weight
            total_weight += weight
            all_topics.extend(analysis.topics)

        avg_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # Get top topics
        topic_counts = Counter(all_topics)
        top_topics = [t for t, _ in topic_counts.most_common(5)]

        return DailySentiment(
            ticker=ticker.upper(),
            date=target_date,
            article_count=len(analyses),
            positive_count=positive,
            negative_count=negative,
            neutral_count=neutral,
            avg_sentiment_score=round(avg_score, 3),
            material_articles=material,
            top_topics=top_topics,
        )

    def group_analyses_by_date(
        self, analyses: list[SentimentAnalysis], articles_by_id: dict[str, date]
    ) -> dict[date, list[SentimentAnalysis]]:
        """Group analyses by their article's publication date.

        Args:
            analyses: List of sentiment analyses
            articles_by_id: Mapping of article_id to publication date

        Returns:
            Dict mapping dates to their analyses
        """
        by_date: dict[date, list[SentimentAnalysis]] = {}

        for analysis in analyses:
            pub_date = articles_by_id.get(analysis.article_id)
            if pub_date:
                if pub_date not in by_date:
                    by_date[pub_date] = []
                by_date[pub_date].append(analysis)

        return by_date

    def calculate_trend(
        self, daily_scores: list[DailySentiment], days_back: int = 30
    ) -> dict:
        """Calculate trend direction and magnitude from daily scores.

        Args:
            daily_scores: List of daily sentiment data
            days_back: Number of days to analyze

        Returns:
            Dict with trend_direction, trend_magnitude, and details
        """
        if not daily_scores:
            return {
                "trend_direction": "stable",
                "trend_magnitude": 0.0,
                "current_score": 0.0,
                "score_period_ago": 0.0,
                "inflection_points": [],
            }

        # Sort by date
        sorted_scores = sorted(daily_scores, key=lambda x: x.date)

        # Filter to requested period
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=days_back)
        recent_scores = [s for s in sorted_scores if s.date >= cutoff]

        if not recent_scores:
            return {
                "trend_direction": "stable",
                "trend_magnitude": 0.0,
                "current_score": 0.0,
                "score_period_ago": 0.0,
                "inflection_points": [],
            }

        # Get first and last scores
        first_score = recent_scores[0].avg_sentiment_score
        last_score = recent_scores[-1].avg_sentiment_score
        score_change = last_score - first_score

        # Determine trend direction
        trend_direction: Literal["improving", "worsening", "stable"]
        if abs(score_change) < 0.1:
            trend_direction = "stable"
        elif score_change > 0:
            trend_direction = "improving"
        else:
            trend_direction = "worsening"

        # Find inflection points (significant day-over-day changes)
        inflection_points: list[dict] = []
        for i in range(1, len(recent_scores)):
            prev = recent_scores[i - 1]
            curr = recent_scores[i]
            change = curr.avg_sentiment_score - prev.avg_sentiment_score

            if abs(change) >= 0.2:  # Significant change threshold
                direction = "improved" if change > 0 else "declined"
                inflection_points.append(
                    {
                        "date": curr.date.isoformat(),
                        "change": round(change, 3),
                        "description": f"Sentiment {direction} by {abs(change):.2f}",
                        "top_topics": curr.top_topics[:3] if curr.top_topics else [],
                    }
                )

        return {
            "trend_direction": trend_direction,
            "trend_magnitude": round(abs(score_change), 3),
            "current_score": round(last_score, 3),
            "score_period_ago": round(first_score, 3),
            "inflection_points": inflection_points,
        }

    def store_daily_sentiment(
        self, ticker: str, daily_sentiment: DailySentiment
    ) -> None:
        """Store a daily sentiment record for trending.

        Args:
            ticker: Stock ticker
            daily_sentiment: The daily sentiment data to store
        """
        self._cache.save_trend_data(ticker, [daily_sentiment])

    def get_trend_data(
        self, ticker: str, days_back: int = 30
    ) -> dict:
        """Get trend data for a ticker.

        Args:
            ticker: Stock ticker
            days_back: Number of days to look back

        Returns:
            Dict with trend analysis and daily scores
        """
        daily_scores = self._cache.get_trend_data(ticker)

        if not daily_scores:
            return {
                "status": "no_data",
                "ticker": ticker.upper(),
                "message": "No historical sentiment data available",
            }

        trend = self.calculate_trend(daily_scores, days_back)

        # Format daily scores for output
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=days_back)
        recent_scores = [s for s in daily_scores if s.date >= cutoff]
        recent_scores = sorted(recent_scores, key=lambda x: x.date)

        return {
            "status": "ok",
            "ticker": ticker.upper(),
            "trend_direction": trend["trend_direction"],
            "trend_magnitude": trend["trend_magnitude"],
            "current_score": trend["current_score"],
            "score_period_ago": trend["score_period_ago"],
            "daily_scores": [
                {
                    "date": s.date.isoformat(),
                    "score": s.avg_sentiment_score,
                    "articles": s.article_count,
                    "material": s.material_articles,
                }
                for s in recent_scores
            ],
            "inflection_points": trend["inflection_points"],
        }
