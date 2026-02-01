"""Tool implementations for the News Sentiment Agent."""

from news_agent.tools.analyze import analyze_sentiment
from news_agent.tools.fetch_news import fetch_news
from news_agent.tools.trends import get_trends

__all__ = ["fetch_news", "analyze_sentiment", "get_trends"]
