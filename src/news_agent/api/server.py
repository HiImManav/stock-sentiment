"""FastAPI server for the News Sentiment Agent."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from news_agent.agent import NewsSentimentAgent
from news_agent.tools.analyze import analyze_sentiment
from news_agent.tools.fetch_news import fetch_news
from news_agent.tools.trends import get_trends

app = FastAPI(
    title="News Sentiment Agent API",
    description="API for analyzing news sentiment for publicly traded companies",
    version="0.1.0",
)

# Global agent instance for session preservation
_agent: NewsSentimentAgent | None = None


def get_agent() -> NewsSentimentAgent:
    """Get or create the global agent instance."""
    global _agent
    if _agent is None:
        _agent = NewsSentimentAgent()
    return _agent


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class FetchRequest(BaseModel):
    """Request model for fetching news."""

    ticker: str
    days_back: int = 30
    force_refresh: bool = False


class AnalyzeRequest(BaseModel):
    """Request model for sentiment analysis."""

    ticker: str
    question: str | None = None
    filter_material_only: bool = True
    max_articles: int = 20


class TrendsRequest(BaseModel):
    """Request model for trend data."""

    ticker: str
    days_back: int = 30


class QueryRequest(BaseModel):
    """Request model for natural language query."""

    query: str


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    service: str


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    answer: str
    session_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", service="news-sentiment-agent")


@app.post("/news/fetch")
def fetch_news_endpoint(request: FetchRequest) -> dict[str, Any]:
    """Fetch news articles for a ticker.

    Returns article count and metadata. Articles are cached for 24 hours.
    """
    result = fetch_news(
        ticker=request.ticker,
        days_back=request.days_back,
        force_refresh=request.force_refresh,
    )

    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))

    return result


@app.post("/news/analyze")
def analyze_sentiment_endpoint(request: AnalyzeRequest) -> dict[str, Any]:
    """Analyze sentiment of cached news articles.

    Articles must be fetched first using /news/fetch.
    """
    result = analyze_sentiment(
        ticker=request.ticker,
        question=request.question,
        filter_material_only=request.filter_material_only,
        max_articles=request.max_articles,
    )

    if result["status"] == "no_articles":
        raise HTTPException(
            status_code=404,
            detail="No cached articles found. Use /news/fetch first.",
        )

    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))

    return result


@app.get("/news/trends/{ticker}")
def get_trends_endpoint(ticker: str, days_back: int = 30) -> dict[str, Any]:
    """Get sentiment trend data for a ticker."""
    result = get_trends(ticker=ticker, days_back=days_back)
    return result


@app.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest) -> QueryResponse:
    """Natural language query to the agent.

    The agent will automatically fetch news, analyze sentiment, and
    synthesize a response.
    """
    agent = get_agent()

    try:
        answer = agent.query(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        answer=answer,
        session_id=agent.memory.session_id,
    )


@app.post("/reset")
def reset_session() -> dict[str, str]:
    """Reset the agent session (clear conversation and memory)."""
    global _agent
    if _agent is not None:
        _agent.reset()
    return {"status": "ok", "message": "Session reset"}
