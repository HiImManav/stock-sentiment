"""FastAPI server for the Orchestration Agent."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from orchestrator.agent import OrchestrationAgent

app = FastAPI(
    title="Orchestration Agent API",
    description=(
        "API for coordinating news sentiment and SEC filings analysis "
        "to provide unified company intelligence"
    ),
    version="0.1.0",
)

# Global agent instance for session preservation
_agent: OrchestrationAgent | None = None


def get_agent() -> OrchestrationAgent:
    """Get or create the global agent instance."""
    global _agent
    if _agent is None:
        _agent = OrchestrationAgent()
    return _agent


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request model for orchestrated query."""

    query: str = Field(..., description="Natural language query about a company")
    ticker: str | None = Field(
        None, description="Stock ticker symbol (e.g., AAPL). Auto-detected if not provided."
    )
    sources: list[Literal["news", "sec"]] | None = Field(
        None,
        description="Sources to query. Options: ['news'], ['sec'], or ['news', 'sec']. Auto-detected if not provided.",
    )
    enable_comparison: bool = Field(
        True, description="Whether to enable discrepancy detection between sources"
    )


class QueryResponse(BaseModel):
    """Response model for orchestrated query."""

    answer: str = Field(..., description="Synthesized response from agent(s)")
    agents_used: list[str] = Field(..., description="List of agents that were queried")
    had_discrepancies: bool = Field(
        ..., description="Whether discrepancies were detected between sources"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score based on source alignment"
    )
    execution_time_ms: float = Field(..., description="Total execution time in milliseconds")
    session_id: str = Field(..., description="Current session identifier")
    query_id: str = Field(..., description="Unique identifier for this query")


class CompareRequest(BaseModel):
    """Request model for explicit comparison."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")


class CompareResponse(BaseModel):
    """Response model for explicit comparison."""

    answer: str = Field(..., description="Synthesized comparison response")
    ticker: str = Field(..., description="Ticker that was compared")
    had_discrepancies: bool = Field(
        ..., description="Whether discrepancies were detected"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    discrepancies: list[dict[str, Any]] = Field(
        default_factory=list, description="List of detected discrepancies"
    )
    agreements: list[dict[str, Any]] = Field(
        default_factory=list, description="List of detected agreements"
    )
    alignment_score: float | None = Field(
        None, description="Alignment score between sources (-1.0 to 1.0)"
    )
    execution_time_ms: float = Field(..., description="Total execution time in milliseconds")
    session_id: str = Field(..., description="Current session identifier")
    query_id: str = Field(..., description="Unique identifier for this query")


class SessionSummaryResponse(BaseModel):
    """Response model for session summary."""

    session_id: str = Field(..., description="Current session identifier")
    query_count: int = Field(..., description="Number of queries in this session")
    tickers_analyzed: list[str] = Field(..., description="Tickers analyzed in this session")
    discrepancy_rate: float = Field(
        ..., description="Percentage of queries with discrepancies"
    )
    average_confidence: float = Field(..., description="Average confidence score")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    service: str


class ResetResponse(BaseModel):
    """Response model for reset endpoint."""

    status: str
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", service="orchestrator")


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Execute an orchestrated query.

    Coordinates between news sentiment and SEC filings agents to provide
    unified company intelligence. Automatically routes queries to the
    appropriate agent(s) based on content.

    The query is classified and routed to:
    - news_agent only: For news/sentiment/headline queries
    - sec_agent only: For SEC filing/10-K/10-Q queries
    - both agents: For comparison or general company queries

    Results are synthesized using an LLM to provide a coherent response
    that integrates insights from all sources.
    """
    agent = get_agent()

    # Determine forced route from sources parameter
    force_route: Literal["news_only", "sec_only", "both"] | None = None
    if request.sources is not None:
        if request.sources == ["news"]:
            force_route = "news_only"
        elif request.sources == ["sec"]:
            force_route = "sec_only"
        elif set(request.sources) == {"news", "sec"}:
            force_route = "both"

    try:
        result = agent.query(
            user_message=request.query,
            ticker=request.ticker,
            force_route=force_route,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        answer=result.response,
        agents_used=result.agents_used,
        had_discrepancies=result.had_discrepancies,
        confidence=result.confidence,
        execution_time_ms=result.execution_time_ms,
        session_id=agent.session_id,
        query_id=result.query_id,
    )


@app.post("/compare", response_model=CompareResponse)
def compare_endpoint(request: CompareRequest) -> CompareResponse:
    """Run an explicit comparison between news and SEC data for a ticker.

    Forces both agents to run and enables comparison mode. Useful for
    getting a comprehensive view of a company by analyzing both recent
    news sentiment and SEC filing data, then identifying any discrepancies.
    """
    agent = get_agent()

    try:
        result = agent.compare(ticker=request.ticker)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Extract discrepancies and agreements from comparison result
    discrepancies: list[dict[str, Any]] = []
    agreements: list[dict[str, Any]] = []
    alignment_score: float | None = None

    if result.comparison is not None:
        # Use ComparisonResult.to_dict() which handles serialization
        comparison_dict = result.comparison.to_dict()
        discrepancies = comparison_dict.get("discrepancies", [])  # type: ignore[assignment]
        agreements = comparison_dict.get("agreements", [])  # type: ignore[assignment]
        alignment_score = result.comparison.overall_alignment

    return CompareResponse(
        answer=result.response,
        ticker=request.ticker.upper(),
        had_discrepancies=result.had_discrepancies,
        confidence=result.confidence,
        discrepancies=discrepancies,
        agreements=agreements,
        alignment_score=alignment_score,
        execution_time_ms=result.execution_time_ms,
        session_id=agent.session_id,
        query_id=result.query_id,
    )


@app.get("/session/summary", response_model=SessionSummaryResponse)
def session_summary_endpoint() -> SessionSummaryResponse:
    """Get a summary of the current session.

    Returns statistics about queries made in this session including
    the number of queries, tickers analyzed, and discrepancy rate.
    """
    agent = get_agent()
    summary = agent.get_session_summary()

    return SessionSummaryResponse(
        session_id=summary["session_id"],
        query_count=summary["query_count"],
        tickers_analyzed=summary["tickers_analyzed"],
        discrepancy_rate=summary["discrepancy_rate"],
        average_confidence=summary["average_confidence"],
    )


@app.get("/session/history")
def session_history_endpoint(
    ticker: str | None = None, limit: int | None = None
) -> list[dict[str, Any]]:
    """Get query history for the current session.

    Args:
        ticker: Optional filter by ticker symbol.
        limit: Optional limit on number of results.

    Returns:
        List of query records with full details.
    """
    agent = get_agent()
    return agent.get_recent_queries(ticker=ticker, limit=limit)


@app.post("/reset", response_model=ResetResponse)
def reset_endpoint() -> ResetResponse:
    """Reset the orchestrator session.

    Clears all memory and resets sub-agent state.
    """
    global _agent
    if _agent is not None:
        _agent.reset()
    return ResetResponse(status="ok", message="Session reset")
