"""FastAPI REST API for the SEC filings agent."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sec_agent.agent import SECFilingsAgent
from sec_agent.tools.fetch_filing import fetch_and_parse_filing
from sec_agent.tools.query_section import list_available_filings

app = FastAPI(title="SEC Filings Agent API", version="0.1.0")

# Shared agent instance for session-based queries.
_agent: SECFilingsAgent | None = None


def _get_agent() -> SECFilingsAgent:
    global _agent
    if _agent is None:
        _agent = SECFilingsAgent()
    return _agent


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    filing_type: str = Field(..., description="Type of SEC filing (10-K, 10-Q, 8-K)")
    question: str = Field(..., description="Natural language question about the filing")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    session_id: str


class FetchRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    filing_type: str = Field(..., description="Type of SEC filing (10-K, 10-Q, 8-K)")


class FetchResponse(BaseModel):
    status: str
    sections_found: list[str]
    chunk_count: int


class FilingsListResponse(BaseModel):
    cached_filings: list[str]


class HealthResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Ask a natural language question about a company's SEC filing."""
    agent = _get_agent()
    prompt = (
        f"Regarding {request.ticker}'s {request.filing_type} filing: "
        f"{request.question}"
    )
    answer = agent.query(prompt)
    return QueryResponse(
        answer=answer,
        sources=[],
        session_id=agent.memory.session_id,
    )


@app.post("/fetch", response_model=FetchResponse)
def fetch(request: FetchRequest) -> FetchResponse:
    """Fetch and cache a SEC filing from EDGAR."""
    result = fetch_and_parse_filing(
        ticker=request.ticker,
        filing_type=request.filing_type,
    )
    if result.get("status") != "ok":
        raise HTTPException(status_code=404, detail=result.get("message", "Fetch failed"))
    return FetchResponse(
        status=result["status"],
        sections_found=result.get("sections_found", []),
        chunk_count=result.get("chunk_count", 0),
    )


@app.get("/filings/{ticker}", response_model=FilingsListResponse)
def get_filings(ticker: str) -> FilingsListResponse:
    """List cached filings for a ticker."""
    result = list_available_filings(ticker=ticker)
    return FilingsListResponse(cached_filings=result.get("cached_filings", []))
