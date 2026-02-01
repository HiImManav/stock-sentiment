"""Tests for the FastAPI REST API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sec_agent.api.server import app, _get_agent

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


# ---------------------------------------------------------------------------
# POST /fetch
# ---------------------------------------------------------------------------


@patch("sec_agent.api.server.fetch_and_parse_filing")
def test_fetch_success(mock_fetch):
    mock_fetch.return_value = {
        "status": "ok",
        "sections_found": ["1A", "7"],
        "chunk_count": 25,
    }
    resp = client.post("/fetch", json={"ticker": "AAPL", "filing_type": "10-K"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["sections_found"] == ["1A", "7"]
    assert data["chunk_count"] == 25
    mock_fetch.assert_called_once_with(ticker="AAPL", filing_type="10-K")


@patch("sec_agent.api.server.fetch_and_parse_filing")
def test_fetch_not_found(mock_fetch):
    mock_fetch.return_value = {
        "status": "error",
        "message": "No 10-K filing found at index 0 for XYZ",
    }
    resp = client.post("/fetch", json={"ticker": "XYZ", "filing_type": "10-K"})
    assert resp.status_code == 404
    assert "No 10-K filing found" in resp.json()["detail"]


def test_fetch_missing_fields():
    resp = client.post("/fetch", json={"ticker": "AAPL"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /filings/{ticker}
# ---------------------------------------------------------------------------


@patch("sec_agent.api.server.list_available_filings")
def test_list_filings(mock_list):
    mock_list.return_value = {
        "status": "ok",
        "cached_filings": ["filings/AAPL/10-K/001/chunks.json"],
        "count": 1,
    }
    resp = client.get("/filings/AAPL")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["cached_filings"]) == 1


@patch("sec_agent.api.server.list_available_filings")
def test_list_filings_empty(mock_list):
    mock_list.return_value = {"status": "ok", "cached_filings": [], "count": 0}
    resp = client.get("/filings/NOPE")
    assert resp.status_code == 200
    assert resp.json()["cached_filings"] == []


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


@patch("sec_agent.api.server._get_agent")
def test_query_success(mock_get_agent):
    mock_agent = MagicMock()
    mock_agent.query.return_value = "Apple's key risks include supply chain issues."
    mock_agent.memory.session_id = "test-session-123"
    mock_get_agent.return_value = mock_agent

    resp = client.post(
        "/query",
        json={
            "ticker": "AAPL",
            "filing_type": "10-K",
            "question": "What are the risk factors?",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "supply chain" in data["answer"]
    assert data["session_id"] == "test-session-123"
    assert isinstance(data["sources"], list)


def test_query_missing_fields():
    resp = client.post("/query", json={"ticker": "AAPL"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Shared agent singleton
# ---------------------------------------------------------------------------


@patch("sec_agent.api.server.SECFilingsAgent")
def test_get_agent_creates_singleton(mock_cls):
    import sec_agent.api.server as mod

    mod._agent = None
    mock_cls.return_value = MagicMock()
    a1 = mod._get_agent()
    a2 = mod._get_agent()
    assert a1 is a2
    mock_cls.assert_called_once()
    mod._agent = None  # cleanup
