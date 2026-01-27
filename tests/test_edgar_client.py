"""Tests for the SEC EDGAR client."""

from __future__ import annotations

import json

import httpx
import pytest

from sec_agent.parser.edgar_client import EdgarClient, FilingMetadata


# -- Fixtures & helpers --------------------------------------------------

SAMPLE_TICKERS_JSON = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}

SAMPLE_SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "8-K", "10-K"],
            "accessionNumber": [
                "0000320193-23-000106",
                "0000320193-23-000077",
                "0000320193-23-000090",
                "0000320193-22-000108",
            ],
            "filingDate": [
                "2023-11-03",
                "2023-08-04",
                "2023-09-15",
                "2022-10-28",
            ],
            "primaryDocument": [
                "aapl-20230930.htm",
                "aapl-20230701.htm",
                "aapl-8k.htm",
                "aapl-20220924.htm",
            ],
        }
    }
}


def _make_transport(routes: dict[str, httpx.Response]) -> httpx.MockTransport:
    """Create a mock transport that maps URL substrings to responses."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for pattern, response in routes.items():
            if pattern in url:
                return response
        return httpx.Response(404, text="Not found")

    return httpx.MockTransport(handler)


# -- Tests ---------------------------------------------------------------


class TestGetCik:
    def test_resolves_known_ticker(self) -> None:
        transport = _make_transport(
            {
                "company_tickers.json": httpx.Response(
                    200, json=SAMPLE_TICKERS_JSON
                )
            }
        )
        client = EdgarClient(user_agent="Test test@test.com")
        client._client = httpx.Client(transport=transport)

        cik = client.get_cik("AAPL")
        assert cik == "0000320193"

    def test_case_insensitive(self) -> None:
        transport = _make_transport(
            {
                "company_tickers.json": httpx.Response(
                    200, json=SAMPLE_TICKERS_JSON
                )
            }
        )
        client = EdgarClient(user_agent="Test test@test.com")
        client._client = httpx.Client(transport=transport)

        assert client.get_cik("aapl") == "0000320193"
        assert client.get_cik("Msft") == "0000789019"

    def test_unknown_ticker_raises(self) -> None:
        transport = _make_transport(
            {
                "company_tickers.json": httpx.Response(
                    200, json=SAMPLE_TICKERS_JSON
                )
            }
        )
        client = EdgarClient(user_agent="Test test@test.com")
        client._client = httpx.Client(transport=transport)

        with pytest.raises(ValueError, match="not found"):
            client.get_cik("ZZZZ")


class TestListFilings:
    def test_filters_by_filing_type(self) -> None:
        transport = _make_transport(
            {
                "submissions/CIK": httpx.Response(
                    200, json=SAMPLE_SUBMISSIONS
                )
            }
        )
        client = EdgarClient(user_agent="Test test@test.com")
        client._client = httpx.Client(transport=transport)

        results = client.list_filings("0000320193", "10-K", count=5)
        assert len(results) == 2
        assert all(r.filing_type == "10-K" for r in results)

    def test_respects_count(self) -> None:
        transport = _make_transport(
            {
                "submissions/CIK": httpx.Response(
                    200, json=SAMPLE_SUBMISSIONS
                )
            }
        )
        client = EdgarClient(user_agent="Test test@test.com")
        client._client = httpx.Client(transport=transport)

        results = client.list_filings("0000320193", "10-K", count=1)
        assert len(results) == 1

    def test_empty_when_no_match(self) -> None:
        transport = _make_transport(
            {
                "submissions/CIK": httpx.Response(
                    200, json=SAMPLE_SUBMISSIONS
                )
            }
        )
        client = EdgarClient(user_agent="Test test@test.com")
        client._client = httpx.Client(transport=transport)

        results = client.list_filings("0000320193", "20-F")
        assert results == []


class TestDownloadFiling:
    def test_constructs_correct_url_and_returns_html(self) -> None:
        filing = FilingMetadata(
            accession_number="0000320193-23-000106",
            filing_type="10-K",
            filing_date="2023-11-03",
            primary_document="aapl-20230930.htm",
            cik="0000320193",
        )
        html = "<html><body>Filing content</body></html>"
        transport = _make_transport(
            {"Archives/edgar/data/": httpx.Response(200, text=html)}
        )
        client = EdgarClient(user_agent="Test test@test.com")
        client._client = httpx.Client(transport=transport)

        result = client.download_filing(filing)
        assert result == html


class TestContextManager:
    def test_context_manager(self) -> None:
        with EdgarClient(user_agent="Test test@test.com") as client:
            assert client._client is not None
