"""Entity resolver for mapping tickers to company names using SEC data."""

from __future__ import annotations

import os
import time
from functools import lru_cache

import httpx

_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_DEFAULT_USER_AGENT = "NewsAgent admin@example.com"
_MIN_INTERVAL = 0.1  # SEC rate limit


class EntityResolver:
    """Resolves stock tickers to official company names using SEC data."""

    def __init__(self, user_agent: str | None = None) -> None:
        self._user_agent = user_agent or os.environ.get(
            "SEC_EDGAR_USER_AGENT", _DEFAULT_USER_AGENT
        )
        self._last_request_time: float = 0.0
        self._client = httpx.Client(
            headers={"User-Agent": self._user_agent},
            timeout=30.0,
        )
        self._ticker_cache: dict[str, dict] | None = None

    def _throttle(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    def _load_ticker_data(self) -> dict[str, dict]:
        """Load and cache the SEC company tickers data."""
        if self._ticker_cache is not None:
            return self._ticker_cache

        self._throttle()
        resp = self._client.get(_SEC_TICKERS_URL)
        resp.raise_for_status()
        data = resp.json()

        # Build a lookup by ticker
        self._ticker_cache = {}
        for entry in data.values():
            ticker = str(entry.get("ticker", "")).upper()
            if ticker:
                self._ticker_cache[ticker] = {
                    "cik": str(entry.get("cik_str", "")).zfill(10),
                    "title": str(entry.get("title", "")),
                }

        return self._ticker_cache

    @lru_cache(maxsize=1000)
    def resolve(self, ticker: str) -> dict:
        """Resolve a ticker to company information.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')

        Returns:
            Dict with 'cik', 'title' (company name), and 'ticker'

        Raises:
            ValueError: If ticker not found
        """
        ticker_data = self._load_ticker_data()
        ticker_upper = ticker.upper()

        if ticker_upper not in ticker_data:
            raise ValueError(f"Ticker '{ticker}' not found in SEC company data")

        info = ticker_data[ticker_upper]
        return {
            "ticker": ticker_upper,
            "cik": info["cik"],
            "company_name": info["title"],
        }

    def get_company_name(self, ticker: str) -> str:
        """Get the official company name for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Official company name from SEC filings
        """
        return self.resolve(ticker)["company_name"]

    def get_cik(self, ticker: str) -> str:
        """Get the CIK (Central Index Key) for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            10-digit zero-padded CIK string
        """
        return self.resolve(ticker)["cik"]

    def build_search_query(self, ticker: str) -> str:
        """Build an optimized search query for a company.

        Creates a query like: "Apple Inc" OR AAPL

        Args:
            ticker: Stock ticker symbol

        Returns:
            Search query string optimized for news search
        """
        info = self.resolve(ticker)
        company_name = info["company_name"]

        # Clean up company name for search
        # Remove common suffixes that might cause false negatives
        clean_name = company_name
        for suffix in [" Inc", " Corp", " Ltd", " LLC", " PLC", " Co"]:
            if clean_name.endswith(suffix):
                clean_name = clean_name[: -len(suffix)]
                break

        # Build query: "Company Name" OR TICKER
        return f'"{clean_name}" OR {ticker.upper()}'

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> EntityResolver:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
