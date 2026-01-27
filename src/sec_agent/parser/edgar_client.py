"""SEC EDGAR API client for fetching filings."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

_DEFAULT_USER_AGENT = "SecAgent admin@example.com"
_BASE_URL = "https://www.sec.gov"
_MIN_INTERVAL = 1.0 / 10  # SEC requires ≤10 req/sec


@dataclass
class FilingMetadata:
    """Metadata for a single SEC filing."""

    accession_number: str
    filing_type: str
    filing_date: str
    primary_document: str
    cik: str


class EdgarClient:
    """Client for SEC EDGAR public API with rate limiting."""

    def __init__(self, user_agent: str | None = None) -> None:
        self._user_agent = user_agent or os.environ.get(
            "SEC_EDGAR_USER_AGENT", _DEFAULT_USER_AGENT
        )
        self._last_request_time: float = 0.0
        self._client = httpx.Client(
            headers={"User-Agent": self._user_agent},
            follow_redirects=True,
            timeout=30.0,
        )

    # -- rate limiting ---------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    def _get(self, url: str) -> httpx.Response:
        self._throttle()
        resp = self._client.get(url)
        resp.raise_for_status()
        return resp

    # -- public API ------------------------------------------------------

    def get_cik(self, ticker: str) -> str:
        """Resolve a stock ticker to a zero-padded 10-digit CIK string."""
        tickers_url = f"{_BASE_URL}/files/company_tickers.json"
        resp = self._get(tickers_url)
        data: dict[str, dict[str, object]] = resp.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if str(entry.get("ticker", "")).upper() == ticker_upper:
                cik_int = int(str(entry["cik_str"]))
                return str(cik_int).zfill(10)
        raise ValueError(f"Ticker '{ticker}' not found in SEC company tickers")

    def list_filings(
        self, cik: str, filing_type: str, count: int = 5
    ) -> list[FilingMetadata]:
        """List recent filings for a CIK and filing type."""
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = self._get(submissions_url)
        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            return []

        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])

        results: list[FilingMetadata] = []
        normalized_type = filing_type.upper().replace("-", "")
        for i, form in enumerate(forms):
            form_normalized = form.upper().replace("-", "")
            if form_normalized == normalized_type:
                results.append(
                    FilingMetadata(
                        accession_number=accessions[i],
                        filing_type=form,
                        filing_date=dates[i],
                        primary_document=primary_docs[i],
                        cik=cik,
                    )
                )
                if len(results) >= count:
                    break
        return results

    def download_filing(self, filing: FilingMetadata) -> str:
        """Download the full filing HTML given its metadata."""
        accession_no_dashes = filing.accession_number.replace("-", "")
        cik_stripped = filing.cik.lstrip("0") or "0"
        url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_stripped}/{accession_no_dashes}/{filing.primary_document}"
        )
        resp = self._get(url)
        return resp.text

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
