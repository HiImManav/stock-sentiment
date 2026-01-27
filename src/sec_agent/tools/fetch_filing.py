"""Tool: fetch and parse a SEC filing, cache chunks in S3."""

from __future__ import annotations

from sec_agent.parser.chunker import chunk_filing
from sec_agent.parser.edgar_client import EdgarClient
from sec_agent.parser.filing_parser import parse_8k, parse_filing
from sec_agent.retrieval.s3_cache import FilingChunks, S3Cache


def fetch_and_parse_filing(
    ticker: str,
    filing_type: str,
    filing_index: int = 0,
    *,
    edgar_client: EdgarClient | None = None,
    s3_cache: S3Cache | None = None,
) -> dict:
    """Fetch a SEC filing from EDGAR, parse it into sections, chunk, and cache.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL).
        filing_type: One of "10-K", "10-Q", "8-K".
        filing_index: 0 = most recent, 1 = second most recent, etc.
        edgar_client: Optional pre-configured EdgarClient.
        s3_cache: Optional pre-configured S3Cache.

    Returns:
        Dict with status, sections found, chunk count, and filing metadata.
    """
    client = edgar_client or EdgarClient()
    cache = s3_cache or S3Cache()
    close_client = edgar_client is None

    try:
        cik = client.get_cik(ticker)
        filings = client.list_filings(cik, filing_type, count=filing_index + 1)

        if not filings or filing_index >= len(filings):
            return {
                "status": "error",
                "message": f"No {filing_type} filing found at index {filing_index} for {ticker}",
            }

        filing_meta = filings[filing_index]

        # Check cache first
        cached = cache.get_cached_filing(
            ticker.upper(), filing_type, filing_meta.accession_number
        )
        if cached is not None:
            return {
                "status": "ok",
                "source": "cache",
                "ticker": ticker.upper(),
                "filing_type": filing_type,
                "accession_number": filing_meta.accession_number,
                "filing_date": filing_meta.filing_date,
                "sections_found": cached.sections,
                "chunk_count": len(cached.chunks),
            }

        # Download and parse
        html = client.download_filing(filing_meta)

        if filing_type.upper() == "8-K":
            sections = parse_8k(html)
        else:
            sections = parse_filing(html, filing_type)

        chunks = chunk_filing(
            sections,
            filing_type=filing_type,
            ticker=ticker.upper(),
            accession_number=filing_meta.accession_number,
        )

        section_names = list({s.item_number for s in sections})

        filing_chunks = FilingChunks(
            ticker=ticker.upper(),
            cik=cik,
            filing_type=filing_type,
            accession_number=filing_meta.accession_number,
            filing_date=filing_meta.filing_date,
            sections=section_names,
            chunks=chunks,
        )

        cache.cache_filing(filing_chunks)

        return {
            "status": "ok",
            "source": "fetched",
            "ticker": ticker.upper(),
            "filing_type": filing_type,
            "accession_number": filing_meta.accession_number,
            "filing_date": filing_meta.filing_date,
            "sections_found": section_names,
            "chunk_count": len(chunks),
        }
    finally:
        if close_client:
            client.close()
