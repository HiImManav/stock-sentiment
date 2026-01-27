"""Tool: semantic search over cached filing chunks and return relevant context."""

from __future__ import annotations

import numpy as np

from sec_agent.parser.chunker import Chunk
from sec_agent.retrieval.embeddings import EmbeddingModel
from sec_agent.retrieval.s3_cache import S3Cache
from sec_agent.retrieval.vector_store import build_index, search


def query_filing(
    ticker: str,
    filing_type: str,
    question: str,
    section_filter: str | None = None,
    *,
    s3_cache: S3Cache | None = None,
    embedding_model: EmbeddingModel | None = None,
    top_k: int = 10,
) -> dict:
    """Search cached filing chunks for relevant context to answer a question.

    Args:
        ticker: Stock ticker symbol.
        filing_type: One of "10-K", "10-Q", "8-K".
        question: Natural language question.
        section_filter: Optional item number to restrict search (e.g., "1A").
        s3_cache: Optional pre-configured S3Cache.
        embedding_model: Optional pre-configured EmbeddingModel.
        top_k: Number of top chunks to retrieve.

    Returns:
        Dict with status, retrieved chunks, and metadata.
    """
    cache = s3_cache or S3Cache()
    embedder = embedding_model or EmbeddingModel()

    # Find cached filings for this ticker
    cached_keys = cache.list_cached_filings(ticker.upper())
    filing_type_upper = filing_type.upper()

    # Filter to the requested filing type
    matching_keys = [k for k in cached_keys if f"/{filing_type_upper}/" in k.upper()]

    if not matching_keys:
        return {
            "status": "error",
            "message": (
                f"No cached {filing_type} filings found for {ticker}. "
                "Call fetch_and_parse_filing first."
            ),
        }

    # Load chunks from the most recent cached filing (last key alphabetically
    # tends to be most recent by accession number)
    matching_keys.sort()
    latest_key = matching_keys[-1]

    # Extract accession number from key: filings/{ticker}/{type}/{accession}/chunks.json
    parts = latest_key.split("/")
    accession_number = parts[3] if len(parts) >= 5 else ""

    filing_data = cache.get_cached_filing(ticker.upper(), filing_type, accession_number)
    if filing_data is None:
        return {
            "status": "error",
            "message": f"Failed to load cached filing for {ticker} {filing_type}",
        }

    chunks = filing_data.chunks
    if not chunks:
        return {
            "status": "error",
            "message": f"No chunks found in cached filing for {ticker} {filing_type}",
        }

    # Build embeddings and index
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(texts)
    query_embedding = embedder.embed_query(question)

    index = build_index(embeddings)
    results = search(
        index, query_embedding, chunks, top_k=top_k, section_filter=section_filter
    )

    return {
        "status": "ok",
        "ticker": ticker.upper(),
        "filing_type": filing_type,
        "accession_number": accession_number,
        "filing_date": filing_data.filing_date,
        "question": question,
        "section_filter": section_filter,
        "chunks_retrieved": len(results),
        "context": [
            {
                "text": c.text,
                "section_name": c.metadata.get("section_name", ""),
                "item_number": c.metadata.get("item_number", ""),
                "chunk_index": c.metadata.get("chunk_index", 0),
            }
            for c in results
        ],
    }


def list_available_filings(
    ticker: str,
    *,
    s3_cache: S3Cache | None = None,
) -> dict:
    """List cached filings for a ticker.

    Returns:
        Dict with status and list of cached filing keys.
    """
    cache = s3_cache or S3Cache()
    keys = cache.list_cached_filings(ticker.upper())
    return {
        "status": "ok",
        "ticker": ticker.upper(),
        "cached_filings": keys,
        "count": len(keys),
    }
