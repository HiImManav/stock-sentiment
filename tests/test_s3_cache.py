"""Tests for the S3 cache layer."""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from sec_agent.parser.chunker import Chunk
from sec_agent.retrieval.s3_cache import FilingChunks, S3Cache

BUCKET = "test-sec-filings"


def _make_filing_chunks() -> FilingChunks:
    return FilingChunks(
        ticker="AAPL",
        cik="0000320193",
        filing_type="10-K",
        accession_number="0000320193-23-000106",
        filing_date="2023-11-03",
        sections=["1A", "7"],
        chunks=[
            Chunk(
                text="Risk factor text here.",
                metadata={
                    "section_name": "Risk Factors",
                    "item_number": "1A",
                    "filing_type": "10-K",
                    "ticker": "AAPL",
                    "accession_number": "0000320193-23-000106",
                    "chunk_index": 0,
                },
                token_count=5,
            ),
            Chunk(
                text="MD&A text here.",
                metadata={
                    "section_name": "MD&A",
                    "item_number": "7",
                    "filing_type": "10-K",
                    "ticker": "AAPL",
                    "accession_number": "0000320193-23-000106",
                    "chunk_index": 0,
                },
                token_count=4,
            ),
        ],
    )


@pytest.fixture()
def s3_cache() -> S3Cache:
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        yield S3Cache(bucket=BUCKET, s3_client=s3)


class TestFilingChunks:
    def test_roundtrip_serialization(self) -> None:
        fc = _make_filing_chunks()
        data = fc.to_dict()
        restored = FilingChunks.from_dict(data)
        assert restored.ticker == fc.ticker
        assert restored.cik == fc.cik
        assert restored.filing_type == fc.filing_type
        assert restored.accession_number == fc.accession_number
        assert restored.filing_date == fc.filing_date
        assert restored.sections == fc.sections
        assert len(restored.chunks) == 2
        assert restored.chunks[0].text == "Risk factor text here."
        assert restored.chunks[0].token_count == 5
        assert restored.chunks[1].metadata["item_number"] == "7"

    def test_to_dict_structure(self) -> None:
        fc = _make_filing_chunks()
        d = fc.to_dict()
        assert d["ticker"] == "AAPL"
        assert isinstance(d["chunks"], list)
        assert d["chunks"][0]["text"] == "Risk factor text here."


class TestS3Cache:
    def test_cache_and_retrieve(self, s3_cache: S3Cache) -> None:
        fc = _make_filing_chunks()
        s3_cache.cache_filing(fc)
        result = s3_cache.get_cached_filing("AAPL", "10-K", "0000320193-23-000106")
        assert result is not None
        assert result.ticker == "AAPL"
        assert len(result.chunks) == 2
        assert result.chunks[0].text == "Risk factor text here."

    def test_get_cached_filing_not_found(self, s3_cache: S3Cache) -> None:
        result = s3_cache.get_cached_filing("AAPL", "10-K", "nonexistent")
        assert result is None

    def test_list_cached_filings_empty(self, s3_cache: S3Cache) -> None:
        keys = s3_cache.list_cached_filings("AAPL")
        assert keys == []

    def test_list_cached_filings(self, s3_cache: S3Cache) -> None:
        fc = _make_filing_chunks()
        s3_cache.cache_filing(fc)
        keys = s3_cache.list_cached_filings("AAPL")
        assert len(keys) == 1
        assert "AAPL" in keys[0]
        assert "chunks.json" in keys[0]

    def test_list_cached_filings_multiple(self, s3_cache: S3Cache) -> None:
        fc1 = _make_filing_chunks()
        fc2 = _make_filing_chunks()
        fc2.accession_number = "0000320193-23-000999"
        s3_cache.cache_filing(fc1)
        s3_cache.cache_filing(fc2)
        keys = s3_cache.list_cached_filings("AAPL")
        assert len(keys) == 2

    def test_overwrite_cached_filing(self, s3_cache: S3Cache) -> None:
        fc = _make_filing_chunks()
        s3_cache.cache_filing(fc)
        fc.sections = ["1A"]
        fc.chunks = [fc.chunks[0]]
        s3_cache.cache_filing(fc)
        result = s3_cache.get_cached_filing("AAPL", "10-K", "0000320193-23-000106")
        assert result is not None
        assert len(result.chunks) == 1
        assert result.sections == ["1A"]

    def test_s3_key_format(self, s3_cache: S3Cache) -> None:
        key = s3_cache._key("AAPL", "10-K", "0000320193-23-000106")
        assert key == "filings/AAPL/10-K/0000320193-23-000106/chunks.json"
