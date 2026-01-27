"""S3 storage layer for caching parsed and chunked SEC filings."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError

from sec_agent.parser.chunker import Chunk


@dataclass
class FilingChunks:
    """A complete set of chunks for a single SEC filing."""

    ticker: str
    cik: str
    filing_type: str
    accession_number: str
    filing_date: str
    sections: list[str]
    chunks: list[Chunk]

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "ticker": self.ticker,
            "cik": self.cik,
            "filing_type": self.filing_type,
            "accession_number": self.accession_number,
            "filing_date": self.filing_date,
            "sections": self.sections,
            "chunks": [
                {
                    "text": c.text,
                    "metadata": c.metadata,
                    "token_count": c.token_count,
                }
                for c in self.chunks
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> FilingChunks:
        """Deserialize from a dict."""
        chunks = [
            Chunk(
                text=c["text"],
                metadata=c["metadata"],
                token_count=c["token_count"],
            )
            for c in data["chunks"]
        ]
        return cls(
            ticker=data["ticker"],
            cik=data["cik"],
            filing_type=data["filing_type"],
            accession_number=data["accession_number"],
            filing_date=data["filing_date"],
            sections=data["sections"],
            chunks=chunks,
        )


class S3Cache:
    """Cache for parsed SEC filing chunks stored in S3."""

    def __init__(self, bucket: str | None = None, s3_client: object | None = None) -> None:
        self._bucket = bucket or os.environ.get("SEC_FILINGS_BUCKET", "sec-filings-cache")
        self._s3 = s3_client or boto3.client("s3")

    def _key(self, ticker: str, filing_type: str, accession_number: str) -> str:
        """Build the S3 object key for a filing."""
        return f"filings/{ticker}/{filing_type}/{accession_number}/chunks.json"

    def get_cached_filing(
        self, ticker: str, filing_type: str, accession_number: str
    ) -> FilingChunks | None:
        """Retrieve a cached filing from S3, or None if not found."""
        key = self._key(ticker, filing_type, accession_number)
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=key)
            data = json.loads(resp["Body"].read().decode("utf-8"))
            return FilingChunks.from_dict(data)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    def cache_filing(self, filing_chunks: FilingChunks) -> None:
        """Store a filing's chunks in S3."""
        key = self._key(
            filing_chunks.ticker,
            filing_chunks.filing_type,
            filing_chunks.accession_number,
        )
        body = json.dumps(filing_chunks.to_dict(), ensure_ascii=False)
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )

    def list_cached_filings(self, ticker: str) -> list[str]:
        """List cached filing accession numbers for a ticker.

        Returns a list of S3 keys under filings/{ticker}/.
        """
        prefix = f"filings/{ticker}/"
        keys: list[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys
