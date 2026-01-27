"""Tests for the SEC filings agent, tools, and orchestration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from moto import mock_aws

import boto3

from sec_agent.agent import (
    SYSTEM_PROMPT,
    TOOL_CONFIG,
    SECFilingsAgent,
    _execute_tool,
)
from sec_agent.parser.chunker import Chunk
from sec_agent.parser.edgar_client import EdgarClient, FilingMetadata
from sec_agent.parser.filing_parser import Section
from sec_agent.retrieval.s3_cache import FilingChunks, S3Cache
from sec_agent.tools.fetch_filing import fetch_and_parse_filing
from sec_agent.tools.query_section import list_available_filings, query_filing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_filing_meta(
    accession: str = "0001-23-000001",
    filing_type: str = "10-K",
    date: str = "2025-01-01",
) -> FilingMetadata:
    return FilingMetadata(
        accession_number=accession,
        filing_type=filing_type,
        filing_date=date,
        primary_document="doc.htm",
        cik="0000012345",
    )


SAMPLE_HTML = """<html><body>
<p>Item 1A. Risk Factors</p>
<p>There are significant risks involved in our operations.
The company faces regulatory, market, and operational risks that could
adversely affect business results. Competition is intense across all segments.
Macroeconomic conditions may impact demand for our products and services.</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>Revenue increased 10% year over year driven by strong product demand.
Operating expenses grew 5% primarily due to increased R&D investment.
The company generated positive free cash flow of $2 billion.</p>
</body></html>"""


# ---------------------------------------------------------------------------
# fetch_and_parse_filing tool tests
# ---------------------------------------------------------------------------


class TestFetchAndParseFiling:
    @mock_aws
    def test_fetch_parse_and_cache(self) -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        cache = S3Cache(bucket="test-bucket", s3_client=s3)

        edgar = MagicMock(spec=EdgarClient)
        meta = _make_filing_meta()
        edgar.get_cik.return_value = "0000012345"
        edgar.list_filings.return_value = [meta]
        edgar.download_filing.return_value = SAMPLE_HTML

        result = fetch_and_parse_filing(
            "AAPL", "10-K", edgar_client=edgar, s3_cache=cache
        )

        assert result["status"] == "ok"
        assert result["source"] == "fetched"
        assert result["ticker"] == "AAPL"
        assert result["chunk_count"] > 0
        assert len(result["sections_found"]) > 0

    @mock_aws
    def test_returns_from_cache(self) -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        cache = S3Cache(bucket="test-bucket", s3_client=s3)

        # Pre-populate cache
        chunks = [Chunk(text="test", metadata={"section_name": "Risk", "item_number": "1A"}, token_count=1)]
        fc = FilingChunks(
            ticker="AAPL", cik="0000012345", filing_type="10-K",
            accession_number="0001-23-000001", filing_date="2025-01-01",
            sections=["1A"], chunks=chunks,
        )
        cache.cache_filing(fc)

        edgar = MagicMock(spec=EdgarClient)
        edgar.get_cik.return_value = "0000012345"
        edgar.list_filings.return_value = [_make_filing_meta()]

        result = fetch_and_parse_filing(
            "AAPL", "10-K", edgar_client=edgar, s3_cache=cache
        )

        assert result["status"] == "ok"
        assert result["source"] == "cache"
        edgar.download_filing.assert_not_called()

    def test_filing_not_found(self) -> None:
        edgar = MagicMock(spec=EdgarClient)
        edgar.get_cik.return_value = "0000012345"
        edgar.list_filings.return_value = []

        result = fetch_and_parse_filing(
            "AAPL", "10-K", edgar_client=edgar, s3_cache=MagicMock(spec=S3Cache)
        )

        assert result["status"] == "error"

    @mock_aws
    def test_8k_parsing(self) -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        cache = S3Cache(bucket="test-bucket", s3_client=s3)

        html_8k = """<html><body>
        <p>Item 2.02 Results of Operations</p>
        <p>The company reported quarterly earnings of $1.50 per share, exceeding
        analyst expectations. Revenue reached $50 billion for the quarter.</p>
        </body></html>"""

        edgar = MagicMock(spec=EdgarClient)
        meta = _make_filing_meta(filing_type="8-K")
        edgar.get_cik.return_value = "0000012345"
        edgar.list_filings.return_value = [meta]
        edgar.download_filing.return_value = html_8k

        result = fetch_and_parse_filing(
            "AAPL", "8-K", edgar_client=edgar, s3_cache=cache
        )

        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# query_filing tool tests
# ---------------------------------------------------------------------------


class TestQueryFiling:
    @mock_aws
    def test_query_returns_chunks(self) -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        cache = S3Cache(bucket="test-bucket", s3_client=s3)

        chunks = [
            Chunk(
                text="Risk factor content about supply chain",
                metadata={"section_name": "Risk Factors", "item_number": "1A", "chunk_index": 0},
                token_count=10,
            ),
            Chunk(
                text="MD&A content about revenue growth",
                metadata={"section_name": "MD&A", "item_number": "7", "chunk_index": 0},
                token_count=10,
            ),
        ]
        fc = FilingChunks(
            ticker="AAPL", cik="0000012345", filing_type="10-K",
            accession_number="0001-23-000001", filing_date="2025-01-01",
            sections=["1A", "7"], chunks=chunks,
        )
        cache.cache_filing(fc)

        embedder = MagicMock()
        embedder.embed_texts.return_value = np.random.rand(2, 1024).astype(np.float32)
        embedder.embed_query.return_value = np.random.rand(1, 1024).astype(np.float32)

        result = query_filing(
            "AAPL", "10-K", "What are the risks?",
            s3_cache=cache, embedding_model=embedder,
        )

        assert result["status"] == "ok"
        assert result["chunks_retrieved"] > 0
        assert len(result["context"]) > 0

    def test_query_no_cached_filing(self) -> None:
        cache = MagicMock(spec=S3Cache)
        cache.list_cached_filings.return_value = []

        result = query_filing(
            "AAPL", "10-K", "What are the risks?",
            s3_cache=cache, embedding_model=MagicMock(),
        )

        assert result["status"] == "error"


class TestListAvailableFilings:
    def test_list_empty(self) -> None:
        cache = MagicMock(spec=S3Cache)
        cache.list_cached_filings.return_value = []

        result = list_available_filings("AAPL", s3_cache=cache)

        assert result["status"] == "ok"
        assert result["count"] == 0

    def test_list_with_filings(self) -> None:
        cache = MagicMock(spec=S3Cache)
        cache.list_cached_filings.return_value = [
            "filings/AAPL/10-K/0001-23-000001/chunks.json"
        ]

        result = list_available_filings("AAPL", s3_cache=cache)

        assert result["status"] == "ok"
        assert result["count"] == 1


# ---------------------------------------------------------------------------
# _execute_tool tests
# ---------------------------------------------------------------------------


class TestExecuteTool:
    def test_unknown_tool(self) -> None:
        result_str = _execute_tool("nonexistent_tool", {})
        result = json.loads(result_str)
        assert result["status"] == "error"
        assert "Unknown tool" in result["message"]

    @patch("sec_agent.agent.fetch_and_parse_filing")
    def test_dispatches_fetch(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {"status": "ok"}
        result_str = _execute_tool(
            "fetch_and_parse_filing", {"ticker": "AAPL", "filing_type": "10-K"}
        )
        result = json.loads(result_str)
        assert result["status"] == "ok"
        mock_fetch.assert_called_once_with(ticker="AAPL", filing_type="10-K")


# ---------------------------------------------------------------------------
# SECFilingsAgent tests
# ---------------------------------------------------------------------------


class TestSECFilingsAgent:
    def test_single_turn_text_response(self) -> None:
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello, I can help with SEC filings."}],
                }
            },
            "stopReason": "end_turn",
        }

        agent = SECFilingsAgent(bedrock_client=mock_client)
        response = agent.query("Hello")

        assert response == "Hello, I can help with SEC filings."
        assert len(agent._messages) == 2  # user + assistant

    def test_tool_use_loop(self) -> None:
        mock_client = MagicMock()

        # First call: model requests tool use
        tool_response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "Let me fetch that filing."},
                        {
                            "toolUse": {
                                "toolUseId": "tool-1",
                                "name": "list_available_filings",
                                "input": {"ticker": "AAPL"},
                            }
                        },
                    ],
                }
            },
            "stopReason": "tool_use",
        }

        # Second call: model returns final answer
        final_response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "No cached filings found for AAPL."}],
                }
            },
            "stopReason": "end_turn",
        }

        mock_client.converse.side_effect = [tool_response, final_response]

        # Mock the tool handler to avoid real S3 calls
        with patch("sec_agent.agent._TOOL_HANDLERS") as mock_handlers:
            mock_handlers.get.return_value = lambda **kw: {
                "status": "ok", "ticker": "AAPL", "cached_filings": [], "count": 0
            }

            agent = SECFilingsAgent(bedrock_client=mock_client)
            response = agent.query("What filings are cached for AAPL?")

        assert response == "No cached filings found for AAPL."
        assert mock_client.converse.call_count == 2

    def test_max_turns_exceeded(self) -> None:
        mock_client = MagicMock()
        # Always return tool use to exhaust turns
        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tool-1",
                                "name": "list_available_filings",
                                "input": {"ticker": "AAPL"},
                            }
                        },
                    ],
                }
            },
            "stopReason": "tool_use",
        }

        with patch("sec_agent.agent._TOOL_HANDLERS") as mock_handlers:
            mock_handlers.get.return_value = lambda **kw: {
                "status": "ok", "ticker": "AAPL", "cached_filings": [], "count": 0
            }

            agent = SECFilingsAgent(bedrock_client=mock_client, max_turns=2)
            response = agent.query("Loop forever")

        assert "unable to complete" in response.lower()

    def test_reset_clears_messages(self) -> None:
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hi"}],
                }
            },
            "stopReason": "end_turn",
        }

        agent = SECFilingsAgent(bedrock_client=mock_client)
        agent.query("Hello")
        assert len(agent._messages) > 0

        agent.reset()
        assert len(agent._messages) == 0

    def test_system_prompt_is_set(self) -> None:
        assert "SEC filings analyst" in SYSTEM_PROMPT

    def test_tool_config_has_all_tools(self) -> None:
        tool_names = {t["toolSpec"]["name"] for t in TOOL_CONFIG["tools"]}
        assert tool_names == {
            "fetch_and_parse_filing",
            "query_filing",
            "list_available_filings",
            "get_memory",
            "save_memory",
        }

    def test_converse_called_with_system_prompt(self) -> None:
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hi"}],
                }
            },
            "stopReason": "end_turn",
        }

        agent = SECFilingsAgent(bedrock_client=mock_client)
        agent.query("Hello")

        call_kwargs = mock_client.converse.call_args
        assert call_kwargs.kwargs["system"] == [{"text": SYSTEM_PROMPT}]
        assert call_kwargs.kwargs["toolConfig"] == TOOL_CONFIG
