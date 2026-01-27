"""Tests for the embedding and vector store retrieval modules."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from sec_agent.parser.chunker import Chunk
from sec_agent.retrieval.embeddings import EmbeddingModel
from sec_agent.retrieval.vector_store import build_index, search


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_bedrock(dim: int = 1024):
    """Return a mock bedrock-runtime client that returns random embeddings."""
    client = MagicMock()

    def _invoke_model(**kwargs):
        body = MagicMock()
        vec = np.random.default_rng(42).random(dim).tolist()
        body.read.return_value = json.dumps({"embedding": vec}).encode()
        return {"body": body}

    client.invoke_model.side_effect = _invoke_model
    return client


def _make_chunks(n: int = 5) -> list[Chunk]:
    return [
        Chunk(
            text=f"Chunk {i} text content.",
            metadata={
                "section_name": "Risk Factors" if i % 2 == 0 else "MD&A",
                "item_number": "1A" if i % 2 == 0 else "7",
                "filing_type": "10-K",
                "ticker": "AAPL",
                "accession_number": "0000320193-23-000106",
                "chunk_index": i,
            },
            token_count=5,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# EmbeddingModel tests
# ---------------------------------------------------------------------------

class TestEmbeddingModel:
    def test_embed_texts(self) -> None:
        client = _make_mock_bedrock()
        model = EmbeddingModel(bedrock_client=client)
        result = model.embed_texts(["hello", "world"])
        assert result.shape == (2, 1024)
        assert result.dtype == np.float32
        assert client.invoke_model.call_count == 2

    def test_embed_texts_empty(self) -> None:
        client = _make_mock_bedrock()
        model = EmbeddingModel(bedrock_client=client)
        result = model.embed_texts([])
        assert result.shape == (0, 1024)
        assert client.invoke_model.call_count == 0

    def test_embed_query(self) -> None:
        client = _make_mock_bedrock()
        model = EmbeddingModel(bedrock_client=client)
        result = model.embed_query("What are the risks?")
        assert result.shape == (1, 1024)
        assert result.dtype == np.float32

    def test_model_id_default(self) -> None:
        client = _make_mock_bedrock()
        model = EmbeddingModel(bedrock_client=client)
        assert model._model_id == "amazon.titan-embed-text-v2:0"

    def test_model_id_custom(self) -> None:
        client = _make_mock_bedrock()
        model = EmbeddingModel(model_id="custom-model", bedrock_client=client)
        assert model._model_id == "custom-model"

    def test_invoke_passes_correct_body(self) -> None:
        client = _make_mock_bedrock()
        model = EmbeddingModel(bedrock_client=client)
        model.embed_query("test input")
        call_kwargs = client.invoke_model.call_args[1]
        body = json.loads(call_kwargs["body"])
        assert body == {"inputText": "test input"}
        assert call_kwargs["contentType"] == "application/json"


# ---------------------------------------------------------------------------
# Vector store tests
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_build_index_and_search(self) -> None:
        rng = np.random.default_rng(123)
        embeddings = rng.random((5, 1024)).astype(np.float32)
        chunks = _make_chunks(5)

        index = build_index(embeddings)
        assert index.ntotal == 5

        query = rng.random((1, 1024)).astype(np.float32)
        results = search(index, query, chunks, top_k=3)
        assert len(results) == 3
        # Results should be Chunk objects
        assert all(isinstance(c, Chunk) for c in results)

    def test_build_index_empty(self) -> None:
        embeddings = np.empty((0, 1024), dtype=np.float32)
        index = build_index(embeddings)
        assert index.ntotal == 0

    def test_search_empty_index(self) -> None:
        embeddings = np.empty((0, 1024), dtype=np.float32)
        index = build_index(embeddings)
        query = np.random.default_rng(0).random((1, 1024)).astype(np.float32)
        results = search(index, query, [], top_k=5)
        assert results == []

    def test_search_with_section_filter(self) -> None:
        rng = np.random.default_rng(456)
        embeddings = rng.random((5, 1024)).astype(np.float32)
        chunks = _make_chunks(5)

        index = build_index(embeddings)
        query = rng.random((1, 1024)).astype(np.float32)
        # Filter to only Risk Factors (item_number "1A") — chunks 0, 2, 4
        results = search(index, query, chunks, top_k=10, section_filter="1A")
        assert all(c.metadata["item_number"] == "1A" for c in results)

    def test_search_top_k_larger_than_index(self) -> None:
        rng = np.random.default_rng(789)
        embeddings = rng.random((3, 1024)).astype(np.float32)
        chunks = _make_chunks(3)

        index = build_index(embeddings)
        query = rng.random((1, 1024)).astype(np.float32)
        results = search(index, query, chunks, top_k=10)
        assert len(results) == 3

    def test_search_returns_correct_chunks(self) -> None:
        """Verify that search results correspond to the right chunk objects."""
        # Create embeddings where chunk 0 is identical to query
        dim = 1024
        query = np.ones((1, dim), dtype=np.float32)
        embeddings = np.zeros((3, dim), dtype=np.float32)
        embeddings[0] = query[0]  # Most similar
        embeddings[1] = -query[0]  # Least similar
        embeddings[2] = query[0] * 0.5

        chunks = _make_chunks(3)
        index = build_index(embeddings)
        results = search(index, query, chunks, top_k=1)
        assert len(results) == 1
        # The most similar should be chunk 0 (same direction as query)
        assert results[0].metadata["chunk_index"] == 0
