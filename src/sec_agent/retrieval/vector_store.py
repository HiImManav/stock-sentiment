"""FAISS-based in-memory vector search for SEC filing chunks."""

from __future__ import annotations

import faiss
import numpy as np

from sec_agent.parser.chunker import Chunk


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product (cosine similarity) index.

    Embeddings are L2-normalized before indexing so that inner product
    equals cosine similarity.
    """
    if embeddings.shape[0] == 0:
        return faiss.IndexFlatIP(1024)
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    chunks: list[Chunk],
    top_k: int = 10,
    section_filter: str | None = None,
) -> list[Chunk]:
    """Search the FAISS index and return ranked chunks.

    If section_filter is provided, only chunks whose item_number matches
    the filter are returned.
    """
    if index.ntotal == 0:
        return []

    # Normalize query for cosine similarity
    query = query_embedding.copy()
    faiss.normalize_L2(query)

    # Search more than top_k if filtering to ensure enough results
    search_k = min(index.ntotal, top_k * 3 if section_filter else top_k)
    scores, indices = index.search(query, search_k)

    results: list[Chunk] = []
    for idx in indices[0]:
        if idx < 0:
            continue
        chunk = chunks[idx]
        if section_filter and chunk.metadata.get("item_number") != section_filter:
            continue
        results.append(chunk)
        if len(results) >= top_k:
            break

    return results
