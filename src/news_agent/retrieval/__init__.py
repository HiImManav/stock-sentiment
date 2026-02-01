"""Retrieval components - reuses sec_agent patterns."""

# For now, we import from sec_agent to share embedding and vector store code
# If needed for standalone deployment, copy the implementations here

try:
    from sec_agent.retrieval.embeddings import EmbeddingModel
    from sec_agent.retrieval.vector_store import build_index, search
except ImportError:
    # Standalone mode - would need local implementations
    EmbeddingModel = None
    build_index = None
    search = None

__all__ = ["EmbeddingModel", "build_index", "search"]
