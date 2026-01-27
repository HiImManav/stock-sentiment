"""Bedrock Titan Text Embeddings V2 for SEC filing chunks."""

from __future__ import annotations

import json
import os

import boto3
import numpy as np


class EmbeddingModel:
    """Wrapper around Bedrock Titan Text Embeddings V2."""

    def __init__(
        self,
        model_id: str | None = None,
        bedrock_client: object | None = None,
    ) -> None:
        self._model_id = model_id or os.environ.get(
            "EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
        )
        self._client = bedrock_client or boto3.client("bedrock-runtime")

    def _invoke(self, text: str) -> list[float]:
        """Invoke the embedding model for a single text."""
        body = json.dumps({"inputText": text})
        response = self._client.invoke_model(
            modelId=self._model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        return result["embedding"]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns an (N, dim) array."""
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        embeddings = [self._invoke(t) for t in texts]
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns a (1, dim) array."""
        vec = self._invoke(query)
        return np.array([vec], dtype=np.float32)
