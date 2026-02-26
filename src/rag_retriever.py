"""
rag_retriever.py
Provides a RAGRetriever class that:
  - Loads the FAISS index + metadata at startup
  - Encodes a query conversation with the same SentenceTransformer model
  - Returns the top-k most similar past conversations with their metadata
"""

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent
INDEX_FILE = DATA_DIR / "faiss_index.bin"
METADATA_FILE = DATA_DIR / "metadata.json"

EMBED_MODEL = "all-MiniLM-L6-v2"


class RAGRetriever:
    def __init__(self) -> None:
        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_FILE}. "
                "Run build_index.py first."
            )
        self.index = faiss.read_index(str(INDEX_FILE))
        with open(METADATA_FILE) as f:
            self.metadata: list[dict[str, Any]] = json.load(f)
        self.model = SentenceTransformer(EMBED_MODEL)

    def retrieve(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top_k most similar conversations to query_text."""
        vec = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        distances, indices = self.index.search(vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.metadata[idx])
            item["similarity"] = float(dist)
            results.append(item)
        return results
