"""
rag_retriever.py
Provides a RAGRetriever class that:
  - Loads the FAISS index + metadata at startup
  - Encodes a query conversation with the same SentenceTransformer model
  - Returns the top-k most similar past conversations with their metadata
"""

import gzip
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent
INDEX_FILE = DATA_DIR / "faiss_index.bin"
METADATA_FILE = DATA_DIR / "metadata.json.gz"

EMBED_MODEL = "all-MiniLM-L6-v2"


class RAGRetriever:
    def __init__(self) -> None:
        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_FILE}. "
                "Run build_index.py first."
            )
        self.index = faiss.read_index(str(INDEX_FILE))
        with gzip.open(METADATA_FILE, "rt", encoding="utf-8") as f:
            raw: list[dict[str, Any]] = json.load(f)
        # Expand compact turns [[speaker, text, event], ...] back to dicts
        for item in raw:
            item["transcript_turns"] = [
                {"speaker": t[0], "text": t[1], "event": t[2]}
                for t in item.get("transcript_turns", [])
            ]
        self.metadata = raw
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
