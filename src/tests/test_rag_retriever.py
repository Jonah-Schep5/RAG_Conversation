"""
tests/test_rag_retriever.py
Tests for the RAGRetriever class.
Requires faiss_index.bin and metadata.json (run build_index.py first).
"""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent
sys.path.insert(0, str(SRC))

INDEX_FILE = SRC / "faiss_index.bin"

pytestmark = pytest.mark.skipif(
    not INDEX_FILE.exists(), reason="FAISS index not built yet – run build_index.py"
)


@pytest.fixture(scope="module")
def retriever():
    from rag_retriever import RAGRetriever
    return RAGRetriever()


class TestRAGRetriever:
    def test_returns_correct_count(self, retriever):
        results = retriever.retrieve("high bill complaint", top_k=5)
        assert len(results) == 5

    def test_top_k_respected(self, retriever):
        for k in [1, 3, 7]:
            results = retriever.retrieve("activate sim card", top_k=k)
            assert len(results) == k

    def test_result_has_required_fields(self, retriever):
        results = retriever.retrieve("I want to cancel my plan", top_k=3)
        required = {"call_id", "category", "sub_category", "transcript_text", "similarity"}
        for r in results:
            assert required.issubset(r.keys()), f"Missing fields: {required - r.keys()}"

    def test_similarity_descending(self, retriever):
        results = retriever.retrieve("billing dispute high charges", top_k=5)
        scores = [r["similarity"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by similarity"

    def test_transcript_text_nonempty(self, retriever):
        results = retriever.retrieve("smartwatch cellular activation problem", top_k=3)
        for r in results:
            assert len(r["transcript_text"]) > 10, "Transcript text is too short"

    def test_different_queries_return_different_results(self, retriever):
        r1 = retriever.retrieve("activate smartwatch cellular", top_k=3)
        r2 = retriever.retrieve("cancel account billing dispute", top_k=3)
        ids1 = {r["call_id"] for r in r1}
        ids2 = {r["call_id"] for r in r2}
        # Different topics should not return identical top results
        assert ids1 != ids2, "Different queries returned identical results"

    def test_missing_index_raises_file_not_found(self, tmp_path, monkeypatch):
        import rag_retriever as rr
        monkeypatch.setattr(rr, "INDEX_FILE", tmp_path / "nonexistent.bin")
        with pytest.raises(FileNotFoundError):
            rr.RAGRetriever()
