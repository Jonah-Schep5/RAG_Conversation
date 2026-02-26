"""
tests/test_build_index.py
Tests for transcript_to_text conversion and FAISS index properties.
Requires faiss_index.bin and metadata.json to exist (run build_index.py first).
"""
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pytest

SRC = Path(__file__).parent.parent
sys.path.insert(0, str(SRC))

from build_index import transcript_to_text

INDEX_FILE = SRC / "faiss_index.bin"
METADATA_FILE = SRC / "metadata.json"


class TestTranscriptToText:
    def test_basic_conversion(self):
        turns = [
            {"speaker": "agent", "text": "Hello, how can I help?", "event": "null"},
            {"speaker": "customer", "text": "My bill is wrong.", "event": "null"},
        ]
        result = transcript_to_text(json.dumps(turns))
        assert "Agent: Hello, how can I help?" in result
        assert "Customer: My bill is wrong." in result

    def test_empty_transcript(self):
        result = transcript_to_text("[]")
        assert result == ""

    def test_invalid_json(self):
        result = transcript_to_text("not valid json")
        assert result == ""

    def test_speaker_capitalized(self):
        turns = [{"speaker": "agent", "text": "Hi", "event": "null"}]
        result = transcript_to_text(json.dumps(turns))
        assert result.startswith("Agent:")

    def test_skips_empty_text_turns(self):
        turns = [
            {"speaker": "agent", "text": "", "event": "hold_start"},
            {"speaker": "customer", "text": "Hello", "event": "null"},
        ]
        result = transcript_to_text(json.dumps(turns))
        assert "Agent:" not in result
        assert "Customer: Hello" in result

    def test_multi_turn_ordering(self):
        turns = [
            {"speaker": "agent", "text": "First", "event": "null"},
            {"speaker": "customer", "text": "Second", "event": "null"},
            {"speaker": "agent", "text": "Third", "event": "null"},
        ]
        result = transcript_to_text(json.dumps(turns))
        lines = result.splitlines()
        assert len(lines) == 3
        assert lines[0].startswith("Agent:")
        assert lines[1].startswith("Customer:")
        assert lines[2].startswith("Agent:")


@pytest.mark.skipif(not INDEX_FILE.exists(), reason="FAISS index not built yet")
class TestFAISSIndex:
    def setup_method(self):
        self.index = faiss.read_index(str(INDEX_FILE))
        with open(METADATA_FILE) as f:
            self.metadata = json.load(f)

    def test_index_has_correct_row_count(self):
        assert self.index.ntotal == 18322, (
            f"Expected 18322 vectors, got {self.index.ntotal}"
        )

    def test_metadata_length_matches_index(self):
        assert len(self.metadata) == self.index.ntotal

    def test_metadata_has_required_keys(self):
        required = {"call_id", "category", "sub_category", "transcript_text"}
        for item in self.metadata[:10]:
            assert required.issubset(item.keys()), f"Missing keys in {item}"

    def test_search_returns_valid_indices(self):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query = "My bill is too high and I want to cancel my plan."
        vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = self.index.search(vec.astype(np.float32), 5)
        assert len(indices[0]) == 5
        assert all(0 <= idx < self.index.ntotal for idx in indices[0])

    def test_similarity_scores_in_range(self):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query = "I need to activate my new SIM card."
        vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        distances, _ = self.index.search(vec.astype(np.float32), 5)
        for score in distances[0]:
            assert -1.1 <= float(score) <= 1.1, f"Unexpected similarity score: {score}"
