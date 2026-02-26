"""
build_index.py
Embeds each conversation in train_data.csv into a FAISS index.

Each chunk = one row = one full conversation (concatenated speaker turns).
Metadata (call_id, category, sub_category, call_transfer, callback_7day)
is stored in a companion JSON file for retrieval at query time.

Output files (written to src/):
  faiss_index.bin   - FAISS flat inner-product index
  metadata.json     - list of dicts, one per row, indexed by FAISS position
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_DIR = Path(__file__).parent
TRAIN_CSV = DATA_DIR / "train_data.csv"
INDEX_FILE = DATA_DIR / "faiss_index.bin"
METADATA_FILE = DATA_DIR / "metadata.json"

EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 256


def transcript_to_text(transcript_json_str: str) -> str:
    """Convert a Transcript_JSON string into a flat plain-text conversation."""
    try:
        turns = json.loads(transcript_json_str)
    except (json.JSONDecodeError, TypeError):
        return ""
    lines = []
    for turn in turns:
        speaker = turn.get("speaker", "unknown").capitalize()
        text = turn.get("text", "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def build_index() -> None:
    print(f"Loading {TRAIN_CSV} …")
    df = pd.read_csv(TRAIN_CSV)
    print(f"  Rows: {len(df)}")

    # Build plain-text conversation strings
    print("Converting transcripts to text …")
    texts = [transcript_to_text(t) for t in tqdm(df["Transcript_JSON"])]

    # Build metadata list (parallel to FAISS positions)
    metadata = []
    for _, row in df.iterrows():
        metadata.append({
            "call_id": str(row.get("Call_ID", "")),
            "category": str(row.get("Category", "")),
            "sub_category": str(row.get("Sub_Category", "")),
            "call_transfer": bool(row.get("Call_Transfer", False)),
            "callback_7day": int(row.get("Customer_Callback_7_Day", 0)),
            "agent_id": str(row.get("Agent_ID", "")),
            "transcript_text": texts[_],   # store full text for retrieval
        })

    # Embed
    print(f"Loading embedding model '{EMBED_MODEL}' …")
    model = SentenceTransformer(EMBED_MODEL)

    print("Embedding conversations (this may take a few minutes) …")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity via inner product
    )

    dim = embeddings.shape[1]
    print(f"Embedding dim: {dim}, total vectors: {len(embeddings)}")

    # Build FAISS index (inner product = cosine similarity after L2-norm)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(INDEX_FILE))
    print(f"FAISS index saved -> {INDEX_FILE}")

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)
    print(f"Metadata saved   -> {METADATA_FILE}")

    print("Done.")


if __name__ == "__main__":
    build_index()
