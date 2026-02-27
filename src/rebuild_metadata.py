"""
rebuild_metadata.py
Regenerates metadata.json.gz from train_data.csv without re-running embeddings.
Run this whenever the metadata schema changes but the FAISS index is unchanged.

transcript_turns is stored as a compact list-of-lists [[speaker, text, event], ...]
instead of list-of-dicts to minimise file size.
"""
import gzip
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from build_index import transcript_to_text

DATA_DIR = Path(__file__).parent
TRAIN_CSV = DATA_DIR / "train_data.csv"
METADATA_FILE = DATA_DIR / "metadata.json.gz"


def rebuild_metadata() -> None:
    print(f"Loading {TRAIN_CSV} …")
    df = pd.read_csv(TRAIN_CSV)
    print(f"  Rows: {len(df)}")

    print("Converting transcripts to text …")
    texts = [transcript_to_text(t) for t in tqdm(df["Transcript_JSON"])]

    print("Building metadata …")
    metadata = []
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        try:
            raw_turns = json.loads(row.get("Transcript_JSON", "[]"))
            # Compact list-of-lists: [speaker, text, event] — no repeated key names
            turns = [
                [t.get("speaker", ""), t.get("text", ""), t.get("event", "null")]
                for t in raw_turns
            ]
        except (json.JSONDecodeError, TypeError):
            turns = []
        metadata.append({
            "call_id": str(row.get("Call_ID", "")),
            "category": str(row.get("Category", "")),
            "sub_category": str(row.get("Sub_Category", "")),
            "call_transfer": bool(row.get("Call_Transfer", False)),
            "callback_7day": int(row.get("Customer_Callback_7_Day", 0)),
            "agent_id": str(row.get("Agent_ID", "")),
            "transcript_text": texts[i],
            "transcript_turns": turns,
        })

    with gzip.open(METADATA_FILE, "wt", encoding="utf-8") as f:
        json.dump(metadata, f)
    size_mb = METADATA_FILE.stat().st_size / 1e6
    print(f"Metadata saved -> {METADATA_FILE}  ({len(metadata)} records, {size_mb:.1f} MB)")


if __name__ == "__main__":
    rebuild_metadata()
