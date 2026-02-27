"""
rebuild_metadata.py
Regenerates metadata.json from train_data.csv without re-running embeddings.
Run this whenever the metadata schema changes but the FAISS index is unchanged.
"""
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from build_index import transcript_to_text

DATA_DIR = Path(__file__).parent
TRAIN_CSV = DATA_DIR / "train_data.csv"
METADATA_FILE = DATA_DIR / "metadata.json"


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
            turns = json.loads(row.get("Transcript_JSON", "[]"))
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

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)
    print(f"Metadata saved -> {METADATA_FILE}  ({len(metadata)} records)")


if __name__ == "__main__":
    rebuild_metadata()
