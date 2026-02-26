"""
split_data.py
Splits the telecom CSV into:
  - train_data.csv  : first 18322 rows  (RAG knowledge base)
  - test_data.csv   : last 1000 rows    (held-out evaluation)
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent
RAW_CSV = DATA_DIR / "telecom_synthetic_call_transcript_data.csv"
TRAIN_CSV = DATA_DIR / "train_data.csv"
TEST_CSV = DATA_DIR / "test_data.csv"

TRAIN_SIZE = 18322
TEST_SIZE = 1000


def split_csv() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(RAW_CSV)
    total = len(df)
    assert total == TRAIN_SIZE + TEST_SIZE, (
        f"Expected {TRAIN_SIZE + TEST_SIZE} rows, got {total}"
    )

    train_df = df.iloc[:TRAIN_SIZE].reset_index(drop=True)
    test_df = df.iloc[TRAIN_SIZE:].reset_index(drop=True)

    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print(f"Train: {len(train_df)} rows -> {TRAIN_CSV}")
    print(f"Test:  {len(test_df)} rows -> {TEST_CSV}")
    return train_df, test_df


if __name__ == "__main__":
    split_csv()
