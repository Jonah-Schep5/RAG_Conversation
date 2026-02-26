"""
tests/test_split_data.py
Tests for the CSV data splitting step.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).parent.parent
sys.path.insert(0, str(SRC))

TRAIN_CSV = SRC / "train_data.csv"
TEST_CSV = SRC / "test_data.csv"
RAW_CSV = SRC / "telecom_synthetic_call_transcript_data.csv"

TRAIN_SIZE = 18322
TEST_SIZE = 1000
EXPECTED_COLS = [
    "Agent_ID", "Pilot", "Agent_AHT", "Call_ID", "Category",
    "Sub_Category", "Call_Start", "Call_End", "Call_Transfer",
    "Customer_Callback_7_Day", "Customer_Callback_IDs_7_Day",
    "CXM7", "Transcript_JSON",
]


class TestSplitFiles:
    def test_train_file_exists(self):
        assert TRAIN_CSV.exists(), "train_data.csv not found – run split_data.py"

    def test_test_file_exists(self):
        assert TEST_CSV.exists(), "test_data.csv not found – run split_data.py"

    def test_train_row_count(self):
        df = pd.read_csv(TRAIN_CSV)
        assert len(df) == TRAIN_SIZE, f"Expected {TRAIN_SIZE}, got {len(df)}"

    def test_test_row_count(self):
        df = pd.read_csv(TEST_CSV)
        assert len(df) == TEST_SIZE, f"Expected {TEST_SIZE}, got {len(df)}"

    def test_total_matches_raw(self):
        raw = pd.read_csv(RAW_CSV)
        train = pd.read_csv(TRAIN_CSV)
        test = pd.read_csv(TEST_CSV)
        assert len(train) + len(test) == len(raw)

    def test_columns_present(self):
        train = pd.read_csv(TRAIN_CSV)
        test = pd.read_csv(TEST_CSV)
        for col in EXPECTED_COLS:
            assert col in train.columns, f"Column '{col}' missing from train"
            assert col in test.columns, f"Column '{col}' missing from test"

    def test_no_overlap_in_call_ids(self):
        train = pd.read_csv(TRAIN_CSV)
        test = pd.read_csv(TEST_CSV)
        overlap = set(train["Call_ID"]) & set(test["Call_ID"])
        # The synthetic dataset contains a small number of duplicate Call_IDs
        # (found 3 in the raw data). We tolerate up to 5 duplicates; a larger
        # number would indicate a split logic error.
        assert len(overlap) <= 5, (
            f"Too many overlapping Call_IDs ({len(overlap)}): {overlap}"
        )

    def test_train_is_first_rows(self):
        raw = pd.read_csv(RAW_CSV)
        train = pd.read_csv(TRAIN_CSV)
        # First Call_ID of train should match raw
        assert train["Call_ID"].iloc[0] == raw["Call_ID"].iloc[0]
        assert train["Call_ID"].iloc[-1] == raw["Call_ID"].iloc[TRAIN_SIZE - 1]

    def test_test_is_last_rows(self):
        raw = pd.read_csv(RAW_CSV)
        test = pd.read_csv(TEST_CSV)
        assert test["Call_ID"].iloc[0] == raw["Call_ID"].iloc[TRAIN_SIZE]
        assert test["Call_ID"].iloc[-1] == raw["Call_ID"].iloc[-1]
