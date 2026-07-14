"""
Unit tests for src/data_cleaning.py.

Run with:
    pytest tests/
or, without pytest installed:
    python tests/test_data_cleaning.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_cleaning import (
    clean_text,
    drop_invalid_rows,
    run_cleaning_pipeline,
    standardize_sentiment_labels,
    tokenize,
    word_frequencies,
)


def test_clean_text_lowercases():
    assert clean_text("AMAZING Game") == "amazing game"


def test_clean_text_removes_urls():
    result = clean_text("check this out http://t.co/abc123 now")
    assert "http" not in result
    assert "abc123" not in result


def test_clean_text_removes_mentions_and_hashtags():
    result = clean_text("@someuser this update is great #hype")
    assert "@" not in result
    assert "#" not in result
    assert "someuser" not in result
    assert "hype" in result


def test_clean_text_removes_stopwords():
    result = clean_text("this is the best and the worst")
    for stopword in ["this", "is", "the", "and"]:
        assert stopword not in result.split()


def test_clean_text_handles_non_string_input():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_tokenize_returns_list():
    tokens = tokenize("I love this Amazing Game!!!")
    assert isinstance(tokens, list)
    assert "amazing" in tokens
    assert "game" in tokens


def test_standardize_sentiment_labels():
    df = pd.DataFrame({"sentiment": [" positive", "NEGATIVE", "Neutral "]})
    result = standardize_sentiment_labels(df)
    assert list(result["sentiment"]) == ["Positive", "Negative", "Neutral"]


def test_drop_invalid_rows_removes_duplicates_and_missing_text():
    df = pd.DataFrame(
        {
            "tweet_id": [1, 1, 2, 3],
            "entity": ["A", "A", "B", "C"],
            "sentiment": ["Positive", "Positive", "Negative", "Neutral"],
            "text": ["great game", "great game", None, "it is fine"],
        }
    )
    result = drop_invalid_rows(df)
    assert len(result) == 2  # duplicate row #2 and the missing-text row #3 are dropped
    assert result.attrs["rows_removed"] == 2


def test_drop_invalid_rows_removes_unknown_sentiment_labels():
    df = pd.DataFrame(
        {
            "tweet_id": [1, 2],
            "entity": ["A", "B"],
            "sentiment": ["Positive", "NotARealLabel"],
            "text": ["nice", "hmm"],
        }
    )
    result = drop_invalid_rows(df)
    assert len(result) == 1
    assert result.iloc[0]["sentiment"] == "Positive"


def test_run_cleaning_pipeline_adds_derived_columns():
    df = pd.DataFrame(
        {
            "tweet_id": [1, 2],
            "entity": ["A", "B"],
            "sentiment": ["positive", "negative"],
            "text": ["I love this so much", "I hate waiting in line"],
        }
    )
    result = run_cleaning_pipeline(df)
    for col in ["clean_text", "char_count", "word_count"]:
        assert col in result.columns
    assert result["sentiment"].isin(["Positive", "Negative"]).all()


def test_word_frequencies_counts_correctly():
    texts = ["great game", "great story", "great graphics"]
    freq = word_frequencies(texts, top_n=5)
    assert freq["great"] == 3


def _run_all_tests_manually() -> None:
    """Fallback runner for environments without pytest installed."""
    tests = [obj for name, obj in globals().items() if name.startswith("test_")]
    passed, failed = 0, 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"PASS: {test_fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test_fn.__name__} -> {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_all_tests_manually()
