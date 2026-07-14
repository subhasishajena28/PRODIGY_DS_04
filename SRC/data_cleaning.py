"""
data_cleaning.py
-----------------
Utilities for loading and cleaning the Twitter / social-media sentiment
dataset (Prodigy InfoTech - Task 4, based on the Kaggle "Twitter Entity
Sentiment Analysis" dataset).

The raw files ship WITHOUT a header row and have four columns in this
fixed order:

    tweet_id, entity, sentiment, text

This module intentionally avoids heavyweight NLP dependencies (nltk,
spaCy, ...) for the core cleaning path so the project keeps working even
on a machine with no internet access to download corpora. A small,
built-in English stopword list is used instead of nltk's downloadable one.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

COLUMN_NAMES = ["tweet_id", "entity", "sentiment", "text"]
VALID_SENTIMENTS = {"Positive", "Negative", "Neutral", "Irrelevant"}

# --------------------------------------------------------------------------
# A compact, dependency-free English stopword list.
# (Covers the same ground as sklearn/nltk's lists for this use case, without
# requiring an extra import or a runtime corpus download.)
# --------------------------------------------------------------------------
STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll
    he's her here here's hers herself him himself his how how's i i'd i'll
    i'm i've if in into is isn't it it's its itself let's me more most
    mustn't my myself no nor not of off on once only or other ought our
    ours ourselves out over own same shan't she she'd she'll she's should
    shouldn't so some such than that that's the their theirs them
    themselves then there there's these they they'd they'll they're
    they've this those through to too under until up very was wasn't we
    we'd we'll we're we've were weren't what what's when when's where
    where's which while who who's whom why why's with won't would
    wouldn't you you'd you'll you're you've your yours yourself
    yourselves rt via u ur im dont didnt doesnt isnt wasnt arent cant
    couldnt wouldnt shouldnt amp
    """.split()
)

_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_SIGN_RE = re.compile(r"#")
_HTML_ENTITY_RE = re.compile(r"&\w+;")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_REPEATED_CHARS_RE = re.compile(r"(.)\1{2,}")


def _short_path(path: Path, keep_parts: int = 3) -> str:
    """Render a path using only its last `keep_parts` components (e.g.
    'data/raw/twitter_training.csv') so log/notebook output stays readable
    and doesn't bake in a machine-specific absolute path.
    """
    parts = path.parts[-keep_parts:]
    return str(Path(*parts))


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    """Load one of the raw twitter_*.csv files (no header row) into a
    DataFrame with standardized column names.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMN_NAMES,
        encoding="utf-8",
        on_bad_lines="skip",
    )
    return df


def load_dataset(raw_dir: str | Path, sample_dir: str | Path) -> tuple[pd.DataFrame, str]:
    """Load the training data, preferring the full raw dataset if present
    and falling back to the small bundled sample otherwise.

    Returns (dataframe, source_description).
    """
    raw_dir = Path(raw_dir)
    sample_dir = Path(sample_dir)

    raw_file = raw_dir / "twitter_training.csv"
    sample_file = sample_dir / "twitter_sentiment_sample.csv"

    if raw_file.exists():
        df = load_raw_csv(raw_file)
        return df, f"full dataset ({_short_path(raw_file)})"

    if sample_file.exists():
        df = pd.read_csv(sample_file)
        return df, f"bundled sample dataset ({_short_path(sample_file)})"

    raise FileNotFoundError(
        "Could not find twitter_training.csv in data/raw/ or a sample csv "
        "in data/sample/. Run `python data/download_data.py` first, or see "
        "data/README.md."
    )


def standardize_sentiment_labels(df: pd.DataFrame, column: str = "sentiment") -> pd.DataFrame:
    """Trim whitespace and normalize casing of the sentiment label column."""
    df = df.copy()
    df[column] = df[column].astype(str).str.strip().str.title()
    return df


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing text/entity/sentiment, duplicate rows, and
    rows whose sentiment label isn't one of the four expected classes.
    """
    df = df.copy()
    before = len(df)

    df = df.dropna(subset=["text", "entity", "sentiment"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    df = df.drop_duplicates(subset=["tweet_id", "entity", "sentiment", "text"])
    df = df[df["sentiment"].isin(VALID_SENTIMENTS)]

    removed = before - len(df)
    df.attrs["rows_removed"] = removed
    return df.reset_index(drop=True)


def clean_text(text: str) -> str:
    """Normalize a single piece of social-media text for analysis:

    - lowercase
    - strip URLs, @mentions, HTML entities, the '#' sign (keeps the word)
    - strip punctuation & digits
    - collapse repeated characters ("soooo" -> "soo")
    - collapse extra whitespace
    - remove stopwords
    """
    if not isinstance(text, str):
        return ""

    t = text.lower()
    t = _URL_RE.sub(" ", t)
    t = _MENTION_RE.sub(" ", t)
    t = _HTML_ENTITY_RE.sub(" ", t)
    t = _HASHTAG_SIGN_RE.sub("", t)
    t = _NON_ALPHA_RE.sub(" ", t)
    t = _REPEATED_CHARS_RE.sub(r"\1\1", t)
    t = _MULTI_SPACE_RE.sub(" ", t).strip()

    tokens = [tok for tok in t.split() if tok not in STOPWORDS and len(tok) > 1]
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    return clean_text(text).split()


def add_derived_columns(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Add clean_text, word_count and char_count columns."""
    df = df.copy()
    df["clean_text"] = df[text_col].apply(clean_text)
    df["char_count"] = df[text_col].astype(str).str.len()
    df["word_count"] = df["clean_text"].str.split().apply(len)
    return df


def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: standardize labels -> drop invalid rows ->
    add derived text columns. Returns the cleaned dataframe.
    """
    df = standardize_sentiment_labels(df)
    df = drop_invalid_rows(df)
    df = add_derived_columns(df)
    return df


def word_frequencies(texts: Iterable[str], top_n: int = 20) -> pd.Series:
    """Return the top_n most frequent tokens across an iterable of
    already-cleaned text strings.
    """
    from collections import Counter

    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(str(t).split())
    return pd.Series(dict(counter.most_common(top_n)))
