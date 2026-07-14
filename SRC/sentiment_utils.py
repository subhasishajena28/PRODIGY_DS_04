"""
sentiment_utils.py
-------------------
Helpers for summarizing the (already labeled) sentiment column by entity,
plus an optional lexicon-based sentiment scorer (VADER) used purely as a
sanity check / bonus comparison against the ground-truth labels supplied
in the dataset.

VADER is intentionally optional: if `vaderSentiment` isn't installed, every
function here degrades gracefully (returns None / skips) instead of
raising, so the core pipeline never depends on it.
"""

from __future__ import annotations

import pandas as pd

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative", "Irrelevant"]


def vader_available() -> bool:
    try:
        import vaderSentiment  # noqa: F401

        return True
    except ImportError:
        return False


def score_with_vader(texts: pd.Series) -> pd.DataFrame | None:
    """Score raw (uncleaned) text with VADER's compound polarity score and
    map it to a Positive/Negative/Neutral label using VADER's own
    recommended thresholds.

    Returns None if vaderSentiment is not installed.
    """
    if not vader_available():
        return None

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    scores = texts.astype(str).apply(analyzer.polarity_scores)
    compound = scores.apply(lambda s: s["compound"])

    def label(c: float) -> str:
        if c >= 0.05:
            return "Positive"
        if c <= -0.05:
            return "Negative"
        return "Neutral"

    return pd.DataFrame(
        {
            "vader_compound": compound,
            "vader_sentiment": compound.apply(label),
        }
    )


def entity_sentiment_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot table of raw counts: rows = entity, columns = sentiment."""
    pivot = (
        df.groupby(["entity", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )
    for col in SENTIMENT_ORDER:
        if col not in pivot.columns:
            pivot[col] = 0
    return pivot[SENTIMENT_ORDER]


def entity_sentiment_percentages(df: pd.DataFrame) -> pd.DataFrame:
    """Same as entity_sentiment_counts but normalized to row percentages."""
    counts = entity_sentiment_counts(df)
    return counts.div(counts.sum(axis=1), axis=0) * 100


def top_entities_by_volume(df: pd.DataFrame, n: int = 10) -> list[str]:
    return df["entity"].value_counts().head(n).index.tolist()


def net_sentiment_score(df: pd.DataFrame) -> pd.Series:
    """A simple 'net sentiment' metric per entity:
        (%Positive - %Negative), ignoring Irrelevant/Neutral.
    Higher = more favorable public opinion, lower/negative = more criticized.
    """
    pct = entity_sentiment_percentages(df)
    return (pct["Positive"] - pct["Negative"]).sort_values(ascending=False)


def agreement_with_vader(df: pd.DataFrame, vader_df: pd.DataFrame) -> float:
    """% of rows where the dataset's labeled sentiment agrees with VADER's
    (Positive/Negative/Neutral only -- Irrelevant rows are excluded since
    VADER has no such class).
    """
    merged = df.loc[df["sentiment"] != "Irrelevant", "sentiment"]
    comparable = vader_df.loc[merged.index, "vader_sentiment"]
    return float((merged.values == comparable.values).mean() * 100)
