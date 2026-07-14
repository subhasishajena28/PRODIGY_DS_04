"""
visualization.py
-----------------
Reusable, consistently-styled plotting functions for the sentiment EDA.
Every function returns the matplotlib Figure it created and optionally
saves it to disk (dpi=150, tight bounding box) so both the notebook and
`run_pipeline.py` can share the exact same plotting code.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SENTIMENT_COLORS = {
    "Positive": "#2ca02c",
    "Negative": "#d62728",
    "Neutral": "#7f7f7f",
    "Irrelevant": "#9467bd",
}
SENTIMENT_ORDER = ["Positive", "Neutral", "Negative", "Irrelevant"]


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlesize"] = 13


def _save(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_sentiment_distribution(df: pd.DataFrame, save_path: str | Path | None = None) -> plt.Figure:
    counts = df["sentiment"].value_counts().reindex(SENTIMENT_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(counts.index, counts.values, color=[SENTIMENT_COLORS[s] for s in counts.index])
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set_title("Overall Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Posts")
    _save(fig, save_path)
    return fig


def plot_entity_sentiment_stacked(
    df: pd.DataFrame, top_n: int = 10, save_path: str | Path | None = None
) -> plt.Figure:
    from src.sentiment_utils import entity_sentiment_percentages, top_entities_by_volume

    top_entities = top_entities_by_volume(df, top_n)
    pct = entity_sentiment_percentages(df).loc[top_entities]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottom = np.zeros(len(pct))
    for sentiment in SENTIMENT_ORDER:
        ax.barh(pct.index, pct[sentiment], left=bottom, label=sentiment, color=SENTIMENT_COLORS[sentiment])
        bottom += pct[sentiment].values
    ax.set_xlabel("Share of Posts (%)")
    ax.set_title(f"Sentiment Mix by Entity/Topic (Top {top_n} by Volume)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Sentiment")
    ax.invert_yaxis()
    _save(fig, save_path)
    return fig


def plot_net_sentiment(net_series: pd.Series, top_n: int = 15, save_path: str | Path | None = None) -> plt.Figure:
    subset = net_series.head(top_n)
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in subset.values]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(subset.index, subset.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Net Sentiment Score  (% Positive − % Negative)")
    ax.set_title("Most Favorably vs. Least Favorably Viewed Entities")
    ax.invert_yaxis()
    _save(fig, save_path)
    return fig


def plot_text_length_distribution(df: pd.DataFrame, save_path: str | Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sentiment in SENTIMENT_ORDER:
        subset = df.loc[df["sentiment"] == sentiment, "word_count"]
        if len(subset):
            sns.kdeplot(subset, label=sentiment, color=SENTIMENT_COLORS[sentiment], fill=True, alpha=0.15, ax=ax)
    ax.set_title("Cleaned Word-Count Distribution by Sentiment")
    ax.set_xlabel("Words per Post (after cleaning)")
    ax.legend(title="Sentiment")
    _save(fig, save_path)
    return fig


def plot_top_words(word_freq: pd.Series, title: str, color: str = "#1f77b4", save_path: str | Path | None = None) -> plt.Figure:
    ordered = word_freq.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.32 * len(ordered))))
    ax.barh(ordered.index, ordered.values, color=color)
    ax.set_title(title)
    ax.set_xlabel("Frequency")
    _save(fig, save_path)
    return fig


def plot_top_words_by_sentiment(
    df: pd.DataFrame,
    sentiments: tuple[str, str] = ("Positive", "Negative"),
    top_n: int = 12,
    save_path: str | Path | None = None,
) -> plt.Figure:
    from src.data_cleaning import word_frequencies

    fig, axes = plt.subplots(1, len(sentiments), figsize=(6 * len(sentiments), 5))
    if len(sentiments) == 1:
        axes = [axes]

    for ax, sentiment in zip(axes, sentiments):
        texts = df.loc[df["sentiment"] == sentiment, "clean_text"]
        freq = word_frequencies(texts, top_n=top_n).sort_values(ascending=True)
        ax.barh(freq.index, freq.values, color=SENTIMENT_COLORS.get(sentiment, "#1f77b4"))
        ax.set_title(f"Top Words — {sentiment}")
        ax.set_xlabel("Frequency")

    fig.tight_layout()
    _save(fig, save_path)
    return fig


def generate_wordcloud(text: str, save_path: str | Path | None = None):
    """Optional word cloud (requires the `wordcloud` package). Returns the
    Figure, or None if wordcloud isn't installed.
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        return None

    wc = WordCloud(width=1000, height=500, background_color="white", colormap="viridis").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud")
    _save(fig, save_path)
    return fig
