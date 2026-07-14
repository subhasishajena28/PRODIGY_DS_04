"""
run_pipeline.py
-----------------
End-to-end command-line pipeline: load raw data -> clean it -> run the EDA
-> save all figures to outputs/figures/ and the cleaned dataset to
outputs/cleaned_data.csv.

This is the non-notebook way to run the whole project, e.g. in CI or from
a terminal:

    python run_pipeline.py
    python run_pipeline.py --top-n 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_cleaning import load_dataset, run_cleaning_pipeline
from src.sentiment_utils import (
    entity_sentiment_percentages,
    net_sentiment_score,
    score_with_vader,
    vader_available,
)
from src.visualization import (
    generate_wordcloud,
    plot_entity_sentiment_stacked,
    plot_net_sentiment,
    plot_sentiment_distribution,
    plot_text_length_distribution,
    plot_top_words_by_sentiment,
    set_style,
)

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full sentiment EDA pipeline.")
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--sample-dir", default=str(ROOT / "data" / "sample"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--top-n", type=int, default=10, help="Top-N entities to plot")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SOCIAL MEDIA SENTIMENT ANALYSIS -- PIPELINE")
    print("=" * 70)

    df_raw, source = load_dataset(args.raw_dir, args.sample_dir)
    print(f"\n[1/4] Loaded {len(df_raw):,} rows from {source}")

    df = run_cleaning_pipeline(df_raw)
    removed = df.attrs.get("rows_removed", 0)
    print(f"[2/4] Cleaned dataset: {len(df):,} rows kept, {removed:,} rows dropped "
          f"(missing/duplicate/invalid label)")

    cleaned_path = output_dir / "cleaned_data.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"      Saved cleaned data -> {cleaned_path}")

    print(f"[3/4] Sentiment breakdown:\n{df['sentiment'].value_counts().to_string()}")

    set_style()
    plot_sentiment_distribution(df, figures_dir / "01_sentiment_distribution.png")
    plot_entity_sentiment_stacked(df, top_n=args.top_n, save_path=figures_dir / "02_entity_sentiment_mix.png")
    plot_net_sentiment(net_sentiment_score(df), save_path=figures_dir / "03_net_sentiment_ranking.png")
    plot_text_length_distribution(df, figures_dir / "04_text_length_distribution.png")
    plot_top_words_by_sentiment(df, save_path=figures_dir / "05_top_words_by_sentiment.png")

    wc_fig = generate_wordcloud(" ".join(df["clean_text"]), figures_dir / "06_wordcloud.png")
    if wc_fig is None:
        print("      (skipped word cloud -- install 'wordcloud' to enable it)")

    if vader_available():
        vader_df = score_with_vader(df["text"])
        df = df.join(vader_df)
        df.to_csv(cleaned_path, index=False)
        print("      VADER lexicon scores added to cleaned_data.csv")
    else:
        print("      (skipped VADER comparison -- install 'vaderSentiment' to enable it)")

    print(f"[4/4] Saved {len(list(figures_dir.glob('*.png')))} figures -> {figures_dir}")
    print("\nDone. See outputs/ for results, or open notebooks/01_sentiment_eda.ipynb.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
