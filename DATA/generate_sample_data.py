"""
generate_sample_data.py
------------------------
Creates data/sample/twitter_sentiment_sample.csv: a small, ORIGINAL,
synthetically-generated dataset that mirrors the schema of the real
Twitter Entity Sentiment Analysis dataset used in Task 4
(tweet_id, entity, sentiment, text) so the notebook and pipeline can run
end-to-end immediately after cloning the repo -- no download required.

This is demo/test data only. It is written from scratch (template
sentences + randomized noise), not copied or scraped from any dataset,
and is deliberately messy (duplicates, missing text, mixed casing,
URLs/@mentions/hashtags, elongated words) so the cleaning steps in
src/data_cleaning.py have something real to do.

For the real 74k-row dataset, run `python data/download_data.py` instead.

Usage:
    python data/generate_sample_data.py
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

random.seed(42)

ENTITIES = [
    "Borderlands", "Overwatch", "Amazon", "Google", "Microsoft",
    "PlayStation5", "Xbox", "Nvidia", "CallOfDuty", "Cyberpunk2077",
]

TEMPLATES = {
    "Positive": [
        "I absolutely love {e}, best experience ever!",
        "{e} just made my day, this update is incredible honestly",
        "Been using {e} all week and it keeps getting better",
        "Huge shoutout to {e} for such an amazing release",
        "{e} never disappoints, highly recommend to everyone",
        "This is exactly why {e} is my favorite, pure quality",
        "Can't stop talking about how good {e} is right now",
        "{e} support team fixed my issue in minutes, awesome service",
        "Just tried the new {e} update and I'm so impressed",
        "{e} community is so wholesome, love being part of it",
        "Genuinely happy with {e} lately, well done team",
        "{e} really outdid themselves this time, fantastic work",
    ],
    "Negative": [
        "{e} keeps crashing on me, so frustrating",
        "Really disappointed with the new {e} update, feels worse now",
        "{e} customer service was unhelpful honestly",
        "Why does {e} keep charging me for things I didn't buy",
        "{e} servers have been down all day, unacceptable",
        "I regret spending money on {e}, total waste",
        "{e} is lagging so bad I can barely use it",
        "Worst experience with {e} support, still unresolved",
        "{e} removed the feature everyone liked, terrible decision",
        "Thinking about quitting {e} for good after this mess",
        "{e} has gone downhill these past few months",
        "Not happy at all with how {e} handled this",
    ],
    "Neutral": [
        "{e} released patch notes earlier today",
        "Anyone know the system requirements for {e}",
        "{e} servers scheduled for maintenance tonight at 10pm",
        "Just saw {e} trending, wonder what happened",
        "{e} announced a new event starting next week",
        "Does {e} work on the older hardware models",
        "Reading through the {e} changelog before the weekend",
        "{e} stock moved slightly after the earnings call",
        "Comparing {e} pricing plans before deciding",
        "This {e} tutorial helped me understand the basics",
        "{e} posted an update on their roadmap today",
        "Looking for reviews on the latest {e} release",
    ],
    "Irrelevant": [
        "Watching a movie tonight, nothing to do with {e}",
        "Grabbed coffee this morning then headed to work",
        "Saw {e} mentioned in passing but this is about lunch",
        "Anyone else stuck in traffic right now lol",
        "Just finished a workout, feeling great today",
        "Random thought, pineapple does not belong on pizza",
        "My cat knocked over a plant again",
        "{e} came up on my timeline but I'm just here for the memes",
        "Can't decide what to cook for dinner tonight",
        "Weather has been so weird all week here",
        "Finally cleaned my room after putting it off forever",
        "Weekend plans involve absolutely nothing and I love it",
    ],
}

NOISY_PREFIXES = ["RT @user123: ", "", "", "", "@friendly_user ", ""]
NOISY_SUFFIXES = [
    "", "", " #gaming", " #brand", " check http://t.co/abc123",
    " via @source", " !!!", " ...", " lolll", "",
]


def _add_noise(sentence: str) -> str:
    prefix = random.choice(NOISY_PREFIXES)
    suffix = random.choice(NOISY_SUFFIXES)
    text = f"{prefix}{sentence}{suffix}"

    if random.random() < 0.12:
        text = text.upper()
    if random.random() < 0.15:
        text = text.replace("so ", "sooo ").replace("good", "goooood")
    return text


def build_dataset(rows_per_entity_sentiment: int = 10) -> pd.DataFrame:
    records = []
    tweet_id = 10000

    for entity in ENTITIES:
        for sentiment, templates in TEMPLATES.items():
            for _ in range(rows_per_entity_sentiment):
                template = random.choice(templates)
                sentence = template.format(e=entity)
                text = _add_noise(sentence)
                records.append(
                    {
                        "tweet_id": tweet_id,
                        "entity": entity,
                        "sentiment": sentiment,
                        "text": text,
                    }
                )
                tweet_id += 1

    df = pd.DataFrame.from_records(records)

    # Shuffle so sentiment/entity blocks aren't in sorted order (more
    # realistic + gives the notebook something to actually explore).
    df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)

    # --- Inject realistic messiness for the cleaning step to handle ---
    # 1) A handful of exact duplicate rows.
    dup_rows = df.sample(n=8, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 2) A few rows with missing text.
    missing_idx = df.sample(n=5, random_state=2).index
    df.loc[missing_idx, "text"] = None

    # 3) A few rows with inconsistent sentiment casing ("positive" vs "Positive").
    casing_idx = df.sample(n=6, random_state=3).index
    df.loc[casing_idx, "sentiment"] = df.loc[casing_idx, "sentiment"].str.lower()

    return df.reset_index(drop=True)


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "sample"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "twitter_sentiment_sample.csv"

    df = build_dataset(rows_per_entity_sentiment=10)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic rows to {out_path}")


if __name__ == "__main__":
    main()
