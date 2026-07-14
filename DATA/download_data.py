"""
download_data.py
------------------
Downloads the full real dataset (twitter_training.csv + twitter_validation.csv)
from the Prodigy InfoTech GitHub repository into data/raw/.

Source:
https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%204

This dataset is the "Twitter Entity Sentiment Analysis" dataset
(originally published on Kaggle by Jaskaran Puri), redistributed by
Prodigy InfoTech for their Data Science internship Task 4.

Usage:
    python data/download_data.py

Requires an internet connection. If this script can't reach GitHub (e.g.
you're offline or behind a restrictive firewall), just download the two
CSV files manually from the link above and place them in data/raw/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

BASE_URL = "https://raw.githubusercontent.com/Prodigy-InfoTech/data-science-datasets/main/Task%204/"
FILES = ["twitter_training.csv", "twitter_validation.csv"]


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url} ...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  -> saved to {dest}  ({size_mb:.2f} MB)")


def main() -> int:
    raw_dir = Path(__file__).resolve().parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        for filename in FILES:
            download(BASE_URL + filename, raw_dir / filename)
    except requests.RequestException as exc:
        print(f"\nDownload failed: {exc}")
        print(
            "\nYou can download the files manually instead from:\n"
            "https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%204\n"
            f"and place them in: {raw_dir}"
        )
        return 1

    print("\nDone. Full dataset is ready in data/raw/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
