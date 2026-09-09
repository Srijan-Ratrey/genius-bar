"""Load the TWCS dump, pull out one brand's threads, and cache them as parquet.

The raw Kaggle csv is ~500MB / 2.8M tweets. Only `build_brand_threads` ever
touches it; everything downstream reads the committed parquet subsample via
`load_threads`, so no grader needs the raw file or Kaggle credentials.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
THREADS_PARQUET = REPO_ROOT / "data" / "apple_threads.parquet"

BRAND = "AppleSupport"
KAGGLE_DATASET = "thoughtvector/customer-support-on-twitter"
# "Tue Oct 31 22:11:45 +0000 2017"
CREATED_AT_FMT = "%a %b %d %H:%M:%S %z %Y"


def find_raw_csv() -> Path:
    """Locate twcs.csv, downloading from Kaggle only if it isn't already local."""
    local = list(RAW_DIR.glob("twcs*.csv"))
    if local:
        return local[0]

    # Loaded here rather than at import time so any ad-hoc script gets the
    # credentials without having to remember to call load_dotenv itself.
    load_dotenv(REPO_ROOT / ".env")

    if not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
        raise SystemExit(
            f"twcs.csv not found in {RAW_DIR} and Kaggle credentials are not set.\n"
            "Either:\n"
            f"  1. set KAGGLE_USERNAME and KAGGLE_KEY in .env (both are required "
            "-- the API key alone will not authenticate), or\n"
            f"  2. download {KAGGLE_DATASET} manually and put twcs.csv in {RAW_DIR}\n"
            "You do not need this file to run `make eval`; the subsample is committed."
        )

    import kagglehub  # imported lazily: it is only needed on this path

    path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    hits = list(path.glob("**/twcs*.csv"))
    if not hits:
        raise SystemExit(f"Downloaded {KAGGLE_DATASET} but found no twcs csv under {path}")
    return hits[0]


def _roots(child_to_parent: dict[int, int]) -> dict[int, int]:
    """Map every tweet to the root of its reply chain.

    Iterative with memoisation: chains run to hundreds of turns in this dataset,
    which is deep enough that the recursive version hits Python's stack limit.
    """
    root_of: dict[int, int] = {}
    for start in child_to_parent:
        if start in root_of:
            continue
        path = []
        node = start
        # Walk up until we reach a tweet with no parent in the dataset (parents
        # are often dangling -- the dump is a subsample of real Twitter), or a
        # node whose root we already know.
        while node in child_to_parent and node not in root_of:
            path.append(node)
            node = child_to_parent[node]
            if len(path) > 10_000:  # cycle guard; malformed data, not expected
                break
        root = root_of.get(node, node)
        for n in path:
            root_of[n] = root
    return root_of


def build_brand_threads(
    csv_path: Path | None = None,
    brand: str = BRAND,
    n_threads: int | None = 15_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Reconstruct the threads `brand` took part in.

    Two passes over the csv. The first reads only the three id/author columns so
    the reply graph fits comfortably in memory; that gives the exact set of
    tweet ids belonging to brand threads. The second re-reads in chunks and
    keeps just those rows. Cheaper and more exact than trying to grow the set
    outward from the brand's own tweets.
    """
    csv_path = csv_path or find_raw_csv()

    graph = pd.read_csv(
        csv_path,
        usecols=["tweet_id", "author_id", "in_response_to_tweet_id"],
        dtype={"tweet_id": "int64", "author_id": "string", "in_response_to_tweet_id": "float64"},
    )

    linked = graph.dropna(subset=["in_response_to_tweet_id"])
    child_to_parent = dict(
        zip(linked["tweet_id"], linked["in_response_to_tweet_id"].astype("int64"))
    )
    root_of = _roots(child_to_parent)
    graph["thread_id"] = [root_of.get(t, t) for t in graph["tweet_id"]]

    brand_threads = set(graph.loc[graph["author_id"] == brand, "thread_id"])
    if not brand_threads:
        raise SystemExit(f"No tweets by author_id={brand!r} in {csv_path}")

    if n_threads is not None and len(brand_threads) > n_threads:
        brand_threads = set(
            pd.Series(sorted(brand_threads)).sample(n=n_threads, random_state=seed)
        )

    thread_of = dict(zip(graph["tweet_id"], graph["thread_id"]))
    keep = {t for t, thread in thread_of.items() if thread in brand_threads}

    chunks = [
        chunk[chunk["tweet_id"].isin(keep)]
        for chunk in pd.read_csv(csv_path, chunksize=250_000)
    ]
    df = pd.concat(chunks, ignore_index=True)

    df["thread_id"] = df["tweet_id"].map(thread_of)
    df["created_at"] = pd.to_datetime(df["created_at"], format=CREATED_AT_FMT, utc=True)

    # Threads are trees, not chains -- a tweet can have several replies. Ordering
    # by timestamp linearises them into the conversation a human would read.
    df = df.sort_values(["thread_id", "created_at", "tweet_id"], ignore_index=True)
    df["turn"] = df.groupby("thread_id").cumcount()

    cols = ["thread_id", "turn", "tweet_id", "author_id", "inbound", "created_at", "text"]
    return df[cols]


# Apple is addressed both by handle and by the dump's anonymised numeric id.
_MENTION = re.compile(r"@\w+")
_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip handles and links, keeping everything that carries intent.

    Emoji, punctuation and casing are deliberately preserved: "WHY IS THIS
    BROKEN 😡" and "why is this broken" are the same intent but very different
    escalation signals, and flattening them would throw that away.

    Handles go because @AppleSupport and its numeric alias @115858 appear in
    most messages and carry no intent -- left in, they dominate TF-IDF.
    """
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    return _WS.sub(" ", text).strip()


def first_inbound(df: pd.DataFrame) -> pd.DataFrame:
    """One row per thread: the opening customer message.

    This is the classifier's real input -- at prediction time a first-touch
    message is all you have. Evaluating on mid-thread messages would leak the
    agent's own earlier replies and inflate every number.
    """
    inbound = df[df["inbound"]].sort_values(["thread_id", "turn"])
    first = inbound.groupby("thread_id", as_index=False).first()
    first["clean"] = first["text"].map(clean_text)
    return first[first["clean"].str.len() > 0].reset_index(drop=True)


def agent_replies(df: pd.DataFrame) -> pd.DataFrame:
    """Brand replies with the customer message each one answers, for grounding."""
    df = df.sort_values(["thread_id", "turn"])
    prev_text = df.groupby("thread_id")["text"].shift(1)
    prev_inbound = df.groupby("thread_id")["inbound"].shift(1)

    replies = df[(df["author_id"] == BRAND) & prev_inbound.fillna(False)].copy()
    replies["customer_text"] = prev_text[replies.index]
    replies["customer_clean"] = replies["customer_text"].map(clean_text)
    replies["reply_clean"] = replies["text"].map(clean_text)
    return replies.reset_index(drop=True)


def load_threads(path: Path = THREADS_PARQUET) -> pd.DataFrame:
    """Read the committed subsample. The only data entry point downstream code uses."""
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run `make data` to build it.")
    return pd.read_parquet(path)


def main() -> None:
    df = build_brand_threads()
    THREADS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(THREADS_PARQUET, index=False)

    size_mb = THREADS_PARQUET.stat().st_size / 1e6
    print(f"wrote {THREADS_PARQUET.relative_to(REPO_ROOT)}  ({size_mb:.1f} MB)")
    print(f"  threads: {df['thread_id'].nunique():,}")
    print(f"  tweets:  {len(df):,}  ({df['inbound'].sum():,} inbound)")
    print(f"  turns/thread: median {df.groupby('thread_id').size().median():.0f}, "
          f"max {df.groupby('thread_id').size().max()}")


if __name__ == "__main__":
    main()
