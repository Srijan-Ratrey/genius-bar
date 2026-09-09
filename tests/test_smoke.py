"""Checks for the parts where a silent bug would quietly corrupt every result."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from genius_bar.data import build_brand_threads

# All timestamps are a real Wednesday so the %a in CREATED_AT_FMT matches.
def _at(hour: int) -> str:
    return f"Wed Nov 01 {hour:02d}:00:00 +0000 2017"


@pytest.fixture
def csv(tmp_path):
    """A miniature twcs.csv covering the cases that actually bite.

    Thread A: a normal alternating customer/AppleSupport exchange, written to
              the csv out of timestamp order.
    Thread B: another brand entirely -- must be dropped.
    Thread C: replies to a tweet that isn't in the dump. The real dataset is a
              subsample, so dangling parents are common, not exotic.
    """
    rows = [
        # thread A, deliberately shuffled
        (3, "cust1", True, _at(12), "still broken", 4.0, 2.0),
        (1, "cust1", True, _at(10), "my iphone won't charge", 2.0, None),
        (4, "AppleSupport", False, _at(13), "let's take this to DM", None, 3.0),
        (2, "AppleSupport", False, _at(11), "have you tried a different cable?", 3.0, 1.0),
        # thread B, no AppleSupport anywhere
        (10, "cust2", True, _at(10), "hey spotify", 11.0, None),
        (11, "SpotifyCares", False, _at(11), "we're on it", None, 10.0),
        # thread C, parent 999 is absent from the dump
        (20, "cust3", True, _at(14), "as I said before", 21.0, 999.0),
        (21, "AppleSupport", False, _at(15), "sorry about that", None, 20.0),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "tweet_id", "author_id", "inbound", "created_at",
            "text", "response_tweet_id", "in_response_to_tweet_id",
        ],
    )
    path = tmp_path / "twcs.csv"
    df.to_csv(path, index=False)
    return path


def test_keeps_only_brand_threads(csv):
    df = build_brand_threads(csv, n_threads=None)
    assert set(df["tweet_id"]) == {1, 2, 3, 4, 20, 21}, "Spotify thread leaked in"


def test_chain_collapses_to_one_thread(csv):
    df = build_brand_threads(csv, n_threads=None)
    a = df[df["tweet_id"].isin([1, 2, 3, 4])]
    assert a["thread_id"].nunique() == 1, "a 4-turn chain split into multiple threads"


def test_turns_ordered_by_time_not_csv_order(csv):
    df = build_brand_threads(csv, n_threads=None)
    a = df[df["tweet_id"].isin([1, 2, 3, 4])].sort_values("turn")
    assert list(a["tweet_id"]) == [1, 2, 3, 4]
    assert list(a["turn"]) == [0, 1, 2, 3]


def test_dangling_parent_still_groups(csv):
    """20 and 21 belong together even though their root tweet is missing."""
    df = build_brand_threads(csv, n_threads=None)
    c = df[df["tweet_id"].isin([20, 21])]
    assert c["thread_id"].nunique() == 1
    assert list(c.sort_values("turn")["tweet_id"]) == [20, 21]


def test_subsample_is_deterministic(csv):
    one = build_brand_threads(csv, n_threads=1, seed=0)
    two = build_brand_threads(csv, n_threads=1, seed=0)
    assert one["thread_id"].nunique() == 1
    assert list(one["tweet_id"]) == list(two["tweet_id"])


# --- llm cache -------------------------------------------------------------
# These guard the reproduce claim: `make eval` must recompute every headline
# number from committed cache with no API key present.

from genius_bar import llm  # noqa: E402


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "LLM_CACHE", tmp_path / "llm")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(llm, "MIN_INTERVAL", 0.0)
    return tmp_path / "llm"


def test_cache_miss_without_key_is_explicit(isolated_cache):
    """A missing key must fail loudly, never silently return a fabricated answer."""
    with pytest.raises(llm.CacheMiss, match="GEMINI_API_KEY"):
        llm.generate("anything", schema={"type": "object"})


def test_cached_prompt_replays_without_key(isolated_cache):
    schema = {"type": "object", "properties": {"intent": {"type": "string"}}}
    key = {"model": llm.FLASH, "prompt": "hello", "schema": schema, "thinking": 0}

    isolated_cache.mkdir(parents=True)
    llm._cache_path(isolated_cache, key).write_text(
        json.dumps({**key, "response": {"intent": "billing"}})
    )

    assert llm.generate("hello", schema) == {"intent": "billing"}


def test_batch_length_mismatch_falls_back_per_item(monkeypatch):
    """A short batch response must not silently misalign results with inputs."""
    calls = []

    def fake_generate(prompt, schema, model=None, thinking=0):
        calls.append(prompt)
        n = prompt.count("|")
        # Simulate a model that drops an item whenever given more than one.
        return [{"v": 1}] * (n - 1) if n > 1 else [{"v": 1}]

    monkeypatch.setattr(llm, "generate", fake_generate)

    out = llm.map_batched(
        ["a", "b", "c"],
        lambda batch: "".join(f"|{x}" for x in batch),
        {"type": "object"},
        batch_size=3,
    )

    assert len(out) == 3, "result count must match input count"
    assert len(calls) == 4, "expected 1 failed batch + 3 per-item retries"
