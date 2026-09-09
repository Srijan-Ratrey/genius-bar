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
    # REPO_ROOT too: _get_client calls load_dotenv(REPO_ROOT/".env"), which would
    # otherwise reload the real key and make this test issue a live request.
    monkeypatch.setattr(llm, "REPO_ROOT", tmp_path)
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
    key = {"model": llm.PRIMARY, "prompt": "hello", "schema": schema, "thinking": 0}

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


# --- metrics ---------------------------------------------------------------

from genius_bar import metrics  # noqa: E402


def test_escalation_cost_is_asymmetric():
    """Missing an escalation must cost more than escalating needlessly.

    If these ever come out equal, the cost model has collapsed into plain
    accuracy and the whole reason for a cost-weighted metric is gone.
    """
    missed = metrics.escalation_metrics([True] * 10, [False] + [True] * 9)
    needless = metrics.escalation_metrics([False] * 10, [True] + [False] * 9)

    assert missed["missed_escalations"] == 1
    assert needless["needless_escalations"] == 1
    assert missed["cost_per_100"] > needless["cost_per_100"]


def test_trivial_never_escalate_is_visibly_bad():
    """The trivial baseline should score 0 recall, not an accidental pass."""
    out = metrics.escalation_metrics([True] * 3 + [False] * 7, [False] * 10)
    assert out["recall"] == 0.0
    assert out["missed_escalations"] == 3


def test_agreement_separates_bias_from_correlation():
    """A judge that is always +1 ranks perfectly but is biased -- report both."""
    human = [1, 2, 3, 4, 5]
    out = metrics.agreement(human, [h + 1 for h in human])
    assert out["spearman"] == pytest.approx(1.0)
    assert out["judge_bias"] == pytest.approx(1.0)
    assert out["exact_match"] == 0.0


def test_agreement_handles_flat_ratings():
    """A judge that gives everything a 4 has undefined correlation, not a crash."""
    out = metrics.agreement([3, 4, 5], [4, 4, 4])
    assert out["spearman"] is None


# --- rate limits vs size limits --------------------------------------------
# The distinction these guard cost a wasted embedding run: splitting a batch
# that was refused for *rate* doubles the request count against a per-minute
# quota, so it accelerates into the limit instead of backing off.

RATE_LIMIT_ERR = (
    "ClientError: 429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
    "embed_content_free_tier_requests, limit: 100 ... 'retryDelay': '27s'"
)
TOO_LARGE_ERR = (
    "ClientError: 400 INVALID_ARGUMENT. * BatchEmbedContentsRequest.requests: "
    "at most 100 requests can be batched"
)


def test_rate_limit_and_size_errors_are_distinguished():
    rate, size = Exception(RATE_LIMIT_ERR), Exception(TOO_LARGE_ERR)
    assert llm._is_rate_limit(rate) and not llm._is_too_large(rate)
    assert llm._is_too_large(size) and not llm._is_rate_limit(size)


def test_retry_delay_read_from_server_response():
    """Honour the server's own figure; guessing shorter just refills the window."""
    assert llm._retry_after(Exception(RATE_LIMIT_ERR)) == pytest.approx(29.0)
    assert llm._retry_after(Exception("429 no delay given"), default=30.0) == 30.0


def test_rate_limited_batch_is_retried_whole_not_split(monkeypatch):
    seen = []

    class FakeModels:
        def embed_content(self, model, contents, config):
            seen.append(len(contents))
            if len(seen) == 1:
                raise Exception(RATE_LIMIT_ERR)
            return type("R", (), {
                "embeddings": [type("E", (), {"values": [0.0, 1.0]})() for _ in contents]
            })()

    monkeypatch.setattr(llm, "_get_client", lambda: type("C", (), {"models": FakeModels()})())
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm, "EMBED_MIN_INTERVAL", 0.0)

    out = llm._embed_live(["a"] * 40, "m", 2)

    assert len(out) == 40
    assert seen == [40, 40], f"batch was split instead of retried whole: {seen}"


def test_oversized_batch_is_split(monkeypatch):
    seen = []

    class FakeModels:
        def embed_content(self, model, contents, config):
            seen.append(len(contents))
            if len(contents) > 20:
                raise Exception(TOO_LARGE_ERR)
            return type("R", (), {
                "embeddings": [type("E", (), {"values": [0.0, 1.0]})() for _ in contents]
            })()

    monkeypatch.setattr(llm, "_get_client", lambda: type("C", (), {"models": FakeModels()})())
    monkeypatch.setattr(llm, "EMBED_MIN_INTERVAL", 0.0)

    out = llm._embed_live(["a"] * 40, "m", 2)

    assert len(out) == 40
    assert max(seen) == 40 and seen[-1] <= 20, f"expected a split: {seen}"


# --- escalation policy -----------------------------------------------------
# The component where a silent bug is most expensive: a wrongly auto-handled
# payment dispute or safety report is an incident, not a metric regression.

from genius_bar.agent import MIN_CONFIDENCE, MIN_EVIDENCE_SCORE, decide, load_intents  # noqa: E402

CONFIDENT, GROUNDED = 0.95, 0.40


@pytest.mark.parametrize("signal", [
    "safety_risk", "legal_or_press_threat", "payment_dispute",
    "irreversible_data_loss", "non_english",
])
def test_hard_signals_escalate_even_when_confident_and_grounded(signal):
    action, reason = decide("update_performance", CONFIDENT, [signal], GROUNDED)
    assert action == "escalate", f"{signal} was auto-handled"
    assert reason, "escalation must always carry a stated reason"


def test_severity_order_determines_the_stated_reason():
    """Safety outranks everything: the reason given must be the most serious one."""
    _, reason = decide(
        "data_loss", 0.1, ["safety_risk", "payment_dispute", "repeat_contact"], 0.0
    )
    assert "safety" in reason.lower()


def test_high_risk_intents_never_auto_handle():
    for name, meta in load_intents().items():
        if meta["risk"] != "high":
            continue
        action, _ = decide(name, CONFIDENT, [], GROUNDED)
        assert action == "escalate", f"high-risk intent {name} was auto-handled"


def test_low_confidence_escalates():
    action, reason = decide("update_performance", MIN_CONFIDENCE - 0.01, [], GROUNDED)
    assert action == "escalate" and "confidence" in reason


def test_ungrounded_draft_escalates():
    """No precedent means a "grounded" draft would be grounded in noise."""
    action, reason = decide("update_performance", CONFIDENT, [], MIN_EVIDENCE_SCORE - 0.01)
    assert action == "escalate" and "precedent" in reason


def test_routine_case_auto_handles():
    action, _ = decide("update_performance", CONFIDENT, [], GROUNDED)
    assert action == "auto", "nothing would ever be automated"


def test_anger_alone_does_not_escalate():
    """A deliberate policy choice, not an oversight.

    Profanity is the default register in this corpus. Escalating on it would
    route 30-40% of traffic to humans and defeat the point of the system, so
    anger is passed to the drafting step to soften tone instead. Documented in
    DECISIONS.md and revisited in the report's failure analysis.
    """
    action, _ = decide("update_performance", CONFIDENT, ["severe_anger"], GROUNDED)
    assert action == "auto"


def test_off_taxonomy_label_does_not_become_a_confident_prediction():
    """A hallucinated intent name must not sail through as high confidence."""
    action, _ = decide("refund_my_money_now", 0.99, [], GROUNDED)
    assert action == "escalate"


def test_ungrounded_support_draft_is_downgraded_to_escalate(monkeypatch):
    """The grounding claim has to be enforced, not just asserted.

    Retrieval score says similar precedent EXISTS; it cannot say the draft used
    any. A support-intent draft that cites nothing is not grounded, whatever
    its score was.
    """
    from genius_bar import agent

    monkeypatch.setattr(agent, "classify", lambda m, **kw: [
        {"intent": "update_performance", "confidence": 0.95, "signals": []},
        {"intent": "not_actionable", "confidence": 0.95, "signals": []},
    ])
    monkeypatch.setattr(agent, "draft", lambda items, **kw: [
        {"draft": "Have you tried restarting?", "used_evidence": []} for _ in items
    ])

    class FakeRetriever:
        def search(self, q, k=5):
            return [{"customer": "c", "reply": "r", "score": 0.9}]

    support, venting = agent.triage(["battery dies fast", "you all suck"], FakeRetriever())

    assert support.action == "escalate" and "not grounded" in support.reason
    assert venting.action == "auto", "not_actionable needs no precedent to cite"
    assert support.grounded is False
