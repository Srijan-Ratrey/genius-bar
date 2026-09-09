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


# --- baselines -------------------------------------------------------------

from genius_bar import baselines  # noqa: E402


def test_trivial_baseline_never_escalates_and_is_one_reply():
    out = baselines.trivial(["battery dead", "unauthorized charge", "boiling hot"])
    assert {t.action for t in out} == {"auto"}, "trivial baseline must never escalate"
    assert len({t.draft for t in out}) == 1, "trivial baseline sends one canned reply"
    assert len({t.intent for t in out}) == 1


def test_baseline_rules_are_ordered_specific_before_catch_all():
    """update_performance keywords appear inside other intents' messages.

    "all my contacts are gone after updating" contains "updating"; if the
    catch-all ran first it would swallow the data_loss case.
    """
    assert baselines.classify_by_rules("all my contacts are gone after updating") == "data_loss"
    assert baselines.classify_by_rules("refund my iTunes charge from the update") == "account_billing"


def test_baseline_escalates_high_risk_intents_too():
    """Kept comparable to the agent, so escalation metrics measure signal quality."""
    action, reason = baselines.escalate_by_rules("been to the genius bar 4 times", "hardware_repair")
    assert action == "escalate" and "high-risk" in reason


def test_simple_baseline_copies_verbatim_and_does_not_synthesise():
    class FakeRetriever:
        def search(self, q, k=5):
            return [{"customer": "c", "reply": "EXACT HISTORICAL REPLY", "score": 0.7}]

    out = baselines.simple(["my battery drains fast"], FakeRetriever())
    assert out[0].draft == "EXACT HISTORICAL REPLY", "the control must not paraphrase"


# --- labelling --------------------------------------------------------------

from genius_bar import label  # noqa: E402


def test_blind_pass_gets_no_model_suggestions():
    """The blind pass is the evidence base; a suggestion would destroy it.

    Those 60 labels are the only ones that can support the judge-agreement and
    label-noise claims. If a model hint reaches them they stop being
    independent, and the agreement number silently becomes a measure of the
    model agreeing with itself.
    """
    todo = [
        {"id": 1, "message": "battery dies", "pass": "blind"},
        {"id": 2, "message": "refund me", "pass": "blind"},
    ]
    assert label._load_suggestions(todo) == {}, "blind items must never be pre-labelled"


def test_jsonl_round_trip_appends(tmp_path):
    """Progress must survive an interrupted session."""
    path = tmp_path / "g.jsonl"
    label.append_jsonl(path, {"id": 1, "intent": "data_loss"})
    label.append_jsonl(path, {"id": 2, "intent": "account_billing"})
    rows = label.read_jsonl(path)
    assert [r["id"] for r in rows] == [1, 2]
    assert label.read_jsonl(tmp_path / "missing.jsonl") == []


# --- report rendering ------------------------------------------------------

from genius_bar import eval as ev  # noqa: E402
from genius_bar import judge as judge_mod  # noqa: E402


def test_report_rows_match_header_width_when_a_system_is_unjudged():
    """The trivial baseline can be unjudged; its row must still line up.

    A row with the wrong number of cells does not raise -- markdown just
    renders a silently wrong table, which is the worst failure mode for a
    file whose whole purpose is reporting numbers honestly.
    """
    def block(judged):
        scores = {k: (3.5 if judged else None) for k in [*judge_mod.RUBRIC, "mean"]}
        return {
            "intent": {"balanced": {"accuracy": 0.5, "macro_f1": 0.4},
                       "weighted": {"accuracy": 0.5}},
            "escalation": {"balanced": {"precision": 0.5, "recall": 0.5,
                                        "missed_escalations": 1,
                                        "needless_escalations": 2,
                                        "cost_per_100": 10.0},
                           "cost_sweep": {"3:1": 1.0, "10:1": 2.0, "30:1": 3.0}},
            "reply": {"n_drafted": 5, "ungrounded_rate": 0.1, "over_limit": 0, **scores},
        }

    md = ev.build_report({
        "n": 10, "n_blind": 5, "n_assisted": 5,
        "systems": {"trivial": block(False), "agent": block(True)},
    })

    header = next(l for l in md.splitlines() if l.startswith("| system | grounded"))
    width = header.count("|")
    for line in md.splitlines():
        if line.startswith(("| trivial |", "| agent |")) and "%" in line:
            assert line.count("|") == width, f"row width {line.count('|')} != {width}: {line}"
    assert "| |" not in md, "empty cell from a desynchronised row"


def test_eval_main_runs_end_to_end(tmp_path, monkeypatch):
    """Smoke-test the whole harness with everything mocked.

    Worth its length: without it, a wiring bug in eval.main would only surface
    after someone spends two hours hand-labelling. Runs no API calls and
    touches no real data file.
    """
    import sys

    from genius_bar import agent, eval as ev, judge as judge_mod

    golden = tmp_path / "golden.jsonl"
    golden.write_text("\n".join(json.dumps(r) for r in [
        {"id": 1, "message": "battery drains since the update", "intent": "update_performance",
         "should_escalate": False, "pass": "blind", "proxy_intent": "update_performance",
         "weight": 12.0, "hard_cases": []},
        {"id": 2, "message": "unauthorized charges on my itunes account", "intent": "account_billing",
         "should_escalate": True, "pass": "blind", "proxy_intent": "account_billing",
         "weight": 1.5, "hard_cases": []},
        {"id": 3, "message": "all my contacts vanished", "intent": "data_loss",
         "should_escalate": True, "pass": "assisted", "proxy_intent": "data_loss",
         "weight": 1.8, "hard_cases": ["very_short"]},
    ]))

    class FakeRetriever:
        corpus = {"reply_clean": []}

        def __len__(self):
            return 0

        def search(self, q, k=5):
            return [{"customer": "battery dies", "reply": "Which iOS version?", "score": 0.42}]

    monkeypatch.setattr(ev, "GOLDEN", golden)
    monkeypatch.setattr(ev, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(ev, "REPLY_RATINGS", tmp_path / "none.jsonl")
    monkeypatch.setattr(ev, "RECHECK", tmp_path / "none2.jsonl")
    monkeypatch.setattr(ev, "Retriever", lambda *a, **k: FakeRetriever())

    # No API: fixed classifications, drafts and judge scores.
    monkeypatch.setattr(agent, "classify", lambda msgs, **kw: [
        {"intent": "update_performance", "confidence": 0.95, "signals": []},
        {"intent": "account_billing", "confidence": 0.9, "signals": ["payment_dispute"]},
        {"intent": "data_loss", "confidence": 0.8, "signals": []},
    ][: len(msgs)])
    monkeypatch.setattr(agent, "draft", lambda items, **kw: [
        {"draft": "Which iOS version are you on?", "used_evidence": [1]} for _ in items
    ])
    monkeypatch.setattr(judge_mod, "judge_replies", lambda items, **kw: [
        ({k: 4 for k in judge_mod.RUBRIC} | {"mean": 4.0, "worst_problem": "terse"})
        if it.get("draft", "").strip() else None
        for it in items
    ])
    monkeypatch.setattr(sys, "argv", ["eval", "--cross-family", "0"])

    ev.main()

    results = json.loads((tmp_path / "reports" / "results.json").read_text())
    assert set(results["systems"]) == {"trivial", "simple", "agent"}
    assert results["n"] == 3

    # The agent must escalate both high-risk cases and auto-handle the routine one.
    preds = [json.loads(l) for l in
             (tmp_path / "reports" / "predictions.jsonl").read_text().splitlines()]
    actions = {p["id"]: p["systems"]["agent"]["action"] for p in preds}
    assert actions == {1: "auto", 2: "escalate", 3: "escalate"}

    # Escalation recall must be perfect here, and reported both ways.
    esc = results["systems"]["agent"]["escalation"]
    assert esc["balanced"]["recall"] == 1.0
    assert "weighted" in esc and "cost_sweep" in esc

    md = (tmp_path / "reports" / "results.md").read_text()
    assert "Intent classification" in md and "Escalation" in md
