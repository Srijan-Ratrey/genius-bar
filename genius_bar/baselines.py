"""Two baselines, each with a specific job.

TRIVIAL -- always predict the majority intent, always send Apple's single
most-frequent reply, never escalate. Its job is to expose how much of any
headline number is class imbalance rather than skill. It is stronger than it
sounds: the modal reply was sent 1,070 times and addresses the iOS 11 bug that
dominates this corpus, so on a naive reply-quality rubric it does not embarrass
itself. That is the point. A system that cannot clear this bar by a wide margin
has not earned its API bill.

SIMPLE -- hand-written keyword rules for intent, the nearest historical reply
copied verbatim, keyword rules for escalation. Deliberately competitive. It uses
the same retriever as the real agent, so the measured gap between them isolates
the drafting step rather than confounding it with a retrieval change.

Neither baseline needs labelled training data, which matters: the golden set is
the test set, and fitting a classifier on it would leak. The keyword rules were
written by hand from reading the corpus, and are exactly the kind of thing a
team would ship in an afternoon before reaching for an LLM.
"""

from __future__ import annotations

import functools
import re

from genius_bar.agent import Triage
from genius_bar.data import clean_text
from genius_bar.retrieve import Retriever, build_corpus

# update_performance, at 38.4% of traffic, is the majority class.
MAJORITY_INTENT = "update_performance"

# Ordered most-specific first: update_performance is a catch-all whose keywords
# ("update", "battery") appear inside messages that are really about something
# else, so it must be tried last.
INTENT_RULES: list[tuple[str, re.Pattern]] = [
    ("account_billing", re.compile(
        r"refund|billing|invoice|receipt|subscription|unauthoris|unauthoriz|"
        r"double.?charged|charged me|my card|itunes (card|credit|account)|"
        r"locked out|cancel my (subscription|plan)", re.I)),
    # Both word orders. The first draft only matched verb-then-noun ("lost my
    # contacts") and so missed "all my contacts are gone" -- an obvious phrasing
    # that any engineer writing these rules would have covered. Leaving it
    # broken would have flattered the LLM system by weakening its competition.
    ("data_loss", re.compile(
        r"(?:(?:lost|missing|deleted|gone|wiped|erased|disappear\w*)\W+(?:\w+\W+){0,6}?"
        r"(?:photos?|contacts?|music|songs|data|notes|messages|backup|library)"
        r"|(?:photos?|contacts?|music|songs|data|notes|messages|backup|library)"
        r"\W+(?:\w+\W+){0,5}?(?:lost|missing|deleted|gone|wiped|erased|disappear\w*))", re.I)),
    ("autocorrect_bug", re.compile(
        r"i⁠️|autocorrect|auto-correct|question mark|glitch"
        r"|keyboard\W+(?:\w+\W+){0,3}?(?:bug|glitch|issue)"
        r"|(?:the )?letter\s+.?i\b|[\"\']i[\"\']", re.I)),
    ("hardware_repair", re.compile(
        r"genius bar|warranty|apple ?care|repair|replac(e|ed|ement)|"
        r"cracked screen|screen (is )?crack|water damage", re.I)),
    ("device_crash_reboot", re.compile(
        r"keeps? restarting|restart(s|ing)? (itself|randomly|constantly)|"
        r"reboot|shut(s|ting)? (down|off) (randomly|by itself)|"
        r"won'?t turn on|overheat|boiling hot", re.I)),
    ("app_or_service_issue", re.compile(
        r"app ?store|icloud|apple music|itunes|safari|imessage|facetime|"
        r"apple pay|siri|mail app|sync(ing)?|won'?t (open|load|download|install)", re.I)),
    ("update_performance", re.compile(
        r"ios ?11|update|battery|drain|freez|slow|lag|charge", re.I)),
]

# What a team would write before building anything cleverer.
ESCALATE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"refund|unauthoris|unauthoriz|charged me|double.?charged|"
                r"my card|fraud", re.I), "keyword rule: payment or refund"),
    (re.compile(r"lawyer|sue|legal|attorney|consumer rights|ombudsman|"
                r"trading standards|press|journalist", re.I), "keyword rule: legal or press"),
    (re.compile(r"overheat|boiling|burn(ed|t|ing)?|caught fire|swollen|"
                r"exploded", re.I), "keyword rule: safety"),
    (re.compile(r"(?:(?:lost|deleted|wiped|gone|missing)\W+(?:\w+\W+){0,6}?"
                r"(?:photos?|contacts?|data|backup)"
                r"|(?:photos?|contacts?|data|backup)\W+(?:\w+\W+){0,5}?"
                r"(?:lost|deleted|wiped|gone|missing))", re.I), "keyword rule: data loss"),
    (re.compile(r"third time|3rd time|as i (said|mentioned)|already (told|asked|tried)|"
                r"still waiting|no (one|response)", re.I), "keyword rule: repeat contact"),
]


@functools.lru_cache(maxsize=1)
def canned_reply() -> str:
    """Apple's single most-frequent reply, taken from the data rather than invented."""
    return build_corpus()["reply_clean"].mode().iat[0]


def trivial(messages: list[str]) -> list[Triage]:
    """Majority intent, modal reply, never escalate."""
    reply = canned_reply()
    return [
        Triage(
            message=m,
            intent=MAJORITY_INTENT,
            confidence=1.0,  # asserted, not calibrated -- that is the baseline's flaw
            signals=[],
            action="auto",
            reason="trivial baseline: never escalates",
            grounded=False,
            draft=reply,
            evidence=[],
        )
        for m in messages
    ]


def classify_by_rules(message: str, fallback: str | None = MAJORITY_INTENT) -> str | None:
    """First matching rule wins; unmatched messages fall back to `fallback`.

    As a CLASSIFIER the fallback is the majority class -- the charitable choice,
    maximising the baseline's accuracy on an imbalanced set so the comparison is
    against the baseline at its best.

    As a STRATIFICATION proxy that fallback is actively harmful: it labels every
    unmatched message update_performance, which both inflates that stratum to
    79% (against a true ~38%) and makes not_actionable unreachable, since no
    message ever falls into it. Sampling therefore passes fallback=None and
    treats "unmatched" as its own stratum.
    """
    text = clean_text(message)
    for intent, pattern in INTENT_RULES:
        if pattern.search(text):
            return intent
    return fallback


def escalate_by_rules(message: str, intent: str | None = None) -> tuple[str, str]:
    """Keyword rules first, then the risk level of the intent it just predicted.

    Giving the baseline the same intent-risk rule the agent uses is deliberate.
    Without it the baseline would auto-handle every hardware_repair message,
    and the escalation comparison would measure a feature the baseline simply
    lacks. With it, both sides escalate high-risk intents and the remaining
    difference is what it should be: whether LLM-read risk signals beat
    keyword matching.
    """
    for pattern, why in ESCALATE_RULES:
        if pattern.search(message):
            return "escalate", why

    if intent is not None:
        from genius_bar.agent import load_intents

        if load_intents().get(intent, {}).get("risk") == "high":
            return "escalate", f"high-risk intent '{intent}' by policy"

    return "auto", "no escalation keyword matched"


def simple(messages: list[str], retriever: Retriever | None = None, k: int = 5) -> list[Triage]:
    """Keyword intent, verbatim nearest historical reply, keyword escalation."""
    retriever = retriever or Retriever()
    out = []
    for m in messages:
        evidence = retriever.search(m, k=k)
        intent = classify_by_rules(m)
        action, reason = escalate_by_rules(m, intent)
        # Copied verbatim -- no synthesis. This is the control that isolates
        # what the LLM's drafting step actually adds.
        reply = evidence[0]["reply"] if evidence else canned_reply()
        out.append(Triage(
            message=m,
            intent=intent,
            confidence=1.0,
            signals=[],
            action=action,
            reason=f"simple baseline: {reason}",
            grounded=bool(evidence),
            draft=reply if action == "auto" else "",
            evidence=evidence,
            used_evidence=[1] if evidence and action == "auto" else [],
        ))
    return out


if __name__ == "__main__":
    msgs = [
        "the new update is draining my battery like crazy",
        "over $90 in unauthorized charges from iTunes",
        "fix the I glitch please",
        "my iphone is boiling hot",
        "all my contacts are gone after updating",
    ]
    print(f"canned reply: {canned_reply()[:100]}\n")
    r = Retriever()
    for t, s in zip(trivial(msgs), simple(msgs, r)):
        print(f"MSG      {t.message[:70]}")
        print(f"  trivial {t.intent:22} {t.action}")
        print(f"  simple  {s.intent:22} {s.action:8} <- {s.reason}")
        print(f"          copies: {s.draft[:85] or '(escalated)'}")
