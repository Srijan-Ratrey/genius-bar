"""The agent: classify, retrieve, draft, then decide auto-handle vs escalate.

Structure worth noting before reading the code.

Classification and risk-signal extraction happen in ONE model call, because
they are the same act of reading. Drafting is a second call, because it needs
the retrieved evidence. Two calls per message, both batchable.

The escalation DECISION is not a model call at all. The model reports signals;
`decide` is ordinary Python that maps signals to an action and a reason. Three
reasons for that:

- It is auditable. Every escalation traces to a named rule, which is what the
  assignment's "with a stated reason" requirement actually needs.
- It is testable as a truth table, so a change in policy is a visible diff
  rather than a prompt edit with unknown blast radius.
- The costs are wildly asymmetric. Auto-answering a payment dispute is an
  incident; a needless escalation costs an agent two minutes. That tradeoff is
  a business decision and belongs in code a human can read and argue with, not
  buried in a model's judgement.

Rule ORDER matters: the first matching rule supplies the reason, so rules run
most-severe first and the stated reason is the most serious applicable one.
"""

from __future__ import annotations

import functools
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from genius_bar import llm
from genius_bar.data import REPO_ROOT, clean_text
from genius_bar.retrieve import Retriever

INTENTS_PATH = REPO_ROOT / "data" / "intents.yaml"

# Below this the classifier is guessing; a human should look.
MIN_CONFIDENCE = 0.60
# Below this, TF-IDF found nothing genuinely similar, so a "grounded" draft
# would be grounded in noise. Calibrated against observed scores: an exact
# match scores 1.0, a good topical match ~0.3-0.45, junk below ~0.15.
MIN_EVIDENCE_SCORE = 0.15
TWEET_LIMIT = 280

SIGNALS = {
    "legal_or_press_threat": "threatens legal action, regulators, or media",
    "safety_risk": "device overheating, burning, swelling, or physical harm",
    "payment_dispute": "disputed, unauthorised or duplicate charge; demands a refund",
    "irreversible_data_loss": "data appears permanently lost, with no backup",
    "repeat_contact": "says they have already asked, or already tried the usual advice",
    "non_english": "written mainly in a language other than English",
    "multi_intent": "contains two or more unrelated requests",
    "severe_anger": "sustained abuse or profanity directed at the brand",
}

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "confidence": {"type": "number", "description": "0.0-1.0"},
        "signals": {
            "type": "array",
            "items": {"type": "string", "enum": list(SIGNALS)},
            "description": "only signals genuinely present",
        },
    },
    "required": ["intent", "confidence", "signals"],
}

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "draft": {"type": "string"},
        "used_evidence": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "1-based indices of the examples actually relied on",
        },
    },
    "required": ["draft", "used_evidence"],
}


@dataclass
class Triage:
    """One message's full result. Serialisable so eval can diff runs."""

    message: str
    intent: str
    confidence: float
    signals: list[str]
    action: str  # "auto" | "escalate"
    reason: str
    grounded: bool = False
    draft: str = ""
    evidence: list[dict] = field(default_factory=list)
    used_evidence: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@functools.lru_cache(maxsize=1)
def load_intents() -> dict[str, dict]:
    entries = yaml.safe_load(INTENTS_PATH.read_text())
    return {e["name"]: e for e in entries}


# --- step 1: classify + read risk signals ----------------------------------


def _classify_prompt(batch: list[str]) -> str:
    intents = load_intents()
    catalogue = "\n".join(
        f"- {name}: {meta['description'].strip()}" for name, meta in intents.items()
    )
    signal_list = "\n".join(f"- {k}: {v}" for k, v in SIGNALS.items())
    messages = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(batch))

    return f"""You are triaging opening messages sent to Apple's support account on Twitter.

For each message return the single best intent, a calibrated confidence, and any
risk signals genuinely present.

INTENTS (use exactly one of these names):
{catalogue}

RISK SIGNALS (include only those actually present; an empty list is normal):
{signal_list}

Guidance:
- Judge what the customer NEEDS, not how politely they said it. An furious
  "iOS 11 RUINED MY PHONE FIX IT" is still update_performance; the anger is a
  separate signal, not a different intent.
- not_actionable means there is nothing to answer -- venting, jokes, feature
  requests. Do not use it merely because the message is rude.
- confidence should be genuinely calibrated. These categories overlap and many
  messages are ambiguous; report ~0.5 when torn rather than defaulting high.
  Low confidence is useful information, not a failure.

Return one object per message, in order.

MESSAGES:
{messages}
"""


def classify(messages: list[str], batch_size: int = 30, model: str = llm.PRIMARY) -> list[dict]:
    """Intent, confidence and risk signals for each message."""
    raw = llm.map_batched(
        [clean_text(m) for m in messages],
        _classify_prompt,
        CLASSIFY_SCHEMA,
        batch_size=batch_size,
        model=model,
    )
    valid = set(load_intents())
    out = []
    for item in raw:
        if not item:
            # A failed call must not silently become a confident prediction.
            out.append({"intent": "not_actionable", "confidence": 0.0, "signals": []})
            continue
        intent = item.get("intent", "")
        out.append({
            # An off-taxonomy label is a real failure mode; surface it as zero
            # confidence so the policy escalates rather than trusting it.
            "intent": intent if intent in valid else "not_actionable",
            "confidence": 0.0 if intent not in valid else float(item.get("confidence", 0.0)),
            "signals": [s for s in item.get("signals", []) if s in SIGNALS],
        })
    return out


# --- step 2: the escalation policy (no model involved) ---------------------


def decide(intent: str, confidence: float, signals: list[str], top_score: float) -> tuple[str, str]:
    """Map triage output to (action, reason). Pure function, ordered most-severe first."""
    intents = load_intents()
    meta = intents.get(intent, {})
    sig = set(signals)

    if "safety_risk" in sig:
        return "escalate", "safety risk reported (overheating or physical harm)"
    if "legal_or_press_threat" in sig:
        return "escalate", "customer threatened legal, regulatory or media action"
    if "payment_dispute" in sig:
        return "escalate", "disputed charge -- money must not be handled automatically"
    if "irreversible_data_loss" in sig:
        return "escalate", "possible permanent data loss; recovery depends on backup state"
    if "non_english" in sig:
        return "escalate", "not in English; this agent drafts English replies only"

    # Defence in depth. classify() already maps off-taxonomy labels to zero
    # confidence, but decide() is the safety-critical function and must not
    # depend on its caller having sanitised the input: an unknown intent has no
    # risk level, so every risk-based rule below would silently pass it.
    if intent not in intents:
        return "escalate", f"unrecognised intent '{intent}'; not in the taxonomy"

    if meta.get("risk") == "high":
        return "escalate", f"intent '{intent}' is high-risk by policy"
    if "repeat_contact" in sig:
        return "escalate", "customer has already asked; the standard reply has failed once"
    if confidence < MIN_CONFIDENCE:
        return "escalate", f"low classifier confidence ({confidence:.2f} < {MIN_CONFIDENCE})"
    if top_score < MIN_EVIDENCE_SCORE:
        return "escalate", (
            f"no similar precedent found (best match {top_score:.2f} < "
            f"{MIN_EVIDENCE_SCORE}); a draft would not be grounded"
        )
    if "multi_intent" in sig:
        return "escalate", "multiple unrelated requests in one message"

    return "auto", f"routine '{intent}', confident and grounded in precedent"


# --- step 3: draft a reply from retrieved precedent ------------------------


def _draft_prompt(batch: list[tuple[str, str, list[dict]]]) -> str:
    intents = load_intents()
    blocks = []
    for n, (message, intent, evidence) in enumerate(batch, 1):
        meta = intents.get(intent, {})
        examples = "\n".join(
            f"    {i}. customer: {e['customer'][:200]}\n"
            f"       Apple replied: {e['reply'][:250]}"
            for i, e in enumerate(evidence, 1)
        ) or "    (no similar precedent found)"
        blocks.append(
            f"--- MESSAGE {n} ---\n"
            f"customer: {message}\n"
            f"intent: {intent}\n"
            f"how Apple usually handles this: {meta.get('resolution_hint', '').strip()}\n"
            f"precedent:\n{examples}"
        )
    body = "\n\n".join(blocks)

    return f"""Draft Apple's next public reply to each customer message below.

Ground every draft in the precedent shown. That precedent is what Apple's
support account ACTUALLY does, which is usually to ask one diagnostic question
or point at one setting -- not to deliver a complete fix. Match that behaviour;
do not invent a more helpful Apple than the evidence supports.

Rules:
- Under {TWEET_LIMIT} characters. This is a public tweet.
- Never invent specifics absent from the precedent: no version numbers, article
  URLs, timelines, refunds or promises of a fix.
- "[support link]" in the precedent means Apple linked an article whose content
  is not available here. You may say you will share a link; never fabricate one.
- If the precedent shows Apple asking a diagnostic question first, ask it.
- For intent not_actionable: acknowledge briefly and do not troubleshoot. Never
  draft steps for a problem the customer did not report.
- Match Apple's register: calm, plain, first person plural ("we"). Do not
  mirror the customer's profanity or match their intensity.
- used_evidence: 1-based indices of the examples you actually relied on. Empty
  if none applied.

Return one object per message, in order.

{body}
"""


def draft(
    items: list[tuple[str, str, list[dict]]], batch_size: int = 10, model: str = llm.PRIMARY
) -> list[dict]:
    """Draft replies for (message, intent, evidence) triples."""
    raw = llm.map_batched(items, _draft_prompt, DRAFT_SCHEMA, batch_size=batch_size, model=model)
    return [r or {"draft": "", "used_evidence": []} for r in raw]


# --- the pipeline ----------------------------------------------------------


def triage(
    messages: list[str],
    retriever: Retriever | None = None,
    k: int = 5,
    model: str = llm.PRIMARY,
) -> list[Triage]:
    """Full pipeline over a list of opening customer messages.

    Drafts are produced only for messages the policy auto-handles. Escalated
    messages go to a human with the reason attached, so drafting one would spend
    quota on text nobody sends -- and would invite an agent to paste a reply the
    policy just judged unsafe to send.
    """
    retriever = retriever or Retriever()
    classified = classify(messages, model=model)

    results: list[Triage] = []
    for message, c in zip(messages, classified):
        evidence = retriever.search(message, k=k)
        top = evidence[0]["score"] if evidence else 0.0
        action, reason = decide(c["intent"], c["confidence"], c["signals"], top)
        results.append(Triage(
            message=message,
            intent=c["intent"],
            confidence=c["confidence"],
            signals=c["signals"],
            action=action,
            reason=reason,
            evidence=evidence,
        ))

    auto = [i for i, r in enumerate(results) if r.action == "auto"]
    if auto:
        drafted = draft(
            [(results[i].message, results[i].intent, results[i].evidence) for i in auto],
            model=model,
        )
        for i, d in zip(auto, drafted):
            r = results[i]
            r.draft = d["draft"]
            r.used_evidence = d.get("used_evidence", [])
            r.grounded = bool(r.used_evidence)

            # MIN_EVIDENCE_SCORE only checks that similar precedent EXISTS. It
            # cannot tell whether the draft actually used any, and in practice
            # the model sometimes writes a fluent generic reply while ignoring
            # every example. For a system whose central claim is groundedness,
            # a support-intent draft citing no precedent is exactly the case a
            # human should see. not_actionable is exempt: acknowledging venting
            # without troubleshooting is correct behaviour, not a failure.
            if not r.grounded and r.intent != "not_actionable":
                r.action = "escalate"
                r.reason = "draft cited no precedent; not grounded despite available evidence"

    return results
