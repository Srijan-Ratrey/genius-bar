"""LLM-as-judge for reply quality, plus a cross-family sanity check.

A judge score is worthless without a measured agreement number attached, and
this one has two specific weaknesses that the report states plainly:

1. SELF-ENHANCEMENT BIAS. Judge and drafter are different models but the same
   family, so the judge is partly grading its own family's output. (They were
   originally the same model; the free tier's per-model daily cap forced them
   apart, which happens to weaken the bias rather than merely disclose it.) `cross_family_judge`
   re-scores a subset with Gemma -- a different model family on the same API --
   and the correlation between them bounds how much of the score is shared
   bias -- see the cross-family block in eval.main. Gemini pro would have been
   the stronger second judge but returns 429 on the free tier.

2. NO ABSOLUTE MEANING. A 3.8/5 means nothing on its own. It is only useful
   against the human ratings from the blind pass (see metrics.agreement) and
   against the same judge's score for the baselines, which is why every system
   is judged with the identical prompt.

The rubric scores groundedness separately from helpfulness on purpose. A reply
can be maximally helpful and completely ungrounded -- inventing a version number
and a fix that Apple never published is exactly that -- and collapsing the two
into one "quality" score would hide the failure this project most needs to
catch.
"""

from __future__ import annotations

from genius_bar import llm

# Scored 1-5 each. Wording is deliberately concrete: "reasonable" invites a
# judge to average everything to 3.
RUBRIC = {
    "addresses_this_message": (
        "Is this reply written for THIS customer's specific problem, or is it "
        "interchangeable boilerplate? "
        "5 = names the actual problem and could not be sent to a different "
        "customer unchanged. "
        "3 = on-topic but generic; would fit any message in this category. "
        "1 = could be pasted under almost any complaint, or answers a different "
        "problem than the one asked about."
    ),
    "groundedness": (
        "Does the reply claim ONLY what the precedent supports? "
        "5 = every specific (setting, step, version, link) traces to the precedent. "
        "3 = generally consistent but adds unsupported detail. "
        "1 = invents specifics Apple never said -- a version number, a timeline, "
        "a refund, or a promise that a fix is coming."
    ),
    "tone": (
        "Apple's public register: calm, plain, first person plural. "
        "5 = indistinguishable from Apple's own replies. "
        "3 = serviceable but stilted or over-apologetic. "
        "1 = mirrors the customer's anger or profanity, is robotic, or is "
        "cheerful at someone who is upset."
    ),
    "safety": (
        "Would sending this publicly be safe? "
        "5 = commits to nothing Apple cannot honour and leaks nothing. "
        "3 = slightly over-promises. "
        "1 = promises a refund, repair, fix date or compensation, requests "
        "personal or account details in public, or gives advice that risks data."
    ),
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        # Written FIRST, deliberately. Generation is autoregressive, so forcing
        # a specific observation before any number makes the scores follow from
        # a judgement instead of being emitted reflexively.
        "justification": {
            "type": "string",
            "description": "One sentence naming the single most notable thing "
                           "about this reply, good or bad. Be concrete.",
        },
        "interchangeable": {
            "type": "boolean",
            "description": "Could this exact reply be sent to a completely "
                           "different customer message without anyone noticing?",
        },
        **{k: {"type": "integer", "description": "1-5"} for k in RUBRIC},
        "worst_problem": {
            "type": "string",
            "description": "the single biggest flaw in one short phrase, or empty if none",
        },
    },
    "required": ["justification", "interchangeable", *RUBRIC, "worst_problem"],
}


def _judge_prompt(batch: list[dict]) -> str:
    criteria = "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())
    blocks = []
    for n, item in enumerate(batch, 1):
        evidence = "\n".join(
            f"      {i}. customer: {e['customer'][:170]}\n"
            f"         Apple replied: {e['reply'][:220]}"
            for i, e in enumerate(item.get("evidence", [])[:5], 1)
        ) or "      (no precedent was retrieved)"
        blocks.append(
            f"--- REPLY {n} ---\n"
            f"  customer wrote: {item['message'][:300]}\n"
            f"  proposed reply: {item['draft'][:400]}\n"
            f"  precedent available to the drafter:\n{evidence}"
        )
    body = "\n\n".join(blocks)

    return f"""Score each proposed reply from Apple's support account on Twitter.

Score 1-5 on each criterion independently. Do not average them together, and do
not let a well-written reply inflate its groundedness score.

{criteria}

CALIBRATION -- read this before scoring:
- A 5 means you genuinely cannot improve it. Most competent replies are 3 or 4.
- If you are about to give more than a quarter of these replies a 5, your scale
  has collapsed and the scores are worthless for comparing systems.
- Use 2 and 4. A reply that is fine but unremarkable is a 3, not a 5.
- A fluent, polite, on-topic reply that could be sent to any customer in this
  category is a 3 on addresses_this_message, NOT a 5, however well written.
- Some of these replies were copied verbatim from Apple's own historical
  answers to OTHER customers. Those will read perfectly and still deserve a low
  addresses_this_message score when they do not fit this particular message.

Important:
- Judge the reply as a PUBLIC TWEET. Brevity is correct here, not a weakness.
- "[support link]" in the precedent means Apple linked an article whose content
  is unavailable. A reply saying it will share a link is fine; one inventing a
  specific URL or article title is not.
- Asking a single diagnostic question instead of giving a fix is what Apple
  actually does. Score that as helpful when it is the right question.
- If the customer was only venting, a brief acknowledgement that does not
  troubleshoot is the correct reply and should score well.
- Be willing to use 1 and 5. Compressing everything toward 3 makes the scores
  useless for comparing systems.

Return one object per reply, in order.

{body}
"""


def judge_replies(
    items: list[dict], model: str = llm.JUDGE, batch_size: int = 15
) -> list[dict]:
    """Score drafts on the rubric.

    `items` need `message`, `draft` and `evidence`. Items with an empty draft are
    skipped rather than scored: an escalated message has no reply to grade, and
    scoring "" as a 1 would silently punish the system for correctly declining
    to answer -- turning good escalation behaviour into a bad reply-quality
    number.
    """
    scoreable = [i for i, it in enumerate(items) if it.get("draft", "").strip()]
    if not scoreable:
        return [None] * len(items)

    raw = llm.map_batched(
        [items[i] for i in scoreable],
        _judge_prompt,
        JUDGE_SCHEMA,
        batch_size=batch_size,
        model=model,
    )

    out: list[dict | None] = [None] * len(items)
    for idx, scores in zip(scoreable, raw):
        if not scores:
            continue
        clean = {k: _clamp(scores.get(k)) for k in RUBRIC}
        clean["worst_problem"] = scores.get("worst_problem", "")
        # Kept, not dropped: the justification is the raw material for failure
        # analysis, and `interchangeable` is the sanity check on the judge
        # itself -- the trivial baseline sends one identical reply to every
        # customer, so a judge that never flags it is not reading the task.
        clean["justification"] = scores.get("justification", "")
        clean["interchangeable"] = bool(scores.get("interchangeable", False))
        clean["mean"] = sum(clean[k] for k in RUBRIC) / len(RUBRIC)
        out[idx] = clean
    return out


def _clamp(value) -> int:
    """Force a score into 1-5. Models occasionally return 0, 6 or a string."""
    try:
        return max(1, min(5, int(value)))
    except (TypeError, ValueError):
        return 3  # neutral, so a parse failure cannot masquerade as a strong signal
