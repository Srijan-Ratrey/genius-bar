"""Grounding corpus and TF-IDF retrieval, shared by the agent and the baseline.

What "grounded" can honestly mean for this brand is narrower than the phrase
suggests, and the corpus is why. Of 19,530 customer->reply pairs:

  53.5%  are DM deflections ("send us a DM") -- no resolution in public at all
  75.9%  contain a t.co link to a support article whose CONTENT IS NOT in the
         dataset, so the actual answer is behind a URL we cannot read
   9.2%  are diagnostic questions ("which iOS version are you running?")
  10.5%  contain concrete steps

So Apple's public support record holds very few completed resolutions. What it
does hold, densely, is Apple's *first-response behaviour*: which question they
ask first, which setting they point at, when they take it private. That is what
this retriever grounds a draft in, and the report says so rather than claiming
resolutions we do not have.

Retrieval is TF-IDF rather than embeddings because the free embedding quota is
1000 texts/day. Both the agent and the simple baseline use this same retriever,
which is deliberate: holding retrieval constant makes the measured gap between
them attributable to the drafting step alone.
"""

from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from genius_bar.data import agent_replies, clean_text, load_threads

# "we've sent you a DM" -- the reply exists but the resolution is private.
DEFLECTION = re.compile(
    r"\b(?:DM|direct message)\b|send us a (?:DM|message)|meet us in", re.I
)
# Concrete guidance the customer can act on.
STEPS = re.compile(
    r"\b(?:try|tap|go to|open|settings|restart|toggle|turn (?:off|on)|"
    r"sign out|reset|update to|navigat\w+|check)\b",
    re.I,
)
# A diagnostic question. Not a resolution, but genuinely how Apple opens.
DIAGNOSTIC = re.compile(
    r"\b(?:which|what) (?:version|model|ios|macos)\b|can you (?:tell|let) us|"
    r"what happens when|are you (?:seeing|getting)",
    re.I,
)
LINK = re.compile(r"\[support link\]")

MIN_REPLY_CHARS = 40


def build_corpus() -> pd.DataFrame:
    """Customer->reply pairs that carry enough content to ground a draft.

    The filter is the single biggest bias in this project and is reported as
    such: dropping DM deflections removes over half the corpus, and those are
    disproportionately the *hard* cases -- the ones Apple judged too complex,
    sensitive or account-specific to answer in public. What remains therefore
    over-represents problems with a tidy public answer.
    """
    pairs = agent_replies(load_threads())
    reply = pairs["reply_clean"]

    has_content = (
        reply.str.contains(STEPS, na=False)
        | reply.str.contains(LINK, na=False)
        | reply.str.contains(DIAGNOSTIC, na=False)
    )
    keep = (
        has_content
        & ~reply.str.contains(DEFLECTION, na=False)
        & (reply.str.len() >= MIN_REPLY_CHARS)
        & (pairs["customer_clean"].str.len() >= 10)
    )
    cols = ["thread_id", "customer_clean", "reply_clean"]
    return pairs.loc[keep, cols].reset_index(drop=True)


class Retriever:
    """TF-IDF nearest-neighbour over historical customer messages.

    Word 1-2 grams with sublinear tf. Support tweets are short and share a tight
    vocabulary ("battery", "iOS 11", "won't charge"), which is the regime where
    TF-IDF holds up well against embeddings.
    """

    def __init__(self, corpus: pd.DataFrame | None = None):
        self.corpus = build_corpus() if corpus is None else corpus.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        self.matrix = self.vectorizer.fit_transform(self.corpus["customer_clean"])

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Top-k historical pairs, most similar first, with duplicate replies dropped.

        Duplicates are kept in the index but collapsed in the results. Apple's
        templated openers appear dozens of times, so their frequency is real
        signal for ranking -- but five copies of the same sentence is not five
        pieces of evidence, and would let one template crowd out the rest of the
        context window.
        """
        vec = self.vectorizer.transform([clean_text(query)])
        scores = (self.matrix @ vec.T).toarray().ravel()

        out: list[dict] = []
        seen: set[str] = set()
        for i in scores.argsort()[::-1]:
            if scores[i] <= 0:
                break
            reply = self.corpus.at[i, "reply_clean"]
            if reply in seen:
                continue
            seen.add(reply)
            out.append({
                "customer": self.corpus.at[i, "customer_clean"],
                "reply": reply,
                "score": float(scores[i]),
            })
            if len(out) == k:
                break
        return out


if __name__ == "__main__":
    r = Retriever()
    print(f"corpus: {len(r.corpus):,} pairs, "
          f"{r.corpus['reply_clean'].nunique():,} unique replies, "
          f"{len(r.vectorizer.vocabulary_):,} features\n")
    for q in ["my battery drains so fast since the update",
              "all my contacts are gone after updating",
              "fix the I glitch"]:
        print(f"Q: {q}")
        for hit in r.search(q, k=2):
            print(f"   [{hit['score']:.3f}] {hit['reply'][:110]}")
        print()
