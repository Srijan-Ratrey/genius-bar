"""Draw the golden evaluation set, stratified, and write it out unlabelled.

THE STRATIFICATION PROBLEM, AND HOW IT IS HANDLED

Stratifying by intent needs intent labels, which is exactly what the golden set
is being built to create. Three ways out, none free:

  pure random      -- data_loss (0.7%) and account_billing (0.9%) get one
                      example each, so per-class F1 and the entire escalation
                      evaluation become meaningless.
  stratify by LLM  -- enriches the set where the LLM is already confident,
                      flattering the system under test. Unacceptable.
  stratify by RULE -- enriches the set where hand-written keyword rules match,
                      which flatters the SIMPLE BASELINE.

The rule proxy is used, because its bias runs *against* the system being sold.
If the LLM agent beats a baseline that the sampling itself favoured, the result
is stronger than the raw number suggests, not weaker. The alternative would
have quietly tilted the comparison the other way.

Because strata are known, natural-distribution metrics are recoverable by
importance weighting (`weight` on each record), so both the balanced and the
production-distribution numbers get reported.

Usage: uv run python scripts/sample_golden.py [--n 180] [--blind 60]
"""

from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from genius_bar.agent import load_intents
from genius_bar.baselines import classify_by_rules
from genius_bar.data import REPO_ROOT, first_inbound, load_threads

OUT_PATH = REPO_ROOT / "data" / "golden_unlabelled.jsonl"

PER_INTENT_FLOOR = 10   # below this, per-class F1 is noise
PER_INTENT_CAP = 27     # ~15% of 180; stops update_performance dominating

# Hard cases, over-sampled on purpose. A golden set of only clean cases measures
# nothing interesting -- these are where the system is expected to struggle and
# where the escalation policy has to earn its keep.
NON_ENGLISH_WORDS = re.compile(
    r"\b(je|j'ai|que|qué|não|nao|para|pero|el|las|und|der|ich|nicht|bir|ben|"
    r"olarak|ile|मेरा|هذا|على|في)\b", re.I
)
NON_LATIN = re.compile(r"[؀-ۿऀ-ॿЀ-ӿ一-鿿぀-ヿ]")
MIN_HARD_CASES = {"non_english": 6, "image_only": 6, "very_short": 6}

# Messages whose rules match nothing. Roughly half the corpus, and the pool
# that not_actionable lives in -- along with genuine misses, which is precisely
# what a human labeller is for.
UNMATCHED = "unmatched"


def tag_hard_cases(text: str, raw: str = "") -> list[str]:
    """Tagged from the RAW text where possible.

    image_only has to read the raw message: clean_text drops the t.co link, so
    a screenshot-plus-"fix this" complaint looks like a plain short message
    afterwards. These are common in the autocorrect cluster and are a genuine
    hard case -- the actual content is in an image nothing here can read.
    """
    tags = []
    if raw and re.search(r"https?://(?:t\.co|pic\.twitter)", raw):
        if len(re.sub(r"https?://\S+", "", raw).strip()) < 45:
            tags.append("image_only")
    if NON_LATIN.search(text) or len(NON_ENGLISH_WORDS.findall(text)) >= 2:
        tags.append("non_english")
    if len(text) < 40:
        tags.append("very_short")
    if len(text) > 240:
        tags.append("very_long")
    return tags


def allocate(shares: dict[str, int], total: int) -> dict[str, int]:
    """Floor every stratum, then fill round-robin up to the cap.

    Round-robin rather than proportional-to-volume. An earlier version always
    gave the next seat to the stratum with the most available messages, which
    left data_loss and account_billing at the floor of 10 while the equally
    rare device_crash_reboot reached 25 -- an arbitrary split between three
    classes that are all rare AND all high-risk escalation cases.

    An even split maximises per-class reliability where it matters most, and
    natural-distribution metrics are recovered afterwards from `weight`, so
    nothing is lost by not sampling proportionally here.
    """
    names = [n for n in shares if shares[n] > 0]
    available = {n: shares[n] for n in names}
    alloc = {n: min(PER_INTENT_FLOOR, available[n]) for n in names}

    while sum(alloc.values()) < total:
        room = [n for n in names if alloc[n] < min(PER_INTENT_CAP, available[n])]
        if not room:
            break
        for n in room:
            if sum(alloc.values()) >= total:
                break
            alloc[n] += 1
    return alloc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=180)
    ap.add_argument("--blind", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    first = first_inbound(load_threads()).copy()
    # fallback=None: unmatched messages must stay visible as their own stratum
    first["proxy_intent"] = first["clean"].map(
        lambda t: classify_by_rules(t, fallback=None) or UNMATCHED
    )
    first["hard_cases"] = [
        tag_hard_cases(c, r) for c, r in zip(first["clean"], first["text"])
    ]

    counts = first["proxy_intent"].value_counts().to_dict()
    strata = list(load_intents()) + [UNMATCHED]
    shares = {name: counts.get(name, 0) for name in strata}
    alloc = allocate(shares, args.n)

    print(f"{'stratum':22} {'available':>10} {'natural':>8} {'allocated':>10}")
    for name in strata:
        nat = shares[name] / len(first)
        print(f"{name:22} {shares[name]:10,} {nat:8.1%} {alloc.get(name, 0):10}")
    print(f"{'TOTAL':22} {len(first):10,} {'':8} {sum(alloc.values()):10}")
    print(f"\nnot_actionable has no rule of its own, so it has no stratum: its\n"
          f"examples come from '{UNMATCHED}', which the human labeller resolves.")

    picked = []
    for name, k in alloc.items():
        pool = first[first["proxy_intent"] == name]
        if not len(pool):
            continue
        picked.append(pool.sample(n=min(k, len(pool)), random_state=args.seed))
    golden = pd.concat(picked, ignore_index=True)

    # Top up under-represented hard cases by swapping, not appending, so the
    # total stays at n.
    for tag, want in MIN_HARD_CASES.items():
        have = golden["hard_cases"].map(lambda t: tag in t).sum()
        if have >= want:
            continue
        pool = first[
            first["hard_cases"].map(lambda t: tag in t)
            & ~first["tweet_id"].isin(golden["tweet_id"])
        ]
        extra = pool.sample(n=min(want - have, len(pool)), random_state=args.seed)
        if not len(extra):
            continue
        # Drop from the largest stratum, which has examples to spare.
        biggest = golden["proxy_intent"].value_counts().idxmax()
        drop = golden[golden["proxy_intent"] == biggest].sample(
            n=min(len(extra), (golden["proxy_intent"] == biggest).sum()),
            random_state=args.seed,
        )
        golden = pd.concat(
            [golden.drop(index=drop.index), extra], ignore_index=True
        )
        print(f"topped up {tag}: +{len(extra)} (swapped out of {biggest})")

    golden = golden.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    blind = set(golden.index[: args.blind])

    # Importance weight: how many real messages each golden example stands for.
    # Lets natural-distribution metrics be recovered from a balanced sample.
    #
    # Computed from the FINAL counts, not from `alloc`. The hard-case top-up
    # swaps examples between strata after allocation, so alloc is stale by then
    # -- using it left the two swapped strata ~18% wrong and made the weights
    # sum to more messages than the corpus contains.
    final_counts = golden["proxy_intent"].value_counts().to_dict()
    weights = {
        name: shares[name] / max(final_counts.get(name, 1), 1) for name in strata
    }

    with OUT_PATH.open("w") as fh:
        for i, row in golden.iterrows():
            fh.write(json.dumps({
                "id": int(row["tweet_id"]),
                "thread_id": int(row["thread_id"]),
                "message": row["clean"],
                "proxy_intent": row["proxy_intent"],
                "hard_cases": row["hard_cases"],
                "weight": round(weights[row["proxy_intent"]], 3),
                "pass": "blind" if i in blind else "assisted",
                # left for the human: intent, should_escalate, notes
            }, ensure_ascii=False) + "\n")

    print(f"\nwrote {OUT_PATH.relative_to(REPO_ROOT)}: {len(golden)} examples "
          f"({len(blind)} blind, {len(golden) - len(blind)} assisted)")
    hard = pd.Series([t for tags in golden["hard_cases"] for t in tags]).value_counts()
    print(f"hard cases: {hard.to_dict()}")
    print("\nNext: uv run python -m genius_bar.label")


if __name__ == "__main__":
    main()
