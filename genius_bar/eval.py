"""Evaluation harness. `make eval` runs this.

Reports every headline number for all three systems, and reports each one twice:

  balanced  -- over the golden set as sampled, which over-represents rare
               high-stakes classes so per-class figures mean something
  weighted  -- importance-weighted back to production distribution

Both, always. The balanced number describes a distribution no real traffic has;
the weighted one is dominated by the majority class. Quoting either alone is
the single easiest way to mislead with this project, so neither is quoted alone.

Intent metrics are additionally broken out over the 60 BLIND labels, which were
made with no model output visible. Those are the only labels that can support a
claim about agreement, so they are never blended into the headline.

This module must run with no API key when the cache is warm -- that is what the
15-minute reproduce depends on. A cache miss raises CacheMiss with instructions
rather than quietly scoring an empty response.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from genius_bar import judge as judge_mod
from genius_bar import llm, metrics
from genius_bar.agent import Triage, load_intents, triage
from genius_bar.baselines import simple, trivial
from genius_bar.data import REPO_ROOT
from genius_bar.retrieve import Retriever

def _rel(path: Path) -> str:
    """Path relative to the repo for display. relative_to() raises when the
    path sits outside REPO_ROOT; relpath never does."""
    return os.path.relpath(path, REPO_ROOT)


GOLDEN = REPO_ROOT / "data" / "golden.jsonl"
RECHECK = REPO_ROOT / "data" / "golden_recheck.jsonl"
REPLY_RATINGS = REPO_ROOT / "data" / "reply_ratings.jsonl"
REPORTS = REPO_ROOT / "reports"

# The 10:1 default is a judgement call, so results are shown across a range.
COST_RATIOS = [3.0, 10.0, 30.0]

# Free-tier ceiling, per model, per day.
FREE_TIER_RPD = 20


def load_golden(path: Path | None = None) -> list[dict]:
    # Resolved at call time, not bound as a default: a default argument captures
    # the module constant at import, so the path could not be overridden.
    path = path or GOLDEN
    if not path.exists():
        raise SystemExit(
            f"{_rel(path)} not found.\n"
            "The golden set is hand-labelled and cannot be generated. Run:\n"
            "  uv run python scripts/sample_golden.py    # once, to draw 180\n"
            "  uv run python -m genius_bar.label         # to label them"
        )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path.name} is empty -- nothing labelled yet.")
    return rows


def preflight(n_golden: int, judge_sample: int) -> None:
    """Print the request cost per model before spending any of it.

    The free tier allows 20 generate requests per day PER MODEL, so the only
    way a full run fits in a day is by putting each stage on its own model.
    Printing this up front turns "it died halfway" into a decision made in
    advance. Anything already cached costs nothing, so reruns shrink.
    """
    est = {
        llm.PRIMARY: -(-n_golden // 30) + -(-(n_golden // 2) // 10),  # classify + draft
        llm.JUDGE: -(-(judge_sample * 2) // 15),                      # ~2 drafts per example
        llm.ALT_JUDGE: -(-min(40, judge_sample) // 10),               # cross-family
    }
    print("\nestimated live requests (cached prompts cost nothing):")
    for model, n in est.items():
        flag = "" if n <= FREE_TIER_RPD else f"  <-- OVER the {FREE_TIER_RPD}/day free cap"
        print(f"  {model:24} ~{n:3}{flag}")
    print()


def run_systems(golden: list[dict], retriever: Retriever) -> dict[str, list[Triage]]:
    messages = [g["message"] for g in golden]
    return {
        "trivial": trivial(messages),
        "simple": simple(messages, retriever),
        "agent": triage(messages, retriever),
    }


def _intent_block(golden: list[dict], preds: list[Triage]) -> dict:
    truth = [g["intent"] for g in golden]
    pred = [t.intent for t in preds]
    weights = [g.get("weight", 1.0) for g in golden]

    blind = [i for i, g in enumerate(golden) if g.get("pass") == "blind"]
    out = {
        "balanced": metrics.intent_metrics(truth, pred),
        "weighted": metrics.intent_metrics(truth, pred, sample_weight=weights),
    }
    if len(blind) >= 20:
        out["blind_only"] = metrics.intent_metrics(
            [truth[i] for i in blind], [pred[i] for i in blind]
        )
    return out


def _escalation_block(golden: list[dict], preds: list[Triage]) -> dict:
    truth = [bool(g["should_escalate"]) for g in golden]
    pred = [t.action == "escalate" for t in preds]
    weights = [g.get("weight", 1.0) for g in golden]

    out = {
        "balanced": metrics.escalation_metrics(truth, pred),
        "weighted": metrics.escalation_metrics(truth, pred, sample_weight=weights),
        "cost_sweep": {
            f"{r:g}:1": metrics.escalation_metrics(truth, pred, cost_missed=r)["cost_per_100"]
            for r in COST_RATIOS
        },
    }
    # Which intents the misses concentrate in -- the actionable part of a miss count.
    missed = [golden[i]["intent"] for i in range(len(truth)) if truth[i] and not pred[i]]
    out["missed_by_intent"] = {k: missed.count(k) for k in sorted(set(missed))}
    return out


def _reply_block(preds: list[Triage], scores: list[dict | None]) -> dict:
    scored = [s for s in scores if s]
    drafted = [t for t in preds if t.draft.strip()]
    block = {
        "n_drafted": len(drafted),
        "n_judged": len(scored),
        # Self-reported by the drafter: a support draft citing no precedent.
        # Tracked because groundedness is this project's central claim.
        "ungrounded_rate": (
            sum(1 for t in drafted if not t.used_evidence) / len(drafted) if drafted else 0.0
        ),
        "over_limit": sum(1 for t in drafted if len(t.draft) > 280),
    }
    # Always present, None when unjudged: lets the report format one uniform
    # row instead of branching on whether scores exist.
    for k in [*judge_mod.RUBRIC, "mean"]:
        block[k] = sum(s[k] for s in scored) / len(scored) if scored else None
    if scored:
        problems: dict[str, int] = {}
        for s in scored:
            p = (s.get("worst_problem") or "").strip().lower()[:60]
            if p:
                problems[p] = problems.get(p, 0) + 1
        block["top_problems"] = dict(sorted(problems.items(), key=lambda kv: -kv[1])[:5])
    return block


def judge_agreement(agent_preds: list[Triage], scores: list[dict | None]) -> dict | None:
    """Compare the judge against human ratings from data/reply_ratings.jsonl."""
    if not REPLY_RATINGS.exists():
        return None
    human = {
        r["id"]: r for r in
        (json.loads(l) for l in REPLY_RATINGS.read_text().splitlines() if l.strip())
    }
    pairs = [
        (human[i]["mean"], s["mean"])
        for i, (t, s) in enumerate(zip(agent_preds, scores))
        if s and i in human and "mean" in human[i]
    ]
    if len(pairs) < 10:
        return {"n": len(pairs), "note": "too few human ratings to report agreement"}
    return metrics.agreement([h for h, _ in pairs], [j for _, j in pairs])


def annotator_self_consistency(golden: list[dict]) -> dict | None:
    """Ceiling on any reported accuracy: the annotator vs their own second pass."""
    if not RECHECK.exists():
        return None
    second = {
        r["id"]: r["intent"] for r in
        (json.loads(l) for l in RECHECK.read_text().splitlines() if l.strip())
    }
    pairs = [(g["intent"], second[g["id"]]) for g in golden if g["id"] in second]
    if len(pairs) < 10:
        return {"n": len(pairs), "note": "too few rechecked examples"}
    return metrics.self_consistency([a for a, _ in pairs], [b for _, b in pairs])


def build_report(results: dict) -> str:
    """Markdown tables. Deliberately terse -- the prose lives in the README.

    Plain loops on purpose. An abstracted table builder was tried and reverted:
    these five tables share a shape but no cell, so the helpers cost as many
    lines as the duplication they removed.
    """
    lines = ["# Results", "", f"Golden set: {results['n']} examples "
             f"({results['n_blind']} blind, {results['n_assisted']} assisted)", ""]

    lines += ["## Intent classification", "",
              "| system | acc (balanced) | macro-F1 | acc (weighted) | acc (blind only) |",
              "|---|---|---|---|---|"]
    for name, r in results["systems"].items():
        i = r["intent"]
        blind = f"{i['blind_only']['accuracy']:.3f}" if "blind_only" in i else "-"
        lines.append(f"| {name} | {i['balanced']['accuracy']:.3f} | "
                     f"{i['balanced']['macro_f1']:.3f} | "
                     f"{i['weighted']['accuracy']:.3f} | {blind} |")

    lines += ["", "## Escalation", "",
              "| system | precision | recall | missed | needless | cost/100 (10:1) |",
              "|---|---|---|---|---|---|"]
    for name, r in results["systems"].items():
        e = r["escalation"]["balanced"]
        lines.append(f"| {name} | {e['precision']:.3f} | {e['recall']:.3f} | "
                     f"{e['missed_escalations']} | {e['needless_escalations']} | "
                     f"{e['cost_per_100']:.1f} |")

    lines += ["", "### Cost sensitivity", "",
              "Ratio = cost of a missed escalation vs a needless one. Lower is better.", "",
              "| system | " + " | ".join(f"{r:g}:1" for r in COST_RATIOS) + " |",
              "|---|" + "---|" * len(COST_RATIOS)]
    for name, r in results["systems"].items():
        lines.append(f"| {name} | " + " | ".join(
            f"{v:.1f}" for v in r["escalation"]["cost_sweep"].values()) + " |")

    lines += ["", "## Reply quality (LLM judge, 1-5)", "",
              "| system | grounded | helpful | tone | safety | mean | drafted "
              "| ungrounded | >280 chars |", "|---|---|---|---|---|---|---|---|---|"]
    for name, r in results["systems"].items():
        q = r["reply"]
        # Both branches must yield the same number of cells, or the row's pipes
        # desynchronise from the header and the table silently renders wrong.
        cells = ([f"{q[k]:.2f}" for k in [*judge_mod.RUBRIC, "mean"]]
                 if q["mean"] is not None else ["-"] * (len(judge_mod.RUBRIC) + 1))
        lines.append("| " + " | ".join([
            name, *cells, str(q["n_drafted"]),
            f"{q['ungrounded_rate']:.1%}", str(q["over_limit"]),
        ]) + " |")

    if results.get("judge_agreement"):
        a = results["judge_agreement"]
        lines += ["", "## Judge vs human", ""]
        if "note" in a:
            lines.append(f"{a['note']} (n={a['n']})")
        else:
            lines += [
                f"- n = {a['n']} rated by hand",
                f"- Spearman = {a['spearman']:.3f}" if a.get("spearman") is not None
                else "- Spearman = undefined (no variance)",
                f"- quadratic kappa = {a['kappa_quadratic']:.3f}",
                f"- judge bias = {a['judge_bias']:+.2f} "
                f"({'generous' if a['judge_bias'] > 0 else 'harsh'} vs human)",
                f"- within 1 point = {a['within_1']:.1%}",
            ]

    if results.get("cross_family"):
        c = results["cross_family"]
        lines += ["", "## Cross-family judge check (Gemma vs Gemini)", "", f"- n = {c['n']}"]
        if "note" not in c:
            lines += [
                f"- Spearman = {c['spearman']:.3f}" if c.get("spearman") is not None
                else "- Spearman = undefined",
                f"- Gemma is {c['judge_bias']:+.2f} vs Gemini", "",
                "Low correlation would mean much of the primary judge's score is",
                "family-specific taste rather than reply quality.",
            ]

    if results.get("self_consistency"):
        sc = results["self_consistency"]
        lines += ["", "## Annotator self-consistency (the ceiling)", ""]
        if "note" in sc:
            lines.append(f"{sc['note']} (n={sc['n']})")
        else:
            lines += [
                f"- n = {sc['n']} re-labelled blind",
                f"- agreement = {sc['agreement']:.3f}, kappa = {sc['kappa']:.3f}", "",
                "No system can be meaningfully scored above this. An intent accuracy",
                "at or above it is measuring label noise, not skill.",
            ]

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N")
    ap.add_argument("--no-judge", action="store_true", help="skip reply scoring")
    ap.add_argument("--cross-family", type=int, default=40, help="0 to skip")
    ap.add_argument(
        "--judge-sample", type=int, default=90,
        help="how many golden examples to score for reply quality (0 = all). "
             "Judging all 180 across three systems exceeds the free-tier daily cap; "
             "90 is enough for a stable mean and keeps a fresh run inside one day.",
    )
    args = ap.parse_args()

    golden = load_golden()
    if args.limit:
        golden = golden[: args.limit]
    print(f"golden set: {len(golden)} labelled examples")

    retriever = Retriever()
    print(f"corpus: {len(retriever.corpus):,} grounding pairs")
    if not args.no_judge:
        preflight(len(golden), args.judge_sample or len(golden))

    try:
        systems = run_systems(golden, retriever)
    except llm.CacheMiss as exc:
        raise SystemExit(f"\n{exc}\n")

    results: dict = {
        "n": len(golden),
        "n_blind": sum(g.get("pass") == "blind" for g in golden),
        "n_assisted": sum(g.get("pass") == "assisted" for g in golden),
        "intents": list(load_intents()),
        "systems": {},
    }

    all_scores: dict[str, list] = {}
    for name, preds in systems.items():
        scores = [None] * len(preds)
        if not args.no_judge:
            # Score a fixed prefix so every system is judged on the SAME
            # examples -- a different sample per system would make the
            # comparison between them meaningless.
            limit = args.judge_sample or len(preds)
            print(f"judging {name} (first {min(limit, len(preds))} examples)...")
            judged = judge_mod.judge_replies([t.to_dict() for t in preds[:limit]])
            scores = judged + [None] * (len(preds) - len(judged))
        all_scores[name] = scores
        results["systems"][name] = {
            "intent": _intent_block(golden, preds),
            "escalation": _escalation_block(golden, preds),
            "reply": _reply_block(preds, scores),
        }

    results["judge_agreement"] = judge_agreement(systems["agent"], all_scores["agent"])
    results["self_consistency"] = annotator_self_consistency(golden)

    if args.cross_family and not args.no_judge:
        print(f"cross-family judging {args.cross_family} agent replies with Gemma...")
        # Gemma, not Gemini: a different model family. Correlation between the
        # two judges bounds how much of the primary score is shared-family
        # taste rather than reply quality.
        agent_dicts = [t.to_dict() for t in systems["agent"]][: args.cross_family]
        alt = judge_mod.judge_replies(agent_dicts, model=llm.ALT_JUDGE, batch_size=3)
        pairs = [
            (p["mean"], a["mean"])
            for p, a in zip(all_scores["agent"], alt) if p and a
        ]
        results["cross_family"] = (
            metrics.agreement([p for p, _ in pairs], [a for _, a in pairs])
            if len(pairs) >= 10 else {"n": len(pairs), "note": "too few paired scores"}
        )

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2))
    (REPORTS / "results.md").write_text(build_report(results))

    # Full per-example output, so failure analysis reads real cases not summaries.
    with (REPORTS / "predictions.jsonl").open("w") as fh:
        for i, g in enumerate(golden):
            fh.write(json.dumps({
                "id": g["id"],
                "message": g["message"],
                "truth": {"intent": g["intent"], "should_escalate": g["should_escalate"],
                          "pass": g.get("pass"), "note": g.get("note", "")},
                "systems": {
                    name: {**systems[name][i].to_dict(), "judge": all_scores[name][i]}
                    for name in systems
                },
            }, ensure_ascii=False) + "\n")

    print("\n" + build_report(results))
    print(f"wrote {_rel(REPORTS)}/results.{{json,md}} and predictions.jsonl")


if __name__ == "__main__":
    main()
