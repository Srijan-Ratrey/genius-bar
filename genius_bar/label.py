"""Terminal labelling tool for the golden set.

Three modes:

  intent  (default)  assign the true intent and the escalate/auto decision
  recheck            re-label examples already done, to measure the annotator
                     against themselves. That self-agreement is the CEILING --
                     no classifier can be meaningfully scored above the rate at
                     which the person who wrote the labels reproduces them.
  reply              rate drafted replies on the judge's own rubric, which is
                     what makes the LLM-judge agreement number possible

The blind/assisted split is enforced here, not just recorded. During the blind
pass no model output is displayed at all -- no rule proxy, no LLM suggestion.
Those 60 labels are the only ones that can carry the judge-agreement and
label-noise claims, so contaminating them with a suggestion would quietly
destroy the evidence they exist to provide.

Every answer is written immediately, so the session can be interrupted and
resumed. Labelling 180 examples is roughly 1.5-2 hours and nobody should have
to do it in one sitting.

Usage:
  uv run python -m genius_bar.label                 # blind pass, then assisted
  uv run python -m genius_bar.label --mode recheck --n 30
  uv run python -m genius_bar.label --mode reply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from genius_bar.agent import load_intents
from genius_bar.data import REPO_ROOT

UNLABELLED = REPO_ROOT / "data" / "golden_unlabelled.jsonl"
GOLDEN = REPO_ROOT / "data" / "golden.jsonl"
RECHECK = REPO_ROOT / "data" / "golden_recheck.jsonl"
REPLY_RATINGS = REPO_ROOT / "data" / "reply_ratings.jsonl"

console = Console()

GUIDELINES = """\
[bold]Label what the customer NEEDS, not how they said it.[/bold]
An furious "iOS 11 RUINED MY PHONE" is still update_performance. Anger is a
separate escalation signal, never a different intent.

[bold]not_actionable[/bold] means there is genuinely nothing to answer -- venting,
jokes, feature requests, brand commentary. NOT merely "the message was rude".

[bold]Escalate[/bold] = a human must see this before anything is sent. Escalate for:
money, permanent data loss, safety (overheating/burns), legal or press threats,
a customer who has already asked once, or a language this agent cannot write.
Do NOT escalate simply because the customer is angry -- that is the default
register in this corpus.

[bold]When genuinely torn, pick the better fit and write a note.[/bold] These categories
overlap by construction (silhouette 0.085), so ambiguity is expected and the
notes are evidence, not failure.
"""


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def show_intent_menu(intents: dict[str, dict]) -> None:
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("#", width=3)
    table.add_column("intent", width=22)
    table.add_column("risk", width=5)
    table.add_column("what it means")
    for i, (name, meta) in enumerate(intents.items(), 1):
        desc = " ".join(meta["description"].split())
        risk = meta["risk"]
        table.add_row(
            str(i),
            name,
            f"[red]{risk}[/red]" if risk == "high" else risk,
            desc[:74] + ("..." if len(desc) > 74 else ""),
        )
    console.print(table)


def ask_label(item: dict, intents: dict[str, dict], suggestion: str | None) -> dict | None:
    """Present one message and collect a label. Returns None to quit."""
    names = list(intents)
    tags = ", ".join(item.get("hard_cases", [])) or "-"

    console.print(Panel(
        item["message"],
        title=f"[bold]{item['pass']}[/bold] pass",
        subtitle=f"hard cases: {tags}",
        border_style="cyan" if item["pass"] == "blind" else "yellow",
    ))
    if suggestion:
        console.print(f"  model suggests: [magenta]{suggestion}[/magenta]")

    while True:
        raw = console.input(
            f"  intent [1-{len(names)}] (s=skip, q=save+quit, ?=menu): "
        ).strip().lower()
        if raw == "q":
            return None
        if raw == "s":
            return {}
        if raw == "?":
            show_intent_menu(intents)
            continue
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            intent = names[int(raw) - 1]
            break
        console.print("  [red]enter a number from the menu[/red]")

    default_escalate = intents[intent]["risk"] == "high"
    hint = "Y/n" if default_escalate else "y/N"
    esc = console.input(f"  escalate to a human? [{hint}]: ").strip().lower()
    should_escalate = default_escalate if esc == "" else esc.startswith("y")

    note = console.input("  note (optional, enter to skip): ").strip()

    return {
        "id": item["id"],
        "message": item["message"],
        "intent": intent,
        "should_escalate": should_escalate,
        "note": note,
        "pass": item["pass"],
        "proxy_intent": item["proxy_intent"],
        "hard_cases": item.get("hard_cases", []),
        "weight": item.get("weight", 1.0),
    }


def label_intents(only_pass: str | None = None) -> None:
    intents = load_intents()
    items = read_jsonl(UNLABELLED)
    if not items:
        raise SystemExit(f"{UNLABELLED} not found. Run scripts/sample_golden.py first.")

    done = {r["id"] for r in read_jsonl(GOLDEN)}
    # Blind first, always: once any model output has been seen, the blind pass
    # is no longer blind.
    order = {"blind": 0, "assisted": 1}
    todo = [i for i in items if i["id"] not in done]
    todo.sort(key=lambda i: order.get(i["pass"], 2))
    if only_pass:
        todo = [i for i in todo if i["pass"] == only_pass]

    if not todo:
        console.print(f"[green]all {len(items)} examples already labelled.[/green]")
        return

    console.print(Panel(GUIDELINES, title="labelling guidelines", border_style="green"))
    show_intent_menu(intents)
    console.print(
        f"\n[bold]{len(todo)} to label[/bold] "
        f"({sum(i['pass'] == 'blind' for i in todo)} blind, "
        f"{sum(i['pass'] == 'assisted' for i in todo)} assisted). "
        "Progress saves after every answer.\n"
    )

    # Suggestions are fetched when the assisted pass is actually reached, not
    # up front: someone labelling only the blind 60 today should spend no
    # generate quota at all, and should not wait on 15 requests to start.
    suggestions: dict[int, str] | None = None

    for n, item in enumerate(todo, 1):
        console.rule(f"{n}/{len(todo)}  (done: {len(done)}/{len(items)})")
        hint = None
        if item["pass"] == "assisted":
            if suggestions is None:
                suggestions = _load_suggestions(todo)
            hint = suggestions.get(item["id"])
        record = ask_label(item, intents, hint)
        if record is None:
            console.print(f"\n[yellow]saved. {len(done)}/{len(items)} labelled.[/yellow]")
            return
        if record:
            append_jsonl(GOLDEN, record)
            done.add(record["id"])

    console.print(f"\n[green]done: {len(done)}/{len(items)} labelled.[/green]")


def _load_suggestions(todo: list[dict]) -> dict[int, str]:
    """Pre-label the assisted items with the model, in one batched pass."""
    assisted = [i for i in todo if i["pass"] == "assisted"]
    if not assisted:
        return {}

    from genius_bar.agent import classify

    console.print(f"pre-labelling {len(assisted)} assisted examples...")
    try:
        preds = classify([i["message"] for i in assisted])
    except Exception as exc:  # quota, network -- labelling should still proceed
        console.print(f"[yellow]could not pre-label ({type(exc).__name__}); "
                      f"continuing without suggestions[/yellow]")
        return {}
    return {
        i["id"]: f"{p['intent']} (conf {p['confidence']:.2f})"
        for i, p in zip(assisted, preds)
    }


def recheck(n: int) -> None:
    """Re-label already-labelled examples to measure annotator self-agreement."""
    intents = load_intents()
    labelled = read_jsonl(GOLDEN)
    if len(labelled) < n:
        raise SystemExit(f"only {len(labelled)} labelled; need {n}. Label more first.")

    import random

    already = {r["id"] for r in read_jsonl(RECHECK)}
    pool = [r for r in labelled if r["id"] not in already]
    random.Random(0).shuffle(pool)
    todo = pool[: n - len(already)]

    if not todo:
        console.print(f"[green]recheck complete ({len(already)} examples).[/green]")
        return

    console.print(Panel(
        "Re-labelling examples you have already done, with your original answer "
        "hidden.\nDisagreement here is not a mistake -- it measures how noisy the "
        "task itself is,\nand that noise is the ceiling on any score reported "
        "against these labels.",
        title="self-consistency recheck", border_style="magenta",
    ))
    show_intent_menu(intents)

    for i, item in enumerate(todo, 1):
        console.rule(f"recheck {i}/{len(todo)}")
        record = ask_label({**item, "pass": "recheck"}, intents, None)
        if record is None:
            break
        if record:
            append_jsonl(RECHECK, record)

    console.print(f"[green]recheck saved to {RECHECK.name}[/green]")


def rate_replies(n: int = 40) -> None:
    """Hand-rate agent drafts on the judge's own rubric.

    This is what makes the judge trustworthy or not. Without human ratings to
    correlate against, a judge score of 3.8/5 is a number with no referent.

    Two things are deliberately hidden while rating:
      - the LLM judge's scores, for the obvious reason
      - which system produced the draft, so a known-baseline reply is not
        marked down for being the baseline's

    Rated examples are drawn from the BLIND pass first. Those labels were made
    with no model output visible, so they are the only ones that can carry an
    agreement claim without a contamination caveat attached.
    """
    from genius_bar.judge import RUBRIC

    preds_path = REPO_ROOT / "reports" / "predictions.jsonl"
    if not preds_path.exists():
        raise SystemExit(
            f"{preds_path.relative_to(REPO_ROOT)} not found.\n"
            "Run `make eval` first -- rating needs drafts to rate."
        )

    rows = read_jsonl(preds_path)
    done = {r["id"] for r in read_jsonl(REPLY_RATINGS)}

    # Blind-pass examples first, and only drafts that actually exist.
    candidates = [
        r for r in rows
        if r["id"] not in done and r["systems"]["agent"].get("draft", "").strip()
    ]
    candidates.sort(key=lambda r: 0 if r["truth"].get("pass") == "blind" else 1)
    todo = candidates[: max(0, n - len(done))]

    if not todo:
        console.print(f"[green]{len(done)} replies already rated.[/green]")
        return

    console.print(Panel(
        "Score each reply 1-5 on four criteria. The model's own scores are\n"
        "hidden, and so is which system wrote the draft.\n\n"
        + "\n".join(f"[bold]{k}[/bold]: {v.splitlines()[0]}" for k, v in RUBRIC.items())
        + "\n\nUse the full 1-5 range. Rating everything 3 makes the correlation\n"
          "meaningless, which defeats the point of doing this by hand.",
        title="reply rating", border_style="green",
    ))

    for i, row in enumerate(todo, 1):
        console.rule(f"{i}/{len(todo)}  (rated: {len(done)})")
        agent = row["systems"]["agent"]
        console.print(Panel(row["message"], title="customer", border_style="cyan"))
        console.print(Panel(agent["draft"], title="proposed reply", border_style="yellow"))
        for j, e in enumerate(agent.get("evidence", [])[:3], 1):
            console.print(f"  precedent {j}: [dim]{e['reply'][:140]}[/dim]")

        scores: dict[str, int] = {}
        quit_now = False
        for criterion in RUBRIC:
            while True:
                raw = console.input(f"  {criterion} [1-5] (q=save+quit): ").strip().lower()
                if raw == "q":
                    quit_now = True
                    break
                if raw.isdigit() and 1 <= int(raw) <= 5:
                    scores[criterion] = int(raw)
                    break
                console.print("  [red]1-5 please[/red]")
            if quit_now:
                break
        if quit_now:
            console.print(f"\n[yellow]saved. {len(done)} rated.[/yellow]")
            return

        append_jsonl(REPLY_RATINGS, {
            "id": row["id"],
            **scores,
            "mean": sum(scores.values()) / len(scores),
        })
        done.add(row["id"])

    console.print(f"[green]done: {len(done)} replies rated.[/green]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["intent", "recheck", "reply"], default="intent")
    ap.add_argument("--n", type=int, default=30, help="recheck size")
    ap.add_argument("--pass", dest="which", choices=["blind", "assisted"], default=None)
    args = ap.parse_args()

    if args.mode == "intent":
        label_intents(args.which)
    elif args.mode == "recheck":
        recheck(args.n)
    else:
        rate_replies(args.n)


if __name__ == "__main__":
    main()
