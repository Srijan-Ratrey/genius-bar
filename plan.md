# genius-bar — AI support agent for AppleSupport

## Context

Hiver SDE Intern take-home. Build an AI support agent for **one brand** from the
Customer Support on Twitter dataset (~3M tweets) that (1) classifies intent,
(2) drafts a reply grounded in that brand's historical resolutions, and
(3) decides auto-handle vs escalate with a stated reason.

The assignment states plainly: **"The proof is worth more than the system."**
So the evaluation harness, golden set, and honest failure analysis are the
primary deliverable — the agent is the thing being measured, not the point.
Plan is weighted accordingly: a deliberately simple agent, a serious harness.

**Locked decisions** (from user):

| Decision | Choice |
|---|---|
| Title / repo | `genius-bar` (package `genius_bar`) |
| Brand | AppleSupport (~106k tweets) |
| LLM | Gemini via Google AI Studio free tier |
| Quota strategy | Batch requests + committed on-disk cache |
| Dataset | Kaggle API (creds in `.env`) |
| Golden set | 180 examples: 60 blind-labelled + 120 model-assisted |
| Env | `uv` virtual environment, Python 3.12 |

---

## Blockers to clear before coding

1. **`KAGGLE_USERNAME` missing.** `.env` has only `KAGGLE_API_TOKEN` (bare
   37-char key). The Kaggle API needs both `KAGGLE_USERNAME` and `KAGGLE_KEY`.
   I will rename to the expected vars and you add the username. Fallback if
   you'd rather not: download `twcs.csv` manually into `data/raw/`.
2. **`GEMINI_API_KEY`** — add to `.env` once you have it from AI Studio.
3. **`.env` is currently unprotected and holds a live key.** `.gitignore` with
   `.env` must land in the very first commit, before the repo is wired up.
4. `genius-bar` borrows Apple's "Genius Bar" trademark. Your call, noted once —
   fine for a take-home, worth renaming if it ever goes public.

---

## Design

### Why AppleSupport is the hard case (and how the plan handles it)

Apple threads are long, multi-turn troubleshooting exchanges, and intents blur
into one large "my device is broken" mass. Two consequences shape the design:

- **Taxonomy must be derived, not invented.** Cluster first, name second, then
  hand-fix. A taxonomy asserted from intuition would not survive questioning.
- **Apple deflects to DM often.** Threads ending in "DM us" contain no
  resolution. These are filtered out of the grounding corpus and that filter is
  reported as a bias — it means the corpus over-represents the easy cases.

### Pipeline

```
twcs.csv ──filter+thread──> apple_threads.parquet
                                   │
              ┌────────────────────┼─────────────────────┐
              ▼                    ▼                     ▼
        taxonomy (once)     grounding corpus       golden set (180)
        intents.yaml        (inbound→resolution)   golden.jsonl
                                   │
                                   ▼
   message ─> classify ─> retrieve k=5 ─> draft ─> escalation policy
                                   │
                                   ▼
                          eval: metrics + LLM-judge + agreement
```

### Escalation is a deterministic policy, not an LLM vibe

The LLM emits structured signals (intent, confidence, risk flags, retrieval
score). A plain Python function maps those to `auto | escalate` plus a reason
string. Auditable, unit-testable, and the "stated reason" requirement falls out
for free. Hard-escalate rules regardless of confidence: data loss, payment
disputes, legal/press threats, self-harm signals, repeat contact.

### Baselines (assignment requires trivial + simple)

- **Trivial** — majority intent, one canned reply, never escalate. Exists to
  expose how much of any headline accuracy number is just class imbalance.
- **Simple** — TF-IDF nearest neighbour: intent by kNN vote, reply copied
  verbatim from the closest historical agent reply, escalation by keyword rules.
  A genuinely strong baseline for short tweets; if the LLM system cannot beat
  copy-paste retrieval, that is the finding and it gets reported.

### Metrics

- **Intent** — accuracy, macro-F1 (imbalance-robust), per-class confusion.
- **Escalation** — precision/recall on "should escalate" plus a cost-weighted
  score. Errors are asymmetric: wrongly auto-handling a refund dispute costs far
  more than needlessly escalating a password reset.
- **Reply** — LLM-judge rubric (groundedness, correctness, tone, actionability,
  safety; 1–5) and pairwise vs the simple baseline.
- **Judge trust** — Cohen's κ and Spearman between judge and your blind ratings
  on the 60-example clean subset. Without this the judge scores mean nothing.

### Quota strategy (the real engineering constraint)

Free tier caps ~250 requests/day on 2.5-flash. A naive run needs 1,000–1,400.

- Batch 5–10 items per request for drafting and judging.
- SHA-256 disk cache on `(model, prompt, schema)`, committed to the repo.
- **Gemini embeddings API instead of local sentence-transformers** — 100 texts
  per request (~50 requests for the whole corpus), cached to `.npz`. This drops
  the ~2GB torch dependency entirely, which is also what keeps the grader's
  reproduce under 15 minutes.
- `tenacity` exponential backoff on 429; a budget counter prints requests used.
- **`make eval` must recompute every headline number from committed cache with
  no API key present.** This is a hard design requirement, not a nicety.

### Golden set: 180 examples

Stratified by intent, thread length, and deliberately over-sampled hard cases
(sarcasm, multi-intent, non-English, no-resolution threads). Stratified sampling
means reported accuracy does **not** reflect production distribution — that goes
in the misleading-number section.

- **60 blind** — you label with no model suggestions. This subset alone carries
  the judge-agreement and label-noise claims.
- **120 assisted** — model pre-labels, you correct in a `rich` TUI.
- Both halves tagged in `golden.jsonl` and reported separately, always.

---

## Repo layout

```
genius-bar/
├── README.md              # includes the full report section
├── plan.md                # this plan, committed for the graders
├── DECISIONS.md           # decision log, appended as we go (target 10–15)
├── pyproject.toml         # uv, Python 3.12
├── Makefile               # data / taxonomy / label / eval / test
├── .env.example  .gitignore
├── data/
│   ├── raw/.gitkeep       # twcs.csv gitignored (~500MB)
│   ├── apple_threads.parquet   # committed subsample
│   ├── intents.yaml            # committed taxonomy
│   └── golden.jsonl            # committed golden set
├── cache/                 # committed LLM + embedding cache
├── genius_bar/
│   ├── data.py            # load, filter, thread reconstruction
│   ├── llm.py             # Gemini client: batching, cache, retry, budget
│   ├── agent.py           # classify → retrieve → draft → escalation policy
│   ├── baselines.py       # trivial + TF-IDF
│   ├── eval.py            # metrics, LLM-judge, agreement
│   └── label.py           # labelling TUI
├── scripts/derive_taxonomy.py  # one-off: cluster + LLM naming
├── tests/test_smoke.py
└── reports/               # generated tables and figures
```

---

## Build order (commit every 3 changes, as requested)

| # | Milestone | Commit after |
|---|---|---|
| 1 | `uv init`, pyproject, `.gitignore` (`.env` first!), `plan.md`, `.env.example` | ✅ commit 1 |
| 2 | `data.py`: Kaggle download, chunked AppleSupport filter, thread reconstruction, parquet subsample | |
| 3 | `llm.py`: Gemini client with cache, batching, retry, budget counter | ✅ commit 2 |
| 4 | `scripts/derive_taxonomy.py` → cluster, LLM-name, hand-fix → `intents.yaml` | |
| 5 | Grounding corpus build (inbound→resolution pairs, DM-deflection filtered) + embeddings | |
| 6 | `agent.py`: classify, retrieve, draft, escalation policy | ✅ commit 3 |
| 7 | `baselines.py`: trivial + TF-IDF | |
| 8 | `label.py` TUI + sample the 180 → **you label the 60 blind seed** | |
| 9 | `eval.py`: intent + escalation metrics vs baselines | ✅ commit 4 |
| 10 | LLM-judge rubric + judge/human agreement on the blind 60 | |
| 11 | Failure analysis: top 5 modes with real examples | |
| 12 | README report: framing, results, failures, misleading-number, next week | ✅ commit 5 |
| 13 | `DECISIONS.md` final pass, smoke test, 15-min reproduce rehearsal from clean clone | ✅ commit 6 |

Steps 8 and 10 need your hands on the keyboard — everything else is unattended.

---

## What I am deliberately not building

Stated up front because the report has to defend it:

- No fine-tuning. No vector database (numpy dot product over a few thousand
  rows is enough — a real index earns its place above ~100k rows).
- No multi-turn dialogue state. Single inbound message in, one draft out.
- No web UI, no serving layer, no Docker. It is a pipeline plus a harness.
- No multilingual handling — non-English is detected and escalated, not
  translated.

---

## Verification

- `uv run pytest` — thread reconstruction, escalation policy truth table,
  cache-hit behaviour.
- `make eval` **from a clean clone with no API key** → reproduces every headline
  number from committed cache in under 15 min. Rehearsed before submission.
- `make eval FRESH=1` — same numbers with live API calls, to prove the cache is
  not stale or fabricated.
- Escalation policy gets an explicit truth table test; it is the component where
  a silent bug is most costly.

## Known risks

- **Judge and generator are both Gemini** → self-enhancement bias. Mitigation:
  cross-check a subset with a second judge from a different model family and
  report the delta. This is a headline-misleading item, not a footnote.
- Single annotator (you) means no inter-annotator agreement. Mitigation: label
  30 examples twice, several days apart, and report self-consistency as a
  ceiling on achievable accuracy.
- Free-tier quota may force a multi-day labelling/eval cadence. The cache makes
  this resumable.
