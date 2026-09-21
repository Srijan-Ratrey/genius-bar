# genius-bar

An AI support agent for **AppleSupport**, built from real customer-support
threads on Twitter. Given an incoming customer message it:

1. **Classifies** it into an 8-intent taxonomy derived from the data by clustering.
2. **Drafts** a reply grounded in how Apple has historically handled similar messages.
3. **Decides** auto-handle vs escalate, with a stated reason, via a deterministic policy.

The assignment behind this says *"the proof is worth more than the system."* So
the agent is deliberately simple and the effort went into the evaluation harness,
the hand-labelled golden set, and finding out where it breaks.

> **Status.** Complete and reproducible. 180 golden examples hand-labelled (60
> blind, 120 assisted), 30 replies hand-rated, 39 tests passing. Every number
> below comes from `make eval`. One planned measurement was **not** taken:
> annotator self-consistency (§7.6).

---

## Quickstart

```bash
uv sync                  # Python 3.12, ~10 deps, no torch
make test                # 36 tests
make eval                # reproduce all headline numbers -- NO API KEY NEEDED
```

`make eval` replays a committed response cache, so it recomputes every number
offline. `make reproduce` is the full grader path (`uv sync && pytest && eval`)
and is the thing held to the 15-minute budget.

To regenerate rather than replay, put `GEMINI_API_KEY` in `.env` (see
[.env.example](.env.example)). To rebuild the data subsample from scratch you
also need Kaggle credentials, but the subsample is committed so you don't.

| target | what it does |
|---|---|
| `make data` | rebuild the AppleSupport subsample from the Kaggle dump |
| `make taxonomy` | re-derive the intent taxonomy draft |
| `make golden` | draw the 180-example golden set |
| `make label` | hand-label it (60 blind, then 120 assisted) |
| `make recheck` | re-label 30 to measure annotator self-consistency |
| `make rate` | hand-rate replies, for judge-vs-human agreement |
| `make eval` | all metrics, all three systems |

---

## Report

### 1. Problem framing

#### What the data actually is

AppleSupport is ~106k tweets across 80,702 threads in the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dump. This project samples 15,000 threads at random (44,284 tweets); the
subsample's turn distribution matches the population, so there is no sampling
bias to argue about.

Four measured facts shaped every design decision:

| fact | figure | consequence |
|---|---|---|
| Threads are short | median **2 turns**; only 24.8% reach 4 | "multi-turn resolution" mostly does not exist here |
| Apple deflects to DM | **53.5%** of replies say "DM us" | over half the corpus has no public resolution |
| Answers are behind links | **75.9%** of replies contain a t.co link whose article is not in the dataset | the actual fix is unreadable |
| One OS release dominates | `update_performance` is **38%** of traffic | the corpus is a late-2017 iOS 11 snapshot |

#### What "good" means for this brand

Given the above, the obvious definition — *does the agent resolve the
customer's problem* — is not measurable from this data, and any system claiming
it would be scored against a record that does not contain resolutions. So "good"
is defined as three narrower things that *are* measurable:

1. **Correct triage.** The intent is right often enough to route the message,
   and where intents genuinely overlap the agent says so via low confidence
   rather than guessing confidently.
2. **A faithful first response.** The draft is what Apple would plausibly send
   *next* — usually one diagnostic question or one setting to check — and
   invents nothing. Not a complete fix, because Apple does not send complete
   fixes on Twitter.
3. **Trustworthy abstention.** Anything involving money, permanent data loss,
   safety, legal threats, or a language the agent cannot write reaches a human,
   every time, with the reason attached. A missed escalation is treated as ~10x
   worse than an unnecessary one.

The headline claim this project can honestly support is therefore about
**grounded first-response drafting and safe triage**, not issue resolution.

#### What I chose not to build

- **Resolution generation.** The single biggest scope cut, and it follows from
  the data rather than from laziness: with 53.5% of replies deflecting to DM and
  the rest linking articles that aren't in the dump, there is nothing to learn
  full resolutions from. Generating them would mean inventing them.
- **Fine-tuning.** 180 labelled examples is an evaluation set, not a training
  set. Spending it on training would leave nothing to measure with.
- **A vector database.** 6,221 grounding pairs is a numpy dot product. An index
  earns its place somewhere above 100k rows.
- **Embedding-based retrieval.** Not a preference — the free embedding quota is
  1,000 texts/**day** (each text counts as a request, not each batch), which is
  several days of quota for one corpus. See DECISIONS.md #36.
- **Multi-turn dialogue state.** One inbound message in, one draft out. The
  median thread is 2 turns, so there is little state to track.
- **Multilingual support.** Non-English is detected and escalated, not
  translated. Apple's own public policy is to redirect these, so escalation
  matches the brand's actual behaviour.
- **Any serving layer, UI, or container.** It is a pipeline and a harness.

### 2. Method

#### Intent taxonomy — derived, not invented

`scripts/derive_taxonomy.py` embeds 600 real opening messages, clusters with
KMeans, and asks the LLM only to *name* what clustering found. The draft
([data/intents_draft.yaml](data/intents_draft.yaml)) was then hand-edited into
[data/intents.yaml](data/intents.yaml). Both are committed so the edits are
auditable.

**Two findings mattered more than the taxonomy itself.**

*The clusters are not well separated.* Best silhouette was **0.085** at k=8
across a k=4..20 sweep ([reports/silhouette.json](reports/silhouette.json)).
Anything under ~0.15 means there is essentially no cluster structure. This
taxonomy is **imposed on a continuum, not discovered in one** — so boundary
cases are genuinely ambiguous, and some classifier "errors" are disagreements a
second human would also have. This caps how high intent accuracy can honestly go.

*Clustering split by register, not intent.* It produced four near-duplicate
groups of "my phone broke after the iOS 11 update" differing mainly in
profanity, and the LLM namer labelled the angry ones "not a support request".
Those are merged by intent here, with anger moved to an escalation signal. Left
alone, the classifier would have learned that swearing changes what a customer
needs.

Final taxonomy, with keyword-floor prevalence over all 14,963 opening messages:

| intent | prevalence | risk | default |
|---|---|---|---|
| `update_performance` | 38.4% | low | auto |
| `not_actionable` | ~20% | low | auto |
| `app_or_service_issue` | 7.5% | low | auto |
| `autocorrect_bug` | 5.5% | low | auto |
| `hardware_repair` | 1.9% | high | escalate |
| `device_crash_reboot` | 1.3% | high | escalate |
| `account_billing` | 0.9% | high | escalate |
| `data_loss` | 0.7% | high | escalate |

`how_to` was proposed and then dropped after measuring it: its matches were
overwhelmingly "how do I fix [battery]", a phrasing of other intents rather than
an intent. `app_or_service_issue` was added despite clustering never surfacing
it — a prevalence probe put it at 7.5%, larger than four classes clustering did
surface. Clustering finds what is textually dominant, which is not the same as
what matters.

#### Grounding corpus

Of 19,530 customer→reply pairs, **6,221 (31.9%)** carry enough content to ground
a draft. The filter drops DM deflections and requires concrete steps, a support
link, or a diagnostic question.

An early measurement said "0.0% of replies link an article", which turned out to
be an artefact of stripping URLs before matching. Since Apple's answer is
frequently a bare link, deleting it turned real resolutions into "Try this out:".
Preserving a `[support link]` marker took the usable corpus from 1,346 pairs to
6,221 — a 4.6x difference from one analysis bug.

**This filter is the project's largest bias.** Dropping DM deflections removes
over half the data, and those are disproportionately the *hard* cases — the ones
Apple judged too complex or account-specific to answer publicly. What remains
over-represents problems with a tidy public answer.

#### Retrieval

TF-IDF (word 1-2 grams, sublinear tf) nearest neighbour over historical customer
messages. Support tweets are short and share a tight vocabulary, which is the
regime where TF-IDF holds up well.

Both the agent and the simple baseline use the **same** retriever. That is
deliberate: with retrieval held constant, the measured gap between "copy the
nearest historical reply" and "draft from the same evidence" isolates the
drafting step instead of confounding it with a retrieval change.

#### Escalation — a deterministic policy, not a model call

The model reports structured risk signals; `agent.decide()` is ordinary Python
mapping signals to an action and a reason. Rules run **most-severe first**, so
the stated reason is the most serious applicable one.

```
safety_risk           -> escalate  (overheating, burns)
legal_or_press_threat -> escalate
payment_dispute       -> escalate  (money is never automatic)
irreversible_data_loss-> escalate
non_english           -> escalate  (this agent writes English only)
unrecognised intent   -> escalate  (defence in depth)
intent risk == high   -> escalate
repeat_contact        -> escalate  (the standard reply already failed once)
confidence < 0.60     -> escalate
evidence score < 0.15 -> escalate  (a draft would not be grounded)
multi_intent          -> escalate
otherwise             -> auto
```

Three reasons for keeping this out of the model: it is auditable (every
escalation traces to a named rule, which is what "with a stated reason"
actually requires); it is testable as a truth table; and the cost asymmetry is a
business decision that belongs in code a human can argue with.

It is **partly robust to intent errors**, and the evaluation sharpened this
claim. The *signal* rules (safety, payment, data loss, legal) fire independently
of the intent, so an overheating complaint escalates even when its intent is
misread. The *intent-risk* rules cannot: all 7 missed escalations came from a
misclassified intent (§6.1).

Two guards came out of observed failures rather than speculation:

- **`decide()` re-validates the intent** even though `classify()` already does.
  A test caught that an off-taxonomy label has no risk level, so every risk rule
  would silently pass it to auto-handling.
- **A support draft citing no precedent is downgraded to escalate.** The first
  end-to-end run produced a fluent, generic reply with `used_evidence: []` that
  passed the score gate. Score proves precedent *exists*, not that the draft
  used it — and groundedness is this project's central claim.

**One policy judgement worth arguing with:** severe anger does *not* escalate on
its own. Profanity is the default register in this corpus, so escalating on it
would route 30–40% of traffic to a human and defeat the system's purpose. Anger
is passed to the drafting step to soften tone instead. There is an explicit test
documenting this choice.

### 3. Golden evaluation set

180 examples. How they were sampled and labelled:

**Sampling.** Stratifying by intent needs intent labels, which is what the
golden set exists to create. Three ways out, none free: pure random gives
`data_loss` and `account_billing` one example each and makes escalation
evaluation meaningless; stratifying by LLM enriches the set where the LLM is
already confident, flattering the system under test; stratifying by hand-written
keyword rules flatters the **simple baseline**.

The rule proxy was chosen **because its bias runs against the system being
sold.** If the agent beats a baseline that the sampling itself favoured, the
result is stronger than the raw number, not weaker.

Nine strata (8 intents + "unmatched", which is 44.7% of the corpus and where
`not_actionable` lives) are filled round-robin, 22–23 each. Hard cases are
deliberately over-sampled: 13 very short, 7 very long, 7 non-English, 6
image-only (a screenshot plus "fix this", where the real content is in an image
nothing here can read). Every record carries an importance `weight`, so
production-distribution metrics are recoverable from the balanced sample.

**Labelling.** Two passes, always reported separately:

- **60 blind** — labelled with no model output visible at all. Not merely
  recorded but *enforced* in the tool: no rule proxy, no LLM suggestion. These
  are the only labels that can carry the judge-agreement and label-noise claims.
- **120 assisted** — model pre-labels, human corrects.

Guidelines shown at the top of every session: label what the customer *needs*,
not how they said it; `not_actionable` means nothing to answer, not merely rude;
escalate for money / data loss / safety / legal / repeat contact / language, but
**not** for anger alone.

**The ceiling.** `make recheck` re-labels 30 examples with the original answer
hidden. That self-agreement is the ceiling on any reported accuracy — no
classifier can be meaningfully scored above the rate at which the person who
wrote the labels reproduces their own.

### 4. Evaluation harness

Every number is reported twice — **balanced** over the golden set as sampled,
and **importance-weighted** back to production distribution. Quoting either
alone is the easiest way to mislead with this project, so neither is quoted
alone. Intent metrics are additionally broken out over the 60 blind labels.

- **Intent** — accuracy, macro-F1 (imbalance-robust), per-class F1, confusion.
- **Escalation** — precision/recall on "should escalate", plus a cost-weighted
  `cost_per_100` at missed:needless ratios of 3:1, 10:1 and 30:1. The 10:1
  default is a judgement call, so results are swept across it. Misses are also
  broken out by intent, because *which* escalations you miss matters more than
  how many.
- **Reply quality** — LLM judge, 1–5 on groundedness, helpfulness, tone, safety.
  Scored separately on purpose: a reply can be maximally helpful and completely
  ungrounded, and one merged "quality" score would hide exactly that failure.
  Empty drafts are skipped rather than scored 1, so correctly declining to answer
  is not punished as bad writing.
**A quota constraint that improved the design.** The free tier allows 20
generate requests per day *per model*, so each stage runs on a different model:
classify and draft on `gemini-3.5-flash`, judging on `gemini-3.5-flash-lite`,
cross-family checking on Gemma. Judge and drafter were originally the same
model. Being forced apart genuinely weakens self-enhancement bias rather than
merely disclosing it. `make eval` prints a per-model request estimate before
spending any quota, and reply quality is scored on a 90-example subset (the
same examples for every system) so a fresh run fits inside one day.

- **Judge trust** — quadratic kappa and Spearman against hand ratings on the
  blind subset, plus judge bias (is it systematically generous?). Both are
  reported because they fail differently: a judge that is consistently one point
  generous looks terrible on kappa and near-perfect on Spearman.
- **Cross-family check** — a subset re-scored by `gemma-4-26b-a4b-it`, a different
  model family. Low correlation would mean much of the primary judge's score is
  family-specific taste rather than quality.

### 5. Results vs baselines

180 hand-labelled examples. Reply quality is judged on the first 90 (the same
90 for every system). Regenerate with `make eval`.

#### Intent classification

| system | acc (balanced) | macro-F1 | acc (weighted) | acc (blind only) |
|---|---|---|---|---|
| trivial | 0.183 | 0.039 | 0.345 | 0.183 |
| simple | 0.611 | 0.583 | 0.471 | 0.600 |
| **agent** | **0.833** | **0.820** | **0.843** | **0.783** |

The agent beats the keyword baseline by 22 points balanced and the majority-class
baseline by 65. Macro-F1 moves more than accuracy (0.820 vs 0.583), which is the
more meaningful gap: the baseline gets its accuracy from the majority class.

**The blind-only column is the trustworthy one.** 0.783 against 0.833 balanced.
Those 60 labels were made with no model output on screen; the other 120 were
made with a suggestion visible. A 5-point gap in the model's favour on labels
the model helped produce is what contamination looks like, and it is why the two
passes are never blended.

#### Escalation

| system | precision | recall | missed | needless | cost/100 (10:1) |
|---|---|---|---|---|---|
| trivial | 0.000 | 0.000 | 74 | 0 | 411.1 |
| simple | 0.660 | 0.865 | 10 | 33 | 73.9 |
| **agent** | **0.698** | **0.905** | **7** | **29** | **55.0** |

| system | 3:1 | 10:1 | 30:1 |
|---|---|---|---|
| trivial | 123.3 | 411.1 | 1233.3 |
| simple | 35.0 | 73.9 | 185.0 |
| agent | 27.8 | 55.0 | 132.8 |

The ordering is stable across every cost ratio, so the result does not depend on
the 10:1 assumption. The agent misses 7 escalations where the keyword baseline
misses 10 — a real but modest improvement, and §6 shows all 7 have one cause.

#### Reply quality

| system | addresses this msg | grounded | tone | safety | mean | interchangeable |
|---|---|---|---|---|---|---|
| trivial | 1.27 | 1.00 | 2.88 | 1.36 | 1.62 | 100% |
| simple | 3.86 | 4.42 | 4.49 | 4.91 | 4.42 | 30% |
| **agent** | **4.53** | 5.00 | 5.00 | 5.00 | 4.88 | **10%** |

**Read this table with the next section in hand.** Three of four criteria are
saturated for the agent (exactly 5.00), so the 4.88 mean is a ceiling artefact,
not a measurement. Only `addresses_this_message` and `interchangeable` carry
information, and they agree: the baseline's copied replies are three times more
likely to be boilerplate that would fit any customer.

#### Is the judge trustworthy? No.

30 replies hand-rated against the judge's own rubric:

| | |
|---|---|
| Spearman | 0.462 |
| quadratic kappa | **0.130** |
| judge bias | **+1.01** (generous) |
| within 1 point | 66.7% |

| criterion | human | judge | bias |
|---|---|---|---|
| addresses_this_message | 3.57 | 4.63 | +1.07 |
| groundedness | 3.70 | **5.00** | **+1.30** |
| tone | 4.23 | 5.00 | +0.77 |
| safety | 4.10 | 5.00 | +0.90 |

Kappa of 0.13 is near chance: **as an absolute scorer the judge is unusable.**
Spearman 0.462 says it keeps weak ranking signal, which is the only reason the
system *ordering* above is defensible. The worst bias is on groundedness — the
central claim of this project — where the judge reports a flat 5.00 against a
human 3.70.

Consequently: **the intent and escalation numbers are the load-bearing results.**
They never route through the judge. The reply-quality numbers support an ordering
and nothing finer.

### 6. Failure analysis

Top five, from `reports/predictions.jsonl`.

#### 1. Escalation inherits every intent error (all 7 misses)

All seven missed escalations trace to a wrong intent — not one was a signal
failure. Examples:

- *"Tried installing macOS High Sierra 4 times, always gets stuck before reboot
  step"* → agent said `update_performance` (conf 0.85), truth
  `device_crash_reboot`. High-risk intent read as low-risk, so no escalation.
- *"I redeemed a 3 month Apple Music subscription on October 7th and now this
  happens!"* → agent said `app_or_service_issue` (conf 0.85), truth
  `account_billing`.

**Hypothesis.** Four of eight intents escalate by risk level alone, so for those
the escalation decision is only as good as the classifier. The signal rules
(safety, payment dispute, data loss, legal) fire independently of intent and are
genuinely robust — an overheating complaint escalates even when its intent is
misread. The intent-risk rules are not. An earlier draft of this README claimed
the policy was "robust to intent errors by design"; that is true only of the
signal half, and the data corrects it.

#### 2. The abstention rule is inert

The agent makes 30 intent errors. The `confidence < 0.60` rule catches **2**.
87% of predictions come back at 0.9 or above.

The signal exists — mean confidence is 0.775 when wrong against 0.904 when right
— but the threshold sits where almost no probability mass lives:

| threshold | escalated | errors caught | cost/100 |
|---|---|---|---|
| 0.60 (current) | 96 | 2/30 | 55.0 |
| 0.85 | 112 | 13/30 | 45.6 |
| 0.90 | 125 | 21/30 | **34.4** |

**Hypothesis.** The model reports confidence in a near-degenerate band despite
a prompt explicitly asking for calibration, and the taxonomy's real ambiguity
(silhouette 0.085) is invisible to it. Note that choosing 0.90 from this table
would be fitting to the evaluation set — see §8.

#### 3. `device_crash_reboot` is absorbed by `update_performance` (5 cases)

The largest single confusion, and the worst per-class F1 alongside
`not_actionable` (both 0.69).

- *"with iOS 11.0.* I'm rebooting for fix stuff like it was Windows 98"*
- *"what is the point of doing an update, when it actually makes your phone
  crash all the time?!"*

**Hypothesis.** These are not really errors. The taxonomy splits them on *cause*
— software regression vs device fault — but a tweet almost never reveals cause.
"My phone reboots after the update" is honestly both. This is the silhouette
0.085 finding showing up as a metric, and a second annotator would likely
disagree here too. It matters because one side is high-risk and the other is not.

#### 4. Venting is over-read as actionable (`not_actionable` F1 0.69)

Five cases where the human saw no answerable request and the agent found one —
three routed to `account_billing`, two to `hardware_repair`.

**Hypothesis.** The prompt says not to use `not_actionable` merely because a
message is rude, and the agent over-corrects: complaint language that mentions
money or a broken device pattern-matches to a service intent even with no
request attached. The cost is asymmetric in an unexpected direction — these
misfires escalate to a human, so they are wasteful rather than dangerous.

#### 5. The judge saturates on fluent text

Before the rubric was rewritten, the judge gave 5s to 98% of agent criteria and
94% of the baseline's — rating a system that *copies historical replies verbatim*
at 4.86/5. Replacing `helpfulness` with `addresses_this_message` and adding a
forced `interchangeable` judgement pulled the baseline down to 3.86 and produced
a clean 100%/30%/10% separation.

**Hypothesis.** Absolute quality rubrics collapse on text that is fluent and
polite, because every criterion is satisfied in isolation. Only criteria that are
*relational* (does this fit THIS customer?) or binary (could this be sent to
anyone?) discriminate. Even after the fix, three criteria remain pinned at 5.00,
so the underlying problem is mitigated rather than solved.

### 7. What is misleading about my headline number?

Mandatory section, and most of it is already known before the numbers exist.

1. **The golden set is balanced, not representative.** Nine strata of 22–23
   each, against a real distribution where one class is 38% and four are under
   2%. Balanced accuracy describes traffic that does not exist. The weighted
   column is the honest one for "what would production see", and it will be
   dominated by `update_performance`.
2. **One trivially easy class is over-represented.** `autocorrect_bug` is one
   known bug with one known answer. Any system, including the trivial baseline,
   scores well on it.
3. **The judge is measurably unreliable, and this is now evidence rather than a
   worry.** Against 30 hand ratings: quadratic kappa **0.130**, Spearman 0.462,
   and a systematic **+1.01** generosity. On groundedness it reports a flat 5.00
   against a human 3.70. Before the rubric was rewritten it rated a
   copy-verbatim baseline 4.86/5. Treat every reply-quality figure as an
   ordering, never a score. The intent and escalation results do not route
   through the judge and are unaffected.
4. **Sampling favours the baseline, not the agent.** Stated plainly because it
   cuts the other way from everything else here: strata came from the same
   keyword rules the simple baseline uses.
5. **The intent taxonomy has no natural boundaries.** Silhouette 0.085. A
   fraction of every reported "error" is genuine ambiguity, so intent accuracy
   is measured against labels that a second annotator would not fully reproduce.
6. **No label-noise estimate was taken at all.** The plan called for
   re-labelling 30 examples to measure annotator self-consistency, which would
   have bounded what any accuracy figure can mean. It was not done, so the
   ceiling is unknown. The indirect evidence that remains: the agent scores
   0.783 on blind labels against 0.833 on assisted ones, and human/keyword-proxy
   agreement is 107/180. Both suggest meaningful label variance, but neither is
   a substitute. This is the weakest point in the evaluation.
7. **The grounding corpus over-represents easy cases.** The filter drops the
   53.5% of threads Apple chose not to answer publicly — disproportionately the
   hard ones. Retrieval is being measured on an easier distribution than production.
8. **"Grounded" means grounded in Apple's first-response behaviour**, not in
   verified resolutions. The resolutions are behind t.co links that are not in
   the dataset.
9. **The corpus is frozen in late 2017.** 38% of traffic is one OS rollout, and
   the single most-sent reply addresses one keyboard bug. These numbers describe
   a moment, not Apple's support stream generally.
10. **Escalation cost ratios are asserted, not measured.** 10:1 is a guess about
    Apple's economics. The sweep shows the ordering holds at 3:1 and 30:1 too,
    so the conclusion survives even though the constant is invented.

11. **Escalation safety is capped by intent accuracy.** All 7 missed escalations
    came from a misclassified intent (§6.1). The headline escalation recall of
    0.905 is not an independent safety property — for the four intent-risk
    classes it is a restatement of the classifier's accuracy on them.

12. **The agent's abstention mechanism is effectively off.** The
    `confidence < 0.60` rule fires on 2 of 180 examples. Any credit given to the
    system for "knowing when it is unsure" is unearned: it almost never claims
    to be unsure (§6.2).

### 8. What I'd do next with one more week

1. **Measure label noise at all.** First the cheap version (re-label 30
   examples to get self-consistency), then a second annotator on 60 for real
   inter-annotator agreement. Right now the accuracy ceiling is simply unknown,
   which is the largest single gap in the evaluation (§7.6).
2. **Revisit the anger policy with data.** Measure whether angry messages that
   were auto-handled actually needed a human, instead of reasoning about it.
3. **Sample the DM-deflected threads.** Understand what Apple refuses to answer
   publicly and whether the agent can recognise that class in advance — that is
   the real escalation signal, learned rather than hand-written.
4. **Recalibrate the confidence gate on a held-out split.** The sweep in §6.2
   suggests 0.90 would cut cost/100 from 55.0 to 34.4 and catch 21 of 30 errors
   instead of 2. That number is *not* usable as it stands — it was chosen by
   looking at the evaluation set, which is exactly the overfitting this report
   warns about elsewhere. The right version is a separate validation split, and
   it would likely move the cost/coverage tradeoff more than any prompt change.
5. **Embedding retrieval as a measured ablation.** One day of quota on a corpus
   subset would put a number on what TF-IDF costs, instead of assuming.
6. **Multi-turn on the 24.8% of threads with 4+ turns.** Enough data exists to
   try, and it is where the "did it actually resolve" question becomes answerable.

---

## Repo layout

```
genius_bar/
  data.py        TWCS loading, thread reconstruction, text cleaning
  llm.py         Gemini client: disk cache, batching, quota handling
  retrieve.py    grounding corpus + TF-IDF retriever
  agent.py       classify -> retrieve -> draft -> escalation policy
  baselines.py   trivial + simple
  judge.py       LLM-as-judge rubric
  metrics.py     intent, cost-weighted escalation, agreement
  eval.py        the harness
  label.py       labelling and rating TUI
scripts/
  derive_taxonomy.py   cluster + name -> intents_draft.yaml
  sample_golden.py     stratified draw of the golden set
data/  intents.yaml, apple_threads.parquet, golden*.jsonl
cache/ committed LLM + embedding responses (this is what makes eval offline)
```

[DECISIONS.md](DECISIONS.md) is the decision log — 54 entries, written as the
work happened rather than reconstructed afterwards. [plan.md](plan.md) is the
original plan.

## Credits

- Dataset: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
  (thoughtvector, Kaggle).
- Models, via Google AI Studio free tier: `gemini-3.5-flash` (classify, draft),
  `gemini-3.5-flash-lite` (judge), `gemini-3.1-flash-lite` (assisted-pass
  pre-labels), `gemma-4-26b-a4b-it` (cross-family judge),
  `gemini-embedding-001` (taxonomy clustering only). Each stage runs on its own
  model because the free tier caps generate requests at **20 per day per
  model** -- see §4.
- Libraries: pandas, scikit-learn (TF-IDF, KMeans, metrics), scipy, google-genai,
  tenacity, rich, pyyaml. No torch.
- "Genius Bar" is an Apple trademark, used here only as a project codename for a
  take-home exercise.
