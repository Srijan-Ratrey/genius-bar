# Decision log

Non-obvious choices and why, appended as the project was built. The assignment
asks for 10-15; the ones that changed the design are marked **load-bearing**.

## Framing

1. **Brand: AppleSupport, not the easier options.** Apple has ~106k tweets but
   its intents blur into one large "my device is broken" mass, and it deflects
   to DM constantly. Delta or SpotifyCares would have produced prettier
   numbers from crisper intents. Apple was chosen because the deflection
   problem is the interesting one: it means a large fraction of threads contain
   no resolution to ground a reply in, and pretending otherwise is how you get
   a system that looks good on a golden set and fails in production.

2. **Escalation is a deterministic Python policy, not an LLM judgement.**
   *load-bearing.* The LLM emits structured signals (intent, confidence, risk
   flags, retrieval score); a plain function maps those to auto/escalate plus a
   reason. Auditable, unit-testable as a truth table, and the assignment's
   "stated reason" requirement falls out for free. Asking the model to decide
   directly would make the most consequential step the least inspectable.

3. **Two baselines with distinct jobs.** The trivial one (majority intent, one
   canned reply, never escalate) exists to show how much of any accuracy number
   is just class imbalance. The simple one (TF-IDF nearest neighbour, reply
   copied verbatim from the closest historical agent reply) is deliberately
   strong -- if the LLM cannot beat copy-paste retrieval, that is the finding
   and it gets reported as one.

4. **Deliberately not built:** fine-tuning, a vector database, multi-turn
   dialogue state, any serving layer, multilingual handling. Non-English is
   detected and escalated rather than translated. A numpy dot product over a
   few thousand rows is enough; a real index earns its place above ~100k.

## Reproducibility

5. **Every LLM response is cached to disk and committed.** *load-bearing.*
   The free tier caps requests per *day* (~250 on 2.5-flash) while a naive run
   needs 1,000-1,400, so the cache started as a quota workaround and turned out
   to be the reproducibility story: `make eval` recomputes every headline number
   with no API key and no network, which is what makes the 15-minute reproduce
   target achievable at all.

6. **Cache entries are readable JSON, not a binary store.** A grader can open
   one and see the exact prompt behind a number instead of trusting it. Costs
   disk, buys inspectability.

7. **`temperature=0` everywhere.** A cache keyed on the prompt is only honest if
   the same prompt would produce the same answer again.

8. **A cache miss with no API key raises, never returns a default.** The failure
   mode I most wanted to avoid is a plausible-looking metric computed from
   silently empty responses.

9. **`thinking_budget=0` by default.** Reasoning tokens burn quota and
   classification does not need them; callers opt in where it helps.

## Data handling

10. **`kagglehub` over the `kaggle` package.** `kaggle` authenticates at import
    time and raises without credentials, which would break the no-key `make
    eval` path. A dependency that fails on import is a dependency that dictates
    your architecture.

11. **Two passes over the csv, first pass reads only three columns.** Building
    the reply graph from just the id/author columns fits comfortably in memory
    and yields the *exact* set of tweet ids in AppleSupport threads. Growing the
    set outward from Apple's own tweets would have been one pass but would miss
    consecutive customer tweets.

12. **Threads linearised by timestamp, not by walking the reply tree.** Threads
    are genuinely trees -- one tweet can have several replies -- but sorting by
    time reproduces the conversation a human would read, in one line instead of
    a tree traversal with branch-ordering rules.

13. **Dangling reply parents are kept, not dropped.** The dump is a subsample of
    real Twitter, so `in_response_to_tweet_id` frequently points at a tweet that
    isn't present. Those threads are grouped under the absent root rather than
    discarded; there is a test for it.

14. **Batch misalignment triggers a per-item retry.** *load-bearing.* If a
    batched request returns the wrong number of results, every result would
    attach to the wrong input and the metrics would be quietly meaningless.
    Spending extra quota to recover is strictly cheaper than that.

## Evaluation

15. **Golden set is 60 blind + 120 model-assisted, always reported separately.**
    Model-assisted labels bias agreement upward, so the 60 blind examples alone
    carry the judge-agreement and label-noise claims. Reporting a single blended
    number would have been the misleading version.

16. **Stratified sampling, over-sampling hard cases.** Reported accuracy
    therefore does *not* reflect production distribution. Disclosed in the
    misleading-number section rather than buried.

## Model selection (all probed against the live API, not assumed)

17. **`gemini-2.5-flash` is retired for new API keys.** It returns 404 with a
    pointer to newer models. Worth recording because every tutorial and most
    training data still names it, so the obvious first choice is now a dead end.

18. **Pinned `gemini-3.7-flash`, not `gemini-flash-latest`.** *load-bearing.*
    An alias would silently change model behind a committed response cache,
    which would make the cached numbers unreproducible *and* wrong in a way
    nobody would notice. `gemini-3.8-flash` was the newer option but returned
    503 "high demand" on every attempt, so it was rejected as unreliable.

19. **Cross-family judge is `gemma-4-31b-it`.** *load-bearing.* Judge and
    generator both being Gemini means self-enhancement bias, which is a headline
    caveat rather than a footnote. Gemma is a different model family on the same
    API, so judge/judge agreement there is much weaker evidence of *shared*
    bias. Gemini pro would have been the stronger judge but returns 429 on the
    free tier -- an availability constraint, not a quality judgement.

20. **Gemma needs fence-stripping.** It ignores `response_mime_type` and wraps
    output in ```json fences even with a schema attached, so JSON parsing falls
    back to stripping them. It also rejects `thinking_config` outright, as does
    `gemini-3.5-flash-lite`, hence the `NO_THINKING_CONFIG` set.

21. **`gemini-embedding-001`, not `gemini-embedding-2`.** The newer model
    returns exactly *one* vector regardless of how many texts you pass, so it
    cannot be batched -- embedding the corpus one text at a time would exhaust
    the daily quota by itself. Caught by an assertion comparing input and output
    counts, which is the only reason it was noticed at all.

22. **Embeddings truncated to 768 dims from 3072 (Matryoshka).** Measured on
    support text before committing to it: paraphrase similarity 0.765 vs 0.767
    at full width, with every relative ordering preserved, for a quarter of the
    storage. 3072 dims would have made the committed cache ~74MB.

23. **Embed batch size 50, not the documented 100.** 100 returns 429 for
    tweet-length input while 50 succeeds reliably. `_embed_live` halves the
    batch and retries on 429 rather than hard-coding a size that will be wrong
    for a different corpus.

24. **Embeddings cached per text, not per batch.** A per-batch cache key would
    be invalidated by any reordering or resampling of the corpus, re-spending
    the entire embedding budget. Per-text means only genuinely new text costs a
    request.

25. **Python pinned to 3.12 via `.python-version`.** uv resolved to 3.14 by
    default, which worked, but graders should run what was actually tested.

## What the data turned out to look like

26. **Sample is 15,000 of 80,702 AppleSupport threads, drawn at random with no
    stratification.** The first attempt at 4,000 was raised after checking the
    grounding material available. The subsample's turn distribution matches the
    population (median 2 turns, 24.8% with >=4 turns vs 24.9% overall), so
    there is no sampling bias to explain away.

27. **53% of AppleSupport's *first* replies are DM deflections.**
    *load-bearing.* Half the corpus contains no public resolution to ground a
    reply in. This is the single most important fact about this brand and it
    directly limits what "grounded reply" can honestly mean here.

28. **Median thread is 2 turns.** Only 24.8% reach 4 turns. The headline "106k
    Apple tweets" badly oversells the amount of usable resolution material;
    the real figure is thousands of threads, not tens of thousands.

## Taxonomy (milestone 4)

29. **Clusters are not well separated, and that is a finding.** *load-bearing.*
    Best silhouette was 0.085 at k=8 across a k=4..20 sweep
    (reports/silhouette.json). Anything below ~0.15 means there is essentially
    no cluster structure. So this taxonomy is *imposed on a continuum*, not
    discovered in one -- which means boundary cases are genuinely ambiguous and
    some classifier "errors" are disagreements a second human would also have.
    Reported rather than buried, because it caps how high intent accuracy can
    honestly go.

30. **The clustering split by register, not intent.** *load-bearing.* KMeans
    produced four near-duplicate groups of "my phone broke after the iOS 11
    update" differing mainly in profanity, and the LLM namer labelled the angry
    ones "not a support request". Merging them by intent and moving anger to an
    escalation signal is the single biggest hand-edit. Had it been left alone,
    the classifier would have learned that swearing changes what a customer
    needs.

31. **`how_to` was proposed, then dropped after measuring it.** The keyword
    probe returned 3.9% but the samples were overwhelmingly "how do I fix
    [battery]" -- a *phrasing* of other intents, not an intent. Kept, it would
    have competed with every other class for the same messages.

32. **`app_or_service_issue` was added, though clustering never surfaced it.**
    A prevalence probe found 7.5% -- bigger than four classes that clustering
    did surface. Clustering finds what is *textually* dominant, which is not the
    same as what matters; the iOS 11 noise drowned it out.

33. **Rare-but-dangerous classes kept despite tiny volume.** data_loss (0.7%),
    account_billing (0.9%) and device_crash_reboot (1.3%) would each get one or
    two examples in a natural-distribution golden set. They are kept as separate
    intents precisely because they are the escalation cases -- the ones where
    being wrong is expensive. Merging them into a generic bucket would have
    optimised the metric at the cost of the decision that matters.

34. **`not_actionable` is a first-class intent.** Venting, insults, jokes and
    feature requests are a large share of traffic. Forcing them into a support
    intent produces confidently irrelevant troubleshooting, which is a worse
    failure than admitting there is nothing to answer.

35. **The corpus is a late-2017 iOS 11 snapshot.** update_performance is 38% of
    traffic because of one OS rollout, and 25% of one cluster was a single
    keyboard bug. The taxonomy is period-specific and would not transfer to
    Apple's support stream today. Any headline accuracy here is accuracy on a
    frozen moment.

## Retrieval

36. **Retrieval is TF-IDF, not embeddings.** *load-bearing.* The free embedding
    quota is 1000 texts per *day*, and each text counts as one request rather
    than each batch -- so a batch of 100 consumes 100 requests and lands exactly
    on the 100-per-minute cap. That was the real cause of the 429s originally
    misread as a payload-size limit. A grounding corpus of a few thousand
    replies is therefore several days of quota, which is no basis for a pipeline
    graders must reproduce. sklearn TF-IDF is already a dependency, costs
    nothing, and runs instantly.

37. **Sharing TF-IDF between the baseline and the system is deliberate.** With
    retrieval held constant, the measured gap between "copy the nearest
    historical reply" and "draft from the same retrieved evidence" isolates
    exactly the LLM's drafting contribution. Changing retrieval *and* generation
    at once would have produced a bigger headline number that attributed
    nothing.

38. **Embeddings kept for offline clustering only.** 604 vectors were already
    paid for before the quota was understood, and 600 messages is ample to
    derive 8 intents. `llm.cached_subset` exists so the clustering could run on
    what was already bought instead of stalling a day for a fresh 1500.

39. **Golden set caps any intent at ~15% and floors the rare ones.** At natural
    distribution, 38% of the set would be one class and four classes would have
    under two examples. Both the capped and the natural-distribution numbers
    get reported; the gap between them is a concrete entry in the
    misleading-headline section.

## Grounding (milestone 5)

40. **"Grounded" had to be redefined downward, honestly.** *load-bearing.* The
    assignment asks for replies grounded in how the brand "historically
    resolved" similar issues. Apple's public record barely contains
    resolutions: 53.5% of replies are DM deflections, and 75.9% contain a t.co
    link to a support article whose content is not in the dataset. What the
    record *does* contain densely is Apple's first-response behaviour -- which
    question they ask first, which setting they point at, when they go private.
    That is what the system grounds in, and the report says so rather than
    claiming resolution quality it cannot have.

41. **URLs are replaced with a marker, not deleted.** An early measurement
    showed "0.0% of replies link an article", which was an artefact of
    `clean_text` stripping URLs before the regex ran. Since Apple's answer is
    frequently a bare link, deleting it turned real resolutions into "Try this
    out:" and made them look empty. Preserving `[support link]` took the usable
    corpus from 1,346 pairs (6.9%) to 6,230 (31.9%).

42. **The corpus filter is the project's biggest bias.** Dropping DM
    deflections removes over half the data, and those are disproportionately
    the *hard* cases -- the ones Apple judged too complex or account-specific to
    answer publicly. What survives over-represents problems with a tidy public
    answer, so retrieval quality is measured on an easier distribution than
    production. Reported, not buried.

43. **Duplicate replies stay in the index but are collapsed in results.**
    Apple's templated openers recur dozens of times, and that frequency is real
    ranking signal -- but five copies of one sentence is not five pieces of
    evidence, and would crowd the draft's context window.

## Escalation policy (milestone 6)

44. **Rule order encodes severity, because order picks the stated reason.**
    First match wins, so rules run most-severe first and a message with several
    problems is escalated for the worst one. There is a test asserting that a
    safety signal outranks a payment dispute.

45. **`decide()` re-validates the intent even though `classify()` already
    did.** A caught test failure, not a hypothetical: an off-taxonomy label has
    no risk level, so every risk-based rule would have silently passed it
    through to auto-handling. `decide` is the safety-critical function and must
    not depend on its caller having sanitised anything.

46. **Severe anger does NOT escalate on its own.** *load-bearing.* Profanity is
    the default register in this corpus, so escalating on it would route 30-40%
    of traffic to a human and defeat the system's purpose. Anger is passed to
    the drafting step to soften tone instead. This is a genuine policy judgement
    that could be wrong, it has an explicit test documenting the choice, and it
    is revisited in the failure analysis.

47. **Drafts are generated only for auto-handled messages.** Drafting for an
    escalated message spends quota on text nobody sends, and worse, invites an
    agent to paste a reply the policy just judged unsafe to send.

48. **A support draft citing no precedent is downgraded to escalate.**
    *load-bearing.* MIN_EVIDENCE_SCORE only proves similar precedent *exists*;
    it cannot tell whether the draft used any. Observed in the first end-to-end
    run: a fluent, generic reply with `used_evidence: []` that had passed the
    score gate. For a system whose central claim is groundedness, that is the
    exact case a human should see. `not_actionable` is exempt, since
    acknowledging venting without troubleshooting is correct.

49. **The escalation policy is robust to intent errors, by design.** In testing,
    an overheating complaint was classified `device_crash_reboot` (arguably
    wrong -- it is thermal, not a reboot) but still escalated correctly, because
    the safety signal fires independently of the intent. Decoupling the risk
    read from the classification is what makes that work.

## Baselines (milestone 7)

50. **The trivial baseline sends Apple's actual modal reply, not an invented
    one.** That reply -- the iOS 11 autocorrect workaround -- was sent 1,070
    times, and it addresses the bug that dominates this corpus. So the trivial
    baseline is genuinely hard to embarrass on a reply-quality rubric, which is
    the point: a system that cannot clear it by a wide margin has not earned its
    API bill.

51. **Baselines use hand-written rules, not a fitted classifier.**
    *load-bearing.* The golden set IS the test set, so fitting anything on it
    would leak. Keyword rules need no labels and are what a competent team ships
    in an afternoon before reaching for an LLM -- which makes them the honest
    thing to beat.

52. **The baseline was strengthened twice after watching it fail.** Its first
    draft only matched verb-then-noun ("lost my contacts") and so missed "all my
    contacts are gone"; and it had no notion of intent risk, so it auto-handled
    every hardware_repair message. Both were fixed. A deliberately weak baseline
    inflates the system's apparent gain, which is a subtler way of lying about
    the headline number than getting the metric wrong.

53. **The baseline's unmatched messages fall to the majority class, not
    not_actionable.** That is the choice which maximises its accuracy on an
    imbalanced set, so the comparison is against the baseline at its best.

54. **Both baselines share the agent's retriever.** With retrieval held
    constant, "copy the nearest historical reply" vs "draft from the same
    evidence" isolates the drafting step. The simple baseline copies verbatim
    and never paraphrases -- there is a test for that, because a paraphrasing
    control would quietly stop being a control.

## Quota, discovered the hard way (milestone 9)

55. **The free tier allows 20 generate requests per DAY, per model.**
    *load-bearing.* Not the ~250 the plan assumed. Found only when the assisted
    pre-labelling failed mid-session:
    `GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue: 20`. A full
    evaluation needs ~136 naive requests, which would have been seven days.

56. **Each stage runs on a different model, because the cap is per model.**
    classify/draft on `gemini-3.5-flash`, judging on `gemini-3.5-flash-lite`,
    pre-labels on `gemini-3.1-flash-lite`, cross-family on
    `gemma-4-26b-a4b-it`. A fresh run is ~15 + ~12 + ~4 requests, each inside
    its own cap. `gemini-3.7-flash` and `gemma-4-31b-it` are retired (exhausted
    and 503 respectively) but named in the code so their committed cache
    entries stay explicable.

57. **The quota constraint improved the design.** Judge and drafter were the
    same model, which is textbook self-enhancement bias. Being forced onto
    different models weakens that bias rather than merely disclosing it. Worth
    recording that the constraint produced a better experiment than the free
    choice did.

58. **Misaligned batches bisect rather than falling back per item.**
    *load-bearing.* With batch sizes raised to 10-30 to conserve quota, a
    per-item fallback would spend more than a day's entire quota recovering
    from one bad batch. Bisecting costs ~log2(n) and still guarantees every
    item is covered.

59. **Reply quality is judged on 90 examples, not all 180.** Judging three
    systems across 180 examples exceeds the daily cap. The same 90 are used for
    every system -- a different sample per system would make the comparison
    between them meaningless.

60. **Labels record whether a suggestion was actually shown.** When
    pre-labelling failed, the assisted pass silently degraded to blind. The
    report claims "120 assisted, model pre-labels, human corrects", and that
    claim is only true for records where a suggestion reached the screen. Now
    stored per record as `suggestion_shown`.
