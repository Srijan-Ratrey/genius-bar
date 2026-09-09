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
