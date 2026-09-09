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
