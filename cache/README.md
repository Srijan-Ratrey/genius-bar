# Why this cache is committed

This directory is checked into git deliberately. It is a **fixture, not a
build artifact**: `make eval` replays it to recompute every headline number in
the README with **no API key and no network**.

Three things depend on that:

- **Reproducibility under 15 minutes.** A fresh run needs ~196 model requests.
  The Gemini free tier allows ~250 generate requests and 1,000 embedded texts
  per *day*, so a grader with their own key would be rate-limited before
  finishing. Replaying the fixture takes seconds.
- **Auditability.** Every entry in `llm/` is readable JSON holding the exact
  `model`, `prompt`, `schema` and `response` behind a number. Open one and
  check the claim rather than trusting the table.
- **Determinism.** All generation runs at `temperature=0`, so a cached response
  is what the same prompt would produce again.

## Layout

    llm/<sha256[:24]>.json   one generate() call, keyed on (model, prompt, schema, thinking)
    embed/<model>-<dim>.npz  embedding vectors keyed per text

The hash covers the prompt *and* the schema, so changing either produces a new
entry rather than silently reusing a stale one.

`embed/` was used only to cluster 600 messages while deriving the intent
taxonomy. Retrieval at runtime is TF-IDF and needs no embeddings, so this file
is finished and will not grow.

## Contents

Prompts contain customer tweets from the public Kaggle dataset. No credentials
are stored here -- `.env` is gitignored and no key is ever interpolated into a
prompt.

## Regenerating

`make clean-cache` deletes it (with a confirmation prompt), after which
`make eval` needs `GEMINI_API_KEY` and will take multiple days on the free tier
to rebuild. This is rarely what you want.
