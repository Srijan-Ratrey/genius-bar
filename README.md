# genius-bar

An AI support agent for **AppleSupport**, built from real customer-support
threads on Twitter. It does three things with an incoming customer message:

1. **Classify** it into an intent taxonomy derived from the data.
2. **Draft** a reply grounded in how Apple historically resolved similar issues.
3. **Decide** auto-handle vs escalate to a human, with a stated reason.

The assignment this was built for says *"the proof is worth more than the
system"*, so the evaluation harness, the hand-labelled golden set, and the
failure analysis are the primary deliverable. The agent is deliberately simple.

> **Status: in progress.** Build order and design rationale are in
> [plan.md](plan.md); non-obvious decisions are logged in
> [DECISIONS.md](DECISIONS.md). The report section lands here at milestone 12.

## Quickstart

```bash
uv sync          # create the venv, install deps
uv run pytest    # smoke tests
make eval        # reproduce headline results (no API key needed)
```

`make eval` replays a committed response cache, so it reproduces every headline
number without a Gemini key and without touching the network. See
[.env.example](.env.example) for the keys needed only to *regenerate* results.
