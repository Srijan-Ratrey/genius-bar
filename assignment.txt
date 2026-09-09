Hiver SDE Intern — Take-Home Assignment
What we are testing: whether you can turn a messy real-world dataset into a working AI system and prove it works. The proof is worth more than the system.

The problem
You are given real customer-support conversations between customers and brands on Twitter. Pick one brand from the dataset and build an AI support agent for it that can:

Classify each incoming customer message into a small set of intents that you define from the data.
Draft a reply grounded in how that brand has historically resolved similar issues.
Decide whether the message should be auto-handled or escalated to a human — with a stated reason.

You then have to convince us the agent is good enough to trust. That is the hard part.
Dataset
Primary: Customer Support on Twitter (Kaggle, thoughtvector/customer-support-on-twitter) — ~3M tweets, multi-turn threads, dozens of brands. Real, noisy, and imperfect.
Optional secondary (for intent work only): Banking77 (Hugging Face PolyAI/banking77) — 13k queries, 77 labelled intents.
You may use any LLM API or open model.
Deliverables
Repo with a runnable pipeline (README must let us reproduce your headline results in under 15 minutes).
Golden evaluation set — 150–250 hand-labelled examples you built yourself, with a short note on how you sampled and labelled them.
Evaluation harness — automated metrics + an LLM-as-judge rubric for reply quality, including evidence of how well your judge agrees with a human.
Report (max 6 pages / or a README section) covering:
Problem framing: what "good" means for this brand, and what you chose not to build.
Results vs. at least two baselines (a trivial one and a simple one).
Failure analysis: your top 5 failure modes with real examples and hypotheses.
"What is misleading about my headline number?" — a mandatory section.
What you'd do next with one more week.
Decision log — a plain list of the 10–15 non-obvious decisions you made and why. Bullet points are fine.
How to submit
Submit everything through this form: https://intelligent-bar-256.notion.site/39492cbf0da2800682cfc78a600a745f?pvs=105
Include the repo link (public, or private with access granted to us) and the report. Do not email submissions.
Rules
You may use AI coding assistants freely. We will ask you to explain and modify your own code live.
Cite anything you borrowed. Borrowing is fine; not knowing what you borrowed is not.
We will not run your code on the full dataset — a subsample is expected and encouraged.

