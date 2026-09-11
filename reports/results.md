# Results

Golden set: 180 examples (60 blind, 120 assisted)

## Intent classification

| system | acc (balanced) | macro-F1 | acc (weighted) | acc (blind only) |
|---|---|---|---|---|
| trivial | 0.183 | 0.039 | 0.345 | 0.183 |
| simple | 0.611 | 0.583 | 0.471 | 0.600 |
| agent | 0.833 | 0.820 | 0.843 | 0.783 |

## Escalation

| system | precision | recall | missed | needless | cost/100 (10:1) |
|---|---|---|---|---|---|
| trivial | 0.000 | 0.000 | 74 | 0 | 411.1 |
| simple | 0.660 | 0.865 | 10 | 33 | 73.9 |
| agent | 0.698 | 0.905 | 7 | 29 | 55.0 |

### Cost sensitivity

Ratio = cost of a missed escalation vs a needless one. Lower is better.

| system | 3:1 | 10:1 | 30:1 |
|---|---|---|---|
| trivial | 123.3 | 411.1 | 1233.3 |
| simple | 35.0 | 73.9 | 185.0 |
| agent | 27.8 | 55.0 | 132.8 |

## Reply quality (LLM judge, 1-5)

| system | grounded | helpful | tone | safety | mean | drafted | ungrounded | >280 chars |
|---|---|---|---|---|---|---|---|---|
| trivial | 1.09 | 2.18 | 2.80 | 1.13 | 1.80 | 180 | 100.0% | 0 |
| simple | 4.81 | 4.77 | 4.91 | 4.95 | 4.86 | 83 | 1.2% | 0 |
| agent | 4.95 | 4.97 | 5.00 | 4.95 | 4.97 | 84 | 1.2% | 0 |

## Cross-family judge check (Gemma vs Gemini)

- n = 22
- Spearman = undefined
- Gemma is -0.22 vs Gemini

Low correlation would mean much of the primary judge's score is
family-specific taste rather than reply quality.
