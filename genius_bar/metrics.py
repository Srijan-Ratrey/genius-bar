"""Metric definitions, kept separate from the harness that calls them.

Three deliberate choices here, since the metric is the claim:

- **Macro-F1 alongside accuracy.** Apple's intent distribution is heavily
  skewed. Accuracy alone mostly measures whether you got the biggest class
  right, which is exactly what the trivial baseline exploits.
- **Escalation is scored by cost, not accuracy.** The two errors are not
  equally bad: auto-answering a payment dispute is a real incident, needlessly
  escalating a password reset costs a few minutes of an agent's time.
- **Judge scores are meaningless without an agreement number.** `agreement`
  exists so the LLM-judge results can be reported with a measured
  human-correlation attached rather than on trust.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# Relative cost of the two escalation errors. Ratio, not currency: a missed
# escalation is treated as ~10x worse than an unnecessary one. The exact number
# is a judgement call, so results are reported across a sweep of it too.
COST_MISSED_ESCALATION = 10.0
COST_NEEDLESS_ESCALATION = 1.0


def intent_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Accuracy plus macro-F1 and per-class F1. Labels come from the union of both."""
    labels = sorted(set(y_true) | set(y_pred))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": float(np.mean([t == p for t, p in zip(y_true, y_pred)])),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {lab: float(s) for lab, s in zip(labels, per_class)},
        "labels": labels,
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def escalation_metrics(
    y_true: list[bool],
    y_pred: list[bool],
    cost_missed: float = COST_MISSED_ESCALATION,
    cost_needless: float = COST_NEEDLESS_ESCALATION,
) -> dict:
    """Precision/recall on "should escalate", plus a cost-weighted error rate.

    `cost_per_100` is the headline: expected cost of running this policy over
    100 messages, in units where one needless escalation costs 1.
    """
    t = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)

    missed = int((t & ~p).sum())      # should have escalated, auto-handled it
    needless = int((~t & p).sum())    # could have auto-handled, escalated anyway

    precision, recall, f1, _ = precision_recall_fscore_support(
        t, p, average="binary", zero_division=0
    )
    total_cost = missed * cost_missed + needless * cost_needless
    return {
        "n": len(t),
        "escalation_rate": float(p.mean()) if len(p) else 0.0,
        "true_escalation_rate": float(t.mean()) if len(t) else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "missed_escalations": missed,
        "needless_escalations": needless,
        "cost_per_100": float(100 * total_cost / len(t)) if len(t) else 0.0,
    }


def agreement(human: list[float], judge: list[float], ordinal: bool = True) -> dict:
    """How well the LLM-judge tracks the human ratings.

    Reports both a chance-corrected exact-match (weighted kappa) and a rank
    correlation, because they fail differently: a judge that is consistently
    one point generous scores badly on kappa but near-perfectly on Spearman,
    and that distinction changes whether the judge is usable.
    """
    h = np.asarray(human, dtype=float)
    j = np.asarray(judge, dtype=float)
    if len(h) != len(j):
        raise ValueError(f"{len(h)} human ratings vs {len(j)} judge ratings")

    out: dict = {
        "n": len(h),
        "mean_human": float(h.mean()),
        "mean_judge": float(j.mean()),
        "judge_bias": float(j.mean() - h.mean()),  # >0 means the judge is generous
        "within_1": float(np.mean(np.abs(h - j) <= 1)),
        "exact_match": float(np.mean(h == j)),
    }

    # Rank correlation is undefined if either side never varies.
    if h.std() > 0 and j.std() > 0:
        rho, pval = spearmanr(h, j)
        out["spearman"] = float(rho)
        out["spearman_p"] = float(pval)
    else:
        out["spearman"] = None
        out["spearman_p"] = None

    if ordinal:
        # Quadratic weights: being 3 points off is much worse than 1 point off.
        out["kappa_quadratic"] = float(
            cohen_kappa_score(h.round().astype(int), j.round().astype(int), weights="quadratic")
        )
    else:
        out["kappa"] = float(cohen_kappa_score(h, j))
    return out


def self_consistency(first: list[str], second: list[str]) -> dict:
    """Agreement of one annotator with themselves across two passes.

    This is the ceiling: no classifier can be meaningfully scored above the rate
    at which the person who wrote the labels reproduces their own.
    """
    return {
        "n": len(first),
        "agreement": float(np.mean([a == b for a, b in zip(first, second)])),
        "kappa": float(cohen_kappa_score(first, second)),
    }
