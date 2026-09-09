"""Derive an intent taxonomy from the data, rather than inventing one.

Deliberately a two-stage process. Clustering finds the structure that is
actually in the messages; the LLM only *names* what clustering found. Asking a
model to invent a taxonomy directly would produce something plausible and
unfalsifiable -- and "why these intents?" is the first question a reviewer asks.

Output is data/intents_draft.yaml, which is then hand-edited into
data/intents.yaml. The draft is committed too, so the edits are auditable.

Usage: uv run python scripts/derive_taxonomy.py [--sample 1500] [--k 14]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from genius_bar import llm
from genius_bar.data import REPO_ROOT, first_inbound, load_threads

DRAFT_PATH = REPO_ROOT / "data" / "intents_draft.yaml"

CLUSTER_SCHEMA = {
    "type": "object",
    "properties": {
        "cluster_id": {"type": "integer"},
        "name": {"type": "string", "description": "snake_case intent name, 1-3 words"},
        "description": {"type": "string", "description": "one sentence, what the customer wants"},
        "is_support_request": {
            "type": "boolean",
            "description": "false for venting, jokes, feature requests and general commentary",
        },
        "coherent": {
            "type": "boolean",
            "description": "false if these examples do not share a single intent",
        },
        "merge_with": {
            "type": "string",
            "description": "name of another cluster this duplicates, or empty string",
        },
    },
    "required": ["cluster_id", "name", "description", "is_support_request", "coherent", "merge_with"],
}


def choose_k(vectors: np.ndarray, candidates: range, seed: int = 0) -> list[tuple[int, float]]:
    """Silhouette score per k, so the cluster count is measured not guessed."""
    scores = []
    for k in candidates:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(vectors)
        scores.append((k, float(silhouette_score(vectors, labels, metric="cosine"))))
    return scores


def describe_clusters(
    texts: list[str], labels: np.ndarray, vectors: np.ndarray, centroids: np.ndarray, per: int = 8
) -> list[dict]:
    """Pull the examples closest to each centroid -- the cluster's clearest cases."""
    out = []
    for cid in range(len(centroids)):
        idx = np.where(labels == cid)[0]
        if not len(idx):
            continue
        # Vectors are unit-norm, so a dot product is cosine similarity.
        order = idx[np.argsort(-(vectors[idx] @ centroids[cid]))]
        out.append({
            "cluster_id": cid,
            "size": int(len(idx)),
            "examples": [texts[i] for i in order[:per]],
        })
    return out


def name_clusters(clusters: list[dict]) -> list[dict]:
    """Ask the model to name each cluster and flag incoherent or duplicate ones.

    It is explicitly allowed to say a cluster is incoherent. A model pushed to
    name everything will invent a label for noise, and that label then looks
    like a real intent for the rest of the project.
    """
    blocks = []
    for c in clusters:
        examples = "\n".join(f"    - {t[:220]}" for t in c["examples"])
        blocks.append(f"  cluster {c['cluster_id']} ({c['size']} messages):\n{examples}")
    body = "\n\n".join(blocks)

    prompt = f"""These are clusters of opening messages customers sent to Apple's
support account on Twitter. Each cluster is a group of semantically similar
messages, with the examples closest to the cluster centre shown.

Name each cluster as a customer-support intent.

Rules:
- name: snake_case, 1-3 words, describing what the CUSTOMER WANTS, not the
  device or feature involved. Prefer "battery_drain" over "iphone_problem".
- is_support_request: false for venting, jokes, insults, feature requests and
  general commentary with no answerable request. These are common here and are
  a legitimate category, not noise to be forced into a support intent.
- coherent: false if the examples plainly do not share one intent. Say so
  rather than inventing a label that papers over a mixed cluster.
- merge_with: if two clusters are the same intent, name the other one. Else "".

Return one object per cluster, in cluster_id order.

{body}
"""
    # A single call returning one object per cluster. map_batched is the wrong
    # tool here: it maps N items to N results, whereas this is 1 prompt -> N
    # results, which it would (correctly) flag as a misalignment.
    result = llm.generate(
        prompt, {"type": "array", "items": CLUSTER_SCHEMA}, thinking=4096
    )
    if not isinstance(result, list):
        raise SystemExit(f"expected a list of clusters, got {type(result).__name__}")
    if len(result) != len(clusters):
        print(f"[warn] named {len(result)} of {len(clusters)} clusters")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1500)
    ap.add_argument("--k", type=int, default=8, help="best silhouette on this corpus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--cached-only",
        action="store_true",
        help="cluster only messages already embedded, spending no quota",
    )
    args = ap.parse_args()

    first = first_inbound(load_threads())
    sample = first.sample(n=min(args.sample, len(first)), random_state=args.seed)
    texts = sample["clean"].tolist()

    if args.cached_only:
        texts = llm.cached_subset(texts)
        print(f"clustering {len(texts)} already-embedded messages (0 quota spent)")
        if len(texts) < 200:
            raise SystemExit(
                f"only {len(texts)} embedded messages available; too few to derive "
                "a taxonomy. Drop --cached-only and re-run when quota resets."
            )
    else:
        print(f"embedding {len(texts)} opening messages "
              f"(free tier allows 1000 texts/day)...")
    vectors = llm.embed(texts)

    print("\nsilhouette by k (cosine):")
    sweep = choose_k(vectors, range(4, 21, 2), args.seed)
    for k, score in sweep:
        print(f"  k={k:2}  {score:.4f}")
    best_k, best_score = max(sweep, key=lambda kv: kv[1])
    print(f"  best: k={best_k} at {best_score:.4f}")
    if best_score < 0.15:
        print("  NOTE: all scores are very low -- these messages do not form\n"
              "  well-separated clusters. Any taxonomy here is imposed on a\n"
              "  continuum, not discovered in it. This is a finding, not a bug.")
    (REPO_ROOT / "reports").mkdir(exist_ok=True)
    (REPO_ROOT / "reports" / "silhouette.json").write_text(
        json.dumps({"n_messages": len(texts), "sweep": sweep,
                    "best_k": best_k, "best_score": best_score}, indent=2))

    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10).fit(vectors)
    centroids = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    clusters = describe_clusters(texts, km.labels_, vectors, centroids)

    print(f"\nnaming {len(clusters)} clusters...")
    named = name_clusters(clusters)
    by_id = {n["cluster_id"]: n for n in named if n}

    draft = []
    for c in clusters:
        n = by_id.get(c["cluster_id"], {})
        draft.append({
            "cluster_id": c["cluster_id"],
            "name": n.get("name", f"cluster_{c['cluster_id']}"),
            "description": n.get("description", ""),
            "is_support_request": n.get("is_support_request", True),
            "coherent": n.get("coherent", True),
            "merge_with": n.get("merge_with", ""),
            "size": c["size"],
            "share": round(c["size"] / len(texts), 4),
            "examples": c["examples"][:5],
        })

    DRAFT_PATH.write_text(yaml.safe_dump(draft, sort_keys=False, allow_unicode=True, width=100))

    print(f"\n{'name':26} {'n':>5} {'share':>6}  support  coherent  merge_with")
    for d in sorted(draft, key=lambda x: -x["size"]):
        print(f"{d['name']:26} {d['size']:5} {d['share']:6.1%}  "
              f"{str(d['is_support_request']):7}  {str(d['coherent']):8}  {d['merge_with']}")
    print(f"\nwrote {DRAFT_PATH.relative_to(REPO_ROOT)} -- hand-edit into data/intents.yaml")


if __name__ == "__main__":
    main()
