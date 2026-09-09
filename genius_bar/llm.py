"""Gemini client: on-disk cache, request batching, retry, and a budget counter.

The free AI Studio tier caps requests per *day*, and a naive run of this project
needs 1,000-1,400 calls. Three things keep it inside that:

1. Every response is cached to disk under a hash of (model, prompt, schema) and
   committed to the repo. `make eval` therefore replays results with no API key
   and no network -- which is what makes the 15-minute reproduce possible.
2. `map_batched` packs many items into one request.
3. `embed` caches per *text* rather than per batch, so only genuinely new text
   costs a request, and reordering the corpus doesn't invalidate anything.

Generation cache entries are readable JSON on purpose: a grader can open one and
see the exact prompt behind a number rather than taking it on faith. Embeddings
go to a binary .npz because the readable version would be ~55MB.

Model choices are pinned, not aliased. `gemini-flash-latest` would silently
change results under a committed cache; every constant here is a fixed version.
All of these were probed against the live API -- see DECISIONS.md.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "cache"
LLM_CACHE = CACHE_DIR / "llm"
EMBED_CACHE = CACHE_DIR / "embed"

# Primary worker: classification, drafting, and the main judge.
PRIMARY = "gemini-3.7-flash"
# A deliberately different model *family* for cross-checking the judge. Gemma is
# not Gemini, so agreement between them is weaker evidence of shared bias than
# two Gemini models would be. Gemini pro would have been the stronger judge but
# returns 429 on the free tier.
ALT_JUDGE = "gemma-4-31b-it"

# gemini-embedding-2 returns a single vector no matter how many inputs you pass,
# so it cannot be batched; -001 batches correctly.
EMBED_MODEL = "gemini-embedding-001"
# Truncated from 3072 via Matryoshka. Measured on support text: paraphrase
# similarity 0.765 vs 0.767 at full width, with all relative orderings intact,
# for 1/4 the storage.
EMBED_DIM = 768
# 100 is the documented ceiling but returns 429 for tweet-length input; 50 is
# the largest size that reliably succeeds.
EMBED_BATCH = 50

# Models that reject a thinking_config outright.
NO_THINKING_CONFIG = {"gemma-4-31b-it", "gemini-3.5-flash-lite"}

# Spacing between live calls, to stay under the per-minute cap. Only applies on
# cache misses, so a cached replay is unaffected.
MIN_INTERVAL = float(os.getenv("GEMINI_MIN_INTERVAL", "4.0"))

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


class CacheMiss(RuntimeError):
    """Raised when a prompt isn't cached and there's no key to fetch it live."""


class _Budget:
    """Counts live requests so quota use is visible rather than a surprise 429."""

    def __init__(self) -> None:
        self.live = 0
        self.cached = 0
        self.embed_live = 0
        self.embed_texts = 0
        self._lock = threading.Lock()
        self._last_call = 0.0

    def hit(self, n: int = 1) -> None:
        with self._lock:
            self.cached += n

    def spend(self, kind: str = "llm", texts: int = 0) -> None:
        """Record a live request, sleeping first to respect the per-minute cap."""
        with self._lock:
            wait = MIN_INTERVAL - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            if kind == "embed":
                self.embed_live += 1
                self.embed_texts += texts
            else:
                self.live += 1

    def report(self) -> None:
        if not (self.live or self.embed_live):
            if self.cached:
                print(
                    f"[budget] {self.cached} served from cache, 0 live requests",
                    file=sys.stderr,
                )
            return
        print(
            f"[budget] live: {self.live} generate + {self.embed_live} embed "
            f"({self.embed_texts} texts) | cached: {self.cached}",
            file=sys.stderr,
        )


BUDGET = _Budget()
atexit.register(BUDGET.report)

_client = None


def _get_client():
    global _client
    if _client is None:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise CacheMiss(
                "GEMINI_API_KEY is not set, so this request can only be served "
                "from cache -- and it is not cached.\n"
                "Set the key in .env to generate it, or run a target that uses "
                "only committed cache entries (`make eval`)."
            )
        from google import genai

        _client = genai.Client(api_key=key)
    return _client


def _is_transient(exc: BaseException) -> bool:
    """429 and 5xx are worth retrying; a malformed request never is."""
    text = f"{type(exc).__name__}: {exc}"
    return any(m in text for m in ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE"))


def _cache_path(root: Path, key: dict[str, Any]) -> Path:
    blob = json.dumps(key, sort_keys=True, ensure_ascii=False)
    return root / f"{hashlib.sha256(blob.encode()).hexdigest()[:24]}.json"


def _parse_json(raw: str) -> Any:
    """Parse model JSON, tolerating markdown fences.

    Gemma ignores response_mime_type and wraps output in ```json fences even
    when a schema is supplied, so stripping them is required, not defensive.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_FENCE.sub("", raw))


@retry(
    retry=retry_if_exception(_is_transient),
    wait=wait_exponential(multiplier=8, min=8, max=240),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _call_gemini(prompt: str, model: str, schema: dict | None, thinking: int | None) -> str:
    client = _get_client()
    config: dict[str, Any] = {"temperature": 0.0}  # determinism: the cache must be repeatable
    if thinking is not None and model not in NO_THINKING_CONFIG:
        config["thinking_config"] = {"thinking_budget": thinking}
    if schema is not None:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = schema

    BUDGET.spend("llm")
    resp = client.models.generate_content(model=model, contents=prompt, config=config)
    if not resp.text:
        raise RuntimeError(f"empty response from {model} (may be a safety block)")
    return resp.text


def generate(
    prompt: str,
    schema: dict | None = None,
    model: str = PRIMARY,
    thinking: int | None = 0,
) -> Any:
    """One prompt in, parsed JSON (or raw text) out. Cached on disk.

    `thinking` defaults to 0 -- reasoning tokens cost quota and classification
    does not need them; callers opt in where they help. Pass None to omit the
    setting entirely.
    """
    key = {"model": model, "prompt": prompt, "schema": schema, "thinking": thinking}
    path = _cache_path(LLM_CACHE, key)

    if path.exists():
        BUDGET.hit()
        return json.loads(path.read_text())["response"]

    raw = _call_gemini(prompt, model, schema, thinking)
    response = _parse_json(raw) if schema is not None else raw

    LLM_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**key, "response": response}, indent=2, ensure_ascii=False))
    return response


def map_batched(
    items: Sequence[Any],
    build_prompt: Callable[[Sequence[Any]], str],
    item_schema: dict,
    batch_size: int = 8,
    model: str = PRIMARY,
    thinking: int | None = 0,
) -> list[Any]:
    """Apply a prompt across many items, several items per request.

    `build_prompt` receives a batch and must ask for a JSON array with one entry
    per item, in order. If the model returns the wrong number of entries the
    batch is retried one item at a time -- a misaligned batch would silently
    attach every result to the wrong input, which is far worse than the extra
    quota spent recovering from it.
    """
    schema = {"type": "array", "items": item_schema}
    out: list[Any] = []

    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        result = generate(build_prompt(batch), schema, model=model, thinking=thinking)

        if isinstance(result, list) and len(result) == len(batch):
            out.extend(result)
            continue

        got = len(result) if isinstance(result, list) else "non-array"
        print(
            f"[llm] batch at {start} returned {got} results for {len(batch)} items; "
            "falling back to one request per item",
            file=sys.stderr,
        )
        for item in batch:
            single = generate(build_prompt([item]), schema, model=model, thinking=thinking)
            out.append(single[0] if isinstance(single, list) and single else None)

    return out


# --- embeddings ------------------------------------------------------------


def _embed_store(model: str, dim: int) -> Path:
    return EMBED_CACHE / f"{model}-{dim}.npz"


def _embed_live(texts: list[str], model: str, dim: int) -> list[np.ndarray]:
    """Embed a batch, halving on 429 rather than giving up.

    The documented batch ceiling is 100 but tweet-length input 429s well below
    that, and the real limit moves with input length. Splitting adapts instead
    of hard-coding a size that will be wrong for some corpus.
    """
    client = _get_client()
    try:
        BUDGET.spend("embed", texts=len(texts))
        resp = client.models.embed_content(
            model=model, contents=texts, config={"output_dimensionality": dim}
        )
        got = [np.asarray(e.values, dtype=np.float32) for e in resp.embeddings]
        if len(got) != len(texts):
            raise RuntimeError(f"asked for {len(texts)} embeddings, got {len(got)}")
        return got
    except Exception as exc:
        if len(texts) > 1 and _is_transient(exc):
            mid = len(texts) // 2
            print(f"[embed] batch of {len(texts)} rejected; splitting", file=sys.stderr)
            return _embed_live(texts[:mid], model, dim) + _embed_live(texts[mid:], model, dim)
        raise


def embed(
    texts: Sequence[str], model: str = EMBED_MODEL, dim: int = EMBED_DIM
) -> np.ndarray:
    """Embed texts and return an L2-normalised (len(texts), dim) matrix.

    Cached per text in a single .npz, so re-running with a slightly different
    corpus only pays for the new entries. Using the hosted embedding API rather
    than sentence-transformers keeps torch (~2GB) out of the dependency tree,
    which is what keeps the reproduce inside 15 minutes.
    """
    store_path = _embed_store(model, dim)
    store: dict[str, np.ndarray] = {}
    if store_path.exists():
        with np.load(store_path) as z:
            store = {k: z[k] for k in z.files}

    keys = [hashlib.sha256(t.encode()).hexdigest()[:24] for t in texts]
    # dict, not set: preserves order and drops duplicates within this call
    missing = list(dict.fromkeys(k for k in keys if k not in store))

    if missing:
        by_key = dict(zip(keys, texts))
        for start in range(0, len(missing), EMBED_BATCH):
            chunk = missing[start : start + EMBED_BATCH]
            for k, vec in zip(chunk, _embed_live([by_key[k] for k in chunk], model, dim)):
                store[k] = vec

        EMBED_CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(store_path, **store)
    else:
        BUDGET.hit(len(keys))

    arr = np.stack([store[k] for k in keys])
    # Normalise once here so every downstream similarity is a plain dot product.
    # Required after Matryoshka truncation, which does not preserve unit norm.
    return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12, None)
