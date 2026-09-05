"""S3: dense retrieval over raw notes with ``text-embedding-3-small``, ranked by cosine.

Every note's ``rendered`` string is embedded once; the question is embedded at query time; the
notes are ranked by cosine similarity, newest first on ties, and the top ones that fit the
budget are shown oldest first.  There is no index structure because 178 vectors do not need
one: the ranking is an exact dot product over unit vectors.

The model is reached through OpenRouter's OpenAI-compatible ``POST /embeddings`` with
anatid's own :class:`anatid.embed.OpenAICompatibleEmbedder`, wrapped in :class:`CachedEmbedder`
so every request is written to disk keyed by a hash of (model, dimensions, text) and a rerun
costs nothing and returns the same vectors.  :func:`default_embedder` builds that pair; the
harness hands the same object to the anatid systems so S3 and S4 rank with identical vectors.

The API key is read from ``OPEN_ROUTER_KEY`` (or ``OPENROUTER_API_KEY``), else from
``open_router_key=`` in the repository's ``.env``.  It is never logged, and it is not part of
any cache key.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from anatid.embed import Embedder, OpenAICompatibleEmbedder
from bench.quality.systems import (
    DEFAULT_BUDGET,
    Hit,
    Note,
    Retrieval,
    count_tokens,
    pack,
)

__all__ = [
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "OPENROUTER_BASE_URL",
    "CachedEmbedder",
    "VectorSystem",
    "default_embedder",
    "openrouter_key",
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIM = 1536

_KEY_VARS = ("OPEN_ROUTER_KEY", "OPENROUTER_API_KEY")
_KEY_NAMES = ("open_router_key", "openrouter_api_key")
_REPO_ROOT = Path(__file__).resolve().parents[3]


def openrouter_key(env_path: Path | str | None = None) -> str:
    """The OpenRouter key: the environment first, then ``.env`` at the repository root.

    Raises :class:`RuntimeError` when neither has it.  The value is returned, never printed.
    """
    for var in _KEY_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value
    env_file = Path(env_path) if env_path is not None else _REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip().lower() in _KEY_NAMES and value.strip():
                return value.strip().strip("\"'")
    raise RuntimeError(
        "no OpenRouter key: set OPEN_ROUTER_KEY or put open_router_key= in "
        f"{env_file} (the file is not read aloud)"
    )


# --------------------------------------------------------------------------------- caching


class CachedEmbedder:
    """An :class:`anatid.embed.Embedder` that remembers every vector on disk.

    The key of a text is ``sha256`` of the JSON ``{"kind", "model", "dim", "input"}``; the file
    ``<cache_dir>/embeddings/<key[:2]>/<key>.json`` holds the vector.  A batch is served from
    the cache text by text and only the misses go to ``inner`` (in one call, in order), so a
    corpus that gained one note costs one embedding.  ``stats`` counts hits, misses, the calls
    made to ``inner`` and the characters it was sent, which is what the report prices.
    """

    def __init__(self, inner: Embedder, cache_dir: Path | str, *, model: str | None = None):
        self.inner = inner
        self.dim = int(inner.dim)
        self.model = model or str(getattr(inner, "model", repr(inner)))
        self.cache_dir = Path(cache_dir) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.calls = 0
        self.chars_sent = 0

    def __repr__(self) -> str:
        return (
            f"CachedEmbedder(model={self.model!r}, dim={self.dim}, cache={str(self.cache_dir)!r})"
        )

    def key(self, text: str) -> str:
        payload = json.dumps(
            {"kind": "embedding", "model": self.model, "dim": self.dim, "input": text},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _read(self, key: str) -> list[float] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        vector = stored.get("embedding") if isinstance(stored, dict) else None
        if not isinstance(vector, list) or len(vector) != self.dim:
            return None
        return [float(x) for x in vector]

    def _write(self, key: str, text: str, vector: list[float]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {"model": self.model, "dim": self.dim, "input": text, "embedding": vector}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = [str(t) for t in texts]
        keys = [self.key(t) for t in items]
        out: list[list[float] | None] = [self._read(k) for k in keys]
        pending = [i for i, v in enumerate(out) if v is None]
        # The same text twice in one batch is one miss, not two.
        first_of: dict[str, int] = {}
        to_send: list[int] = []
        for i in pending:
            if keys[i] not in first_of:
                first_of[keys[i]] = i
                to_send.append(i)
        self.hits += len(items) - len(pending)
        self.misses += len(to_send)
        if to_send:
            self.calls += 1
            self.chars_sent += sum(len(items[i]) for i in to_send)
            fresh = self.inner.embed([items[i] for i in to_send])
            for i, vector in zip(to_send, fresh, strict=True):
                self._write(keys[i], items[i], vector)
            for i in pending:
                out[i] = fresh[to_send.index(first_of[keys[i]])]
        return [v for v in out if v is not None]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def stats(self) -> dict[str, Any]:
        looked_up = self.hits + self.misses
        return {
            "model": self.model,
            "dim": self.dim,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / looked_up) if looked_up else None,
            "calls": self.calls,
            "chars_sent": self.chars_sent,
        }


def default_embedder(
    cache_dir: Path | str,
    *,
    api_key: str | None = None,
    model: str = EMBEDDING_MODEL,
    dim: int = EMBEDDING_DIM,
    base_url: str = OPENROUTER_BASE_URL,
    batch_size: int = 64,
    timeout: float = 60.0,
) -> CachedEmbedder:
    """``text-embedding-3-small`` through OpenRouter, cached under ``cache_dir``.

    The key is resolved with :func:`openrouter_key` when not given.  The same object should be
    handed to every system that embeds, so they all rank with the same vectors.
    """
    inner = OpenAICompatibleEmbedder(
        base_url,
        api_key or openrouter_key(),
        model,
        dim,
        batch_size=batch_size,
        timeout=timeout,
    )
    return CachedEmbedder(inner, cache_dir, model=model)


# ---------------------------------------------------------------------------------- system


class VectorSystem:
    """Cosine over one embedding per note.  See the module docstring."""

    name: str

    def __init__(
        self,
        embedder: Embedder,
        *,
        order: str = "chronological",
        name: str | None = None,
    ) -> None:
        self.embedder = embedder
        self.order = order
        self.name = name or "vector"
        self._notes: list[Note] = []
        self._matrix: np.ndarray | None = None
        self.build_seconds = 0.0

    def __repr__(self) -> str:
        return f"VectorSystem(name={self.name!r}, embedder={self.embedder!r}, notes={len(self._notes)})"

    @staticmethod
    def _unit(rows: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return rows / norms

    # ------------------------------------------------------------------------------- build

    def build(self, notes: Sequence[Note]) -> None:
        started = time.perf_counter()
        self._notes = list(notes)
        vectors = self.embedder.embed([note.rendered for note in self._notes])
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(self._notes):
            raise RuntimeError(
                f"embedder returned shape {matrix.shape} for {len(self._notes)} notes"
            )
        self._matrix = self._unit(matrix) if len(self._notes) else matrix
        self.build_seconds = time.perf_counter() - started

    # ---------------------------------------------------------------------------- retrieve

    def rank(self, question: str) -> list[Hit]:
        """Every note by cosine to the question, best first, newest first on ties."""
        if self._matrix is None:
            raise RuntimeError("VectorSystem.build() has not run")
        if not self._notes:
            return []
        query = np.asarray(self.embedder.embed_one(question), dtype=np.float64)
        norm = float(np.linalg.norm(query))
        if norm > 0.0:
            query = query / norm
        scores = self._matrix @ query
        # Descending score, then descending seq: sort on the negatives, stable on the key.
        order = sorted(
            range(len(self._notes)),
            key=lambda i: (-float(scores[i]), -self._notes[i].seq),
        )
        return [
            Hit.of_note(self._notes[i], rank=position, score=float(scores[i]))
            for position, i in enumerate(order, start=1)
        ]

    def retrieve(self, question: str, *, budget: int | None = DEFAULT_BUDGET) -> Retrieval:
        started = time.perf_counter()
        ranked = self.rank(question)
        text, kept, skipped = pack(ranked, budget=budget, order=self.order)
        meta: dict[str, Any] = {
            "metric": "cosine",
            "tie_break": "newest first",
            "order": self.order,
            "build_seconds": self.build_seconds,
        }
        stats = getattr(self.embedder, "stats", None)
        if isinstance(stats, dict):
            meta["embedder"] = stats
        else:
            meta["embedder"] = repr(self.embedder)
        return Retrieval(
            system=self.name,
            question=question,
            budget=budget,
            text=text,
            hits=kept,
            tokens=count_tokens(text),
            considered=len(ranked),
            skipped=skipped,
            seconds=time.perf_counter() - started,
            meta=meta,
        )

    def close(self) -> None:
        """Nothing to release; the cache lives on disk."""
