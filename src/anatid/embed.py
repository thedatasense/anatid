"""Embeddings as a protocol: what anatid asks of an embedding model, and two implementations.

anatid stores a ``FLOAT[N]`` per memory and scores the vector arm of :meth:`anatid.Anatid.recall`
with cosine similarity over it.  Where the vectors come from is not the database's business,
and the core never calls a model on its own.  What it offers instead is one seam:

* :class:`Embedder` is the protocol.  ``embed(texts)`` returns one vector per text,
  ``embed_one(text)`` returns one vector, and ``dim`` says how long the vectors are so a
  mismatch with the database's ``embedding_dim`` is caught at :meth:`anatid.Anatid.open` rather
  than at the first write.
* ``Anatid.open(embedder=...)`` stores an implementation on the handle.  With one set,
  :meth:`~anatid.Anatid.remember` and :meth:`~anatid.Anatid.supersede` embed content they were
  not given an embedding for, and :meth:`~anatid.Anatid.recall` embeds the query so the vector
  arm runs.  An embedding passed explicitly always wins, and a handle with no embedder behaves
  exactly as before.

Two implementations ship:

:class:`OpenAICompatibleEmbedder`
    Any endpoint that speaks the OpenAI ``POST /embeddings`` shape: OpenAI itself, OpenRouter,
    Ollama, LM Studio, vLLM, text-embeddings-inference.  Standard library HTTP only; no client
    package is imported.
:class:`HashEmbedder`
    A deterministic, offline stand-in built on feature hashing of the words in the text.  It
    exists so tests, demos and CI can exercise the vector arm with no model and no network.  Two
    texts are similar under it exactly when they share words; it knows nothing about meaning,
    and it is not an embedding model in any quality sense.

Everything raised here on purpose is an :class:`EmbedderError` (transport, HTTP status, or a
response that does not have the documented shape) or an
:class:`~anatid.errors.EmbeddingDimensionError` (vectors of the wrong length).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from .errors import AnatidError, EmbeddingDimensionError

__all__ = [
    "Embedder",
    "EmbedderError",
    "HashEmbedder",
    "OpenAICompatibleEmbedder",
]


class EmbedderError(AnatidError):
    """An embedder could not produce vectors: transport failure, HTTP error, or a malformed
    response.  ``status`` is the HTTP status when there was one."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@runtime_checkable
class Embedder(Protocol):
    """What :class:`anatid.Anatid` asks of an embedding model.

    ``dim`` is the length of every vector the embedder returns.  ``embed`` takes a sequence of
    texts and returns one ``list[float]`` per text, in the same order.  ``embed_one`` is the
    single-text form.  Implementations are called from whichever thread runs the verb, so they
    must be safe to share between threads; both implementations here hold no mutable state.
    """

    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]: ...


def _check_vectors(vectors: list[list[float]], *, dim: int, expected: int) -> list[list[float]]:
    if len(vectors) != expected:
        raise EmbedderError(f"embedder returned {len(vectors)} vectors for {expected} texts")
    for vec in vectors:
        if len(vec) != dim:
            raise EmbeddingDimensionError(
                f"embedder returned a vector of {len(vec)} dimensions, expected {dim}",
                expected=dim,
                got=len(vec),
            )
    return vectors


# --------------------------------------------------------------------------- OpenAI-compatible


class OpenAICompatibleEmbedder:
    """``POST {base_url}/embeddings`` in the OpenAI request and response shape.

    ``base_url``
        The API root, e.g. ``https://api.openai.com/v1``, ``https://openrouter.ai/api/v1`` or
        ``http://localhost:11434/v1``.  A URL that already ends in ``/embeddings`` is used as
        is.
    ``api_key``
        Sent as ``Authorization: Bearer ...`` when given.  Local servers usually need none.
    ``model``
        The model name the endpoint expects, e.g. ``text-embedding-3-small``.
    ``dim``
        The length every returned vector must have.  It is the database's ``embedding_dim``.
        A vector of another length raises :class:`~anatid.errors.EmbeddingDimensionError`
        instead of being written.
    ``request_dimensions``
        Also send ``"dimensions": dim`` in the request body.  OpenAI's ``text-embedding-3``
        models honour it and shorten their output; many compatible servers reject unknown
        fields, so it is off by default.
    ``timeout``
        Seconds per request.
    ``batch_size``
        Texts per request in :meth:`embed`.

    Only the standard library is used (:mod:`urllib.request`), so this class adds no
    dependency to anatid and no client library's retry or logging policy.  A failed request
    raises :class:`EmbedderError` with the HTTP status when there was one.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        dim: int,
        *,
        request_dimensions: bool = False,
        timeout: float = 30.0,
        batch_size: int = 128,
    ) -> None:
        root = base_url.strip().rstrip("/")
        if not root:
            raise ValueError("base_url is empty")
        self.url = root if root.endswith("/embeddings") else root + "/embeddings"
        self.api_key = api_key or None
        self.model = model
        self.dim = int(dim)
        self.request_dimensions = bool(request_dimensions)
        self.timeout = float(timeout)
        self.batch_size = max(1, int(batch_size))
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

    def __repr__(self) -> str:
        return f"OpenAICompatibleEmbedder(url={self.url!r}, model={self.model!r}, dim={self.dim})"

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """One HTTP round trip.  Split out so a test can drive the class without a socket."""
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:  # noqa: BLE001 - the status is the message here
                pass
            raise EmbedderError(
                f"{self.url} answered HTTP {exc.code} for model {self.model!r}"
                + (f": {detail}" if detail else ""),
                status=int(exc.code),
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise EmbedderError(f"could not reach {self.url}: {exc}") from exc
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EmbedderError(f"{self.url} returned a body that is not JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise EmbedderError(f"{self.url} returned {type(parsed).__name__}, expected an object")
        return parsed

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        body: dict[str, Any] = {"model": self.model, "input": texts}
        if self.request_dimensions:
            body["dimensions"] = self.dim
        parsed = self._post(body)
        data = parsed.get("data")
        if not isinstance(data, list):
            raise EmbedderError(f"{self.url} returned no 'data' list; keys were {sorted(parsed)}")
        slots: list[list[float] | None] = [None] * len(texts)
        for position, item in enumerate(data):
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise EmbedderError(f"{self.url}: data[{position}] carries no 'embedding' list")
            index = item.get("index", position)
            if not isinstance(index, int) or not 0 <= index < len(texts):
                raise EmbedderError(f"{self.url}: data[{position}] has index {index!r}")
            slots[index] = [float(x) for x in item["embedding"]]
        if any(v is None for v in slots):
            raise EmbedderError(
                f"{self.url} returned {len(data)} embeddings for {len(texts)} texts"
            )
        return _check_vectors(
            [v for v in slots if v is not None], dim=self.dim, expected=len(texts)
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = [str(t) for t in texts]
        out: list[list[float]] = []
        for start in range(0, len(items), self.batch_size):
            out.extend(self._embed_batch(items[start : start + self.batch_size]))
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# --------------------------------------------------------------------------- hash stand-in

_WORD = re.compile(r"\w+", re.UNICODE)


class HashEmbedder:
    """A deterministic, offline stand-in for an embedding model.  Not for quality.

    Each word of the lower-cased text is hashed (BLAKE2b) to one of ``dim`` positions and a
    sign, the signs are summed per position, and the vector is L2-normalised.  Consequences,
    stated plainly:

    * The same text always gives the same vector, on every machine and every run.
    * Two texts score high under cosine exactly when they share words.  Synonyms, paraphrase
      and word order are invisible to it.
    * Collisions are frequent at small ``dim``; that is fine for a demo and useless for
      ranking a real corpus.

    It exists so tests, examples and the MCP server's ``--embed-hash`` flag can exercise the
    vector arm end to end with no model and no network.  Use a real model for anything whose
    ranking matters.  An empty text (no words) maps to the unit vector on position 0 rather
    than to the zero vector, because a zero vector makes cosine similarity undefined.
    """

    def __init__(self, dim: int, *, seed: int = 0) -> None:
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.seed = int(seed)
        self._salt = self.seed.to_bytes(8, "big", signed=True)

    def __repr__(self) -> str:
        return f"HashEmbedder(dim={self.dim}, seed={self.seed})"

    def embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in _WORD.findall(str(text).lower()):
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8, salt=self._salt).digest()
            position = int.from_bytes(digest[:4], "big") % self.dim
            vec[position] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [x / norm for x in vec]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]
