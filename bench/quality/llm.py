"""One model client for the answer-quality benchmark: OpenAI-compatible, cached, accounted.

Every model call the benchmark makes goes through :class:`LLMClient`: the extractor that
turns notes into patches, the embedder behind the vector arm and the vector baseline, the
answerer, and the judge.  Three things about it matter for the fairness of the benchmark:

* **Every call is cached on disk**, keyed by a SHA-256 of the request (endpoint, model,
  messages, parameters).  A second run with the same inputs makes no network call, costs
  nothing and returns the same bytes, so every number in the report is reproducible from the
  cache alone (``offline=True`` refuses to go to the network at all).  The hit rate is recorded.
* **Every call is accounted**: calls, cache hits, prompt and completion tokens, embedding
  tokens, wall time spent waiting on the network, and cost.  Cost comes from the provider's
  ``usage.cost`` field when it returns one (OpenRouter does, when asked), otherwise from an
  optional price table, otherwise it is reported as unknown rather than as zero.
* **The key never appears anywhere**: it is read from ``OPEN_ROUTER_KEY`` or from the
  ``open_router_key`` line of the repository's ``.env``, kept on the transport object, and is
  not part of the cache key, the cached file, any log line or any result file.

Tokens are counted with ``tiktoken`` (``cl100k_base``) when it is importable.  When it is not,
the documented estimate is one token per four characters, rounded up, applied identically to
every system; :attr:`TokenCounter.name` says which one a run used.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_EMBED_DIM",
    "ChatResult",
    "DiskCache",
    "HttpTransport",
    "LLMClient",
    "LLMError",
    "OfflineCacheMiss",
    "Stats",
    "TokenCounter",
    "load_api_key",
    "request_key",
]

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CHAT_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_EMBED_MODEL = "openai/text-embedding-3-small"
DEFAULT_EMBED_DIM = 1536

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"

#: Retry these HTTP statuses (rate limits and provider hiccups) with exponential backoff.
RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 6


class LLMError(RuntimeError):
    """A model call failed after retries, or the reply had no usable content."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class OfflineCacheMiss(LLMError):
    """``offline=True`` and the request was not in the cache."""


# --------------------------------------------------------------------------- the key


def load_api_key(env_var: str = "OPEN_ROUTER_KEY", *, dotenv: Path | None = None) -> str:
    """The OpenRouter key from ``OPEN_ROUTER_KEY`` or from ``.env`` (``open_router_key=...``).

    The value is placed in ``os.environ[env_var]`` for the rest of the process and returned.
    It is never printed.  Raises ``LookupError`` when neither source has it.
    """
    value = os.environ.get(env_var, "").strip()
    if value:
        return value
    path = dotenv if dotenv is not None else REPO_ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, raw = line.partition("=")
            if sep and key.strip().lower() in ("open_router_key", "openrouter_api_key"):
                value = raw.strip().strip("\"'")
                if value:
                    os.environ[env_var] = value
                    return value
    raise LookupError(
        f"no API key: set {env_var} or put open_router_key=... in {path} (never commit it)"
    )


# --------------------------------------------------------------------------- tokens


class TokenCounter:
    """Count tokens the same way for every system.

    ``tiktoken``'s ``cl100k_base`` when importable; otherwise ``ceil(len(text) / 4)``, the
    common four-characters-per-token estimate.  :attr:`name` records which one is in use so
    the report can say so.  ``mode`` pins the choice: ``"auto"`` prefers tiktoken and falls
    back, ``"chars4"`` uses the estimate whatever is installed, and ``"tiktoken"`` raises
    ``RuntimeError`` when the package is missing.  A run's counter is part of its budget, so a
    rerun that means to reproduce a report must pin the counter the report names; the runner's
    ``--counter`` flag and the report's reproduce command do that.
    """

    MODES = ("auto", "chars4", "tiktoken")

    def __init__(self, *, prefer_tiktoken: bool = True, mode: str | None = None) -> None:
        if mode is None:
            mode = "auto" if prefer_tiktoken else "chars4"
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self._encode: Callable[[str], list[int]] | None = None
        self.name = "chars/4"
        if mode in ("auto", "tiktoken"):
            try:
                import tiktoken  # type: ignore[import-not-found]

                enc = tiktoken.get_encoding("cl100k_base")
                self._encode = enc.encode
                self.name = "tiktoken:cl100k_base"
            except Exception as exc:  # noqa: BLE001 - any failure means "estimate"
                self._encode = None
                if mode == "tiktoken":
                    raise RuntimeError(
                        "--counter tiktoken asked for tiktoken's cl100k_base and it is not "
                        f"importable ({type(exc).__name__}: {exc})"
                    ) from exc

    @classmethod
    def for_name(cls, name: str) -> TokenCounter:
        """The counter a summary's ``tokenizer`` field names, pinned."""
        return cls(mode="tiktoken" if name.startswith("tiktoken") else "chars4")

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encode is not None:
            return len(self._encode(text))
        return math.ceil(len(text) / 4)


# --------------------------------------------------------------------------- the cache


def provider_failure(endpoint: str, response: Mapping[str, Any]) -> str | None:
    """Why a reply is the provider reporting a failure rather than the model answering, or None.

    OpenRouter (and the OpenAI-compatible providers behind it) can return HTTP 200 with a
    ``choices`` entry whose ``finish_reason`` is ``"error"`` and whose content is empty, or with
    an ``error`` object beside the choices, when the upstream model failed mid-request.  Such a
    reply must never be cached or scored as the model's answer: the 0.4.0 build of the committed
    corpus lost one note that way (``standup/boreal/2026-03-20``, six questions rest on it),
    and every offline replay reproduced the loss.  Embedding replies are validated by their
    caller; this covers chat completions.
    """
    if endpoint != "/chat/completions":
        return None
    err = response.get("error")
    if isinstance(err, Mapping) and err:
        return f"error: {err.get('message') or err.get('code') or 'unspecified'}"
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return "no choices"
    first = choices[0] if isinstance(choices[0], Mapping) else {}
    err = first.get("error")
    if isinstance(err, Mapping) and err:
        return f"choice error: {err.get('message') or err.get('code') or 'unspecified'}"
    if first.get("finish_reason") == "error":
        return "finish_reason is 'error'"
    return None


def request_key(endpoint: str, body: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of ``{endpoint, body}``; the cache file name."""
    payload = json.dumps(
        {"endpoint": endpoint, "body": body},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskCache:
    """JSON files under ``root/<first two hex>/<key>.json``; ``get`` returns None on a miss."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.hits = 0
        self.misses = 0

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.path(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.misses += 1
            return None
        self.hits += 1
        return data

    def put(self, key: str, value: Mapping[str, Any]) -> None:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)


# --------------------------------------------------------------------------- transport


class HttpTransport:
    """``POST {base_url}{path}`` with a bearer token; standard library only.

    The key lives here and nowhere else.  ``post`` returns the parsed JSON object or raises
    :class:`LLMError` with the status.  Retries :data:`RETRY_STATUSES` and transport errors
    with exponential backoff, up to :data:`MAX_ATTEMPTS`.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        *,
        timeout: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout = float(timeout)
        self._sleep = sleep
        self._headers = dict(headers or {})

    def __repr__(self) -> str:
        return f"HttpTransport({self.base_url!r})"

    def post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        headers.update(self._headers)
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        url = self.base_url + path
        last: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                break
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")[:500]
                except Exception:  # noqa: BLE001
                    pass
                last = LLMError(f"{url} answered HTTP {exc.code}: {detail}", status=int(exc.code))
                if exc.code not in RETRY_STATUSES:
                    raise last from exc
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                last = LLMError(f"could not reach {url}: {exc}")
            self._sleep(min(60.0, 2.0**attempt))
        else:
            assert last is not None
            raise last
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMError(f"{url} returned a body that is not JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMError(f"{url} returned {type(parsed).__name__}, expected an object")
        if "error" in parsed and "choices" not in parsed and "data" not in parsed:
            err = parsed["error"]
            message = err.get("message") if isinstance(err, dict) else str(err)
            code = err.get("code") if isinstance(err, dict) else None
            raise LLMError(f"{url} returned an error: {message}", status=code)
        return parsed


# --------------------------------------------------------------------------- accounting


@dataclass
class Stats:
    """Running totals.  ``wall_s`` is time spent waiting on the network in this run;
    ``recorded_wall_s`` adds the latency each cached call had when it was first made, so a
    replay reports the same network time as the run that filled the cache."""

    calls: int = 0
    cache_hits: int = 0
    network_calls: int = 0
    chat_calls: int = 0
    embed_calls: int = 0
    embed_texts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embed_tokens: int = 0
    cost_usd: float = 0.0
    cost_usd_charged: float = 0.0
    cost_known: bool = True
    wall_s: float = 0.0
    recorded_wall_s: float = 0.0
    #: Chat replies the provider marked as failed (``finish_reason: "error"``, an ``error``
    #: object) and this client refused to take as an answer; each was retried.
    provider_failures: int = 0

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / self.calls if self.calls else 0.0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["hit_rate"] = round(self.hit_rate, 4)
        return out

    def diff(self, earlier: Stats) -> Stats:
        """This minus ``earlier``: the accounting of one phase."""
        return Stats(
            calls=self.calls - earlier.calls,
            cache_hits=self.cache_hits - earlier.cache_hits,
            network_calls=self.network_calls - earlier.network_calls,
            chat_calls=self.chat_calls - earlier.chat_calls,
            embed_calls=self.embed_calls - earlier.embed_calls,
            embed_texts=self.embed_texts - earlier.embed_texts,
            prompt_tokens=self.prompt_tokens - earlier.prompt_tokens,
            completion_tokens=self.completion_tokens - earlier.completion_tokens,
            embed_tokens=self.embed_tokens - earlier.embed_tokens,
            cost_usd=self.cost_usd - earlier.cost_usd,
            cost_usd_charged=self.cost_usd_charged - earlier.cost_usd_charged,
            cost_known=self.cost_known and earlier.cost_known,
            wall_s=self.wall_s - earlier.wall_s,
            recorded_wall_s=self.recorded_wall_s - earlier.recorded_wall_s,
            provider_failures=self.provider_failures - earlier.provider_failures,
        )

    def copy(self) -> Stats:
        return Stats(**asdict(self))


@dataclass(frozen=True)
class ChatResult:
    """One chat completion: the text, whether it came from the cache, and the accounting.
    ``latency_s`` is the network time of the call when it was made, so a cached result carries
    the latency recorded with it rather than zero."""

    content: str
    cached: bool
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    latency_s: float
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


# --------------------------------------------------------------------------- the client


class LLMClient:
    """Chat completions and embeddings through one cached, accounted OpenAI-compatible client.

    ``transport`` is anything with ``post(path, body) -> dict``; the tests pass a fake and the
    benchmark passes :class:`HttpTransport`.  ``offline=True`` turns every cache miss into
    :class:`OfflineCacheMiss`, which is how a rerun proves it made no network call.
    ``prices`` maps a model name to ``(usd per million prompt tokens, usd per million
    completion tokens)`` for providers that do not return ``usage.cost``.
    """

    def __init__(
        self,
        *,
        chat_model: str = DEFAULT_CHAT_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        embed_dim: int = DEFAULT_EMBED_DIM,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        transport: Any | None = None,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        offline: bool = False,
        counter: TokenCounter | None = None,
        prices: Mapping[str, tuple[float, float]] | None = None,
        embed_batch_size: int = 64,
        request_usage: bool = True,
        clock: Callable[[], float] = time.perf_counter,
        retry_sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.embed_dim = int(embed_dim)
        self.cache = DiskCache(cache_dir)
        self.offline = bool(offline)
        self.counter = counter or TokenCounter()
        self.prices = dict(prices or {})
        self.embed_batch_size = max(1, int(embed_batch_size))
        self.request_usage = bool(request_usage)
        self._clock = clock
        self._retry_sleep = retry_sleep
        self.stats = Stats()
        self._lock = threading.RLock()
        if transport is None:
            if not offline:
                api_key = api_key or load_api_key()
            transport = HttpTransport(base_url, api_key)
        self.transport = transport

    # ------------------------------------------------------------------ helpers

    def snapshot(self) -> Stats:
        with self._lock:
            return self.stats.copy()

    def _post_cached(
        self, endpoint: str, body: dict[str, Any]
    ) -> tuple[dict[str, Any], bool, float]:
        """The response for ``body``, from the cache or the network.  ``(response, cached, latency)``."""
        key = request_key(endpoint, body)
        hit = self.cache.get(key)
        recorded = float(hit.get("latency_s") or 0.0) if hit is not None else 0.0
        usable = hit is not None and isinstance(hit.get("response"), dict)
        if usable and provider_failure(endpoint, hit["response"]) is not None:
            # A failed reply that an earlier run cached (before this check existed) is not an
            # answer: treat it as a miss and fetch again, so the store built from it heals.
            usable = False
            with self._lock:
                self.stats.provider_failures += 1
        with self._lock:
            self.stats.calls += 1
            if usable:
                self.stats.cache_hits += 1
                self.stats.recorded_wall_s += recorded
        if usable:
            assert hit is not None
            return hit["response"], True, recorded
        if self.offline:
            raise OfflineCacheMiss(
                f"offline run and {endpoint} request {key[:12]} is not in the cache at {self.cache.root}"
            )
        wire = dict(body)
        if endpoint == "/chat/completions" and self.request_usage:
            # OpenRouter returns usage.cost when asked; other providers ignore the field.
            wire["usage"] = {"include": True}
        latency = 0.0
        for attempt in range(MAX_ATTEMPTS):
            t0 = self._clock()
            response = self.transport.post(endpoint, wire)
            latency = self._clock() - t0
            with self._lock:
                self.stats.network_calls += 1
                self.stats.wall_s += latency
                self.stats.recorded_wall_s += latency
            why = provider_failure(endpoint, response)
            if why is None:
                break
            # The provider answered 200 with a failure inside: not cached, retried with the
            # same backoff the transport uses for a failing status, then reported.
            with self._lock:
                self.stats.provider_failures += 1
            if attempt + 1 >= MAX_ATTEMPTS:
                raise LLMError(
                    f"{endpoint} request {key[:12]} failed {MAX_ATTEMPTS} times at the provider "
                    f"({why}); the reply was not cached"
                )
            self._retry_sleep(min(60.0, 2.0**attempt))
        self.cache.put(
            key,
            {
                "endpoint": endpoint,
                "body": body,
                "response": response,
                "latency_s": latency,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        return response, False, latency

    def _cost(self, model: str, usage: Mapping[str, Any], *, embed: bool = False) -> float | None:
        raw = usage.get("cost")
        if isinstance(raw, (int, float)):
            # OpenRouter reports the price it charged; for a pass-through it can be 0 while the
            # upstream price is in cost_details.  Take the larger so nothing is under-reported.
            details = usage.get("cost_details")
            upstream = details.get("upstream_inference_cost") if isinstance(details, dict) else None
            if isinstance(upstream, (int, float)):
                return max(float(raw), float(upstream))
            return float(raw)
        price = self.prices.get(model)
        if price is None:
            return None
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        return (prompt * price[0] + completion * price[1]) / 1_000_000

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.0,
        response_format: Mapping[str, Any] | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ChatResult:
        """One chat completion at ``temperature`` (default 0), cached by its full request."""
        body: dict[str, Any] = {
            "model": model or self.chat_model,
            "messages": [dict(m) for m in messages],
            "temperature": float(temperature),
        }
        if response_format is not None:
            body["response_format"] = dict(response_format)
        if max_tokens is not None:
            body["max_tokens"] = int(max_tokens)
        if extra:
            body.update(extra)
        response, cached, latency = self._post_cached("/chat/completions", body)
        choices = response.get("choices") or []
        if not choices:
            raise LLMError(f"the model returned no choices: {json.dumps(response)[:300]}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
        content = str(content or "").strip()
        usage = response.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        cost = self._cost(body["model"], usage)
        with self._lock:
            self.stats.chat_calls += 1
            self.stats.prompt_tokens += prompt_tokens
            self.stats.completion_tokens += completion_tokens
            if cost is None:
                self.stats.cost_known = False
            else:
                self.stats.cost_usd += cost
                if not cached:
                    self.stats.cost_usd_charged += cost
        return ChatResult(
            content=content,
            cached=cached,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latency_s=latency,
            finish_reason=choices[0].get("finish_reason"),
            raw=response,
        )

    # ------------------------------------------------------------------ embeddings

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per text, each text cached on its own so a batch is never re-embedded."""
        items = [str(t) for t in texts]
        out: list[list[float] | None] = [None] * len(items)
        pending: list[int] = []
        for i, text in enumerate(items):
            body = {"model": self.embed_model, "input": text, "dimensions": self.embed_dim}
            key = request_key("/embeddings", body)
            hit = self.cache.get(key)
            with self._lock:
                self.stats.calls += 1
                if hit is not None and isinstance(hit.get("response"), dict):
                    self.stats.cache_hits += 1
            if hit is not None and isinstance(hit.get("response"), dict):
                vec = _first_embedding(hit["response"])
                self._account_embedding(hit["response"], cached=True)
                with self._lock:
                    self.stats.recorded_wall_s += float(hit.get("latency_s") or 0.0)
                out[i] = vec
            else:
                pending.append(i)
        if pending and self.offline:
            raise OfflineCacheMiss(
                f"offline run and {len(pending)} embedding request(s) are not in the cache"
            )
        for start in range(0, len(pending), self.embed_batch_size):
            batch = pending[start : start + self.embed_batch_size]
            body = {
                "model": self.embed_model,
                "input": [items[i] for i in batch],
                "dimensions": self.embed_dim,
            }
            t0 = self._clock()
            response = self.transport.post("/embeddings", body)
            latency = self._clock() - t0
            with self._lock:
                self.stats.network_calls += 1
                self.stats.wall_s += latency
                self.stats.recorded_wall_s += latency
            data = response.get("data")
            if not isinstance(data, list) or len(data) != len(batch):
                raise LLMError(
                    f"embeddings: expected {len(batch)} vectors, got "
                    f"{len(data) if isinstance(data, list) else 'no data'}"
                )
            usage = response.get("usage") or {}
            per_text = {
                k: (int(v) / len(batch) if isinstance(v, (int, float)) else v)
                for k, v in usage.items()
            }
            for position, item in enumerate(data):
                index = item.get("index", position)
                if not isinstance(index, int) or not 0 <= index < len(batch):
                    raise LLMError(f"embeddings: data[{position}] has index {index!r}")
                vec = [float(x) for x in item["embedding"]]
                if len(vec) != self.embed_dim:
                    raise LLMError(
                        f"embeddings: got a {len(vec)}-dimensional vector, expected {self.embed_dim}"
                    )
                single = {
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": vec}],
                    "model": response.get("model", self.embed_model),
                    "usage": per_text,
                }
                one_body = {
                    "model": self.embed_model,
                    "input": items[batch[index]],
                    "dimensions": self.embed_dim,
                }
                self.cache.put(
                    request_key("/embeddings", one_body),
                    {
                        "endpoint": "/embeddings",
                        "body": one_body,
                        "response": single,
                        "latency_s": latency / len(batch),
                    },
                )
                self._account_embedding(single, cached=False)
                out[batch[index]] = vec
        if any(v is None for v in out):
            raise LLMError("embeddings: a vector is missing from the response")
        return [v for v in out if v is not None]

    def _account_embedding(self, response: Mapping[str, Any], *, cached: bool) -> None:
        usage = response.get("usage") or {}
        tokens = usage.get("prompt_tokens", usage.get("total_tokens", 0))
        with self._lock:
            self.stats.embed_calls += 1
            self.stats.embed_texts += 1
            self.stats.embed_tokens += round(float(tokens or 0))
        cost = self._cost(self.embed_model, usage, embed=True)
        with self._lock:
            if cost is None:
                self.stats.cost_known = False
            else:
                self.stats.cost_usd += cost
                if not cached:
                    self.stats.cost_usd_charged += cost

    # ------------------------------------------------------------------ adapters

    def chat_client(self) -> Any:
        """An object with ``chat.completions.create(**kwargs)`` for
        :class:`anatid.ingest.OpenAICompatibleExtractor`, routed through :meth:`chat`."""
        return _ChatClientAdapter(self)

    def embedder(self) -> Any:
        """An :class:`anatid.embed.Embedder` (``dim``, ``embed``, ``embed_one``) routed through
        :meth:`embed`, so the vectors anatid stores are the cached ones."""
        return _EmbedderAdapter(self)


def _first_embedding(response: Mapping[str, Any]) -> list[float]:
    data = response.get("data") or []
    if not data or not isinstance(data[0], dict) or "embedding" not in data[0]:
        raise LLMError("cached embedding response has no vector")
    return [float(x) for x in data[0]["embedding"]]


class _ChatClientAdapter:
    """The slice of the ``openai`` client surface the extractor uses."""

    def __init__(self, client: LLMClient) -> None:
        outer = client

        class _Completions:
            def create(self, **kwargs: Any) -> dict[str, Any]:
                extra = dict(kwargs.get("extra_body") or {})
                result = outer.chat(
                    kwargs["messages"],
                    temperature=float(kwargs.get("temperature", 0.0) or 0.0),
                    response_format=kwargs.get("response_format"),
                    max_tokens=kwargs.get("max_tokens"),
                    model=kwargs.get("model"),
                    extra=extra or None,
                )
                return {
                    "choices": [{"message": {"content": result.content}}],
                    "usage": {
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                    },
                }

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()
        self._client = client

    def snapshot(self) -> dict[str, Any]:
        """The client's running totals as plain numbers, for callers that difference them."""
        return self._client.stats.to_dict()


class _EmbedderAdapter:
    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self.dim = client.embed_dim

    def __repr__(self) -> str:
        return f"CachedEmbedder(model={self._client.embed_model!r}, dim={self.dim})"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._client.embed(texts)

    def embed_one(self, text: str) -> list[float]:
        return self._client.embed([text])[0]
