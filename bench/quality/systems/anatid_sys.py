"""S4 anatid, its three single-arm ablations, and the S5 anatid-gold oracle.

The answer-quality benchmark asks whether an agent answers better from anatid than from a
Markdown file, BM25 or a vector index, at the same model, prompt and 1,200-token memory budget.
This module is anatid's side of that comparison.

S4 ``anatid``
    Every note goes through :func:`anatid.ingest.ingest` with
    :class:`~anatid.ingest.OpenAICompatibleExtractor` on GLM 5.3 Flash proposing the patch,
    the note's source label as ``writer`` and ``source``, and the note's date as ``now``, so
    every memory carries the validity interval and the writer that answer temporal and
    provenance questions.  The handle is opened with
    :class:`~anatid.embed.OpenAICompatibleEmbedder` on text-embedding-3-small, so the vector
    arm has real vectors.  At question time ``db.recall(question)`` runs every arm, with the
    graph arm seeded from the entities the question names, and :mod:`.render` packs the hits,
    each with its superseded predecessors, into the budget.
S4t / S4v / S4g ``anatid-text`` / ``anatid-vector`` / ``anatid-graph``
    The same stored memory, read through one arm at a time via
    :func:`anatid.recall.hybrid_recall`: BM25 alone, cosine alone, or the 2-hop graph
    expansion alone from the entities the question names.  They exist to say which arm earns
    its place.
S5 ``anatid-gold``
    The same pipeline fed the gold patches through :class:`~anatid.ingest.ScriptedExtractor`
    instead of the model.  It separates extraction quality from retrieval quality.  It is an
    oracle, and the report must call it one; it is not the product.

Reproducibility.  Every model call and every embedding request is cached on disk under a hash
of the full request (:class:`RequestCache`), so a rerun makes no network calls and produces the
same database.  For that to hold the memory ids the extractor sees in its prompt have to be
the same on every run, so the build installs a counter as anatid's id allocator
(:func:`deterministic_ids`, through the public :func:`anatid.ids.set_allocator`) and every row
is stamped with the note's date rather than the wall clock.  Costs and wall time are recorded
twice: what this run spent (``live``) and what the calls cost when they were first made
(``recorded``), so a cached rerun still reports the price of a cold one.

Nothing here decides what the report says.  The harness owns the prompt template, the judge
and the numbers; this module hands it a context string per question and says how it was made.

Standalone use, from the repository root::

    python -m bench.quality.systems.anatid_sys build --mode gold
    python -m bench.quality.systems.anatid_sys build --mode glm
    python -m bench.quality.systems.anatid_sys ask --mode gold "Which team owns the ledger now?"
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import hashlib
import itertools
import json
import logging
import os
import pathlib
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from anatid import Anatid, Memory, RecallHits, __version__ as anatid_version
from anatid.embed import Embedder, EmbedderError, OpenAICompatibleEmbedder
from anatid.errors import AnatidError
from anatid.ids import reset_allocator, set_allocator
from anatid.ingest import (
    ExtractionError,
    MemoryPatch,
    OpenAICompatibleExtractor,
    PatchReceipt,
    ScriptedExtractor,
    ingest,
)
from anatid.recall import auto_seeds, hybrid_recall

from .render import (
    DEFAULT_HISTORY_SHARE,
    LEGEND,
    RenderedContext,
    TokenCounter,
    render_context,
    tokenizer_name,
)

__all__ = [
    "ARMS",
    "BASE_URL",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_K",
    "DEFAULT_STORE_DIR",
    "EMBED_DIM",
    "EMBED_MODEL",
    "ENV_KEY",
    "MODEL",
    "SYSTEM_CODES",
    "TENANT",
    "AnatidStore",
    "AnatidSystem",
    "BuildStats",
    "CacheStats",
    "CachedChatClient",
    "CachedEmbedder",
    "Note",
    "RequestCache",
    "Retrieval",
    "build_glm_store",
    "build_gold_store",
    "close_systems",
    "deterministic_ids",
    "load_gold_patches",
    "load_notes",
    "load_openrouter_key",
    "make_chat_client",
    "make_embedder",
    "make_systems",
    "note_time",
    "request_hash",
]

log = logging.getLogger("bench.quality.anatid")

# --------------------------------------------------------------------------- configuration

#: The answering and extracting model, through OpenRouter's OpenAI-compatible endpoint.
MODEL = "z-ai/glm-5.3-flash"
#: The embedding model, through the same endpoint (``POST /embeddings``), 1536 dimensions.
EMBED_MODEL = "openai/text-embedding-3-small"
EMBED_DIM = 1536
BASE_URL = "https://openrouter.ai/api/v1"
#: The environment variable the key is read into.  It is never printed or logged.
ENV_KEY = "OPEN_ROUTER_KEY"
TENANT = 1
#: Hits asked of recall per question.  The token budget, not ``k``, is what cuts the context:
#: forty hits with their history is well past 1,200 tokens.
DEFAULT_K = 40
DEFAULT_CANDIDATES = 50
#: Where the counter allocator starts for each store.  Two stores are two files, so the bases
#: could coincide; they differ so an id in a log says which store it belongs to.
ID_BASE_GLM = 1_000_000
ID_BASE_GOLD = 2_000_000

_QUALITY_DIR = pathlib.Path(__file__).resolve().parents[1]
_REPO_ROOT = _QUALITY_DIR.parents[1]
DEFAULT_CACHE_DIR = _QUALITY_DIR / "cache"
DEFAULT_STORE_DIR = _QUALITY_DIR / "stores"

#: System name -> the arms its recall runs.  ``"all"`` is ``db.recall(question)`` unchanged.
ARMS: dict[str, tuple[str, ...]] = {
    "all": ("text", "vector", "graph"),
    "text": ("text",),
    "vector": ("vector",),
    "graph": ("graph",),
}

#: The names the report uses, and the codes the benchmark plan gave them.
SYSTEM_CODES: dict[str, str] = {
    "anatid": "S4",
    "anatid-text": "S4t",
    "anatid-vector": "S4v",
    "anatid-graph": "S4g",
    "anatid-gold": "S5",
}


# --------------------------------------------------------------------------- the key


def load_openrouter_key(env_path: pathlib.Path | None = None) -> str:
    """The OpenRouter key: ``OPEN_ROUTER_KEY`` in the environment, else ``open_router_key=``
    in the repository's ``.env``.  The value is exported into the environment and returned; it
    is never written anywhere else."""
    value = os.environ.get(ENV_KEY, "").strip()
    if value:
        return value
    path = env_path or (_REPO_ROOT / ".env")
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, rest = line.partition("=")
            if name.strip().lower() in ("open_router_key", "openrouter_api_key", ENV_KEY.lower()):
                secret = rest.strip().strip("'\"")
                if secret:
                    os.environ[ENV_KEY] = secret
                    return secret
    raise RuntimeError(f"no OpenRouter key: set {ENV_KEY} or put open_router_key=... in {path}")


# --------------------------------------------------------------------------- the cache


def request_hash(payload: Any) -> str:
    """SHA-256 of the canonical JSON of ``payload``: the cache key of one request."""
    text = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    """Counts for one cached client.  ``live`` is this run; ``recorded`` includes cache hits at
    the time they originally cost."""

    calls: int = 0
    hits: int = 0
    misses: int = 0
    live_seconds: float = 0.0
    recorded_seconds: float = 0.0

    @property
    def hit_rate(self) -> float | None:
        return None if self.calls == 0 else self.hits / self.calls

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["hit_rate"] = self.hit_rate
        return out


class RequestCache:
    """One JSON file per request under ``directory/namespace/<hash[:2]>/<hash>.json``.

    An entry holds the request, the response and how long the call took, so every prompt the
    benchmark ever sent is on disk next to its reply.
    """

    def __init__(self, directory: pathlib.Path | str, namespace: str) -> None:
        self.root = pathlib.Path(directory) / namespace
        self.namespace = namespace

    def path(self, key: str) -> pathlib.Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.path(key)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as fh:
            entry = json.load(fh)
        return entry if isinstance(entry, dict) and "response" in entry else None

    def put(self, key: str, request: Any, response: Any, elapsed_s: float) -> dict[str, Any]:
        entry = {
            "key": key,
            "namespace": self.namespace,
            "created_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            "elapsed_s": float(elapsed_s),
            "request": request,
            "response": response,
        }
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(entry, fh, ensure_ascii=False, default=str)
        tmp.replace(path)
        return entry


@dataclass
class UsageTally:
    """Token and cost totals read from the ``usage`` object of chat replies."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    cost_known: int = 0  # replies that carried a cost

    def add(self, usage: Mapping[str, Any] | None) -> None:
        self.calls += 1
        if not isinstance(usage, Mapping):
            return
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details")
        if isinstance(details, Mapping):
            self.reasoning_tokens += int(details.get("reasoning_tokens") or 0)
        cost = usage.get("cost")
        if isinstance(cost, (int, float)):
            self.cost_usd += float(cost)
            self.cost_known += 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Completions:
    def __init__(self, owner: CachedChatClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> dict[str, Any]:
        return self._owner.create(**kwargs)


class _Chat:
    def __init__(self, owner: CachedChatClient) -> None:
        self.completions = _Completions(owner)


class CachedChatClient:
    """``client.chat.completions.create`` with a disk cache in front of it.

    Anything with ``chat.completions.create`` works as ``inner``; the extractor and the
    answering harness both accept this object where they accept an ``openai.OpenAI``.  The
    reply is returned as a plain dict (the SDK object's ``model_dump()``), which
    :class:`~anatid.ingest.OpenAICompatibleExtractor` reads as well as it reads the object.
    ``extra_body`` is merged into every request; ``{"usage": {"include": True}}`` is what
    makes OpenRouter report the cost of each call.  A transport error is retried ``retries``
    times with exponential backoff, then raised: a network failure must stop the build, not be
    stored as "this note said nothing".
    """

    def __init__(
        self,
        inner: Any,
        cache: RequestCache,
        *,
        extra_body: Mapping[str, Any] | None = None,
        retries: int = 3,
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.extra_body = dict(extra_body or {"usage": {"include": True}})
        self.retries = max(0, int(retries))
        self.stats = CacheStats()
        self.live = UsageTally()
        self.recorded = UsageTally()
        self.chat = _Chat(self)

    def _request(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        request = dict(kwargs)
        if self.extra_body:
            merged = dict(request.get("extra_body") or {})
            for key, value in self.extra_body.items():
                merged.setdefault(key, value)
            request["extra_body"] = merged
        return request

    def _call(self, request: dict[str, Any]) -> Any:
        attempt = 0
        while True:
            try:
                return self.inner.chat.completions.create(**request)
            except Exception as exc:  # noqa: BLE001 - retried, then re-raised below
                attempt += 1
                if attempt > self.retries:
                    raise
                delay = 2.0**attempt
                log.warning(
                    "chat call failed (%s: %s); retry %d/%d in %.0fs",
                    type(exc).__name__,
                    str(exc)[:200],
                    attempt,
                    self.retries,
                    delay,
                )
                time.sleep(delay)

    def create(self, **kwargs: Any) -> dict[str, Any]:
        request = self._request(kwargs)
        key = request_hash({"kind": "chat.completions", "request": request})
        self.stats.calls += 1
        entry = self.cache.get(key)
        if entry is None:
            started = time.perf_counter()
            response = self._call(request)
            elapsed = time.perf_counter() - started
            data = response.model_dump() if hasattr(response, "model_dump") else dict(response)
            entry = self.cache.put(key, request, data, elapsed)
            self.stats.misses += 1
            self.stats.live_seconds += elapsed
            self.live.add(data.get("usage"))
        else:
            self.stats.hits += 1
        self.stats.recorded_seconds += float(entry.get("elapsed_s") or 0.0)
        self.recorded.add(entry["response"].get("usage"))
        return entry["response"]

    def snapshot(self) -> dict[str, Any]:
        return {
            "cache": self.stats.to_dict(),
            "live": self.live.to_dict(),
            "recorded": self.recorded.to_dict(),
        }


class CachedEmbedder:
    """An :class:`~anatid.embed.Embedder` with a disk cache in front of another one.

    Texts are cached one by one under a hash of ``(model, dim, text)``; a batch call embeds
    only the misses, in one request to ``inner``.  ``stats.calls`` counts texts, not requests.
    """

    def __init__(self, inner: Embedder, cache: RequestCache) -> None:
        self.inner = inner
        self.cache = cache
        self.dim = int(inner.dim)
        self.model = str(getattr(inner, "model", repr(inner)))
        self.stats = CacheStats()
        self.requests = 0

    def __repr__(self) -> str:
        return f"CachedEmbedder({self.inner!r})"

    def _key(self, text: str) -> str:
        return request_hash(
            {"kind": "embeddings", "model": self.model, "dim": self.dim, "text": text}
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = [str(t) for t in texts]
        out: list[list[float] | None] = [None] * len(items)
        missing: list[int] = []
        for i, text in enumerate(items):
            entry = self.cache.get(self._key(text))
            if entry is None:
                missing.append(i)
                continue
            out[i] = [float(x) for x in entry["response"]]
            self.stats.hits += 1
            self.stats.recorded_seconds += float(entry.get("elapsed_s") or 0.0)
        self.stats.calls += len(items)
        if missing:
            started = time.perf_counter()
            vectors = self.inner.embed([items[i] for i in missing])
            elapsed = time.perf_counter() - started
            self.requests += 1
            each = elapsed / len(missing)
            for i, vec in zip(missing, vectors, strict=True):
                self.cache.put(
                    self._key(items[i]),
                    {"model": self.model, "dim": self.dim, "text": items[i]},
                    vec,
                    each,
                )
                out[i] = vec
            self.stats.misses += len(missing)
            self.stats.live_seconds += elapsed
            self.stats.recorded_seconds += elapsed
        return [v for v in out if v is not None]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def snapshot(self) -> dict[str, Any]:
        """Counters only, so a difference of two snapshots is a difference of counts."""
        out = self.stats.to_dict()
        out["requests"] = self.requests
        return out


def make_embedder(
    api_key: str | None = None, *, cache_dir: pathlib.Path | str = DEFAULT_CACHE_DIR
) -> CachedEmbedder:
    """text-embedding-3-small through OpenRouter, cached.  Reads the key when not given."""
    inner = OpenAICompatibleEmbedder(
        BASE_URL, api_key or load_openrouter_key(), EMBED_MODEL, EMBED_DIM, timeout=60.0
    )
    return CachedEmbedder(inner, RequestCache(cache_dir, "embeddings"))


def make_chat_client(
    api_key: str | None = None,
    *,
    cache_dir: pathlib.Path | str = DEFAULT_CACHE_DIR,
    extra_body: Mapping[str, Any] | None = None,
) -> CachedChatClient:
    """An ``openai.OpenAI`` on OpenRouter behind :class:`CachedChatClient`."""
    from openai import OpenAI

    inner = OpenAI(base_url=BASE_URL, api_key=api_key or load_openrouter_key(), timeout=180.0)
    return CachedChatClient(inner, RequestCache(cache_dir, "chat"), extra_body=extra_body)


# --------------------------------------------------------------------------- determinism


@contextlib.contextmanager
def deterministic_ids(start: int) -> Iterator[Callable[[], int]]:
    """Make :func:`anatid.ids.new_id` a counter from ``start`` for the duration of the block.

    The extractor's prompt lists existing memories by id, so with time-based ids the second
    run of a build would send different prompts and miss the cache.  The counter is installed
    through the public :func:`anatid.ids.set_allocator` and restored on exit.  It is process
    wide: do not run another anatid writer in the same process while a build is under way.
    """
    counter = itertools.count(int(start))
    lock = threading.Lock()

    def allocate() -> int:
        with lock:
            return next(counter)

    set_allocator(allocate)
    try:
        yield allocate
    finally:
        reset_allocator()


def note_time(date: str | _dt.date, seq: int) -> _dt.datetime:
    """The instant a note's facts take effect: 09:00 UTC on its date plus ``seq`` minutes, so
    two notes on one day keep their arrival order.  The same rule ``gen_corpus --verify``
    uses, so the S5 store is the graph the corpus was verified against."""
    day = date if isinstance(date, _dt.date) else _dt.date.fromisoformat(str(date))
    base = _dt.datetime.combine(day, _dt.time(9, 0, tzinfo=_dt.timezone.utc))
    return base + _dt.timedelta(minutes=int(seq))


# --------------------------------------------------------------------------- the corpus


@dataclass(frozen=True)
class Note:
    """The fields of one ``notes.jsonl`` row the systems need."""

    note_id: str
    seq: int
    date: str
    source: str
    rendered: str

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Note:
        return cls(
            note_id=str(row["note_id"]),
            seq=int(row["seq"]),
            date=str(row["date"]),
            source=str(row["source"]),
            rendered=str(row["rendered"]),
        )


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_notes(path: pathlib.Path | None = None) -> list[Note]:
    """``notes.jsonl`` in arrival order."""
    from bench.quality import NOTES_PATH

    return [Note.from_dict(row) for row in _read_jsonl(path or NOTES_PATH)]


def load_gold_patches(path: pathlib.Path | None = None) -> list[dict[str, Any]]:
    """``gold_patches.jsonl`` rows, one per note, in note order."""
    from bench.quality import GOLD_PATCHES_PATH

    return _read_jsonl(path or GOLD_PATCHES_PATH)


# --------------------------------------------------------------------------- the build


@dataclass
class BuildStats:
    """What ingestion did and cost, kept apart from answering."""

    mode: str
    notes: int = 0
    build_seconds: float = 0.0
    reused: bool = False
    episodes: int = 0
    memories_created: int = 0
    memories_closed: int = 0
    corrections: int = 0
    relations_opened: int = 0
    relations_closed: int = 0
    aliases: int = 0
    downgraded_corrections: int = 0
    dedupe_drops: int = 0
    parser_notes: int = 0
    empty_patches: int = 0
    extraction_failures: int = 0
    apply_failures: int = 0
    model: dict[str, Any] = field(default_factory=dict)
    embeddings: dict[str, Any] = field(default_factory=dict)
    graph: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BuildStats:
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in names})


def _sanitize(patch: MemoryPatch) -> MemoryPatch:
    """The review hook for the model's patches: a second correction of the same memory in one
    patch would make ``supersede`` refuse to fork the chain and roll the whole patch back, so
    only the first is kept and the drop is noted."""
    seen: set[int] = set()
    kept = []
    notes: list[str] = []
    for corr in patch.corrections:
        if corr.old_id is not None:
            if corr.old_id in seen:
                notes.append(
                    f"bench: dropped a second correction of memory {corr.old_id} in one patch "
                    f"({corr.new_content!r})"
                )
                continue
            seen.add(corr.old_id)
        kept.append(corr)
    if len(kept) == len(patch.corrections):
        return patch
    return patch.replace(corrections=tuple(kept), notes=patch.notes + tuple(notes))


def _tally(stats: BuildStats, receipt: PatchReceipt) -> None:
    stats.episodes += 1
    stats.memories_created += len(receipt.memories_created)
    stats.memories_closed += len(receipt.memories_closed)
    stats.corrections += len(receipt.corrections)
    stats.relations_opened += len(receipt.relations_opened)
    stats.relations_closed += len(receipt.relations_closed)
    stats.aliases += len(receipt.aliases)
    if receipt.changes == 0:
        stats.empty_patches += 1
    for line in receipt.patch.notes:
        if line.startswith("correction of"):
            stats.downgraded_corrections += 1
        elif line.startswith("dedupe:"):
            stats.dedupe_drops += 1
        elif line.startswith("parser:"):
            stats.parser_notes += 1


def _ingest_note(
    db: Anatid, note: Note, extractor: Any, when: _dt.datetime, stats: BuildStats
) -> dict[str, Any]:
    """Ingest one note; when the model's patch cannot be read or applied, store the episode
    alone so the record shows the note was read, and count the failure."""
    started = time.perf_counter()
    failure: str | None = None
    receipt: PatchReceipt | None = None
    try:
        receipt = ingest(
            db,
            note.rendered,
            extractor=extractor,
            writer=note.source,
            source=note.source,
            now=when,
            review=_sanitize,
        )
    except ExtractionError as exc:
        failure = f"extraction: {exc}"
        stats.extraction_failures += 1
    except EmbedderError:
        # A transport failure, not a fact about the note: it must stop the build rather than
        # be stored as "this note said nothing".
        raise
    except AnatidError as exc:
        failure = f"apply: {type(exc).__name__}: {exc}"
        stats.apply_failures += 1
    if receipt is None:
        log.warning("%s (%s): %s; storing the episode only", note.note_id, note.source, failure)
        receipt = MemoryPatch(source_text=note.rendered).apply(
            db, writer=note.source, source=note.source, now=when
        )
    _tally(stats, receipt)
    return {
        "note_id": note.note_id,
        "source": note.source,
        "date": note.date,
        "at": when.isoformat(),
        "episode_id": str(receipt.episode_id),
        "memories_created": [str(i) for i in receipt.memories_created],
        "memories_closed": [str(i) for i in receipt.memories_closed],
        "corrections": [[str(o), str(n)] for o, n in receipt.corrections],
        "relations_opened": len(receipt.relations_opened),
        "relations_closed": len(receipt.relations_closed),
        "aliases": [list(pair) for pair in receipt.aliases],
        "notes": list(receipt.patch.notes),
        "patch": receipt.patch.describe(),
        "failure": failure,
        "seconds": round(time.perf_counter() - started, 4),
    }


def _sha(items: Iterable[Any]) -> str:
    return request_hash(list(items))


def _manifest(
    *,
    mode: str,
    notes: Sequence[Note],
    embedder: Embedder,
    id_base: int,
    patches: Sequence[Mapping[str, Any]] | None,
    extractor: OpenAICompatibleExtractor | None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "anatid": anatid_version,
        "tenant": TENANT,
        "id_base": id_base,
        "notes": len(notes),
        "notes_sha": _sha((n.note_id, n.seq, n.date, n.source, n.rendered) for n in notes),
        "patches_sha": None if patches is None else _sha(patches),
        "embed_model": str(getattr(embedder, "model", repr(embedder))),
        "dim": int(embedder.dim),
        "model": None if extractor is None else extractor.model,
        "system_prompt_sha": None if extractor is None else request_hash(extractor.system_prompt),
        "extractor": None
        if extractor is None
        else {
            "temperature": extractor.temperature,
            "response_format": extractor.response_format,
            "max_existing": extractor.max_existing,
            "extra_body": extractor.extra_body,
        },
    }


@dataclass
class AnatidStore:
    """A built store: the open handle, where it lives, and what building it cost."""

    name: str
    mode: str
    path: pathlib.Path
    db: Anatid
    stats: BuildStats
    manifest: dict[str, Any]

    @property
    def receipts_path(self) -> pathlib.Path:
        return self.path.with_suffix(".receipts.jsonl")

    def close(self) -> None:
        if not self.db.closed:
            self.db.close()


def _sidecars(path: pathlib.Path) -> list[pathlib.Path]:
    return [
        path,
        path.with_name(path.name + ".wal"),
        path.with_suffix(".manifest.json"),
        path.with_suffix(".stats.json"),
        path.with_suffix(".receipts.jsonl"),
    ]


def _build_store(
    notes: Sequence[Note],
    *,
    name: str,
    mode: str,
    path: pathlib.Path,
    extractor_factory: Callable[[], Any],
    embedder: Embedder,
    id_base: int,
    manifest: dict[str, Any],
    chat: CachedChatClient | None,
    rebuild: bool,
    progress: Callable[[int, int, dict[str, Any]], None] | None,
) -> AnatidStore:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = path.with_suffix(".manifest.json")
    stats_path = path.with_suffix(".stats.json")
    receipts_path = path.with_suffix(".receipts.jsonl")

    if not rebuild and path.exists() and manifest_path.exists() and stats_path.exists():
        with manifest_path.open(encoding="utf-8") as fh:
            stored = json.load(fh)
        if stored == manifest:
            with stats_path.open(encoding="utf-8") as fh:
                stats = BuildStats.from_dict(json.load(fh))
            stats.reused = True
            db = Anatid.open(path, tenant=TENANT, embedding_dim=embedder.dim, embedder=embedder)
            log.info("%s: reused %s (%d notes)", name, path.name, stats.notes)
            return AnatidStore(name, mode, path, db, stats, manifest)
        log.info("%s: manifest changed; rebuilding %s", name, path.name)

    for sidecar in _sidecars(path):
        if sidecar.exists():
            sidecar.unlink()

    stats = BuildStats(mode=mode, notes=len(notes))
    chat_before = chat.snapshot() if chat is not None else None
    embed_before = embedder.snapshot() if isinstance(embedder, CachedEmbedder) else None
    started = time.perf_counter()
    extractor = extractor_factory()
    with (
        deterministic_ids(id_base),
        Anatid.open(path, tenant=TENANT, embedding_dim=embedder.dim, embedder=embedder) as db,
        receipts_path.open("w", encoding="utf-8") as receipts,
    ):
        for index, note in enumerate(notes, start=1):
            record = _ingest_note(db, note, extractor, note_time(note.date, note.seq), stats)
            receipts.write(json.dumps(record, ensure_ascii=False) + "\n")
            receipts.flush()
            if progress is not None:
                progress(index, len(notes), record)
        stats.graph = db.stats()
    stats.build_seconds = time.perf_counter() - started
    if chat is not None and chat_before is not None:
        stats.model = _delta(chat.snapshot(), chat_before)
    if isinstance(embedder, CachedEmbedder) and embed_before is not None:
        stats.embeddings = _delta(embedder.snapshot(), embed_before)
        stats.embeddings.update({"model": embedder.model, "dim": embedder.dim})

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, ensure_ascii=False)

    db = Anatid.open(path, tenant=TENANT, embedding_dim=embedder.dim, embedder=embedder)
    log.info(
        "%s: built %s in %.1fs: %d memories, %d corrections, %d failures",
        name,
        path.name,
        stats.build_seconds,
        stats.memories_created,
        stats.corrections,
        stats.extraction_failures + stats.apply_failures,
    )
    return AnatidStore(name, mode, path, db, stats, manifest)


def _delta(after: Any, before: Any) -> Any:
    """``after - before`` over nested dicts of numbers; non-numbers are taken from ``after``."""
    if isinstance(after, dict) and isinstance(before, dict):
        return {k: _delta(v, before.get(k)) for k, v in after.items()}
    if isinstance(after, bool) or isinstance(before, bool):
        return after
    if isinstance(after, (int, float)) and isinstance(before, (int, float)):
        return after - before
    return after


def build_glm_store(
    notes: Sequence[Note],
    *,
    embedder: Embedder,
    chat: CachedChatClient,
    store_dir: pathlib.Path | str = DEFAULT_STORE_DIR,
    name: str = "anatid",
    rebuild: bool = False,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    extractor_kwargs: Mapping[str, Any] | None = None,
) -> AnatidStore:
    """S4: ingest every note with the model proposing the patches.

    The extractor is :class:`~anatid.ingest.OpenAICompatibleExtractor` on :data:`MODEL` at
    temperature 0 with a ``json_object`` reply, given ``chat`` as its client so every call is
    cached.  ``extractor_kwargs`` overrides its keyword arguments.
    """
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "client": chat,
        "temperature": 0.0,
        "response_format": "json_object",
    }
    kwargs.update(extractor_kwargs or {})

    def factory() -> OpenAICompatibleExtractor:
        return OpenAICompatibleExtractor(**kwargs)

    manifest = _manifest(
        mode="glm",
        notes=notes,
        embedder=embedder,
        id_base=ID_BASE_GLM,
        patches=None,
        extractor=factory(),
    )
    return _build_store(
        notes,
        name=name,
        mode="glm",
        path=pathlib.Path(store_dir) / f"{name}.anatid",
        extractor_factory=factory,
        embedder=embedder,
        id_base=ID_BASE_GLM,
        manifest=manifest,
        chat=chat,
        rebuild=rebuild,
        progress=progress,
    )


def build_gold_store(
    notes: Sequence[Note],
    patches: Sequence[Mapping[str, Any]],
    *,
    embedder: Embedder,
    store_dir: pathlib.Path | str = DEFAULT_STORE_DIR,
    name: str = "anatid-gold",
    rebuild: bool = False,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> AnatidStore:
    """S5, the oracle: ingest every note with its gold patch through
    :class:`~anatid.ingest.ScriptedExtractor`.  ``patches`` are the ``gold_patches.jsonl``
    rows in note order; each must belong to the note at the same position."""
    if len(patches) != len(notes):
        raise ValueError(f"{len(patches)} gold patches for {len(notes)} notes")
    for note, row in zip(notes, patches, strict=True):
        if str(row.get("note_id")) != note.note_id:
            raise ValueError(f"gold patch {row.get('note_id')!r} does not belong to {note.note_id}")

    def factory() -> ScriptedExtractor:
        return ScriptedExtractor([MemoryPatch.from_dict(row["patch"]) for row in patches])

    manifest = _manifest(
        mode="gold",
        notes=notes,
        embedder=embedder,
        id_base=ID_BASE_GOLD,
        patches=[row["patch"] for row in patches],
        extractor=None,
    )
    return _build_store(
        notes,
        name=name,
        mode="gold",
        path=pathlib.Path(store_dir) / f"{name}.anatid",
        extractor_factory=factory,
        embedder=embedder,
        id_base=ID_BASE_GOLD,
        manifest=manifest,
        chat=None,
        rebuild=rebuild,
        progress=progress,
    )


# --------------------------------------------------------------------------- the systems


@dataclass
class Retrieval:
    """One question's retrieved context and how it was made.  ``context`` is the memory text
    the answer prompt receives; everything else is the record."""

    system: str
    question: str
    budget: int
    context: str
    tokens: int
    tokenizer: str
    arms: tuple[str, ...]
    seeds: tuple[str, ...]
    hits_total: int
    hits_rendered: int
    history_rendered: int
    history_available: int
    history_tokens: int
    history_exhausted: bool
    truncated: bool
    memory_ids: list[str]
    lines: list[dict[str, Any]]
    recall_notes: tuple[str, ...]
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["arms"] = list(self.arms)
        out["seeds"] = list(self.seeds)
        out["recall_notes"] = list(self.recall_notes)
        return out


class AnatidSystem:
    """One benchmark system over a built store: S4 with every arm, one ablation, or S5.

    ``arms`` is a key of :data:`ARMS`.  ``"all"`` calls ``db.recall(question)`` exactly as an
    application would, graph arm auto-seeded from the question.  A single arm goes through
    :func:`anatid.recall.hybrid_recall` with only that arm's input: the query for BM25, the
    question's embedding for cosine, the entities the question names (found by
    :func:`anatid.recall.auto_seeds`) for the graph.  ``retrieve`` renders the hits into the
    budget with :func:`.render.render_context`: each hit followed by the memories it superseded
    (read once per memory from :meth:`anatid.Anatid.provenance` and kept), history capped at
    ``history_share`` of the budget.
    """

    def __init__(
        self,
        store: AnatidStore,
        *,
        arms: str = "all",
        name: str | None = None,
        k: int = DEFAULT_K,
        candidates: int = DEFAULT_CANDIDATES,
        hops: int = 2,
        history_share: float = DEFAULT_HISTORY_SHARE,
        max_history: int | None = None,
        legend: str | None = LEGEND,
    ) -> None:
        if arms not in ARMS:
            raise ValueError(f"arms must be one of {sorted(ARMS)}, got {arms!r}")
        self.store = store
        self.arms = arms
        self.name = name or (store.name if arms == "all" else f"{store.name}-{arms}")
        self.k = int(k)
        self.candidates = int(candidates)
        self.hops = int(hops)
        self.history_share = float(history_share)
        self.max_history = max_history
        self.legend = legend
        self._history: dict[int, list[Memory]] = {}

    def __repr__(self) -> str:
        return f"AnatidSystem({self.name!r}, arms={ARMS[self.arms]}, store={self.store.path.name})"

    @property
    def code(self) -> str:
        return SYSTEM_CODES.get(self.name, self.name)

    @property
    def db(self) -> Anatid:
        return self.store.db

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "code": self.code,
            "arms": list(ARMS[self.arms]),
            "store": self.store.path.name,
            "mode": self.store.mode,
            "k": self.k,
            "candidates": self.candidates,
            "hops": self.hops,
            "history_share": self.history_share,
            "max_history": self.max_history,
            "legend": self.legend,
            "tokenizer": tokenizer_name(),
        }

    # ------------------------------------------------------------------ recall

    def recall(self, question: str) -> RecallHits:
        """The hits for ``question`` through this system's arms, in fused rank order."""
        db = self.db
        if self.arms == "all":
            return db.recall(question, k=self.k, candidates=self.candidates, hops=self.hops)
        ns = db.resolve_tenant(None)
        common: dict[str, Any] = {
            "tenant_id": ns.tenant_id,
            "dim": db.config.embedding_dim,
            "k": self.k,
            "candidates": self.candidates,
            "hops": self.hops,
            "backend": db.csr,
        }
        if self.arms == "text":
            return hybrid_recall(
                db.connection, query=question, embedding=None, seed_entity=None, **common
            )
        if self.arms == "vector":
            if db.embedder is None:
                raise RuntimeError("the vector arm needs an embedder on the handle")
            return hybrid_recall(
                db.connection,
                query=None,
                embedding=db.embedder.embed_one(question),
                seed_entity=None,
                **common,
            )
        seeds = auto_seeds(db.connection, tenant_id=ns.tenant_id, query=question)
        if not seeds:
            return RecallHits((), notes=("graph arm: the question names no known entity",))
        return hybrid_recall(
            db.connection,
            query=None,
            embedding=None,
            seed_entity=[sid for sid, _name in seeds],
            **common,
        )

    def history(self, memory_id: int) -> list[Memory]:
        """The memories ``memory_id`` superseded, newest first, from the provenance chain."""
        cached = self._history.get(memory_id)
        if cached is None:
            chain = self.db.provenance(memory_id).chain
            cached = list(chain[1:])
            self._history[memory_id] = cached
        return cached

    # ------------------------------------------------------------------ retrieve

    def render(
        self, hits: Iterable[Any], *, budget: int, count: TokenCounter | None = None
    ) -> RenderedContext:
        return render_context(
            hits,
            budget=budget,
            history=self.history,
            count=count,
            legend=self.legend,
            history_share=self.history_share,
            max_history=self.max_history,
        )

    def retrieve(
        self, question: str, *, budget: int, count: TokenCounter | None = None
    ) -> Retrieval:
        """The memory context for ``question`` within ``budget`` tokens, with its record.

        ``count`` is the token counter; leave it out to use the one every system shares
        (:func:`.render.count_tokens`), or pass the harness's so all systems agree.
        """
        started = time.perf_counter()
        hits = self.recall(question)
        rendered = self.render(hits, budget=budget, count=count)
        return Retrieval(
            system=self.name,
            question=question,
            budget=budget,
            context=rendered.text,
            tokens=rendered.tokens,
            tokenizer=rendered.tokenizer,
            arms=tuple(hits.arms),
            seeds=tuple(hits.seeds),
            hits_total=rendered.hits_total,
            hits_rendered=rendered.hits_rendered,
            history_rendered=rendered.history_rendered,
            history_available=rendered.history_available,
            history_tokens=rendered.history_tokens,
            history_exhausted=rendered.history_exhausted,
            truncated=rendered.truncated,
            memory_ids=[str(i) for i in rendered.memory_ids],
            lines=[line.to_dict() for line in rendered.lines],
            recall_notes=tuple(hits.notes),
            seconds=time.perf_counter() - started,
        )


def make_systems(
    *,
    notes: Sequence[Note] | None = None,
    patches: Sequence[Mapping[str, Any]] | None = None,
    include: Iterable[str] = tuple(SYSTEM_CODES),
    store_dir: pathlib.Path | str = DEFAULT_STORE_DIR,
    cache_dir: pathlib.Path | str = DEFAULT_CACHE_DIR,
    api_key: str | None = None,
    embedder: Embedder | None = None,
    chat: CachedChatClient | None = None,
    rebuild: bool = False,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    k: int = DEFAULT_K,
    history_share: float = DEFAULT_HISTORY_SHARE,
    max_history: int | None = None,
) -> dict[str, AnatidSystem]:
    """Build (or reopen) the stores and return ``{name: system}`` for the names in ``include``.

    The four S4 systems share one store; S5 has its own.  ``embedder`` and ``chat`` default
    to the cached OpenRouter clients (:func:`make_embedder`, :func:`make_chat_client`); pass
    the harness's own to share one cache.  Close what this returns with :func:`close_systems`.
    """
    wanted = [name for name in include if name in SYSTEM_CODES]
    unknown = sorted(set(include) - set(SYSTEM_CODES))
    if unknown:
        raise ValueError(f"unknown anatid systems {unknown}; choose from {sorted(SYSTEM_CODES)}")
    if not wanted:
        return {}
    notes = list(notes) if notes is not None else load_notes()
    embedder = embedder or make_embedder(api_key, cache_dir=cache_dir)
    out: dict[str, AnatidSystem] = {}

    glm_names = [n for n in wanted if n != "anatid-gold"]
    if glm_names:
        chat = chat or make_chat_client(api_key, cache_dir=cache_dir)
        store = build_glm_store(
            notes,
            embedder=embedder,
            chat=chat,
            store_dir=store_dir,
            rebuild=rebuild,
            progress=progress,
        )
        for name in glm_names:
            arms = "all" if name == "anatid" else name.split("-", 1)[1]
            out[name] = AnatidSystem(
                store,
                arms=arms,
                name=name,
                k=k,
                history_share=history_share,
                max_history=max_history,
            )
    if "anatid-gold" in wanted:
        patches = list(patches) if patches is not None else load_gold_patches()
        store = build_gold_store(
            notes,
            patches,
            embedder=embedder,
            store_dir=store_dir,
            rebuild=rebuild,
            progress=progress,
        )
        out["anatid-gold"] = AnatidSystem(
            store,
            arms="all",
            name="anatid-gold",
            k=k,
            history_share=history_share,
            max_history=max_history,
        )
    return out


def close_systems(systems: Mapping[str, AnatidSystem]) -> None:
    """Close every store behind ``systems`` once."""
    seen: set[int] = set()
    for system in systems.values():
        if id(system.store) not in seen:
            seen.add(id(system.store))
            system.store.close()


# --------------------------------------------------------------------------- command line


def _emit(text: str) -> None:
    sys.stdout.write(text + "\n")


def _log_progress(index: int, total: int, record: dict[str, Any]) -> None:
    if index % 10 == 0 or index == total or record.get("failure"):
        log.info(
            "%3d/%d %s: +%d memories, %d corrections%s",
            index,
            total,
            record["note_id"],
            len(record["memories_created"]),
            len(record["corrections"]),
            f"; FAILED {record['failure']}" if record.get("failure") else "",
        )


def _cli_systems(args: argparse.Namespace, include: Iterable[str]) -> dict[str, AnatidSystem]:
    notes = load_notes()
    patches = load_gold_patches()
    store_dir = pathlib.Path(args.store_dir)
    if args.limit:
        notes = notes[: args.limit]
        patches = patches[: args.limit]
        store_dir = store_dir / f"first{args.limit}"
    return make_systems(
        notes=notes,
        patches=patches,
        include=include,
        store_dir=store_dir,
        cache_dir=pathlib.Path(args.cache_dir),
        rebuild=getattr(args, "rebuild", False),
        progress=_log_progress,
    )


def _names_for(mode: str) -> list[str]:
    if mode == "gold":
        return ["anatid-gold"]
    if mode == "glm":
        return ["anatid", "anatid-text", "anatid-vector", "anatid-graph"]
    return list(SYSTEM_CODES)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument(
        "--limit", type=int, default=0, help="use only the first N notes (a smoke test)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="ingest the corpus into the stores")
    build.add_argument("--mode", choices=("glm", "gold", "both"), default="both")
    build.add_argument("--rebuild", action="store_true", help="ignore an existing store")

    ask = sub.add_parser("ask", help="show the context one system would hand the model")
    ask.add_argument("--mode", choices=("glm", "gold"), default="gold")
    ask.add_argument("--arms", choices=sorted(ARMS), default="all")
    ask.add_argument("--budget", type=int, default=1200)
    ask.add_argument("question", nargs="+")

    stats = sub.add_parser("stats", help="print a store's build statistics")
    stats.add_argument("--mode", choices=("glm", "gold", "both"), default="both")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("anatid.recall").setLevel(logging.ERROR)

    if args.command == "build":
        systems = _cli_systems(args, _names_for(args.mode))
        try:
            for name in _names_for(args.mode):
                if name in ("anatid", "anatid-gold"):
                    _emit(json.dumps({name: systems[name].store.stats.to_dict()}, indent=2))
        finally:
            close_systems(systems)
        return 0

    if args.command == "stats":
        systems = _cli_systems(args, _names_for(args.mode))
        try:
            for name in ("anatid", "anatid-gold"):
                if name in systems:
                    _emit(json.dumps({name: systems[name].store.stats.to_dict()}, indent=2))
        finally:
            close_systems(systems)
        return 0

    name = "anatid-gold" if args.mode == "gold" else "anatid"
    systems = _cli_systems(args, [name])
    try:
        system = systems[name]
        if args.arms != "all":
            system = AnatidSystem(system.store, arms=args.arms)
        question = " ".join(args.question)
        result = system.retrieve(question, budget=args.budget)
        _emit(result.context)
        _emit("")
        summary = {
            k: v for k, v in result.to_dict().items() if k not in ("context", "lines", "question")
        }
        _emit(json.dumps(summary, indent=2))
    finally:
        close_systems(systems)
    return 0


if __name__ == "__main__":
    sys.exit(main())
