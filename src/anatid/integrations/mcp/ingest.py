"""Ingestion over MCP: the extractor the configuration names, and the patches waiting for review.

:mod:`anatid.ingest` turns a note into a :class:`~anatid.ingest.MemoryPatch` and applies it in
one transaction.  Over MCP that is two tool calls, because the client is the reviewer:
``ingest(text)`` proposes a patch and hands back its diff and a ``patch_id``, and
``apply_patch(patch_id)`` commits it, unchanged or edited.  Nothing is written between the two
calls, and a patch nobody applies is forgotten when the server process exits.  This module holds
the two pieces the tools in :mod:`anatid.integrations.mcp.server` need for that:

* :func:`extractor_from_config`, which builds an
  :class:`~anatid.ingest.OpenAICompatibleExtractor` from ``ANATID_EXTRACT_BASE_URL``,
  ``ANATID_EXTRACT_MODEL`` and ``ANATID_EXTRACT_API_KEY``, or returns None when none of them is
  set.  The ingest tools are registered only when there is an extractor, so a server nobody
  configured for ingestion does not offer a tool that would fail on first use.
* :class:`PendingPatches`, the bounded in-memory table of proposed patches keyed by id.  It
  hands a proposal to exactly one ``apply_patch`` call (:meth:`~PendingPatches.claim`) and keeps
  the receipt of an applied one (:meth:`~PendingPatches.settle`), so two calls that name the
  same ``patch_id``, concurrently or as a retry after a lost reply, write it once and both get
  the same receipt.

The endpoint settings are environment-only, like the embedding ones: a URL, a model and a key
are exactly what an MCP client's config block carries, and a key on a command line is visible
in ``ps``.  The variables, with the embedding ones beside them for comparison:

==============================  ==========================================================
``ANATID_EXTRACT_BASE_URL``     the OpenAI-compatible endpoint, e.g. ``https://openrouter.ai/api/v1``
``ANATID_EXTRACT_MODEL``        the chat model that proposes patches, e.g. ``z-ai/glm-5.3-flash``
``ANATID_EXTRACT_API_KEY``      its key; optional for a local endpoint
``ANATID_EXTRACT_REASONING``    ``1`` sends ``{"reasoning": {"enabled": true}}`` in ``extra_body``
==============================  ==========================================================

The pipeline runs on the file's own connection (``existing_context`` reads with SQL and
``MemoryPatch.apply`` needs one transaction spanning several verbs), so it is available to an
``anatid-mcp`` that opens the file itself and refused, at startup, to one that talks to a
server over a socket; see :func:`anatid.integrations.mcp.backend.check_backend_config`.
"""

from __future__ import annotations

import datetime as _dt
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping

from anatid.errors import NotFoundError
from anatid.ids import new_id
from anatid.ingest import Extractor, MemoryPatch, OpenAICompatibleExtractor, PatchReceipt
from anatid.types import utcnow

__all__ = [
    "ENV_EXTRACT_API_KEY",
    "ENV_EXTRACT_BASE_URL",
    "ENV_EXTRACT_MODEL",
    "ENV_EXTRACT_REASONING",
    "AppliedPatch",
    "ExtractorConfigError",
    "PendingPatch",
    "PendingPatches",
    "describe_extractor",
    "extraction_configured",
    "extractor_from_config",
]

ENV_EXTRACT_BASE_URL = "ANATID_EXTRACT_BASE_URL"
ENV_EXTRACT_MODEL = "ANATID_EXTRACT_MODEL"
ENV_EXTRACT_API_KEY = "ANATID_EXTRACT_API_KEY"
ENV_EXTRACT_REASONING = "ANATID_EXTRACT_REASONING"

#: How many proposed patches one server keeps, and separately how many receipts of applied ones.
#: The oldest is dropped when a table is full; a client that proposes and never applies does not
#: grow the process without bound, and neither does one that applies forever.
DEFAULT_PENDING_LIMIT = 64

_TRUE = ("1", "true", "yes", "on")


class ExtractorConfigError(ValueError):
    """Half an extraction endpoint was configured, or the package it needs is missing."""


def extraction_configured(env: Mapping[str, str]) -> bool:
    """Whether ``env`` names an extraction endpoint at all (fully or by half)."""
    return bool(
        (env.get(ENV_EXTRACT_BASE_URL) or "").strip() or (env.get(ENV_EXTRACT_MODEL) or "").strip()
    )


def extractor_from_config(*, env: Mapping[str, str] | None = None) -> Extractor | None:
    """The extractor the environment asks for, or None when it asks for none.

    Raises :class:`ExtractorConfigError` when only one of the URL and the model is set, so a
    typo in a config block is reported at startup rather than as a failing tool call, and when
    the ``openai`` package the extractor needs is not installed.
    """
    if env is None:
        import os

        env = os.environ
    base_url = (env.get(ENV_EXTRACT_BASE_URL) or "").strip()
    model = (env.get(ENV_EXTRACT_MODEL) or "").strip()
    api_key = (env.get(ENV_EXTRACT_API_KEY) or "").strip() or None
    if not base_url and not model:
        return None
    if not base_url or not model:
        missing = ENV_EXTRACT_MODEL if not model else ENV_EXTRACT_BASE_URL
        raise ExtractorConfigError(
            f"an extraction endpoint needs both {ENV_EXTRACT_BASE_URL} and {ENV_EXTRACT_MODEL}; "
            f"{missing} is not set"
        )
    reasoning = (env.get(ENV_EXTRACT_REASONING) or "").strip().lower() in _TRUE
    try:
        return OpenAICompatibleExtractor(
            model=model,
            base_url=base_url,
            api_key=api_key,
            extra_body={"reasoning": {"enabled": True}} if reasoning else None,
        )
    except ImportError as exc:
        raise ExtractorConfigError(
            f"{ENV_EXTRACT_MODEL} is set but the extractor needs the openai package: "
            f"pip install openai"
        ) from exc


def describe_extractor(extractor: Extractor | None) -> dict[str, Any] | None:
    """What the ``stats`` tool can say about the server's extractor.  Never the key."""
    if extractor is None:
        return None
    if isinstance(extractor, OpenAICompatibleExtractor):
        base_url = getattr(getattr(extractor, "client", None), "base_url", None)
        return {
            "kind": "openai_compatible",
            "model": extractor.model,
            "url": None if base_url is None else str(base_url),
        }
    return {"kind": type(extractor).__name__, "model": getattr(extractor, "model", None)}


@dataclass(frozen=True)
class PendingPatch:
    """One proposed patch, waiting for ``apply_patch``.

    ``text`` is the note it was proposed from and becomes the episode when the patch is applied;
    ``source`` and ``writer`` are what the proposing call named, and ``apply_patch`` may override
    the writer.
    """

    patch_id: int
    patch: MemoryPatch
    text: str
    source: str | None
    writer: str | None
    proposed_at: _dt.datetime


@dataclass(frozen=True)
class AppliedPatch:
    """A proposal ``apply_patch`` committed, kept so a repeated call is answered with its receipt.

    ``receipt.patch`` is the version that landed, edited or not, and ``receipt.at`` is when.
    """

    proposal: PendingPatch
    receipt: PatchReceipt

    @property
    def patch_id(self) -> int:
        return self.proposal.patch_id


class PendingPatches:
    """Proposed patches keyed by id, newest kept, thread safe.

    The id is an anatid id (63-bit, minted by :func:`anatid.ids.new_id`), so it crosses the
    wire as a decimal string like every other id the tools hand out.

    A proposal is applied at most once.  :meth:`claim` takes it out of the table before anything
    is written, so two calls that name the same id cannot both apply it: the second waits for
    the first to finish and is handed its :class:`AppliedPatch`, or, when the first failed and
    :meth:`restore` put the proposal back, claims the proposal itself.  :meth:`settle` records
    the receipt, and the newest ``limit`` receipts are kept, so a client that retries an
    ``apply_patch`` whose reply it lost gets the receipt again rather than a second copy of
    every memory.  ``len`` and ``in`` count and find the proposals still waiting.
    """

    def __init__(self, *, limit: int = DEFAULT_PENDING_LIMIT) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self.limit = int(limit)
        self._items: OrderedDict[int, PendingPatch] = OrderedDict()
        self._applied: OrderedDict[int, AppliedPatch] = OrderedDict()
        self._in_flight: set[int] = set()
        self._lock = threading.Lock()
        #: Signalled whenever an in-flight proposal is settled or restored.
        self._outcome = threading.Condition(self._lock)

    def put(
        self,
        patch: MemoryPatch,
        *,
        text: str,
        source: str | None = None,
        writer: str | None = None,
    ) -> PendingPatch:
        """Remember a proposal and return it with its id.  Drops the oldest when full."""
        entry = PendingPatch(
            patch_id=new_id(),
            patch=patch,
            text=text,
            source=source,
            writer=writer,
            proposed_at=utcnow(),
        )
        with self._lock:
            self._items[entry.patch_id] = entry
            self._trim(self._items)
        return entry

    def get(self, patch_id: int) -> PendingPatch:
        """The pending patch with this id, or :class:`~anatid.errors.NotFoundError`."""
        with self._lock:
            entry = self._items.get(int(patch_id))
        if entry is None:
            raise NotFoundError(self._missing(patch_id))
        return entry

    def claim(self, patch_id: int) -> PendingPatch | AppliedPatch:
        """Take a proposal out of the table to apply it, or the record of its apply.

        Exactly one caller is handed the :class:`PendingPatch`: while that caller holds it the
        proposal is in neither table, and the caller must follow with :meth:`settle` or
        :meth:`restore`.  A call for an id another thread holds blocks until that thread's
        outcome is known, then gets the :class:`AppliedPatch` or, after a failure, the proposal.
        An id that is neither pending, in flight nor applied raises
        :class:`~anatid.errors.NotFoundError`.
        """
        key = int(patch_id)
        with self._outcome:
            while key in self._in_flight:
                self._outcome.wait()
            applied = self._applied.get(key)
            if applied is not None:
                return applied
            entry = self._items.pop(key, None)
            if entry is None:
                raise NotFoundError(self._missing(key))
            self._in_flight.add(key)
            return entry

    def restore(self, entry: PendingPatch) -> None:
        """Put a claimed proposal back, after applying it failed, so it can be edited and retried."""
        with self._outcome:
            self._in_flight.discard(entry.patch_id)
            self._items[entry.patch_id] = entry
            self._trim(self._items)
            self._outcome.notify_all()

    def settle(self, entry: PendingPatch, receipt: PatchReceipt) -> AppliedPatch:
        """Record that a claimed proposal was applied, and return the record."""
        applied = AppliedPatch(proposal=entry, receipt=receipt)
        with self._outcome:
            self._in_flight.discard(entry.patch_id)
            self._applied[entry.patch_id] = applied
            self._trim(self._applied)
            self._outcome.notify_all()
        return applied

    def discard(self, patch_id: int) -> bool:
        """Forget a proposal.  Returns whether one was there."""
        with self._lock:
            return self._items.pop(int(patch_id), None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __contains__(self, patch_id: object) -> bool:
        with self._lock:
            return isinstance(patch_id, int) and patch_id in self._items

    def _trim(self, table: OrderedDict[int, Any]) -> None:
        """Drop the oldest entries until ``table`` fits.  Called with the lock held."""
        while len(table) > self.limit:
            table.popitem(last=False)

    def _missing(self, patch_id: int) -> str:
        return (
            f"no pending patch {patch_id}: it was never proposed by this server, or it was "
            f"proposed or applied more than {self.limit} proposals ago and has been forgotten. "
            f"Call ingest again to propose a fresh one."
        )
