"""An MCP server that gives any MCP client (Claude Code, Claude Desktop, Cursor) anatid memory.

Built against **mcp 2.x**, whose ergonomic server class is
:class:`mcp.server.mcpserver.MCPServer` -- what was ``FastMCP`` in mcp 1.x.  (``mcp 2`` ships
``mcp.server.fastmcp`` as a stub whose only job is to raise ``ModuleNotFoundError`` pointing at
the rename, so there is no ambiguity about which entry point this targets.)  The low-level
:class:`mcp.server.lowlevel.Server` is the other, older entry point; it is not used here.

Configuration is environment-first, because that is what a client's JSON config block can set:

===============================  =========================================================
``ANATID_DB``                    database file, or ``:memory:``.  Default ``~/.anatid/memory.anatid``
``ANATID_SOCKET``                a running ``anatid-server``'s Unix socket, instead of a file
``ANATID_HTTP_URL``              a running ``anatid-server``'s HTTP listener, instead of a file
``ANATID_TOKEN`` / ``_FILE``     bearer token for ``ANATID_HTTP_URL``, or a file holding one
``ANATID_TENANT``                tenant id (int).  Default ``0``
``ANATID_EMBEDDING_DIM``         ``N`` in ``FLOAT[N]``, only when creating a new file.  Default 1536
``ANATID_EMBED_BASE_URL``        an OpenAI-compatible ``/embeddings`` endpoint; turns the vector arm on
``ANATID_EMBED_MODEL``           the embedding model at that endpoint
``ANATID_EMBED_API_KEY``         its key, optional for a local endpoint
``ANATID_EMBED_HASH``            ``1`` embeds with the offline hash stand-in instead (demos, tests)
``ANATID_EXTRACT_BASE_URL``      an OpenAI-compatible chat endpoint; registers ``ingest`` and ``apply_patch``
``ANATID_EXTRACT_MODEL``         the chat model that proposes memory patches from text
``ANATID_EXTRACT_API_KEY``       its key, optional for a local endpoint
``ANATID_EXTRACT_REASONING``     ``1`` sends ``{"reasoning": {"enabled": true}}`` with every request
``ANATID_READ_ONLY``             ``1`` opens the database read-only; write tools are not registered
``ANATID_ENABLE_SQL``            ``1`` registers the raw-SQL escape hatch.  **Default off**
``ANATID_SQL_TOOL``              legacy name; ``off`` still forces the escape hatch off
``ANATID_MAX_ROWS``              row cap for the SQL tool.  Default ``200``
``ANATID_SQL_TIMEOUT``           seconds before a SQL-tool statement is interrupted.  Default ``30``
``ANATID_SQL_MEMORY_LIMIT``      ``SET memory_limit`` for the SQL tool's database.  Default ``1GB``
``ANATID_MCP_TRANSPORT``         ``stdio`` (default), ``streamable-http`` or ``sse``
``ANATID_MCP_HOST`` / ``_PORT``  bind address for the HTTP transports.  Default ``127.0.0.1:8765``
``ANATID_MCP_AUTH``              declares that an authenticating proxy fronts an HTTP transport
===============================  =========================================================

Backends.  A file is opened in this process (the embedded profile: one MCP server per file,
because DuckDB gives one process exclusive use of a database file).  A socket or URL connects to
a running ``anatid-server`` that holds the file, so any number of MCP clients share one memory.
:func:`anatid.integrations.mcp.backend.open_backend` makes that choice; the tools do not branch
on it.  See ``docs/mcp.md``, "Sharing one memory between clients".

The SQL escape hatch is **opt-in**.  ``sql`` runs arbitrary read-only SQL, and a default-on
arbitrary-SQL primitive is exactly what a prompt-injected model wants; it is registered only when
``ANATID_ENABLE_SQL=1`` (or ``--enable-sql``, or ``ServerConfig(sql_tool=True)``) says so.
Enabling it also hardens the database -- ``enable_external_access=false``, a ``memory_limit`` and
a per-statement timeout; see :mod:`anatid.integrations.mcp.sqlgate`.

Transport security.  ``stdio`` is a pipe to the client process and needs no authentication.  The
HTTP transports are a network listener, and anatid implements no authentication of its own, so
binding one to a **non-loopback** address is refused unless ``ANATID_MCP_AUTH`` (or ``--auth``)
declares that something in front of it authenticates -- see :func:`check_transport_security` and
https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization.

Tenancy.  The server is **pinned** to ``ANATID_TENANT``; no tool takes a ``tenant`` argument.
``Isolation.SCOPED`` -- a ``tenant_id`` column predicate that anatid's verbs add -- is scoping,
not isolation: DuckDB has no row-level security, so anything that reaches raw SQL sees the whole
file.  Real isolation is one file per tenant (:class:`anatid.DatabasePool`), which for MCP means
one server process per tenant, each with its own ``ANATID_DB``.

Destructive tools.  ``forget``, ``prune``, ``unrelate`` and ``correct`` carry
``ToolAnnotations(destructiveHint=True)`` so a client can prompt before running them.
Annotations are per *tool*, not per *argument*, so all four are marked even though only
``forget(hard=true)`` and ``prune(dry_run=false)`` actually remove anything, and ``correct``
closes edges only when ``remove_relations`` names some.  Marking the tool is the conservative
reading, and the descriptions say which argument makes it bite.  ``unrelate`` and ``correct``
delete nothing: they close edge versions, which a read with ``as_of`` before the close still
traverses.  The graph stops saying something it said, and that is what the hint is for.
Every other write tool is ``destructiveHint=False``; every read tool is ``readOnlyHint=True``.

Graph maintenance.  ``relate``, ``unrelate`` and ``correct`` are the tools an agent maintains the
graph with.  Memories are filed under the entities they name and nothing links those entities
until an edge does: "Ada leads Kestrel", "Kestrel owns the ingest service" and "Bo maintains the
ingest service", remembered by name alone, leave ``recall(seed_entity="Ada")`` with the first
fact only, until ``relate(Ada, Kestrel)`` and ``relate(Kestrel, ingest service)`` bring the other
two within two hops.  ``correct`` is :meth:`anatid.Anatid.correct`: one ``supersede`` plus the
edges the correction closes and opens, in one transaction.  It is registered when the backend
has the verb, which both the embedded handle and ``AnatidClient`` do since 0.4.0.

Recall by default.  ``recall(query)`` with no ``seed_entity`` runs the text arm and the graph
arm: the query's words are matched against this tenant's entity names (longest name first, at
most three) and the graph arm expands from each match.  The result's ``seeds`` lists the names
it used and ``arms`` the arms that ran.  With ``ANATID_EMBED_*`` set the handle embeds every
``remember`` and every query, so the vector arm runs too and nothing about the tools changes.
The embedder belongs to the process that holds the file, so it is configured on an
``anatid-mcp`` that opens the file itself and refused, with the reason, on one that talks to a
server over a socket.

Ingestion.  With ``ANATID_EXTRACT_*`` set, ``ingest(text)`` hands the note and the facts the
graph already holds about the entities it names to the configured model, which proposes a
:class:`~anatid.ingest.MemoryPatch`; the tool returns the patch, its diff and a ``patch_id`` and
writes nothing.  ``apply_patch(patch_id)`` commits that patch, unchanged or edited, in one
transaction with the note stored first as the episode every row cites.  Declining is not calling
``apply_patch``.  The pipeline runs on the file's own connection, so like the SQL escape hatch
it is refused for a socket backend.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import functools
import ipaddress
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import duckdb
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

import anatid
from anatid import Anatid, AsOf
from anatid.errors import AnatidError
from anatid.types import (
    Edge,
    Entity,
    Episode,
    FtsStatus,
    ForgetReceipt,
    Memory,
    Provenance,
    PruneReport,
    RecallHit,
    RecallHits,
)

from ..wire import (
    WireEntityRef,
    WireId,
    wire_id,
    wire_ids,
    wire_unsafe_ints,
)
from anatid.ingest import Extractor, MemoryPatch
from anatid.ingest import propose as propose_patch

from .backend import (
    BackendConfigError,
    BackendError,
    check_ingest_backend,
    check_sql_backend,
    open_backend,
    start_hint,
)
from .embedding import add_cli_arguments as add_embedding_arguments
from .embedding import describe_embedder
from .ingest import (
    AppliedPatch,
    ExtractorConfigError,
    PendingPatches,
    describe_extractor,
    extractor_from_config,
)
from .sqlgate import (
    DEFAULT_MEMORY_LIMIT,
    DEFAULT_TIMEOUT_SECONDS,
    ENFORCEMENT,
    SqlGateway,
    SqlNotAllowed,
    SqlTimeout,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anatid.server.client import AnatidClient

__all__ = [
    "build_server",
    "main",
    "ServerConfig",
    "DEFAULT_DB_PATH",
    "HTTP_TRANSPORTS",
    "InsecureTransport",
    "check_transport_security",
    "is_loopback_host",
    "normalise_host",
]

DEFAULT_DB_PATH = "~/.anatid/memory.anatid"

#: Transports that open a network listener rather than talking over the client's pipes.
HTTP_TRANSPORTS = frozenset({"streamable-http", "sse", "http"})

INSTRUCTIONS = """\
anatid is a bitemporal graph memory for agents, stored in one embedded DuckDB file.

Write with `remember` (facts, with the entities they are about) and `relate` (entity->entity
edges, which is what makes graph recall reach further than one hop). Read with `recall`
(hybrid: BM25 text + graph expansion, plus cosine when the server has an embedder, fused with
RRF) and `context` (everything about one entity). A `recall` with a query and no seed_entity
finds its own seeds: entity names that occur in the query, longest first, at most three, and the
graph arm expands from each; the result's `seeds` says which, and `arms` says which arms ran.
Name a seed_entity to expand from exactly that one. Correct a memory with `supersede`, which
keeps the old version and the audit trail; `provenance` walks that chain back to the raw source
text.

Memories are filed under the entities they name, and nothing links those entities to each other
until you `relate` them. Add an edge when two entities are connected in the world (a person and
their team, a team and the service it owns) so that a `recall` seeded on either one reaches the
facts about the other; traversal is undirected and goes up to two hops. Retire an edge with
`unrelate` when the relationship itself has ended. When a correction changes who is connected
to what, use `correct` instead of `supersede`: it supersedes the memory and closes and opens the
edges you name, in one transaction, so the graph never says two things at once.

Nothing is ever silently overwritten. `forget` closes a memory's validity by default and only
erases when you pass hard=true; `prune` is a dry run unless you pass dry_run=false.

Time travel is a filter over valid-time and transaction-time columns, so pass `as_of` (an
ISO-8601 timestamp) to any read to ask what the database believed then.

Every id -- memory_id, entity_id, edge_id, episode_id -- is a decimal string such as
"883768514279557120", in what you send and in what you get back. anatid ids are 63-bit
integers and a JSON number cannot carry one exactly, so hand an id back the way you received
it, as a string, never as a number.

Text search is current the moment you write. A new memory is matched by the very next `recall`
with nothing rebuilt, and a memory you `supersede` or `forget` leaves the text results on that
same read. `rebuild_fts_index` compacts the writes made since the last build into a new BM25
generation, which buys read speed rather than visibility, so call it after a large batch of
writes and never to make a write findable. On a recall, `pending_fts_rows` counts documents the
text arm re-read from the memories themselves and searched, and `bm25_stale` is the rare case
where the text arm could not answer exactly, which takes a database with no generation ever
built and more than 100,000 memories in this tenant. Graph and vector arms are always current
too.\
"""

#: Appended to :data:`INSTRUCTIONS` only when an extractor is configured, for the same reason as
#: :data:`SQL_INSTRUCTIONS`: a model is told about a tool exactly when it is there.
INGEST_INSTRUCTIONS = """

To store a note, a message or a document rather than one fact at a time, call `ingest(text)`.
It proposes a memory patch (facts to add, facts to correct, edges to open and close, names
that mean an existing entity) resolved against what the graph already holds, and returns the
patch, its diff and a patch_id without writing anything. Show the diff to the person you are
working for. `apply_patch(patch_id)` commits it in one transaction, with the note stored as the
episode every new row cites; pass an edited `patch` to change it first. To decline, do not
call apply_patch. apply_patch is safe to repeat: a patch_id applied already is answered with
the receipt of that apply, `already_applied` true, and nothing is written twice."""

#: Appended to :data:`INSTRUCTIONS` only when the operator opted the escape hatch in, so a model
#: talking to a default server is never told about a tool that is not there.
SQL_INSTRUCTIONS = """

`sql` is a read-only escape hatch for questions the verbs do not answer. It is off by default
and this server has it on."""


# --------------------------------------------------------------------------- serialization
#
# Every id leaves as a decimal string (:mod:`anatid.integrations.wire`).  anatid ids are
# 63-bit, JSON numbers are doubles in most clients, and an id that goes out as a number comes
# back rounded with nothing raised anywhere -- so `wire_id` is not optional politeness, it is
# what makes a returned id usable as the argument to the next call.  Counts, ranks, scores and
# `version` (a small per-row counter, not an id) stay numbers.


def _iso(value: _dt.datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _memory(m: Memory, *, with_embedding: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "memory_id": wire_id(m.memory_id),
        "content": m.content,
        "kind": m.kind,
        "created_at": _iso(m.created_at),
        "valid_from": _iso(m.valid_from),
        "valid_to": _iso(m.valid_to),
        "tx_from": _iso(m.tx_from),
        "tx_to": _iso(m.tx_to),
        "is_current": m.is_current,
        "writer": m.writer,
        "episode_id": wire_id(m.episode_id),
        "confidence": None if m.confidence is None else float(m.confidence),
        "access_count": m.access_count,
        "last_access_at": _iso(m.last_access_at),
        "tenant_id": wire_id(m.tenant_id),
        "version": m.version,
    }
    if with_embedding and m.embedding is not None:
        out["embedding"] = [float(x) for x in m.embedding]
    return out


def _entity(e: Entity) -> dict[str, Any]:
    return {
        "entity_id": wire_id(e.entity_id),
        "name": e.name,
        "kind": e.kind,
        "valid_from": _iso(e.valid_from),
        "valid_to": _iso(e.valid_to),
        "is_current": e.is_current,
        "writer": e.writer,
        "episode_id": wire_id(e.episode_id),
        "confidence": None if e.confidence is None else float(e.confidence),
        "tenant_id": wire_id(e.tenant_id),
    }


def _episode(ep: Episode) -> dict[str, Any]:
    return {
        "episode_id": wire_id(ep.episode_id),
        "content": ep.content,
        "source": ep.source,
        "kind": ep.kind,
        "created_at": _iso(ep.created_at),
        "writer": ep.writer,
        "tenant_id": wire_id(ep.tenant_id),
    }


def _edge(e: Edge) -> dict[str, Any]:
    return {
        "edge_id": wire_id(e.edge_id),
        "edge_type": e.edge_type.value,
        "src": wire_id(e.src),
        "dst": wire_id(e.dst),
        "rel_kind": e.rel_kind,
        "weight": None if e.weight is None else float(e.weight),
        "valid_from": _iso(e.valid_from),
        "valid_to": _iso(e.valid_to),
        "is_current": e.is_current,
        "tenant_id": wire_id(e.tenant_id),
        "version": e.version,
    }


def _hit(h: RecallHit) -> dict[str, Any]:
    return {
        "memory_id": wire_id(h.memory_id),
        "content": h.content,
        "score": float(h.score),
        "rank": h.rank,
        "sources": list(h.sources),
        "about": list(h.about),
        "vector_rank": h.vector_rank,
        "text_rank": h.text_rank,
        "graph_rank": h.graph_rank,
        "vector_score": None if h.vector_score is None else float(h.vector_score),
        "text_score": None if h.text_score is None else float(h.text_score),
        "memory": _memory(h.memory),
    }


def _hits(hits: RecallHits) -> dict[str, Any]:
    return {
        "hits": [_hit(h) for h in hits],
        "count": len(hits),
        "arms": list(hits.arms),
        "seeds": list(hits.seeds),
        "weights": dict(hits.weights),
        "bm25_available": hits.bm25_available,
        "bm25_stale": hits.bm25_stale,
        "pending_fts_rows": hits.pending_fts_rows,
        "as_of": _asof_json(hits.as_of),
        "notes": list(hits.notes),
    }


def _asof_json(scope: AsOf | None) -> dict[str, Any] | None:
    if scope is None:
        return None
    return {
        "valid_time": _iso(scope.valid_time),
        "tx_time": _iso(scope.tx_time),
        "is_current": scope.is_current,
    }



def _applied_patch(applied: AppliedPatch, *, already_applied: bool) -> dict[str, Any]:
    """The ``apply_patch`` result: the receipt, and whether this call wrote it or replayed it."""
    receipt = applied.receipt
    out: dict[str, Any] = {
        "applied": True,
        "already_applied": already_applied,
        "patch_id": wire_id(applied.patch_id),
        "summary": receipt.describe(),
        "diff": receipt.patch.describe(),
        "changes": receipt.changes,
        **receipt.to_dict(),
    }
    if already_applied:
        out["note"] = (
            f"patch {applied.patch_id} was applied already, at {receipt.at.isoformat()}; this is "
            f"the receipt of that apply and nothing was written now. To store a changed version "
            f"of the note, call ingest again."
        )
    return out

def _receipt(r: ForgetReceipt) -> dict[str, Any]:
    return {
        "memory_id": wire_id(r.memory_id),
        "tenant_id": wire_id(r.tenant_id),
        "hard": r.hard,
        "at": _iso(r.at),
        "memories_deleted": r.memories_deleted,
        "about_edges_deleted": r.about_edges_deleted,
        "supersedes_edges_deleted": r.supersedes_edges_deleted,
        "episodes_deleted": r.episodes_deleted,
        "audit_rows_deleted": r.audit_rows_deleted,
        "audit_rows_written": r.audit_rows_written,
        "memory_versions_deleted": r.memory_versions_deleted,
        "about_edge_versions_deleted": r.about_edge_versions_deleted,
        "derived_rows_deleted": r.derived_rows_deleted,
        "invalidated_generations": r.invalidated_generations,
        "rows_removed": r.rows_removed,
        "reason": r.reason,
    }


def _provenance(p: Provenance) -> dict[str, Any]:
    return {
        "memory_id": wire_id(p.memory_id),
        "depth": p.depth,
        "chain": [_memory(m) for m in p.chain],
        "root": None if p.root is None else _memory(p.root),
        "source_text": p.source_text,
        "episodes": [_episode(e) for e in p.episodes],
        "edges": [_edge(e) for e in p.edges],
        "writers": list(p.writers),
        "versions": [_memory(m) for m in p.versions],
    }


def _prune(rep: PruneReport) -> dict[str, Any]:
    return {
        "dry_run": rep.dry_run,
        "hard": rep.hard,
        "at": _iso(rep.at),
        "count": rep.count,
        "memory_ids": wire_ids(rep.memory_ids),
        "receipts": [_receipt(r) for r in rep.receipts],
        "older_than": _iso(rep.older_than),
        "max_access_count": rep.max_access_count,
    }


def _fts(s: FtsStatus) -> dict[str, Any]:
    return {
        "available": s.available,
        "stale": s.stale,
        "indexed_rows": s.indexed_rows,
        "current_rows": s.current_rows,
        "pending_rows": s.pending_rows,
        # Row count alone cancels out (one insert + one hard purge); the id watermark does not.
        "indexed_max_id": wire_id(s.indexed_max_id),
        "current_max_id": wire_id(s.current_max_id),
        "indexed_at": _iso(s.indexed_at),
        "newest_row_at": _iso(s.newest_row_at),
        "policy": s.policy,
    }


def _parse_ts(value: str | None, *, field: str) -> _dt.datetime | None:
    """Parse an ISO-8601 timestamp from a tool argument into anatid's naive-UTC convention."""
    if value is None or value == "":
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = _dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"{field} must be an ISO-8601 timestamp such as '2026-09-02T14:30:00Z'; got {value!r}"
        ) from exc
    return anatid.to_utc_naive(parsed)


def _entity_ref(value: str) -> int | str:
    """Resolve one entity argument: a decimal string is an entity_id, anything else a name.

    The argument reaches this as a string whatever the client sent, because the parameter is
    annotated :class:`~anatid.integrations.wire.WireEntityRef`, which declares a string in the
    schema and turns an integer into its decimal form before the tool body runs.
    """
    text = value.strip()
    if text and (text.isdigit() or (text[0] == "-" and text[1:].isdigit())):
        return int(text)
    return value


@dataclass
class Relation:
    """One RELATES_TO edge as the ``correct`` tool takes it: ``{src, dst, rel_kind?}``.

    ``src`` and ``dst`` are :class:`~anatid.integrations.wire.WireEntityRef`, so each is an
    entity name or a decimal entity_id string, with an integer id still accepted.  The server
    builds this into the tool's input schema and hands the tool body instances of this class.
    """

    src: WireEntityRef
    dst: WireEntityRef
    rel_kind: str | None = None


def _relation_tuple(value: Any) -> tuple[int | str, int | str, str | None]:
    """A :class:`Relation`, or the mapping a client sent for one, resolved for the verb."""
    if isinstance(value, dict):
        return (_entity_ref(str(value["src"])), _entity_ref(str(value["dst"])),
                value.get("rel_kind"))
    return (_entity_ref(str(value.src)), _entity_ref(str(value.dst)), value.rel_kind)


def _guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Turn anatid's own failures into MCP tool errors the model can read and act on.

    Anything anatid raises deliberately -- ``NotFoundError`` for an unknown id,
    ``ConflictError`` for a lost MVCC race, ``TenantIsolationError``, a ``ValueError`` for a
    bad timestamp or a prune with no policy -- is an *anticipated* failure, so it becomes
    :class:`~mcp.server.mcpserver.exceptions.ToolError` and its text reaches the client as an
    ``isError`` result.  Anything else stays a crash: mcp reports only "Error executing tool
    <name>" and logs the traceback server-side, which is what you want for a bug.

    ``functools.wraps`` is load-bearing: ``MCPServer`` builds the tool's input schema from
    ``inspect.signature``, which follows ``__wrapped__``, so the wrapper is invisible to the
    generated schema.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except (AnatidError, SqlNotAllowed, ValueError) as exc:
            raise ToolError(str(exc) or exc.__class__.__name__) from exc

    return wrapper


# --------------------------------------------------------------------------- configuration


_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off", "")


def _flag(value: str | None, default: bool) -> bool:
    """Parse an environment flag.  Anything unrecognised is the default, not a crash."""
    if value is None:
        return default
    text = value.strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return default


class ServerConfig:
    """Resolved server configuration -- environment, then CLI overrides.

    ``sql_tool`` is **False** unless something says otherwise: ``ANATID_ENABLE_SQL=1``, the
    legacy ``ANATID_SQL_TOOL=on``, or an explicit ``sql_tool=True``.  ``ANATID_SQL_TOOL=off``
    keeps its old meaning and still forces the tool off, so a config block written against
    0.1.0 that switched the tool *off* keeps working unchanged; one that relied on the old
    default-on now has to say so.
    """

    def __init__(
        self,
        *,
        db: str | None = None,
        socket: str | None = None,
        http_url: str | None = None,
        token: str | None = None,
        token_file: str | None = None,
        tenant: int | None = None,
        embedding_dim: int | None = None,
        embed_hash: bool | None = None,
        read_only: bool | None = None,
        sql_tool: bool | None = None,
        max_rows: int | None = None,
        sql_timeout: float | None = None,
        sql_memory_limit: str | None = None,
        transport: str | None = None,
        host: str | None = None,
        port: int | None = None,
        auth: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        e = os.environ if env is None else env
        #: The mapping this configuration was read from.  The embedding and extraction
        #: settings (``ANATID_EMBED_*``, ``ANATID_EXTRACT_*``) are read from it by
        #: :func:`~anatid.integrations.mcp.backend.open_backend` and :func:`build_server`
        #: rather than copied here, so their parsing lives next to the objects they build.
        self.env: Mapping[str, str] = e
        #: ``--embed-hash``: True selects the offline hash embedder, None lets ``ANATID_EMBED_HASH``
        #: decide.  The endpoint settings have no flag, because a key belongs in the
        #: environment rather than in ``ps`` output.
        self.embed_hash = embed_hash
        self.db = db if db is not None else e.get("ANATID_DB", DEFAULT_DB_PATH)
        #: Whether a file was NAMED, by flag or by environment, as opposed to defaulted.  A
        #: socket client opens no file, so a named file next to a socket is refused rather
        #: than ignored; the default path next to a socket is simply unused.
        self.db_explicit = db is not None or "ANATID_DB" in e
        # The server profile.  An empty string means unset, so a config block that clears
        # ANATID_SOCKET="" falls back to the file the way an absent variable would.
        self.socket = (socket if socket is not None else e.get("ANATID_SOCKET", "")) or None
        self.http_url = (http_url if http_url is not None else e.get("ANATID_HTTP_URL", "")) or None
        self.token = (token if token is not None else e.get("ANATID_TOKEN", "")) or None
        self.token_file = (
            token_file if token_file is not None else e.get("ANATID_TOKEN_FILE", "")
        ) or None
        self.tenant = tenant if tenant is not None else int(e.get("ANATID_TENANT", "0"))
        self.embedding_dim = (
            embedding_dim if embedding_dim is not None
            else int(e.get("ANATID_EMBEDDING_DIM", str(anatid.DEFAULT_EMBEDDING_DIM)))
        )
        self.read_only = (
            read_only if read_only is not None
            else _flag(e.get("ANATID_READ_ONLY"), False)
        )
        if sql_tool is not None:
            self.sql_tool = bool(sql_tool)
        else:
            # OFF by default.  ANATID_ENABLE_SQL is the name that says what it does; the legacy
            # ANATID_SQL_TOOL still enables on "on" -- and an explicit ANATID_SQL_TOOL=off wins
            # over everything, because a config block that says "off" must never end up on.
            legacy = e.get("ANATID_SQL_TOOL")
            enabled = _flag(e.get("ANATID_ENABLE_SQL"), _flag(legacy, False))
            if legacy is not None and not _flag(legacy, True):
                enabled = False
            self.sql_tool = enabled
        self.max_rows = max_rows if max_rows is not None else int(e.get("ANATID_MAX_ROWS", "200"))
        self.sql_timeout = (
            float(sql_timeout) if sql_timeout is not None
            else float(e.get("ANATID_SQL_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS)))
        )
        self.sql_memory_limit = (
            sql_memory_limit if sql_memory_limit is not None
            else e.get("ANATID_SQL_MEMORY_LIMIT", DEFAULT_MEMORY_LIMIT)
        ) or None
        self.transport = transport or e.get("ANATID_MCP_TRANSPORT", "stdio")
        self.host = host or e.get("ANATID_MCP_HOST", "127.0.0.1")
        self.port = port if port is not None else int(e.get("ANATID_MCP_PORT", "8765"))
        auth_value = auth if auth is not None else e.get("ANATID_MCP_AUTH")
        self.auth = None if auth_value is None or auth_value.strip().lower() in _FALSE \
            else auth_value.strip()

    @property
    def authenticated(self) -> bool:
        """Whether the operator declared that requests to an HTTP transport are authenticated.

        anatid ships no authentication of its own, so this is a *declaration* that something in
        front of the server -- a reverse proxy, an API gateway, an OAuth resource server per the
        MCP authorization spec -- does it.  It is deliberately not a password: the point is that
        exposing an unauthenticated memory server on a public interface has to be a decision
        somebody typed, not a default.
        """
        return self.auth is not None

    @property
    def remote(self) -> bool:
        """Whether the tools talk to a running ``anatid-server`` rather than open a file."""
        return self.socket is not None or self.http_url is not None

    @property
    def target(self) -> str:
        """What this server opens or connects to, for messages: the socket, the URL or the file."""
        if self.socket is not None:
            return self.socket
        if self.http_url is not None:
            return self.http_url
        return self.db

    def resolved_db(self) -> str:
        """Expand ``~`` and create the parent directory for a file-backed database."""
        if self.db == ":memory:":
            return ":memory:"
        p = Path(self.db).expanduser()
        if not self.read_only:
            p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    def __repr__(self) -> str:                                        # pragma: no cover
        return (f"<ServerConfig db={self.db!r} socket={self.socket!r} "
                f"http_url={self.http_url!r} tenant={self.tenant} read_only={self.read_only} "
                f"sql_tool={self.sql_tool} transport={self.transport!r} host={self.host!r} "
                f"authenticated={self.authenticated}>")


# --------------------------------------------------------------------------- transport security


class InsecureTransport(RuntimeError):
    """The requested transport would expose the database on a network with no authentication."""


def normalise_host(host: str) -> str:
    """Reduce a bind host to the one spelling everything downstream reasons about.

    Surrounding whitespace goes; then the trailing dot of a fully-qualified name ("localhost.",
    which RFC 1034 calls the same name as "localhost"); then the brackets of the URL form of an
    IPv6 literal ("[::1]"); then case, because hostnames are case-insensitive (RFC 4343) and a
    hex IPv6 literal is too.  The dot is taken off before the brackets so that "[::1]." reduces
    the same way "[::1]" does.

    Case is the one that bit: :func:`is_loopback_host` used to hand the name straight to
    ``getaddrinfo``, which made "LOCALHOST" a question about the platform resolver rather than
    about the address.  glibc folds case when it reads /etc/hosts and musl does not, so the same
    configuration served on Debian and refused to start on Alpine.
    """
    text = host.strip()
    if len(text) > 1 and text.endswith("."):                 # localhost. / 127.0.0.1. / [::1].
        text = text[:-1]
    if text.startswith("[") and text.endswith("]"):          # [::1]
        text = text[1:-1]
    return text.lower()


def is_loopback_host(host: str | None) -> bool:
    """True when ``host`` can only be reached from this machine.

    The host is reduced by :func:`normalise_host` first, so "LOCALHOST", "Localhost",
    "localhost." and "localhost" are one question, and so are "::1", "[::1]" and "[::1].".

    A literal address is then decided by :mod:`ipaddress`, which knows that loopback is the
    whole of 127.0.0.0/8 and not just 127.0.0.1, and that ``::1`` is its IPv6 counterpart; an
    address with an interface scope ("::1%lo0") is decided by the address part.  A name is
    resolved and must map to loopback addresses *only* -- "localhost" normally does, a name that
    also resolves to a LAN address does not.  The name is still resolved rather than accepted on
    sight, deliberately: this function's answer decides whether an unauthenticated listener is
    allowed, and a host whose /etc/hosts points "localhost" somewhere else must not be told that
    binding it is private.

    An empty host means "every interface" in every server framework there is, so it is not
    loopback; a name that will not resolve is not loopback either, because refusing to serve is
    the safe answer to "I cannot tell".
    """
    if host is None:
        return False
    text = normalise_host(host)
    if not text or text == "*":
        return False
    try:
        return ipaddress.ip_address(text.split("%")[0]).is_loopback
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(text, None)
    except OSError:
        return False
    addresses = {info[4][0] for info in infos}
    if not addresses:
        return False
    for address in addresses:
        try:
            if not ipaddress.ip_address(str(address).split("%")[0]).is_loopback:
                return False
        except ValueError:                                   # pragma: no cover - exotic family
            return False
    return True


def check_transport_security(cfg: ServerConfig) -> None:
    """Refuse to serve an unauthenticated HTTP transport on a non-loopback interface.

    ``stdio`` is a pipe between the client and this process: there is nothing to authenticate
    and nothing to reach it from.  The HTTP transports are a listener, and this server has no
    authentication of its own -- so binding one to anything but loopback publishes every memory
    in the file, plus every write tool, to whoever can route to the port.  The MCP authorization
    spec (2025-06-18) puts authorization on the HTTP transports for exactly this reason.

    ``ANATID_MCP_AUTH`` / ``--auth`` is the operator's declaration that something in front does
    authenticate; with it, this passes.  Raises :class:`InsecureTransport` otherwise.
    """
    if cfg.transport not in HTTP_TRANSPORTS:
        return
    if is_loopback_host(cfg.host) or cfg.authenticated:
        return
    raise InsecureTransport(
        f"refusing to start: transport {cfg.transport!r} would bind {cfg.host!r}:{cfg.port}, "
        f"which is not a loopback address, and no authentication is configured. Anyone who can "
        f"reach that port would get this database's memories and its write tools. Either bind "
        f"127.0.0.1 (ANATID_MCP_HOST=127.0.0.1) and reach it through an SSH tunnel or a reverse "
        f"proxy, or -- if an authenticating proxy or OAuth resource server really is in front of "
        f"it -- say so with ANATID_MCP_AUTH=<description> / --auth <description>. anatid "
        f"implements no authentication of its own; see "
        f"https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization"
    )


# --------------------------------------------------------------------------- the server


def build_server(
    db: Anatid | AnatidClient,
    config: ServerConfig | None = None,
    *,
    extractor: Extractor | None = None,
) -> MCPServer:
    """Register anatid's verbs as MCP tools on a new :class:`~mcp.server.mcpserver.MCPServer`.

    ``db`` is an already-open :class:`anatid.Anatid`, or an
    :class:`~anatid.server.client.AnatidClient` connected to a running ``anatid-server``
    (:func:`~anatid.integrations.mcp.backend.open_backend` returns whichever the configuration
    names).  The two answer the same verbs, so no tool below asks which it has.  The server does
    **not** take ownership of ``db`` -- :func:`main` opens and closes it; a test can pass an
    in-memory handle and close it itself.  Every tool is pinned to ``db.namespace``; see the
    module docstring on tenancy.

    Raises :class:`InsecureTransport` when ``config`` asks for an HTTP transport on a
    non-loopback interface with no authentication declared -- here as well as in :func:`main`,
    so a program that builds the server itself and calls ``server.run(...)`` cannot skip the
    check.  Registering the ``sql`` tool additionally hardens ``db``'s DuckDB instance; see
    :class:`~anatid.integrations.mcp.sqlgate.SqlGateway`.

    ``extractor`` is an :class:`~anatid.ingest.Extractor`; given one, or with
    ``ANATID_EXTRACT_*`` set in ``config.env``, the ``ingest`` and ``apply_patch`` tools are
    registered.  They need the embedded handle (the pipeline runs on the file's own connection),
    so an extractor with an ``AnatidClient`` is refused here the way the SQL tool is.
    """
    cfg = config or ServerConfig()
    check_transport_security(cfg)
    check_sql_backend(db, cfg)
    if extractor is None and not db.read_only:
        extractor = extractor_from_config(env=cfg.env)
    if extractor is not None:
        check_ingest_backend(db, cfg)
    tenant_id = db.namespace.tenant_id
    server = MCPServer(
        "anatid",
        title="anatid graph memory",
        version=anatid.__version__,
        instructions=INSTRUCTIONS
        + (INGEST_INSTRUCTIONS if extractor is not None and not db.read_only else "")
        + (SQL_INSTRUCTIONS if cfg.sql_tool else ""),
    )
    #: Proposed patches waiting for ``apply_patch``; exists only with an extractor.
    pending: PendingPatches | None = (
        PendingPatches() if extractor is not None and not db.read_only else None
    )

    # Built before any tool is registered so `stats` can report the hardening that building it
    # applied.  Constructing the gateway is what disables external access on `db`, so it happens
    # only when the operator opted the escape hatch in.
    gateway = SqlGateway(
        lambda: db.connection,
        max_rows=cfg.max_rows,
        timeout=cfg.sql_timeout,
        memory_limit=cfg.sql_memory_limit,
    ) if cfg.sql_tool else None

    # Field names as the pydantic model declares them; they serialise to the protocol's
    # camelCase (readOnlyHint, ...) through the model's alias generator.
    read_only_tool = ToolAnnotations(read_only_hint=True, destructive_hint=False,
                                     idempotent_hint=True, open_world_hint=False)
    write_tool = ToolAnnotations(read_only_hint=False, destructive_hint=False,
                                 idempotent_hint=False, open_world_hint=False)
    destructive_tool = ToolAnnotations(read_only_hint=False, destructive_hint=True,
                                       idempotent_hint=False, open_world_hint=False)
    # `ingest` changes nothing in the database and sends the note to an external model.
    propose_tool = ToolAnnotations(read_only_hint=True, destructive_hint=False,
                                   idempotent_hint=False, open_world_hint=True)

    # ------------------------------------------------------------------ writes

    if not db.read_only:

        @server.tool(
            title="Remember a fact",
            annotations=write_tool,
            description=(
                "Write one memory and the ABOUT edges linking it to the entities it concerns. "
                "Entities are given by name and created on demand, so `entities=[\"Ada\", "
                "\"DuckDB\"]` is enough; a decimal string is treated as an existing entity_id. "
                "Pass `episode` to record the raw source text first and attach it as this "
                "memory's provenance -- evidence before belief. Appends never conflict. Every "
                "id in the result is a decimal string: anatid ids are 63-bit and a JSON number "
                "would round them."
            ),
        )
        @_guard
        def remember(
            content: str,
            entities: list[WireEntityRef] | None = None,
            kind: str = "fact",
            writer: str | None = None,
            episode: str | None = None,
            episode_source: str | None = None,
            confidence: float = 1.0,
            valid_from: str | None = None,
        ) -> dict[str, Any]:
            m = db.remember(
                content,
                entities=[_entity_ref(x) for x in (entities or [])],
                kind=kind,
                writer=writer,
                episode=episode,
                episode_source=episode_source,
                confidence=confidence,
                valid_from=_parse_ts(valid_from, field="valid_from"),
            )
            about = db.entities_of(m.memory_id)
            return {"memory": _memory(m), "about": [_entity(e) for e in about]}

        @server.tool(
            title="Relate two entities",
            annotations=write_tool,
            description=(
                "Create a RELATES_TO edge between two entities, given by name (created on "
                "demand) or as a decimal entity_id string. Traversal is undirected, so one edge "
                "reaches both ways. These edges are what let `recall(seed_entity=...)` and "
                "`context(hops>0)` reach beyond the memories filed directly under an entity."
            ),
        )
        @_guard
        def relate(
            src: WireEntityRef,
            dst: WireEntityRef,
            rel_kind: str | None = None,
            writer: str | None = None,
            confidence: float = 1.0,
        ) -> dict[str, Any]:
            edge = db.relate(
                _entity_ref(src), _entity_ref(dst),
                rel_kind=rel_kind, writer=writer, confidence=confidence,
            )
            return {"edge": _edge(edge)}

        @server.tool(
            title="Supersede a memory",
            annotations=write_tool,
            description=(
                "Replace a memory with a corrected version. The old row is not deleted or "
                "rewritten: its current version is closed on the transaction axis and a new "
                "version with validity ending now is added, a SUPERSEDES edge records the "
                "replacement, and `provenance` and `as_of` still see it. Leave `entities` unset "
                "to inherit the old memory's ABOUT set, and `kind` unset to inherit its kind."
            ),
        )
        @_guard
        def supersede(
            old_id: WireId,
            content: str,
            entities: list[WireEntityRef] | None = None,
            kind: str | None = None,
            writer: str | None = None,
            episode: str | None = None,
            confidence: float = 1.0,
        ) -> dict[str, Any]:
            m = db.supersede(
                int(old_id), content,
                entities=None if entities is None else [_entity_ref(x) for x in entities],
                kind=kind, writer=writer, episode=episode, confidence=confidence,
            )
            return {"memory": _memory(m), "superseded": wire_id(int(old_id))}

        @server.tool(
            title="Unrelate two entities",
            annotations=destructive_tool,
            description=(
                "DESTRUCTIVE for what the graph says; it deletes nothing. Retire the "
                "RELATES_TO edges between two entities, given by name or as decimal entity_id "
                "strings. Use it when the relationship itself has ended in the world (Bo no "
                "longer maintains the ingest service). A fact that changed while the two stay "
                "connected calls for `supersede` or `correct` instead. Every current edge "
                "between the two, in either direction, is closed, or only those with `rel_kind` "
                "when it is given. Closing writes a new edge version and deletes nothing: "
                "`recall` and `context` stop walking the edge from now on, and a read with "
                "`as_of` before now still does. An unknown entity name is an error. Returns "
                "how many edge versions closed; 0 means nothing was current between the two."
            ),
        )
        @_guard
        def unrelate(
            src: WireEntityRef,
            dst: WireEntityRef,
            rel_kind: str | None = None,
        ) -> dict[str, Any]:
            closed = db.unrelate(_entity_ref(src), _entity_ref(dst), rel_kind=rel_kind)
            return {"src": src, "dst": dst, "rel_kind": rel_kind, "edges_closed": closed}

        if hasattr(db, "correct"):

            @server.tool(
                title="Correct a memory and its edges",
                annotations=destructive_tool,
                description=(
                    "Replace a memory with a corrected version and move the RELATES_TO edges "
                    "that change with it, in one transaction: `supersede` on old_id, then close "
                    "every edge named in remove_relations, then open every edge named in "
                    "add_relations. If any step fails nothing lands, so the graph never says "
                    "two things at once. Use it instead of `supersede` when a correction "
                    "changes who is connected to what (a maintainer changes, an allergy moves "
                    "from one ingredient to another); use `supersede` when only the wording or "
                    "a detail changes. Each relation is {src, dst, rel_kind?}, entities by name "
                    "(created on demand when opening, must exist when closing) or as decimal "
                    "entity_id strings; a remove entry closes every current edge between its "
                    "two entities, or only those with its rel_kind when given. Leave `entities` "
                    "unset to inherit the old memory's ABOUT set. remove_relations is what "
                    "makes this tool destructive: it closes edges the way `unrelate` does, "
                    "deleting nothing. The result lists the edges opened, the relations that "
                    "closed at least one edge, and `edges_closed`, the number of edge versions "
                    "closed in total."
                ),
            )
            @_guard
            def correct(
                old_id: WireId,
                content: str,
                entities: list[WireEntityRef] | None = None,
                kind: str | None = None,
                writer: str | None = None,
                episode: str | None = None,
                add_relations: list[Relation] | None = None,
                remove_relations: list[Relation] | None = None,
            ) -> dict[str, Any]:
                receipt = db.correct(
                    int(old_id), content,
                    entities=None if entities is None else [_entity_ref(x) for x in entities],
                    kind=kind, writer=writer, episode=episode,
                    add_relations=[_relation_tuple(r) for r in add_relations or ()],
                    remove_relations=[_relation_tuple(r) for r in remove_relations or ()],
                )
                return {
                    "memory": _memory(receipt.new),
                    "superseded": wire_id(int(old_id)),
                    "old": _memory(receipt.old),
                    "opened": [_edge(e) for e in receipt.opened],
                    "closed": [
                        {"src": str(src), "dst": str(dst), "rel_kind": rel_kind}
                        for src, dst, rel_kind in receipt.closed
                    ],
                    "edges_closed": receipt.edges_closed,
                }

        @server.tool(
            title="Reinforce a memory",
            annotations=write_tool,
            description=(
                "Record that a memory was used: bump access_count and last_access_at, and "
                "optionally set confidence. This is a same-row UPDATE, so two concurrent "
                "reinforcements of one memory race and the loser gets a retryable conflict."
            ),
        )
        @_guard
        def reinforce(
            memory_id: WireId, amount: int = 1, confidence: float | None = None
        ) -> dict[str, Any]:
            m = db.reinforce(int(memory_id), amount=int(amount), confidence=confidence)
            return {"memory": _memory(m)}

        @server.tool(
            title="Forget a memory",
            annotations=destructive_tool,
            description=(
                "DESTRUCTIVE. With hard=false (the default) this is a soft forget: the memory's "
                "validity is closed at now by adding a new version (the previous version stays "
                "readable through `as_of` on the transaction axis), its ABOUT edges are closed "
                "the same way, an audit row is written, and `as_of` before now still returns it "
                "-- reversible in the sense that the history survives. With hard=true this is a "
                "right-to-erasure purge: every version of the memory "
                "row (the embedding is a column of it), every ABOUT and SUPERSEDES edge, the "
                "episode if nothing else cites it, and the memory's own audit rows are deleted. "
                "Afterwards no row anywhere references that memory_id, in any as-of view. The "
                "returned receipt is the only record; log it outside the database if you need one."
            ),
        )
        @_guard
        def forget(
            memory_id: WireId, hard: bool = False, reason: str | None = None,
            writer: str | None = None,
        ) -> dict[str, Any]:
            r = db.forget(int(memory_id), hard=bool(hard), reason=reason, writer=writer)
            return {"receipt": _receipt(r)}

        @server.tool(
            title="Prune memories by policy",
            annotations=destructive_tool,
            description=(
                "DESTRUCTIVE. Forget every memory matching an age and/or usage policy. At least "
                "one of older_than (ISO-8601) or max_access_count must be given -- anatid will "
                "not delete a whole tenant because an argument was forgotten. dry_run=true (the "
                "default) only reports the memory_ids that would go and changes nothing; "
                "dry_run=false actually forgets them, softly unless hard=true, which purges."
            ),
        )
        @_guard
        def prune(
            older_than: str | None = None,
            max_access_count: int | None = None,
            dry_run: bool = True,
            hard: bool = False,
            kinds: list[str] | None = None,
            limit: int | None = None,
            reason: str = "prune",
            writer: str | None = None,
        ) -> dict[str, Any]:
            rep = db.prune(
                older_than=_parse_ts(older_than, field="older_than"),
                max_access_count=max_access_count,
                dry_run=bool(dry_run), hard=bool(hard),
                kinds=kinds or None, limit=limit, reason=reason, writer=writer,
            )
            return _prune(rep)

        @server.tool(
            title="Rebuild the BM25 index",
            annotations=write_tool,
            description=(
                "Compact the writes made since the last build into a new BM25 generation over "
                "memories.content, and record the watermark. This buys read speed rather than "
                "visibility: a memory is matched by the very next `recall` without it. Call it "
                "after a large batch of `remember` calls, when text recall feels slow, or "
                "whenever `recall` reports bm25_stale."
            ),
        )
        @_guard
        def rebuild_fts_index() -> dict[str, Any]:
            return {"fts": _fts(db.rebuild_fts_index())}

        # -------------------------------------------------------------- ingestion

        if extractor is not None and pending is not None:
            assert isinstance(db, Anatid)          # check_ingest_backend ran above
            handle: Anatid = db
            model_extractor: Extractor = extractor
            pending_patches: PendingPatches = pending

            @server.tool(
                title="Propose memory changes from text",
                annotations=propose_tool,
                description=(
                    "Turn a note, a message or a document into a proposed memory patch, "
                    "WITHOUT writing anything. The text and the current facts about the "
                    "entities it names go to the configured extraction model, which proposes "
                    "facts to add, facts to correct (by memory_id), RELATES_TO edges to open and "
                    "close, and names that mean an existing entity. The proposal is then "
                    "resolved against the database: aliases are rewritten, a correction whose "
                    "target is gone becomes a new fact, and facts and edges the graph already "
                    "holds are dropped, each with a note saying so. The result carries "
                    "`patch_id`, `diff` (one line per operation, for a person to review) and "
                    "`patch` (the same as JSON, ids as decimal strings). Show the diff, then "
                    "call `apply_patch(patch_id)` to commit it, with an edited `patch` to "
                    "change it first; to decline, call nothing. A proposal lives in this "
                    "server's memory until it is applied or the process exits."
                ),
            )
            @_guard
            def ingest(
                text: str,
                source: str | None = None,
                writer: str | None = None,
            ) -> dict[str, Any]:
                if not text or not text.strip():
                    raise ToolError("ingest needs text: a note, a message or a document")
                patch = propose_patch(handle, text, extractor=model_extractor)
                entry = pending_patches.put(patch, text=text, source=source, writer=writer)
                return {
                    "patch_id": wire_id(entry.patch_id),
                    "applied": False,
                    "operations": patch.operations,
                    "is_empty": patch.is_empty,
                    "diff": patch.describe(),
                    "patch": patch.to_dict(),
                    "notes": list(patch.notes),
                    "next": (
                        "review the diff; apply_patch(patch_id) commits it, apply_patch("
                        "patch_id, patch=<edited>) commits a changed version, and not calling "
                        "apply_patch declines it"
                    ),
                }

            @server.tool(
                title="Apply a proposed memory patch",
                annotations=destructive_tool,
                description=(
                    "Commit a patch that `ingest` proposed, in one transaction: the note is "
                    "stored first as an episode, then aliases, new facts, corrections "
                    "(supersede), edges closed and edges opened, every row citing that episode. "
                    "If any step fails nothing lands and the proposal stays pending, so it can "
                    "be edited and applied again. Pass `patch` to apply an edited version of "
                    "the proposal (the `patch` object `ingest` returned, changed as needed; "
                    "memory ids stay decimal strings). DESTRUCTIVE for what the graph says "
                    "when the patch corrects a memory or removes a relation: those close the "
                    "old version, the way `supersede` and `unrelate` do, and delete nothing. "
                    "The result is the receipt: every id created or closed, as decimal strings, "
                    "and `summary` in one sentence. A patch is applied at most once, even when "
                    "two calls name it at the same moment: a patch_id applied already is "
                    "answered with the receipt of that apply, `already_applied` true and "
                    "nothing written, so repeating a call whose reply was lost is safe. An "
                    "unknown patch_id is an error; call `ingest` again to propose afresh."
                ),
            )
            @_guard
            def apply_patch(
                patch_id: WireId,
                patch: dict[str, Any] | None = None,
                writer: str | None = None,
            ) -> dict[str, Any]:
                # `claim` takes the proposal out of the table before anything is written and
                # is the only way to get it, so a second call for the same id, concurrent or a
                # retry, cannot apply it again: it waits for this call's outcome and is handed
                # the receipt, or the proposal back when this call failed.
                claimed = pending_patches.claim(int(patch_id))
                if isinstance(claimed, AppliedPatch):
                    return _applied_patch(claimed, already_applied=True)
                entry = claimed
                try:
                    to_apply = (
                        entry.patch if patch is None
                        else MemoryPatch.from_dict(patch, source_text=entry.text)
                    )
                    receipt = to_apply.apply(
                        handle,
                        writer=writer or entry.writer or "anatid-mcp",
                        episode=entry.text,
                        source=entry.source,
                    )
                except BaseException:
                    pending_patches.restore(entry)
                    raise
                return _applied_patch(pending_patches.settle(entry, receipt), already_applied=False)

    # ------------------------------------------------------------------ reads

    @server.tool(
        title="Recall memories",
        annotations=read_only_tool,
        description=(
            "Hybrid retrieval over this tenant's memories: BM25 over the text, graph expansion "
            "(up to `hops`) from seed_entity, and cosine over `embedding` if you supply one or "
            "the server has an embedder -- whichever arms have input, fused with weighted "
            "Reciprocal Rank Fusion: the vector arm leads when it runs (vector 1.0, graph 0.5, "
            "text 0.25), the text arm otherwise (text 1.0, graph 0.5), and `weights` in the "
            "result says what was used; pass `arm_weights` such as {\"text\": 0} to override a "
            "weight by name. With a query and no seed_entity the graph arm seeds itself: entity names "
            "that occur in the query, longest first, at most three, and `seeds` in the result "
            "says which; name a seed_entity to expand from exactly that one. Pass `as_of` "
            "(ISO-8601) to ask what the database believed at that time. The result reports which "
            "arms ran. A memory written a moment ago is already findable by the text arm: "
            "`pending_fts_rows` counts documents that arm re-read from the memories themselves "
            "and searched, and `bm25_stale` is the rare case where the text arm could not "
            "answer exactly, which affects the text arm alone."
        ),
    )
    @_guard
    def recall(
        query: str | None = None,
        k: int = 10,
        seed_entity: WireEntityRef | None = None,
        hops: int = 2,
        kinds: list[str] | None = None,
        as_of: str | None = None,
        embedding: list[float] | None = None,
        candidates: int = 50,
        arm_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        hits = db.recall(
            query,
            k=int(k),
            seed_entity=anatid.AUTO_SEED if seed_entity is None else _entity_ref(seed_entity),
            hops=int(hops),
            kinds=kinds or None,
            as_of=_parse_ts(as_of, field="as_of"),
            embedding=embedding,
            candidates=int(candidates),
            arm_weights=arm_weights or None,
        )
        return _hits(hits)

    @server.tool(
        title="Context for an entity",
        annotations=read_only_tool,
        description=(
            "Everything known about one entity, newest first. hops=0 (the default) returns the "
            "memories filed directly under it; hops=1 widens to its RELATES_TO neighbours; "
            "hops=2 is the full two-hop graph recall. Pass `as_of` (ISO-8601) for a "
            "point-in-time view. The entity is given by name, or as a decimal entity_id string."
        ),
    )
    @_guard
    def context(
        entity: WireEntityRef,
        limit: int = 20,
        hops: int = 0,
        kinds: list[str] | None = None,
        as_of: str | None = None,
    ) -> dict[str, Any]:
        rows = db.context(
            _entity_ref(entity), limit=int(limit), hops=int(hops),
            kinds=kinds or None, as_of=_parse_ts(as_of, field="as_of"),
        )
        return {"entity": entity, "hops": int(hops), "count": len(rows),
                "memories": [_memory(m) for m in rows]}

    @server.tool(
        title="Get one memory",
        annotations=read_only_tool,
        description=(
            "Fetch one memory by id, with the entities it is ABOUT. Pass `as_of` (ISO-8601) to "
            "read the version that was current then. Returns memory=null if the id is unknown "
            "in this tenant, or was hard-purged."
        ),
    )
    @_guard
    def get(memory_id: WireId, as_of: str | None = None) -> dict[str, Any]:
        scope = _parse_ts(as_of, field="as_of")
        m = db.get(int(memory_id), as_of=scope, with_embedding=False)
        if m is None:
            return {"memory": None, "about": []}
        about = db.entities_of(int(memory_id), as_of=scope)
        return {"memory": _memory(m), "about": [_entity(e) for e in about]}

    @server.tool(
        title="Provenance of a memory",
        annotations=read_only_tool,
        description=(
            "Walk a memory's SUPERSEDES chain back to the original assertion and the raw source "
            "text (episode) behind it. `chain` is newest-first; `root` is the first thing "
            "anybody asserted; `source_text` is the oldest episode's content."
        ),
    )
    @_guard
    def provenance(memory_id: WireId) -> dict[str, Any]:
        return _provenance(db.provenance(int(memory_id)))

    @server.tool(
        title="Database stats",
        annotations=read_only_tool,
        description=(
            "Row counts for this tenant, the graph-expansion path in use ('sql' or the C++ "
            "'extension'), the schema/embedding metadata, and the BM25 index's staleness."
        ),
    )
    @_guard
    def stats() -> dict[str, Any]:
        info = db.info()
        sql_policy: dict[str, Any] = {
            "enabled": cfg.sql_tool,
            "opt_in": True,
            "max_rows": cfg.max_rows,
            "enforcement": ENFORCEMENT,
        }
        if gateway is not None:          # built exactly when cfg.sql_tool is set
            sql_policy["limits"] = {
                "timeout_seconds": gateway.timeout,
                "memory_limit": gateway.hardening.get("memory_limit"),
                "external_access_disabled": gateway.hardening.get("external_access_disabled"),
            }
        return {
            "path": db.path,
            "tenant_id": wire_id(tenant_id),
            "isolation": db.namespace.isolation.value,
            "read_only": db.read_only,
            "counts": db.stats(),
            "expand_path": db.expand_path,
            "fts": _fts(db.fts_status()),
            "schema_version": info.schema_version,
            "embedding_dim": info.embedding_dim,
            "anatid_version": info.anatid_version,
            "duckdb_version": info.duckdb_version,
            "embedder": describe_embedder(getattr(db, "embedder", None)),
            "extractor": describe_extractor(extractor),
            "pending_patches": 0 if pending is None else len(pending),
            "sql_tool": sql_policy,
        }

    # ------------------------------------------------------------------ escape hatch

    if gateway is not None:
        memory_text = gateway.hardening.get("memory_limit") or "DuckDB's default"
        limits_text = (
            f"The database also runs with enable_external_access=false and "
            f"memory_limit={memory_text}, and a statement that runs longer than "
            f"{gateway.timeout:g}s is interrupted. "
            if gateway.timeout else
            f"The database also runs with enable_external_access=false and "
            f"memory_limit={memory_text}. "
        )

        @server.tool(
            name="sql",
            title="Read-only SQL",
            annotations=read_only_tool,
            description=(
                "READ-ONLY SQL over the anatid database, for questions the memory verbs do not "
                "answer. This tool is OFF BY DEFAULT and this server was started with it "
                "explicitly enabled (ANATID_ENABLE_SQL=1). Only SELECT and EXPLAIN run. This is "
                "enforced by DuckDB, not by a "
                "regex, and every check fails closed: (1) duckdb's own parser classifies every "
                "statement in the text and any "
                "type other than SELECT/EXPLAIN is refused -- so INSERT, UPDATE, DELETE, ATTACH, "
                "COPY, CREATE, DROP, PRAGMA, INSTALL and a second statement after a semicolon "
                "are all rejected before anything runs; an EXPLAIN must wrap a SELECT, because "
                "EXPLAIN ANALYZE executes what it explains; (2) the statement's parse tree is "
                "scanned for filesystem and external-connector functions (read_csv, "
                "read_parquet, glob, postgres_scan, ...) and for base-table names that are not "
                "plain identifiers, and a statement DuckDB cannot serialize into a parse tree is "
                "REFUSED rather than run unscanned; (3) what is "
                "left runs inside BEGIN TRANSACTION READ ONLY on a private cursor and is always "
                "ROLLBACKed, so DuckDB's transaction manager refuses any write regardless. "
                + limits_text +
                "NOTE: raw SQL is not tenant-filtered -- add `WHERE tenant_id = <n>` yourself. "
                "Tables: memories, entities, episodes, edges_about (memory->entity), "
                "edges_relates (entity->entity), edges_supersedes (new->old), anatid_audit, "
                "anatid_meta, and the view relates_undirected. Current rows are "
                "`valid_to IS NULL AND tx_to IS NULL`. A cell whose integer is too large for "
                "a JSON number to carry exactly (above 2**53, which every anatid id is) comes "
                "back as a decimal string, so a memory_id you SELECT here is the same string "
                "the memory verbs take and return; smaller integers such as counts stay "
                "numbers."
            ),
        )
        @_guard
        def sql(query: str, limit: int = 100) -> dict[str, Any]:
            # A rejection is an anticipated failure: SqlNotAllowed propagates through _guard
            # and reaches the client as an isError result carrying the reason, so the model
            # can see why and rewrite the query. Nothing has run at that point.
            try:
                # The columns of an arbitrary SELECT are not known here, so unlike the verb
                # tools this cannot name its id fields: every integer too large for a JSON
                # number becomes a decimal string instead, which is exactly the anatid ids
                # and nothing else. The tool description says so.
                return wire_unsafe_ints(gateway.run(query, limit=limit))
            except SqlTimeout as exc:
                # Ran, cost too much, was interrupted, nothing committed. Same shape as a
                # rejection so the model rewrites the query rather than retrying it verbatim.
                raise ToolError(str(exc)) from exc
            except duckdb.Error as exc:
                # A query that passed the gate but that DuckDB will not run -- an unknown
                # column, a bad cast, an ambiguous name. That is the caller's SQL to fix, so
                # DuckDB's own message goes back rather than a generic "error executing tool".
                raise ToolError(f"DuckDB rejected this query: {str(exc).strip()}") from exc

    return server


# --------------------------------------------------------------------------- entry point


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="anatid-mcp",
        description="Serve an anatid graph-memory database over the Model Context Protocol.",
        epilog="Every option also reads an environment variable, which is what MCP client "
               "config blocks can set: ANATID_DB, ANATID_SOCKET, ANATID_HTTP_URL, ANATID_TOKEN, "
               "ANATID_TOKEN_FILE, ANATID_TENANT, ANATID_EMBEDDING_DIM, ANATID_EMBED_HASH, "
               "ANATID_READ_ONLY, ANATID_ENABLE_SQL, ANATID_SQL_TOOL, ANATID_MAX_ROWS, "
               "ANATID_SQL_TIMEOUT, ANATID_SQL_MEMORY_LIMIT, ANATID_MCP_TRANSPORT, "
               "ANATID_MCP_HOST, ANATID_MCP_PORT, ANATID_MCP_AUTH. Environment only, because "
               "they carry keys: ANATID_EMBED_BASE_URL, ANATID_EMBED_MODEL, ANATID_EMBED_API_KEY "
               "(the vector arm) and ANATID_EXTRACT_BASE_URL, ANATID_EXTRACT_MODEL, "
               "ANATID_EXTRACT_API_KEY, ANATID_EXTRACT_REASONING (the ingest tools).",
    )
    p.add_argument("--db", help=f"database file or ':memory:' (env ANATID_DB, default {DEFAULT_DB_PATH})")
    p.add_argument("--socket", metavar="PATH",
                   help="talk to the anatid-server listening on this Unix socket instead of "
                        "opening a file, so several MCP clients share one memory (env "
                        "ANATID_SOCKET). Not combinable with --db or --enable-sql")
    p.add_argument("--http-url", metavar="URL",
                   help="talk to the anatid-server listening on this HTTP address, for a server "
                        "on another host (env ANATID_HTTP_URL). Needs --token or --token-file")
    p.add_argument("--token", metavar="TOKEN",
                   help="bearer token for --http-url (env ANATID_TOKEN). It is visible in the "
                        "process list; --token-file is not")
    p.add_argument("--token-file", metavar="PATH",
                   help="read the bearer token for --http-url from the first token in this "
                        "file (env ANATID_TOKEN_FILE); an anatid-server token file works as is")
    p.add_argument("--tenant", type=int, help="tenant id (env ANATID_TENANT, default 0)")
    p.add_argument("--embedding-dim", type=int, help="FLOAT[N] width for a NEW database file")
    add_embedding_arguments(p)
    p.add_argument("--read-only", action="store_true", default=None,
                   help="open the database read-only; no write tools are registered")
    p.add_argument("--enable-sql", action="store_true", default=None,
                   help="register the read-only SQL escape hatch (env ANATID_ENABLE_SQL=1). "
                        "OFF by default; enabling it also sets enable_external_access=false, a "
                        "memory_limit and a statement timeout on the database")
    p.add_argument("--no-sql-tool", action="store_true",
                   help="force the SQL escape hatch off even if the environment enables it")
    p.add_argument("--max-rows", type=int, help="row cap for the SQL tool (default 200)")
    p.add_argument("--sql-timeout", type=float,
                   help=f"seconds before a SQL-tool statement is interrupted "
                        f"(default {DEFAULT_TIMEOUT_SECONDS:g}; 0 disables)")
    p.add_argument("--sql-memory-limit",
                   help=f"SET memory_limit for the SQL tool's database (default "
                        f"{DEFAULT_MEMORY_LIMIT})")
    p.add_argument("--transport", choices=("stdio", "streamable-http", "sse"),
                   help="default stdio (env ANATID_MCP_TRANSPORT)")
    p.add_argument("--host", help="bind host for the HTTP transports")
    p.add_argument("--port", type=int, help="bind port for the HTTP transports")
    p.add_argument("--auth", metavar="DESCRIPTION",
                   help="declare that an authenticating proxy fronts an HTTP transport. Without "
                        "it, binding a non-loopback address is refused: anatid implements no "
                        "authentication of its own (env ANATID_MCP_AUTH)")
    p.add_argument("--version", action="version", version=f"anatid-mcp {anatid.__version__}")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point (``anatid-mcp``); also usable as ``python -m`` glue.

    Opens the configured database, or connects to the configured ``anatid-server``, registers
    the tools, and serves until the transport closes.  Returns a process exit code.
    """
    args = _parse_args(argv)
    if args.no_sql_tool:
        sql_tool: bool | None = False              # an explicit off beats an enabling environment
    elif args.enable_sql:
        sql_tool = True
    else:
        sql_tool = None                            # environment decides; the default is off
    cfg = ServerConfig(
        db=args.db,
        socket=args.socket,
        http_url=args.http_url,
        token=args.token,
        token_file=args.token_file,
        tenant=args.tenant,
        embedding_dim=args.embedding_dim,
        embed_hash=args.embed_hash,
        read_only=args.read_only,
        sql_tool=sql_tool,
        max_rows=args.max_rows,
        sql_timeout=args.sql_timeout,
        sql_memory_limit=args.sql_memory_limit,
        transport=args.transport,
        host=args.host,
        port=args.port,
        auth=args.auth,
    )

    # Before the database is even opened: a refusal to serve must not create a file, and the
    # operator must see why on stderr rather than as a traceback.
    try:
        check_transport_security(cfg)
    except InsecureTransport as exc:
        print(f"anatid-mcp: {exc}", file=sys.stderr)
        return 2

    # One decision: a file is opened here (embedded), a socket or URL is a client of the
    # anatid-server that holds the file.  Either way `db` answers the same verbs below.
    try:
        db = open_backend(cfg)
    except BackendError as exc:
        # Two backends named, the SQL tool asked of a socket, or a file another process holds:
        # each message says what to run instead, and nothing was opened.
        print(f"anatid-mcp: {exc}", file=sys.stderr)
        return 2
    except (AnatidError, OSError, duckdb.Error) as exc:
        # stdout is the JSON-RPC channel on the stdio transport; diagnostics go to stderr.
        # AnatidError covers ServerUnavailable, the client's "no socket at ..." with the
        # start command in it; duckdb.Error covers every other reason DuckDB refuses a file.
        print(f"anatid-mcp: cannot open {cfg.target!r}: {exc}", file=sys.stderr)
        hint = start_hint(cfg)
        if hint is not None:
            print(f"anatid-mcp: {hint}", file=sys.stderr)
        return 2

    try:
        try:
            server = build_server(db, cfg)
        except (BackendConfigError, ExtractorConfigError) as exc:
            # Half an extraction endpoint, or one asked of a socket backend: say so and stop,
            # before a tool that would fail on first use is offered to a model.
            print(f"anatid-mcp: {exc}", file=sys.stderr)
            return 2
        if cfg.transport == "stdio":
            server.run("stdio")
        elif cfg.transport == "streamable-http":
            server.run("streamable-http", host=cfg.host, port=cfg.port)
        elif cfg.transport == "sse":
            server.run("sse", host=cfg.host, port=cfg.port)
        else:                                                          # pragma: no cover
            print(f"anatid-mcp: unknown transport {cfg.transport!r}", file=sys.stderr)
            return 2
    except KeyboardInterrupt:                                          # pragma: no cover
        return 130
    finally:
        db.close()
    return 0


if __name__ == "__main__":                                             # pragma: no cover
    raise SystemExit(main())
