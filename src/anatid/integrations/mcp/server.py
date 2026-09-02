"""An MCP server that gives any MCP client (Claude Code, Claude Desktop, Cursor) anatid memory.

Built against **mcp 2.x**, whose ergonomic server class is
:class:`mcp.server.mcpserver.MCPServer` -- what was ``FastMCP`` in mcp 1.x.  (``mcp 2`` ships
``mcp.server.fastmcp`` as a stub whose only job is to raise ``ModuleNotFoundError`` pointing at
the rename, so there is no ambiguity about which entry point this targets.)  The low-level
:class:`mcp.server.lowlevel.Server` is the other, older entry point; it is not used here.

Configuration is environment-first, because that is what a client's JSON config block can set:

===========================  =============================================================
``ANATID_DB``                database file, or ``:memory:``.  Default ``~/.anatid/memory.anatid``
``ANATID_TENANT``            tenant id (int).  Default ``0``
``ANATID_EMBEDDING_DIM``     ``N`` in ``FLOAT[N]``, only used when creating a new file.  Default 1536
``ANATID_READ_ONLY``         ``1`` opens the whole database read-only; write tools are not registered
``ANATID_SQL_TOOL``          ``off`` removes the SQL escape hatch entirely.  Default ``on``
``ANATID_MAX_ROWS``          row cap for the SQL tool.  Default ``200``
``ANATID_MCP_TRANSPORT``     ``stdio`` (default), ``streamable-http`` or ``sse``
``ANATID_MCP_HOST`` / ``_PORT``  bind address for the HTTP transports.  Default ``127.0.0.1:8765``
===========================  =============================================================

Tenancy.  The server is **pinned** to ``ANATID_TENANT``; no tool takes a ``tenant`` argument.
``Isolation.SCOPED`` -- a ``tenant_id`` column predicate that anatid's verbs add -- is scoping,
not isolation: DuckDB has no row-level security, so anything that reaches raw SQL sees the whole
file.  Real isolation is one file per tenant (:class:`anatid.DatabasePool`), which for MCP means
one server process per tenant, each with its own ``ANATID_DB``.

Destructive tools.  ``forget`` and ``prune`` carry ``ToolAnnotations(destructiveHint=True)`` so a
client can prompt before running them.  Annotations are per *tool*, not per *argument*, so both
are marked even though only ``forget(hard=true)`` and ``prune(dry_run=false)`` actually remove
anything -- marking the tool is the conservative reading, and the descriptions say which argument
makes it bite.  Every other write tool is ``destructiveHint=False``; every read tool is
``readOnlyHint=True``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import functools
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

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

from .sqlgate import ENFORCEMENT, SqlGateway, SqlNotAllowed

__all__ = ["build_server", "main", "ServerConfig", "DEFAULT_DB_PATH"]

DEFAULT_DB_PATH = "~/.anatid/memory.anatid"

INSTRUCTIONS = """\
anatid is a bitemporal graph memory for agents, stored in one embedded DuckDB file.

Write with `remember` (facts, with the entities they are about) and `relate` (entity->entity
edges, which is what makes graph recall reach further than one hop). Read with `recall`
(hybrid: BM25 text + graph expansion, fused with RRF) and `context` (everything about one
entity). Correct a memory with `supersede`, which keeps the old version and the audit trail;
`provenance` walks that chain back to the raw source text.

Nothing is ever silently overwritten. `forget` closes a memory's validity by default and only
erases when you pass hard=true; `prune` is a dry run unless you pass dry_run=false.

Time travel is a filter over valid-time and transaction-time columns, so pass `as_of` (an
ISO-8601 timestamp) to any read to ask what the database believed then.

BM25 in DuckDB is not incremental: rows written since the last index build are invisible to
the text arm of `recall`, which reports this as bm25_stale. Call `rebuild_fts_index` to catch
up. Graph and vector arms are always current.

`sql` is a read-only escape hatch for questions the verbs do not answer.\
"""


# --------------------------------------------------------------------------- serialization


def _iso(value: _dt.datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _memory(m: Memory, *, with_embedding: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "memory_id": m.memory_id,
        "content": m.content,
        "kind": m.kind,
        "created_at": _iso(m.created_at),
        "valid_from": _iso(m.valid_from),
        "valid_to": _iso(m.valid_to),
        "tx_from": _iso(m.tx_from),
        "tx_to": _iso(m.tx_to),
        "is_current": m.is_current,
        "writer": m.writer,
        "episode_id": m.episode_id,
        "confidence": None if m.confidence is None else float(m.confidence),
        "access_count": m.access_count,
        "last_access_at": _iso(m.last_access_at),
        "tenant_id": m.tenant_id,
    }
    if with_embedding and m.embedding is not None:
        out["embedding"] = [float(x) for x in m.embedding]
    return out


def _entity(e: Entity) -> dict[str, Any]:
    return {
        "entity_id": e.entity_id,
        "name": e.name,
        "kind": e.kind,
        "valid_from": _iso(e.valid_from),
        "valid_to": _iso(e.valid_to),
        "is_current": e.is_current,
        "writer": e.writer,
        "episode_id": e.episode_id,
        "confidence": None if e.confidence is None else float(e.confidence),
        "tenant_id": e.tenant_id,
    }


def _episode(ep: Episode) -> dict[str, Any]:
    return {
        "episode_id": ep.episode_id,
        "content": ep.content,
        "source": ep.source,
        "kind": ep.kind,
        "created_at": _iso(ep.created_at),
        "writer": ep.writer,
        "tenant_id": ep.tenant_id,
    }


def _edge(e: Edge) -> dict[str, Any]:
    return {
        "edge_id": e.edge_id,
        "edge_type": e.edge_type.value,
        "src": e.src,
        "dst": e.dst,
        "rel_kind": e.rel_kind,
        "weight": None if e.weight is None else float(e.weight),
        "valid_from": _iso(e.valid_from),
        "valid_to": _iso(e.valid_to),
        "is_current": e.is_current,
        "tenant_id": e.tenant_id,
    }


def _hit(h: RecallHit) -> dict[str, Any]:
    return {
        "memory_id": h.memory_id,
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


def _receipt(r: ForgetReceipt) -> dict[str, Any]:
    return {
        "memory_id": r.memory_id,
        "tenant_id": r.tenant_id,
        "hard": r.hard,
        "at": _iso(r.at),
        "memories_deleted": r.memories_deleted,
        "about_edges_deleted": r.about_edges_deleted,
        "supersedes_edges_deleted": r.supersedes_edges_deleted,
        "episodes_deleted": r.episodes_deleted,
        "audit_rows_deleted": r.audit_rows_deleted,
        "audit_rows_written": r.audit_rows_written,
        "rows_removed": r.rows_removed,
        "reason": r.reason,
    }


def _provenance(p: Provenance) -> dict[str, Any]:
    return {
        "memory_id": p.memory_id,
        "depth": p.depth,
        "chain": [_memory(m) for m in p.chain],
        "root": None if p.root is None else _memory(p.root),
        "source_text": p.source_text,
        "episodes": [_episode(e) for e in p.episodes],
        "edges": [_edge(e) for e in p.edges],
        "writers": list(p.writers),
    }


def _prune(rep: PruneReport) -> dict[str, Any]:
    return {
        "dry_run": rep.dry_run,
        "hard": rep.hard,
        "at": _iso(rep.at),
        "count": rep.count,
        "memory_ids": list(rep.memory_ids),
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
        "indexed_max_id": s.indexed_max_id,
        "current_max_id": s.current_max_id,
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
    """Tool arguments carry entities as strings; a decimal string is an entity_id."""
    text = value.strip()
    if text and (text.isdigit() or (text[0] == "-" and text[1:].isdigit())):
        return int(text)
    return value


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


class ServerConfig:
    """Resolved server configuration -- environment, then CLI overrides."""

    def __init__(
        self,
        *,
        db: str | None = None,
        tenant: int | None = None,
        embedding_dim: int | None = None,
        read_only: bool | None = None,
        sql_tool: bool | None = None,
        max_rows: int | None = None,
        transport: str | None = None,
        host: str | None = None,
        port: int | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        e = os.environ if env is None else env
        self.db = db if db is not None else e.get("ANATID_DB", DEFAULT_DB_PATH)
        self.tenant = tenant if tenant is not None else int(e.get("ANATID_TENANT", "0"))
        self.embedding_dim = (
            embedding_dim if embedding_dim is not None
            else int(e.get("ANATID_EMBEDDING_DIM", str(anatid.DEFAULT_EMBEDDING_DIM)))
        )
        self.read_only = (
            read_only if read_only is not None
            else e.get("ANATID_READ_ONLY", "0").lower() in ("1", "true", "yes", "on")
        )
        self.sql_tool = (
            sql_tool if sql_tool is not None
            else e.get("ANATID_SQL_TOOL", "on").lower() not in ("0", "false", "no", "off")
        )
        self.max_rows = max_rows if max_rows is not None else int(e.get("ANATID_MAX_ROWS", "200"))
        self.transport = transport or e.get("ANATID_MCP_TRANSPORT", "stdio")
        self.host = host or e.get("ANATID_MCP_HOST", "127.0.0.1")
        self.port = port if port is not None else int(e.get("ANATID_MCP_PORT", "8765"))

    def resolved_db(self) -> str:
        """Expand ``~`` and create the parent directory for a file-backed database."""
        if self.db == ":memory:":
            return ":memory:"
        p = Path(self.db).expanduser()
        if not self.read_only:
            p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    def __repr__(self) -> str:                                        # pragma: no cover
        return (f"<ServerConfig db={self.db!r} tenant={self.tenant} read_only={self.read_only} "
                f"sql_tool={self.sql_tool} transport={self.transport!r}>")


# --------------------------------------------------------------------------- the server


def build_server(db: Anatid, config: ServerConfig | None = None) -> MCPServer:
    """Register anatid's verbs as MCP tools on a new :class:`~mcp.server.mcpserver.MCPServer`.

    ``db`` is an already-open :class:`anatid.Anatid`.  The server does **not** take ownership of
    it -- :func:`main` opens and closes it; a test can pass an in-memory handle and close it
    itself.  Every tool is pinned to ``db.namespace``; see the module docstring on tenancy.
    """
    cfg = config or ServerConfig()
    tenant_id = db.namespace.tenant_id
    server = MCPServer(
        "anatid",
        title="anatid graph memory",
        version=anatid.__version__,
        instructions=INSTRUCTIONS,
    )

    read_only_tool = ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                                     idempotentHint=True, openWorldHint=False)
    write_tool = ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                 idempotentHint=False, openWorldHint=False)
    destructive_tool = ToolAnnotations(readOnlyHint=False, destructiveHint=True,
                                       idempotentHint=False, openWorldHint=False)

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
                "memory's provenance -- evidence before belief. Appends never conflict."
            ),
        )
        @_guard
        def remember(
            content: str,
            entities: list[str] | None = None,
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
            src: str,
            dst: str,
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
                "Replace a memory with a corrected version. The old row is not deleted: its "
                "validity is closed, a SUPERSEDES edge records the replacement, and `provenance` "
                "and `as_of` still see it. Leave `entities` unset to inherit the old memory's "
                "ABOUT set, and `kind` unset to inherit its kind."
            ),
        )
        @_guard
        def supersede(
            old_id: int,
            content: str,
            entities: list[str] | None = None,
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
            return {"memory": _memory(m), "superseded": int(old_id)}

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
            memory_id: int, amount: int = 1, confidence: float | None = None
        ) -> dict[str, Any]:
            m = db.reinforce(int(memory_id), amount=int(amount), confidence=confidence)
            return {"memory": _memory(m)}

        @server.tool(
            title="Forget a memory",
            annotations=destructive_tool,
            description=(
                "DESTRUCTIVE. With hard=false (the default) this is a soft forget: the memory's "
                "validity is closed at now, its ABOUT edges are closed, an audit row is written, "
                "and `as_of` before now still returns it -- reversible in the sense that the "
                "history survives. With hard=true this is a right-to-erasure purge: the memory "
                "row (the embedding is a column of it), every ABOUT and SUPERSEDES edge, the "
                "episode if nothing else cites it, and the memory's own audit rows are deleted. "
                "Afterwards no row anywhere references that memory_id, in any as-of view. The "
                "returned receipt is the only record; log it outside the database if you need one."
            ),
        )
        @_guard
        def forget(
            memory_id: int, hard: bool = False, reason: str | None = None,
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
                "Rebuild the full-text index over memories.content and record the watermark. "
                "DuckDB's fts index is NOT incremental: rows written since the last build are "
                "invisible to the text arm of `recall` until this runs. Call it after a batch of "
                "`remember` calls, or whenever `recall` reports bm25_stale."
            ),
        )
        @_guard
        def rebuild_fts_index() -> dict[str, Any]:
            return {"fts": _fts(db.rebuild_fts_index())}

    # ------------------------------------------------------------------ reads

    @server.tool(
        title="Recall memories",
        annotations=read_only_tool,
        description=(
            "Hybrid retrieval over this tenant's memories: BM25 over the text, graph expansion "
            "from seed_entity (up to `hops`), and cosine over `embedding` if you supply one -- "
            "whichever arms have input, fused with Reciprocal Rank Fusion. Pass `as_of` "
            "(ISO-8601) to ask what the database believed at that time. The result reports which "
            "arms ran and whether the BM25 index is stale; a stale index means recent memories "
            "are missing from the text arm only."
        ),
    )
    @_guard
    def recall(
        query: str | None = None,
        k: int = 10,
        seed_entity: str | None = None,
        hops: int = 2,
        kinds: list[str] | None = None,
        as_of: str | None = None,
        embedding: list[float] | None = None,
        candidates: int = 50,
    ) -> dict[str, Any]:
        hits = db.recall(
            query,
            k=int(k),
            seed_entity=None if seed_entity is None else _entity_ref(seed_entity),
            hops=int(hops),
            kinds=kinds or None,
            as_of=_parse_ts(as_of, field="as_of"),
            embedding=embedding,
            candidates=int(candidates),
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
        entity: str,
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
    def get(memory_id: int, as_of: str | None = None) -> dict[str, Any]:
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
    def provenance(memory_id: int) -> dict[str, Any]:
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
        return {
            "path": db.path,
            "tenant_id": tenant_id,
            "isolation": db.namespace.isolation.value,
            "read_only": db.read_only,
            "counts": db.stats(),
            "expand_path": db.expand_path,
            "fts": _fts(db.fts_status()),
            "schema_version": info.schema_version,
            "embedding_dim": info.embedding_dim,
            "anatid_version": info.anatid_version,
            "duckdb_version": info.duckdb_version,
            "sql_tool": {
                "enabled": cfg.sql_tool,
                "max_rows": cfg.max_rows,
                "enforcement": ENFORCEMENT,
            },
        }

    # ------------------------------------------------------------------ escape hatch

    if cfg.sql_tool:
        gateway = SqlGateway(lambda: db.connection, max_rows=cfg.max_rows)

        @server.tool(
            name="sql",
            title="Read-only SQL",
            annotations=read_only_tool,
            description=(
                "READ-ONLY SQL over the anatid database, for questions the memory verbs do not "
                "answer. Only SELECT and EXPLAIN run. This is enforced by DuckDB, not by a "
                "regex: (1) duckdb's own parser classifies every statement in the text and any "
                "type other than SELECT/EXPLAIN is refused -- so INSERT, UPDATE, DELETE, ATTACH, "
                "COPY, CREATE, DROP, PRAGMA, INSTALL and a second statement after a semicolon "
                "are all rejected before anything runs; an EXPLAIN must wrap a SELECT, because "
                "EXPLAIN ANALYZE executes what it explains; (2) the statement's parse tree is "
                "scanned for filesystem and external-connector functions (read_csv, "
                "read_parquet, glob, postgres_scan, ...) and refused if any appear; (3) what is "
                "left runs inside BEGIN TRANSACTION READ ONLY on a private cursor and is always "
                "ROLLBACKed, so DuckDB's transaction manager refuses any write regardless. "
                "NOTE: raw SQL is not tenant-filtered -- add `WHERE tenant_id = <n>` yourself. "
                "Tables: memories, entities, episodes, edges_about (memory->entity), "
                "edges_relates (entity->entity), edges_supersedes (new->old), anatid_audit, "
                "anatid_meta, and the view relates_undirected. Current rows are "
                "`valid_to IS NULL AND tx_to IS NULL`."
            ),
        )
        @_guard
        def sql(query: str, limit: int = 100) -> dict[str, Any]:
            # A rejection is an anticipated failure: SqlNotAllowed propagates through _guard
            # and reaches the client as an isError result carrying the reason, so the model
            # can see why and rewrite the query. Nothing has run at that point.
            try:
                return gateway.run(query, limit=limit)
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
               "config blocks can set: ANATID_DB, ANATID_TENANT, ANATID_EMBEDDING_DIM, "
               "ANATID_READ_ONLY, ANATID_SQL_TOOL, ANATID_MAX_ROWS, ANATID_MCP_TRANSPORT, "
               "ANATID_MCP_HOST, ANATID_MCP_PORT.",
    )
    p.add_argument("--db", help=f"database file or ':memory:' (env ANATID_DB, default {DEFAULT_DB_PATH})")
    p.add_argument("--tenant", type=int, help="tenant id (env ANATID_TENANT, default 0)")
    p.add_argument("--embedding-dim", type=int, help="FLOAT[N] width for a NEW database file")
    p.add_argument("--read-only", action="store_true", default=None,
                   help="open the database read-only; no write tools are registered")
    p.add_argument("--no-sql-tool", action="store_true", help="do not register the SQL escape hatch")
    p.add_argument("--max-rows", type=int, help="row cap for the SQL tool (default 200)")
    p.add_argument("--transport", choices=("stdio", "streamable-http", "sse"),
                   help="default stdio (env ANATID_MCP_TRANSPORT)")
    p.add_argument("--host", help="bind host for the HTTP transports")
    p.add_argument("--port", type=int, help="bind port for the HTTP transports")
    p.add_argument("--version", action="version", version=f"anatid-mcp {anatid.__version__}")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point (``anatid-mcp``); also usable as ``python -m`` glue.

    Opens the configured database, registers the tools, and serves until the transport closes.
    Returns a process exit code.
    """
    args = _parse_args(argv)
    cfg = ServerConfig(
        db=args.db,
        tenant=args.tenant,
        embedding_dim=args.embedding_dim,
        read_only=args.read_only,
        sql_tool=False if args.no_sql_tool else None,
        max_rows=args.max_rows,
        transport=args.transport,
        host=args.host,
        port=args.port,
    )

    try:
        db = Anatid.open(
            cfg.resolved_db(),
            tenant=cfg.tenant,
            embedding_dim=cfg.embedding_dim,
            read_only=cfg.read_only,
        )
    except (AnatidError, OSError) as exc:
        # stdout is the JSON-RPC channel on the stdio transport; diagnostics go to stderr.
        print(f"anatid-mcp: cannot open {cfg.db!r}: {exc}", file=sys.stderr)
        return 2

    try:
        server = build_server(db, cfg)
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
