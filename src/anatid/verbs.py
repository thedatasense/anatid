"""The memory verbs.

Each **write** verb (``remember``, ``supersede``, ``forget``, ``relate``, ``reinforce``,
``episode``, ``entity_id``) is one transaction.  ``prune`` is not -- it is a query plus one
transaction per memory, and says so.  **Read** verbs (``recall``, ``recall_2hop``, ``context``,
``get``, ``provenance``, ``stats``) open no transaction: ``recall`` runs its staleness probe,
its arms, its hydration and its ABOUT lookup as separate statements, so another thread's commit
can land between them.  Wrap the call in ``with db.transaction():`` when you need one snapshot.

Every value a verb writes or filters on travels as a bound parameter -- no statement is ever
built by concatenating caller data.  (Two narrow exceptions, both documented where they occur:
integer id lists in an ``IN`` clause and the C++ extension's bind-time arguments, where a bound
parameter defeats the index or the function.  Both go through ``int()`` first, so only digits
can reach the SQL.)  DDL is the other half of that promise: identifiers go through
:func:`anatid.schema.quote_ident` and column *types* through :func:`anatid.schema.check_type`.

Every verb takes an explicit ``now=`` (and reads take ``as_of=``) defaulting to UTC now, so a run
can be made bit-for-bit deterministic by passing timestamps in.

The verbs live on a mixin so they are methods of :class:`anatid.Anatid`; the module-level
functions of the same name are thin wrappers for callers who prefer ``verbs.remember(db, ...)``.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Sequence

from . import recall as _recall
from .errors import ConflictError, EmbeddingDimensionError, NotFoundError
from .ids import new_id
from .schema import (
    ENTITY_COLUMNS,
    EPISODE_COLUMNS,
    MEMORY_COLUMNS,
    temporal_predicate,
)
from .types import (
    RELATES_TO,
    SUPERSEDES,
    AsOf,
    Edge,
    Entity,
    Episode,
    ForgetReceipt,
    Memory,
    Namespace,
    Provenance,
    PruneReport,
    RecallHits,
    utcnow,
    to_utc_naive,
)

log = logging.getLogger("anatid.verbs")

__all__ = [
    "MemoryVerbs",
    "AsOfView",
    "remember",
    "recall",
    "recall_2hop",
    "context",
    "supersede",
    "reinforce",
    "forget",
    "prune",
    "as_of",
    "provenance",
]

_MEM_SELECT = ", ".join(MEMORY_COLUMNS)
_MEM_SELECT_NO_EMB = ", ".join(
    ("NULL AS embedding" if c == "embedding" else c) for c in MEMORY_COLUMNS)
_ENT_SELECT = ", ".join(ENTITY_COLUMNS)
_EPI_SELECT = ", ".join(EPISODE_COLUMNS)

_INSERT_MEMORY_TPL = (
    "INSERT INTO memories (memory_id, tenant_id, content, kind, embedding, created_at, "
    "valid_from, valid_to, tx_from, tx_to, writer, episode_id, confidence, access_count, "
    "last_access_at) VALUES (?, ?, ?, ?, ?::FLOAT[{dim}], ?, ?, NULL, ?, NULL, ?, ?, ?, 0, NULL)"
)

_INSERT_ABOUT = (
    "INSERT INTO edges_about (edge_id, src, dst, tenant_id, weight, valid_from, valid_to, "
    "tx_from, tx_to, writer, episode_id, confidence) "
    "SELECT eid, ?, dst, ?, ?, ?, NULL, ?, NULL, ?, ?, ? "
    "FROM (SELECT unnest(?::BIGINT[]) AS eid, unnest(?::BIGINT[]) AS dst)"
)

_MAX_CHAIN = 10_000          # cycle guard for the SUPERSEDES walk


def _count(result) -> int:
    """Row count reported by a DuckDB DML statement."""
    row = result.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


class AsOfView:
    """Read-only view of a database at one point in time (``db.as_of(t)``).

    Every read on this object carries the same :class:`~anatid.types.AsOf` scope.  There is no
    engine support behind it: DuckDB has no ``AS OF SYSTEM TIME``, so this simply passes the
    scope to the verbs, which compile it into a WHERE clause over the bitemporal columns.

    ::

        before = db.as_of(t0)
        before.recall_2hop(seed)      # the answer as of t0
        before.get(mid).content       # the content believed at t0
    """

    __slots__ = ("_db", "scope")

    def __init__(self, db: "MemoryVerbs", scope: AsOf) -> None:
        self._db = db
        self.scope = scope

    def __repr__(self) -> str:
        return f"<AsOfView valid_time={self.scope.valid_time} tx_time={self.scope.tx_time}>"

    def recall(self, query=None, **kw) -> RecallHits:
        kw.setdefault("as_of", self.scope)
        return self._db.recall(query, **kw)

    def recall_2hop(self, seed_entity, **kw) -> list[Memory]:
        kw.setdefault("as_of", self.scope)
        return self._db.recall_2hop(seed_entity, **kw)

    def recall_2hop_ids(self, seed_entity, **kw):
        kw.setdefault("as_of", self.scope)
        return self._db.recall_2hop_ids(seed_entity, **kw)

    def context(self, entity, **kw) -> list[Memory]:
        kw.setdefault("as_of", self.scope)
        return self._db.context(entity, **kw)

    def get(self, memory_id, **kw) -> Memory | None:
        kw.setdefault("as_of", self.scope)
        return self._db.get(memory_id, **kw)

    def entities_of(self, memory_id, **kw) -> list[Entity]:
        kw.setdefault("as_of", self.scope)
        return self._db.entities_of(memory_id, **kw)

    def provenance(self, memory_id, **kw) -> Provenance:
        return self._db.provenance(memory_id, **kw)


class MemoryVerbs:
    """The verb surface of :class:`anatid.Anatid`.

    The mixin expects its host to provide ``execute()``, ``transaction()``, ``connection``,
    ``resolve_tenant()``, ``config`` and ``csr`` -- see :class:`anatid.database.Anatid`.
    """

    # -- provided by Anatid -------------------------------------------------
    config: Any
    csr: Any
    connection: Any

    def execute(self, sql: str, params: Sequence[Any] | None = None, *, con=None): ...  # noqa: E704
    def transaction(self, con=None): ...  # noqa: E704
    def resolve_tenant(self, tenant=None) -> Namespace: ...  # noqa: E704

    # ------------------------------------------------------------------ internals

    def _check_embedding(self, embedding: Sequence[float] | None) -> str | None:
        if embedding is None:
            return None
        dim = self.config.embedding_dim
        if len(embedding) != dim:
            raise EmbeddingDimensionError(
                f"embedding has {len(embedding)} dimensions, database is FLOAT[{dim}]",
                expected=dim, got=len(embedding))
        return _recall.embedding_literal(embedding)

    def entity_id(
        self,
        value: "int | str | Entity",
        *,
        tenant: int | Namespace | None = None,
        create: bool = False,
        kind: str | None = None,
        now: _dt.datetime | None = None,
        writer: str | None = None,
        episode_id: int | None = None,
    ) -> int:
        """Resolve an entity reference to an ``entity_id``.

        ``int`` is taken as an id verbatim (no existence check -- that is a scan anatid will not
        do on every write).  ``Entity`` uses its id.  ``str`` is looked up by
        ``(tenant_id, name)`` among currently-valid entities and, with ``create=True``, inserted
        if missing.
        """
        ns = self.resolve_tenant(tenant)
        if isinstance(value, Entity):
            return int(value.entity_id)
        if isinstance(value, bool):
            raise TypeError("entity reference must be int | str | Entity, not bool")
        if isinstance(value, int):
            return int(value)
        if not isinstance(value, str):
            raise TypeError(f"entity reference must be int | str | Entity, got {type(value).__name__}")
        row = self.execute(
            "SELECT entity_id FROM entities WHERE tenant_id = ? AND name = ? "
            "AND valid_to IS NULL AND tx_to IS NULL ORDER BY entity_id LIMIT 1",
            [ns.tenant_id, value]).fetchone()
        if row is not None:
            return int(row[0])
        if not create:
            raise NotFoundError(f"no entity named {value!r} in tenant {ns.tenant_id}")
        at = to_utc_naive(now) or utcnow()
        eid = new_id()
        self.execute(
            "INSERT INTO entities (entity_id, tenant_id, kind, name, valid_from, valid_to, "
            "tx_from, tx_to, writer, episode_id, confidence) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?, ?)",
            [eid, ns.tenant_id, kind, value, at, at, writer, episode_id, 1.0])
        return eid

    # ------------------------------------------------------------------ episodes

    def episode(
        self,
        content: str,
        *,
        source: str | None = None,
        kind: str | None = None,
        writer: str | None = None,
        tenant: int | Namespace | None = None,
        now: _dt.datetime | None = None,
        episode_id: int | None = None,
    ) -> Episode:
        """Record raw source material before anything is derived from it.

        "Evidence before belief": write the episode, then pass its ``episode_id`` to
        :meth:`remember` so every derived fact points back at the text it came from and
        :meth:`provenance` can walk there.
        """
        ns = self.resolve_tenant(tenant)
        at = to_utc_naive(now) or utcnow()
        eid = int(episode_id) if episode_id is not None else new_id()
        with self.transaction():
            self.execute(
                "INSERT INTO episodes (episode_id, tenant_id, source, content, kind, created_at, "
                "valid_from, valid_to, tx_from, tx_to, writer) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?)",
                [eid, ns.tenant_id, source, content, kind, at, at, at, writer])
        return Episode(episode_id=eid, tenant_id=ns.tenant_id, content=content, source=source,
                       kind=kind, created_at=at, valid_from=at, tx_from=at, writer=writer)

    def get_episode(self, episode_id: int, *, tenant: int | Namespace | None = None) -> Episode | None:
        """Fetch one episode, or None."""
        ns = self.resolve_tenant(tenant)
        row = self.execute(
            f"SELECT {_EPI_SELECT} FROM episodes WHERE episode_id = ? AND tenant_id = ?",
            [int(episode_id), ns.tenant_id]).fetchone()
        return None if row is None else Episode.from_row(row)

    # ------------------------------------------------------------------ entities & edges

    def upsert_entity(
        self,
        name: str,
        *,
        kind: str | None = None,
        tenant: int | Namespace | None = None,
        writer: str | None = None,
        episode_id: int | None = None,
        now: _dt.datetime | None = None,
    ) -> Entity:
        """Find or create an entity by name and return it."""
        ns = self.resolve_tenant(tenant)
        with self.transaction():
            eid = self.entity_id(name, tenant=ns, create=True, kind=kind, now=now,
                                 writer=writer, episode_id=episode_id)
        got = self.get_entity(eid, tenant=ns)
        if got is None:      # pragma: no cover - only if another writer purged it in between
            raise NotFoundError(f"entity {eid} vanished during upsert")
        return got

    def get_entity(
        self,
        entity: "int | str | Entity",
        *,
        tenant: int | Namespace | None = None,
    ) -> Entity | None:
        """Fetch one entity by id or name, or None."""
        ns = self.resolve_tenant(tenant)
        if isinstance(entity, str):
            row = self.execute(
                f"SELECT {_ENT_SELECT} FROM entities WHERE tenant_id = ? AND name = ? "
                "AND valid_to IS NULL AND tx_to IS NULL ORDER BY entity_id LIMIT 1",
                [ns.tenant_id, entity]).fetchone()
        else:
            eid = entity.entity_id if isinstance(entity, Entity) else int(entity)
            row = self.execute(
                f"SELECT {_ENT_SELECT} FROM entities WHERE entity_id = ? AND tenant_id = ?",
                [int(eid), ns.tenant_id]).fetchone()
        return None if row is None else Entity.from_row(row)

    def relate(
        self,
        src: "int | str | Entity",
        dst: "int | str | Entity",
        *,
        rel_kind: str | None = None,
        tenant: int | Namespace | None = None,
        writer: str | None = None,
        episode_id: int | None = None,
        confidence: float = 1.0,
        now: _dt.datetime | None = None,
        valid_from: _dt.datetime | None = None,
        create_missing: bool = True,
        edge_id: int | None = None,
    ) -> Edge:
        """Add a ``RELATES_TO`` edge between two entities -- the edges 2-hop recall traverses.

        Traversal is undirected (both directions are expanded), so ``relate(a, b)`` makes ``b``
        reachable from ``a`` and vice versa.  Marks the optional CSR snapshot stale.
        """
        ns = self.resolve_tenant(tenant)
        at = to_utc_naive(now) or utcnow()
        vf = to_utc_naive(valid_from) or at
        with self.transaction():
            s = self.entity_id(src, tenant=ns, create=create_missing, now=at, writer=writer)
            d = self.entity_id(dst, tenant=ns, create=create_missing, now=at, writer=writer)
            eid = int(edge_id) if edge_id is not None else new_id()
            self.execute(
                "INSERT INTO edges_relates (edge_id, src, dst, tenant_id, rel_kind, valid_from, "
                "valid_to, tx_from, tx_to, writer, episode_id, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?, ?)",
                [eid, s, d, ns.tenant_id, rel_kind, vf, at, writer, episode_id, confidence])
        self.csr.note_edge_write()
        return Edge(edge_id=eid, edge_type=RELATES_TO, src=s, dst=d, tenant_id=ns.tenant_id,
                    rel_kind=rel_kind, valid_from=vf, tx_from=at, writer=writer,
                    episode_id=episode_id, confidence=confidence)

    def entities_of(
        self,
        memory_id: int,
        *,
        tenant: int | Namespace | None = None,
        as_of: AsOf | _dt.datetime | None = None,
    ) -> list[Entity]:
        """Entities a memory is ABOUT, ordered by entity_id."""
        ns = self.resolve_tenant(tenant)
        scope = AsOf.coerce(as_of)
        aw, ap = temporal_predicate("a", scope)
        rows = self.execute(
            f"SELECT {', '.join('e.' + c for c in ENTITY_COLUMNS)} FROM edges_about a "
            f"JOIN entities e ON e.entity_id = a.dst AND e.tenant_id = a.tenant_id "
            f"WHERE a.src = ? AND a.tenant_id = ? AND {aw} ORDER BY e.entity_id",
            [int(memory_id), ns.tenant_id] + ap).fetchall()
        return [Entity.from_row(r) for r in rows]

    # ------------------------------------------------------------------ remember

    def remember(
        self,
        content: str,
        *,
        entities: Sequence["int | str | Entity"] = (),
        kind: str = "fact",
        embedding: Sequence[float] | None = None,
        writer: str | None = None,
        episode_id: int | None = None,
        episode: str | None = None,
        episode_source: str | None = None,
        confidence: float = 1.0,
        now: _dt.datetime | None = None,
        valid_from: _dt.datetime | None = None,
        created_at: _dt.datetime | None = None,
        tenant: int | Namespace | None = None,
        memory_id: int | None = None,
        entity_kind: str | None = None,
        weight: float = 1.0,
        create_entities: bool = True,
    ) -> Memory:
        """Write one memory and its ABOUT edges, in a single transaction.

        ``entities``
            Names (created on demand), ``entity_id`` ints, or :class:`~anatid.types.Entity`
            objects.  Each becomes one ABOUT edge, which is what makes the memory reachable by
            graph recall.
        ``episode`` / ``episode_source``
            Raw source text to record first and attach as this memory's provenance.  Mutually
            exclusive with an explicit ``episode_id``.
        ``now``
            Defaults to UTC now and sets ``valid_from``, ``tx_from`` and ``created_at`` unless
            those are given.  Pass it to make a run deterministic.

        Appends never conflict under DuckDB's optimistic MVCC, so concurrent ``remember()`` from
        many writers all commit.
        """
        ns = self.resolve_tenant(tenant)
        at = to_utc_naive(now) or utcnow()
        vf = to_utc_naive(valid_from) or at
        ca = to_utc_naive(created_at) or at
        emb = self._check_embedding(embedding)
        mid = int(memory_id) if memory_id is not None else new_id()
        if episode is not None and episode_id is not None:
            raise ValueError("pass either episode (raw text to record) or episode_id, not both")

        with self.transaction():
            if episode is not None:
                ep_id = new_id()
                self.execute(
                    "INSERT INTO episodes (episode_id, tenant_id, source, content, kind, "
                    "created_at, valid_from, valid_to, tx_from, tx_to, writer) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?)",
                    [ep_id, ns.tenant_id, episode_source, episode, "source", ca, vf, at, writer])
                episode_id = ep_id

            self.execute(
                _INSERT_MEMORY_TPL.format(dim=self.config.embedding_dim),
                [mid, ns.tenant_id, content, kind, emb, ca, vf, at, writer,
                 None if episode_id is None else int(episode_id), confidence])

            dsts = [self.entity_id(e, tenant=ns, create=create_entities, kind=entity_kind,
                                   now=at, writer=writer, episode_id=episode_id)
                    for e in entities]
            if dsts:
                self.execute(
                    _INSERT_ABOUT,
                    [mid, ns.tenant_id, weight, vf, at, writer,
                     None if episode_id is None else int(episode_id), confidence,
                     [new_id() for _ in dsts], dsts])

        return Memory(
            memory_id=mid, tenant_id=ns.tenant_id, content=content, kind=kind,
            embedding=None if embedding is None else tuple(float(x) for x in embedding),
            created_at=ca, valid_from=vf, valid_to=None, tx_from=at, tx_to=None,
            writer=writer, episode_id=None if episode_id is None else int(episode_id),
            confidence=confidence, access_count=0, last_access_at=None)

    # ------------------------------------------------------------------ reads

    def get(
        self,
        memory_id: int,
        *,
        tenant: int | Namespace | None = None,
        as_of: AsOf | _dt.datetime | None = None,
        with_embedding: bool = True,
    ) -> Memory | None:
        """Fetch one memory by id under the given time scope, or None."""
        ns = self.resolve_tenant(tenant)
        scope = AsOf.coerce(as_of)
        cols = _MEM_SELECT if with_embedding else _MEM_SELECT_NO_EMB
        if scope.is_current:
            # No validity filter: get() by id should return a superseded row too, and say so via
            # Memory.is_current. The as-of forms below are the ones that hide history.
            row = self.execute(
                f"SELECT {cols} FROM memories WHERE memory_id = ? AND tenant_id = ?",
                [int(memory_id), ns.tenant_id]).fetchone()
        else:
            w, wp = temporal_predicate("m", scope)
            row = self.execute(
                f"SELECT {cols} FROM memories m WHERE m.memory_id = ? AND m.tenant_id = ? AND {w}",
                [int(memory_id), ns.tenant_id] + wp).fetchone()
        return None if row is None else Memory.from_row(row)

    def recall_2hop_ids(
        self,
        seed_entity: "int | str | Entity",
        *,
        tenant: int | Namespace | None = None,
        limit: int = 20,
        as_of: AsOf | _dt.datetime | None = None,
        hops: int = 2,
        kinds: Sequence[str] | None = None,
    ) -> list[tuple[int, _dt.datetime]]:
        """The benchmarked 2-hop recall exactly as measured: ``[(memory_id, created_at)]``.

        Two columns only, because hydrating full rows would drag the ``FLOAT[N]`` embedding into
        the TOP_N.  Use :meth:`recall_2hop` for :class:`~anatid.types.Memory` objects.
        """
        ns = self.resolve_tenant(tenant)
        scope = AsOf.coerce(as_of)
        seed = self.entity_id(seed_entity, tenant=ns, create=False) \
            if not isinstance(seed_entity, int) else int(seed_entity)
        return _recall.recall_2hop_ids(
            self.connection, tenant_id=ns.tenant_id, seed_entity_id=seed, limit=limit,
            hops=hops, as_of=scope, backend=self.csr, kinds=kinds)

    def recall_2hop(
        self,
        seed_entity: "int | str | Entity",
        *,
        tenant: int | Namespace | None = None,
        limit: int = 20,
        as_of: AsOf | _dt.datetime | None = None,
        hops: int = 2,
        kinds: Sequence[str] | None = None,
        with_embedding: bool = False,
    ) -> list[Memory]:
        """Memories ABOUT any entity within ``hops`` of the seed, newest first.

        The graph query the whole engine choice was decided on: at 1M memories / 2.3M edges /
        10 tenants it runs in 2.88 ms p50 on plain DuckDB SQL and 2.04 ms with the optional C++
        CSR extension, against LadybugDB's tuned 7.35 ms.

        Traversal is undirected over currently-valid same-tenant ``RELATES_TO`` edges; results are
        ordered ``created_at DESC, memory_id DESC``.
        """
        pairs = self.recall_2hop_ids(seed_entity, tenant=tenant, limit=limit, as_of=as_of,
                                     hops=hops, kinds=kinds)
        ns = self.resolve_tenant(tenant)
        rows = _recall.hydrate(self.connection, [m for m, _ in pairs],
                               tenant_id=ns.tenant_id, with_embedding=with_embedding)
        return [rows[m] for m, _ in pairs if m in rows]

    def context(
        self,
        entity: "int | str | Entity",
        *,
        limit: int = 20,
        as_of: AsOf | _dt.datetime | None = None,
        tenant: int | Namespace | None = None,
        hops: int = 0,
        kinds: Sequence[str] | None = None,
        with_embedding: bool = False,
    ) -> list[Memory]:
        """Memories directly ABOUT one entity, newest first.

        ``hops=0`` (the default) is the entity's own memories.  ``hops=1`` widens to its
        neighbours, ``hops=2`` is :meth:`recall_2hop`.
        """
        return self.recall_2hop(entity, tenant=tenant, limit=limit, as_of=as_of, hops=hops,
                                kinds=kinds, with_embedding=with_embedding)

    def recall(
        self,
        query: str | None = None,
        *,
        tenant: int | Namespace | None = None,
        k: int = 10,
        embedding: Sequence[float] | None = None,
        seed_entity: "int | str | Entity | None" = None,
        hops: int = 2,
        as_of: AsOf | _dt.datetime | None = None,
        kinds: Sequence[str] | None = None,
        candidates: int = _recall.DEFAULT_CANDIDATES,
        rrf_k: int = _recall.RRF_K,
        with_embedding: bool = False,
        include_about: bool = True,
        on_stale_fts: str = "report",
    ) -> RecallHits:
        """Hybrid retrieval: cosine + BM25 + graph expansion, fused with RRF (k=60).

        Arms run when their input exists -- ``embedding`` for the vector arm, ``query`` plus an
        fts index for BM25, ``seed_entity`` for the graph arm -- and are fused by Reciprocal Rank
        Fusion.  The result is a ``list[RecallHit]`` that also reports how it was answered:

        * ``hits.bm25_stale`` -- rows written since the last :meth:`~anatid.Anatid.rebuild_fts_index`
          are invisible to the text arm (DuckDB's fts index is not incremental).  A warning is
          also logged on ``anatid.recall``.  ``on_stale_fts="error"`` raises instead.
        * ``hits.pending_fts_rows`` -- how many rows that is.
        * ``hits.arms`` -- which arms actually ran.

        The vector arm is a brute-force scan of the tenant's embeddings: comfortable to roughly
        1e5 memories per tenant, linear beyond that.
        """
        ns = self.resolve_tenant(tenant)
        scope = AsOf.coerce(as_of)
        seed = None
        if seed_entity is not None:
            seed = int(seed_entity) if isinstance(seed_entity, int) and not isinstance(seed_entity, bool) \
                else self.entity_id(seed_entity, tenant=ns, create=False)
        if embedding is not None:
            self._check_embedding(embedding)
        return _recall.hybrid_recall(
            self.connection, tenant_id=ns.tenant_id, query=query, embedding=embedding,
            dim=self.config.embedding_dim, k=k, seed_entity=seed, hops=hops, as_of=scope,
            kinds=kinds, candidates=candidates, rrf_k=rrf_k, backend=self.csr,
            with_embedding=with_embedding, include_about=include_about,
            on_stale_fts=on_stale_fts)

    def as_of(self, timestamp: _dt.datetime | AsOf, *, tx_time: _dt.datetime | None = None) -> AsOfView:
        """Scope reads to a point in time.

        ``db.as_of(t).recall_2hop(seed)`` answers as the database believed at ``t``.

        This is anatid's own filter over ``valid_from``/``valid_to`` and ``tx_from``/``tx_to``.
        **DuckDB has no ``AS OF SYSTEM TIME``**; nothing rewinds, and rows removed by a hard
        purge are absent from every as-of view too.  Pass ``tx_time`` to separate the two axes
        (what was true then vs. what the database knew then).
        """
        if isinstance(timestamp, AsOf):
            scope = timestamp
        else:
            ts = to_utc_naive(timestamp)
            scope = AsOf(valid_time=ts, tx_time=to_utc_naive(tx_time) if tx_time else ts)
        return AsOfView(self, scope)

    # ------------------------------------------------------------------ mutations

    def supersede(
        self,
        old_id: int,
        content: str,
        *,
        entities: Sequence["int | str | Entity"] | None = None,
        kind: str | None = None,
        embedding: Sequence[float] | None = None,
        writer: str | None = None,
        episode_id: int | None = None,
        episode: str | None = None,
        episode_source: str | None = None,
        confidence: float = 1.0,
        now: _dt.datetime | None = None,
        tenant: int | Namespace | None = None,
        memory_id: int | None = None,
        close_about_edges: bool = False,
        allow_fork: bool = False,
    ) -> Memory:
        """Replace a memory with a newer one, in one transaction.

        Inserts the new memory, closes the old one's ``valid_to`` at ``now``, and records a
        ``SUPERSEDES`` edge (new -> old) so :meth:`provenance` can walk the chain.  Reads at the
        current time see only the new memory; ``db.as_of(t)`` for ``t`` before ``now`` still sees
        the old one.

        ``entities=None`` (default) inherits the old memory's ABOUT entities; pass a sequence to
        replace them, or ``()`` for none.  ``kind=None`` inherits the old kind.

        The ``UPDATE`` on the old row is what can lose a write-write race: if another transaction
        is *concurrently* superseding the same memory, this one raises
        :class:`~anatid.errors.ConflictError` and is safe to retry.

        The *serialized* version of that race -- a retry after a client timeout, or two agents
        acting on the same stale read -- does not conflict in the engine at all: the ``UPDATE``
        simply matches nothing, because ``valid_to`` is already set.  anatid checks the row count
        and raises :class:`~anatid.errors.ConflictError` naming the memory that already superseded
        it, rather than committing a second current memory and leaving two heads on one chain.
        Pass ``allow_fork=True`` if branching really is what you want.
        """
        ns = self.resolve_tenant(tenant)
        at = to_utc_naive(now) or utcnow()
        old = self.get(int(old_id), tenant=ns, with_embedding=False)
        if old is None:
            raise NotFoundError(f"memory {old_id} not found in tenant {ns.tenant_id}")
        if entities is None:
            entities = [e.entity_id for e in self.entities_of(int(old_id), tenant=ns)]
        emb = self._check_embedding(embedding)
        new_mid = int(memory_id) if memory_id is not None else new_id()
        if episode is not None and episode_id is not None:
            raise ValueError("pass either episode (raw text to record) or episode_id, not both")

        with self.transaction():
            if episode is not None:
                ep_id = new_id()
                self.execute(
                    "INSERT INTO episodes (episode_id, tenant_id, source, content, kind, "
                    "created_at, valid_from, valid_to, tx_from, tx_to, writer) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?)",
                    [ep_id, ns.tenant_id, episode_source, episode, "source", at, at, at, writer])
                episode_id = ep_id

            self.execute(
                _INSERT_MEMORY_TPL.format(dim=self.config.embedding_dim),
                [new_mid, ns.tenant_id, content, kind if kind is not None else old.kind, emb,
                 at, at, at, writer, None if episode_id is None else int(episode_id), confidence])

            dsts = [self.entity_id(e, tenant=ns, create=True, now=at, writer=writer,
                                   episode_id=episode_id) for e in entities]
            if dsts:
                self.execute(
                    _INSERT_ABOUT,
                    [new_mid, ns.tenant_id, 1.0, at, at, writer,
                     None if episode_id is None else int(episode_id), confidence,
                     [new_id() for _ in dsts], dsts])

            closed = _count(self.execute(
                "UPDATE memories SET valid_to = ? WHERE memory_id = ? AND tenant_id = ? "
                "AND valid_to IS NULL",
                [at, int(old_id), ns.tenant_id]))
            if closed == 0 and not allow_fork:
                head = self.execute(
                    "SELECT src FROM edges_supersedes WHERE dst = ? AND tenant_id = ? "
                    "ORDER BY tx_from, edge_id LIMIT 1",
                    [int(old_id), ns.tenant_id]).fetchone()
                by = f" (memory {int(head[0])} already superseded it)" if head else ""
                raise ConflictError(
                    f"memory {int(old_id)} is not current in tenant {ns.tenant_id}, so this "
                    f"supersede would fork the chain into two heads{by}. Re-read the current "
                    f"memory and supersede that one, or pass allow_fork=True.")
            if close_about_edges:
                self.execute(
                    "UPDATE edges_about SET valid_to = ? WHERE src = ? AND tenant_id = ? "
                    "AND valid_to IS NULL", [at, int(old_id), ns.tenant_id])
            self.execute(
                "INSERT INTO edges_supersedes (edge_id, src, dst, tenant_id, tx_from, writer) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [new_id(), new_mid, int(old_id), ns.tenant_id, at, writer])
            # The counterpart id goes in a COLUMN, never into `reason`: forget(hard=True) has to
            # be able to find and delete every audit row that names an erased memory, and it
            # cannot search free text for it.
            self.execute(
                "INSERT INTO anatid_audit (audit_id, tenant_id, memory_id, related_memory_id, "
                "action, reason, writer, happened_at) "
                "VALUES (?, ?, ?, ?, 'supersede', ?, ?, ?)",
                [new_id(), ns.tenant_id, int(old_id), new_mid,
                 'superseded', writer, at])

        return Memory(
            memory_id=new_mid, tenant_id=ns.tenant_id, content=content,
            kind=kind if kind is not None else old.kind,
            embedding=None if embedding is None else tuple(float(x) for x in embedding),
            created_at=at, valid_from=at, valid_to=None, tx_from=at, tx_to=None, writer=writer,
            episode_id=None if episode_id is None else int(episode_id), confidence=confidence)

    def reinforce(
        self,
        memory_id: int,
        *,
        amount: int = 1,
        confidence: float | None = None,
        now: _dt.datetime | None = None,
        tenant: int | Namespace | None = None,
    ) -> Memory:
        """Record that a memory was used: bump ``access_count`` and ``last_access_at``.

        Optionally raise (or lower) ``confidence``.  This is a same-row ``UPDATE``, so two
        concurrent reinforcements of the same memory race and the loser gets
        :class:`~anatid.errors.ConflictError` -- retry it.
        """
        ns = self.resolve_tenant(tenant)
        at = to_utc_naive(now) or utcnow()
        with self.transaction():
            if confidence is None:
                n = _count(self.execute(
                    "UPDATE memories SET access_count = coalesce(access_count, 0) + ?, "
                    "last_access_at = ? WHERE memory_id = ? AND tenant_id = ?",
                    [int(amount), at, int(memory_id), ns.tenant_id]))
            else:
                n = _count(self.execute(
                    "UPDATE memories SET access_count = coalesce(access_count, 0) + ?, "
                    "last_access_at = ?, confidence = ? WHERE memory_id = ? AND tenant_id = ?",
                    [int(amount), at, float(confidence), int(memory_id), ns.tenant_id]))
            if n == 0:
                raise NotFoundError(f"memory {memory_id} not found in tenant {ns.tenant_id}")
        got = self.get(int(memory_id), tenant=ns, with_embedding=False)
        if got is None:  # pragma: no cover - concurrent purge
            raise NotFoundError(f"memory {memory_id} vanished during reinforce")
        return got

    def forget(
        self,
        memory_id: int,
        *,
        hard: bool = False,
        reason: str | None = None,
        writer: str | None = None,
        now: _dt.datetime | None = None,
        tenant: int | Namespace | None = None,
    ) -> ForgetReceipt:
        """Stop believing a memory (soft), or erase it (hard).

        **Soft** (default): closes the memory's ``valid_to`` and its ABOUT edges' ``valid_to`` at
        ``now`` and writes an ``anatid_audit`` row.  History is intact -- ``db.as_of(t)`` before
        ``now`` still returns it, and :meth:`provenance` still walks through it.

        **Hard** (``hard=True``): a right-to-erasure purge.  Deletes the ``memories`` row (the
        embedding is a column of it), every ABOUT edge, every SUPERSEDES edge in either
        direction, the episode if no other row cites it, and every ``anatid_audit`` row that
        names the memory -- as its own ``memory_id`` *or* as the ``related_memory_id`` of some
        other memory's supersede, which is why that id is a column and not free text.  When it
        returns, **no row in the memory graph references that memory_id** -- including in every
        as-of view, because there is nothing left to find.  That is the point: an audit trail
        that retained the id would defeat the erasure.

        The purge covers the tables anatid owns.  It reaches anything else in the file only
        through the hooks you register with :meth:`~anatid.Anatid.register_erasure_hook`, which
        run inside this transaction; ``AnatidSession`` registers one so a purge also removes the
        transcript rows that quoted the memory's id and content back into the model's context.
        Rows those hooks deleted are reported as ``extra_rows_deleted``.  The returned
        :class:`~anatid.types.ForgetReceipt` is the record; log it outside the database if you
        need one.
        """
        ns = self.resolve_tenant(tenant)
        at = to_utc_naive(now) or utcnow()
        mid = int(memory_id)

        if not hard:
            with self.transaction():
                n = _count(self.execute(
                    "UPDATE memories SET valid_to = ? WHERE memory_id = ? AND tenant_id = ? "
                    "AND valid_to IS NULL", [at, mid, ns.tenant_id]))
                if n == 0 and self.get(mid, tenant=ns, with_embedding=False) is None:
                    raise NotFoundError(f"memory {mid} not found in tenant {ns.tenant_id}")
                self.execute(
                    "UPDATE edges_about SET valid_to = ? WHERE src = ? AND tenant_id = ? "
                    "AND valid_to IS NULL", [at, mid, ns.tenant_id])
                self.execute(
                    "INSERT INTO anatid_audit (audit_id, tenant_id, memory_id, action, reason, "
                    "writer, happened_at) VALUES (?, ?, ?, 'forget_soft', ?, ?, ?)",
                    [new_id(), ns.tenant_id, mid, reason, writer, at])
            return ForgetReceipt(memory_id=mid, tenant_id=ns.tenant_id, hard=False, at=at,
                                 memories_deleted=0, audit_rows_written=1, reason=reason)

        with self.transaction():
            row = self.execute(
                "SELECT episode_id, content FROM memories WHERE memory_id = ? AND tenant_id = ?",
                [mid, ns.tenant_id]).fetchone()
            if row is None:
                raise NotFoundError(f"memory {mid} not found in tenant {ns.tenant_id}")
            ep = None if row[0] is None else int(row[0])
            # Read the content BEFORE the delete: erasure hooks need it to find verbatim copies
            # of the text (a tool-result row quotes the content, not just the id).
            purged_content = row[1]

            about = _count(self.execute(
                "DELETE FROM edges_about WHERE src = ? AND tenant_id = ?", [mid, ns.tenant_id]))
            sup = _count(self.execute(
                "DELETE FROM edges_supersedes WHERE (src = ? OR dst = ?) AND tenant_id = ?",
                [mid, mid, ns.tenant_id]))
            audit = _count(self.execute(
                "DELETE FROM anatid_audit WHERE (memory_id = ? OR related_memory_id = ?) "
                "AND tenant_id = ?", [mid, mid, ns.tenant_id]))
            mems = _count(self.execute(
                "DELETE FROM memories WHERE memory_id = ? AND tenant_id = ?", [mid, ns.tenant_id]))

            episodes = 0
            if ep is not None:
                # Only orphaned evidence goes; an episode several memories cite stays.
                still = self.execute(
                    "SELECT (SELECT count(*) FROM memories WHERE episode_id = ? AND tenant_id = ?)"
                    " + (SELECT count(*) FROM entities WHERE episode_id = ? AND tenant_id = ?)"
                    " + (SELECT count(*) FROM edges_about WHERE episode_id = ? AND tenant_id = ?)"
                    " + (SELECT count(*) FROM edges_relates WHERE episode_id = ? AND tenant_id = ?)",
                    [ep, ns.tenant_id] * 4).fetchone()[0]
                if int(still) == 0:
                    episodes = _count(self.execute(
                        "DELETE FROM episodes WHERE episode_id = ? AND tenant_id = ?",
                        [ep, ns.tenant_id]))

            # Tables anatid does not own (conversation transcripts, caller-defined labels).
            # Inside the transaction on purpose: a hook that raises aborts the whole purge
            # rather than committing a half-erased file.
            extra = 0
            for hook in getattr(self, "erasure_hooks", ()):
                extra += int(hook(self, mid, ns.tenant_id, purged_content) or 0)

        return ForgetReceipt(memory_id=mid, tenant_id=ns.tenant_id, hard=True, at=at,
                             memories_deleted=mems, about_edges_deleted=about,
                             supersedes_edges_deleted=sup, episodes_deleted=episodes,
                             audit_rows_deleted=audit, extra_rows_deleted=extra, reason=reason)

    def prune(
        self,
        *,
        older_than: _dt.datetime | None = None,
        max_access_count: int | None = None,
        dry_run: bool = True,
        hard: bool = False,
        kinds: Sequence[str] | None = None,
        limit: int | None = None,
        tenant: int | Namespace | None = None,
        writer: str | None = None,
        reason: str | None = "prune",
        now: _dt.datetime | None = None,
    ) -> PruneReport:
        """Forget memories matching an age and/or usage policy.

        ``dry_run=True`` (the default) only reports what would go -- read
        :attr:`~anatid.types.PruneReport.memory_ids` before running it for real.  At least one of
        ``older_than`` / ``max_access_count`` must be given: anatid will not delete a whole
        tenant because an argument was forgotten.

        ``hard=True`` purges instead of closing validity; see :meth:`forget`.

        **Not atomic.**  Unlike every write verb, ``prune`` is one ``SELECT`` plus one
        transaction per memory it forgets.  A failure part-way (a ``ConflictError`` on one row, a
        crash) leaves the earlier ones committed and does not return a report at all -- run a
        ``dry_run`` first and keep its ``memory_ids`` if you need to resume.
        """
        ns = self.resolve_tenant(tenant)
        at = to_utc_naive(now) or utcnow()
        if older_than is None and max_access_count is None:
            raise ValueError("prune needs older_than and/or max_access_count")

        where = ["tenant_id = ?", "valid_to IS NULL", "tx_to IS NULL"]
        params: list[Any] = [ns.tenant_id]
        if older_than is not None:
            where.append("created_at < ?")
            params.append(to_utc_naive(older_than))
        if max_access_count is not None:
            where.append("coalesce(access_count, 0) <= ?")
            params.append(int(max_access_count))
        if kinds:
            where.append(f"kind IN ({', '.join('?' for _ in kinds)})")
            params += [str(k) for k in kinds]
        sql = (f"SELECT memory_id FROM memories WHERE {' AND '.join(where)} "
               f"ORDER BY created_at ASC, memory_id ASC")
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        ids = [int(r[0]) for r in self.execute(sql, params).fetchall()]

        if dry_run:
            return PruneReport(dry_run=True, hard=hard, at=at, memory_ids=tuple(ids),
                               older_than=to_utc_naive(older_than),
                               max_access_count=max_access_count)
        receipts = [self.forget(m, hard=hard, reason=reason, writer=writer, now=at, tenant=ns)
                    for m in ids]
        return PruneReport(dry_run=False, hard=hard, at=at, memory_ids=tuple(ids),
                           receipts=tuple(receipts), older_than=to_utc_naive(older_than),
                           max_access_count=max_access_count)

    # ------------------------------------------------------------------ provenance

    def provenance(
        self,
        memory_id: int,
        *,
        tenant: int | Namespace | None = None,
        max_depth: int = _MAX_CHAIN,
    ) -> Provenance:
        """Walk a memory's SUPERSEDES chain back to the original assertion and its evidence.

        Returns the chain newest-first, the episodes behind it, the SUPERSEDES edges traversed
        and every distinct writer involved -- the answer to "where did this belief come from and
        who put it there".

        A hard purge anywhere in the chain truncates it: the erased link is genuinely gone.
        """
        ns = self.resolve_tenant(tenant)
        chain: list[Memory] = []
        edges: list[Edge] = []
        seen: set[int] = set()
        cur: int | None = int(memory_id)
        while cur is not None and cur not in seen and len(chain) < max_depth:
            seen.add(cur)
            mem = self.get(cur, tenant=ns, with_embedding=False)
            if mem is None:
                break
            chain.append(mem)
            row = self.execute(
                "SELECT edge_id, dst, tx_from, writer FROM edges_supersedes "
                "WHERE src = ? AND tenant_id = ? ORDER BY tx_from, edge_id LIMIT 1",
                [cur, ns.tenant_id]).fetchone()
            if row is None:
                break
            edges.append(Edge(edge_id=int(row[0]), edge_type=SUPERSEDES, src=cur,
                              dst=int(row[1]), tenant_id=ns.tenant_id, tx_from=row[2],
                              writer=row[3]))
            cur = int(row[1])
        if not chain:
            raise NotFoundError(f"memory {memory_id} not found in tenant {ns.tenant_id}")

        ep_ids = [m.episode_id for m in chain if m.episode_id is not None]
        episodes: list[Episode] = []
        if ep_ids:
            marks = ", ".join("?" for _ in ep_ids)
            rows = {int(r[0]): Episode.from_row(r) for r in self.execute(
                f"SELECT {_EPI_SELECT} FROM episodes WHERE episode_id IN ({marks}) "
                f"AND tenant_id = ?", [*ep_ids, ns.tenant_id]).fetchall()}
            episodes = [rows[e] for e in ep_ids if e in rows]

        writers: list[str] = []
        for m in chain:
            if m.writer and m.writer not in writers:
                writers.append(m.writer)
        return Provenance(memory_id=int(memory_id), chain=tuple(chain),
                          episodes=tuple(episodes), edges=tuple(edges), writers=tuple(writers))

    # ------------------------------------------------------------------ misc

    def stats(self, *, tenant: int | Namespace | None = None, all_tenants: bool = False) -> dict:
        """Row counts for this tenant (or the whole file), plus the active expansion path."""
        ns = self.resolve_tenant(tenant)
        out: dict[str, Any] = {}
        tables = ["memories", "entities", "episodes", "edges_about", "edges_relates",
                  "edges_supersedes", "anatid_audit"]
        for t in tables:
            if all_tenants:
                out[t] = int(self.execute(f"SELECT count(*) FROM {t}").fetchone()[0])
            else:
                out[t] = int(self.execute(
                    f"SELECT count(*) FROM {t} WHERE tenant_id = ?", [ns.tenant_id]).fetchone()[0])
        out["current_memories"] = int(self.execute(
            "SELECT count(*) FROM memories WHERE tenant_id = ? AND valid_to IS NULL "
            "AND tx_to IS NULL", [ns.tenant_id]).fetchone()[0])
        out["tenant_id"] = ns.tenant_id
        out["expand_path"] = self.csr.active
        return out


# --------------------------------------------------------------------------- function forms

def remember(db: MemoryVerbs, content: str, **kw) -> Memory:
    """Function form of :meth:`MemoryVerbs.remember`."""
    return db.remember(content, **kw)


def recall(db: MemoryVerbs, query: str | None = None, **kw) -> RecallHits:
    """Function form of :meth:`MemoryVerbs.recall`."""
    return db.recall(query, **kw)


def recall_2hop(db: MemoryVerbs, seed_entity, **kw) -> list[Memory]:
    """Function form of :meth:`MemoryVerbs.recall_2hop`."""
    return db.recall_2hop(seed_entity, **kw)


def context(db: MemoryVerbs, entity, **kw) -> list[Memory]:
    """Function form of :meth:`MemoryVerbs.context`."""
    return db.context(entity, **kw)


def supersede(db: MemoryVerbs, old_id: int, content: str, **kw) -> Memory:
    """Function form of :meth:`MemoryVerbs.supersede`."""
    return db.supersede(old_id, content, **kw)


def reinforce(db: MemoryVerbs, memory_id: int, **kw) -> Memory:
    """Function form of :meth:`MemoryVerbs.reinforce`."""
    return db.reinforce(memory_id, **kw)


def forget(db: MemoryVerbs, memory_id: int, **kw) -> ForgetReceipt:
    """Function form of :meth:`MemoryVerbs.forget`."""
    return db.forget(memory_id, **kw)


def prune(db: MemoryVerbs, **kw) -> PruneReport:
    """Function form of :meth:`MemoryVerbs.prune`."""
    return db.prune(**kw)


def as_of(timestamp: _dt.datetime, *, tx_time: _dt.datetime | None = None) -> AsOf:
    """Build an :class:`~anatid.types.AsOf` scope for a timestamp.

    ``db.as_of(t)`` returns a bound view instead; this is the plain value, for passing as
    ``as_of=`` to any read.  Remember what it is: anatid's WHERE filter over the bitemporal
    columns, not a DuckDB feature.
    """
    ts = to_utc_naive(timestamp)
    return AsOf(valid_time=ts, tx_time=to_utc_naive(tx_time) if tx_time is not None else ts)


def provenance(db: MemoryVerbs, memory_id: int, **kw) -> Provenance:
    """Function form of :meth:`MemoryVerbs.provenance`."""
    return db.provenance(memory_id, **kw)
