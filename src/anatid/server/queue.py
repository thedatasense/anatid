"""Per-tenant write queues, batching, backpressure and idempotency keys.

The shape
---------
One process owns the files, so every write for a tenant arrives on some connection thread and
has to reach that tenant's DuckDB handle.  Letting the connection threads write directly would
work -- ``Anatid`` is thread-safe and DuckDB allows many writer threads in one process -- but it
gives up three things this module exists for.

*Backpressure that is visible.*  A queue with a bound tells a client it is going too fast.  An
unbounded one tells it nothing until memory runs out, and a blocking one tells it nothing until
its own timeout fires.  A full queue here answers :class:`~anatid.server.protocol.BusyError`,
which is retryable and carries the wait the server estimates from the rate it is actually
draining that tenant.

*Batching.*  A ``remember`` is 1.77 ms in process, and most of that is transaction overhead
rather than the row.  Writes for one tenant that are queued at the same moment commit in one
transaction, which is the difference between paying that once and paying it per write.  It is
not unconditional: a verb that must not share a transaction (a hard purge, a prune, an index
rebuild) is marked ``batchable=False`` and gets a transaction of its own.

*Fairness.*  A tenant is served, then goes to the back of the ring whether or not it still has
work.  A tenant with a thousand queued writes therefore yields after ``batch_max`` of them, and
a tenant with three writes waits for at most one other tenant's batch per tenant ahead of it,
not for the thousand.  ``tests/test_server_queue.py`` runs two tenants at very different rates
and asserts the quiet one still finishes.

What it does not add
--------------------
Serializability.  DuckDB gives optimistic snapshot isolation with write-write aborts, and
funnelling writes through one process does not change that.  What changes is where a conflict
can happen: between two threads of one process, where a per-tenant in-flight flag means the
queue never runs two batches for one tenant at once, instead of between two processes, where it
could not happen at all because the second process could not open the file.

Idempotency
-----------
A retried write must not write twice.  :class:`IdempotencyStore` records the key and the
original response IN THE SAME TRANSACTION as the write, in a table in the anatid file itself, so
a crash between the write and the record is not a state that exists and a server restart does not
turn a retry into a duplicate.  Only successes are recorded: a write that failed did not happen,
so its retry should actually run.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import random
import threading
import time
from collections import deque
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..types import utcnow
from ..visibility import tenant_sql
from . import protocol
from .protocol import BusyError, DeadlineExceeded, IdempotencyConflict, ShuttingDown

log = logging.getLogger("anatid.server.queue")

__all__ = [
    "IDEMPOTENCY_TABLE",
    "IDEMPOTENCY_DDL",
    "ensure_idempotency_table",
    "install_migration",
    "IdempotencyRecord",
    "IdempotencyStore",
    "QueuedWrite",
    "QueueStats",
    "DrainReport",
    "WriteQueue",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_BATCH_MAX",
    "DEFAULT_WORKERS",
    "DEFAULT_IDEMPOTENCY_TTL",
]

#: Queued writes a single tenant may have outstanding before the server answers busy.
DEFAULT_MAX_DEPTH = 256

#: Writes for one tenant that may share a transaction.  Above roughly this the batch stops
#: amortising the transaction and starts holding a write open behind unrelated work.
DEFAULT_BATCH_MAX = 32

#: Worker threads draining the queues.  Each serves one tenant at a time and no tenant is served
#: by two at once, so this is the number of tenants that can be written in parallel.
DEFAULT_WORKERS = 4

#: How long an idempotency key is remembered, in seconds.  A day: long enough to cover a client
#: that retries after a restart, short enough that the table does not become a log.
DEFAULT_IDEMPOTENCY_TTL = 24 * 60 * 60.0


# --------------------------------------------------------------------------- idempotency


#: The table idempotency records live in, inside the anatid file itself.  Not a sidecar: a
#: sidecar cannot be written in the same transaction as the memory, and a record that is not in
#: the write's transaction is a record that a crash can separate from its write.
IDEMPOTENCY_TABLE = "anatid_idempotency"

#: The DDL, as one idempotent statement.  It lives here rather than in :mod:`anatid.schema`
#: because the server owns it: a build with no server never needs the table, and an anatid file
#: that never met a server should not carry it.
IDEMPOTENCY_DDL = (
    f"CREATE TABLE IF NOT EXISTS {IDEMPOTENCY_TABLE} ("
    " tenant_id INTEGER NOT NULL,"
    " idempotency_key VARCHAR NOT NULL,"
    " verb VARCHAR NOT NULL,"
    " request_digest VARCHAR NOT NULL,"
    " response BLOB NOT NULL,"
    " created_at TIMESTAMP NOT NULL,"
    " expires_at TIMESTAMP NOT NULL,"
    " PRIMARY KEY (tenant_id, idempotency_key))"
)

#: An index for the expiry sweep.  The primary key already covers the lookup.
IDEMPOTENCY_INDEX_DDL = (
    f"CREATE INDEX IF NOT EXISTS idx_{IDEMPOTENCY_TABLE}_expires "
    f"ON {IDEMPOTENCY_TABLE} (expires_at)"
)


def ensure_idempotency_table(con: Any) -> None:
    """Create the idempotency table and its index on ``con`` if they are not there.

    Idempotent DDL run once per handle rather than a registered schema migration, and that is a
    deliberate choice, not a shortcut.  :data:`anatid.schema.SCHEMA_VERSION` is 4;
    ``MIGRATIONS[5]`` would only run if that constant moved to 5, and moving it is an edit to
    ``schema.py`` that would collide with the other work in flight on this tree.  Registering a
    step that never runs is worse than not registering one, because a file would then claim a
    version whose table it does not have.  :func:`install_migration` is the hook for the release
    that does bump it; until then this runs at server start, before any write transaction opens,
    because DuckDB refuses to commit a transaction that modifies a table's rows and then alters
    the same table.
    """
    con.execute(IDEMPOTENCY_DDL)
    con.execute(IDEMPOTENCY_INDEX_DDL)


def install_migration(to_version: int) -> None:
    """Register the idempotency DDL as the schema migration step to ``to_version``.

    For the release that raises :data:`anatid.schema.SCHEMA_VERSION`: call this once at import
    time with the new version, in the same change that raises the constant.  It is a no-op to
    call twice with the same version, and it raises if some other module already claimed the
    step, which is the collision worth failing on.
    """
    from .. import schema as _schema

    existing = _schema.MIGRATIONS.get(to_version)
    if existing is not None:
        if getattr(existing, "anatid_idempotency_migration", False):
            return
        raise ValueError(
            f"migration to schema version {to_version} is already registered by "
            f"{getattr(existing, '__module__', '?')}.{getattr(existing, '__name__', '?')}"
        )

    def _step(con: Any) -> None:
        ensure_idempotency_table(con)

    # The marker makes a second call with the same version a no-op instead of a collision.
    setattr(_step, "anatid_idempotency_migration", True)  # noqa: B010
    _schema.register_migration(to_version)(_step)


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    """One remembered write: the key, what it was for, and the response it produced.

    ``response`` is the original result encoded with :func:`anatid.server.protocol.dumps`.
    Replaying it rebuilds an equal value, not the identical object: a replayed ``remember``
    returns a ``Memory`` with the same fields as the one the first attempt returned, which is
    what a client that never saw the first response needs.
    """

    tenant_id: int
    key: str
    verb: str
    digest: str
    response: bytes
    created_at: Any
    expires_at: Any

    def value(self) -> Any:
        """The recorded result, decoded."""
        return protocol.loads(self.response)


class IdempotencyStore:
    """Idempotency keys, scoped per tenant, persisted in the anatid file.

    The contract:

    * A key is scoped to one tenant.  The same key from two tenants is two writes, and a
      principal that cannot name a tenant cannot reach its keys either.
    * A key that already succeeded returns the ORIGINAL response.  The write does not run again.
    * A key reused with different arguments is refused with
      :class:`~anatid.server.protocol.IdempotencyConflict`.  Returning the first write's result
      for the second write's arguments would hide a client bug behind a success.
    * Only successes are recorded.  A write that raised did not happen, so its retry runs.  The
      exception is not remembered, because remembering it would need a second transaction after
      the first rolled back, and a crash between those two is the state this class exists to
      make impossible.
    * Records expire after ``ttl`` seconds.  Expired ones are ignored on lookup and removed by
      :meth:`purge_expired`, which the queue calls every ``purge_every`` batches.
    """

    def __init__(
        self,
        *,
        ttl: float = DEFAULT_IDEMPOTENCY_TTL,
        purge_every: int = 512,
        now: Callable[[], Any] = utcnow,
    ) -> None:
        if ttl <= 0:
            raise ValueError("ttl must be greater than zero seconds")
        self.ttl = float(ttl)
        self.purge_every = max(1, int(purge_every))
        self._now = now
        self._ensured: set[tuple[str, int]] = set()
        self._lock = threading.Lock()

    # -- table lifecycle ---------------------------------------------------------------

    def ensure_table(self, db: Any) -> None:
        """Make sure ``db`` has the table.  Cheap after the first call per handle.

        Call it outside a write transaction.  DuckDB refuses to commit a transaction that
        modifies a table's rows and then alters that table, so the queue calls this before it
        opens a batch's transaction, never inside one.

        The result is remembered per handle, because the statement costs 137 us even when the
        table is already there and a batch would otherwise pay it every time.  The memo is
        keyed on the file's path AND the handle's identity: on identity alone a recycled
        ``id()`` could make a second tenant's handle inherit the first one's note and skip the
        DDL for a file that does not have the table.
        """
        marker = (str(getattr(db, "path", "")), id(db))
        with self._lock:
            if marker in self._ensured:
                return
        ensure_idempotency_table(db.connection)
        with self._lock:
            self._ensured.add(marker)

    def forget_handle(self, db: Any) -> None:
        """Drop the "already ensured" note for a handle that is being closed or evicted."""
        with self._lock:
            self._ensured.discard((str(getattr(db, "path", "")), id(db)))

    # -- keys --------------------------------------------------------------------------

    @staticmethod
    def digest_for(verb: str, args: Mapping[str, Any]) -> str:
        """A stable digest of one call, used to catch a key reused for a different request.

        Encoded through the protocol codec first, so a datetime or a ``Memory`` in the arguments
        digests as what goes on the wire rather than as its ``repr``, and sorted, so two clients
        that build the same call in a different key order agree.
        """
        body = json.dumps(
            protocol.encode_value(dict(args)),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(f"{verb}\0{body}".encode()).hexdigest()

    # -- read and write ----------------------------------------------------------------

    def lookup(
        self, db: Any, tenant_id: int, key: str, *, verb: str, digest: str
    ) -> IdempotencyRecord | None:
        """The record for ``key`` in ``tenant_id``, or None when there is none or it expired.

        Raises :class:`~anatid.server.protocol.IdempotencyConflict` when a record exists whose
        request digest is not ``digest``.
        """
        # The tenant predicate comes from anatid.visibility like every other read's does.
        # This table carries no valid or transaction time on purpose -- an idempotency record
        # is a fact about a request, not a belief about the world, and it is never read "as of"
        # anything -- so it takes tenant_sql(), the helper for exactly that case, rather than
        # Visibility.predicate(), which would filter on columns this table does not have.
        row = db.execute(
            f"SELECT tenant_id, idempotency_key, verb, request_digest, response, created_at,"
            f" expires_at FROM {IDEMPOTENCY_TABLE}"
            f" WHERE {tenant_sql()} AND idempotency_key = ?",
            [int(tenant_id), str(key)],
        ).fetchone()
        if row is None:
            return None
        record = IdempotencyRecord(
            tenant_id=int(row[0]),
            key=str(row[1]),
            verb=str(row[2]),
            digest=str(row[3]),
            response=bytes(row[4]),
            created_at=row[5],
            expires_at=row[6],
        )
        if record.expires_at is not None and record.expires_at <= self._now():
            return None
        if record.digest != digest:
            raise IdempotencyConflict(
                f"idempotency key {key!r} was already used for a different {record.verb} "
                f"request in tenant {tenant_id}. A key identifies one write; reusing it with "
                f"different arguments would return the first write's result for the second "
                f"write's request. Use a new key.",
                key=key,
                recorded_verb=record.verb,
            )
        return record

    def record(
        self, db: Any, tenant_id: int, key: str, *, verb: str, digest: str, value: Any
    ) -> IdempotencyRecord:
        """Remember ``value`` as the response for ``key``.

        Call this inside the write's own transaction.  That is the whole mechanism: the memory
        row and the record that says the memory was written commit together or neither commits,
        so there is no instant at which the write exists and the key does not.
        """
        now = self._now()
        expires = now + _timedelta_seconds(self.ttl)
        blob = protocol.dumps(value)
        db.execute(
            f"INSERT INTO {IDEMPOTENCY_TABLE}"
            f" (tenant_id, idempotency_key, verb, request_digest, response, created_at,"
            f" expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [int(tenant_id), str(key), str(verb), str(digest), blob, now, expires],
        )
        return IdempotencyRecord(
            tenant_id=int(tenant_id),
            key=str(key),
            verb=str(verb),
            digest=str(digest),
            response=blob,
            created_at=now,
            expires_at=expires,
        )

    def purge_expired(self, db: Any, *, now: Any = None) -> int:
        """Delete every record whose window has closed.  Returns how many rows went."""
        cutoff = self._now() if now is None else now
        before = self.count(db)
        db.execute(f"DELETE FROM {IDEMPOTENCY_TABLE} WHERE expires_at <= ?", [cutoff])
        return before - self.count(db)

    def count(self, db: Any, tenant_id: int | None = None) -> int:
        """How many records the file holds, optionally for one tenant."""
        if tenant_id is None:
            row = db.execute(f"SELECT count(*) FROM {IDEMPOTENCY_TABLE}").fetchone()
        else:
            row = db.execute(
                f"SELECT count(*) FROM {IDEMPOTENCY_TABLE} WHERE {tenant_sql()}",
                [int(tenant_id)],
            ).fetchone()
        return 0 if row is None else int(row[0])


def _timedelta_seconds(seconds: float) -> Any:
    import datetime as _dt

    return _dt.timedelta(seconds=seconds)


# --------------------------------------------------------------------------- queue items


@dataclass(slots=True)
class QueuedWrite:
    """One write waiting for its tenant's turn.

    ``work`` is called with the tenant's :class:`~anatid.Anatid` handle, inside a transaction
    the queue has already opened, and returns whatever the verb returns.  It must not open or
    commit a transaction of its own -- ``db.transaction()`` is re-entrant and will join the
    queue's, which is what the verbs already do.

    ``batchable`` is False for a verb that must not share a transaction with a neighbour: a hard
    purge, a prune, an index rebuild.  Those get a transaction to themselves.

    ``deadline`` is an absolute value on the queue's monotonic clock.  It is checked before the
    transaction opens and not after: a write that has started is finished or aborted by DuckDB,
    and killing it half-way is neither available nor desirable.
    """

    tenant_id: int
    verb: str
    work: Callable[[Any], Any]
    future: "Future[Any]" = field(default_factory=Future)
    idempotency_key: str | None = None
    digest: str | None = None
    batchable: bool = True
    deadline: float | None = None
    enqueued_at: float = 0.0

    def expired(self, now: float) -> bool:
        return self.deadline is not None and now >= self.deadline


@dataclass(frozen=True, slots=True)
class QueueStats:
    """A snapshot of what the queue has done and is holding.

    ``depth`` is per tenant and only names tenants with something queued.  ``max_tenant_depth``
    is the fullest single queue, which is what readiness compares against its high-water mark:
    the total across tenants says nothing about whether any one tenant is being refused.
    """

    running: bool
    workers: int
    max_depth: int
    batch_max: int
    queued: int
    in_flight: int
    depth: dict[int, int] = field(default_factory=dict)
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    rejected: int = 0
    expired: int = 0
    replayed: int = 0
    batches: int = 0
    batched_items: int = 0
    split_batches: int = 0

    @property
    def max_tenant_depth(self) -> int:
        return max(self.depth.values(), default=0)

    @property
    def mean_batch(self) -> float:
        """Writes per transaction, averaged over every batch run.  1.0 means no batching happened."""
        return self.batched_items / self.batches if self.batches else 0.0


@dataclass(frozen=True, slots=True)
class DrainReport:
    """What a :meth:`WriteQueue.drain` or :meth:`WriteQueue.stop` managed, and what it did not.

    ``drained`` counts the writes THIS call finished, successfully or not.  It is a delta taken
    across the call and not the queue's lifetime total, so a server that has served a million
    writes and drains three on the way out reports three.  ``QueueStats.completed`` is where the
    lifetime numbers live.

    ``abandoned`` is the count of writes still queued when the timeout ran out; their futures
    are completed with :class:`~anatid.server.protocol.ShuttingDown`, so no caller is left
    waiting on a future nothing will ever finish.  ``abandoned_by_tenant`` names them, because
    "we dropped 40 writes" without saying whose is not an operational report.

    ``drained`` and ``abandoned`` partition what was outstanding: for a :meth:`WriteQueue.stop`
    they add up to everything the queue had left to do when the stop began plus anything it
    accepted before ``stop_accepting`` took effect, with nothing counted twice and nothing
    dropped.
    """

    drained: int
    abandoned: int
    timed_out: bool
    duration_s: float
    abandoned_by_tenant: dict[int, int] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return self.abandoned == 0 and not self.timed_out


#: Registered here, where the types are defined, and not in the module that happens to return
#: them.  ``queue_stats`` is a public verb, so a server built with ``AnatidServer(pool=...)`` has
#: to be able to encode its answer; when this lived in :mod:`anatid.server.cli` the verb worked
#: or failed depending on whether the process had imported the command line at all.
protocol.register_dataclass(QueueStats)
protocol.register_dataclass(DrainReport)


# --------------------------------------------------------------------------- the queue


class WriteQueue:
    """Bounded per-tenant write queues drained by a fixed pool of worker threads.

    ``open_db`` is called with a tenant id and returns that tenant's handle.  With
    :class:`anatid.DatabasePool` that is ``pool.get``; with a single shared file it is a lambda
    returning the one handle.  It is called on a worker thread, so it has to be thread-safe --
    ``DatabasePool.get`` is.

    Scheduling is one ring of tenants that have work and are not already being written.  A
    worker takes the tenant at the front, drains up to ``batch_max`` of its writes, runs them,
    and puts the tenant at the BACK if it still has more.  Two consequences, and they are the
    reason for the shape: no tenant is ever written by two workers at once, so a tenant's writes
    keep their order and never conflict with each other; and a tenant with a full queue yields
    after ``batch_max`` writes, so a tenant with three waits for at most one batch per tenant
    ahead of it.
    """

    def __init__(
        self,
        open_db: Callable[[int], Any],
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        batch_max: int = DEFAULT_BATCH_MAX,
        workers: int = DEFAULT_WORKERS,
        idempotency: IdempotencyStore | None = None,
        high_water: float = 0.8,
        min_retry_after: float = 0.005,
        max_retry_after: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        name: str = "anatid-write",
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if batch_max < 1:
            raise ValueError("batch_max must be at least 1")
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if not 0.0 < high_water <= 1.0:
            raise ValueError("high_water is a fraction of max_depth in (0, 1]")
        self.open_db = open_db
        self.max_depth = int(max_depth)
        self.batch_max = int(batch_max)
        self.workers = int(workers)
        self.idempotency = idempotency
        self.high_water = float(high_water)
        self.min_retry_after = float(min_retry_after)
        self.max_retry_after = float(max_retry_after)
        self.name = name
        self._clock = clock

        self._cv = threading.Condition(threading.Lock())
        #: Counters are bumped from every worker thread.  ``x += 1`` is a load, an add and a
        #: store, and two threads can interleave them and lose one, so the counters get a lock
        #: of their own rather than a number that is quietly a little bit wrong.
        self._counters = threading.Lock()
        self._queues: dict[int, deque[QueuedWrite]] = {}
        self._ring: deque[int] = deque()
        self._in_ring: set[int] = set()
        self._serving: set[int] = set()
        #: The batch each busy worker is running, so a shutdown that cannot wait for it can
        #: still complete its futures rather than leave a caller holding one forever.
        self._inflight: dict[int, list[QueuedWrite]] = {}
        self._rate: dict[int, float] = {}
        #: Writes refused per tenant since that tenant was last below its high-water mark: the
        #: size of the crowd competing for the next slot, which is what the wait hint scales on.
        self._refusals: dict[int, int] = {}
        #: The depth at which a queue counts as nearly full.  One number, so the wait hint and
        #: readiness cannot disagree about where the mark is.
        self._high_water_depth = max(1, int(self.max_depth * self.high_water))
        self._threads: list[threading.Thread] = []
        self._running = False
        # Accepting from construction, not from start().  A queue that refused work until its
        # workers were running would make "queue everything, then start draining" impossible,
        # and that is exactly the shape a deterministic scheduling test needs and a server needs
        # while it is still opening files.
        self._accepting = True
        #: Set by :meth:`abandon` and cleared by :meth:`start`.  A drain already waiting on
        #: another thread reads it and gives up, which is what makes a second SIGTERM mean
        #: something.
        self._abandon_requested = False
        self._batches_since_purge = 0

        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._rejected = 0
        self._expired = 0
        self._replayed = 0
        self._batches = 0
        self._batched_items = 0
        self._split_batches = 0

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> "WriteQueue":
        """Start the worker threads.  Idempotent."""
        with self._cv:
            if self._running:
                return self
            self._running = True
            self._accepting = True
            self._abandon_requested = False
            self._threads = [
                threading.Thread(target=self._worker, name=f"{self.name}-{i}", daemon=True)
                for i in range(self.workers)
            ]
        for t in self._threads:
            t.start()
        return self

    def __enter__(self) -> "WriteQueue":
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def accepting(self) -> bool:
        """True while :meth:`submit` will take new work.  Set False first by :meth:`stop`."""
        return self._accepting

    def stop_accepting(self) -> None:
        """Refuse new writes but keep draining.  The first half of a graceful shutdown."""
        with self._cv:
            self._accepting = False
            self._cv.notify_all()

    def abandon(self) -> None:
        """Stop waiting: a :meth:`drain` in progress returns now, with what is left unrun.

        The escape hatch a second SIGTERM pulls.  A drain is a wait with a budget, and an
        operator who signals twice has decided that the budget is still too long; without this
        the second signal has nothing to shorten, because the wait is happening on another
        thread with its own deadline already fixed.  New writes are refused from here on, and
        :meth:`stop` is what completes the leftover futures with
        :class:`~anatid.server.protocol.ShuttingDown`.
        """
        with self._cv:
            self._abandon_requested = True
            self._accepting = False
            self._cv.notify_all()

    def drain(self, timeout: float | None = None) -> DrainReport:
        """Wait for EVERY tenant's queue to empty, without stopping the workers.

        Returns when everything queued at the moment of the call, and everything queued while
        waiting, has run -- or when ``timeout`` seconds pass, in which case the report says how
        many writes were left and whose they were.  Nothing is abandoned by :meth:`drain`; it
        reports, and :meth:`stop` is what completes the leftovers.

        The wait is process-wide, so a busy tenant delays it however quiet the others are.  Use
        :meth:`drain_tenant` when the thing being waited for belongs to one tenant.
        """
        started = self._clock()
        deadline = None if timeout is None else started + float(timeout)
        baseline = self._finished()
        with self._cv:
            while self._queued_locked() or self._serving:
                remaining = None if deadline is None else deadline - self._clock()
                if self._abandon_requested or (remaining is not None and remaining <= 0):
                    left = self._depth_locked()
                    return DrainReport(
                        drained=self._finished() - baseline,
                        abandoned=sum(left.values()),
                        timed_out=True,
                        duration_s=self._clock() - started,
                        abandoned_by_tenant=left,
                    )
                self._cv.wait(timeout=0.01 if remaining is None else min(0.01, remaining))
        return DrainReport(
            drained=self._finished() - baseline,
            abandoned=0,
            timed_out=False,
            duration_s=self._clock() - started,
        )

    def drain_tenant(self, tenant_id: int, timeout: float | None = None) -> DrainReport:
        """Wait for ONE tenant's queue to empty and its batch to finish.

        The difference from :meth:`drain` is the whole reason this exists: a backup pauses the
        tenant it copies and nothing else, so it must not wait behind another tenant's traffic.
        Writes for other tenants keep being served throughout and do not extend this wait.

        Nothing is abandoned here either.  A report with ``timed_out`` true means the budget ran
        out with ``abandoned`` writes still queued for this tenant, and the caller has to decide
        what that means for what it was about to do.

        ``abandoned`` counts this tenant only.  ``drained`` counts every write the queue finished
        while this call waited, whoever it belonged to, because the workers do not stop serving
        other tenants to satisfy this wait and pretending otherwise would need a per-tenant
        counter nothing else wants.
        """
        tenant_id = int(tenant_id)
        started = self._clock()
        deadline = None if timeout is None else started + float(timeout)
        baseline = self._finished()
        with self._cv:
            while self._tenant_busy_locked(tenant_id):
                remaining = None if deadline is None else deadline - self._clock()
                if self._abandon_requested or (remaining is not None and remaining <= 0):
                    q = self._queues.get(tenant_id)
                    left = 0 if q is None else len(q)
                    return DrainReport(
                        drained=self._finished() - baseline,
                        abandoned=left,
                        timed_out=True,
                        duration_s=self._clock() - started,
                        abandoned_by_tenant={tenant_id: left} if left else {},
                    )
                self._cv.wait(timeout=0.01 if remaining is None else min(0.01, remaining))
        return DrainReport(
            drained=self._finished() - baseline,
            abandoned=0,
            timed_out=False,
            duration_s=self._clock() - started,
        )

    def _tenant_busy_locked(self, tenant_id: int) -> bool:
        q = self._queues.get(tenant_id)
        return bool(q) or tenant_id in self._serving

    def stop(self, *, timeout: float | None = 30.0, drain: bool = True) -> DrainReport:
        """Stop accepting, optionally drain, then join the workers.

        Whatever is still queued when ``timeout`` runs out is completed with
        :class:`~anatid.server.protocol.ShuttingDown` and counted in the report, so no caller is
        left holding a future that will never finish.  ``drain=False`` abandons immediately,
        which is what a second SIGTERM should do.

        ``drained`` counts what this stop finished and ``abandoned`` what it gave up on, and the
        two are taken so that they partition the work outstanding when the stop began.  The
        count is taken AFTER the workers are joined, so a write that finished between the end of
        the drain and the abandon is counted as drained rather than falling between them.
        """
        started = self._clock()
        baseline = self._finished()
        self.stop_accepting()
        report = self.drain(timeout) if drain else DrainReport(0, 0, False, 0.0)
        with self._cv:
            self._running = False
            self._cv.notify_all()
        # Abandon BEFORE joining, not after.  A worker that is told to stop while its queues
        # still hold work would otherwise keep taking batches, and the join would wait for the
        # backlog this call has already decided not to wait for.
        abandoned = self._abandon_everything()
        for t in self._threads:
            t.join(timeout=None if timeout is None else max(1.0, timeout))
        # Whatever a worker was still running when the join gave up.  Rare, and the report is
        # the only honest thing to do with it: the write may or may not have committed.
        stranded = self._abandon_inflight()
        for tenant_id, count in stranded.items():
            abandoned[tenant_id] = abandoned.get(tenant_id, 0) + count
        self._threads = []
        return DrainReport(
            drained=self._finished() - baseline,
            abandoned=sum(abandoned.values()),
            timed_out=report.timed_out or bool(abandoned),
            duration_s=self._clock() - started,
            abandoned_by_tenant=abandoned,
        )

    def _abandon_inflight(self) -> dict[int, int]:
        """Complete the futures of any batch a worker did not finish before the join gave up."""
        with self._cv:
            batches = {t: list(items) for t, items in self._inflight.items()}
        counts: dict[int, int] = {}
        for tenant_id, items in batches.items():
            pending = [i for i in items if not i.future.done()]
            if not pending:
                continue
            counts[tenant_id] = len(pending)
            for item in pending:
                with contextlib.suppress(InvalidStateError):
                    item.future.set_exception(
                        ShuttingDown(
                            f"the server stopped while this {item.verb} for tenant "
                            f"{tenant_id} was running; whether it committed is not known "
                            f"here. Re-send it with an idempotency key to find out."
                        )
                    )
        return counts

    def _abandon_everything(self) -> dict[int, int]:
        with self._cv:
            leftovers = [(t, list(q)) for t, q in self._queues.items() if q]
            for q in self._queues.values():
                q.clear()
            self._ring.clear()
            self._in_ring.clear()
            self._refusals.clear()
        counts: dict[int, int] = {}
        for tenant_id, items in leftovers:
            counts[tenant_id] = len(items)
            for item in items:
                if not item.future.done():
                    with contextlib.suppress(InvalidStateError):
                        item.future.set_exception(
                            ShuttingDown(
                                f"the server stopped before this {item.verb} for tenant "
                                f"{tenant_id} ran; nothing was written, send it again"
                            )
                        )
        return counts

    # -- submitting --------------------------------------------------------------------

    def submit(
        self,
        tenant_id: int,
        verb: str,
        work: Callable[[Any], Any],
        *,
        idempotency_key: str | None = None,
        digest: str | None = None,
        batchable: bool = True,
        deadline: float | None = None,
    ) -> "Future[Any]":
        """Queue one write and return a future for its result.

        ``deadline`` is seconds of budget from now, not an absolute time: the request's budget
        is measured by this process, so a client clock cannot expire a write on arrival or fail
        to expire one ever.

        Raises :class:`~anatid.server.protocol.BusyError` when the tenant's queue is full, with
        a suggested wait, and :class:`~anatid.server.protocol.ShuttingDown` when the queue has
        stopped accepting.  It never blocks: a caller that wants to wait can wait on the wait it
        was given.
        """
        item = QueuedWrite(
            tenant_id=int(tenant_id),
            verb=verb,
            work=work,
            idempotency_key=idempotency_key,
            digest=digest,
            batchable=bool(batchable),
            deadline=None if deadline is None else self._clock() + float(deadline),
            enqueued_at=self._clock(),
        )
        with self._cv:
            if not self._accepting:
                raise ShuttingDown(
                    "the server has stopped accepting writes and is draining; retry against "
                    "another instance"
                )
            q = self._queues.get(item.tenant_id)
            if q is None:
                q = self._queues[item.tenant_id] = deque()
            if len(q) >= self.max_depth:
                self._rejected += 1
                refused = self._refusals.get(item.tenant_id, 0) + 1
                self._refusals[item.tenant_id] = refused
                wait = self._retry_after_locked(item.tenant_id, len(q), refused)
                raise BusyError(
                    f"tenant {item.tenant_id} has {len(q)} writes queued, its limit of "
                    f"{self.max_depth}, and {refused} writes have been refused since it was "
                    f"last below its high-water mark. The write was NOT performed. Retry after "
                    f"about {wait:.3f}s, or send less.",
                    retry_after=wait,
                    tenant_id=item.tenant_id,
                    depth=len(q),
                    max_depth=self.max_depth,
                )
            q.append(item)
            if len(q) < self._high_water_depth:
                # Back under the mark, so the crowd this tenant was refusing has dispersed and
                # the next one starts counting again from one.
                self._refusals.pop(item.tenant_id, None)
            self._submitted += 1
            if item.tenant_id not in self._in_ring and item.tenant_id not in self._serving:
                self._ring.append(item.tenant_id)
                self._in_ring.add(item.tenant_id)
            self._cv.notify()
        return item.future

    def _retry_after_locked(self, tenant_id: int, depth: int, refused: int) -> float:
        """How long this caller should wait, from the drain rate AND the size of the crowd.

        Not a constant, because a constant is either a lie about a fast tenant or a lie about a
        slow one.  The rate is an exponentially weighted average of writes per second measured
        over the batches already run for this tenant; with no measurement yet the queue guesses
        from ``batch_max`` and a millisecond a write, which errs low and gets corrected by the
        first batch.

        The overflow alone is not the answer, and this used to give it anyway.  Time to drain the
        overflow is the time until the queue has room, which is the time until ONE more write
        fits; it does not depend on how many callers are waiting for that room, so sixteen
        clients were each told to come back in five milliseconds, came back together, and fifteen
        of them were refused again.  Measured before this changed: 16 threads against a queue of
        4, 231 of 640 writes got through with the shipped three attempts.  So the wait is
        multiplied by ``refused``, the number of writes this tenant has turned away since it was
        last below its high-water mark, which is the closest thing the queue has to a count of
        the callers competing for the next slot.

        Then it is jittered down by up to half.  Identical advice makes a crowd retry in step,
        which reproduces the collision it was given to avoid; spreading the returns over an
        interval is what actually clears a queue.  The number a caller gets is therefore an
        upper bound it should not beat, not a reservation, and ``BusyError`` says so.
        """
        rate = self._rate.get(tenant_id, 0.0)
        if rate <= 0.0:
            rate = float(self.batch_max) * 1000.0
        over = max(1, depth - self._high_water_depth + 1)
        wait = (over * max(1, refused)) / rate
        wait *= 0.5 + 0.5 * random.random()
        return min(self.max_retry_after, max(self.min_retry_after, wait))

    # -- introspection -----------------------------------------------------------------

    def depth(self, tenant_id: int) -> int:
        """How many writes are queued for one tenant."""
        with self._cv:
            q = self._queues.get(int(tenant_id))
            return 0 if q is None else len(q)

    def _depth_locked(self) -> dict[int, int]:
        return {t: len(q) for t, q in self._queues.items() if q}

    def _queued_locked(self) -> int:
        return sum(len(q) for q in self._queues.values())

    def stats(self) -> QueueStats:
        """A consistent snapshot of the counters and the depths."""
        with self._cv:
            return QueueStats(
                running=self._running,
                workers=self.workers,
                max_depth=self.max_depth,
                batch_max=self.batch_max,
                queued=self._queued_locked(),
                in_flight=len(self._serving),
                depth=self._depth_locked(),
                submitted=self._submitted,
                completed=self._completed,
                failed=self._failed,
                rejected=self._rejected,
                expired=self._expired,
                replayed=self._replayed,
                batches=self._batches,
                batched_items=self._batched_items,
                split_batches=self._split_batches,
            )

    def above_high_water(self) -> bool:
        """True when some tenant's queue is at or above ``high_water * max_depth``.

        Readiness reads this.  A server whose queues are nearly full is alive and should stay
        alive; it is not ready for more traffic, and saying so is what lets a load balancer take
        it out of rotation instead of a client discovering it one busy response at a time.
        """
        with self._cv:
            return any(len(q) >= self._high_water_depth for q in self._queues.values())

    # -- the workers -------------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            with self._cv:
                while self._running and not self._ring:
                    self._cv.wait(timeout=0.05)
                if not self._ring:
                    if not self._running:
                        return
                    continue
                tenant_id = self._ring.popleft()
                self._in_ring.discard(tenant_id)
                batch = self._take_batch_locked(tenant_id)
                if not batch:
                    continue
                self._serving.add(tenant_id)
                self._inflight[tenant_id] = list(batch)
            started = self._clock()
            try:
                self._run_batch(tenant_id, batch)
            except BaseException as exc:  # a bug in this module, not in a verb
                log.exception("write queue worker failed on tenant %s", tenant_id)
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)
            finally:
                elapsed = max(1e-6, self._clock() - started)
                with self._cv:
                    self._serving.discard(tenant_id)
                    self._inflight.pop(tenant_id, None)
                    self._note_rate_locked(tenant_id, len(batch) / elapsed)
                    q = self._queues.get(tenant_id)
                    if q and tenant_id not in self._in_ring:
                        # The BACK of the ring, not the front: this is the whole fairness
                        # property.  A tenant with a thousand queued writes gets batch_max of
                        # them and then waits behind every other tenant that has work.
                        self._ring.append(tenant_id)
                        self._in_ring.add(tenant_id)
                    self._cv.notify_all()

    def _bump(self, name: str, amount: int = 1) -> None:
        with self._counters:
            setattr(self, name, getattr(self, name) + amount)

    def _finished(self) -> int:
        """Lifetime count of writes that reached an answer, succeeded or failed.

        Drain reports subtract two readings of this rather than quoting it, because "how many
        writes has this queue ever run" is not the answer to "what did this shutdown drain".
        """
        with self._counters:
            return self._completed + self._failed

    def _note_rate_locked(self, tenant_id: int, observed: float) -> None:
        previous = self._rate.get(tenant_id)
        self._rate[tenant_id] = observed if previous is None else 0.7 * previous + 0.3 * observed

    def _take_batch_locked(self, tenant_id: int) -> list[QueuedWrite]:
        q = self._queues.get(tenant_id)
        if not q:
            return []
        batch: list[QueuedWrite] = []
        while q and len(batch) < self.batch_max:
            head = q[0]
            if not head.batchable:
                if batch:
                    break  # it gets the next turn, and a transaction of its own
                batch.append(q.popleft())
                break
            batch.append(q.popleft())
        return batch

    # -- running a batch ---------------------------------------------------------------

    def _run_batch(self, tenant_id: int, batch: Sequence[QueuedWrite]) -> None:
        now = self._clock()
        live: list[QueuedWrite] = []
        for item in batch:
            if item.expired(now):
                self._bump("_expired")
                item.future.set_exception(
                    DeadlineExceeded(
                        f"this {item.verb} for tenant {tenant_id} waited "
                        f"{now - item.enqueued_at:.3f}s in the queue and its deadline passed "
                        f"before it ran. Nothing was written.",
                        waited=now - item.enqueued_at,
                    )
                )
            else:
                live.append(item)
        if not live:
            return
        db = self.open_db(tenant_id)
        if self.idempotency is not None:
            # Before the transaction, never inside one: DuckDB refuses to commit a transaction
            # that modified a table's rows and then altered that table.  Memoised per handle.
            self.idempotency.ensure_table(db)
        self._bump("_batches")
        self._bump("_batched_items", len(live))
        if len(live) == 1:
            self._run_alone(db, tenant_id, live[0])
        else:
            self._run_together(db, tenant_id, live)
        self._maybe_purge(db)

    def _run_together(self, db: Any, tenant_id: int, live: list[QueuedWrite]) -> None:
        """One transaction for the whole batch, falling back to one each when it fails.

        The fallback is the part that matters.  A batch that aborts committed nothing, so
        re-running each write in its own transaction cannot double-write, and the write that
        actually failed fails alone: its neighbours are not lost and are not attributed its
        error.  ``tests/test_server_queue.py`` puts a failing write in the middle of a batch and
        asserts exactly that.
        """
        try:
            results = self._commit(db, tenant_id, live)
        except Exception as exc:
            log.debug(
                "batch of %d for tenant %s failed (%s: %s), re-running one at a time",
                len(live),
                tenant_id,
                type(exc).__name__,
                exc,
            )
            self._bump("_split_batches")
            for item in live:
                self._run_alone(db, tenant_id, item)
            return
        for item, result in zip(live, results, strict=True):
            item.future.set_result(result)
        self._bump("_completed", len(live))

    def _run_alone(self, db: Any, tenant_id: int, item: QueuedWrite) -> None:
        try:
            result = self._commit(db, tenant_id, [item])[0]
        except Exception as exc:
            self._bump("_failed")
            if not item.future.done():
                item.future.set_exception(exc)
            return
        self._bump("_completed")
        if not item.future.done():
            item.future.set_result(result)

    def _commit(self, db: Any, tenant_id: int, items: Sequence[QueuedWrite]) -> list[Any]:
        """Run ``items`` in one transaction, replaying anything an idempotency key already covers."""
        results: list[Any] = []
        store = self.idempotency
        with db.transaction():
            for item in items:
                key = item.idempotency_key
                if store is not None and key is not None and item.digest is not None:
                    record = store.lookup(db, tenant_id, key, verb=item.verb, digest=item.digest)
                    if record is not None:
                        self._bump("_replayed")
                        results.append(record.value())
                        continue
                value = item.work(db)
                if store is not None and key is not None and item.digest is not None:
                    store.record(
                        db, tenant_id, key, verb=item.verb, digest=item.digest, value=value
                    )
                results.append(value)
        return results

    def _maybe_purge(self, db: Any) -> None:
        store = self.idempotency
        if store is None:
            return
        self._batches_since_purge += 1
        if self._batches_since_purge < store.purge_every:
            return
        self._batches_since_purge = 0
        try:
            removed = store.purge_expired(db)
        except Exception:
            log.warning("could not purge expired idempotency keys", exc_info=True)
            return
        if removed:
            log.debug("purged %d expired idempotency keys", removed)


def wait_all(futures: Iterable["Future[Any]"], timeout: float | None = None) -> list[Any]:
    """Every future's result, in order, raising the first exception any of them carries.

    A convenience for a caller that submitted a group of writes and wants them all or the
    reason it did not get them.
    """
    return [f.result(timeout=timeout) for f in futures]
