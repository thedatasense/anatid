"""Backups a running anatid server can actually stand behind, and the restore beside them.

Why a module and not ``cp``
---------------------------
A DuckDB file is a file, so copying it looks solved.  It is not.  While the server holds a
tenant's file it holds an EXCLUSIVE lock on it (measured on duckdb 1.5.5: a second process gets
``duckdb.IOException`` even asking for read-only), and the committed state is split between the
database file and its ``.wal`` sidecar, which DuckDB folds in on its own schedule.  ``cp`` of
those two files, taken a few milliseconds apart while four threads write, is a copy of a moment
no client ever observed.  Measured here on the 100k-memory spike file: ``shutil.copy2`` takes
11 ms against 494 ms for a coordinated copy, and the 483 ms it saves buys a backup with no
stated guarantee at all.  (Every number in this module is a median of four runs of
``tests/test_backup.py::test_backup_cost_on_the_hundred_thousand_memory_file``, which prints
the whole table under ``-s``.)

The three ways to take one, and what each promises
--------------------------------------------------
Every path below is DuckDB writing the copy from inside the process that owns the file, because
that is the only process that can read it.  They differ in where the upper boundary of the copy
sits, not in whether the copy is torn -- none of them is ever torn.

``backup(..., quiesce=True)``  (:attr:`Guarantee.QUIESCED`, the default)
    A barrier is queued on the tenant's own write queue.  Everything the queue had for that
    tenant runs first, the barrier then holds the tenant's single serving slot for the length of
    the copy, and everything submitted afterwards runs after it.  So: **every write acknowledged
    before the call is in the copy, and no write that commits after the barrier closes is in
    it.**  The boundary is a point in the tenant's own write order, which is the thing an
    operator can reason about.  Costs: the tenant's writes pause for the length of the copy
    (494 ms on 42 MiB), and one queue worker is occupied for that time.

``backup(..., quiesce=False)``  (:attr:`Guarantee.SNAPSHOT`)
    ``COPY FROM DATABASE`` with nothing paused.  DuckDB runs it in its own transaction, so what
    lands in the copy is a consistent committed state and an uncommitted transaction is never in
    it (asserted by ``test_an_unquiesced_copy_never_holds_an_uncommitted_transaction``, which
    holds an open INSERT on a second connection and finds it absent from the copy).  Every write
    acknowledged before the call is in it.  Writes acknowledged *during* the call may or may not
    be, and there is no way for the caller to find out which.  Nothing pauses.

``export(...)``  (:attr:`Guarantee.EXPORT`)
    ``EXPORT DATABASE ... (FORMAT PARQUET)`` into a directory: ``schema.sql``, ``load.sql`` and
    one Parquet file per table.  Same snapshot semantics as above, and it is the one path that
    runs happily inside a transaction.  Measured on the 100k file: 202 ms and 44.4 MiB against
    494 ms and 42.3 MiB for the database copy.  What it gives up is that the result is not an
    anatid file: :func:`restore` puts it back with ``IMPORT DATABASE``, which rebuilds the file
    at the *reading* build's storage format, and rebuilding costs what writing the rows costs
    (616 ms against 95 ms to put a file copy back and check it).  That is also the reason to
    keep one: it is the copy that survives a DuckDB storage-format change, and DuckDB has
    broken storage compatibility before.

The fourth way, which does not exist
------------------------------------
A read-only snapshot copy taken beside the writer.  It is unavailable at every level, measured:

    another process, read-only, while the server holds the file
        ``duckdb.IOException: Could not set lock on file ...: Conflicting lock is held``
    this process, ``duckdb.connect(path, read_only=True)``
        ``duckdb.ConnectionException: Can't open a connection to same database file with a
        different configuration than existing connections``
    the owning handle attaching its own file read-only
        ``duckdb.BinderException: Unique file handle conflict``

So there is no arrangement in which a separate reader takes the copy.  The owner takes it, or
nobody does, and that is why this module needs the server rather than a path.

What it does not cover
----------------------
Writes that do not go through the server's queue.  A quiesced backup pauses the queue's tenant,
and an embedded caller holding the same :class:`~anatid.Anatid` handle in-process can write
straight past it.  That is the same handle the server's own workers use, so the copy is still a
consistent snapshot; it is the *boundary* that becomes as vague as the snapshot guarantee.  With
``database=`` (one shared file, tenants as namespaces) the same applies to every tenant except
the one being quiesced: their rows are in the copy at whatever committed state the snapshot
found them in.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import fnmatch
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import duckdb

from ..database import Anatid, DatabasePool
from ..errors import AnatidError
from ..schema import SCHEMA_VERSION, current_version, quote_ident
from ..types import DoctorReport, utcnow
from . import protocol
from .protocol import BusyError, ShuttingDown
from .queue import WriteQueue

if TYPE_CHECKING:  # pragma: no cover - import cycle: server imports nothing from here
    from .server import AnatidServer

log = logging.getLogger("anatid.server.backup")

__all__ = [
    "Guarantee",
    "BackupError",
    "QuiesceTimeout",
    "QuiesceUnavailable",
    "DestinationInUse",
    "BackupUnreadable",
    "BackupReport",
    "BackupSet",
    "BackupInfo",
    "RestoreReport",
    "PruneReport",
    "BackupCoordinator",
    "sidecars",
    "in_use",
    "inspect",
    "restore",
    "prune",
    "DEFAULT_FILE_TEMPLATE",
    "DEFAULT_KEEP",
]

#: How a per-tenant file is named inside a directory when the caller gives a directory.
DEFAULT_FILE_TEMPLATE = "tenant-{tenant}.anatid"

#: How many backups :func:`prune` keeps when the caller does not say.
DEFAULT_KEEP = 7

#: Substrings that mark a DuckDB open failure as "someone else has this file" rather than
#: "this file is broken".  Matched case-insensitively against the first line of the message.
#: Two different exceptions say it: ``IOException`` across processes, ``ConnectionException``
#: inside this one, where DuckDB's instance cache refuses a second configuration.
_IN_USE_MARKERS = ("conflicting lock", "could not set lock", "different configuration")

#: What DuckDB leaves beside a database file.  ``.wal`` is the write-ahead log; ``.tmp`` is the
#: spill directory a large query creates.  Neither belongs in a backup taken by DuckDB itself
#: (a ``COPY FROM DATABASE`` destination is folded and self-contained, measured), and both have
#: to travel with a file copied by :func:`restore`.
_SIDECAR_SUFFIXES = (".wal", ".tmp")


# --------------------------------------------------------------------------- errors


class BackupError(AnatidError):
    """Something went wrong taking, checking or restoring a backup."""


class QuiesceTimeout(BackupError):
    """The tenant's write queue did not reach the backup barrier in time.

    The barrier waits behind everything already queued for that tenant, so this means the queue
    is deep, its writes are slow, or its workers are busy elsewhere.  Nothing was copied and
    nothing was paused.  Raise ``acquire_timeout``, or take the backup with ``quiesce=False``
    and accept the weaker boundary :attr:`Guarantee.SNAPSHOT` describes.
    """

    #: Nothing was copied and nothing was paused, so the same call can be made again unchanged.
    retryable = True


class QuiesceUnavailable(BackupError):
    """The barrier this backup needs cannot be held with the workers the queue has.

    Quiescing N tenants at once occupies N of the queue's worker threads for the length of the
    copy, so it needs ``workers >= N``.  Asking for more than that would wait for a worker that
    is itself waiting for this call, which is a deadlock, so it is refused instead.

    Also raised when the tenant's queue is full or has stopped accepting.  Nothing was copied and
    nothing was paused either way, and the queue being full is a transient condition, so this is
    retryable; a worker count too low for the tenants named is not, and a caller that retries it
    unchanged will be refused again with the same message.
    """

    retryable = True


class DestinationInUse(BackupError):
    """The path being written to is a database some process currently holds open.

    Restoring over a file a server owns would either fail half way or, worse, succeed and leave
    that server serving from a file it no longer matches.
    """


class BackupUnreadable(BackupError):
    """A backup file will not open, or does not look like an anatid database."""


# --------------------------------------------------------------------------- reports


class Guarantee(str, Enum):
    """Where the upper boundary of a copy sits.  See this module's docstring for the detail."""

    #: Every write acknowledged before the call is in it; nothing that commits after the barrier
    #: closes is in it.  The boundary is a point in the tenant's own write order.
    QUIESCED = "quiesced"
    #: A consistent committed state, never torn, never holding an uncommitted transaction.  A
    #: write acknowledged while the copy ran may or may not be in it.
    SNAPSHOT = "snapshot"
    #: The snapshot guarantee, written as Parquet plus DDL rather than as a database file.
    EXPORT = "export"


@dataclass(frozen=True)
class BackupReport:
    """One copy that was taken, and what can be said about it.

    ``guarantee`` is the guarantee for the WHOLE copy, so it is
    :attr:`Guarantee.QUIESCED` only when every tenant whose rows are in the file was paused for
    it.  With a :class:`~anatid.DatabasePool` that is one tenant and one file.  With a single
    shared file it is every tenant in the file, which a server cannot enumerate, so a shared-file
    copy reports :attr:`Guarantee.SNAPSHOT` and names in :attr:`quiesced_tenants` the tenants
    whose rows do have the stronger boundary.
    """

    path: Path
    tenant: int | None
    guarantee: Guarantee
    bytes: int
    seconds: float
    #: Seconds the tenants below were paused.  0.0 when nothing was paused.
    quiesced_for: float = 0.0
    #: Seconds spent waiting for the barrier to reach the front of the tenant's queue.
    waited_for: float = 0.0
    #: The tenants whose writes were held for the copy.  Their rows carry the quiesced boundary
    #: even when the copy as a whole does not.
    quiesced_tenants: tuple[int, ...] = ()
    schema_version: int | None = None
    expected_schema_version: int = SCHEMA_VERSION
    #: Row counts read back out of the copy, when ``verify`` was on.
    counts: dict[str, int] = field(default_factory=dict)
    #: The copy's own :meth:`~anatid.Anatid.doctor` report, when ``doctor`` was on.
    doctor: DoctorReport | None = None
    started_at: _dt.datetime = field(default_factory=utcnow)
    detail: str = ""

    @property
    def quiesced(self) -> bool:
        return self.guarantee is Guarantee.QUIESCED

    @property
    def megabytes(self) -> float:
        return self.bytes / (1024 * 1024)

    def __str__(self) -> str:
        where = "" if self.tenant is None else f"tenant {self.tenant} "
        return (
            f"{where}{self.guarantee.value} backup {self.path.name}: "
            f"{self.megabytes:.1f} MiB in {self.seconds * 1e3:.0f} ms"
        )


class BackupSet(tuple):  # type: ignore[type-arg]  # tuple[BackupReport, ...] needs 3.9+ at runtime
    """The reports from one :meth:`BackupCoordinator.backup` call.

    A tuple, so it iterates and indexes like the list it is, with the two questions an operator
    asks of a whole run answered on it.  It is always a ``BackupSet``, whether one tenant was
    copied or twenty, because a return type that changes shape with an argument makes every
    caller test the argument again.
    """

    __slots__ = ()

    @property
    def one(self) -> BackupReport:
        """The single report, when there is exactly one.  Raises otherwise."""
        if len(self) != 1:
            raise BackupError(
                f"this backup covered {len(self)} tenants; use it as a sequence, or name a "
                f"tenant to get one report"
            )
        return self[0]

    @property
    def paths(self) -> list[Path]:
        return [r.path for r in self]

    @property
    def bytes(self) -> int:
        return sum(r.bytes for r in self)

    @property
    def seconds(self) -> float:
        return sum(r.seconds for r in self)

    def __repr__(self) -> str:
        return f"<BackupSet {len(self)} copies, {self.bytes / (1024 * 1024):.1f} MiB>"


@dataclass(frozen=True)
class BackupInfo:
    """What :func:`inspect` found in a file or an export directory."""

    path: Path
    ok: bool
    kind: str  # "database", "export", "missing", "unreadable"
    schema_version: int | None = None
    expected_schema_version: int = SCHEMA_VERSION
    bytes: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    doctor: DoctorReport | None = None
    #: Set when some process holds the file open; the first line of DuckDB's own message.
    held_by: str | None = None
    detail: str = ""

    @property
    def needs_migration(self) -> bool:
        """True when this build would migrate the file on the first read-write open."""
        return self.schema_version is not None and self.schema_version < SCHEMA_VERSION

    @property
    def too_new(self) -> bool:
        """True when the file was written by a build newer than this one."""
        return self.schema_version is not None and self.schema_version > SCHEMA_VERSION


@dataclass(frozen=True)
class RestoreReport:
    """What :func:`restore` put where, and what it found there."""

    source: Path
    destination: Path
    kind: str  # "database" or "export"
    schema_version: int | None
    expected_schema_version: int
    bytes: int
    seconds: float
    sidecars: tuple[Path, ...] = ()
    doctor: DoctorReport | None = None
    detail: str = ""

    @property
    def needs_migration(self) -> bool:
        """True when opening the restored file read-write will run a schema migration.

        :func:`restore` never migrates.  A restore that silently upgraded the file would make
        the backup unreadable by the build that wrote it, which is the one build you know can
        read it.
        """
        return (
            self.schema_version is not None and self.schema_version < self.expected_schema_version
        )


@dataclass(frozen=True)
class PruneReport:
    """What :func:`prune` kept and what it removed."""

    directory: Path
    kept: tuple[Path, ...] = ()
    pruned: tuple[Path, ...] = ()
    #: Files that would not open, with the reason.  A bad backup is prunable; the newest good
    #: one never is.
    bad: dict[str, str] = field(default_factory=dict)
    #: Files some process holds open, with DuckDB's message.  Never pruned.
    held: dict[str, str] = field(default_factory=dict)
    #: Files that were chosen for pruning but could not be removed, with the reason.
    failed: dict[str, str] = field(default_factory=dict)
    freed_bytes: int = 0
    dry_run: bool = False
    detail: str = ""

    @property
    def kept_good(self) -> tuple[Path, ...]:
        return tuple(p for p in self.kept if p.name not in self.bad)


# --------------------------------------------------------------------------- file helpers


def sidecars(path: str | os.PathLike[str]) -> list[Path]:
    """The ``.wal`` and ``.tmp`` companions of ``path`` that exist right now.

    A copy that takes the database file and leaves the write-ahead log behind loses every
    committed write DuckDB has not folded in yet, which on a busy file is most of the recent
    ones.  :func:`restore` moves these with the file; a backup taken by DuckDB itself has none,
    because ``COPY FROM DATABASE`` writes a folded, self-contained destination (measured: no
    ``.wal`` beside a fresh copy).
    """
    base = Path(os.fspath(path))
    out: list[Path] = []
    for suffix in _SIDECAR_SUFFIXES:
        beside = Path(f"{base}{suffix}")
        if beside.exists():
            out.append(beside)
    return out


def in_use(path: str | os.PathLike[str]) -> str | None:
    """DuckDB's own message if some process holds ``path`` open, else None.

    Detected by trying to open it read-only, which is the cheapest question that gets a true
    answer: a process holding the file read-write excludes a read-only open (measured), and this
    process holding it excludes one too, because DuckDB's instance cache refuses a second
    configuration for one file.  So both the cross-process and the same-process case answer.

    A file that is merely broken is not "in use": the message is checked against
    :data:`_IN_USE_MARKERS` and anything else is left for :func:`inspect` to report as unreadable.
    A path that does not exist is not in use either.
    """
    p = Path(os.fspath(path))
    if not p.exists() or p.is_dir():
        return None
    try:
        con = duckdb.connect(str(p), read_only=True)
    except Exception as exc:
        first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        lowered = first.lower()
        if any(marker in lowered for marker in _IN_USE_MARKERS):
            return first
        return None
    con.close()
    return None


def _is_export_dir(path: Path) -> bool:
    return path.is_dir() and (path / "schema.sql").is_file()


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _file_bytes(path: Path) -> int:
    total = path.stat().st_size if path.is_file() else 0
    for extra in sidecars(path):
        total += _dir_bytes(extra) if extra.is_dir() else extra.stat().st_size
    return total


def inspect(
    path: str | os.PathLike[str],
    *,
    doctor: bool = False,
    counts: bool = True,
) -> BackupInfo:
    """Open a backup read-only and report what is in it.

    Read-only and ``ensure=False``, so inspecting a backup never migrates it, never creates a
    write-ahead log beside it and never touches a byte.  An export directory is recognised by
    its ``schema.sql`` and reported without being imported, because importing one costs a whole
    new database.

    ``doctor=True`` additionally runs :meth:`anatid.Anatid.doctor` on the copy, which is the
    check that answers "would anatid's verbs trust this file".  Measured on the 100k-memory
    spike file: 54 ms.
    """
    p = Path(os.fspath(path)).expanduser()
    if not p.exists():
        return BackupInfo(path=p, ok=False, kind="missing", detail=f"{p} does not exist")
    if _is_export_dir(p):
        tables = sorted(f.stem for f in p.glob("*.parquet"))
        return BackupInfo(
            path=p,
            ok=bool(tables),
            kind="export",
            bytes=_dir_bytes(p),
            counts={},
            detail=(
                f"EXPORT DATABASE directory, {len(tables)} tables; restore rebuilds the file "
                f"with IMPORT DATABASE at this build's storage format"
            ),
        )
    if p.is_dir():
        return BackupInfo(
            path=p, ok=False, kind="unreadable", detail=f"{p} is a directory with no schema.sql"
        )
    held = in_use(p)
    if held is not None:
        return BackupInfo(path=p, ok=False, kind="database", held_by=held, detail=held)
    try:
        handle = Anatid.open(p, read_only=True, ensure=False, fts=False, accelerators=False)
    except Exception as exc:
        first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        return BackupInfo(path=p, ok=False, kind="unreadable", bytes=_file_bytes(p), detail=first)
    try:
        version = current_version(handle.connection)
        rows: dict[str, int] = {}
        if counts:
            for table in ("memories", "entities", "episodes", "edges_about", "edges_relates"):
                with contextlib.suppress(duckdb.Error):
                    row = handle.execute(f"SELECT count(*) FROM {quote_ident(table)}").fetchone()
                    rows[table] = int(row[0]) if row else 0
        report = handle.doctor() if doctor else None
    finally:
        handle.close()
    ok = version is not None and (report is None or report.ok)
    detail = "readable"
    if version is None:
        detail = "no anatid_meta row: this is a DuckDB file, but not an anatid database"
    elif version != SCHEMA_VERSION:
        detail = f"schema version {version}, this build expects {SCHEMA_VERSION}"
    elif report is not None and not report.ok:
        detail = f"doctor found {len(report.errors)} errors: " + ", ".join(
            f.check for f in report.errors
        )
    return BackupInfo(
        path=p,
        ok=ok,
        kind="database",
        schema_version=version,
        bytes=_file_bytes(p),
        counts=rows,
        doctor=report,
        detail=detail,
    )


# --------------------------------------------------------------------------- the copy itself


def _copy_database(db: Anatid, destination: Path) -> None:
    """``COPY FROM DATABASE`` from the handle that owns the file into a fresh file.

    The same statement :meth:`anatid.DatabasePool.backup` runs, spelled out here because a
    server built on ``database=`` has a handle and no pool.  DuckDB runs it in its own
    transaction, so what lands in ``destination`` is committed data only, and it writes a
    folded, self-contained file: there is no ``.wal`` beside the result.

    The ``DETACH`` is in a ``finally`` because an attached alias that outlives a failed copy
    would make the next backup fail on a name that is already taken, and would keep an open file
    handle on a destination the caller thinks it can delete.
    """
    literal = str(destination).replace("'", "''")
    row = db.execute("SELECT current_database()").fetchone()
    catalog = quote_ident(str(row[0]) if row else "memory")
    alias = "anatid_backup"
    db.execute(f"ATTACH '{literal}' AS {quote_ident(alias)}")
    try:
        db.execute(f"COPY FROM DATABASE {catalog} TO {quote_ident(alias)}")
    finally:
        with contextlib.suppress(Exception):
            db.execute(f"DETACH {quote_ident(alias)}")


def _export_database(db: Anatid, destination: Path, fmt: str) -> None:
    """``EXPORT DATABASE`` into a directory: ``schema.sql``, ``load.sql``, one file per table.

    The one copy path that runs inside a transaction (measured: ``COPY FROM DATABASE`` cannot,
    because its ``DETACH`` raises ``TransactionException`` there).  That is not why it is here,
    though -- it is here because its output survives a DuckDB storage-format change and a
    database file does not.
    """
    literal = str(destination).replace("'", "''")
    if fmt.lower() not in ("parquet", "csv"):
        raise ValueError(f"export format must be 'parquet' or 'csv', got {fmt!r}")
    db.execute(f"EXPORT DATABASE '{literal}' (FORMAT {fmt.upper()})")


# --------------------------------------------------------------------------- the barrier


class _Quiesce:
    """Hold the write queue's serving slot for one or more tenants.

    The whole mechanism is the queue's own scheduling rule, used rather than duplicated: a
    tenant is never served by two workers at once, and its writes are taken in the order they
    were submitted.  So a non-batchable item submitted for tenant T runs after everything
    already queued for T and before everything queued afterwards, and while its ``work`` has not
    returned, no other write for T can run.  That item is this barrier.  Nothing in
    :mod:`anatid.server.queue` had to change to make it work.

    Two bounds keep it from becoming a hang.  ``acquire_timeout`` is how long the caller waits
    for the barrier to reach the front; a caller that gives up sets both the abandon flag and
    the release flag, so a barrier that starts a moment later returns immediately instead of
    holding a worker nobody is waiting for.  ``hold_timeout`` is the ceiling on the hold itself,
    so a caller that dies mid-copy costs the tenant that long and not forever.

    The barrier's ``work`` runs inside the transaction the queue opens, and writes nothing.  An
    open empty transaction on one connection does not stop another connection attaching, copying
    or detaching (measured), which is what lets the copy run on the calling thread while the
    barrier holds the worker.
    """

    def __init__(
        self,
        queue: WriteQueue,
        tenants: Sequence[int],
        *,
        acquire_timeout: float,
        hold_timeout: float,
    ) -> None:
        self.queue = queue
        self.tenants = [int(t) for t in tenants]
        self.acquire_timeout = float(acquire_timeout)
        self.hold_timeout = float(hold_timeout)
        self._held: dict[int, threading.Event] = {t: threading.Event() for t in self.tenants}
        self._release = threading.Event()
        self._abandon = threading.Event()
        self._futures: list[Any] = []
        self.waited_for = 0.0
        self.expired = False
        self.active = False
        #: True when a barrier was wanted but the queue's workers are not running, so nothing it
        #: holds can commit and the copy already has the boundary a barrier would have given it.
        self.inert = False

    def _work(self, tenant_id: int) -> Any:
        def hold(_db: Any) -> str:
            if self._abandon.is_set():
                return "abandoned"
            self._held[tenant_id].set()
            if not self._release.wait(self.hold_timeout):
                self.expired = True
                log.warning(
                    "backup barrier for tenant %s released itself after %.1fs; the copy was "
                    "still running, so this backup has the snapshot guarantee and not the "
                    "quiesced one",
                    tenant_id,
                    self.hold_timeout,
                )
                return "expired"
            return "released"

        return hold

    def __enter__(self) -> "_Quiesce":
        if not self.tenants:
            return self
        if len(self.tenants) > self.queue.workers:
            raise QuiesceUnavailable(
                f"quiescing {len(self.tenants)} tenants at once needs {len(self.tenants)} of "
                f"the write queue's worker threads and it has {self.queue.workers}. Back the "
                f"tenants up one at a time, raise ServerConfig(workers=...), or pass "
                f"quiesce=False and take the snapshot guarantee instead."
            )
        started = time.monotonic()
        deadline = started + self.acquire_timeout
        try:
            for tenant_id in self.tenants:
                try:
                    future = self.queue.submit(
                        tenant_id,
                        "backup.quiesce",
                        self._work(tenant_id),
                        batchable=False,
                    )
                except ShuttingDown as exc:
                    raise QuiesceUnavailable(
                        f"the write queue has stopped accepting, so the backup barrier for "
                        f"tenant {tenant_id} cannot be queued behind the writes that are still "
                        f"draining. Wait for the shutdown to finish and back the file up "
                        f"offline, or pass quiesce=False and take the snapshot guarantee."
                    ) from exc
                except BusyError as exc:
                    raise QuiesceUnavailable(
                        f"tenant {tenant_id}'s write queue is full ({exc}), so the barrier "
                        f"cannot be queued. Retry when the queue has drained, or pass "
                        f"quiesce=False."
                    ) from exc
                self._futures.append(future)
            for tenant_id in self.tenants:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._held[tenant_id].wait(remaining):
                    raise QuiesceTimeout(
                        f"the backup barrier for tenant {tenant_id} did not reach the front of "
                        f"its write queue within {self.acquire_timeout}s (depth "
                        f"{self.queue.depth(tenant_id)}). Nothing was copied and nothing was "
                        f"paused. Raise acquire_timeout, or pass quiesce=False."
                    )
        except BaseException:
            self._abort()
            raise
        self.waited_for = time.monotonic() - started
        self.active = True
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._release.set()
        self.active = False
        for future in self._futures:
            with contextlib.suppress(Exception):
                future.result(timeout=max(1.0, self.hold_timeout))
        self._futures = []

    def _abort(self) -> None:
        """Give up on a barrier that never closed, without leaving a worker holding one."""
        self._abandon.set()
        self._release.set()
        for future in self._futures:
            with contextlib.suppress(Exception):
                future.result(timeout=5.0)
        self._futures = []


# --------------------------------------------------------------------------- coordinator


class BackupCoordinator:
    """Backups taken with the running server's cooperation.

    ::

        coordinator = BackupCoordinator(server)
        report = coordinator.backup("/backups/2026-09-04/tenant-1.anatid", tenant=1).one
        print(report.guarantee, report.megabytes, report.seconds)

    Every method here is synchronous and safe to call from any thread: the coordination is the
    write queue's, which is threads and futures rather than asyncio, so a backup does not need
    an event loop and can be tested without one.  :meth:`abackup` and :meth:`aexport` are the
    awaitable wrappers for a caller that is already on the loop, and they run the work on a
    thread so the loop keeps answering requests while the copy runs.

    ``acquire_timeout``
        How long to wait for the barrier to reach the front of a tenant's write queue before
        giving up with :class:`QuiesceTimeout`.  Nothing is copied and nothing is paused when it
        expires.
    ``hold_timeout``
        The ceiling on the pause itself.  A copy that outlasts it carries on, the tenant's
        writes resume, and the report says :attr:`Guarantee.SNAPSHOT` rather than claiming a
        boundary it no longer has.
    ``file_template``
        How a per-tenant file is named when the destination is a directory.
    """

    def __init__(
        self,
        server: "AnatidServer",
        *,
        acquire_timeout: float = 30.0,
        hold_timeout: float = 600.0,
        file_template: str = DEFAULT_FILE_TEMPLATE,
    ) -> None:
        self.server = server
        self.acquire_timeout = float(acquire_timeout)
        self.hold_timeout = float(hold_timeout)
        self.file_template = file_template

    # -- what there is to back up ------------------------------------------------------

    @property
    def pool(self) -> DatabasePool | None:
        return self.server.pool

    @property
    def shared_file(self) -> bool:
        """True when this server holds one file for every tenant rather than one file each.

        It changes what a backup IS, not just how it is taken: with a pool, backing up tenant 3
        copies tenant 3's file and nothing else; with a shared file there is one file, so the
        copy holds every tenant in it whichever tenant was named.
        """
        return self.pool is None

    def tenants(self) -> list[int]:
        """Every tenant this server has a reason to back up, in ascending order.

        The configured ones, the ones whose file the pool has resolved or opened, and the ones
        with something in the write queue.  A tenant that has never been touched has no file,
        and a backup of a file that does not exist is not a backup.

        With one shared file this is the tenants that can be *quiesced*, not a list of files:
        the queue only knows a tenant it has been asked to write, so a tenant that is idle now
        and starts writing during the copy is not in it.  That is exactly why a shared-file copy
        never claims the quiesced guarantee for the whole file.
        """
        found: set[int] = {int(t) for t in self.server.config.tenants}
        pool = self.pool
        if pool is not None:
            found.update(int(t) for t in pool.known_tenants())
            found.update(int(t) for t in pool.registry())
        else:
            db = self.server.database
            if db is not None:
                found.add(int(db.namespace.tenant_id))
        found.update(int(t) for t in self.server.queue.stats().depth)
        return sorted(found)

    def _handle(self, tenant_id: int | None) -> Anatid:
        pool = self.pool
        if pool is not None:
            if tenant_id is None:  # unreachable: a pool backup always names a tenant
                raise BackupError("a pooled backup needs a tenant")
            return pool.get(tenant_id)
        db = self.server.database
        if db is None:  # unreachable: AnatidServer requires one of pool= / database=
            raise BackupError("this server has no database handle")
        return db

    def _source_path(self, tenant_id: int | None) -> Path:
        db = self._handle(tenant_id)
        raw = os.fspath(db.path)
        if raw == ":memory:":
            raise BackupError(
                f"tenant {tenant_id} is an in-memory database; there is nothing on disk to back "
                f"up and nothing to restore it into"
            )
        return Path(raw)

    def _destination(
        self, destination: str | os.PathLike[str], tenant_id: int | None, source: Path
    ) -> Path:
        """Where one copy goes.

        A directory destination gets ``file_template`` inside it, which is what makes
        ``backup(dir)`` for every tenant and ``backup(dir, tenant=3)`` for one land in the same
        place under the same name.  A shared file, which is not any one tenant's, keeps the
        source file's own name instead.
        """
        dest = Path(os.fspath(destination)).expanduser()
        if not dest.is_dir():
            return dest
        if self.shared_file:
            return dest / source.name
        return dest / self.file_template.format(tenant=tenant_id)

    def _normalise(self, tenant: int | Iterable[int] | None) -> list[int] | None:
        if tenant is None:
            return None
        if isinstance(tenant, int):
            return [int(tenant)]
        return [int(t) for t in tenant]

    # -- taking one --------------------------------------------------------------------

    def backup(
        self,
        destination: str | os.PathLike[str],
        *,
        tenant: int | Iterable[int] | None = None,
        quiesce: bool = True,
        overwrite: bool = False,
        verify: bool = True,
        doctor: bool = False,
    ) -> BackupSet:
        """Copy a tenant's database, or every tenant's, and say what the copy promises.

        ``tenant`` takes one id, several ids, or ``None`` for every tenant :meth:`tenants` names.

        With a :class:`~anatid.DatabasePool` that is one copy per tenant, and ``destination`` has
        to be a directory when more than one is named.  Each tenant is quiesced on its own, one
        at a time, so a ten-tenant backup pauses each tenant for the length of its own copy and
        never all ten at once.  The copy holds that tenant and nothing else, so ``quiesce=True``
        gives it :attr:`Guarantee.QUIESCED`: every write acknowledged before the call is in it,
        and no write that commits after the barrier closes is.

        With one shared file there is one copy however many tenants are named, because there is
        one file, and it holds every tenant in that file.  The named tenants are still paused for
        it and are named in :attr:`BackupReport.quiesced_tenants`, so their rows do carry the
        stronger boundary; the copy as a whole reports :attr:`Guarantee.SNAPSHOT`, because a
        tenant nobody named could have committed while it ran.  The exception is a handle opened
        with :attr:`~anatid.types.Isolation.FILE_PER_TENANT`, where the file holds exactly the
        one tenant and pausing it does cover the file.

        ``quiesce=False`` pauses nothing anywhere and always gives :attr:`Guarantee.SNAPSHOT`.
        Read this module's docstring before choosing it.

        ``verify=True`` reopens each copy read-only and reads its schema version and row counts,
        because a backup nobody has opened is a hypothesis.  ``doctor=True`` additionally runs
        the full integrity check on it (54 ms on a 100k-memory file).
        """
        named = self._normalise(tenant)
        if self.shared_file:
            pause = named if named is not None else self.tenants()
            return BackupSet(
                [
                    self._backup_one(
                        None if named is None or len(named) != 1 else named[0],
                        pause if quiesce else [],
                        destination,
                        overwrite=overwrite,
                        verify=verify,
                        doctor=doctor,
                    )
                ]
            )
        targets = named if named is not None else self.tenants()
        if not targets:
            raise BackupError(
                "this server has no tenant with a file to back up: none configured, none open "
                "and none in the write queue"
            )
        if len(targets) > 1:
            dest_root = Path(os.fspath(destination)).expanduser()
            if dest_root.exists() and not dest_root.is_dir():
                raise BackupError(
                    f"backing up {len(targets)} tenants needs a directory to put them in; "
                    f"{dest_root} is a file"
                )
            dest_root.mkdir(parents=True, exist_ok=True)
        reports = [
            self._backup_one(
                t,
                [t] if quiesce else [],
                destination,
                overwrite=overwrite,
                verify=verify,
                doctor=doctor,
            )
            for t in targets
        ]
        return BackupSet(reports)

    def _backup_one(
        self,
        tenant_id: int | None,
        pause: Sequence[int],
        destination: str | os.PathLike[str],
        *,
        overwrite: bool,
        verify: bool,
        doctor: bool,
    ) -> BackupReport:
        source = self._source_path(tenant_id)
        dest = self._destination(destination, tenant_id, source)
        self._prepare_destination(dest, overwrite=overwrite)
        started_at = utcnow()
        with self._barrier(pause) as barrier:
            copy_started = time.monotonic()
            db = self._handle(tenant_id)
            pool = self.pool
            try:
                if pool is not None and tenant_id is not None:
                    # The pool's own copy: same statement, plus its audit event and file mode.
                    pool.backup(tenant_id, dest, overwrite=True)
                else:
                    _copy_database(db, dest)
            except Exception as exc:
                with contextlib.suppress(OSError):
                    dest.unlink()
                raise BackupError(f"could not copy {source} to {dest}: {exc}") from exc
            seconds = time.monotonic() - copy_started
            held = barrier.active and not barrier.expired
            quiesced_for = seconds if held else 0.0
        report = self._finish(
            dest,
            tenant_id,
            self._guarantee(pause, held=held, inert=barrier.inert),
            seconds,
            quiesced_for,
            barrier.waited_for,
            tuple(pause) if held or barrier.inert else (),
            started_at,
            verify=verify,
            doctor=doctor,
            inert=barrier.inert,
        )
        log.info("%s", report)
        return report

    def _guarantee(self, pause: Sequence[int], *, held: bool, inert: bool) -> Guarantee:
        """QUIESCED only when every tenant in the copy was paused for it.

        With a pool the copy holds one tenant, so pausing it is the whole file.  With a shared
        file the copy holds every tenant in the file, and pausing the ones this server happens to
        know is not the same thing -- unless the handle is file-per-tenant, in which case the
        file holds exactly the one tenant it is bound to.
        """
        if not pause or not (held or inert):
            return Guarantee.SNAPSHOT
        if not self.shared_file:
            return Guarantee.QUIESCED
        db = self.server.database
        covers_file = (
            db is not None
            and db.namespace.is_isolated
            and list(pause) == [int(db.namespace.tenant_id)]
        )
        return Guarantee.QUIESCED if covers_file else Guarantee.SNAPSHOT

    def _prepare_destination(self, dest: Path, *, overwrite: bool) -> None:
        if dest.exists():
            if not overwrite:
                raise BackupError(
                    f"{dest} exists; pass overwrite=True to replace it. A backup that "
                    f"overwrites silently is one crash away from leaving you with neither copy."
                )
            held = in_use(dest)
            if held is not None:
                raise DestinationInUse(f"{dest} is open in another process: {held}")
            dest.unlink()
        dest.parent.mkdir(parents=True, exist_ok=True)

    def _barrier(self, tenants: Sequence[int]) -> _Quiesce:
        """The barrier for ``tenants``, or an inert one when nothing needs pausing.

        A queue whose workers are not running drains nothing, so no queued write can commit
        while the copy runs and the boundary is already where a barrier would put it.
        Submitting one there would wait for a worker that will never take it, so the barrier is
        built inert instead: it pauses nothing, holds nothing, and still reports the quiesced
        guarantee, because on a stopped queue that is what the copy actually has.
        """
        running = self.server.queue.running
        barrier = _Quiesce(
            self.server.queue,
            list(tenants) if running else [],
            acquire_timeout=self.acquire_timeout,
            hold_timeout=self.hold_timeout,
        )
        barrier.inert = bool(tenants) and not running
        return barrier

    def _finish(
        self,
        dest: Path,
        tenant_id: int | None,
        guarantee: Guarantee,
        seconds: float,
        quiesced_for: float,
        waited_for: float,
        quiesced_tenants: tuple[int, ...],
        started_at: _dt.datetime,
        *,
        verify: bool,
        doctor: bool,
        inert: bool = False,
    ) -> BackupReport:
        size = _dir_bytes(dest) if dest.is_dir() else dest.stat().st_size
        info = inspect(dest, doctor=doctor) if (verify or doctor) else None
        if info is not None and not info.ok:
            raise BackupUnreadable(
                f"the copy at {dest} did not verify: {info.detail}. It has been left in place "
                f"for inspection; do not treat it as a backup."
            )
        detail = "" if info is None else info.detail
        note = ""
        if guarantee is Guarantee.SNAPSHOT and not quiesced_tenants:
            note = (
                "taken without pausing writes: every write acknowledged before the call is in "
                "it, and a write acknowledged during it may or may not be"
            )
        elif guarantee is Guarantee.SNAPSHOT:
            note = (
                f"one shared file, so the copy holds every tenant in it; tenants "
                f"{list(quiesced_tenants)} were paused and their rows carry the quiesced "
                f"boundary, the rest carry the snapshot one"
            )
        elif inert:
            note = "the write queue was not running, so nothing could commit while this ran"
        if note:
            detail = f"{note}; {detail}" if detail else note
        return BackupReport(
            path=dest,
            tenant=tenant_id,
            guarantee=guarantee,
            bytes=size,
            seconds=seconds,
            quiesced_for=quiesced_for,
            waited_for=waited_for,
            quiesced_tenants=quiesced_tenants,
            schema_version=None if info is None else info.schema_version,
            counts={} if info is None else dict(info.counts),
            doctor=None if info is None else info.doctor,
            started_at=started_at,
            detail=detail,
        )

    # -- the online export -------------------------------------------------------------

    def export(
        self,
        destination: str | os.PathLike[str],
        *,
        tenant: int | Iterable[int] | None = None,
        fmt: str = "parquet",
        overwrite: bool = False,
    ) -> BackupSet:
        """``EXPORT DATABASE`` one tenant's file, or every tenant's, without pausing anything.

        The copy that is not a database file: a directory of Parquet plus the DDL to rebuild it,
        which is why it is worth keeping beside the file copies.  A DuckDB storage-format change
        makes a database file unreadable by the new build and leaves this one perfectly readable.
        Measured on the 100k-memory spike file: 202 ms and 44.4 MiB, against 494 ms and 42.3 MiB
        for :meth:`backup`.

        It always carries :attr:`Guarantee.EXPORT`, which is the snapshot guarantee: nothing is
        paused, so a write acknowledged while it runs may or may not be in it.  There is no
        quiesced variant, because a barrier would pause the tenant for the export and buy a
        boundary the format is not the right tool for anyway; take :meth:`backup` when the
        boundary is what matters.

        Like :meth:`backup`, a shared file is exported once and the export holds every tenant in
        it; ``tenant`` selects a file, and there is only one.
        """
        named = self._normalise(tenant)
        targets: list[int | None]
        if self.shared_file:
            targets = [None if named is None or len(named) != 1 else named[0]]
        else:
            found = named if named is not None else self.tenants()
            if not found:
                raise BackupError("this server has no tenant with a file to export")
            targets = list(found)
        root = Path(os.fspath(destination)).expanduser()
        reports: list[BackupReport] = []
        for tenant_id in targets:
            dest = root / f"tenant-{tenant_id}" if len(targets) > 1 else root
            if dest.exists():
                if not overwrite:
                    raise BackupError(f"{dest} exists; pass overwrite=True to replace it")
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            started_at = utcnow()
            started = time.monotonic()
            try:
                _export_database(self._handle(tenant_id), dest, fmt)
            except Exception as exc:
                shutil.rmtree(dest, ignore_errors=True)
                raise BackupError(f"could not export tenant {tenant_id} to {dest}: {exc}") from exc
            seconds = time.monotonic() - started
            info = inspect(dest)
            reports.append(
                BackupReport(
                    path=dest,
                    tenant=tenant_id,
                    guarantee=Guarantee.EXPORT,
                    bytes=info.bytes,
                    seconds=seconds,
                    started_at=started_at,
                    detail=info.detail,
                )
            )
            log.info("%s", reports[-1])
        return BackupSet(reports)

    # -- awaitable wrappers ------------------------------------------------------------

    async def abackup(self, destination: str | os.PathLike[str], **kwargs: Any) -> BackupSet:
        """:meth:`backup` on a worker thread, so the event loop keeps serving during the copy."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.backup(destination, **kwargs))

    async def aexport(self, destination: str | os.PathLike[str], **kwargs: Any) -> BackupSet:
        """:meth:`export` on a worker thread."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.export(destination, **kwargs))

    # -- putting one back --------------------------------------------------------------

    def owned_paths(self) -> set[Path]:
        """Every database file this server currently has open, resolved.

        :meth:`restore` refuses to write over one of these.  The filesystem answer (:func:`in_use`)
        catches the same thing from another process, but a server restoring over its own open
        file would get a lock error from DuckDB and a half-written file from the copy, and
        saying so before either happens is better than reporting both afterwards.
        """
        out: set[Path] = set()
        pool = self.pool
        if pool is not None:
            for tenant_id in pool.known_tenants():
                with contextlib.suppress(Exception):
                    out.add(pool.path_for(tenant_id).resolve())
            return out
        db = self.server.database
        if db is not None and not db.closed and os.fspath(db.path) != ":memory:":
            out.add(Path(os.fspath(db.path)).resolve())
        return out

    def restore(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        doctor: bool = True,
    ) -> RestoreReport:
        """:func:`restore`, additionally refusing any file this server has open."""
        return restore(
            source,
            destination,
            overwrite=overwrite,
            doctor=doctor,
            forbid=self.owned_paths(),
        )

    def prune(self, directory: str | os.PathLike[str], **kwargs: Any) -> PruneReport:
        """:func:`prune`, additionally protecting any file this server has open."""
        protect: set[Any] = set(kwargs.pop("protect", ()))
        protect.update(self.owned_paths())
        return prune(directory, protect=protect, **kwargs)


# --------------------------------------------------------------------------- restore


def restore(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    doctor: bool = True,
    forbid: Iterable[str | os.PathLike[str]] = (),
) -> RestoreReport:
    """Put a backup back, after checking that it is one and that nothing is using the target.

    The order is the point, and it is the order that keeps a bad restore from being worse than
    no restore:

    1. the source is opened read-only and has to be a readable anatid database (or an
       ``EXPORT DATABASE`` directory), so a truncated copy is refused before anything is touched;
    2. a destination that is the source itself, or a directory, is refused -- the first would
       delete the backup and then fail to copy it, and the second is a path the operator meant
       to put a file name after;
    3. the destination is refused if some process holds it open, whether that is this process or
       another one -- restoring over a file a server owns would leave that server serving from a
       file it no longer matches;
    4. an existing destination is refused unless ``overwrite=True``;
    5. only then is anything written, and the write-ahead log travels with the file.

    Nothing is migrated.  The report says what schema version came back and whether this build
    would migrate it on the first read-write open, and that decision stays with the operator: a
    restore that silently upgraded the file would make it unreadable by the build that wrote it,
    which is the one build known to be able to read it.

    An ``EXPORT DATABASE`` directory is restored with ``IMPORT DATABASE`` into a new file, which
    rebuilds it at this build's storage format.  That is the path that survives a DuckDB
    storage-format change.
    """
    src = Path(os.fspath(source)).expanduser()
    dest = Path(os.fspath(destination)).expanduser()
    started = time.monotonic()

    info = inspect(src, doctor=False, counts=False)
    if info.kind == "missing":
        raise BackupUnreadable(f"there is no backup at {src}")
    if not info.ok:
        raise BackupUnreadable(
            f"{src} will not serve as a backup: {info.detail}. Nothing was written to {dest}."
        )

    # Checked here, before anything is unlinked, because the failure is silent otherwise: with
    # overwrite=True the destination is removed and then copied onto from the source, so a
    # source and destination that are the same file delete the backup and then fail to copy it,
    # leaving neither.  Resolved, so a symlink or a "." in the path does not get past it.
    if src.resolve() == dest.resolve():
        raise BackupError(
            f"{src} is both the source and the destination of this restore. Name the file to "
            f"restore INTO; a restore onto the backup itself would delete it."
        )
    if dest.is_dir():
        raise BackupError(
            f"{dest} is a directory. A restore writes one database FILE, so name the file to "
            f"write (for instance {dest / src.name}), not the directory to write it in."
        )

    forbidden = {Path(os.fspath(p)).expanduser().resolve() for p in forbid}
    resolved_dest = dest.resolve() if dest.exists() else (dest.parent.resolve() / dest.name)
    if resolved_dest in forbidden:
        raise DestinationInUse(
            f"{dest} is open in this process; close the handle (or stop the server) before "
            f"restoring over it. A server serving from a file that has been replaced under it "
            f"answers from a catalog that no longer describes the file."
        )
    held = in_use(dest)
    if held is not None:
        raise DestinationInUse(f"{dest} is open in another process: {held}")

    if dest.exists() and not overwrite:
        raise BackupError(
            f"{dest} exists; pass overwrite=True to replace it. Restoring over a live database "
            f"is the one operation with no undo, so it is never the default."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []
    if info.kind == "export":
        if dest.exists():
            dest.unlink()
        for extra in sidecars(dest):
            _remove(extra)
        con = duckdb.connect(str(dest))
        try:
            literal = str(src).replace("'", "''")
            con.execute(f"IMPORT DATABASE '{literal}'")
        except Exception as exc:
            con.close()
            with contextlib.suppress(OSError):
                dest.unlink()
            raise BackupUnreadable(f"could not import {src} into {dest}: {exc}") from exc
        con.close()
    else:
        if dest.exists():
            dest.unlink()
        for extra in sidecars(dest):
            _remove(extra)
        shutil.copy2(src, dest)
        for extra in sidecars(src):
            if extra.is_dir():
                continue  # a .tmp spill directory is scratch space, not state
            # The suffix comes from the known list rather than from Path.suffixes, which would
            # read "2026-09-04.anatid.wal" as two suffixes and has to be told which to take.
            suffix = next(s for s in _SIDECAR_SUFFIXES if extra.name.endswith(s))
            target = Path(f"{dest}{suffix}")
            shutil.copy2(extra, target)
            moved.append(target)

    back = inspect(dest, doctor=doctor)
    if not back.ok:
        raise BackupUnreadable(
            f"the restored file at {dest} does not open cleanly: {back.detail}. The source "
            f"{src} is untouched."
        )
    return RestoreReport(
        source=src,
        destination=dest,
        kind=info.kind,
        schema_version=back.schema_version,
        expected_schema_version=SCHEMA_VERSION,
        bytes=back.bytes,
        seconds=time.monotonic() - started,
        sidecars=tuple(moved),
        doctor=back.doctor,
        detail=back.detail,
    )


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        with contextlib.suppress(OSError):
            path.unlink()


# --------------------------------------------------------------------------- retention


def prune(
    directory: str | os.PathLike[str],
    *,
    keep: int = DEFAULT_KEEP,
    pattern: str = "*.anatid",
    verify: bool = True,
    dry_run: bool = False,
    protect: Iterable[str | os.PathLike[str]] = (),
) -> PruneReport:
    """Keep the newest ``keep`` backups in ``directory``, delete the rest, and never the last good one.

    Newest is by modification time, which is when the copy finished.  The rules, in the order
    they apply:

    1. a file some process holds open is never pruned, whatever its age -- it is a live database
       that happens to match the pattern, or a backup being written right now;
    2. a file in ``protect`` is never pruned;
    3. the newest ``keep`` files are kept;
    4. if none of those is verifiably good, the newest good file is kept as well.

    Rule 4 is the one that matters, and it is why ``verify`` defaults to on.  "Keep the last
    seven" is a retention policy right up to the morning seven broken copies have pushed the
    last good one out of the window; ``prune`` will not be the thing that deletes it.  With
    ``verify=False`` no file is checked, so every file counts as good and rule 4 only protects
    the newest.

    ``dry_run=True`` reports exactly what it would remove and removes nothing.
    """
    root = Path(os.fspath(directory)).expanduser()
    if not root.is_dir():
        raise BackupError(f"{root} is not a directory")
    if keep < 0:
        raise ValueError("keep must be >= 0")
    protected = {Path(os.fspath(p)).expanduser().resolve() for p in protect}

    candidates = sorted(
        (p for p in root.iterdir() if _matches(p, pattern)),
        key=lambda p: (p.stat().st_mtime, p.name),
        reverse=True,
    )

    bad: dict[str, str] = {}
    held: dict[str, str] = {}
    good: list[Path] = []
    for path in candidates:
        busy = in_use(path) if path.is_file() else None
        if busy is not None:
            held[path.name] = busy
            continue
        if not verify:
            good.append(path)
            continue
        info = inspect(path, counts=False)
        if info.ok:
            good.append(path)
        else:
            bad[path.name] = info.detail

    keepable = [p for p in candidates if p.name not in held]
    kept = list(keepable[:keep])
    if not any(p in good for p in kept):
        newest_good = next((p for p in keepable if p in good), None)
        if newest_good is not None and newest_good not in kept:
            kept.append(newest_good)
    kept.extend(p for p in keepable if p.resolve() in protected and p not in kept)
    kept_set = set(kept)
    doomed = [p for p in keepable if p not in kept_set]

    pruned: list[Path] = []
    failed: dict[str, str] = {}
    freed = 0
    for path in doomed:
        size = _dir_bytes(path) if path.is_dir() else path.stat().st_size
        if dry_run:
            pruned.append(path)
            freed += size
            continue
        try:
            _remove_strict(path)
        except OSError as exc:
            failed[path.name] = str(exc)
            continue
        pruned.append(path)
        freed += size

    held_names = [p for p in candidates if p.name in held]
    detail = (
        f"kept {len(kept)} of {len(candidates)} backups"
        + (f", pruned {len(pruned)}" if pruned else ", pruned none")
        + (f", {len(bad)} would not open" if bad else "")
        + (f", {len(held_names)} in use and left alone" if held_names else "")
    )
    return PruneReport(
        directory=root,
        kept=tuple(sorted(kept, key=lambda p: p.name)),
        pruned=tuple(sorted(pruned, key=lambda p: p.name)),
        bad=bad,
        held=held,
        failed=failed,
        freed_bytes=freed,
        dry_run=dry_run,
        detail=detail,
    )


def _matches(path: Path, pattern: str) -> bool:
    """A candidate for pruning: matches the pattern, and is not a sidecar of another backup.

    A ``.wal`` beside a backup is part of that backup, so it never gets its own decision; it
    goes when its database file goes, in :func:`_remove_strict`.
    """
    if not fnmatch.fnmatch(path.name, pattern):
        return False
    return not any(path.name.endswith(suffix) for suffix in _SIDECAR_SUFFIXES)


def _remove_strict(path: Path) -> None:
    """Delete a backup and its sidecars, raising rather than swallowing an OSError."""
    for extra in sidecars(path):
        if extra.is_dir():
            shutil.rmtree(extra)
        else:
            extra.unlink()
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


# --------------------------------------------------------------------------- on the wire
#
# Registered here, where the types are defined.  A backup is reachable over the protocol (the
# operator verb in anatid.server.cli), so a client has to be able to name what came back and what
# went wrong.  Without these a QuiesceTimeout arrives as a generic RemoteError, which loses the
# one thing it says: nothing was copied and nothing was paused, so it is safe to try again.

protocol.register_enum(Guarantee)
for _dc in (BackupReport, BackupInfo, RestoreReport):
    protocol.register_dataclass(_dc)
del _dc
#: Under a tag of its own, because ``anatid.types.PruneReport`` (what ``prune()`` removed from a
#: tenant) already holds ``PruneReport`` and the two are unrelated.  The class name stays as it is:
#: it reads correctly at ``anatid.server.backup.PruneReport``, and it is the WIRE that needs the
#: tags to be distinct, not the module.
protocol.register_dataclass(PruneReport, "BackupPruneReport")
for _exc in (
    BackupError,
    QuiesceTimeout,
    QuiesceUnavailable,
    DestinationInUse,
    BackupUnreadable,
):
    protocol.register_error(_exc)
del _exc
